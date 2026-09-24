"""Specific star-formation rates: empirical low-z COSMOS2025 p(logsSFR | logM) per type, evolved with redshift.

Star-forming galaxies are shifted by the change in the main-sequence ridge between the low-z reference redshift and
their own; passive galaxies follow a weak linear redshift slope about the low-z distribution.
"""

import numpy as np

from .cosmos_catalogue import standardise_cosmos_columns
from .utils import build_kd_tree, pick_kernel_weighted_neighbour, query_nearest


class LowRedshiftSSFRPDF:
    """Kernel-weighted nearest-neighbour sampler of logsSFR given logM, kept separately for 'sf' and 'pa' (passive)."""

    def __init__(self, samples, k=160):
        self.samples, self.trees, self.k = {}, {}, int(k)
        for kind, df in samples.items():
            d = df[["logM", "logsSFR"]].dropna().reset_index(drop=True)
            if len(d) == 0:
                raise ValueError(f"No low-z sSFR training rows for {kind}")
            self.samples[kind], self.trees[kind] = d, build_kd_tree(d[["logM"]].to_numpy(float))

    def sample(self, kind, logM, rng):
        logM = np.asarray(logM, float)
        out = np.full(len(logM), np.nan)
        if len(logM) == 0:
            return out
        d = self.samples[kind]
        k = min(self.k, len(d))
        dist, ind = query_nearest(self.trees[kind], logM[:, None], k, workers=-1)
        if k == 1:
            dist, ind = dist[:, None], ind[:, None]
        vals = d["logsSFR"].to_numpy(float)
        for i in range(len(logM)):
            out[i] = vals[pick_kernel_weighted_neighbour(dist[i], ind[i], rng, fallback_scale=0.15, min_scale=0.08)]
        return out


def build_low_redshift_ssfr_pdf(cosmos, cfg):
    """Fit the per-type p(logsSFR | logM) to COSMOS galaxies at ssfr_pdf_z_min <= z < ssfr_pdf_z_max."""
    c, _ = standardise_cosmos_columns(cosmos, cfg)
    d = c[["z", "logM", "logsSFR"]].replace([np.inf, -np.inf], np.nan).dropna()
    d = d[(d["z"] >= cfg["ssfr_pdf_z_min"]) & (d["z"] < cfg["ssfr_pdf_z_max"])
          & d["logM"].between(cfg["logm_min"], cfg["logm_max"])].copy()
    samples = {"sf": d[d["logsSFR"] > cfg["ssfr_split"]].copy(), "pa": d[d["logsSFR"] <= cfg["ssfr_split"]].copy()}
    print(f"Low-z COSMOS2025 sSFR PDF rows: SF={len(samples['sf']):,}, passive={len(samples['pa']):,}, "
          f"z<{cfg['ssfr_pdf_z_max']}")
    return LowRedshiftSSFRPDF(samples, k=cfg["ssfr_pdf_k"])


def main_sequence_log_sfr(logM, z, grids):
    """Star-forming main-sequence ridge log SFR(logM, cosmic age) in the Speagle et al. (2014) form."""
    age = np.interp(np.asarray(z, float), grids["z_age"], grids["age_gyr"])
    return (0.84 - 0.026 * age) * np.asarray(logM, float) - (6.51 - 0.11 * age)


def ssfr_redshift_offset(logM, z, kind, grids, cfg, phys):
    """logsSFR shift from the low-z reference redshift to z: main-sequence evolution ("sf") or a linear slope ("pa")."""
    logM, z = np.asarray(logM, float), np.asarray(z, float)
    zref, out = cfg["ssfr_reference_z"], np.zeros(len(logM), float)
    sf = np.asarray(kind) == "sf"
    if sf.any():
        ridge = main_sequence_log_sfr(logM[sf], z[sf], grids) - logM[sf]
        ridge_ref = main_sequence_log_sfr(logM[sf], np.full(sf.sum(), zref), grids) - logM[sf]
        out[sf] = ridge - ridge_ref
    if (~sf).any():
        out[~sf] = phys["passive_logssfr_z_slope"] * (z[~sf] - zref)
    return out


def draw_log_ssfr(logM, z, kind, ssfr_pdf, grids, cfg, phys, rng):
    """Low-z draw per type plus the redshift offset, clipped to [ssfr_min, ssfr_max]: (low-z draw, offset, logsSFR)."""
    sf = kind == "sf"
    low_z_draw = np.full(len(logM), np.nan)
    low_z_draw[sf] = ssfr_pdf.sample("sf", logM[sf], rng)
    low_z_draw[~sf] = ssfr_pdf.sample("pa", logM[~sf], rng)
    offset = ssfr_redshift_offset(logM, z, kind, grids, cfg, phys)
    return low_z_draw, offset, np.clip(low_z_draw + offset, cfg["ssfr_min"], cfg["ssfr_max"])
