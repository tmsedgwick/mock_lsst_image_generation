"""Empirical rest-frame SED cloning: COSMOS donors -> conditional rest-frame PDF -> observed LSST photometry.

Rather than learning observed-frame ugrizy directly, each COSMOS donor's observed mags and angular size are converted
to rest-frame absolute mags (at REST_NODES) and a physical Re. Mock galaxies draw a donor from their (logM, logsSFR)
neighbourhood, i.e. sample p(rest SED, Re_kpc, ellipticity | logM, logsSFR), and the cloned SED is projected to
observed ugrizy at the mock redshift. This is deliberately simple SED cloning, not a stellar-population model.
"""

import numpy as np
import pandas as pd

from .cosmology import arcsec_to_kpc, build_cosmology_grids, kpc_to_arcsec
from .cosmos_catalogue import standardise_cosmos_columns
from .donor_selection import apply_donor_quality_cuts, apply_rest_frame_safety_cuts
from .photometry import (REST_COLS, component_surface_brightness, mag_to_flux_njy, observed_to_rest_abs_mags,
                         rest_abs_to_observed_mags)
from .utils import build_kd_tree, numeric, pick_kernel_weighted_neighbour, query_nearest


def build_rest_frame_donor_table(cosmos, cfg, grids=None):
    """Quality-cut COSMOS donors with rest-frame absolute mags, physical Re, ellipticity and SED extrapolation."""
    grids = build_cosmology_grids(cfg) if grids is None else grids
    c, obs_bands = standardise_cosmos_columns(cosmos, cfg)
    obs_waves, obs_cols = np.array([wave for _, wave, _ in obs_bands], float), [col for _, _, col in obs_bands]
    c = apply_donor_quality_cuts(c, obs_cols, cfg)
    rows = []
    for _, row in c.iterrows():
        z, obs_mags = row["z"], row[obs_cols].to_numpy(float)
        if not np.isfinite(z) or z <= 0 or np.isfinite(obs_mags).sum() < cfg["min_training_bands"]:
            continue
        rest_abs, edge = observed_to_rest_abs_mags(obs_mags, obs_waves, z, grids, cfg)
        re_kpc = arcsec_to_kpc(row["Re_arcsec"], z, grids)
        rows.append(dict(z_cosmos=z, log1pz=np.log10(1.0 + z), logM=row["logM"], logSFR=row["logSFR"], logsSFR=row["logsSFR"],
                         Re_arcsec=row["Re_arcsec"], Re_kpc=re_kpc, ellipticity=row["ellipticity"],
                         rest_edge_distance=edge, n_obs_bands=int(np.isfinite(obs_mags).sum()),
                         logRe_kpc=np.log10(re_kpc) if re_kpc > 0 else np.nan, **dict(zip(REST_COLS, rest_abs))))
    donors = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    donors = donors.dropna(subset=["logM", "logSFR", "logsSFR", "logRe_kpc", "ellipticity", *REST_COLS])
    donors = apply_rest_frame_safety_cuts(donors, cfg)
    donors = donors[donors["Re_kpc"].between(1e-3, 200.0) & donors["ellipticity"].between(0.0, 0.95)
                    & donors["logM"].between(5.0, 13.0)]
    return donors.reset_index(drop=True)


class RestFrameEmpiricalPDF:
    """p(rest SED, logRe_kpc, ellipticity | xcols): kernel-weighted draws among the k nearest donors in the
    standardised xcols space (inputs clipped to the donor percentile range). xweights (default all 1) multiply each
    standardised coordinate, so a larger weight makes donors match more closely in that quantity."""

    def __init__(self, xcols=None, k=256, clip_percentiles=(0.5, 99.5), seed=42, xweights=None):
        self.xcols = list(xcols) if xcols is not None else ["logM", "logsSFR"]
        self.xweights = np.ones(len(self.xcols)) if xweights is None else np.asarray(xweights, float)
        self.k, self.clip_percentiles, self.seed = k, clip_percentiles, seed

    def fit(self, donors):
        self.train = donors.copy()
        self.ycols = [*REST_COLS, *([] if "logRe_kpc" in self.xcols else ["logRe_kpc"]), "ellipticity"]
        missing = [c for c in [*self.xcols, *self.ycols] if c not in self.train.columns]
        if missing:
            raise KeyError(f"Training table missing columns: {missing}")
        X = self.train[self.xcols].to_numpy(float)
        self.xlo = np.nanpercentile(X, self.clip_percentiles[0], axis=0)
        self.xhi = np.nanpercentile(X, self.clip_percentiles[1], axis=0)
        Xc = np.clip(X, self.xlo, self.xhi)
        self.xmed, self.xscale = np.nanmedian(Xc, axis=0), np.nanstd(Xc, axis=0)
        self.xscale[self.xscale == 0] = 1.0
        self.xscale = self.xscale / self.xweights
        self.Xs = (Xc - self.xmed) / self.xscale
        self.tree = build_kd_tree(self.Xs)
        self.k = min(self.k, len(self.train))
        return self

    def sample(self, query, rng=None, chunk=10000):
        """Donor properties (ycols) for each query row, plus the standardised clipping distance and donor index."""
        rng = np.random.default_rng(self.seed) if rng is None else rng
        X = query[self.xcols].to_numpy(float)
        Xc = np.clip(X, self.xlo, self.xhi)
        Xs = (Xc - self.xmed) / self.xscale
        picked = np.empty(len(query), dtype=int)
        for start in range(0, len(query), chunk):
            stop = min(start + chunk, len(query))
            dist, ind = query_nearest(self.tree, Xs[start:stop], self.k, workers=-1)
            if self.k == 1:
                dist, ind = dist[:, None], ind[:, None]
            for j in range(stop - start):
                picked[start + j] = pick_kernel_weighted_neighbour(dist[j], ind[j], rng, fallback_scale=1.0,
                                                                   min_scale=1e-6)
        y = self.train.iloc[picked][self.ycols].reset_index(drop=True)
        y["empirical_edge_distance"] = np.sqrt((((X - Xc) / self.xscale) ** 2).sum(axis=1))
        y["donor_index"] = picked
        y["donor_logM"] = self.train["logM"].to_numpy(float)[picked]
        y["donor_z"] = self.train["z_cosmos"].to_numpy(float)[picked]
        return y


def sample_rest_frame_properties(mock, pdf, rng):
    """Append a cloned donor's rest-frame SED, Re (kpc) and ellipticity to every mock galaxy."""
    rest = pdf.sample(mock[pdf.xcols].copy(), rng=rng)
    out = pd.concat([mock.reset_index(drop=True), rest.reset_index(drop=True)], axis=1)
    if "logRe_kpc" not in out.columns:
        raise KeyError("logRe_kpc missing after empirical sampling.")
    out["Re_kpc"] = 10.0 ** out["logRe_kpc"]
    out["ellipticity_total"] = out["ellipticity"].clip(0.0, 0.95)
    return out


def project_to_observed_frame(rest_mock, cfg, grids):
    """Observed mags/fluxes per band, angular Re, large-Re ellipticity cap and mean surface brightness."""
    m = rest_mock.copy()
    obs_mag, edge = np.empty((len(m), len(cfg["out_bands"]))), np.empty(len(m))
    for i, (rest_abs, z) in enumerate(zip(m[REST_COLS].to_numpy(float), m["z"].to_numpy(float))):
        obs_mag[i], edge[i] = rest_abs_to_observed_mags(rest_abs, z, grids, cfg)
    m["projection_edge_distance"] = edge
    for j, band in enumerate(cfg["out_bands"]):
        m[f"mag_{band}_total"], m[f"flux_{band}_total"] = obs_mag[:, j], mag_to_flux_njy(obs_mag[:, j])
    m["re_total_kpc"] = m["Re_kpc"]
    m["re_total_arcsec"] = kpc_to_arcsec(m["Re_kpc"], m["z"], grids)

    re_cap_min, ell_cap = cfg["large_re_ellipticity_cap_arcsec"], cfg["large_re_ellipticity_cap"]
    if re_cap_min is not None and ell_cap is not None:
        re_arcsec, ell = numeric(m, "re_total_arcsec"), numeric(m, "ellipticity_total")
        large_re = np.isfinite(re_arcsec) & (re_arcsec > re_cap_min)
        needs_cap = large_re & np.isfinite(ell) & (ell > ell_cap)
        if np.any(needs_cap):
            n_cap = int(np.sum(needs_cap))
            print(f"Capping ellipticity to <={ell_cap} for {n_cap:,} galaxies with Re>{re_cap_min} arcsec", flush=True)
        m.loc[large_re, "ellipticity_total"] = np.minimum(ell[large_re], ell_cap)
        m.loc[large_re, "ellipticity"] = m.loc[large_re, "ellipticity_total"]

    for band in cfg["out_bands"]:
        m[f"sb_{band}_total"] = component_surface_brightness(m, band, "total", cfg)
    return m
