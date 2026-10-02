"""Extended dwarf irregulars, ultra-diffuse galaxies (UDGs) and almost-dark galaxies, added in generous numbers.

These populations are rare and extended, so a mock frame of realistic abundance holds almost none; they are added on
top of the main catalogue so a detector sees enough of them to learn them. Their abundance (cfg['lsb_populations'],
per deg^2) is a training choice, not a measurement; each galaxy's properties are drawn within its population's
observed parameter space:
  * extended dIrr: Re 1-3 kpc, central g-band surface brightness mu0_g 22.5-24, star-forming, irregular;
  * UDG (van Dokkum et al. 2015: Re >= 1.5 kpc, mu0_g >= 24): Re 1.5-5 kpc, mu0_g 24-27; in the field mostly blue,
    irregular and star-forming (HI-bearing UDGs, Leisman et al. 2017), the rest red and smooth;
  * almost dark (ALFALFA, Cannon et al. 2015): HI clouds with a barely visible, very blue, irregular stellar
    counterpart, e.g. AGC 229385 (peak mu_g 26.5, Re ~ 2.4 kpc; Janowiecki et al. 2015) and AGC 229101 (peak mu_g 26.6,
    M* = 10^7.3 Msun, M_HI / M* ~ 100, two clumps; Leisman et al. 2021): Re 1.5-3.5 kpc, mu0_g 26-28, nearby
    (z < 0.03, as found by HI surveys), the bluest donors, strongly irregular. Their HI is not modelled.
Redshifts are volume-weighted within each population's range; Re (kpc) is log-uniform, mu0_g uniform, the Sersic index
uniform in SERSIC_N and the axis ratio in AXIS_RATIO. Each galaxy takes a real donor's SED (all colours together),
rescaled so the bulgeless disc has the drawn central surface brightness; its stellar mass follows from the donor's
mass-to-light ratio. Star-forming galaxies are Irr, with irregular modes that their clumps follow; quiescent UDGs are
smooth (En). Rows are flagged by lsb_population.
"""

import numpy as np
import pandas as pd
from scipy.special import gamma, gammaincinv

from .bulge_disc_decomposition import add_bulge_disc_components
from .hubble_types import draw_structure
from .photometry import REST_COLS
from .rest_frame_sed_sampling import project_to_observed_frame

SERSIC_N = (0.6, 1.2)
AXIS_RATIO = (0.45, 1.0)


def volume_weighted_redshifts(n, z_range, grids, rng):
    """n redshifts in z_range drawn in proportion to the survey's comoving volume."""
    z = grids["z"]
    inside = (z >= z_range[0]) & (z <= z_range[1])
    cdf = np.cumsum(grids["dVdz"][inside])
    return np.interp(rng.random(n) * cdf[-1], cdf, z[inside])


def sersic_total_mag(mu0, re_arcsec, n):
    """Total magnitude of a Sersic profile with central surface brightness mu0 (mag/arcsec^2), circularised half-light
    radius re_arcsec and index n: L = 2 pi n Gamma(2n) Re^2 I0 / b_n^(2n)."""
    b = gammaincinv(2 * n, 0.5)
    return mu0 - 2.5 * np.log10(2 * np.pi * n * gamma(2 * n) * np.asarray(re_arcsec) ** 2 / b ** (2 * n))


def donor_pool(donors, kind):
    """Row positions of donors usable for a population: star-forming dwarfs, the bluest third of them, or quiescent."""
    logm, ssfr = donors["logM"].to_numpy(float), donors["logsSFR"].to_numpy(float)
    star_forming = np.flatnonzero((logm <= 9.5) & (ssfr >= -10.0))
    if kind == "star_forming":
        return star_forming
    if kind == "bluest":
        g_r = (donors["Mg"] - donors["Mr"]).to_numpy(float)[star_forming]
        return star_forming[g_r <= np.percentile(g_r, 33)]
    if kind == "quiescent":
        return np.flatnonzero((logm <= 10.5) & (ssfr <= -10.5))
    raise ValueError(f"unknown donor pool {kind!r}")


def draw_population(name, spec, n, donors, grids, cfg, rng):
    """n galaxies of one population: redshift, size, shape, target mu0_g, star-forming or not, and a donor SED."""
    quiescent = rng.random(n) < spec["quiescent_frac"]
    picks = np.empty(n, int)
    for is_q, kind in [(False, spec["donors"]), (True, "quiescent")]:
        pool, rows = donor_pool(donors, kind), np.flatnonzero(quiescent == is_q)
        picks[rows] = pool[rng.integers(0, len(pool), len(rows))]
    d = donors.iloc[picks].reset_index(drop=True)
    npix = cfg["npix"]
    t = pd.DataFrame(dict(
        lsb_population=name, z=volume_weighted_redshifts(n, spec["z"], grids, rng),
        x_pix=rng.uniform(0, npix, n), y_pix=rng.uniform(0, npix, n), web_env="field",
        Re_kpc=10 ** rng.uniform(*np.log10(spec["re_kpc"]), n), mu0_g=rng.uniform(*spec["mu0_g"], n),
        sersic_n=rng.uniform(*SERSIC_N, n), ellipticity=1 - rng.uniform(*AXIS_RATIO, n), pa_deg=rng.uniform(0, 180, n),
        type=np.where(quiescent, "passive", "star_forming"), kind=np.where(quiescent, "pa", "sf"), BT=0.0,
        logsSFR=d["logsSFR"].to_numpy(float), donor_index=picks, donor_logM=d["logM"].to_numpy(float),
        donor_z=d["z_cosmos"].to_numpy(float), empirical_edge_distance=0.0))
    t[REST_COLS] = d[REST_COLS].to_numpy(float)
    t["logRe_kpc"], t["ellipticity_total"] = np.log10(t["Re_kpc"]), t["ellipticity"]
    return t


def scale_to_surface_brightness(t, cfg, grids):
    """Shift each donor SED (all bands together) so the disc has its drawn mu0_g; stellar mass keeps the donor M/L."""
    obs = project_to_observed_frame(t, cfg, grids)
    shift = sersic_total_mag(t["mu0_g"], obs["re_total_arcsec"], t["sersic_n"]) - obs["mag_g_total"]
    t[REST_COLS] = t[REST_COLS].add(shift, axis=0)
    t["logM"] = t["donor_logM"] - 0.4 * shift
    t["logSFR"] = t["logM"] + t["logsSFR"]
    return project_to_observed_frame(t, cfg, grids)


def give_structure(t, rng, irregularity):
    """Hubble type and structure: star-forming galaxies are Irr with the population's irregularity, quiescent En."""
    en = np.clip(np.round(10 * t["ellipticity_total"].to_numpy(float)), 0, 7).astype(int)
    types = np.where(t["type"] == "star_forming", "Irr", np.char.add("E", en.astype(str))).astype(object)
    s = draw_structure(types, rng).set_index(t.index)
    irregular = types == "Irr"
    s.loc[irregular, "irregularity"] = np.clip(rng.normal(*irregularity, irregular.sum()), 0.1, 0.95)
    t["hubble_type"] = types
    t[s.columns] = s
    return t


def add_lsb_galaxies(mock, donors, cfg, phys, grids, rng):
    """Catalogue with each population of cfg['lsb_populations'] appended (Poisson numbers for the frame area), as
    bulgeless discs ready for rendering, clumps and the rest of the pipeline; new ids follow the existing ones."""
    tables = []
    for name, spec in cfg["lsb_populations"].items():
        n = rng.poisson(spec["per_deg2"] * grids["area_deg2"])
        if n == 0:
            continue
        t = scale_to_surface_brightness(draw_population(name, spec, n, donors, grids, cfg, rng), cfg, grids)
        t = add_bulge_disc_components(t, cfg, phys, rng)
        t["n_disc"] = t["sersic_n"]
        if "hubble_type" in mock.columns:
            t = give_structure(t, rng, spec["irregularity"])
        tables.append(t)
    if not tables:
        return mock
    new = pd.concat(tables, ignore_index=True)
    new["id"] = len(mock) + np.arange(len(new))
    assert mock["id"].tolist() == list(range(len(mock))), "ids must be 0..N-1 before LSB galaxies are appended"
    print(f"Added LSB galaxies: {new['lsb_population'].value_counts().to_dict()}", flush=True)
    return pd.concat([mock, new], ignore_index=True)
