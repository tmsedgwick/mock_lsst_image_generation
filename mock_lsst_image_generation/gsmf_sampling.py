"""Evolving star-forming and quiescent galaxy stellar mass functions (GSMFs), and stellar-mass draws from them.

Both are continuous ("continuity"-style, cf. Leja et al. 2020) fits to COSMOS2020 per-redshift-bin Schechter
parameters: each parameter is a polynomial in z (np.polyval order, highest power first), giving a closed-form
Phi(logM, z) in Mpc^-3 dex^-1 calibrated over 0.2 < z < 5.5 (evaluation redshifts are clamped to that range).
  * star-forming: double Schechter whose second component tapers off above z ~ 2.6;
  * quiescent: a shallow main component at all z plus a steep low-mass upturn that tapers off above z ~ 1.5.
"""

from typing import TypedDict

import numpy as np

from .config import CONFIG
from .cosmology import trapz

SF_LOGMSTAR = [0.025027, -0.266354, 0.659328, 10.501996]
SF_ALPHA1 = [0.007457, -0.065600, -1.331207]
SF_LOGPHI1 = [0.029046, -0.326406, -2.903912]
SF_LOGPHI2 = [-0.207248, -3.298156]
SF_ALPHA2 = -0.162
Q_LOGMSTAR = [0.020554, -0.041798, -0.307767, 11.103041]
Q_ALPHA_MAIN = [-0.080275, 0.435367, 0.153467, -0.824239]
Q_LOGPHI_MAIN = [0.067958, -0.528446, 0.515016, -3.109924]
Q_LOGPHI_UP = [-0.28553, -5.59659]
Q_ALPHA_UP = -2.0
LN10 = np.log(10.0)


def clamp_gsmf_redshift(z, cfg):
    """Clamp redshifts to the GSMF calibration range [gsmf_z_eval_min, gsmf_z_eval_max]."""
    z = np.asarray(z, float)
    if cfg["gsmf_z_eval_min"] is not None:
        z = np.maximum(z, cfg["gsmf_z_eval_min"])
    if cfg["gsmf_z_eval_max"] is not None:
        z = np.minimum(z, cfg["gsmf_z_eval_max"])
    return z


def sf_second_component_taper(z):
    """Smoothly switches off the low-mass second SF Schechter component above z ~ 2.6 (1 -> 0)."""
    return 0.5 * (1.0 - np.tanh((np.asarray(z, float) - 2.6) / 0.18))


def quiescent_upturn_taper(z):
    """Smoothly switches off the low-mass quiescent upturn above z ~ 1.5 (1 -> 0)."""
    return 0.5 * (1.0 - np.tanh((np.asarray(z, float) - 1.5) / 0.15))


def star_forming_gsmf(logM, z, cfg=CONFIG, second_component=True):
    """Star-forming Phi(logM, z) in Mpc^-3 dex^-1."""
    z = clamp_gsmf_redshift(z, cfg)
    m = 10.0 ** (np.asarray(logM, float) - np.polyval(SF_LOGMSTAR, z))
    phi = 10.0 ** np.polyval(SF_LOGPHI1, z) * m ** (1.0 + np.polyval(SF_ALPHA1, z))
    if second_component:
        phi = phi + 10.0 ** np.polyval(SF_LOGPHI2, z) * sf_second_component_taper(z) * m ** (1.0 + SF_ALPHA2)
    return LN10 * np.exp(-m) * phi


def quiescent_gsmf(logM, z, cfg=CONFIG, upturn=True):
    """Quiescent Phi(logM, z) in Mpc^-3 dex^-1."""
    z = clamp_gsmf_redshift(z, cfg)
    m = 10.0 ** (np.asarray(logM, float) - np.polyval(Q_LOGMSTAR, z))
    phi = 10.0 ** np.polyval(Q_LOGPHI_MAIN, z) * m ** (1.0 + np.polyval(Q_ALPHA_MAIN, z))
    if upturn:
        phi = phi + 10.0 ** np.polyval(Q_LOGPHI_UP, z) * quiescent_upturn_taper(z) * m ** (1.0 + Q_ALPHA_UP)
    return LN10 * np.exp(-m) * phi


def mass_cdfs_from_gsmf(phi, m_grid):
    """Per-redshift normalised mass CDFs of phi[z, logM] (uniform if phi integrates to zero) and the integrals."""
    phi = np.maximum(np.nan_to_num(np.asarray(phi, float), nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    incr = 0.5 * (phi[:, 1:] + phi[:, :-1]) * np.diff(m_grid)[None, :]
    cdf = np.c_[np.zeros(len(phi)), np.cumsum(incr, axis=1)]
    norm = cdf[:, -1].copy()
    good = norm > 0
    cdf[good] /= norm[good, None]
    if (~good).any():
        cdf[~good] = np.linspace(0.0, 1.0, len(m_grid))[None, :]
    cdf[:, -1] = 1.0
    return cdf, norm


class GSMFTables(TypedDict):
    """Tabulated GSMF quantities: arrays over the redshift grid z (and mass grid m), plus the expected galaxy count."""
    z: np.ndarray
    z_eval: np.ndarray
    m: np.ndarray
    n_sf: np.ndarray
    n_q: np.ndarray
    n_total: np.ndarray
    p_sf: np.ndarray
    cdf_sf: np.ndarray
    cdf_q: np.ndarray
    n_expected: float


def build_gsmf_sampling_tables(cfg, grids) -> GSMFTables:
    """Number densities, SF fraction and mass CDFs of both types on a (z, logM) grid, and the expected galaxy count
    in the survey light cone."""
    z_grid = np.linspace(max(cfg["z_min"], 1e-4), cfg["z_max"], cfg["gsmf_n_z"])
    m_grid = np.linspace(cfg["logm_min"], cfg["logm_max"], cfg["gsmf_n_m"])
    z_eval = clamp_gsmf_redshift(z_grid, cfg)
    cdf_sf, n_sf = mass_cdfs_from_gsmf(star_forming_gsmf(m_grid[None, :], z_eval[:, None], cfg), m_grid)
    cdf_q, n_q = mass_cdfs_from_gsmf(quiescent_gsmf(m_grid[None, :], z_eval[:, None], cfg), m_grid)
    n_total = n_sf + n_q
    n_expected = trapz(n_total * np.interp(z_grid, grids["z"], grids["dVdz"]), z_grid)
    p_sf = np.divide(n_sf, n_total, out=np.zeros_like(n_total), where=n_total > 0)
    return GSMFTables(z=z_grid, z_eval=z_eval, m=m_grid, n_sf=n_sf, n_q=n_q, n_total=n_total, p_sf=p_sf,
                      cdf_sf=cdf_sf, cdf_q=cdf_q, n_expected=float(n_expected))


def draw_gsmf_masses(z, kind, tables, rng):
    """Inverse-CDF logM draws at the nearest tabulated redshift, for kind 'sf' (star-forming) or quiescent."""
    z = np.asarray(z, float)
    out = np.full(len(z), np.nan)
    if len(z) == 0:
        return out
    cdfs, z_grid = tables["cdf_sf"] if kind == "sf" else tables["cdf_q"], tables["z"]
    idx = np.clip(np.searchsorted(z_grid, z), 1, len(z_grid) - 1)
    nearest = np.where(np.abs(z - z_grid[idx - 1]) <= np.abs(z - z_grid[idx]), idx - 1, idx)
    for iz in np.unique(nearest):
        s = nearest == iz
        out[s] = np.interp(rng.random(s.sum()), cdfs[iz], tables["m"])
    return out
