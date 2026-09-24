"""Bulge + disc structure: B/T, per-band light fractions, component sizes, ellipticities and Sersic indices."""

from typing import cast

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammainc, gammaincinv

from .photometry import component_surface_brightness, flux_colour, flux_njy_to_mag

SERSIC_B4 = float(gammaincinv(8.0, 0.5))  # b_n for a de Vaucouleurs (n = 4) bulge
SERSIC_B1 = float(gammaincinv(2.0, 0.5))  # b_n for an exponential (n = 1) disc
_RE_RATIO_TABLES = {}


def mean_bulge_to_total(logM, logSFR, z):
    """Mean stellar-mass B/T versus mass, SFR and redshift (logistic polynomial derived from Dimauro et al. 2022)."""
    m = (np.asarray(logM, float) - 10.5) / 1.5
    s = (np.asarray(logSFR, float) - 0.5) / 2.5
    t = np.asarray(z, float) - 1.0
    eta = (0.04884986 + 1.61120039 * m - 1.79274182 * s - 0.06611559 * t + 0.40782181 * t**2 - 0.24717860 * m * t
           - 0.91687005 * s * t - 0.04682984 * m**2 + 0.75914923 * s**2 - 0.66947451 * m * s + 0.23730185 * m**2 * t
           + 0.23472053 * s**2 * t - 0.52307049 * m * s * t)
    return 1.0 / (1.0 + np.exp(-eta))


def draw_bulge_to_total(logM, logSFR, z, rng, phys):
    """Stellar-mass B/T from a Beta distribution about mean_bulge_to_total (concentration phys['bt_conc'])."""
    mu = np.clip(mean_bulge_to_total(logM, logSFR, z), 0.02, 0.98)
    return rng.beta(mu * phys["bt_conc"], (1.0 - mu) * phys["bt_conc"])


def mass_to_light_bulge_fraction(bt_mass, ml_ratio):
    """Light B/T from mass B/T, given ml_ratio = (M/L)_bulge / (M/L)_disc."""
    bt_mass = np.clip(np.asarray(bt_mass, float), 1e-4, 1 - 1e-4)
    return bt_mass / (bt_mass + ml_ratio * (1.0 - bt_mass))


def sersic_enclosed_fraction(r, re, n, bn):
    """Fraction of a Sersic profile's light inside radius r."""
    return gammainc(2.0 * n, bn * (r / re) ** (1.0 / n))


def bulge_disc_half_light_radius(bt, re_bulge, re_disc):
    """Half-light radius of an n=4 bulge (light fraction bt) plus an n=1 disc."""
    if bt >= 0.999:
        return re_bulge
    if bt <= 0.001:
        return re_disc

    def enclosed_minus_half(r):
        return bt * sersic_enclosed_fraction(r, re_bulge, 4.0, SERSIC_B4) + (1.0 - bt) * sersic_enclosed_fraction(
            r, re_disc, 1.0, SERSIC_B1) - 0.5

    return cast(float, brentq(enclosed_minus_half, 1e-3 * min(re_bulge, re_disc), 30.0 * max(re_bulge, re_disc),
                              xtol=1e-5))


def total_to_disc_re_ratio(bt, bulge_re_frac):
    """Re_total / Re_disc as a function of light B/T when Re_bulge = bulge_re_frac * Re_disc (tabulated once)."""
    key = round(float(bulge_re_frac), 4)
    if key not in _RE_RATIO_TABLES:
        grid = np.linspace(0.0, 1.0, 201)
        _RE_RATIO_TABLES[key] = (grid, np.array([bulge_disc_half_light_radius(b, bulge_re_frac, 1.0) for b in grid]))
    grid, ratio = _RE_RATIO_TABLES[key]
    return np.interp(np.asarray(bt, float), grid, ratio)


def draw_spheroid_ellipticity(n, rng, phys):
    p = phys["bulge_ba"]
    return 1.0 - np.clip(rng.normal(p["mean"], p["sig"], n), p["lo"], p["hi"])


def draw_coupled_bulge_ellipticity(q_disc, rng, phys, cfg):
    """Bulge ellipticity: an intrinsic spheroid draw, pulled towards (disc q + offset) for inclined/edge-on discs, so
    the bulge is flatter when the disc is flatter but never as flat as the disc."""
    q_disc = np.clip(np.asarray(q_disc, float), 0.08, 1.0)
    q_sph = 1.0 - draw_spheroid_ellipticity(len(q_disc), rng, phys)
    q0 = cfg["bulge_edgeon_q0"]
    w = cfg["bulge_disc_q_coupling"] * np.clip((q0 - q_disc) / max(q0 - 0.08, 1e-6), 0.0, 1.0)
    q_bulge = (1.0 - w) * q_sph + w * (q_disc + cfg["bulge_disc_q_offset"])
    return 1.0 - np.clip(q_bulge, cfg["bulge_q_min"], cfg["bulge_q_max"])


def add_bulge_disc_components(mock, cfg, phys, rng):
    """Split each galaxy into bulge + disc: per-band fluxes from light B/T, component Re that reproduce the total Re,
    ellipticities consistent with the total, Sersic indices, surface brightnesses and adjacent-band colours."""
    m = mock.copy()
    bands = cfg["out_bands"]
    bt_mass = np.clip(m["BT"].to_numpy(float), 0.0, 1.0)
    # Stellar-mass B/T -> light B/T; the r-band light B/T sets the geometry.
    bt_struct = np.clip(mass_to_light_bulge_fraction(bt_mass, phys["bulge_ml_ratio"]["r"]), 0.0, 1.0)
    has_disc = (1.0 - bt_struct) > 1e-4
    m["BT_mass"], m["BT_light_struct"] = bt_mass, bt_struct
    for band in bands:
        flux_total = m[f"flux_{band}_total"].to_numpy(float)
        bt_light = np.clip(mass_to_light_bulge_fraction(bt_mass, phys["bulge_ml_ratio"][band]), 0.0, 1.0)
        flux_bulge = bt_light * flux_total
        flux_disc = np.where(has_disc, flux_total - flux_bulge, 0.0)
        m[f"BT_light_{band}"] = bt_light
        m[f"flux_{band}_bulge"], m[f"mag_{band}_bulge"] = flux_bulge, flux_njy_to_mag(flux_bulge)
        m[f"flux_{band}_disc"], m[f"mag_{band}_disc"] = flux_disc, flux_njy_to_mag(flux_disc)

    frac, re_total = cfg["bulge_re_frac"], m["re_total_arcsec"].to_numpy(float)
    re_disc = re_total / total_to_disc_re_ratio(np.clip(bt_struct, 0.001, 0.999), frac)
    m["re_disc_arcsec"] = np.where(has_disc, re_disc, np.nan)
    m["re_bulge_arcsec"] = np.where(has_disc, frac * re_disc, re_total)

    ell_total = m["ellipticity_total"].to_numpy(float)
    q_total = 1.0 - ell_total
    # First pass: approximate disc flattening assuming the old round-bulge model. Second pass: the bulge shape
    # partially follows that inferred disc inclination.
    ell_bulge0 = np.where(has_disc, draw_spheroid_ellipticity(len(m), rng, phys), ell_total)
    q_disc0 = np.clip((q_total - bt_struct * (1.0 - ell_bulge0)) / np.maximum(1.0 - bt_struct, 1e-3), 0.08, 1.0)
    ell_bulge = np.where(has_disc, draw_coupled_bulge_ellipticity(q_disc0, rng, phys, cfg), ell_total)
    q_disc = np.clip((q_total - bt_struct * (1.0 - ell_bulge)) / np.maximum(1.0 - bt_struct, 1e-3), 0.08, 1.0)
    m["ellipticity"], m["ellipticity_bulge"] = ell_total, ell_bulge
    m["ellipticity_disc"] = np.where(has_disc, 1.0 - q_disc, np.nan)
    m["n_disc"] = np.where(has_disc, 1.0, np.nan)
    # Low-B/T systems get pseudobulge-like Sersic indices.
    m["n_bulge"] = np.where(bt_struct < 0.5, rng.uniform(1.0, 2.0, len(m)), rng.uniform(2.5, 4.0, len(m)))

    for band in bands:
        m[f"sb_{band}_bulge"] = component_surface_brightness(m, band, "bulge", cfg)
        m[f"sb_{band}_disc"] = np.where(has_disc, component_surface_brightness(m, band, "disc", cfg), np.nan)
    for b1, b2 in zip(bands[:-1], bands[1:]):
        m[f"{b1}{b2}_total"] = flux_colour(m[f"flux_{b1}_total"], m[f"flux_{b2}_total"])
        m[f"{b1}{b2}_bulge"] = flux_colour(m[f"flux_{b1}_bulge"], m[f"flux_{b2}_bulge"])
        m[f"{b1}{b2}_disc"] = np.where(has_disc, flux_colour(m[f"flux_{b1}_disc"], m[f"flux_{b2}_disc"]), np.nan)
    return m


def refresh_disc_photometry(m, cfg):
    """Recompute disc mags, surface brightnesses and colours in place after light is moved out of the disc."""
    bands = cfg["out_bands"]
    for b in bands:
        m[f"mag_{b}_disc"] = flux_njy_to_mag(m[f"flux_{b}_disc"])
        sb = component_surface_brightness(m, b, "disc", cfg)
        m[f"sb_{b}_disc"] = np.where(np.isfinite(m["re_disc_arcsec"]), sb, np.nan)
    for b1, b2 in zip(bands[:-1], bands[1:]):
        colour = flux_colour(m[f"flux_{b1}_disc"], m[f"flux_{b2}_disc"])
        m[f"{b1}{b2}_disc"] = np.where(m[f"flux_{b2}_disc"] > 0, colour, np.nan)
