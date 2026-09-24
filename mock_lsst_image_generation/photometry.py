"""Magnitudes, fluxes, colours, surface brightness and the rest-frame <-> observed-frame SED projection.

Catalogue fluxes are in nJy (AB zeropoint 31.4). The same log-flux interpolation / K-correction pair is used in both
directions, so a donor projected back to its own redshift recovers its observed magnitudes:

    observed -> rest:  M(lambda_rest) = m((1+z) lambda_rest) - DM(z) + 2.5 log10(1+z)
    rest -> observed:  m(lambda_obs) = M(lambda_obs / (1+z)) + DM(z) - 2.5 log10(1+z)
"""

import numpy as np

from .cosmology import distance_modulus

ZP_NJY = 31.4
LSST_BAND_WAVE_A = dict(u=3670.0, g=4825.0, r=6222.0, i=7545.0, z=8679.0, y=9711.0)

# Rest-frame SED nodes (absolute-magnitude columns). UV nodes are needed for sensible observed ugrizy at z~3.
REST_NODES = [("M0900", 900.0), ("M1216", 1216.0), ("M1500", 1500.0), ("M1900", 1900.0), ("M2200", 2200.0),
              ("M2500", 2500.0), ("Mu", 3670.0), ("Mg", 4825.0), ("Mr", 6222.0), ("Mi", 7545.0), ("Mz", 8679.0),
              ("My", 9711.0)]
REST_COLS = [name for name, _ in REST_NODES]
REST_WAVE_A = np.array([wave for _, wave in REST_NODES], float)


def mag_to_flux_njy(mag):
    return 10.0 ** ((ZP_NJY - np.asarray(mag, float)) / 2.5)


def flux_njy_to_mag(flux):
    return ZP_NJY - 2.5 * np.log10(np.maximum(np.asarray(flux, float), 1e-30))


def abmag_to_relative_flux(mag):
    return 10.0 ** (-0.4 * np.asarray(mag, float))


def relative_flux_to_abmag(flux):
    return -2.5 * np.log10(np.maximum(np.asarray(flux, float), 1e-300))


def flux_colour(f1, f2):
    """Colour m1 - m2 from two fluxes (floored to avoid log of zero)."""
    return -2.5 * np.log10(np.maximum(f1, 1e-30) / np.maximum(f2, 1e-30))


def mean_surface_brightness(mag, re_arcsec, ellipticity=0.0, mode="circularized"):
    """Mean surface brightness within Re (mag/arcsec^2); 'circularized' ignores the axis ratio."""
    q = 1.0 if mode == "circularized" else np.clip(1.0 - np.asarray(ellipticity, float), 0.05, 1.0)
    area = 2.0 * np.pi * np.asarray(re_arcsec, float) ** 2 * q
    return np.asarray(mag, float) + 2.5 * np.log10(np.maximum(area, 1e-30))


def component_surface_brightness(m, band, component, cfg):
    """Mean surface brightness of a catalogue component ('total', 'bulge' or 'disc') in one band."""
    return mean_surface_brightness(m[f"mag_{band}_{component}"], m[f"re_{component}_arcsec"],
                                   m[f"ellipticity_{component}"], cfg["sb_re_mode"])


def sed_edge_distance(query_wave, source_wave):
    """How far (dex in wavelength) each query wavelength lies outside the sampled range; 0 inside it."""
    q, w = np.asarray(query_wave, float), np.asarray(source_wave, float)
    below, above = np.log10(w.min()) - np.log10(q), np.log10(q) - np.log10(w.max())
    return np.maximum(np.maximum(below, 0.0), np.maximum(above, 0.0))


def interpolate_sed_mags(wave_in, mag_in, wave_out, allow_extrapolation=True):
    """Interpolate AB mags linearly in log(flux) vs log(wavelength); extrapolate with the end slopes if allowed."""
    wave_in, mag_in, wave_out = np.asarray(wave_in, float), np.asarray(mag_in, float), np.asarray(wave_out, float)
    ok = np.isfinite(wave_in) & np.isfinite(mag_in)
    wave_in, mag_in = wave_in[ok], mag_in[ok]
    if len(wave_in) < 2:
        return np.full_like(wave_out, np.nan, dtype=float)
    order = np.argsort(wave_in)
    x = np.log10(wave_in[order])
    y = np.log10(np.maximum(abmag_to_relative_flux(mag_in[order]), 1e-300))
    xq = np.log10(wave_out)
    yq = np.interp(xq, x, y)
    if allow_extrapolation:
        left, right = xq < x[0], xq > x[-1]
        yq[left] = y[0] + (y[1] - y[0]) / max(x[1] - x[0], 1e-12) * (xq[left] - x[0])
        yq[right] = y[-1] + (y[-1] - y[-2]) / max(x[-1] - x[-2], 1e-12) * (xq[right] - x[-1])
    else:
        yq[(xq < x[0]) | (xq > x[-1])] = np.nan
    return relative_flux_to_abmag(10.0 ** yq)


def observed_to_rest_abs_mags(obs_mags, obs_waves, z, grids, cfg):
    """Observed apparent mags -> rest-frame absolute mags at REST_NODES, plus the worst SED extrapolation (dex)."""
    obs_mags, obs_waves = np.asarray(obs_mags, float), np.asarray(obs_waves, float)
    ok = np.isfinite(obs_mags) & np.isfinite(obs_waves)
    needed_obs_wave = REST_WAVE_A * (1.0 + z)
    interp_m = interpolate_sed_mags(obs_waves[ok], obs_mags[ok], needed_obs_wave, cfg["allow_sed_extrapolation"])
    edge = np.nanmax(sed_edge_distance(needed_obs_wave, obs_waves[ok])) if ok.any() else np.nan
    return interp_m - distance_modulus(z, grids) + 2.5 * np.log10(1.0 + z), edge


def rest_abs_to_observed_mags(rest_abs_mags, z, grids, cfg):
    """Rest-frame absolute mags at REST_NODES -> observed mags in cfg['out_bands'], plus the worst extrapolation."""
    needed_rest_wave = np.array([LSST_BAND_WAVE_A[b] for b in cfg["out_bands"]], float) / (1.0 + z)
    interp_M = interpolate_sed_mags(REST_WAVE_A, rest_abs_mags, needed_rest_wave, cfg["allow_sed_extrapolation"])
    obs_m = interp_M + distance_modulus(z, grids) - 2.5 * np.log10(1.0 + z)
    return obs_m, np.nanmax(sed_edge_distance(needed_rest_wave, REST_WAVE_A))
