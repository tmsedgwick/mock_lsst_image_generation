"""Flat LambdaCDM distance, volume and age grids, plus survey-frame geometry."""

import numpy as np

C_KMS = 299792.458
ARCSEC_PER_RAD = 206265.0
trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def hubble_efunction(z, Om0):
    """E(z) = H(z) / H0 for flat LambdaCDM."""
    return np.sqrt(Om0 * (1.0 + np.asarray(z, float)) ** 3 + (1.0 - Om0))


def build_cosmology_grids(cfg, nz=5000, zmax_age=30.0, nz_age=6000):
    """Tabulate comoving/luminosity/angular-diameter distance (Mpc), dV/dz over the frame (Mpc^3) and age (Gyr)."""
    z = np.linspace(0.0, cfg["z_max"], nz)
    invE = 1.0 / hubble_efunction(z, cfg["Om0"])
    dc = np.r_[0.0, np.cumsum(0.5 * (invE[1:] + invE[:-1]) * np.diff(z))] * C_KMS / cfg["H0"]
    area_deg2 = (cfg["npix"] * cfg["pixscale"] / 3600.0) ** 2
    omega_sr = area_deg2 * (np.pi / 180.0) ** 2
    dVdz = omega_sr * dc**2 * C_KMS / (cfg["H0"] * hubble_efunction(z, cfg["Om0"]))
    za = np.linspace(0.0, zmax_age, nz_age)
    integ = 1.0 / ((1.0 + za) * hubble_efunction(za, cfg["Om0"]))
    cum = np.r_[0.0, np.cumsum(0.5 * (integ[1:] + integ[:-1]) * np.diff(za))]
    return dict(z=z, dc_mpc=dc, dl_mpc=dc * (1.0 + z), da_mpc=dc / (1.0 + z), dVdz=dVdz, area_deg2=area_deg2, z_age=za,
                age_gyr=9.778 / (cfg["H0"] / 100.0) * (cum[-1] - cum))


def distance_modulus(z, grids):
    dl = np.interp(np.asarray(z, float), grids["z"], grids["dl_mpc"])
    return 5.0 * np.log10(np.maximum(dl, 1e-12)) + 25.0


def angular_diameter_distance_kpc(z, grids):
    return 1000.0 * np.interp(np.asarray(z, float), grids["z"], grids["da_mpc"])


def arcsec_to_kpc(size_arcsec, z, grids):
    return np.asarray(size_arcsec, float) * angular_diameter_distance_kpc(z, grids) / ARCSEC_PER_RAD


def kpc_to_arcsec(size_kpc, z, grids):
    return np.asarray(size_kpc, float) / np.maximum(angular_diameter_distance_kpc(z, grids), 1e-30) * ARCSEC_PER_RAD


def comoving_distance_from_redshift(z, grids):
    return np.interp(np.asarray(z, float), grids["z"], grids["dc_mpc"])


def redshift_from_comoving_distance(chi, grids):
    return np.interp(np.asarray(chi, float), grids["dc_mpc"], grids["z"])


def field_width_rad(cfg):
    """Angular side length of the square survey frame."""
    return cfg["npix"] * cfg["pixscale"] / ARCSEC_PER_RAD


def comoving_to_pixel(offset_mpc, chi, cfg):
    """Transverse comoving offset from the frame centre at comoving distance chi -> pixel coordinate."""
    return 0.5 * cfg["npix"] + (offset_mpc / chi) / (cfg["pixscale"] / ARCSEC_PER_RAD)
