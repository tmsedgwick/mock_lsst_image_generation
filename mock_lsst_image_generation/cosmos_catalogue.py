"""Load the COSMOS2025 catalogue and map its columns onto the standard names used for donor selection."""

from pathlib import Path

import numpy as np
import pandas as pd

from .utils import numeric

# Row/column subset of cosmos2025_cat.npz shipped with the repo; gives identical catalogues at the default CONFIG
# (it keeps every row that survives the donor pre-cuts or the low-z sSFR PDF cuts).
DEFAULT_COSMOS_PATH = Path(__file__).resolve().parent.parent / "data" / "cosmos2025_subset.npz"

COSMOS2025_MAG_MAP = {
    "u": "mag_model_cfht-u", "g": "mag_model_hsc-g", "r": "mag_model_hsc-r", "i": "mag_model_hsc-i",
    "z": "mag_model_hsc-z", "y": "mag_model_hsc-y", "J": "mag_model_uvista-j", "H": "mag_model_uvista-h",
    "Ks": "mag_model_uvista-ks", "IRAC1": "mag_model_irac-ch1", "IRAC2": "mag_model_irac-ch2",
    "F115W": "mag_model_f115w", "F150W": "mag_model_f150w", "F277W": "mag_model_f277w", "F444W": "mag_model_f444w",
}
COSMOS2025_MAG_ERR_MAP = {"g": "mag_err_model_hsc-g", "r": "mag_err_model_hsc-r", "i": "mag_err_model_hsc-i"}
COSMOS2025_BASE_COLUMNS = [
    "id", "ra", "dec", "zfinal", "type", "mass_med", "mass_l68", "mass_u68", "sfr_med", "sfr_l68", "sfr_u68",
    "ssfr_med", "ssfr_l68", "ssfr_u68", "radius_sersic", "axratio_sersic", "sersic", "fwhm", "kron1_a",
    "flag_star", "flag_blend", "warn_flag", "flag_chandra",
]
COSMOS2025_COLUMNS = list(dict.fromkeys([*COSMOS2025_BASE_COLUMNS, *COSMOS2025_MAG_MAP.values(),
                                         *COSMOS2025_MAG_ERR_MAP.values()]))
COSMOS2025_FLOAT_SENTINELS = (-999.0, -998.0, -99.9, -99.0, 998.0, 999.0)

# Observed bands usable as SED constraints: name -> (effective wavelength in A, candidate columns). Every band found is
# used, so add more columns here if available.
OBS_BAND_ALIASES = {
    "FUV": (1530.0, ["m_FUV", "FUV_MAG", "GALEX_FUV_MAG", "CFHT_FUV_MAG"]),
    "NUV": (2310.0, ["m_NUV", "NUV_MAG", "GALEX_NUV_MAG", "CFHT_NUV_MAG"]),
    "u": (3670.0, ["m_u", "mag_model_cfht-u", "HSC_u_MAG_AUTO", "CFHT_u_MAG_AUTO", "u_MAG_AUTO"]),
    "g": (4825.0, ["m_g", "mag_model_hsc-g", "HSC_g_MAG_AUTO", "CFHT_g_MAG_AUTO", "g_MAG_AUTO"]),
    "r": (6222.0, ["m_r", "mag_model_hsc-r", "HSC_r_MAG_AUTO", "CFHT_r_MAG_AUTO", "r_MAG_AUTO"]),
    "i": (7545.0, ["m_i", "mag_model_hsc-i", "HSC_i_MAG_AUTO", "CFHT_i_MAG_AUTO", "i_MAG_AUTO"]),
    "z": (8679.0, ["m_z", "mag_model_hsc-z", "HSC_z_MAG_AUTO", "CFHT_z_MAG_AUTO", "z_MAG_AUTO"]),
    "y": (9711.0, ["m_y", "mag_model_hsc-y", "UVISTA_Y_MAG_AUTO", "mag_model_uvista-y", "y_MAG_AUTO"]),
    "J": (12500.0, ["m_J", "mag_model_uvista-j", "UVISTA_J_MAG_AUTO", "J_MAG_AUTO"]),
    "H": (16300.0, ["m_H", "mag_model_uvista-h", "UVISTA_H_MAG_AUTO", "H_MAG_AUTO"]),
    "Ks": (21500.0, ["m_Ks", "mag_model_uvista-ks", "UVISTA_Ks_MAG_AUTO", "UVISTA_K_MAG_AUTO", "Ks_MAG_AUTO",
                     "K_MAG_AUTO"]),
    "F115W": (11500.0, ["m_F115W", "mag_model_f115w", "mag_auto_f115w"]),
    "F150W": (15000.0, ["m_F150W", "mag_model_f150w", "mag_auto_f150w"]),
    "F277W": (27700.0, ["m_F277W", "mag_model_f277w", "mag_auto_f277w"]),
    "F444W": (44400.0, ["m_F444W", "mag_model_f444w", "mag_auto_f444w"]),
    "IRAC1": (35600.0, ["m_IRAC1", "mag_model_irac-ch1", "SPLASH_1_MAG", "IRAC_CH1_MAG", "ch1_mag"]),
    "IRAC2": (45100.0, ["m_IRAC2", "mag_model_irac-ch2", "SPLASH_2_MAG", "IRAC_CH2_MAG", "ch2_mag"]),
}
OBS_MAG_ERR_ALIASES = {
    "g": ["obs_err_g", "merr_g", "mag_err_model_hsc-g", "HSC_g_MAGERR_AUTO", "g_MAGERR_AUTO"],
    "r": ["obs_err_r", "merr_r", "mag_err_model_hsc-r", "HSC_r_MAGERR_AUTO", "r_MAGERR_AUTO"],
    "i": ["obs_err_i", "merr_i", "mag_err_model_hsc-i", "HSC_i_MAGERR_AUTO", "i_MAGERR_AUTO"],
}


def mask_sentinel_values(series):
    """Numeric copy of a COSMOS2025 column with missing-value sentinels (e.g. -99, 999) set to NaN."""
    s = pd.to_numeric(series, errors="coerce")
    values, bad = s.to_numpy(float), np.zeros(len(s), dtype=bool)
    for sentinel in COSMOS2025_FLOAT_SENTINELS:
        bad |= np.isclose(values, sentinel, rtol=0.0, atol=1e-8)
    return s.mask(bad)


def load_cosmos2025_catalogue(path=DEFAULT_COSMOS_PATH, columns=COSMOS2025_COLUMNS):
    """Read the COSMOS2025 .npz, mask sentinels, and add convenience columns (z, logM, sizes in arcsec, m_<band>)."""
    with np.load(path, allow_pickle=True) as npz:
        missing = [col for col in columns if col not in npz.files]
        if missing:
            raise KeyError(f"Missing COSMOS2025 columns: {missing}")
        data = {}
        for col in columns:
            arr = npz[col]
            data[col] = np.char.decode(arr, "utf-8") if arr.dtype.kind == "S" else arr
    c = pd.DataFrame(data)
    for col in c.columns:
        if pd.api.types.is_float_dtype(c[col]):
            c[col] = mask_sentinel_values(c[col])
    # Older (COSMOS2020-era) convenience names mapped onto the COSMOS2025 LePhare/model columns.
    for name, source in [("C2025_ID", "id"), ("z", "zfinal"), ("logmass", "mass_med"), ("logM", "mass_med"),
                         ("LogSFR", "sfr_med"), ("logSFR", "sfr_med"), ("logsSFR", "ssfr_med")]:
        c[name] = c[source]
    c["radius_sersic_arcsec"] = c["radius_sersic"] * 3600.0
    c["half_light_radius_arcsec"] = c["radius_sersic_arcsec"]
    c["ellipticity"] = 1.0 - numeric(c, "axratio_sersic").clip(0.05, 1.0)
    for band, col in COSMOS2025_MAG_MAP.items():
        c[f"m_{band}"] = c[col]
    for band, col in COSMOS2025_MAG_ERR_MAP.items():
        c[f"merr_{band}"] = c[col]
    re = numeric(c, "half_light_radius_arcsec")
    c["mu_r"] = c["m_r"] + 2.5 * np.log10(2.0 * np.pi * np.maximum(re, 1e-6) ** 2)
    return c.replace([np.inf, -np.inf], np.nan)


def first_existing_column(df, options, required_name=None):
    """First of options present in df; raise KeyError if none is and required_name is given."""
    hit = next((col for col in options if col in df.columns), None)
    if hit is None and required_name is not None:
        raise KeyError(f"Missing {required_name}. Tried: {options}")
    return hit


def standardise_cosmos_columns(cosmos, cfg):
    """Copy of a COSMOS table with standard columns z, logM, logSFR, logsSFR, Re_arcsec, ellipticity, obs_<band>
    and obs_err_<g,r,i>, plus the list of (band, wavelength_A, column) observed bands found.

    With cfg['cosmos_galaxies_only'], only LePhare galaxies (type 0) are kept: QSOs (type 2) and stars (type 1) have
    SEDs no galaxy has, and galaxy-template masses that mean nothing.
    """
    c = cosmos.copy()
    if cfg["cosmos_galaxies_only"]:
        if "type" not in c.columns:
            raise KeyError("cosmos_galaxies_only needs the LePhare 'type' column (0 = galaxy, 1 = star, 2 = QSO).")
        c = c[numeric(c, "type").eq(0)]
    aliases = {
        "logM": [cfg["mass_col"], "mass_med", "mass_minchi2", "logmass", "lmass", "lp_mass_med", "lp_mass_best",
                 "ez_mass"],
        "z": [cfg["z_col"], "zfinal", "z", "lp_zBEST", "ez_z_phot"],
        "Re_arcsec": [cfg["re_arcsec_col"], "radius_sersic_arcsec", "half_light_radius_arcsec",
                      "half_light_radius_arcsecs"],
    }
    for target, options in aliases.items():
        c[target] = numeric(c, first_existing_column(c, options, target))
    ssfr_options = [x for x in [cfg["ssfr_col"], "ssfr_med", "logsSFR", "ssfr_minchi2"] if x is not None]
    sfr_options = [x for x in [cfg["sfr_col"], "sfr_med", "LogSFR", "logSFR", "sfr_minchi2"] if x is not None]
    ssfr_col, sfr_col = first_existing_column(c, ssfr_options), first_existing_column(c, sfr_options)
    if ssfr_col is not None:
        c["logsSFR"] = numeric(c, ssfr_col)
    if sfr_col is not None:
        sfr = numeric(c, sfr_col)
        c["logSFR"] = sfr if cfg["sfr_is_log"] else np.log10(np.clip(sfr, 1e-8, None))
    elif ssfr_col is not None:
        c["logSFR"] = c["logM"] + c["logsSFR"]
    else:
        raise KeyError(f"Missing COSMOS SFR/sSFR columns. Tried SFR={sfr_options}, sSFR={ssfr_options}")
    if ssfr_col is None:
        c["logsSFR"] = c["logSFR"] - c["logM"]

    ell_col = first_existing_column(c, [cfg["ellipticity_col"], "ellipticity"])
    if ell_col is not None:
        c["ellipticity"] = numeric(c, ell_col)
    elif "axratio_sersic" in c.columns:
        c["ellipticity"] = 1.0 - numeric(c, "axratio_sersic").clip(0.05, 1.0)
    elif {"ACS_A_WORLD", "ACS_B_WORLD"}.issubset(c.columns):
        a, b = numeric(c, "ACS_A_WORLD"), numeric(c, "ACS_B_WORLD")
        c["ellipticity"] = 1.0 - np.minimum(a, b) / np.maximum(a, b)
    elif {"ERRX2_IMAGE", "ERRY2_IMAGE"}.issubset(c.columns):
        x2, y2 = numeric(c, "ERRX2_IMAGE"), numeric(c, "ERRY2_IMAGE")
        c["ellipticity"] = 1.0 - np.sqrt(np.minimum(x2, y2) / np.maximum(x2, y2))
    else:
        c["ellipticity"] = np.nan

    found = []
    for name, (wave, options) in OBS_BAND_ALIASES.items():
        hit = first_existing_column(c, options)
        if hit is not None:
            c[f"obs_{name}"] = numeric(c, hit)
            found.append((name, wave, f"obs_{name}"))
    for name, options in OBS_MAG_ERR_ALIASES.items():
        hit = first_existing_column(c, options)
        if hit is not None:
            c[f"obs_err_{name}"] = numeric(c, hit)
    if len(found) < cfg["min_training_bands"]:
        raise ValueError(f"Only found {len(found)} photometric bands. Need at least {cfg['min_training_bands']}.")
    return c, found
