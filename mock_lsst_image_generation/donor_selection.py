"""Quality cuts selecting reliable COSMOS 'donor' galaxies, and safety cuts rejecting broken rest-frame SEDs."""

import numpy as np
import pandas as pd

from .photometry import REST_COLS
from .utils import combined_mask, numeric, print_cut_summary, within

OBS_LSST_COLS = [f"obs_{b}" for b in "ugrizy"]


def consecutive_triplets(cols):
    """(a, b, c), (b, c, d), ... for the colour-curvature check."""
    return list(zip(cols, cols[1:], cols[2:]))


def colour_limits_mask(df, limits):
    """Rows whose colours (blue - red) lie in [lo, hi] for each (blue, red, lo, hi) present; also returns n used."""
    ok, n_used = pd.Series(True, index=df.index), 0
    for blue, red, lo, hi in limits:
        if blue in df.columns and red in df.columns:
            colour = numeric(df, blue) - numeric(df, red)
            ok &= np.isfinite(colour) & colour.between(lo, hi)
            n_used += 1
    return ok, n_used


def colour_curvature_mask(df, triplets, max_curvature):
    """Rows without sharp three-band SED zig-zags, |m_blue - 2 m_mid + m_red| < max_curvature; also returns n used.

    The curvature is large when the middle band is a spike or trough relative to its neighbours, which catches
    purple/green RGB artefacts that pass simple adjacent-colour limits.
    """
    ok, n_used = pd.Series(True, index=df.index), 0
    if max_curvature is None:
        return ok, 0
    for blue, mid, red in triplets:
        if {blue, mid, red}.issubset(df.columns):
            curv = numeric(df, blue) - 2.0 * numeric(df, mid) + numeric(df, red)
            ok &= np.isfinite(curv) & (np.abs(curv) < max_curvature)
            n_used += 1
    return ok, n_used


def robust_colour_plane_mask(df, xcol, ycol, max_sigma, base_mask=None, min_fit=300):
    """Clip outliers in a two-colour plane with a robust covariance ellipse (fit to base_mask rows, pre-clipped)."""
    if max_sigma is None:
        return pd.Series(True, index=df.index)
    if xcol not in df.columns or ycol not in df.columns:
        raise KeyError(f"Missing colour-plane columns: {xcol}, {ycol}")
    x, y = numeric(df, xcol), numeric(df, ycol)
    finite = np.isfinite(x) & np.isfinite(y)
    fit = finite if base_mask is None else (finite & np.asarray(base_mask, dtype=bool))
    if int(fit.sum()) < min_fit:
        fit = finite
    if int(fit.sum()) < 10:
        return pd.Series(finite, index=df.index)
    vals = np.column_stack([x.to_numpy(float), y.to_numpy(float)])
    fit_vals = vals[fit]
    centre = np.nanmedian(fit_vals, axis=0)
    q16, q84 = np.nanpercentile(fit_vals, [16, 84], axis=0)
    scale = 0.5 * (q84 - q16)
    scale = np.where(scale > 1e-3, scale, np.nanstd(fit_vals, axis=0))
    scale = np.where(scale > 1e-3, scale, 1.0)
    preclip = fit & np.all(np.abs((vals - centre) / scale) < 3.5, axis=1)
    if int(preclip.sum()) < min_fit:
        preclip = fit
    cov = np.cov(vals[preclip].T)
    if not np.all(np.isfinite(cov)):
        cov = np.diag(scale ** 2)
    delta = vals - centre
    d2 = np.einsum("ij,jk,ik->i", delta, np.linalg.pinv(cov + np.eye(2) * 1e-4), delta)
    return pd.Series(finite & np.isfinite(d2) & (d2 <= float(max_sigma) ** 2), index=df.index)


def apply_donor_quality_cuts(c, obs_cols, cfg):
    """Keep COSMOS galaxies with valid z/mass/sSFR, bright well-measured photometry, sane shapes and typical colours.

    Cuts are applied in order; the g-r/r-i colour-plane ellipse is fitted to the rows surviving all earlier cuts.
    """
    z = numeric(c, "z")
    checks = [
        ("redshift", within(z, None, cfg["cosmos_training_z_max"]) & (z > 0)),
        ("logM", within(numeric(c, "logM"), cfg["cosmos_training_logm_min"], cfg["cosmos_training_logm_max"])),
        ("logsSFR", within(numeric(c, "logsSFR"), cfg["cosmos_training_logssfr_min"],
                           cfg["cosmos_training_logssfr_max"])),
    ]
    mag_max = cfg["cosmos_training_mag_max"]
    if mag_max is not None and len(obs_cols):
        obs = c[obs_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        finite, below = np.isfinite(obs), obs < mag_max
        if cfg["cosmos_training_require_all_obs_bands"]:
            mag_ok = finite.all(axis=1) & below.all(axis=1)
        else:
            mag_ok = ((~finite) | below).all(axis=1) & ((finite & below).sum(axis=1) >= cfg["min_training_bands"])
        checks.append((f"all observed mags < {mag_max}", pd.Series(mag_ok, index=c.index)))
    r_min, r_max = cfg["cosmos_training_r_mag_min"], cfg["cosmos_training_r_mag_max"]
    if r_min is not None or r_max is not None:
        if "obs_r" not in c.columns:
            raise KeyError("COSMOS training r-band magnitude cut requested, but obs_r is unavailable.")
        checks.append((f"{r_min} < observed r < {r_max}", within(numeric(c, "obs_r"), r_min, r_max, inclusive=False)))
    r_err_max = cfg["cosmos_training_r_magerr_max"]
    if r_err_max is not None:
        if "obs_err_r" not in c.columns:
            raise KeyError("COSMOS training r-band magnitude-error cut requested, but obs_err_r is unavailable.")
        checks.append((f"observed r mag error < {r_err_max}",
                       within(numeric(c, "obs_err_r"), None, r_err_max, inclusive=False)))
    max_re_over_kron = cfg["cosmos_max_sersic_re_over_kron"]
    if max_re_over_kron is not None:
        if "kron1_a" not in c.columns:
            raise KeyError("cosmos_max_sersic_re_over_kron needs the COSMOS2025 Kron semi-major axis column kron1_a.")
        re_over_kron = numeric(c, "Re_arcsec") / numeric(c, "kron1_a")
        checks.append((f"Sersic Re <= {max_re_over_kron} x Kron semi-major axis", re_over_kron <= max_re_over_kron))
    ell_min, ell_max = cfg["cosmos_training_ellipticity_min"], cfg["cosmos_training_ellipticity_max"]
    if ell_min is not None or ell_max is not None:
        checks.append((f"{ell_min} < ellipticity < {ell_max}",
                       within(numeric(c, "ellipticity"), ell_min, ell_max, inclusive=False)))
    colour_ok, n_used = colour_limits_mask(c, cfg["cosmos_apparent_colour_limits"] or [])
    if n_used:
        checks.append(("apparent adjacent colours", colour_ok))
    curvature_ok, n_used = colour_curvature_mask(c, consecutive_triplets(OBS_LSST_COLS),
                                                 cfg["cosmos_observed_colour_curvature_max"])
    if n_used:
        checks.append(("observed colour curvature", curvature_ok))

    if {"obs_g", "obs_r", "obs_i"}.issubset(c.columns):
        c["obs_g_minus_r"] = numeric(c, "obs_g") - numeric(c, "obs_r")
        c["obs_r_minus_i"] = numeric(c, "obs_r") - numeric(c, "obs_i")
        err_max = cfg["cosmos_gr_ri_colour_error_max"]
        if err_max is not None:
            missing = sorted({"obs_err_g", "obs_err_r", "obs_err_i"} - set(c.columns))
            if missing:
                raise KeyError(f"Missing COSMOS HSC g/r/i magnitude-error columns for colour-error cut: {missing}")
            eg, er, ei = numeric(c, "obs_err_g"), numeric(c, "obs_err_r"), numeric(c, "obs_err_i")
            sig_gr, sig_ri = np.sqrt(eg ** 2 + er ** 2), np.sqrt(er ** 2 + ei ** 2)
            checks.append((f"sigma(g-r), sigma(r-i) < {err_max}", np.isfinite(sig_gr) & np.isfinite(sig_ri)
                           & (sig_gr < err_max) & (sig_ri < err_max)))
        plane_sigma = cfg["cosmos_gr_ri_colour_plane_sigma"]
        if plane_sigma is not None:
            plane_ok = robust_colour_plane_mask(c, "obs_g_minus_r", "obs_r_minus_i", plane_sigma,
                                                base_mask=combined_mask(checks, c.index),
                                                min_fit=cfg["cosmos_gr_ri_colour_plane_min_fit"])
            checks.append((f"g-r/r-i colour plane < {plane_sigma} sigma", plane_ok))

    keep = combined_mask(checks, c.index)
    if not keep.all():
        print_cut_summary(f"COSMOS donor start cuts kept {int(keep.sum()):,} / {len(c):,} rows", checks)
    return c.loc[keep].copy()


def rest_frame_sed_checks(df, cfg):
    """Named masks rejecting broken rest-frame SEDs: large extrapolation, extreme mags or colours, and zig-zags."""
    checks, has_rest = [], set(REST_COLS).issubset(df.columns)
    max_edge = cfg["max_rest_edge_distance"]
    if max_edge is not None and "rest_edge_distance" in df.columns:
        checks.append((f"rest_edge_distance <= {max_edge}", numeric(df, "rest_edge_distance").le(max_edge)))
    rest_min, rest_max = cfg["rest_abs_mag_min"], cfg["rest_abs_mag_max"]
    if rest_min is not None and has_rest:
        checks.append((f"all rest nodes >= {rest_min}", df[REST_COLS].ge(rest_min).all(axis=1)))
    faint_limits = cfg["rest_abs_mag_faint_limits"] or {}
    if (rest_max is not None or faint_limits) and has_rest:
        max_by_col = pd.Series(np.inf if rest_max is None else float(rest_max), index=REST_COLS, dtype=float)
        for col, lim in faint_limits.items():
            if col in max_by_col.index:
                max_by_col.loc[col] = float(lim)
        checks.append(("rest faint absolute-mag limits", df[REST_COLS].le(max_by_col, axis=1).all(axis=1)))
    limits = cfg["rest_colour_limits"] or []
    if limits:
        checks.append(("rest-frame colour limits", colour_limits_mask(df, limits)[0]))
    curvature_ok, n_used = colour_curvature_mask(df, consecutive_triplets(REST_COLS), cfg["rest_colour_curvature_max"])
    if n_used:
        checks.append(("rest-frame colour curvature", curvature_ok))
    return checks


def apply_rest_frame_safety_cuts(donors, cfg):
    """Drop donors whose rest-frame SED fails rest_frame_sed_checks."""
    checks = rest_frame_sed_checks(donors, cfg)
    keep = combined_mask(checks, donors.index)
    if not keep.all():
        print_cut_summary(f"Rest-frame training safety cuts kept {int(keep.sum()):,} / {len(donors):,} donors", checks)
        donors = donors.loc[keep]
    return donors
