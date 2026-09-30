"""Star-forming clumps: compact blue Gaussian knots placed in the discs of resolved star-forming galaxies.

The clump light fraction rises with sSFR and towards low mass; clumps avoid the disc centre, follow the disc
inclination and position angle, and take their light from the disc (whose Re is refit so the half-light radius of
disc + clumps is unchanged). Clump positions are drawn in proportion to the disc's light, so to its number of stars:
for a smooth disc that is the exponential profile, and for a galaxy with a Hubble type's arms, bar or irregularity
(hubble_types.py) it is that structured light, so clumps crowd onto the arms.
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .bulge_disc_decomposition import SERSIC_B1, refresh_disc_photometry, sersic_enclosed_fraction
from .galaxy_structure import has_structure, structure_weight
from .photometry import LSST_BAND_WAVE_A
from .utils import is_resolved, resolved_mask


def sample_clump_colours(n, rng, ug_mu=-0.25, gr_mu=0.05, ug_sig=0.5, gr_sig=0.3, rho=0.35, ug_range=(-0.55, 0.15),
                         gr_range=(-0.20, 0.22)):
    """Rest-frame (u-g, g-r) clump colours from a correlated Gaussian truncated to the given ranges."""
    cov = np.array([[ug_sig**2, rho * ug_sig * gr_sig], [rho * ug_sig * gr_sig, gr_sig**2]])
    out = []
    while sum(len(x) for x in out) < n:
        x = rng.multivariate_normal([ug_mu, gr_mu], cov, size=max(n, 8))
        ok = (x[:, 0] >= ug_range[0]) & (x[:, 0] <= ug_range[1]) & (x[:, 1] >= gr_range[0]) & (x[:, 1] <= gr_range[1])
        out.append(x[ok])
    x = np.vstack(out)[:n]
    return x[:, 0], x[:, 1]


def target_clump_light_fraction(logM, logSFR, f_ref=0.02, ssfr_ref=-10.0, ssfr_strength=0.30, dwarf_amp=0.65,
                                dwarf_turnover=8.85, dwarf_width=0.25, f_min=0.0, f_max=0.12):
    """Fraction of rest-frame u disc light in clumps: grows with sSFR, boosted for dwarfs, gated off when passive."""
    logsSFR = logSFR - logM
    logf = np.log10(f_ref) + ssfr_strength * (logsSFR - ssfr_ref)
    logf += dwarf_amp / (1.0 + np.exp((logM - dwarf_turnover) / dwarf_width))
    sf_gate = 1.0 / (1.0 + 10.0 ** (-(logsSFR + 10.8) / 0.25))
    return np.clip((10.0 ** logf) * sf_gate, f_min, f_max)


def clump_relative_flux(rest_wave, ug, gr):
    """Clump SED (relative flux, u = 1) at rest wavelength(s), interpolated and extrapolated in log-log from u, g, r."""
    fu = 1.0
    fg = fu / 10.0**(-0.4 * ug)
    fr = fg / 10.0**(-0.4 * gr)
    x = np.log10(np.array([3670.0, 4825.0, 6222.0]))
    y = np.log10(np.array([fu, fg, fr]))
    xq = np.log10(np.asarray(rest_wave, float))
    yq = np.interp(xq, x, y)
    yq = np.where(xq < x[0], y[0] + (y[1] - y[0]) / (x[1] - x[0]) * (xq - x[0]), yq)
    yq = np.where(xq > x[-1], y[-1] + (y[-1] - y[-2]) / (x[-1] - x[-2]) * (xq - x[-1]), yq)
    return 10.0 ** yq


def follow_structure(row, rd, r0, rmax, n, n_candidates=200):
    """Disc-plane radii and azimuths of n clumps for a galaxy with arms / bar / irregularity: candidates drawn from
    the smooth disc (as draw_galaxy_clumps does) are resampled in proportion to structure_weight, so positions follow
    the structured disc light. Uses the galaxy's own random stream (structure_seed), leaving the catalogue's intact."""
    rng = np.random.default_rng([int(row["structure_seed"]), 1])
    rad = rng.gamma(2.0, rd, size=n * n_candidates)
    keep = (rng.random(len(rad)) < 1.0 - np.exp(-(rad / max(r0, 1e-6))**2)) & (rad <= rmax)
    rad = rad[keep]
    phi = rng.uniform(0, 2*np.pi, len(rad))
    weight = structure_weight(row, rad * np.cos(phi), rad * np.sin(phi))
    pick = rng.choice(len(rad), size=n, replace=False, p=weight / weight.sum())
    return rad[pick], phi[pick]


def draw_galaxy_clumps(row, cfg, rng):
    """Clump records (offsets, radius, width, colours, per-band flux) for one galaxy; [] if it gets none."""
    if row.get("type", "") != "star_forming" or not is_resolved(row, cfg):
        return []
    re = float(row["re_disc_arcsec"])
    if not np.isfinite(re) or re <= 0:
        return []
    bands, z = cfg["out_bands"], float(row["z"])
    target_f = target_clump_light_fraction(float(row["logM"]), float(row["logSFR"]))
    mean_single = cfg["clump_mean_single_u_frac"]
    n = min(rng.poisson(target_f / mean_single), cfg["clump_n_max_per_gal"])
    if n == 0:
        return []

    # Radii from a gamma profile with a central hole, projected with the disc axis ratio and position angle.
    q, pa, rd = np.clip(1.0 - float(row["ellipticity_disc"]), 0.08, 1.0), np.deg2rad(float(row["pa_deg"])), re / 1.678
    rmax, r0 = cfg["clump_max_r_re"] * re, cfg["clump_central_hole_re"] * re
    rad = []
    while len(rad) < n:
        r_try = rng.gamma(2.0, rd, size=max(8, 3 * n))
        keep = rng.random(len(r_try)) < 1.0 - np.exp(-(r_try / max(r0, 1e-6))**2)
        rad.extend(r_try[keep & (r_try <= rmax)].tolist())
    rad = np.array(rad[:n])
    phi = rng.uniform(0, 2*np.pi, n)
    if has_structure(row, 0.0):
        rad, phi = follow_structure(row, rd, r0, rmax, n)
    # Disc plane to sky as GalSim draws the disc: its shear keeps area, stretching the major axis by 1 / sqrt(q).
    x0, y0 = rad * np.cos(phi) / np.sqrt(q), np.sqrt(q) * rad * np.sin(phi)
    dx, dy = x0 * np.cos(pa) - y0 * np.sin(pa), x0 * np.sin(pa) + y0 * np.cos(pa)
    ug, gr = sample_clump_colours(n, rng)

    # Normalise fluxes in the observed band closest to rest-frame u.
    rest_waves = {b: LSST_BAND_WAVE_A[b] / (1.0 + z) for b in bands}
    anchor = min(bands, key=lambda b: abs(np.log(rest_waves[b] / 3670.0)))
    anchor_flux = float(row[f"flux_{anchor}_disc"])
    if not np.isfinite(anchor_flux) or anchor_flux <= 0:
        return []
    local_sb = np.exp(-rad / rd)
    local_weight = local_sb ** cfg["clump_flux_radius_bias"]
    local_weight /= np.nanmean(local_weight)
    scatter = cfg["clump_flux_scatter"]
    f_anchor = anchor_flux * mean_single * local_weight * rng.lognormal(mean=-0.5 * scatter**2, sigma=scatter, size=n)
    sigma = 0.04 * re + 0.025 * rad + rng.normal(0, 0.08 * re, n)
    sigma = np.clip(sigma, cfg["clump_sigma_floor_arcsec"], 0.18 * re)

    clumps = []
    for j in range(n):
        f_anchor_sed = clump_relative_flux(rest_waves[anchor], ug[j], gr[j])
        c = dict(parent_id=int(row["id"]), clump_id=j, dx_arcsec=dx[j], dy_arcsec=dy[j], r_ell_arcsec=rad[j],
                 r_over_re=rad[j] / re, sigma_arcsec=sigma[j], u_g=ug[j], g_r=gr[j], local_disc_sb_rel=local_sb[j],
                 target_f_clump_u=target_f)
        for b in bands:
            c[f"flux_{b}_clump"] = f_anchor[j] * clump_relative_flux(rest_waves[b], ug[j], gr[j]) / f_anchor_sed
        clumps.append(c)
    # Never let clumps take more than 90% of the disc light in any band.
    mx = max(sum(c[f"flux_{b}_clump"] for c in clumps) / max(float(row[f"flux_{b}_disc"]), 1e-30) for b in bands)
    if mx > 0.90:
        for c in clumps:
            for b in bands:
                c[f"flux_{b}_clump"] *= 0.90 / mx
    return clumps


def refit_disc_re_without_clumps(pre_re, pre_flux_r, clumps):
    """Disc Re for which residual disc + clumps keep the pre-clump r-band half-light radius (else pre_re)."""
    if len(clumps) == 0 or not np.isfinite(pre_re) or pre_re <= 0:
        return pre_re
    disc_flux = pre_flux_r - clumps["flux_r_clump"].sum()
    if disc_flux <= 0:
        return pre_re
    need = 0.5 * pre_flux_r - clumps.loc[clumps["r_ell_arcsec"] <= pre_re, "flux_r_clump"].sum()
    if need <= 0 or need >= disc_flux:
        return pre_re
    frac = need / disc_flux
    try:
        return brentq(lambda r: sersic_enclosed_fraction(pre_re, r, 1.0, SERSIC_B1) - frac, 0.05 * pre_re,
                      50.0 * pre_re, xtol=1e-4)
    except ValueError:
        return pre_re


def add_sf_clumps(mock, cfg, rng):
    """Add clumps to resolved star-forming discs. Returns (galaxies with clump/disc columns updated, clump table)."""
    m = mock.copy()
    m.attrs.clear()
    bands = cfg["out_bands"]
    for b in bands:
        m[f"flux_{b}_disc_preclump"], m[f"flux_{b}_clump"] = m[f"flux_{b}_disc"], 0.0
    m["re_disc_arcsec_preclump"], m["N_clumps"], m["target_f_clump_u"] = m["re_disc_arcsec"], 0, 0.0

    re_disc = m["re_disc_arcsec"].to_numpy(float)
    is_host = m["type"].eq("star_forming").to_numpy() & resolved_mask(m, cfg) & np.isfinite(re_disc) & (re_disc > 0)
    hosts = m.loc[is_host]
    print(f"Drawing SF clumps for {len(hosts):,} resolved star-forming galaxies", flush=True)
    clumps = pd.DataFrame([c for row in hosts.to_dict("records") for c in draw_galaxy_clumps(row, cfg, rng)])
    if len(clumps) == 0:
        print("Added 0 SF clumps to 0 galaxies", flush=True)
        return m, clumps

    clumps = clumps.merge(m[["id", "x_pix", "y_pix"]], left_on="parent_id", right_on="id", how="left")
    clumps["x_pix_clump"] = clumps["x_pix"] + clumps["dx_arcsec"] / cfg["pixscale"]
    clumps["y_pix_clump"] = clumps["y_pix"] + clumps["dy_arcsec"] / cfg["pixscale"]
    clumps = clumps.drop(columns=["id", "x_pix", "y_pix"])
    grouped = clumps.groupby("parent_id", sort=False)
    for b in bands:
        m[f"flux_{b}_clump"] = m["id"].map(grouped[f"flux_{b}_clump"].sum()).fillna(0.0).to_numpy(float)
        m[f"flux_{b}_disc"] = np.maximum(m[f"flux_{b}_disc_preclump"].to_numpy(float)
                                         - m[f"flux_{b}_clump"].to_numpy(float), 0.0)
    m["N_clumps"] = m["id"].map(grouped.size()).fillna(0).astype(int)
    m["target_f_clump_u"] = m["id"].map(grouped["target_f_clump_u"].first()).fillna(0.0)

    print(f"Refitting residual disc Re for {(m['N_clumps'] > 0).sum():,} clump hosts", flush=True)
    new_re = m["re_disc_arcsec"].to_numpy(float).copy()
    row_of_id = pd.Series(m.index.to_numpy(), index=m["id"].to_numpy())
    for pid, g in grouped:
        i = int(row_of_id.loc[pid])
        new_re[i] = refit_disc_re_without_clumps(m.at[i, "re_disc_arcsec_preclump"], m.at[i, "flux_r_disc_preclump"], g)
    m["re_disc_arcsec"] = new_re
    refresh_disc_photometry(m, cfg)
    print(f"Added {len(clumps):,} SF clumps to {(m['N_clumps'] > 0).sum():,} galaxies", flush=True)
    return m, clumps
