"""Place galaxies on cosmic-web sites, which gives the mock its clustering (correlation function) and environments.

Default route (draw_web_positions): candidate web sites fill the light cone and are thinned by the evolving GSMF number
density at each site's redshift, so positions (and redshifts) come first and physical properties are drawn after.
Optional route (assign_clustered_positions): existing galaxies keep their redshifts and are matched, shell by shell, to
web sites by rank, so that massive, passive and luminous galaxies land in the densest environments.
"""

import numpy as np
import pandas as pd

from .cosmic_web_generation import field_sites, generate_web_sites
from .cosmology import (comoving_distance_from_redshift, comoving_to_pixel, field_width_rad,
                        redshift_from_comoving_distance)

# Galaxy column <- web-site column copied on assignment.
SITE_COLUMNS = {"web_env": "web_env", "web_density_score": "web_density_score", "web_site_id": "web_site_id",
                "web_node_id": "web_node_id", "web_edge_id": "web_edge_id", "dist_node_mpc": "dist_node_mpc",
                "dist_filament_mpc": "dist_filament_mpc", "chi_site_assigned_mpc": "chi_site"}


def draw_web_positions(n_gal, tables, cfg, grids, rng):
    """n_gal light-cone positions: web sites kept with probability ~ GSMF number density at their redshift."""
    if n_gal <= 0:
        return pd.DataFrame()
    chi_lo = max(comoving_distance_from_redshift(cfg["z_min"], grids), 1e-6)
    chi_hi = comoving_distance_from_redshift(cfg["z_max"], grids)
    site_cfg = dict(cfg, web_candidate_oversample=max(1, int(cfg["gsmf_position_oversample"])))
    sites, nodes, edges = generate_web_sites(n_gal, chi_lo, chi_hi, site_cfg, rng)

    z_site = redshift_from_comoving_distance(sites["chi_site"].to_numpy(float), grids)
    w = np.maximum(np.interp(z_site, tables["z"], tables["n_total"], left=0.0, right=0.0), 0.0)
    if cfg["web_site_keep_bias"] != 0.0:
        w *= np.exp(cfg["web_site_keep_bias"] * sites["web_density_score"].to_numpy(float))
    if len(sites) > n_gal:
        if np.isfinite(w).all() and w.sum() > 0:
            choose = rng.choice(len(sites), n_gal, replace=False, p=w / w.sum())
            sites, z_site = sites.iloc[choose].reset_index(drop=True), z_site[choose]
        else:
            sites = sites.sample(n=n_gal, random_state=int(rng.integers(1e9))).reset_index(drop=True)
            z_site = redshift_from_comoving_distance(sites["chi_site"].to_numpy(float), grids)

    chi = np.maximum(sites["chi_site"].to_numpy(float), 1e-6)
    half = 0.5 * field_width_rad(cfg) * chi
    x_comov = np.clip(sites["x_comov"].to_numpy(float), -0.999 * half, 0.999 * half)
    y_comov = np.clip(sites["y_comov"].to_numpy(float), -0.999 * half, 0.999 * half)
    out = pd.DataFrame({
        "z": z_site, "chi_mpc": chi, "x_comov_mpc": x_comov, "y_comov_mpc": y_comov,
        "chi_site_assigned_mpc": sites["chi_site"].to_numpy(float), "web_env": sites["web_env"].to_numpy(object),
        "web_density_score": sites["web_density_score"].to_numpy(float),
        "web_site_id": sites["web_site_id"].to_numpy(int), "web_node_id": sites["web_node_id"].to_numpy(int),
        "web_edge_id": sites["web_edge_id"].to_numpy(int), "dist_node_mpc": sites["dist_node_mpc"].to_numpy(float),
        "dist_filament_mpc": sites["dist_filament_mpc"].to_numpy(float),
    })
    out["x_pix"] = np.clip(comoving_to_pixel(x_comov, chi, cfg), 0.0, cfg["npix"] - 1e-3)
    out["y_pix"] = np.clip(comoving_to_pixel(y_comov, chi, cfg), 0.0, cfg["npix"] - 1e-3)
    print("Initial clustered site environment fractions:")
    print(out["web_env"].value_counts(normalize=True).sort_index())
    return out


def zscore(x):
    """(x - median) / std, with non-finite results (or zero spread) mapped to 0."""
    x = np.asarray(x, float)
    med, sig = np.nanmedian(x), np.nanstd(x)
    if not np.isfinite(sig) or sig <= 0:
        return np.zeros_like(x)
    return np.nan_to_num((x - med) / sig, nan=0.0, posinf=0.0, neginf=0.0)


def galaxy_environment_score(m, rng, cfg):
    """Environment score (high = dense): stellar mass, low sSFR, passivity and luminosity, plus Gaussian scatter."""
    passive = m["type"].eq("passive").to_numpy(float) if "type" in m.columns else np.zeros(len(m))
    if "Mi" in m.columns:
        lum = -m["Mi"].to_numpy(float)
    elif "mag_i_total" in m.columns:
        lum = -m["mag_i_total"].to_numpy(float)
    else:
        lum = np.zeros(len(m))
    # Deliberately weaker than the first attempt (0.95 logM - 0.75 sSFR + 0.55 passive + 0.55 lum + 0.30 SB), which put
    # many galaxies of the same mass and colour together so that groups looked like star clusters.
    score = (0.9 * zscore(m["logM"].to_numpy(float)) - 0.5 * zscore(m["logsSFR"].to_numpy(float)) + 0.4 * passive
             + 0.15 * zscore(lum))
    return score + rng.normal(0.0, cfg["env_assign_scatter"], len(m))


def assign_clustered_positions(mock, cfg, grids, rng):
    """Re-place existing galaxies on web sites in comoving shells of web_assign_bin_mpc, pairing galaxies sorted by
    environment score with sites sorted by density. Each galaxy keeps its redshift and takes the site's direction."""
    if not cfg["clustered_positions"]:
        return mock
    m = mock.copy()
    m["chi_mpc"] = comoving_distance_from_redshift(m["z"].to_numpy(float), grids)
    chi_min, chi_max = max(np.nanmin(m["chi_mpc"]), 1e-6), np.nanmax(m["chi_mpc"])
    sites, nodes, edges = generate_web_sites(len(m), chi_min, chi_max, cfg, rng)

    n = len(m)
    out = {"x_comov_mpc": np.full(n, np.nan), "y_comov_mpc": np.full(n, np.nan),
           "chi_site_assigned_mpc": np.full(n, np.nan), "web_env": np.repeat("unassigned", n).astype(object),
           "web_density_score": np.full(n, np.nan), "web_site_id": np.full(n, -1, int),
           "web_node_id": np.full(n, -1, int), "web_edge_id": np.full(n, -1, int), "dist_node_mpc": np.full(n, np.nan),
           "dist_filament_mpc": np.full(n, np.nan), "x_pix": np.full(n, np.nan), "y_pix": np.full(n, np.nan)}
    used_site = np.zeros(len(sites), bool)
    bin_width, n_filler_total = cfg["web_assign_bin_mpc"], 0
    for chi_lo in np.arange(np.floor(chi_min / bin_width) * bin_width, chi_max + bin_width, bin_width):
        chi_hi = chi_lo + bin_width
        idx = m.index[(m["chi_mpc"] >= chi_lo) & (m["chi_mpc"] < chi_hi)].to_numpy()
        if len(idx) == 0:
            continue
        cand_idx = sites.index[(~used_site) & (sites["chi_site"] >= chi_lo) & (sites["chi_site"] < chi_hi)].to_numpy()
        pool = sites.loc[cand_idx].copy()
        pool["_orig_site_index"] = cand_idx
        if len(pool) < len(idx):  # too few web sites in this shell: top up with field sites
            n_missing = len(idx) - len(pool)
            filler = field_sites(n_missing, chi_lo, chi_hi, cfg, rng)
            filler.insert(0, "web_site_id", -1)
            filler["_orig_site_index"] = -1
            pool = pd.concat([pool, filler], ignore_index=True)
            n_filler_total += n_missing
        if len(pool) > len(idx):
            w = np.exp(cfg["web_site_keep_bias"] * pool["web_density_score"].to_numpy(float))
            pool = pool.iloc[rng.choice(len(pool), len(idx), replace=False, p=w / w.sum())].reset_index(drop=True)
        real = pool["_orig_site_index"].to_numpy(int)
        used_site[real[real >= 0]] = True

        # Low-score galaxies go to low-density sites, high-score galaxies to dense sites.
        gi = idx[np.argsort(galaxy_environment_score(m.loc[idx], rng, cfg))]
        pool = pool.iloc[np.argsort(pool["web_density_score"].to_numpy(float))].reset_index(drop=True)
        chi_gal = m.loc[gi, "chi_mpc"].to_numpy(float)
        chi_site = np.maximum(pool["chi_site"].to_numpy(float), 1e-6)
        # Preserve the site's angular direction but the galaxy's exact redshift.
        half = 0.5 * field_width_rad(cfg) * np.maximum(chi_gal, 1e-6)
        xx = np.clip(pool["x_comov"].to_numpy(float) / chi_site * chi_gal, -0.999 * half, 0.999 * half)
        yy = np.clip(pool["y_comov"].to_numpy(float) / chi_site * chi_gal, -0.999 * half, 0.999 * half)
        out["x_comov_mpc"][gi], out["y_comov_mpc"][gi] = xx, yy
        out["x_pix"][gi] = comoving_to_pixel(xx, np.maximum(chi_gal, 1e-6), cfg)
        out["y_pix"][gi] = comoving_to_pixel(yy, np.maximum(chi_gal, 1e-6), cfg)
        for col, site_col in SITE_COLUMNS.items():
            out[col][gi] = pool[site_col].to_numpy(out[col].dtype)

    for col in ("x_pix", "y_pix"):
        out[col] = np.clip(out[col], 0.0, cfg["npix"] - 1e-3)
    for col, values in out.items():
        m[col] = values
    print("Clustered placement environment fractions:")
    print(m["web_env"].value_counts(normalize=True).sort_index())
    if n_filler_total:
        print(f"WARNING: used {n_filler_total:,} fallback field sites due to sparse candidate bins.")
    return m
