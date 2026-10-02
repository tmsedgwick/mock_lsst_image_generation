"""Place galaxies on cosmic-web sites, which gives the mock its clustering (correlation function) and environments.

Default route (draw_web_positions): candidate web sites fill the light cone and are thinned by the evolving GSMF number
density at each site's redshift, so positions (and redshifts) come first and physical properties are drawn after.
Optional route (assign_clustered_positions): existing galaxies keep their redshifts and are matched, shell by shell, to
web sites. With cfg['environment_by_class'] (the default), each galaxy's environment is drawn from P(environment |
class), so stellar mass, star formation and environment form one joint distribution: classes are red / blue x low /
high surface brightness, judged as Thuruthipilly et al. 2024 selected them (z < 0.1; see reference_classes),
and their probabilities of living in structure (clusters or filaments) are set so each class's angular clustering
matches that paper (see config). The web is built with the matching mix of sites; within an environment the most
massive galaxies take the densest sites. Without it, galaxies are matched by rank of an environment score.
"""

import numpy as np
import pandas as pd

from .cosmic_web_generation import field_sites, generate_web_sites
from .cosmology import (comoving_distance_from_redshift, comoving_to_pixel, distance_modulus, field_width_rad,
                        kpc_to_arcsec, redshift_from_comoving_distance)
from .photometry import LSST_BAND_WAVE_A, REST_COLS, REST_WAVE_A

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


RED_LSB_RANK_BOOST = 1.0
Z_CLASS_MAX = 0.1  # galaxies are classified as seen at min(z, Z_CLASS_MAX): exactly the T24 selection (z < 0.1)


def magnitude_at(rest_mags, band, z, grids):
    """Observed AB magnitude in band at redshift z (one per galaxy), interpolating each rest-frame SED in log
    wavelength (as the observed photometry is projected)."""
    x, lw = np.log10(REST_WAVE_A), np.log10(LSST_BAND_WAVE_A[band] / (1 + z))
    j = np.clip(np.searchsorted(x, lw), 1, len(x) - 1)
    t = (lw - x[j - 1]) / (x[j] - x[j - 1])
    rows = np.arange(len(z))
    m_abs = (1 - t) * rest_mags[rows, j - 1] + t * rest_mags[rows, j]
    return m_abs + distance_modulus(z, grids) - 2.5 * np.log10(1 + z)


def reference_classes(m, grids):
    """'red_hsb', 'blue_hsb', 'red_lsb' or 'blue_lsb' for each galaxy, with the Thuruthipilly et al. 2024 cuts applied
    as it would be seen at min(z, Z_CLASS_MAX): LSB if the mean g-band surface brightness within Re exceeds 24.2
    mag/arcsec^2 and Re > 2.5"; red if g - i > 0.6 (LSB) or > 1.0 (others). Below Z_CLASS_MAX this is exactly their
    selection; above it a galaxy is judged as it would look at Z_CLASS_MAX, so the class stays intrinsic."""
    z = np.minimum(m["z"].to_numpy(float), Z_CLASS_MAX)
    rest = m[REST_COLS].to_numpy(float)
    g, i = magnitude_at(rest, "g", z, grids), magnitude_at(rest, "i", z, grids)
    re_arcsec = kpc_to_arcsec(m["Re_kpc"].to_numpy(float), z, grids)
    lsb = (g + 2.5 * np.log10(2 * np.pi * np.maximum(re_arcsec, 1e-6) ** 2) > 24.2) & (re_arcsec > 2.5)
    red = g - i > np.where(lsb, 0.6, 1.0)
    return np.char.add(np.where(red, "red_", "blue_"), np.where(lsb, "lsb", "hsb"))


def draw_environments(m, cfg, grids, rng):
    """Environment ('cluster_core', 'cluster_outskirts', 'filament' or 'field') for each galaxy, from its class's
    P(structure) and P(cluster | structure) in cfg['environment_by_class']; red LSB galaxies avoid cluster cores."""
    classes = reference_classes(m, grids)
    p_structure, p_cluster = (np.array([cfg["environment_by_class"][c][k] for c in classes]) for k in (0, 1))
    u_structure, u_cluster, u_core = rng.random((3, len(m)))
    env = np.where(u_structure < p_structure, np.where(u_cluster < p_cluster, "cluster", "filament"), "field").astype(object)
    in_cluster = env == "cluster"
    core = in_cluster & (u_core < cfg["web_core_frac"]) & (classes != "red_lsb")
    env[in_cluster] = "cluster_outskirts"
    env[core] = "cluster_core"
    return env, classes


def assign_by_environment(mock, cfg, grids, rng):
    """Place galaxies on web sites of their drawn environment, shell by shell (see the module docstring)."""
    m = mock.copy()
    m["chi_mpc"] = comoving_distance_from_redshift(m["z"].to_numpy(float), grids)
    env_wanted, classes = draw_environments(m, cfg, grids, rng)
    site_cfg = dict(cfg, web_f_cluster=float(np.mean(np.char.startswith(env_wanted.astype(str), "cluster"))),
                    web_f_filament=float(np.mean(env_wanted == "filament")))
    chi_min, chi_max = max(np.nanmin(m["chi_mpc"]), 1e-6), np.nanmax(m["chi_mpc"])
    sites, _, _ = generate_web_sites(len(m), chi_min, chi_max, site_cfg, rng)
    site_env = sites["web_env"].replace({"cluster_central": "cluster_core"}).to_numpy(object)

    n = len(m)
    out = {"x_comov_mpc": np.full(n, np.nan), "y_comov_mpc": np.full(n, np.nan),
           "chi_site_assigned_mpc": np.full(n, np.nan), "web_env": np.repeat("unassigned", n).astype(object),
           "web_density_score": np.full(n, np.nan), "web_site_id": np.full(n, -1, int),
           "web_node_id": np.full(n, -1, int), "web_edge_id": np.full(n, -1, int), "dist_node_mpc": np.full(n, np.nan),
           "dist_filament_mpc": np.full(n, np.nan), "x_pix": np.full(n, np.nan), "y_pix": np.full(n, np.nan)}
    used = np.zeros(len(sites), bool)
    bin_width, n_filler = cfg["web_assign_bin_mpc"], 0
    logm = m["logM"].to_numpy(float)
    for chi_lo in np.arange(np.floor(chi_min / bin_width) * bin_width, chi_max + bin_width, bin_width):
        chi_hi = chi_lo + bin_width
        in_shell = (m["chi_mpc"].to_numpy() >= chi_lo) & (m["chi_mpc"].to_numpy() < chi_hi)
        site_in_shell = (sites["chi_site"].to_numpy() >= chi_lo) & (sites["chi_site"].to_numpy() < chi_hi)
        for env in ["cluster_core", "cluster_outskirts", "filament", "field"]:
            gi = np.flatnonzero(in_shell & (env_wanted == env))
            if len(gi) == 0:
                continue
            cand = np.flatnonzero(site_in_shell & ~used & (site_env == env))
            if len(cand) > len(gi):
                cand = rng.choice(cand, len(gi), replace=False)
            used[cand] = True
            pool = sites.iloc[cand].copy()
            if len(pool) < len(gi):  # too few sites of this environment in the shell: the rest go to the field
                filler = field_sites(len(gi) - len(pool), chi_lo, chi_hi, cfg, rng)
                filler.insert(0, "web_site_id", -1)
                pool = pd.concat([pool, filler], ignore_index=True)
                n_filler += len(filler)
            # Within an environment the most massive galaxies take the densest sites; red LSB galaxies (the most strongly
            # clustered, T24) rank as if RED_LSB_RANK_BOOST dex more massive.
            gi = gi[np.argsort(logm[gi] + RED_LSB_RANK_BOOST * (classes[gi] == "red_lsb"))]
            pool = pool.iloc[np.argsort(pool["web_density_score"].to_numpy(float))].reset_index(drop=True)
            place(out, gi, pool, m["chi_mpc"].to_numpy(float)[gi], cfg)

    for col in ("x_pix", "y_pix"):
        out[col] = np.clip(out[col], 0.0, cfg["npix"] - 1e-3)
    for col, values in out.items():
        m[col] = values
    m["environment_class"] = classes
    print("Environment by class placement fractions:")
    print(m["web_env"].value_counts(normalize=True).sort_index())
    if n_filler:
        print(f"Placed {n_filler:,} galaxies on field sites for lack of sites of their drawn environment.")
    return m


def place(out, gi, pool, chi_gal, cfg):
    """Put galaxies gi on pool's sites (in order): the site's direction at each galaxy's own distance."""
    chi_site = np.maximum(pool["chi_site"].to_numpy(float), 1e-6)
    half = 0.5 * field_width_rad(cfg) * np.maximum(chi_gal, 1e-6)
    xx = np.clip(pool["x_comov"].to_numpy(float) / chi_site * chi_gal, -0.999 * half, 0.999 * half)
    yy = np.clip(pool["y_comov"].to_numpy(float) / chi_site * chi_gal, -0.999 * half, 0.999 * half)
    out["x_comov_mpc"][gi], out["y_comov_mpc"][gi] = xx, yy
    out["x_pix"][gi] = comoving_to_pixel(xx, np.maximum(chi_gal, 1e-6), cfg)
    out["y_pix"][gi] = comoving_to_pixel(yy, np.maximum(chi_gal, 1e-6), cfg)
    for col, site_col in SITE_COLUMNS.items():
        out[col][gi] = pool[site_col].to_numpy(out[col].dtype)


def assign_clustered_positions(mock, cfg, grids, rng):
    """Re-place existing galaxies on web sites in comoving shells of web_assign_bin_mpc, pairing galaxies sorted by
    environment score with sites sorted by density. Each galaxy keeps its redshift and takes the site's direction."""
    if not cfg["clustered_positions"]:
        return mock
    if cfg.get("environment_by_class"):
        return assign_by_environment(mock, cfg, grids, rng)
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
