"""Tidal bridges between interacting galaxy pairs.

Close projected pairs of resolved galaxies are selected until per-environment target fractions are met, and both
galaxies' position angles are aligned with the pair axis. The lower-mass (donor) galaxy loses a few per cent of its
disc light into a chain of Gaussian blobs along a curved bridge: from each core out to an anchor point on the
major axis facing the companion, and straight across between the two anchors.
"""

import numpy as np
import pandas as pd

from .bulge_disc_decomposition import refresh_disc_photometry
from .utils import build_kd_tree, is_resolved, resolved_mask


def wrap_position_angle(pa):
    return pa % 180.0


def interaction_environment(web_env):
    """Collapse web environments to the three interaction classes: field, filament, cluster."""
    web_env = str(web_env)
    if web_env == "filament":
        return "filament"
    return "cluster" if web_env.startswith("cluster") else "field"


def pair_position_angle_deg(row1, row2):
    dx, dy = float(row2["x_pix"] - row1["x_pix"]), float(row2["y_pix"] - row1["y_pix"])
    return wrap_position_angle(np.rad2deg(np.arctan2(dy, dx)))


def major_axis_unit_vector(pa_deg):
    pa = np.deg2rad(pa_deg)
    return np.array([np.cos(pa), np.sin(pa)], float)


def facing_anchor_point(centre, other_centre, re_pix, pa_deg, anchor_radius=2.0):
    """Point anchor_radius * Re along the major axis, on the side facing the other galaxy."""
    centre, other_centre = np.asarray(centre, float), np.asarray(other_centre, float)
    u = major_axis_unit_vector(pa_deg)
    rhat = other_centre - centre
    rhat /= max(np.hypot(*rhat), 1e-8)
    return centre + (1.0 if np.dot(u, rhat) >= 0 else -1.0) * anchor_radius * re_pix * u


def cubic_bezier(p0, p1, p2, p3, t):
    t = np.asarray(t, float)[:, None]
    return (1 - t)**3 * p0 + 3 * (1 - t)**2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3


def core_to_anchor_curve(core, anchor, other_anchor, curve_strength=0.7, inward_pull=0.45, n=9):
    """n points on a Bezier curve leaving the core towards the anchor and arriving along the bridge direction."""
    core, anchor, other_anchor = np.asarray(core, float), np.asarray(anchor, float), np.asarray(other_anchor, float)
    u_core = anchor - core
    d_core = max(np.hypot(*u_core), 1e-8)
    u_core /= d_core
    u_bridge = other_anchor - anchor
    d_bridge = max(np.hypot(*u_bridge), 1e-8)
    u_bridge /= d_bridge
    c1 = core + curve_strength * d_core * u_core
    c2 = anchor - inward_pull * min(d_core, d_bridge) * u_bridge
    return cubic_bezier(core, c1, c2, anchor, np.linspace(0, 1, n))


def tidal_bridge_path(core1, anchor1, anchor2, core2, cfg):
    """Blob centres along core1 -> anchor1 -> anchor2 -> core2, with flux weights and width scales peaking mid-way."""
    curve = dict(curve_strength=cfg["tidal_curve_strength"], inward_pull=cfg["tidal_inward_pull"],
                 n=cfg["tidal_n_curve"])
    left = core_to_anchor_curve(core1, anchor1, anchor2, **curve)
    t = np.linspace(0, 1, cfg["tidal_n_straight"])[:, None]
    mid = (1 - t) * np.asarray(anchor1, float) + t * np.asarray(anchor2, float)
    right = core_to_anchor_curve(core2, anchor2, anchor1, **curve)[::-1]
    pts = np.vstack([left[:-1], mid[:-1], right])
    u = np.linspace(0, 1, len(pts))
    w = np.sin(np.pi * u)
    w[0] = w[-1] = 0.12
    w /= w.sum()
    return pts, w, 0.80 + 0.45 * np.sin(np.pi * u)


def select_tidal_pairs(mock, cfg, rng):
    """Pick non-overlapping close pairs (closest projected first) until each environment's target fraction of
    resolved galaxies is interacting; aligns both galaxies' position angles with the pair axis (with scatter)."""
    m = mock.reset_index(drop=True).copy()
    xy, chi = m[["x_comov_mpc", "y_comov_mpc"]].to_numpy(float), m["chi_mpc"].to_numpy(float)
    ok = np.isfinite(xy).all(axis=1) & np.isfinite(chi) & resolved_mask(m, cfg)
    idx_ok = np.where(ok)[0]
    if len(idx_ok) < 2:
        print("Tidal close-pair candidates: 0")
        print("Selected 0 tidal/interacting pairs")
        return m, pd.DataFrame()

    rp_min, rp_max = cfg["tidal_pair_rp_min_mpc"], cfg["tidal_pair_rp_max_mpc"]
    dchi_max = cfg["tidal_pair_dchi_max_mpc"]
    cand = []
    for a, b in build_kd_tree(xy[ok]).query_pairs(rp_max):
        i, j = idx_ok[a], idx_ok[b]
        rp, dchi = np.hypot(*(xy[i] - xy[j])), abs(chi[i] - chi[j])
        if rp < rp_min or dchi > dchi_max:
            continue
        # Prioritise the closest projected pairs, then smaller line-of-sight offsets.
        cand.append((rp / rp_max + 0.25 * dchi / dchi_max + 1e-4 * rng.random(), rp, dchi, i, j))
    cand.sort()

    env = m["web_env"].map(interaction_environment).to_numpy(object)
    target_frac = {"field": cfg["tidal_target_field"], "filament": cfg["tidal_target_filament"],
                   "cluster": cfg["tidal_target_cluster"]}
    target_n = {k: int(np.round(target_frac[k] * np.sum(ok & (env == k)))) for k in target_frac}
    used, chosen_n, rows = set(), {k: 0 for k in target_frac}, []
    for _, rp, dchi, i, j in cand:
        e1, e2 = env[i], env[j]
        # Keep selecting until the relevant environment budgets are filled.
        if i in used or j in used or (chosen_n[e1] >= target_n[e1] and chosen_n[e2] >= target_n[e2]):
            continue
        pa0 = pair_position_angle_deg(m.loc[i], m.loc[j])
        pa1 = wrap_position_angle(pa0 + rng.normal(0.0, cfg["tidal_pa_scatter_deg"]))
        pa2 = wrap_position_angle(pa0 + rng.normal(0.0, cfg["tidal_pa_scatter_deg"]))
        m.loc[i, "pa_deg"], m.loc[j, "pa_deg"] = pa1, pa2
        used |= {i, j}
        chosen_n[e1] += 1
        chosen_n[e2] += 1
        rows.append({"i": i, "j": j, "id1": int(m.loc[i, "id"]), "id2": int(m.loc[j, "id"]), "rp_mpc": rp,
                     "dchi_mpc": dchi, "env1": m.loc[i, "web_env"], "env2": m.loc[j, "web_env"], "env_class1": e1,
                     "env_class2": e2, "pa1_deg": pa1, "pa2_deg": pa2})
    pairs = pd.DataFrame(rows)
    print(f"Tidal close-pair candidates: {len(cand):,}")
    print("Target interacting galaxies:", target_n)
    print("Chosen interacting galaxies:", chosen_n)
    print(f"Selected {len(pairs):,} tidal/interacting pairs")
    return m, pairs


def add_tidal_streams(mock, pairs, cfg, rng):
    """Move a random fraction (tidal_flux_frac_min..max) of each donor's disc light into bridge blobs.

    Returns (galaxies with tidal and interaction columns, a table with one row per blob and band).
    """
    m = mock.copy()
    bands = cfg["out_bands"]
    for b in bands:
        m[f"flux_{b}_disc_pretidal"], m[f"flux_{b}_tidal"] = m[f"flux_{b}_disc"], 0.0
    m["is_interacting"], m["interaction_partner_id"], m["tidal_role"] = False, -1, "none"
    if len(pairs) == 0:
        return m, pd.DataFrame()

    rows = []
    for pair_id, p in pairs.reset_index(drop=True).iterrows():
        i, j = int(p["i"]), int(p["j"])
        g1, g2 = m.iloc[i], m.iloc[j]
        if not is_resolved(g1, cfg) or not is_resolved(g2, cfg):
            continue
        if not np.isfinite(g1["re_disc_arcsec"]) or not np.isfinite(g2["re_disc_arcsec"]):
            continue
        donor_i, other_i = (i, j) if g1["logM"] < g2["logM"] else (j, i)
        donor = m.iloc[donor_i]
        re1_pix, re2_pix = float(g1["re_disc_arcsec"]) / cfg["pixscale"], float(g2["re_disc_arcsec"]) / cfg["pixscale"]
        if re1_pix <= 0 or re2_pix <= 0:
            continue
        c1, c2 = np.array([g1["x_pix"], g1["y_pix"]], float), np.array([g2["x_pix"], g2["y_pix"]], float)
        p1 = facing_anchor_point(c1, c2, re1_pix, g1["pa_deg"], cfg["tidal_anchor_radius_re"])
        p2 = facing_anchor_point(c2, c1, re2_pix, g2["pa_deg"], cfg["tidal_anchor_radius_re"])
        pts, w, sig_scale = tidal_bridge_path(c1, p1, p2, c2, cfg)
        sig0 = np.clip(rng.uniform(cfg["tidal_column_sig_re_frac_min"], cfg["tidal_column_sig_re_frac_max"])
                       * min(re1_pix, re2_pix), cfg["tidal_column_sig_pix_min"], cfg["tidal_column_sig_pix_max"])
        frac = rng.uniform(cfg["tidal_flux_frac_min"], cfg["tidal_flux_frac_max"])
        for b in bands:
            fcol = frac * float(donor[f"flux_{b}_disc"])
            if not np.isfinite(fcol) or fcol <= 0:
                continue
            m.loc[donor_i, f"flux_{b}_tidal"] += fcol
            m.loc[donor_i, f"flux_{b}_disc"] = max(float(m.loc[donor_i, f"flux_{b}_disc"]) - fcol, 0.0)
            for k, ((x, y), wk, sk) in enumerate(zip(pts, w, sig_scale)):
                rows.append({"tidal_pair_id": pair_id, "tidal_blob_id": k, "donor_id": int(donor["id"]),
                             "gal1_id": int(g1["id"]), "gal2_id": int(g2["id"]), "band": b, "x_pix_tidal": x,
                             "y_pix_tidal": y, "sigma_pix": sig0 * sk, "flux_tidal": fcol * wk,
                             "frac_total_light_in_column": frac})
        m.loc[[i, j], "is_interacting"] = True
        m.loc[i, "interaction_partner_id"], m.loc[j, "interaction_partner_id"] = int(g2["id"]), int(g1["id"])
        m.loc[donor_i, "tidal_role"], m.loc[other_i, "tidal_role"] = "donor", "companion"

    tidal = pd.DataFrame(rows)
    refresh_disc_photometry(m, cfg)
    print(f"Added {len(tidal):,} tidal blobs from {tidal['tidal_pair_id'].nunique() if len(tidal) else 0:,} pairs")
    return m, tidal
