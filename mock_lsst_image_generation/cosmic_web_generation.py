"""A coherent toy cosmic web spanning the survey light cone, and candidate galaxy sites drawn from it.

Web nodes (clusters) are partly clustered around a few supercluster centres and partly uniform; filaments join them
along a minimum spanning tree plus short extra links. Candidate sites are placed at node centres, in cluster cores
and outskirts, along filaments and in the field, each with a web_density_score that later biases site selection.
Coordinates are comoving Mpc: x, y transverse to the line of sight and chi along it.
"""

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree

from .cosmology import field_width_rad, trapz
from .utils import build_kd_tree, query_nearest


def inside_parent_volume(xyz, cfg, chi_lo, chi_hi, pad=None):
    """Points within [chi_lo, chi_hi] inside the light cone widened by pad (default web_parent_pad_mpc) per side."""
    chi = xyz[:, 2]
    half = 0.5 * field_width_rad(cfg) * np.asarray(chi, float) + (cfg["web_parent_pad_mpc"] if pad is None else pad)
    return (chi >= chi_lo) & (chi <= chi_hi) & (np.abs(xyz[:, 0]) <= half) & (np.abs(xyz[:, 1]) <= half)


def inside_lightcone(xyz, cfg, chi_lo=None, chi_hi=None):
    chi = xyz[:, 2]
    half = 0.5 * field_width_rad(cfg) * np.maximum(chi, 1e-6)
    ok = (np.abs(xyz[:, 0]) <= half) & (np.abs(xyz[:, 1]) <= half)
    if chi_lo is not None:
        ok &= chi >= chi_lo
    if chi_hi is not None:
        ok &= chi <= chi_hi
    return ok


def parent_volume(cfg, chi_lo, chi_hi, pad=None, n=4096):
    """Comoving volume (Mpc^3) of the padded light cone between chi_lo and chi_hi."""
    chi = np.linspace(chi_lo, chi_hi, n)
    width = field_width_rad(cfg) * chi + 2.0 * (cfg["web_parent_pad_mpc"] if pad is None else pad)
    return trapz(width**2, chi)


def sample_uniform_in_parent(n, cfg, rng, chi_lo, chi_hi, pad=None):
    """n points uniform in volume inside the padded light cone, as (x, y, chi) rows."""
    if n <= 0:
        return np.empty((0, 3))
    pad = cfg["web_parent_pad_mpc"] if pad is None else pad
    grid = np.linspace(chi_lo, chi_hi, 4096)
    area = (field_width_rad(cfg) * grid + 2.0 * pad) ** 2
    cdf = np.r_[0.0, np.cumsum(0.5 * (area[1:] + area[:-1]) * np.diff(grid))]
    chi = np.interp(rng.random(n) * cdf[-1], cdf, grid)
    half = 0.5 * field_width_rad(cfg) * chi + pad
    x = rng.uniform(-half, half)
    y = rng.uniform(-half, half)
    return np.column_stack([x, y, chi])


def sample_web_nodes(n_nodes, cfg, rng, chi_lo, chi_hi):
    """Node positions: a fraction web_frac_lss scattered about Pareto-weighted supercluster centres, rest uniform."""
    n_lss = int(cfg["web_frac_lss"] * n_nodes)
    centres = sample_uniform_in_parent(cfg["web_n_super"], cfg, rng, chi_lo, chi_hi)
    weights = rng.pareto(1.5, len(centres)) + 0.5
    weights /= weights.sum()
    nodes, tries = [], 0
    while len(nodes) < n_lss and tries < max(200000, 200 * n_lss):
        p = centres[rng.choice(len(centres), p=weights)] + rng.normal(0.0, cfg["web_super_sigma"], 3)
        if inside_parent_volume(p[None, :], cfg, chi_lo, chi_hi)[0]:
            nodes.append(p)
        tries += 1
    nodes_lss = np.vstack(nodes) if len(nodes) else np.empty((0, 3))
    if len(nodes_lss) < n_lss:
        nodes_lss = np.vstack([nodes_lss, sample_uniform_in_parent(n_lss - len(nodes_lss), cfg, rng, chi_lo, chi_hi)])
    return np.vstack([nodes_lss, sample_uniform_in_parent(n_nodes - n_lss, cfg, rng, chi_lo, chi_hi)])


def connected_web_edges(nodes, cfg):
    """Filaments as (i, j, length) rows: the minimum spanning tree of a kNN graph (k grown until the graph is
    connected), plus links to each node's web_k_extra nearest neighbours closer than web_max_extra Mpc."""
    n = len(nodes)
    if n < 2:
        return np.empty((0, 3))
    tree = build_kd_tree(nodes)
    k = min(cfg["web_k_connect"] + 1, n)
    while True:
        d, ind = query_nearest(tree, nodes, k)
        rows, cols, vals = np.repeat(np.arange(n), k - 1), ind[:, 1:].ravel(), d[:, 1:].ravel()
        ok = np.isfinite(vals) & (cols != rows)
        rows, cols, vals = rows[ok], cols[ok], vals[ok]
        graph = coo_matrix((np.r_[vals, vals], (np.r_[rows, cols], np.r_[cols, rows])), shape=(n, n)).tocsr()
        if connected_components(graph, directed=False)[0] == 1 or k >= n:
            break
        k = min(n, 2 * k - 1)
    mst = minimum_spanning_tree(graph).tocoo()
    edges = set()
    for i, j in zip(mst.row, mst.col):
        edges.add(tuple(sorted((int(i), int(j)))))
    d, ind = query_nearest(tree, nodes, min(cfg["web_k_extra"] + 1, n))
    for i in range(n):
        for j, dist in zip(ind[i, 1:], d[i, 1:]):
            if dist <= cfg["web_max_extra"]:
                edges.add(tuple(sorted((int(i), int(j)))))
    return np.array([(i, j, np.linalg.norm(nodes[i] - nodes[j])) for i, j in edges], float)


def sample_filament_points(nodes, edges, n, rng, sig_filament):
    """n points along length-weighted edges with Gaussian cross-section sig_filament (Mpc).

    Returns positions, edge ids, distance from the filament axis and distance to the nearer node.
    """
    if n <= 0 or len(edges) == 0:
        return np.empty((0, 3)), np.empty(0, int), np.empty(0, float), np.empty(0, float)
    w = edges[:, 2]
    edge_id = rng.choice(len(edges), n, p=w / w.sum())
    e = edges[edge_id].astype(int)
    p0, p1, edge_len = nodes[e[:, 0]], nodes[e[:, 1]], edges[edge_id, 2]
    t = rng.uniform(0.0, 1.0, n)
    base = p0 + (p1 - p0) * t[:, None]
    v = p1 - p0
    v /= np.linalg.norm(v, axis=1)[:, None]
    a = rng.normal(size=(n, 3))
    a -= (a * v).sum(1)[:, None] * v
    a /= np.linalg.norm(a, axis=1)[:, None]
    b = np.cross(v, a)
    u1 = rng.normal(0.0, sig_filament, n)
    u2 = rng.normal(0.0, sig_filament, n)
    xyz = base + u1[:, None] * a + u2[:, None] * b
    return xyz, edge_id, np.sqrt(u1**2 + u2**2), np.minimum(t * edge_len, (1.0 - t) * edge_len)


def site_table(xyz, env, score, node_id, edge_id, dist_node, dist_fil):
    """Candidate galaxy sites (comoving x, y, line-of-sight chi) with their environment label and density score."""
    return pd.DataFrame({"x_comov": xyz[:, 0], "y_comov": xyz[:, 1], "chi_site": xyz[:, 2], "web_env": env,
                         "web_density_score": score, "web_node_id": node_id, "web_edge_id": edge_id,
                         "dist_node_mpc": dist_node, "dist_filament_mpc": dist_fil})


def node_centre_sites(nodes, richness, cfg, chi_lo, chi_hi):
    """One 'cluster_central' site on every node inside the light cone (None if there are none)."""
    keep = inside_lightcone(nodes, cfg, chi_lo, chi_hi)
    if keep.any():
        n = len(nodes)
        score = 5.0 + 0.7 * np.log10(np.maximum(richness * n, 1e-6))
        zeros = np.zeros(n)
        return site_table(nodes[keep], np.repeat("cluster_central", n)[keep], score[keep], np.arange(n)[keep],
                          np.full(n, -1, int)[keep], zeros[keep], zeros[keep])


def cluster_sites(nodes, richness, n_cluster, cfg, rng, chi_lo, chi_hi):
    """Sites around richness-weighted nodes: compact 'cluster_core' or extended 'cluster_outskirts' radii."""
    ci = rng.choice(len(nodes), n_cluster, p=richness)
    dirs = rng.normal(size=(n_cluster, 3))
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    core = rng.random(n_cluster) < cfg["web_core_frac"]
    rad = np.empty(n_cluster)
    rad[core] = rng.gamma(1.2, 0.35, core.sum())
    rad[~core] = rng.gamma(1.6, cfg["web_r_cluster"] / 2.2, (~core).sum())
    xyz = nodes[ci] + rad[:, None] * dirs
    score = np.where(core, 4.0, 3.0) + 0.5 * np.log10(np.maximum(richness[ci] * len(nodes), 1e-6)) - 0.04 * rad
    keep = inside_lightcone(xyz, cfg, chi_lo, chi_hi)
    if keep.any():
        return site_table(xyz[keep], np.where(core, "cluster_core", "cluster_outskirts")[keep], score[keep],
                          ci[keep], np.full(keep.sum(), -1, int), rad[keep], rad[keep])


def filament_sites(nodes, edges, n_filament, cfg, rng, chi_lo, chi_hi):
    """Sites scattered about node-to-node filaments; density score falls with distance from the filament and node."""
    xyz, edge_id, dist_fil, dist_node = sample_filament_points(nodes, edges, n_filament, rng, cfg["web_sig_filament"])
    score = 2.0 - 0.15 * dist_fil - 0.01 * dist_node
    keep = inside_lightcone(xyz, cfg, chi_lo, chi_hi)
    if keep.any():
        return site_table(xyz[keep], np.repeat("filament", len(xyz))[keep], score[keep], np.full(keep.sum(), -1, int),
                          edge_id[keep], dist_node[keep], dist_fil[keep])


def field_sites(n, chi_lo, chi_hi, cfg, rng):
    """n 'field' sites uniform inside the (unpadded) light cone."""
    xyz = sample_uniform_in_parent(n, cfg, rng, chi_lo, chi_hi, pad=0.0)
    return site_table(xyz, np.repeat("field", n), rng.normal(0.0, 0.15, n), np.full(n, -1, int), np.full(n, -1, int),
                      np.full(n, np.nan), np.full(n, np.nan))


def generate_web_sites(n_gal, chi_lo, chi_hi, cfg, rng):
    """Generate the web between chi_lo and chi_hi and at least n_gal * web_candidate_oversample candidate sites.

    Returns (sites, nodes, edges); if more sites than needed are drawn, the kept ones are biased to high density.
    """
    parent_vol = parent_volume(cfg, chi_lo, chi_hi)
    n_nodes = max(20, int(cfg["web_node_boost"] * cfg["web_n_nodes_100mpc"] * parent_vol / 100.0**3))
    print(f"Generating coherent global web: parent V={parent_vol:,.0f} Mpc^3, nodes={n_nodes:,}")
    nodes = sample_web_nodes(n_nodes, cfg, rng, chi_lo, chi_hi)
    richness = rng.pareto(1.7, len(nodes)) + 0.3
    richness /= richness.sum()
    edges = connected_web_edges(nodes, cfg)
    print(f"Global web edges: {len(edges):,}")

    target = max(n_gal, int(n_gal * cfg["web_candidate_oversample"]))
    all_sites: list[pd.DataFrame] = []
    n_try = max(5000, target)
    for attempt in range(8):
        # NB: sites from earlier attempts are kept, so node-centre sites repeat after a retry.
        n_cluster, n_filament = int(n_try * cfg["web_f_cluster"]), int(n_try * cfg["web_f_filament"])
        parts = [node_centre_sites(nodes, richness, cfg, chi_lo, chi_hi),
                 cluster_sites(nodes, richness, n_cluster, cfg, rng, chi_lo, chi_hi),
                 filament_sites(nodes, edges, n_filament, cfg, rng, chi_lo, chi_hi),
                 field_sites(n_try - n_cluster - n_filament, chi_lo, chi_hi, cfg, rng)]
        all_sites += [part for part in parts if part is not None]
        n_sites = sum(len(part) for part in all_sites)
        if n_sites >= target:
            break
        n_try *= 2
        print(f"Only {n_sites:,} candidate sites after attempt {attempt + 1}; increasing to {n_try:,}")

    sites = pd.concat(all_sites, ignore_index=True)
    if len(sites) > target:
        w = np.exp(cfg["web_site_keep_bias"] * sites["web_density_score"].to_numpy(float))
        sites = sites.iloc[rng.choice(len(sites), target, replace=False, p=w / w.sum())].reset_index(drop=True)
    sites.insert(0, "web_site_id", np.arange(len(sites)))
    print("Candidate site environment fractions:")
    print(sites["web_env"].value_counts(normalize=True).sort_index())
    return sites, nodes, edges
