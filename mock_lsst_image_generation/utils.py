"""Small helpers shared across modules: numeric columns, named cut masks, neighbour sampling, resolvedness."""

import numpy as np
import pandas as pd
from scipy.spatial import KDTree


def build_kd_tree(points):
    """scipy KDTree with cKDTree's leaf size (16): the same tree as cKDTree, so tied neighbours keep the same order."""
    return KDTree(points, leafsize=16)


def query_nearest(tree, points, k, workers=1) -> tuple[np.ndarray, np.ndarray]:
    """Distances and indices of the k nearest tree points to each query point, always as arrays."""
    dist, ind = tree.query(points, k=k, workers=workers)
    return np.asarray(dist), np.asarray(ind)


def numeric(df, col):
    """Column as floats, with unparsable entries -> NaN."""
    return pd.to_numeric(df[col], errors="coerce")


def within(values, lo=None, hi=None, inclusive=True):
    """Finite values inside [lo, hi] (or (lo, hi) if not inclusive); a None bound is ignored."""
    ok = np.isfinite(values)
    if lo is not None:
        ok &= values >= lo if inclusive else values > lo
    if hi is not None:
        ok &= values <= hi if inclusive else values < hi
    return ok


def combined_mask(checks, index):
    """AND of every mask in a list of (name, mask) checks."""
    keep = pd.Series(True, index=index)
    for _, ok in checks:
        keep &= ok
    return keep


def print_cut_summary(message, checks):
    print(message)
    for name, ok in checks:
        print(f"  failed {name}: {int((~ok).sum()):,}")


def pick_kernel_weighted_neighbour(dist, ind, rng, fallback_scale, min_scale):
    """Draw one of k nearest neighbours with Gaussian weights in distance, scaled by the median non-zero distance."""
    scale = max(np.nanmedian(dist[dist > 0]) if np.any(dist > 0) else fallback_scale, min_scale)
    w = np.exp(-0.5 * (dist / scale) ** 2)
    return ind[rng.choice(len(ind), p=w / w.sum())]


def resolution_fwhm_arcsec(cfg):
    """PSF FWHM a galaxy's size must exceed to count as resolved (psf_fwhm_arcsec may be per-band or a number)."""
    fwhm = cfg["psf_fwhm_arcsec"]
    return float(fwhm[cfg["resolved_psf_band"]]) if isinstance(fwhm, dict) else float(fwhm)


def is_resolved(row, cfg):
    """True if the galaxy's Re exceeds the PSF FWHM, i.e. it can host star-forming regions and tidal features."""
    re = float(row.get(cfg["resolved_re_col"], np.nan))
    return np.isfinite(re) and re > resolution_fwhm_arcsec(cfg)


def resolved_mask(m, cfg):
    """Vectorised is_resolved: Re above the PSF FWHM in cfg['resolved_psf_band']."""
    re = numeric(m, cfg["resolved_re_col"]).to_numpy(float)
    return np.isfinite(re) & (re > resolution_fwhm_arcsec(cfg))
