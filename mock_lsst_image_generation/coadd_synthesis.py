"""Turn one noise-free render into LSST-like coadds at any survey depth and seeing.

Coadding is linear, so a coadd is the scene convolved with the visit-averaged PSF plus the stacked noise. The scene is
therefore rendered once at a narrow base PSF (the expensive GalSim step), and each (depth, seeing) combination is made
from it by broadening to the target PSF (Gaussian FWHMs add in quadrature) and adding noise for its number of visits.
Depth and seeing are independent axes of the grid, so a detector trained on it cannot learn a spurious link between
them. The nominal 10-year coadd is the special case of every visit at the nominal PSF.
"""

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
TEN_YEAR_NOMINAL = "10y_nominal"


def visits_at(survey_fraction, cfg):
    """Visits per band accumulated after this fraction of the 10-year survey (at least one)."""
    return {band: max(1, int(round(cfg["visits_10yr"][band] * survey_fraction))) for band in cfg["bands"]}


def jittered_band_fwhm(fwhm_r, seed, cfg):
    """Per-band PSF FWHM for one seeing value: the r-band FWHM scaled by the nominal band ratios, each with a small
    random jitter, and kept above the base PSF so the broadening kernel exists."""
    rng = np.random.default_rng(seed)
    nominal = cfg["nominal_fwhm"]
    fwhm = {}
    for band in cfg["bands"]:
        jitter = float(rng.normal(0.0, cfg["fwhm_ratio_jitter"]))
        fwhm[band] = float(max(fwhm_r * (nominal[band] / nominal["r"]) * (1.0 + jitter), cfg["base_fwhm"] + 0.02))
    return fwhm


def coadd_grid(catalogue_index, cfg) -> dict[str, dict[str, Any]]:
    """Every (epoch, seeing) combination for one catalogue, as {key: settings}, with key e.g. '1y_fwhm110'.

    Each combination's jitter and noise come from its own seed, SeedSequence([catalogue_index, epoch index, seeing
    index]), so any coadd can be rebuilt on its own (e.g. on the fly during training) and is identical every time.
    """
    grid = {}
    for epoch_index, (epoch, survey_fraction) in enumerate(cfg["epochs"].items()):
        for fwhm_index, fwhm_r in enumerate(cfg["fwhm_grid_r"]):
            seed_entropy = [catalogue_index, epoch_index, fwhm_index]
            jitter_seed, noise_seed = np.random.SeedSequence(seed_entropy).spawn(2)
            grid[f"{epoch}_fwhm{int(round(fwhm_r * 100)):03d}"] = dict(
                epoch=epoch, survey_fraction=survey_fraction, f_r=fwhm_r, n_visit=visits_at(survey_fraction, cfg),
                fwhm_arcsec=jittered_band_fwhm(fwhm_r, jitter_seed, cfg), seed_entropy=seed_entropy,
                noise_seed=noise_seed)
    return grid


def ten_year_nominal(catalogue_index, cfg) -> dict[str, dict[str, Any]]:
    """The full-depth coadd at the nominal PSF in every band, in the same form as a coadd_grid entry."""
    seed_entropy = [catalogue_index, 999, 999]  # outside the grid's (epoch, seeing) indices, so never shared with it
    return {TEN_YEAR_NOMINAL: dict(epoch="10y", survey_fraction=1.0, f_r=cfg["nominal_fwhm"]["r"],
                                   n_visit=visits_at(1.0, cfg), fwhm_arcsec=dict(cfg["nominal_fwhm"]),
                                   seed_entropy=seed_entropy, noise_seed=np.random.SeedSequence(seed_entropy))}


def broaden(image, target_fwhm, cfg):
    """Broaden a base-PSF image to the per-band target FWHM (arcsec) with a Gaussian of the quadrature difference."""
    base_fwhm = cfg["base_fwhm"]
    broadened = np.empty_like(image)
    for i, band in enumerate(cfg["bands"]):
        target = float(target_fwhm[band])
        if target < base_fwhm:
            raise ValueError(f"target FWHM {target:.3f} < base {base_fwhm:.3f} for band {band}")
        sigma_pix = (np.sqrt(target ** 2 - base_fwhm ** 2) * FWHM_TO_SIGMA) / cfg["pixscale"]
        broadened[i] = image[i] if sigma_pix <= 0 else gaussian_filter(image[i], sigma=sigma_pix, mode="constant",
                                                                       cval=0.0, truncate=5.0)
    return broadened


def sky_sigma(band, n_visit, cfg):
    """Per-pixel sky noise (nJy) of an n_visit coadd: the 10-year value, from the 5-sigma point-source depth over a
    nominal-PSF aperture, scaled by sqrt(10-year visits / n_visit). It depends on depth only, never on the seeing."""
    n_pix_psf = 1.13 * (cfg["nominal_fwhm"][band] / cfg["pixscale"]) ** 2
    sigma_10yr = 10 ** (-0.4 * (cfg["depth_10yr"][band] - cfg["zeropoint"])) / (5 * np.sqrt(n_pix_psf))
    return sigma_10yr * np.sqrt(cfg["visits_10yr"][band] / n_visit)


def add_noise(image, n_visit, seed, cfg):
    """Noisy coadd and its variance: sky Gaussian noise plus source Poisson noise averaged over n_visit visits."""
    rng = np.random.default_rng(seed)
    signal, variance = image.copy(), np.zeros_like(image)
    for i, band in enumerate(cfg["bands"]):
        visits = int(n_visit[band])
        sky = sky_sigma(band, visits, cfg)
        source = np.maximum(image[i], 0)
        signal[i] += rng.poisson(visits * source).astype(np.float32) / visits - source
        signal[i] += rng.normal(0.0, sky, image[i].shape).astype(np.float32)
        variance[i] = sky ** 2 + source / visits
    return signal, variance


def synthesise_coadd(base_image, coadd, cfg):
    """(signal, variance) float32 cubes for one coadd_grid / ten_year_nominal entry, from the base-PSF render."""
    signal, variance = add_noise(broaden(base_image, coadd["fwhm_arcsec"], cfg), coadd["n_visit"], coadd["noise_seed"],
                                 cfg)
    return signal.astype(np.float32, copy=False), variance.astype(np.float32, copy=False)
