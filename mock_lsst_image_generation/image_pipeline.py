"""Render each mock catalogue once and write the coadds needed for training and evaluation.

For catalogue <name>, <out_dir>/<name>/ holds:
  base_clean_signal.npy           noise-free render at the narrow base PSF (float32, band x y x, nJy per pixel)
  base_meta.json                  bands, canvas origin in frame pixels, shape, pixel scale, zeropoint and base PSF
  coadd_manifest.json             every coadd: visits and PSF FWHM per band, noise seed, and whether it was saved
  <name>_<coadd>_signal.npy       saved coadds (nJy per pixel) ...
  <name>_<coadd>_variance.npy     ... and their per-pixel variance

Modes:
  training   (default) every (epoch, seeing) coadd goes in the manifest, but only calib/test coadds are saved. Training
             code rebuilds the train/valid ones on the fly from the base render and manifest, so they cost no disk.
  all        every coadd is saved for every catalogue.
  ten_year   only the nominal 10-year coadd, saved for every catalogue.
"""

import gc
import json
from pathlib import Path

import numpy as np

from .catalogue_pipeline import load_mock_catalogue
from .coadd_synthesis import coadd_grid, synthesise_coadd, ten_year_nominal
from .config import CATALOGUE_STEM, IMAGE_CONFIG, catalogue_index
from .galaxy_rendering import render_catalogue

MODES = ("training", "all", "ten_year")


def available_catalogues(catalogue_dir, stem=CATALOGUE_STEM):
    """Names of the catalogues saved in catalogue_dir, in train, valid, calib, test, extra1, ... order."""
    names = []
    for path in Path(catalogue_dir).glob(f"{stem}_*.csv"):
        name = path.stem[len(stem) + 1:]
        try:
            names.append((catalogue_index(name), name))
        except ValueError:  # the _clumps / _tidal / _tidal_pairs companions
            pass
    return [name for _, name in sorted(names)]


def write_json(path, content):
    with Path(path).open("w") as file:
        json.dump(content, file, indent=2)


def generate_catalogue_images(catalogue, name, out_dir, mode="training", cfg=None, n_workers=1):
    """Render one catalogue and write its base image, manifest and coadds (see the module docstring)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    cfg = {**IMAGE_CONFIG, **(cfg or {})}
    for band in cfg["bands"]:
        if min(cfg["fwhm_grid_r"]) * cfg["nominal_fwhm"][band] / cfg["nominal_fwhm"]["r"] <= cfg["base_fwhm"]:
            raise ValueError(f"base_fwhm {cfg['base_fwhm']} must be narrower than every target PSF ({band} band)")
    out_dir = Path(out_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n===== [{name}] rendering at the base PSF, FWHM = {cfg['base_fwhm']}\" =====", flush=True)
    base_image, origin = render_catalogue(catalogue, {band: cfg["base_fwhm"] for band in cfg["bands"]}, cfg, n_workers)
    np.save(out_dir / "base_clean_signal.npy", base_image)
    write_json(out_dir / "base_meta.json", dict(split=name, origin=list(origin), bands=cfg["bands"],
                                                base_fwhm=cfg["base_fwhm"], shape=list(base_image.shape),
                                                ps=cfg["pixscale"], zp=cfg["zeropoint"]))

    index = catalogue_index(name)
    coadds = ten_year_nominal(index, cfg) if mode == "ten_year" else coadd_grid(index, cfg)
    save = mode != "training" or name in cfg["saved_catalogues"]
    manifest_coadds = {}
    for key, coadd in coadds.items():
        if save:
            signal, variance = synthesise_coadd(base_image, coadd, cfg)
            np.save(out_dir / f"{name}_{key}_signal.npy", signal)
            np.save(out_dir / f"{name}_{key}_variance.npy", variance)
            del signal, variance
            gc.collect()
        manifest_coadds[key] = {**{k: v for k, v in coadd.items() if k != "noise_seed"},
                                   "fwhm_arcsec": {band: round(v, 4) for band, v in coadd["fwhm_arcsec"].items()}}
    write_json(out_dir / "coadd_manifest.json", dict(split=name, base_fwhm=cfg["base_fwhm"], bands=cfg["bands"],
                                                     materialised=save, combos=manifest_coadds))
    print(f"[{name}] {len(coadds)} coadds in the manifest, {len(coadds) if save else 0} saved -> {out_dir}")


def generate_mock_images(catalogue_dir, out_dir, names=None, mode="training", cfg=None, n_workers=1,
                         stem=CATALOGUE_STEM):
    """Generate images for the named catalogues in catalogue_dir (default: every catalogue there)."""
    available = available_catalogues(catalogue_dir, stem)
    if not available:
        raise FileNotFoundError(f"No {stem}_<name>.csv catalogues in {catalogue_dir}: run generate_mock_catalogues.py")
    missing = sorted(set(names or []) - set(available))
    if missing:
        raise FileNotFoundError(f"Catalogues {missing} not found in {catalogue_dir}; available: {available}")
    for name in names or available:
        catalogue = load_mock_catalogue(Path(catalogue_dir) / f"{stem}_{name}.csv")
        generate_catalogue_images(catalogue, name, out_dir, mode, cfg, n_workers)
