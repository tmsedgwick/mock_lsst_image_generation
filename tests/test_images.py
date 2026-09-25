"""Checks that images render with the catalogue's flux, coadds carry the stated noise, and each mode writes the right
files. Uses a small 300 x 300 px catalogue, rendered in-process."""

import contextlib
import io
import json

import numpy as np
import pytest

from mock_lsst_image_generation import (IMAGE_CONFIG, build_mock_catalogue, generate_catalogue_images,
                                        load_cosmos2025_catalogue)
from mock_lsst_image_generation.coadd_synthesis import TEN_YEAR_NOMINAL, broaden, synthesise_coadd, ten_year_nominal
from mock_lsst_image_generation.galaxy_rendering import render_catalogue

N_COADDS = len(IMAGE_CONFIG["epochs"]) * len(IMAGE_CONFIG["fwhm_grid_r"])


@pytest.fixture(scope="module")
def catalogue():
    with contextlib.redirect_stdout(io.StringIO()):
        return build_mock_catalogue(load_cosmos2025_catalogue(), dict(seed=42, npix=300))


@pytest.fixture(scope="module")
def base_image(catalogue):
    return render_catalogue(catalogue, {band: IMAGE_CONFIG["base_fwhm"] for band in IMAGE_CONFIG["bands"]},
                            IMAGE_CONFIG)[0]


def test_render_keeps_catalogue_flux(catalogue, base_image):
    total_r = catalogue.galaxies["flux_r_total"].sum() + catalogue.clumps["flux_r_clump"].sum()
    assert base_image.shape == (6, 300, 301) and base_image.dtype == np.float32
    assert 0.9 < base_image[2].sum() / total_r <= 1.0  # a little light falls off the frame edges


def test_coadd_noise_matches_variance(base_image):
    coadd = ten_year_nominal(0, IMAGE_CONFIG)[TEN_YEAR_NOMINAL]
    signal, variance = synthesise_coadd(base_image, coadd, IMAGE_CONFIG)
    pulls = (signal - broaden(base_image, coadd["fwhm_arcsec"], IMAGE_CONFIG)) / np.sqrt(variance)
    assert np.std(pulls) == pytest.approx(1.0, abs=0.02)
    assert np.array_equal(signal, synthesise_coadd(base_image, coadd, IMAGE_CONFIG)[0])  # same seed, same noise


@pytest.mark.parametrize("name, mode, n_saved", [("train", "training", 0), ("test", "training", N_COADDS),
                                                 ("train", "all", N_COADDS), ("train", "ten_year", 1)])
def test_modes_write_expected_files(catalogue, tmp_path, name, mode, n_saved):
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        generate_catalogue_images(catalogue, name, tmp_path, mode)
    out = tmp_path / name
    manifest = json.loads((out / "coadd_manifest.json").read_text())
    assert (out / "base_clean_signal.npy").exists() and (out / "base_meta.json").exists()
    assert len(manifest["combos"]) == (1 if mode == "ten_year" else N_COADDS)
    assert len(list(out.glob(f"{name}_*_signal.npy"))) == len(list(out.glob(f"{name}_*_variance.npy"))) == n_saved
    if mode == "ten_year":
        assert manifest["combos"][TEN_YEAR_NOMINAL]["fwhm_arcsec"] == IMAGE_CONFIG["nominal_fwhm"]
