"""Checks that images render with the catalogue's flux, coadds carry the stated noise, and each mode writes the right
files. Uses a small 300 x 300 px catalogue, rendered in-process."""

import contextlib
import io
import json

import numpy as np
import pytest

from mock_lsst_image_generation import (IMAGE_CONFIG, build_mock_catalogue, generate_catalogue_images,
                                        load_cosmos2025_catalogue)
from mock_lsst_image_generation.coadd_synthesis import broaden, single_coadd, survey_fraction, synthesise_coadd
from mock_lsst_image_generation.galaxy_rendering import render_catalogue

N_COADDS = len(IMAGE_CONFIG["epochs"]) * len(IMAGE_CONFIG["fwhm_grid_r"])


@pytest.fixture(scope="module")
def catalogue():
    with contextlib.redirect_stdout(io.StringIO()):
        return build_mock_catalogue(load_cosmos2025_catalogue(), dict(seed=42, npix=300, stars=False))


@pytest.fixture(scope="module")
def rendered(catalogue):
    """(base image, {component: its light alone})."""
    components = {}
    image = render_catalogue(catalogue, {band: IMAGE_CONFIG["base_fwhm"] for band in IMAGE_CONFIG["bands"]},
                             IMAGE_CONFIG, components_out=components)[0]
    return image, components


@pytest.fixture(scope="module")
def base_image(rendered):
    return rendered[0]


def test_component_images_hold_their_own_light(catalogue, rendered):
    image, components = rendered
    assert set(components) == {"clumps", "tidal", "spikes"}
    assert all(light.shape == image.shape for light in components.values())
    assert 0 < components["clumps"][2].sum() <= catalogue.clumps["flux_r_clump"].sum()
    assert components["clumps"].min() > -1e-3 * components["clumps"].max()  # only rendering ringing below 0
    assert not components["spikes"].any()  # this catalogue has no stars
    assert components["tidal"].sum() < image.sum()


def test_render_keeps_catalogue_flux(catalogue, base_image):
    total_r = catalogue.galaxies["flux_r_total"].sum() + catalogue.clumps["flux_r_clump"].sum()
    # The canvas encloses galaxy centres, so its dimensions depend on the realisation.
    g = catalogue.galaxies
    nx = int(np.ceil(g["x_pix"].max()) - np.floor(g["x_pix"].min())) + 1
    ny = int(np.ceil(g["y_pix"].max()) - np.floor(g["y_pix"].min())) + 1
    assert base_image.shape == (len(IMAGE_CONFIG["bands"]), ny, nx)
    assert base_image.dtype == np.float32
    assert 0.9 < base_image[2].sum() / total_r <= 1.0  # a little light falls off the frame edges


def test_coadd_noise_matches_variance(base_image):
    coadd = single_coadd(0, IMAGE_CONFIG, epoch="10y")["10y_nominal"]
    signal, variance = synthesise_coadd(base_image, coadd, IMAGE_CONFIG)
    pulls = (signal - broaden(base_image, coadd["fwhm_arcsec"], IMAGE_CONFIG)) / np.sqrt(variance)
    assert np.std(pulls) == pytest.approx(1.0, abs=0.02)
    assert np.array_equal(signal, synthesise_coadd(base_image, coadd, IMAGE_CONFIG)[0])  # same seed, same noise


def test_single_coadd_options():
    assert survey_fraction("10y") == 1.0 and survey_fraction("6m") == 0.05
    with pytest.raises(ValueError, match="choose from 1m, 2m"):
        survey_fraction("12y")
    coadd = single_coadd(0, IMAGE_CONFIG, n_exp=92, psf_fwhm=1.3)["nexp92_fwhm130"]
    assert coadd["n_visit"]["r"] == 92 and coadd["fwhm_arcsec"]["r"] == 1.3
    for bad in [dict(n_exp=92), dict(epoch="10y", psf_fwhm=1.3), dict(n_exp=92, psf_fwhm=0.3)]:
        with pytest.raises(ValueError):
            single_coadd(0, IMAGE_CONFIG, **bad)


@pytest.mark.parametrize("name, mode, n_saved", [("train", "training", 0), ("test", "training", N_COADDS),
                                                 ("2", "training", N_COADDS), ("train", "all", N_COADDS),
                                                 ("train", "single", 1)])
def test_modes_write_expected_files(catalogue, tmp_path, name, mode, n_saved):
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        generate_catalogue_images(catalogue, name, tmp_path, mode, epoch="10y" if mode == "single" else None)
    out = tmp_path / name
    manifest = json.loads((out / "coadd_manifest.json").read_text())
    assert (out / "base_clean_signal.npy").exists() and (out / "base_meta.json").exists()
    assert len(manifest["combos"]) == (1 if mode == "single" else N_COADDS)
    assert len(list(out.glob(f"{name}_*_signal.npy"))) == len(list(out.glob(f"{name}_*_variance.npy"))) == n_saved
    if mode == "single":
        assert manifest["combos"]["10y_nominal"]["fwhm_arcsec"] == IMAGE_CONFIG["nominal_fwhm"]
