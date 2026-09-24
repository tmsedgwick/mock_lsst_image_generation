"""Fast checks that the catalogue pipeline still runs and still produces the same catalogues.

The regression test pins the seed-42, 1000 x 1000 px realisation built from the shipped COSMOS subset. If you change
the model on purpose, rerun it, check the new catalogue is what you intended, and update REFERENCE.
"""

import contextlib
import io

import numpy as np
import pytest

from mock_lsst_image_generation import (CONFIG, DEFAULT_COSMOS_PATH, build_mock_catalogue, catalogue_splits,
                                        generate_mock_catalogues, load_cosmos2025_catalogue)

REFERENCE = dict(n_galaxies=4322, n_clumps=6344, n_tidal_pairs=2, n_tidal_blobs=324, n_donors=18729,
                 sums=dict(z=9603.46518814514, logM=34476.06135959138, mag_r_total=118111.28531702518,
                           re_total_arcsec=1434.8673316593267))


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


@pytest.fixture(scope="module")
def cosmos():
    return load_cosmos2025_catalogue()


@pytest.fixture(scope="module")
def catalogue(cosmos):
    return quiet(build_mock_catalogue, cosmos, dict(seed=42, npix=1000))


def test_shipped_cosmos_subset_loads(cosmos):
    assert DEFAULT_COSMOS_PATH.exists()
    assert len(cosmos) == 87286
    assert {"z", "logM", "logsSFR", "m_r"}.issubset(cosmos.columns)


def test_catalogue_splits_are_stable():
    assert catalogue_splits(4) == dict(train=42, valid=101, calib=202, test=303)
    assert catalogue_splits(6) == {**catalogue_splits(4), "extra1": 404, "extra2": 505}
    with pytest.raises(ValueError):
        catalogue_splits(0)


def test_catalogue_matches_reference(catalogue):
    g = catalogue.galaxies
    assert len(g) == REFERENCE["n_galaxies"]
    assert len(catalogue.clumps) == REFERENCE["n_clumps"]
    assert len(catalogue.tidal_pairs) == REFERENCE["n_tidal_pairs"]
    assert len(catalogue.tidal_blobs) == REFERENCE["n_tidal_blobs"]
    assert len(catalogue.donors) == REFERENCE["n_donors"]
    for col, ref in REFERENCE["sums"].items():
        assert g[col].sum() == pytest.approx(ref, rel=1e-9), col


def test_catalogue_is_physically_sane(catalogue):
    g = catalogue.galaxies
    npix = 1000
    assert g["id"].tolist() == list(range(len(g)))
    assert g["x_pix"].between(0, npix).all() and g["y_pix"].between(0, npix).all()
    assert g["z"].between(CONFIG["z_min"], CONFIG["z_max"]).all()
    assert g["logM"].between(CONFIG["logm_min"], CONFIG["logm_max"]).all()
    for band in CONFIG["out_bands"]:
        assert np.isfinite(g[f"mag_{band}_total"]).all() and (g[f"flux_{band}_total"] > 0).all(), band
    assert (g["sb_r_total"] <= CONFIG["render_mu_r_max"]).all()
    assert set(catalogue.clumps["parent_id"]).issubset(g["id"])


def test_same_seed_same_catalogue_different_seed_differs(cosmos, catalogue):
    again = quiet(build_mock_catalogue, cosmos, dict(seed=42, npix=1000)).galaxies
    assert again.equals(catalogue.galaxies)
    other = quiet(build_mock_catalogue, cosmos, dict(seed=7, npix=1000)).galaxies
    assert not other.equals(catalogue.galaxies)


def test_generate_writes_all_csvs(cosmos, tmp_path):
    quiet(generate_mock_catalogues, cosmos, 1, tmp_path, cfg=dict(npix=300))
    stem = tmp_path / "mock_catalogue_train"
    for suffix in ["", "_clumps", "_tidal", "_tidal_pairs"]:
        assert (tmp_path / f"{stem.name}{suffix}.csv").exists(), suffix
