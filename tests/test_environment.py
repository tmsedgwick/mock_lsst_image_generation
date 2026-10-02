"""Checks of the class-dependent environment draw (joint mass / star formation / environment)."""

import numpy as np
import pandas as pd

from mock_lsst_image_generation import CONFIG
from mock_lsst_image_generation.cosmology import build_cosmology_grids
from mock_lsst_image_generation.environment_sampling import draw_environments, reference_classes
from mock_lsst_image_generation.photometry import REST_COLS, REST_WAVE_A

GRIDS = build_cosmology_grids(CONFIG)


def galaxies(red, lsb, n=4000):
    """n galaxies at z = 0.05 with power-law SEDs: red or blue, compact or large and faint (LSB)."""
    slope = -6.0 if red else 0.5  # AB mag change per dex of wavelength (g - i ~ 1.2 or -0.1)
    sed = (-14.5 if lsb else -19.0) + slope * np.log10(REST_WAVE_A / 6222.0)
    table = pd.DataFrame(np.tile(sed, (n, 1)), columns=REST_COLS)
    return table.assign(z=0.05, Re_kpc=6.0 if lsb else 1.5, logM=9.0)


def test_classes_follow_the_t24_cuts():
    for red in [True, False]:
        for lsb in [True, False]:
            expected = ("red" if red else "blue") + ("_lsb" if lsb else "_hsb")
            assert (reference_classes(galaxies(red, lsb, 10), GRIDS) == expected).all()


def test_environment_probabilities_follow_the_class():
    rng = np.random.default_rng(0)
    for (red, lsb), name in [((True, True), "red_lsb"), ((False, False), "blue_hsb")]:
        env, _ = draw_environments(galaxies(red, lsb), CONFIG, GRIDS, rng)
        p_structure, p_cluster = CONFIG["environment_by_class"][name]
        assert abs(np.mean(env != "field") - p_structure) < 0.03
        in_structure = env[env != "field"]
        assert abs(np.mean(np.char.startswith(in_structure.astype(str), "cluster")) - p_cluster) < 0.05
    env, _ = draw_environments(galaxies(True, True), CONFIG, GRIDS, rng)
    assert not (env == "cluster_core").any()  # red LSB galaxies live in cluster outskirts (T24)
