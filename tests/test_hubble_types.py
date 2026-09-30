"""Checks of the Hubble types drawn for catalogues and of the arms / bars / clumps drawn for them."""

import numpy as np
import pandas as pd

from mock_lsst_image_generation.galaxy_rendering import render_galaxy
from mock_lsst_image_generation.galaxy_structure import structure_images
from mock_lsst_image_generation.hubble_types import (BROAD, add_hubble_types, broad_probabilities, class_density,
                                                     draw_structure)

TYPES = {f"E{n}" for n in range(8)} | {"S0", "SB0", "Sa", "Sb", "Sc", "SBa", "SBb", "SBc", "Irr"}


def test_probabilities_are_normalised_everywhere():
    logM, z = np.meshgrid(np.linspace(6, 13, 30), np.linspace(0, 6, 30))
    for passive in [False, True]:
        p = broad_probabilities(logM.ravel(), z.ravel(), np.full(logM.size, passive))
        assert np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(axis=1), 1)


def test_low_redshift_fractions_match_gama():
    """Kelvin et al. 2014 (GAMA, z ~ 0.04, log M > 9): E 19%, S0-Sa 18%, Sab-Scd 28%, Sd-Irr 31%. Our lowest-redshift
    HC16 bin (0.2 < z < 0.5), summed over star-forming and passive galaxies, should be close."""
    logM = np.linspace(9, 12, 300)
    counts = np.array([sum(np.trapezoid(class_density(logM, cls, "0.2-0.5", state), logM) for state in ["sf", "q"])
                       for cls in BROAD])
    assert np.allclose(counts / counts.sum(), [0.19, 0.18, 0.28, 0.31], atol=0.13)


def test_dwarfs_are_mostly_irregular_and_massive_passive_galaxies_mostly_spheroids():
    dwarf = broad_probabilities([8.0], [0.5], [False])[0]
    giant = broad_probabilities([11.0], [0.5], [True])[0]
    assert dwarf[BROAD.index("IRR")] > 0.5 and giant[BROAD.index("SPH")] > 0.4


def test_add_hubble_types_labels_galaxies_only_and_is_reproducible():
    rng = np.random.default_rng(0)
    n = 2000
    table = pd.DataFrame(dict(type=np.where(rng.random(n) < 0.3, "passive", "star_forming"), logM=rng.uniform(7, 11.5, n),
                              z=rng.uniform(0, 3, n), ellipticity_total=rng.uniform(0, 0.8, n)))
    table.loc[:9, "type"] = "star"
    typed = add_hubble_types(table, np.random.default_rng(1))
    assert typed.loc[:9, "hubble_type"].isna().all()
    assert set(typed.loc[10:, "hubble_type"]) <= TYPES and len(set(typed.loc[10:, "hubble_type"])) > 10
    assert typed.equals(add_hubble_types(table, np.random.default_rng(1)))


def galaxy_record(hubble_type, **overrides):
    structure = draw_structure(np.array([hubble_type], dtype=object), np.random.default_rng(3)).iloc[0].to_dict()
    record = dict(x_img=60.3, y_img=58.7, pa_deg=30.0, re_bulge_arcsec=0.5, re_disc_arcsec=2.0,
                  ellipticity_bulge=0.2, ellipticity_disc=0.4, n_bulge=4.0, n_disc=1.0, hubble_type=hubble_type,
                  **structure, **{f"flux_{band}_{c}": 1e4 for band in "gri" for c in ["bulge", "disc"]})
    return {**record, **overrides}


def test_structure_moves_disc_light_without_changing_its_flux():
    offsets = np.arange(-60, 61) * 0.2
    for hubble_type in ["SBb", "Sc", "Irr"]:
        bar, arms, depth = structure_images(galaxy_record(hubble_type), offsets, offsets)
        assert 0 < depth <= 1
        assert abs(bar.sum()) < 1e-6 and abs(arms.sum()) < 1e-6
        assert np.abs(bar).max() + np.abs(arms).max() > 1e-4


def test_structure_never_makes_negative_light():
    """Edge-on and face-on discs of every structured type: the stamp stays non-negative in every band."""
    psf = dict(g=0.5, r=0.5, i=0.5)
    for hubble_type in ["SBb", "Sa", "Sc", "Irr"]:
        for ellipticity in [0.0, 0.5, 0.85]:
            record = galaxy_record(hubble_type, ellipticity_disc=ellipticity)
            _, _, plain = render_galaxy(record, psf, "gri", 0.2, 25)
            _, _, cube = render_galaxy(record, psf, "gri", 0.2, 25, structure_min_re_arcsec=0.4)
            assert cube.min() > -1e-3 * plain.max(), (hubble_type, ellipticity)


def test_rendered_spiral_keeps_band_fluxes():
    psf = dict(g=0.5, r=0.5, i=0.5)
    record = galaxy_record("SBc")
    _, _, plain = render_galaxy(record, psf, "gri", 0.2, 25)
    _, _, spiral = render_galaxy(record, psf, "gri", 0.2, 25, structure_min_re_arcsec=0.4)
    assert np.allclose(spiral.sum(axis=(1, 2)), plain.sum(axis=(1, 2)), rtol=0.01)
    assert np.abs(spiral - plain).max() > 0.05 * plain.max()
    assert spiral.min() > -1e-3 * plain.max()
