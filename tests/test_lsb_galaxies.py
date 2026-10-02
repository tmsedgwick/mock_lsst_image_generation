"""Checks of the extended dwarfs, UDGs and almost-dark galaxies added for training."""

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.special import gammaincinv

from mock_lsst_image_generation import CONFIG, PHYS
from mock_lsst_image_generation.cosmology import build_cosmology_grids
from mock_lsst_image_generation.lsb_galaxies import (add_lsb_galaxies, sersic_total_mag,
                                                     volume_weighted_redshifts)
from mock_lsst_image_generation.photometry import REST_COLS, REST_WAVE_A

CFG = {**CONFIG, "npix": 5000}
GRIDS = build_cosmology_grids(CFG)


def fake_donors(n=300, seed=0):
    """Power-law rest-frame SEDs: star-forming dwarfs of varied blueness, plus quiescent galaxies."""
    rng = np.random.default_rng(seed)
    quiescent = np.arange(n) >= 0.8 * n
    slope = np.where(quiescent, -2.0, rng.uniform(0.2, 1.0, n))  # AB mag change per dex of wavelength
    sed = -17.0 + slope[:, None] * np.log10(REST_WAVE_A / 6222.0)[None, :]
    donors = pd.DataFrame(sed, columns=REST_COLS)
    donors["logM"] = rng.uniform(7.5, 9.0, n)
    donors["logsSFR"] = np.where(quiescent, -11.5, -9.5)
    donors["z_cosmos"] = rng.uniform(0.1, 0.5, n)
    return donors


def test_sersic_total_mag_matches_the_integrated_profile():
    for n in [0.6, 1.0, 1.2]:
        b = gammaincinv(2 * n, 0.5)
        flux, _ = quad(lambda r: 2 * np.pi * r * np.exp(-b * (r / 2.0) ** (1 / n)), 0, np.inf)  # I0 = 1, Re = 2"
        assert np.isclose(sersic_total_mag(0.0, 2.0, n), -2.5 * np.log10(flux), atol=1e-6)


def test_redshifts_are_volume_weighted_within_their_range():
    z = volume_weighted_redshifts(5000, (0.005, 0.06), GRIDS, np.random.default_rng(1))
    assert z.min() >= 0.005 and z.max() <= 0.06 and np.median(z) > 0.0325  # most volume is at the far end


def test_populations_have_their_drawn_sizes_surface_brightnesses_and_colours():
    mock = pd.DataFrame(dict(id=np.arange(10), hubble_type="Sb"))
    out = add_lsb_galaxies(mock, fake_donors(), CFG, PHYS, GRIDS, np.random.default_rng(2))
    new = out.iloc[10:]
    assert new["id"].tolist() == list(range(10, len(out)))
    for name, spec in CFG["lsb_populations"].items():
        rows = new[new["lsb_population"] == name]
        expected = spec["per_deg2"] * GRIDS["area_deg2"]
        assert abs(len(rows) - expected) < 5 * np.sqrt(expected), name
        assert rows["z"].between(*spec["z"]).all() and rows["Re_kpc"].between(*spec["re_kpc"]).all()
        recovered = sersic_total_mag(0.0, rows["re_disc_arcsec"], rows["n_disc"])
        mu0 = rows["mag_g_disc"] - recovered  # central surface brightness of the rendered disc
        assert np.allclose(mu0, rows["mu0_g"], atol=0.02), name
        assert (rows["flux_r_bulge"] < 1e-3 * rows["flux_r_disc"]).all()
    sf = new["type"] == "star_forming"
    assert (new.loc[sf, "hubble_type"] == "Irr").all() and (new.loc[sf, "irregularity"] > 0).all()
    assert new.loc[~sf, "hubble_type"].str.match(r"^E\d$").all()
    g_r = (new["mag_g_total"] - new["mag_r_total"]).groupby(new["lsb_population"]).median()
    assert g_r["almost_dark"] < g_r["extended_dirr"]  # almost-dark galaxies take the bluest donors
