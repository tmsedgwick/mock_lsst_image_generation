"""Checks of the BCG-like core-Sersic sources (ICL_DoubleSersic_Injection.ipynb)."""

import numpy as np
import pandas as pd

from mock_lsst_image_generation import CONFIG
from mock_lsst_image_generation.bcg_sources import add_bcgs, core_sersic_image, half_light_area_arcsec2, render_bcg

CFG = {**CONFIG, "npix": 5000, "bcg_per_deg2": 100}


def test_profile_is_normalised_and_centrally_peaked():
    img = core_sersic_image(re_pix=25.0, half=150, pa_rad=0.3)
    assert np.isclose(img.sum(), 1.0) and np.unravel_index(img.argmax(), img.shape) == (150, 150)


def test_rendered_sources_keep_fluxes_and_hit_their_surface_brightness():
    out = add_bcgs(pd.DataFrame(dict(id=np.arange(3))), CFG, np.random.default_rng(0))
    bcg = out.iloc[3:]
    assert len(bcg) > 0 and (bcg["profile"] == "core_sersic").all() and (bcg["hubble_type"] == "cD").all()
    psf = CFG["psf_fwhm_arcsec"]
    for _, r in bcg.head(3).iterrows():
        record = {**r.to_dict(), "x_img": 100.0, "y_img": 100.0}
        _, _, cube = render_bcg(record, psf, CFG["out_bands"], CFG["pixscale"])
        r_band = cube[CFG["out_bands"].index("r")]
        assert np.isclose(r_band.sum(), r["flux_r_total"], rtol=1e-3)
        mu = r["mag_r_total"] + 2.5 * np.log10(half_light_area_arcsec2(r_band, CFG["pixscale"]))
        assert abs(mu - r["bcg_mu_r"]) < 0.05
    g_r = bcg["mag_g_total"] - bcg["mag_r_total"]
    assert np.allclose(g_r, 0.65)
