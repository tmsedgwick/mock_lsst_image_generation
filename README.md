# mock_lsst_image_generation

Mock LSST-like galaxy catalogues and images for training and testing source detection and deblending, built from
physical models and COSMOS2025 data. Every galaxy is a known truth entry, so any detector's completeness and purity can
be measured exactly.

Two commands: one builds the catalogues, the other renders them into images. Both require their output folder; keep
it outside the repo, since a full image set can be ~200 GB.

```bash
python scripts/generate_mock_catalogues.py --out-dir ~/mocks/catalogues
python scripts/generate_mock_images.py --catalogue-dir ~/mocks/catalogues --out-dir ~/mocks/images
```

## What the catalogue models

- **Positions:** galaxies placed on a cosmic web (clusters, filaments, field) along a light cone
- **Type and stellar mass:** evolving star-forming and quiescent stellar mass functions
- **Star formation, colours and sizes:** drawn from similar real galaxies in COSMOS2025 and shifted to each mock
  galaxy's redshift, giving LSST ugrizy photometry
- **Structure:** bulge + disc
- **Extras:** star-forming clumps and tidal bridges between interacting pairs
- **Stars:** bright LSST-like stars with band-dependent diffraction spikes, halos and saturated cores, calibrated on
  a real LSSTCam deep coadd (`stars.py`, settings in `STAR_CONFIG`)
- **Hubble types:** each galaxy gets a type (E0-E7, S0, Sa-Sc, barred SB0-SBc, Irr), drawn with probabilities
  from published morphological mass functions (Huertas-Company et al. 2016), T-type ratios (Nair & Abraham 2010)
  and bar fractions (Erwin 2018; Melvin et al. 2014). Resolved discs are drawn with the spiral arms, bar or
  irregular clumps of their type, as light moved within the disc so band fluxes are unchanged (`hubble_types.py`,
  `galaxy_structure.py`)

`catalogue_pipeline.build_mock_catalogue` ties these together. All settings live in `config.py` (`CONFIG`, `PHYS`,
and `IMAGE_CONFIG` for the images); pass a partial dict to override any of them.

## Repository layout

| Path | Contents |
|---|---|
| `mock_lsst_image_generation/` | The package: one module per model ingredient, `config.py` for all settings |
| `scripts/generate_mock_catalogues.py` | Command: build the catalogues |
| `scripts/generate_mock_images.py` | Command: render the catalogues into images |
| `data/cosmos2025_subset.npz` | Shipped COSMOS2025 input (see below) |
| `tests/` | pytest suite, run by GitHub Actions on every push |

## Install

Create a virtual environment inside the repo and install the pinned dependencies plus the package itself:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the exact versions the pipeline was verified with (including GalSim), so catalogues and
images are reproducible. To use your own versions instead, run `pip install -e .`. Point your editor's Python interpreter at
`.venv/bin/python` so it can resolve the imports.

The repo ships `data/cosmos2025_subset.npz` (17 MB), used by default. It holds only the columns the pipeline reads and
the 87k rows (of 784k) that can pass its COSMOS cuts, and gives byte-identical catalogues to the full
`cosmos2025_cat.npz` (~1.5 GB) at the default settings. If you loosen the COSMOS training/sSFR-PDF cuts in `CONFIG`,
pass the full catalogue instead.

## Generate catalogues

```bash
python scripts/generate_mock_catalogues.py --out-dir ~/mocks/catalogues   # add --cosmos /path/to/cosmos2025_cat.npz for the full table
```

This builds four catalogues by default: `train`, `valid`, `calib` and `test`. Pass `--n-catalogues N` to build any
number; catalogues beyond the fourth are named `extra1`, `extra2`, and so on. Each catalogue is an independent
realisation of a 5000 × 5000 pixel frame (0.2″ pixels), and only the seed differs. Catalogue k always gets the same
name and seed, so asking for more never changes the ones you already have.

If you are not training a model, `--numbered` names them `1`, `2`, `3`, ... instead (same seeds, so `1` is the same
realisation as `train`).

Each catalogue writes four CSVs to `--out-dir`:

| File | Contents |
|---|---|
| `mock_catalogue_<name>.csv` | Galaxies: position, redshift, environment, physical properties, and bulge/disc/total photometry and structure. Stars are extra rows with `type` = `star` |
| `…_clumps.csv` | Star-forming clumps: parent id, offsets, width, per-band flux |
| `…_tidal.csv` | Tidal-bridge blobs: one row per blob and band |
| `…_tidal_pairs.csv` | The interacting pairs |

Fluxes are in nJy (AB zeropoint 31.4). Sizes are half-light radii in arcsec. `x_pix` and `y_pix` are frame pixel
coordinates.

Star rows have positions, `mag_<band>_total` / `flux_<band>_total` for ugrizy, and the parameters that shape their
spikes, halo and saturation (galaxy-only columns are empty for them). Stars have their own random stream, so the
galaxies are identical with or without them; pass `dict(stars=False)` to `build_mock_catalogue` to leave them out.
Their spike angles (shared by every star in an image) and saturated cores are drawn when the images are made.

Galaxy rows also carry `hubble_type` and the parameters that shape its structure (`n_arms`, `arm_pitch_deg`,
`arm_strength`, `arm_sharpness`, `arm_phase_deg`, `arm_start_h`, `barred`, `irregularity`, `structure_seed`), from
their own random stream; pass `dict(hubble_types=False)` to leave them out. Structure is drawn on discs with
`re_disc_arcsec` of at least `IMAGE_CONFIG["structure_min_re_arcsec"]` (0.4").

## Generate images

```bash
# every catalogue, the coadds needed for training
python scripts/generate_mock_images.py --catalogue-dir ~/mocks/catalogues --out-dir ~/mocks/images

# only these catalogues
python scripts/generate_mock_images.py --catalogue-dir ~/mocks/catalogues --out-dir ~/mocks/images --catalogue train test

# save every coadd of every catalogue
python scripts/generate_mock_images.py --catalogue-dir ~/mocks/catalogues --out-dir ~/mocks/images --all-images

# one coadd per catalogue: 10 years, nominal PSF
python scripts/generate_mock_images.py --catalogue-dir ~/mocks/catalogues --out-dir ~/mocks/images --epoch 10y

# alternative to epoch flag. this example generates one coadd with 50 r-band visits, r FWHM 1.3". Other bands scaled proportionally to 10 yr Nexp ratios and expected FWHMs.
python scripts/generate_mock_images.py --catalogue-dir ~/mocks/catalogues --out-dir ~/mocks/images --n-exp 50 --psf-fwhm 1.3
```

For a single coadd, give either `--epoch` (`1m` to `11m` or `1y` to `10y`, at the nominal PSF in every band) or both
`--n-exp` and `--psf-fwhm` (r-band visits and FWHM in arcsec; the other bands scale with LSST's 10-year visit plan and
the nominal PSF ratios).

Each catalogue is rendered once with GalSim (bulge + disc Sersic profiles, Gaussian clumps and tidal blobs) at a
narrow 0.45″ PSF. Coadds are then made from that render for a grid of 7 survey depths (1 month, 6 months, 1, 3, 5, 8
and 10 years of visits) × 6 seeings (r-band PSF FWHM 0.7″ to 2.0″), by broadening it to the target PSF and adding sky
and source noise for that many visits. Depth and seeing vary independently, so a detector cannot learn to link them.

By default every coadd is listed in each catalogue's manifest and saved, except for `train` and `valid`: the training
code rebuilds those on the fly from the saved base render and manifest, identically every time, so they take no disk
space. At the full 5000 × 5000 pixel size each saved coadd (signal + variance) is 1.2 GB, so for the four standard
catalogues the default run needs about 100 GB, `--all-images` about 200 GB and a single coadd about 7 GB. Saving this partial dataset with a small psf allows for the data to be degraded on the fly. These degraded images can then removed from memory after they have been processed.

Rendering is the slow step; it runs in parallel over `--n-workers` processes (default: up to 8). Outputs go to
`<out-dir>/<catalogue>/`:

| File | Contents |
|---|---|
| `base_clean_signal.npy` | Noise-free render at the 0.45″ base PSF, float32 (band, y, x) in nJy per pixel |
| `base_meta.json` | Bands, canvas origin in frame pixels, shape, pixel scale, zeropoint, base PSF |
| `coadd_manifest.json` | Every coadd: visits and PSF FWHM per band, noise seed, and whether it was saved |
| `<catalogue>_<coadd>_signal.npy` | A saved coadd (nJy per pixel), e.g. `test_1y_fwhm110_signal.npy`, `test_10y_nominal_signal.npy` or `test_nexp50_fwhm130_signal.npy` |
| `<catalogue>_<coadd>_variance.npy` | Its per-pixel variance |

Bands are in `ugrizy` order. A catalogue position (`x_pix`, `y_pix`) falls in array pixel
`[:, round(y_pix) - origin_y, round(x_pix) - origin_x]`, with `origin` from `base_meta.json`.

## Example commands

Run from the repo folder with the environment active (`source .venv/bin/activate`). The folders under `~/mocks` are
just examples; use any location outside the repo.

**Catalogues**

```bash
# Default: 4 catalogues (train, valid, calib, test), 5000 x 5000 px
python scripts/generate_mock_catalogues.py --out-dir ~/mocks/catalogues

# 6 catalogues: train, valid, calib, test, extra1, extra2
python scripts/generate_mock_catalogues.py --out-dir ~/mocks/catalogues --n-catalogues 6

# Numbered instead of train/valid/...: mock_catalogue_1, _2, _3
python scripts/generate_mock_catalogues.py --out-dir ~/mocks/catalogues --n-catalogues 3 --numbered

# One small, quick catalogue (1000 x 1000 px, 'train')
python scripts/generate_mock_catalogues.py --out-dir ~/mocks/catalogues --n-catalogues 1 --npix 1000

# Use the full COSMOS2025 catalogue instead of the shipped subset
python scripts/generate_mock_catalogues.py --out-dir ~/mocks/catalogues --cosmos /path/to/cosmos2025_cat.npz
```

**Images: the full depth x seeing grid**

```bash
# Every catalogue; all 42 coadds listed, saved for all but train/valid (rebuilt on the fly in training)
python scripts/generate_mock_images.py --catalogue-dir ~/mocks/catalogues --out-dir ~/mocks/images

# Save every coadd of every catalogue (~200 GB at full size)
python scripts/generate_mock_images.py --catalogue-dir ~/mocks/catalogues --out-dir ~/mocks/images --all-images

# Only some catalogues
python scripts/generate_mock_images.py --catalogue-dir ~/mocks/catalogues --out-dir ~/mocks/images --catalogue test
python scripts/generate_mock_images.py --catalogue-dir ~/mocks/catalogues --out-dir ~/mocks/images --catalogue calib test
python scripts/generate_mock_images.py --catalogue-dir ~/mocks/catalogues --out-dir ~/mocks/images --catalogue 2
```

**Images: one coadd per catalogue**

```bash
# 10-year depth at the nominal PSF
python scripts/generate_mock_images.py --catalogue-dir ~/mocks/catalogues --out-dir ~/mocks/images --epoch 10y

# Other depths: 1m ... 11m, 1y ... 10y
python scripts/generate_mock_images.py --catalogue-dir ~/mocks/catalogues --out-dir ~/mocks/images --epoch 1y
python scripts/generate_mock_images.py --catalogue-dir ~/mocks/catalogues --out-dir ~/mocks/images --epoch 6m --catalogue test

# Custom depth and seeing: 50 r-band visits at 1.3" r-band FWHM (both flags required)
python scripts/generate_mock_images.py --catalogue-dir ~/mocks/catalogues --out-dir ~/mocks/images --n-exp 50 --psf-fwhm 1.3

# Worst-case seeing at full 10-year depth (184 r-band visits)
python scripts/generate_mock_images.py --catalogue-dir ~/mocks/catalogues --out-dir ~/mocks/images --n-exp 184 --psf-fwhm 2.0

# More parallel rendering processes (default: up to 8)
python scripts/generate_mock_images.py --catalogue-dir ~/mocks/catalogues --out-dir ~/mocks/images --n-workers 10
```

**Quick end-to-end test**

```bash
python scripts/generate_mock_catalogues.py --out-dir ~/mocks/test/catalogues --n-catalogues 1 --npix 1000
python scripts/generate_mock_images.py --catalogue-dir ~/mocks/test/catalogues --out-dir ~/mocks/test/images --epoch 10y
```

**Help and tests**

```bash
python scripts/generate_mock_catalogues.py --help
python scripts/generate_mock_images.py --help
pytest -q
```

## Python API

```python
from mock_lsst_image_generation import build_mock_catalogue, load_cosmos2025_catalogue, save_mock_catalogue

cosmos = load_cosmos2025_catalogue()  # shipped subset; or pass a path
cat = build_mock_catalogue(cosmos, dict(seed=7, npix=2000))  # any CONFIG key can be overridden
cat.galaxies, cat.clumps, cat.tidal_blobs, cat.tidal_pairs
save_mock_catalogue(cat, "my_mock.csv")

from mock_lsst_image_generation import generate_catalogue_images
generate_catalogue_images(cat, "test", "my_images", mode="single", epoch="10y", n_workers=8)  # or "training" / "all"
```

## Tests

```bash
pip install pytest
pytest -q
```

The tests (about a minute) build small catalogues from the shipped subset and check they are sane, reproducible, and
unchanged from the reference realisation pinned in `tests/test_catalogue.py`. `tests/test_images.py` renders a small
catalogue and checks the image keeps the catalogue's flux, the coadd noise matches its variance, and each image mode
and single-coadd option writes the right files. GitHub Actions runs them on every push
(`.github/workflows/tests.yml`). If you change the model on purpose, the reference test will fail: check the new
catalogue, then update `REFERENCE`.

## Licence and data credit

The code is released under the MIT licence (see `LICENSE`). `data/cosmos2025_subset.npz` is a column/row subset of
the public COSMOS2025 catalogue (COSMOS-Web collaboration); please cite the COSMOS2025 catalogue paper if you use it.
