# mock_lsst_image_generation

Mock LSST-like galaxy catalogues and images for training and testing source detection and deblending, built from
physical models and COSMOS2025 data. Every galaxy is a known truth entry, so any detector's completeness and purity can
be measured exactly.

Two commands: one builds the catalogues, the other renders them into images.

```bash
python scripts/generate_mock_catalogues.py   # -> catalogues/
python scripts/generate_mock_images.py       # -> images/
```

## What the catalogue models

- **Positions:** galaxies placed on a cosmic web (clusters, filaments, field) along a light cone
- **Type and stellar mass:** evolving star-forming and quiescent stellar mass functions
- **Star formation, colours and sizes:** drawn from similar real galaxies in COSMOS2025 and shifted to each mock
  galaxy's redshift, giving LSST ugrizy photometry
- **Structure:** bulge + disc
- **Extras:** star-forming clumps and tidal bridges between interacting pairs

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
python scripts/generate_mock_catalogues.py   # add --cosmos /path/to/cosmos2025_cat.npz for the full table
```

This builds four catalogues by default: `train`, `valid`, `calib` and `test`. Pass `--n-catalogues N` to build any
number; catalogues beyond the fourth are named `extra1`, `extra2`, and so on. Each catalogue is an independent
realisation of a 5000 × 5000 pixel frame (0.2″ pixels), and only the seed differs. Catalogue k always gets the same
name and seed, so asking for more never changes the ones you already have.

Each catalogue writes four CSVs to `catalogues/` (change with `--out-dir`):

| File | Contents |
|---|---|
| `mock_catalogue_<name>.csv` | Galaxies: position, redshift, environment, physical properties, and bulge/disc/total photometry and structure |
| `…_clumps.csv` | Star-forming clumps: parent id, offsets, width, per-band flux |
| `…_tidal.csv` | Tidal-bridge blobs: one row per blob and band |
| `…_tidal_pairs.csv` | The interacting pairs |

Fluxes are in nJy (AB zeropoint 31.4). Sizes are half-light radii in arcsec. `x_pix` and `y_pix` are frame pixel
coordinates.

## Generate images

```bash
python scripts/generate_mock_images.py                         # every catalogue, the coadds needed for training
python scripts/generate_mock_images.py --catalogue train test  # only these catalogues
python scripts/generate_mock_images.py --all-images            # save every coadd of every catalogue
python scripts/generate_mock_images.py --ten-year-only         # just the nominal 10-year coadd of each
```

Each catalogue is rendered once with GalSim (bulge + disc Sersic profiles, Gaussian clumps and tidal blobs) at a
narrow 0.45″ PSF. Coadds are then made from that render for a grid of 7 survey depths (1 month, 6 months, 1, 3, 5, 8
and 10 years of visits) × 6 seeings (r-band PSF FWHM 0.7″ to 2.0″), by broadening it to the target PSF and adding sky
and source noise for that many visits. Depth and seeing vary independently, so a detector cannot learn to link them.

By default every coadd is listed in each catalogue's manifest, but only the `calib` and `test` coadds are saved. The
training code rebuilds `train`/`valid` coadds on the fly from the saved base render and manifest, identically every
time, so they take no disk space. At the full 5000 × 5000 pixel size each saved coadd (signal + variance) is 1.2 GB,
so the default run needs about 100 GB, `--all-images` about 200 GB and `--ten-year-only` about 7 GB.

Rendering is the slow step; it runs in parallel over `--n-workers` processes (default: up to 8). Outputs go to
`images/<catalogue>/` (change with `--out-dir`; read catalogues from elsewhere with `--catalogue-dir`):

| File | Contents |
|---|---|
| `base_clean_signal.npy` | Noise-free render at the 0.45″ base PSF, float32 (band, y, x) in nJy per pixel |
| `base_meta.json` | Bands, canvas origin in frame pixels, shape, pixel scale, zeropoint, base PSF |
| `coadd_manifest.json` | Every coadd: visits and PSF FWHM per band, noise seed, and whether it was saved |
| `<catalogue>_<coadd>_signal.npy` | A saved coadd, e.g. `test_1y_fwhm110_signal.npy` (nJy per pixel) |
| `<catalogue>_<coadd>_variance.npy` | Its per-pixel variance |

Bands are in `ugrizy` order. A catalogue position (`x_pix`, `y_pix`) falls in array pixel
`[:, round(y_pix) - origin_y, round(x_pix) - origin_x]`, with `origin` from `base_meta.json`.

## Python API

```python
from mock_lsst_image_generation import build_mock_catalogue, load_cosmos2025_catalogue, save_mock_catalogue

cosmos = load_cosmos2025_catalogue()  # shipped subset; or pass a path
cat = build_mock_catalogue(cosmos, dict(seed=7, npix=2000))  # any CONFIG key can be overridden
cat.galaxies, cat.clumps, cat.tidal_blobs, cat.tidal_pairs
save_mock_catalogue(cat, "my_mock.csv")

from mock_lsst_image_generation import generate_catalogue_images
generate_catalogue_images(cat, "test", "my_images", mode="ten_year", n_workers=8)  # or "training" / "all"
```

## Tests

```bash
pip install pytest
pytest -q
```

The tests (about a minute) build small catalogues from the shipped subset and check they are sane, reproducible, and
unchanged from the reference realisation pinned in `tests/test_catalogue.py`. `tests/test_images.py` renders a small
catalogue and checks the image keeps the catalogue's flux, the coadd noise matches its variance, and each image mode
writes the right files. GitHub Actions runs them on every push
(`.github/workflows/tests.yml`). If you change the model on purpose, the reference test will fail: check the new
catalogue, then update `REFERENCE`.

## Licence and data credit

The code is released under the MIT licence (see `LICENSE`). `data/cosmos2025_subset.npz` is a column/row subset of
the public COSMOS2025 catalogue (COSMOS-Web collaboration); please cite the COSMOS2025 catalogue paper if you use it.
