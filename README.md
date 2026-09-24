# mock_lsst_image_generation

Mock LSST-like galaxy catalogues (and, next, GalSim images) for training and testing source detectors.
Every galaxy is a known truth entry, so any detector's completeness and purity can be measured exactly.

## What the catalogue models

| Ingredient | Model | Module |
|---|---|---|
| Positions and environment | A coherent light-cone cosmic web (nodes, filaments, cluster cores and outskirts, field). Its sites are thinned by the evolving GSMF number density, which gives the mock realistic clustering and a `web_env` label per galaxy | `cosmic_web_generation`, `environment_sampling` |
| Type and stellar mass | Evolving star-forming and quiescent GSMFs (continuous fits to COSMOS2020) | `gsmf_sampling` |
| sSFR | Empirical low-z COSMOS2025 p(sSFR \| M) per type, evolved along the main sequence | `ssfr_sampling` |
| SED and size | Rest-frame SED and size cloned from quality-cut COSMOS2025 donors, then projected to LSST ugrizy | `donor_selection`, `rest_frame_sed_sampling`, `photometry` |
| Structure | Bulge + disc: B/T from a Dimauro et al. (2022)-derived fit, per-band light fractions, coupled ellipticities | `bulge_disc_decomposition` |
| Star-forming clumps | Blue Gaussian knots in resolved star-forming discs | `sf_clump_generation` |
| Tidal features | Curved bridges of Gaussian blobs between interacting pairs | `tidal_stream_generation` |

`catalogue_pipeline.build_mock_catalogue` ties these together. All settings live in `config.py`
(`CONFIG`, `PHYS`); pass a partial dict to override any of them.

## Repository layout

| Path | Contents |
|---|---|
| `mock_lsst_image_generation/` | The package: one module per model ingredient (table above), `config.py` for all settings |
| `scripts/generate_mock_catalogues.py` | Command-line entry point |
| `data/cosmos2025_subset.npz` | Shipped COSMOS2025 input (see below) |
| `tests/` | pytest suite, run by GitHub Actions on every push |

## Install

Create a virtual environment inside the repo and install the pinned dependencies plus the package itself:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the exact numpy/pandas/scipy versions the pipeline was verified with, so catalogues are
reproducible. To use your own versions instead, run `pip install -e .`. Point your editor's Python interpreter at
`.venv/bin/python` so it can resolve the imports.

The repo ships `data/cosmos2025_subset.npz` (17 MB), used by default. It holds only the columns the pipeline reads and
the 87k rows (of 784k) that can pass its COSMOS cuts, and gives byte-identical catalogues to the full
`cosmos2025_cat.npz` (~1.5 GB) at the default settings. If you loosen the COSMOS training/sSFR-PDF cuts in `CONFIG`,
pass the full catalogue instead.

## Generate catalogues

```bash
python scripts/generate_mock_catalogues.py --out-dir catalogues   # add --cosmos /path/to/cosmos2025_cat.npz for the full table
```

This builds four catalogues by default: `train`, `valid`, `calib` and `test`. Pass `--n-catalogues N` to build any
number; catalogues beyond the fourth are named `extra1`, `extra2`, and so on. Each catalogue is an independent
realisation of a 5000 × 5000 pixel frame (0.2″ pixels), and only the seed differs. Catalogue k always gets the same
name and seed, so asking for more never changes the ones you already have.

Each catalogue writes four CSVs:

| File | Contents |
|---|---|
| `forward_mock_restframe_empirical_clustered_<name>.csv` | Galaxies: position, redshift, environment, physical properties, and bulge/disc/total photometry and structure |
| `…_clumps.csv` | Star-forming clumps: parent id, offsets, width, per-band flux |
| `…_tidal.csv` | Tidal-bridge blobs: one row per blob and band |
| `…_tidal_pairs.csv` | The interacting pairs |

Fluxes are in nJy (AB zeropoint 31.4). Sizes are half-light radii in arcsec. `x_pix` and `y_pix` are frame pixel
coordinates.

## Python API

```python
from mock_lsst_image_generation import build_mock_catalogue, load_cosmos2025_catalogue, save_mock_catalogue

cosmos = load_cosmos2025_catalogue()  # shipped subset; or pass a path
cat = build_mock_catalogue(cosmos, dict(seed=7, npix=2000))  # any CONFIG key can be overridden
cat.galaxies, cat.clumps, cat.tidal_blobs, cat.tidal_pairs
save_mock_catalogue(cat, "my_mock.csv")
```

## Tests

```bash
pip install pytest
pytest -q
```

The tests (about 15 s) build small catalogues from the shipped subset and check they are sane, reproducible, and
unchanged from the reference realisation pinned in `tests/test_catalogue.py`. GitHub Actions runs them on every push
(`.github/workflows/tests.yml`). If you change the model on purpose, the reference test will fail: check the new
catalogue, then update `REFERENCE`.

## Licence and data credit

The code is released under the MIT licence (see `LICENSE`). `data/cosmos2025_subset.npz` is a column/row subset of
the public COSMOS2025 catalogue (COSMOS-Web collaboration); please cite the COSMOS2025 catalogue paper if you use it.
