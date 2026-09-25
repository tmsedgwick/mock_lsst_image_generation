"""Generate independent mock catalogues from COSMOS2025 (default 4: train, valid, calib, test).

Each catalogue is an independent realisation of the same forward model; only the seed differs. The first four are
named train, valid, calib and test, any further ones extra1, extra2, ... (or, with --numbered, all are named 1, 2, 3,
... for use outside model training), and each is written as
<out-dir>/mock_catalogue_<name>{,_clumps,_tidal,_tidal_pairs}.csv

    python scripts/generate_mock_catalogues.py --out-dir ~/mocks/catalogues
    python scripts/generate_mock_catalogues.py --out-dir ~/mocks/catalogues --n-catalogues 2 --numbered

COSMOS2025 is read from the subset shipped in data/ unless --cosmos is given.
"""

import argparse
from pathlib import Path

from mock_lsst_image_generation import CONFIG, N_CATALOGUES, generate_mock_catalogues, load_cosmos2025_catalogue
from mock_lsst_image_generation.cosmos_catalogue import DEFAULT_COSMOS_PATH


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cosmos", type=Path, default=DEFAULT_COSMOS_PATH,
                        help="COSMOS2025 catalogue (.npz) (default: %(default)s)")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="output directory; keep it outside the repo, catalogues are large")
    parser.add_argument("--n-catalogues", type=int, default=N_CATALOGUES,
                        help="number of catalogues to generate (default: %(default)s)")
    parser.add_argument("--npix", type=int, default=CONFIG["npix"], help="frame side in pixels (default: %(default)s)")
    parser.add_argument("--numbered", action="store_true",
                        help="name the catalogues 1, 2, 3, ... instead of train, valid, calib, test, extra1, ...")
    args = parser.parse_args()
    cosmos = load_cosmos2025_catalogue(args.cosmos)
    generate_mock_catalogues(cosmos, args.n_catalogues, args.out_dir, cfg=dict(npix=args.npix), numbered=args.numbered)


if __name__ == "__main__":
    main()
