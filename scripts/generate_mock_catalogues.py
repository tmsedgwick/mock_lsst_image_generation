"""Generate independent mock catalogues from COSMOS2025 (default 4: train, valid, calib, test).

Each catalogue is an independent realisation of the same forward model; only the seed differs. The first four are
named train, valid, calib and test, any further ones extra1, extra2, ..., and each is written as
<out-dir>/forward_mock_restframe_empirical_clustered_<name>{,_clumps,_tidal,_tidal_pairs}.csv

    python scripts/generate_mock_catalogues.py --out-dir catalogues --n-catalogues 6

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
    parser.add_argument("--out-dir", type=Path, default=Path("."), help="output directory (default: current)")
    parser.add_argument("--n-catalogues", type=int, default=N_CATALOGUES,
                        help="number of catalogues to generate (default: %(default)s)")
    parser.add_argument("--npix", type=int, default=CONFIG["npix"], help="frame side in pixels (default: %(default)s)")
    args = parser.parse_args()
    cosmos = load_cosmos2025_catalogue(args.cosmos)
    generate_mock_catalogues(cosmos, args.n_catalogues, args.out_dir, cfg=dict(npix=args.npix))


if __name__ == "__main__":
    main()
