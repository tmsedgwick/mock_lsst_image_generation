"""Render mock catalogues into LSST-like coadd images (run generate_mock_catalogues.py first).

Each catalogue is rendered once with GalSim; coadds at every survey depth (1 month to 10 years) and seeing (r-band
FWHM 0.7" to 2.0") are then made from that render. By default every catalogue in --catalogue-dir is processed, all
coadds are listed in each manifest, and only the calib and test coadds are saved to disk: training code rebuilds the
train/valid ones on the fly from the saved base render, so they take no disk space.

    python scripts/generate_mock_images.py                         # all catalogues, coadds needed for training
    python scripts/generate_mock_images.py --catalogue train test  # only these catalogues
    python scripts/generate_mock_images.py --all-images            # save every coadd of every catalogue
    python scripts/generate_mock_images.py --ten-year-only         # just the nominal 10-year coadd of each

Outputs go to <out-dir>/<catalogue>/ (see mock_lsst_image_generation/image_pipeline.py for the file list).
"""

import argparse
import os
from pathlib import Path

from mock_lsst_image_generation import generate_mock_images


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalogue-dir", type=Path, default=Path("catalogues"),
                        help="where generate_mock_catalogues.py wrote the catalogues (default: %(default)s)")
    parser.add_argument("--out-dir", type=Path, default=Path("images"), help="output directory (default: %(default)s)")
    parser.add_argument("--catalogue", nargs="+", metavar="NAME",
                        help="only these catalogues, e.g. train or extra1 (default: all in --catalogue-dir)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--all-images", action="store_true", help="save every coadd of every catalogue")
    mode.add_argument("--ten-year-only", action="store_true", help="only make the nominal 10-year coadd")
    parser.add_argument("--n-workers", type=int, default=min(8, os.cpu_count() or 1),
                        help="parallel rendering processes (default: %(default)s)")
    args = parser.parse_args()
    mode = "all" if args.all_images else "ten_year" if args.ten_year_only else "training"
    generate_mock_images(args.catalogue_dir, args.out_dir, args.catalogue, mode, n_workers=args.n_workers)


if __name__ == "__main__":
    main()
