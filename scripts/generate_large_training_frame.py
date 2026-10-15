"""Generate one large training frame (e.g. 14000 x 14000 px, 0.6 deg^2) as n x n independent parts stitched together.

A large frame holds hundreds of each rare class (UDGs, extended dIrrs, almost-dark galaxies, BCGs) at their default,
uncrowded rates, which a 5000 x 5000 px frame cannot. Rendering one large frame at once needs the whole image and its
component images in memory (~19 GB at 14000 px); here each part is built and rendered on its own (--part-npix, default
7000) and the parts are then stitched on disk into one split:

    python scripts/generate_large_training_frame.py --out-dir ~/mocks_large --parts-per-side 2 --part-npix 7000

writes <out-dir>/catalogues/mock_catalogue_train*.csv and <out-dir>/images/train/ (base image, component images,
base_meta.json, coadd_manifest.json; coadds are rebuilt on the fly in training, as for the usual train split), with the
parts kept under <out-dir>/parts/. A finished part is not rebuilt if the script is run again. Each 7000 px part takes
about an hour on a laptop; parts are independent, so they can be built in parallel with --parts, e.g. in two shells

    python scripts/generate_large_training_frame.py --out-dir ~/mocks_large --parts 0 1
    python scripts/generate_large_training_frame.py --out-dir ~/mocks_large --parts 2 3

and the frame is stitched by whichever run finds all parts finished (or by a final run without --parts).

Part (i, j) is shifted by (part_npix i, part_npix j) px and its ids (galaxies, stars, star-forming region parents, tidal
donors and pairs) are offset so they stay unique. Galaxies within a few px of a seam lose the light that fell outside
their own part's canvas, and large-scale structure does not continue across seams.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mock_lsst_image_generation import load_cosmos2025_catalogue
from mock_lsst_image_generation.catalogue_pipeline import build_mock_catalogue, save_mock_catalogue
from mock_lsst_image_generation.cosmos_catalogue import DEFAULT_COSMOS_PATH
from mock_lsst_image_generation.image_pipeline import generate_catalogue_images

IMAGES = ("base_clean_signal.npy", "base_sfregions_signal.npy", "base_tidal_signal.npy", "base_spikes_signal.npy")
SUFFIXES = ("", "_sfregions", "_tidal", "_tidal_pairs")


def part_done(out_dir, k):
    return (out_dir / "parts" / f"part{k}" / "images" / "train" / "coadd_manifest.json").exists()


def build_parts(out_dir, parts, part_npix, first_seed, cosmos_path, n_workers):
    cosmos = load_cosmos2025_catalogue(cosmos_path)
    for k in parts:
        part = out_dir / "parts" / f"part{k}"
        if part_done(out_dir, k):
            print(f"part {k} already built", flush=True)
            continue
        print(f"\n===== part {k}: seed {first_seed + k}, {part_npix} x {part_npix} px =====", flush=True)
        catalogue = build_mock_catalogue(cosmos, dict(npix=part_npix, seed=first_seed + k))
        (part / "catalogues").mkdir(parents=True, exist_ok=True)
        save_mock_catalogue(catalogue, part / "catalogues" / "mock_catalogue_train.csv")
        generate_catalogue_images(catalogue, "train", part / "images", mode="training", n_workers=n_workers)


def read_table(path):
    try:
        return pd.read_csv(path, low_memory=False) if path.exists() and path.stat().st_size else None
    except pd.errors.EmptyDataError:
        return None


def stitch(out_dir, n_side, part_npix):
    parts = [out_dir / "parts" / f"part{k}" for k in range(n_side * n_side)]
    metas = [json.loads((part / "images" / "train" / "base_meta.json").read_text()) for part in parts]
    images = out_dir / "images" / "train"
    images.mkdir(parents=True, exist_ok=True)
    shape = (len(metas[0]["bands"]), n_side * part_npix, n_side * part_npix)
    for name in IMAGES:
        frame = np.lib.format.open_memmap(images / name, mode="w+", dtype=np.float32, shape=shape)
        for k, (part, meta) in enumerate(zip(parts, metas)):
            image = np.load(part / "images" / "train" / name, mmap_mode="r")
            ox = meta["origin"][0] + part_npix * (k % n_side)
            oy = meta["origin"][1] + part_npix * (k // n_side)
            x0, y0 = max(ox, 0), max(oy, 0)
            x1, y1 = min(ox + image.shape[2], shape[2]), min(oy + image.shape[1], shape[1])
            frame[:, y0:y1, x0:x1] += image[:, y0 - oy:y1 - oy, x0 - ox:x1 - ox]
        frame.flush()
        del frame
    (images / "base_meta.json").write_text(json.dumps({**metas[0], "origin": [0, 0], "shape": list(shape)}, indent=2))
    manifest = json.loads((parts[0] / "images" / "train" / "coadd_manifest.json").read_text())
    (images / "coadd_manifest.json").write_text(json.dumps({**manifest, "materialised": False}, indent=2))

    tables = {suffix: [] for suffix in SUFFIXES}
    id_offset = pair_offset = 0
    for k, part in enumerate(parts):
        dx, dy = part_npix * (k % n_side), part_npix * (k // n_side)
        stem = part / "catalogues" / "mock_catalogue_train"
        galaxies = read_table(Path(f"{stem}.csv"))
        galaxies = galaxies.assign(id=galaxies["id"] + id_offset, x_pix=galaxies["x_pix"] + dx,
                                   y_pix=galaxies["y_pix"] + dy)
        if "interaction_partner_id" in galaxies:
            galaxies["interaction_partner_id"] += id_offset
        tables[""].append(galaxies)
        shifts = {"_sfregions": dict(parent_id=id_offset, x_pix_sfregion=dx, y_pix_sfregion=dy),
                  "_tidal": dict(donor_id=id_offset, gal1_id=id_offset, gal2_id=id_offset, tidal_pair_id=pair_offset,
                                 x_pix_tidal=dx, y_pix_tidal=dy),
                  "_tidal_pairs": dict(tidal_pair_id=pair_offset, gal1_id=id_offset, gal2_id=id_offset)}
        for suffix, shift in shifts.items():
            table = read_table(Path(f"{stem}{suffix}.csv"))
            if table is not None:
                tables[suffix].append(table.assign(**{c: table[c] + s for c, s in shift.items() if c in table}))
        id_offset = int(galaxies["id"].max()) + 1
        if tables["_tidal_pairs"] and "tidal_pair_id" in tables["_tidal_pairs"][-1]:
            pair_offset = int(tables["_tidal_pairs"][-1]["tidal_pair_id"].max()) + 1
    (out_dir / "catalogues").mkdir(parents=True, exist_ok=True)
    for suffix, found in tables.items():
        if found:
            pd.concat(found, ignore_index=True).to_csv(out_dir / "catalogues" / f"mock_catalogue_train{suffix}.csv",
                                                       index=False)
    galaxies = pd.concat(tables[""], ignore_index=True)
    print(f"Stitched {n_side} x {n_side} parts: {shape[2]} x {shape[1]} px, {len(galaxies):,} catalogue rows "
          f"(ids unique: {galaxies['id'].is_unique}) -> {out_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--parts-per-side", type=int, default=2, help="n: the frame is n x n parts (default 2)")
    parser.add_argument("--part-npix", type=int, default=7000, help="side of each part in pixels (default 7000)")
    parser.add_argument("--seed", type=int, default=4242, help="seed of the first part; the others follow it")
    parser.add_argument("--cosmos", type=Path, default=DEFAULT_COSMOS_PATH, help="COSMOS2025 .npz")
    parser.add_argument("--workers", type=int, default=4, help="rendering processes (default 4)")
    parser.add_argument("--parts", type=int, nargs="+",
                        help="build only these parts (0 to n^2 - 1, row by row), e.g. to build parts in parallel")
    args = parser.parse_args()
    n_parts = args.parts_per_side ** 2
    parts = args.parts if args.parts is not None else list(range(n_parts))
    if not all(0 <= k < n_parts for k in parts):
        parser.error(f"--parts must lie between 0 and {n_parts - 1}")
    build_parts(args.out_dir, parts, args.part_npix, args.seed, args.cosmos, args.workers)
    missing = [k for k in range(n_parts) if not part_done(args.out_dir, k)]
    if missing:
        print(f"Parts {missing} are not built yet: stitching waits for them", flush=True)
        return
    stitch(args.out_dir, args.parts_per_side, args.part_npix)


if __name__ == "__main__":
    main()
