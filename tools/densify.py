#!/usr/bin/env python
"""Densify a sparse reconstruction with plane-sweep multi-view stereo.

    python tools/reconstruct.py --images photos/ --out output/scene     # writes reconstruction.npz
    python tools/densify.py --recon output/scene/reconstruction.npz --out output/scene/dense.ply

Image paths are read from the reconstruction (``--images`` overrides the folder).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from structura.dense import MVSConfig, densify  # noqa: E402
from structura.io import load_reconstruction, write_ply  # noqa: E402
from structura.pointcloud import remove_statistical_outliers, voxel_downsample  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recon", required=True)
    p.add_argument("--images", default=None, help="folder with the original images (default: paths in the npz)")
    p.add_argument("--out", default="dense.ply")
    p.add_argument("--depths", type=int, default=96, help="number of depth planes")
    p.add_argument("--downscale", type=int, default=2)
    p.add_argument("--neighbors", type=int, default=4)
    p.add_argument("--min-ncc", type=float, default=0.5)
    p.add_argument("--min-consistent", type=int, default=1)
    p.add_argument("--voxel", type=float, default=0.0, help="voxel size for downsampling (0 = off)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")

    rec, names = load_reconstruction(args.recon)
    images = {}
    for i in rec.cameras:
        path = Path(names.get(i, ""))
        if args.images:
            path = Path(args.images) / path.name
        img = cv2.imread(str(path)) if path.name else None
        if img is None:
            logging.warning("image %d (%s) not found, skipped", i, path)
            continue
        k = rec.cameras[i].intrinsics
        if (img.shape[1], img.shape[0]) != (k.width, k.height):  # reconstruct.py may have downscaled
            img = cv2.resize(img, (k.width, k.height), interpolation=cv2.INTER_AREA)
        images[i] = img
    cfg = MVSConfig(num_depths=args.depths, downscale=args.downscale, num_neighbors=args.neighbors,
                    min_ncc=args.min_ncc, min_consistent=args.min_consistent)  # fmt: skip
    tic = time.time()
    pts, cols, maps = densify(images, rec.cameras, rec.points, cfg)
    if len(pts) > 50:
        keep = remove_statistical_outliers(pts)
        pts, cols = pts[keep], cols[keep]
    if args.voxel > 0 and len(pts):
        pts, cols = voxel_downsample(pts, args.voxel, cols)
    write_ply(args.out, pts, cols)
    logging.info("%d depth maps, %d dense points (%.1fs) -> %s", len(maps), len(pts), time.time() - tic, args.out)


if __name__ == "__main__":
    main()
