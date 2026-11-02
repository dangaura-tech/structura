#!/usr/bin/env python
"""Reconstruct a sparse 3-D model from a folder of images.

    python tools/reconstruct.py --images path/to/images --out output/scene [--focal 1.2] [--gif]

Outputs: ``points.ply`` (coloured sparse cloud), ``colmap/`` (cameras/images/points3D.txt,
loadable in the COLMAP GUI) and optionally ``turntable.gif``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from structura import IncrementalSfM, Intrinsics, SfMConfig  # noqa: E402
from structura.features import detect, match_all  # noqa: E402
from structura.io import export_colmap_text, write_ply  # noqa: E402
from structura.pointcloud import remove_statistical_outliers  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", required=True)
    p.add_argument("--out", default="output/reconstruction")
    p.add_argument("--focal", type=float, default=1.2, help="focal length prior as a multiple of max(w, h)")
    p.add_argument("--max-size", type=int, default=1600, help="downscale images so the long side <= this")
    p.add_argument("--features", choices=["sift", "orb"], default="sift")
    p.add_argument("--max-features", type=int, default=8000)
    p.add_argument("--ratio", type=float, default=0.8)
    p.add_argument("--gif", action="store_true", help="render a turntable GIF of the result")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")

    files = sorted(f for f in Path(args.images).iterdir() if f.suffix.lower() in EXTS)
    if len(files) < 2:
        raise SystemExit("need at least two images")
    tic = time.time()
    feats, intr = {}, {}
    for i, f in enumerate(files):
        img = cv2.imread(str(f))
        s = min(1.0, args.max_size / max(img.shape[:2]))
        if s < 1:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        feats[i] = detect(img, args.features, args.max_features)
        intr[i] = Intrinsics.from_image_size(img.shape[1], img.shape[0], args.focal)
        logging.info("%s: %d keypoints", f.name, len(feats[i].keypoints))
    matches = match_all(feats, ratio=args.ratio)
    logging.info("matched %d image pairs", len(matches))

    kps = {i: f.keypoints for i, f in feats.items()}
    sfm = IncrementalSfM(intr, kps, matches, SfMConfig(), colors={i: f.colors for i, f in feats.items()})
    rec = sfm.run()

    out = Path(args.out)
    keep = remove_statistical_outliers(rec.points) if rec.num_points > 20 else slice(None)
    write_ply(str(out / "points.ply"), rec.points[keep], rec.colors[keep])
    export_colmap_text(rec, kps, str(out / "colmap"), names={i: f.name for i, f in enumerate(files)})
    if args.gif:
        from structura.render import save_gif, turntable

        save_gif(turntable(rec.points[keep], rec.colors[keep], rec.cameras), str(out / "turntable.gif"))
    logging.info(
        "registered %d/%d images, %d points, mean reprojection error %.2f px, %.1fs -> %s",
        len(rec.cameras), len(files), rec.num_points, rec.mean_reprojection_error(kps), time.time() - tic, out,
    )  # fmt: skip


if __name__ == "__main__":
    main()
