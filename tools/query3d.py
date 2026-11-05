#!/usr/bin/env python
"""Query a reconstruction with natural language using dense CLIP features.

    python tools/reconstruct.py --images photos/ --out output/scene        # writes reconstruction.npz
    python tools/query3d.py --recon output/scene/reconstruction.npz --query "a red chair" --gif
    python tools/query3d.py --recon ... --labels "chair" "table" "floor" "wall"   # open-vocabulary segmentation

Requires ``pip install -e ".[clip]"``; the first run downloads the OpenCLIP weights.
Outputs ``<out>/query_<slug>.ply`` (points coloured by relevance), optional turntable GIF, and
``segmentation.ply`` for ``--labels``.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from structura.io import load_reconstruction, write_ply  # noqa: E402
from structura.semantics import fuse_point_features, relevance, segment  # noqa: E402
from structura.semantics.fusion import heat_colors  # noqa: E402

LABEL_COLORS = np.array(
    [[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48], [145, 30, 180],
     [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 212]], np.uint8,
)  # fmt: skip


def build_encoder(args):
    if args.encoder == "mock":
        from structura.semantics import MockColorEncoder

        return MockColorEncoder()
    from structura.semantics import OpenCLIPEncoder

    return OpenCLIPEncoder(args.model, args.pretrained, args.device, args.short_side)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recon", required=True, help="reconstruction.npz written by tools/reconstruct.py")
    p.add_argument("--images", default=None, help="override the image folder stored in the reconstruction")
    q = p.add_mutually_exclusive_group(required=True)
    q.add_argument("--query", nargs="+", help="one or more text queries")
    q.add_argument("--labels", nargs="+", help="labels for open-vocabulary segmentation")
    p.add_argument("--encoder", choices=["openclip", "mock"], default="openclip")
    p.add_argument("--model", default="ViT-B-16")
    p.add_argument("--pretrained", default="laion2b_s34b_b88k")
    p.add_argument("--device", default="cpu")
    p.add_argument("--short-side", type=int, default=448)
    p.add_argument("--threshold", type=float, default=0.55, help="relevance above which points are highlighted")
    p.add_argument("--out", default=None, help="output folder (default: next to the reconstruction)")
    p.add_argument("--gif", action="store_true")
    args = p.parse_args()

    rec, names = load_reconstruction(args.recon)
    images = {}
    for i in rec.cameras:
        path = Path(args.images) / Path(names[i]).name if args.images else Path(names.get(i, ""))
        img = cv2.imread(str(path))
        if img is None:
            raise SystemExit(f"cannot read image for camera {i}: {path}")
        k = rec.cameras[i].intrinsics
        if (img.shape[1], img.shape[0]) != (k.width, k.height):  # reconstruct.py may have downscaled
            img = cv2.resize(img, (k.width, k.height), interpolation=cv2.INTER_AREA)
        images[i] = img
    encoder = build_encoder(args)
    feats, views = fuse_point_features(rec.points, rec.cameras, images, encoder)
    print(f"fused features for {int((views > 0).sum())} / {rec.num_points} points over {len(images)} views")

    out = Path(args.out) if args.out else Path(args.recon).parent
    out.mkdir(parents=True, exist_ok=True)
    if args.labels:
        lab = segment(feats, encoder, args.labels)
        colors = np.where((lab >= 0)[:, None], LABEL_COLORS[lab % len(LABEL_COLORS)], 40).astype(np.uint8)
        write_ply(str(out / "segmentation.ply"), rec.points, colors)
        for k, name in enumerate(args.labels):
            print(f"{name:>20s}: {(lab == k).sum()} points")
        return
    from structura.render import save_gif, turntable

    for text in args.query:
        rel = relevance(feats, encoder, text)
        colors = heat_colors(rel, rec.colors, args.threshold)
        slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
        write_ply(str(out / f"query_{slug}.ply"), rec.points, colors)
        print(f"'{text}': {(rel > args.threshold).sum()} points above {args.threshold} -> query_{slug}.ply")
        if args.gif:
            save_gif(turntable(rec.points, colors, rec.cameras), str(out / f"query_{slug}.gif"))


if __name__ == "__main__":
    main()
