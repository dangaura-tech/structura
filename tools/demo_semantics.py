#!/usr/bin/env python
"""Synthetic demo of language-driven 3-D queries (no pretrained weights needed).

Reconstructs the synthetic scene (sphere, box, torus, floor), renders photo-like views,
fuses dense features from the *mock colour encoder* into the points, and highlights the
query result in a turntable GIF. With real photos and `--encoder openclip`, the same code
runs through `tools/query3d.py`.

    python tools/demo_semantics.py --query "the red sphere" --gif assets/query3d_demo.gif
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from structura import IncrementalSfM  # noqa: E402
from structura.render import render_frame, save_gif, turntable  # noqa: E402
from structura.semantics import MockColorEncoder, fuse_point_features, relevance  # noqa: E402
from structura.semantics.fusion import heat_colors  # noqa: E402
from structura.synthetic import make_scene  # noqa: E402


def render_views(scene, size):
    return {i: np.asarray(render_frame(scene.points, scene.colors, cam, size, point_radius=3.0))[:, :, ::-1].copy()
            for i, cam in scene.cameras.items()}  # fmt: skip


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--query", default="the red sphere")
    p.add_argument("--cameras", type=int, default=12)
    p.add_argument("--points-per-object", type=int, default=1500)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--gif", default="")
    args = p.parse_args()

    scene = make_scene(args.cameras, args.points_per_object, seed=1)
    rec = IncrementalSfM(scene.intrinsics, scene.keypoints, scene.matches, colors=scene.keypoint_colors).run()
    k = next(iter(scene.intrinsics.values()))
    images = render_views(scene, (k.width, k.height))
    encoder = MockColorEncoder()
    feats, views = fuse_point_features(rec.points, rec.cameras, images, encoder, tracks=rec.point_tracks)
    rel = relevance(feats, encoder, args.query)
    hits = rel > args.threshold
    print(f"{rec.num_points} points, {int((views > 0).sum())} with features, {int(hits.sum())} match '{args.query}'")
    if args.gif:
        colors = heat_colors(rel, rec.colors, args.threshold)
        Path(args.gif).parent.mkdir(parents=True, exist_ok=True)
        save_gif(turntable(rec.points, colors, rec.cameras, frames=60, size=(640, 420)), args.gif)
        print(f"wrote {args.gif}")


if __name__ == "__main__":
    main()
