#!/usr/bin/env python
"""End-to-end demo on a synthetic scene with exact ground truth (no data needed).

    python tools/demo_synthetic.py --cameras 12 --outlier-ratio 0.1 --noise 0.5 --gif assets/sfm_turntable.gif

Prints registration and accuracy statistics after aligning the reconstruction to the ground
truth with a similarity transform (SfM is only defined up to scale and rigid motion).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from structura import IncrementalSfM  # noqa: E402
from structura.geometry import umeyama  # noqa: E402
from structura.pointcloud import chamfer_distance  # noqa: E402
from structura.synthetic import make_scene  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cameras", type=int, default=12)
    p.add_argument("--points-per-object", type=int, default=500)
    p.add_argument("--noise", type=float, default=0.5, help="keypoint noise (px)")
    p.add_argument("--outlier-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--gif", default="", help="write a turntable GIF of the reconstruction")
    p.add_argument("--ply", default="", help="write the reconstruction as PLY")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")

    scene = make_scene(args.cameras, args.points_per_object, noise_px=args.noise,
                       outlier_ratio=args.outlier_ratio, seed=args.seed)  # fmt: skip
    tic = time.time()
    rec = IncrementalSfM(scene.intrinsics, scene.keypoints, scene.matches, colors=scene.keypoint_colors).run()
    elapsed = time.time() - tic

    ids = sorted(rec.cameras)
    est = np.array([rec.cameras[i].center for i in ids])
    gt = np.array([scene.cameras[i].center for i in ids])
    s, R, t = umeyama(est, gt)
    aligned_c = est @ (s * R).T + t
    aligned_p = rec.points @ (s * R).T + t
    rot_err = [
        np.degrees(np.arccos(np.clip((np.trace(scene.cameras[i].R.T @ rec.cameras[i].R @ R.T) - 1) / 2, -1, 1)))
        for i in ids
    ]
    print(f"registered images      : {len(ids)} / {args.cameras}")
    print(f"3-D points             : {rec.num_points}")
    print(f"mean reprojection error: {rec.mean_reprojection_error(scene.keypoints):.3f} px")
    print(f"camera position error  : {np.linalg.norm(aligned_c - gt, axis=1).mean():.4f} (scene units)")
    print(f"camera rotation error  : {np.mean(rot_err):.3f} deg")
    print(f"chamfer distance to GT : {chamfer_distance(aligned_p, scene.points):.4f}")
    print(f"time                   : {elapsed:.1f} s")
    if args.ply:
        from structura.io import write_ply

        write_ply(args.ply, rec.points, rec.colors)
    if args.gif:
        from structura.render import save_gif, turntable

        Path(args.gif).parent.mkdir(parents=True, exist_ok=True)
        save_gif(turntable(rec.points, rec.colors, rec.cameras, frames=60, size=(640, 420)), args.gif)
        print(f"turntable written to {args.gif}")


if __name__ == "__main__":
    main()
