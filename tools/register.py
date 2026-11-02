#!/usr/bin/env python
"""Rigidly align two point clouds (PLY) with ICP and write the transformed source.

python tools/register.py source.ply target.ply --out aligned.ply --voxel 0.02 --point-to-plane
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from structura.io import read_ply, write_ply  # noqa: E402
from structura.pointcloud import icp, voxel_downsample  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source")
    p.add_argument("target")
    p.add_argument("--out", default="aligned.ply")
    p.add_argument("--voxel", type=float, default=0.0, help="downsample both clouds before ICP")
    p.add_argument("--max-distance", type=float, default=0.5)
    p.add_argument("--point-to-plane", action="store_true")
    args = p.parse_args()

    src, src_col = read_ply(args.source)
    dst, _ = read_ply(args.target)
    s_ds = voxel_downsample(src, args.voxel) if args.voxel else src
    d_ds = voxel_downsample(dst, args.voxel) if args.voxel else dst
    res = icp(s_ds, d_ds, args.max_distance, point_to_plane=args.point_to_plane)
    write_ply(args.out, src @ res.R.T + res.t, src_col)
    np.set_printoptions(precision=5, suppress=True)
    print(f"ICP: {res.iterations} iterations, rmse {res.rmse:.5f}, fitness {res.fitness:.3f}")
    print("T =\n", res.T)


if __name__ == "__main__":
    main()
