#!/usr/bin/env python
"""Image-based end-to-end demo: render photo-like views of a textured scene, then run the
full pipeline from pixels (SIFT -> matching -> verification -> SfM -> PLY/GIF).

    python tools/demo_images.py --views 10 --out output/demo_images --assets assets
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from structura.synthetic import render_textured_views  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--views", type=int, default=10)
    p.add_argument("--out", default="output/demo_images")
    p.add_argument("--assets", default="", help="also write README assets (input strip + turntable GIF) here")
    args = p.parse_args()

    out = Path(args.out)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    images, _ = render_textured_views(args.views)
    for i, im in enumerate(images):
        cv2.imwrite(str(img_dir / f"view_{i:02d}.png"), im)
    print(f"rendered {len(images)} views to {img_dir}")

    script = Path(__file__).with_name("reconstruct.py")
    subprocess.run(
        [sys.executable, str(script), "--images", str(img_dir), "--out", str(out / "reconstruction"),
         "--focal", "0.9", "--gif"],
        check=True,
    )  # fmt: skip

    if args.assets:
        a = Path(args.assets)
        a.mkdir(parents=True, exist_ok=True)
        picks = np.linspace(0, len(images) - 1, 4).astype(int)
        strip = np.hstack([cv2.resize(images[k], (320, 240), interpolation=cv2.INTER_AREA) for k in picks])
        cv2.imwrite(str(a / "input_views.jpg"), strip, [cv2.IMWRITE_JPEG_QUALITY, 88])
        (a / "image_sfm_turntable.gif").write_bytes((out / "reconstruction" / "turntable.gif").read_bytes())
        print(f"assets written to {a}")


if __name__ == "__main__":
    main()
