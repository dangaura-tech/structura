"""PLY point clouds and COLMAP-compatible text export."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from .reconstruction import Reconstruction


def write_ply(path: str, points: np.ndarray, colors: Optional[np.ndarray] = None, binary: bool = True) -> None:
    points = np.asarray(points, np.float32)
    n = len(points)
    header = ["ply", f"format {'binary_little_endian' if binary else 'ascii'} 1.0", f"element vertex {n}",
              "property float x", "property float y", "property float z"]  # fmt: skip
    if colors is not None:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    header.append("end_header")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode())
        if binary:
            fields = [("xyz", np.float32, (3,))]
            if colors is not None:
                fields.append(("rgb", np.uint8, (3,)))
            arr = np.empty(n, dtype=np.dtype(fields).newbyteorder("<"))
            arr["xyz"] = points
            if colors is not None:
                arr["rgb"] = colors
            f.write(arr.tobytes())
        else:
            for i in range(n):
                row = " ".join(f"{v:.6f}" for v in points[i])
                if colors is not None:
                    row += " " + " ".join(str(int(c)) for c in colors[i])
                f.write((row + "\n").encode())


def read_ply(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Read the vertices (and RGB if present) written by :func:`write_ply` or similar tools."""
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline().decode().strip()
            header.append(line)
            if line == "end_header":
                break
        fmt = next(h.split()[1] for h in header if h.startswith("format"))
        n = int(next(h.split()[2] for h in header if h.startswith("element vertex")))
        props = [h.split()[1:] for h in header if h.startswith("property")]
        types = {"float": "<f4", "double": "<f8", "uchar": "u1", "int": "<i4", "uint8": "u1", "float32": "<f4"}
        dtype = np.dtype([(name, types[t]) for t, name in props])
        if fmt == "ascii":
            data = np.loadtxt(f, max_rows=n, ndmin=2)
            arr = np.zeros(n, dtype)
            for k, name in enumerate(dtype.names):
                arr[name] = data[:, k]
        else:
            arr = np.frombuffer(f.read(dtype.itemsize * n), dtype=dtype, count=n)
    pts = np.c_[arr["x"], arr["y"], arr["z"]].astype(np.float64)
    cols = np.c_[arr["red"], arr["green"], arr["blue"]].astype(np.uint8) if "red" in dtype.names else None
    return pts, cols


def export_colmap_text(rec: Reconstruction, keypoints: Dict[int, np.ndarray], out_dir: str, names=None) -> None:
    """Write cameras.txt / images.txt / points3D.txt (COLMAP text model).

    Distortion-free cameras are written as ``PINHOLE``; cameras with radial distortion as
    ``OPENCV`` (fx fy cx cy k1 k2 p1 p2, tangential terms zero), which COLMAP reads directly.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ids = sorted(rec.cameras)
    obs: Dict[int, list] = {i: [] for i in ids}
    for p, track in enumerate(rec.point_tracks):
        for img, kp in track.items():
            if img in rec.cameras:
                obs[img].append((kp, p))
    with open(out / "cameras.txt", "w") as f:
        f.write("# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n")
        for i in ids:
            k = rec.cameras[i].intrinsics
            if k.has_distortion:
                f.write(f"{i + 1} OPENCV {k.width} {k.height} {k.fx} {k.fy} {k.cx} {k.cy} {k.k1} {k.k2} 0 0\n")
            else:
                f.write(f"{i + 1} PINHOLE {k.width} {k.height} {k.fx} {k.fy} {k.cx} {k.cy}\n")
    with open(out / "images.txt", "w") as f:
        f.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for i in ids:
            c = rec.cameras[i]
            x, y, z, w = Rotation.from_matrix(c.R).as_quat()
            name = names[i] if names else f"image_{i:04d}.jpg"
            f.write(f"{i + 1} {w} {x} {y} {z} {c.t[0]} {c.t[1]} {c.t[2]} {i + 1} {name}\n")
            pid = dict(obs[i])  # keypoint index -> point index; POINT2D_IDX is the keypoint index
            f.write(
                " ".join(
                    f"{xy[0]:.2f} {xy[1]:.2f} {pid[kp] + 1 if kp in pid else -1}" for kp, xy in enumerate(keypoints[i])
                )
                + "\n"
            )
    with open(out / "points3D.txt", "w") as f:
        f.write("# POINT3D_ID X Y Z R G B ERROR TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for p, (X, col, track) in enumerate(zip(rec.points, rec.colors, rec.point_tracks, strict=True)):
            tr = " ".join(f"{img + 1} {kp}" for img, kp in track.items() if img in rec.cameras)
            f.write(f"{p + 1} {X[0]} {X[1]} {X[2]} {col[0]} {col[1]} {col[2]} 0 {tr}\n")


def save_reconstruction(path: str, rec: Reconstruction, image_names: Optional[Dict[int, str]] = None) -> None:
    """Save cameras, points and colours (not tracks) to a compact ``.npz`` for later querying."""
    import json

    ids = sorted(rec.cameras)
    k = [rec.cameras[i].intrinsics for i in ids]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        points=rec.points,
        colors=rec.colors,
        camera_ids=np.array(ids, int),
        R=np.stack([rec.cameras[i].R for i in ids]) if ids else np.zeros((0, 3, 3)),
        t=np.stack([rec.cameras[i].t for i in ids]) if ids else np.zeros((0, 3)),
        K=np.array([[c.fx, c.fy, c.cx, c.cy, c.width, c.height, c.k1, c.k2] for c in k], float).reshape(-1, 8),
        names=json.dumps({str(i): n for i, n in (image_names or {}).items()}),
    )


def load_reconstruction(path: str) -> Tuple[Reconstruction, Dict[int, str]]:
    import json

    from .camera import Camera, Intrinsics

    d = np.load(path, allow_pickle=False)
    cams = {}
    for i, R, t, k in zip(d["camera_ids"], d["R"], d["t"], d["K"], strict=True):
        dist = (float(k[6]), float(k[7])) if len(k) >= 8 else (0.0, 0.0)  # files written before distortion support
        cams[int(i)] = Camera(Intrinsics(k[0], k[1], k[2], k[3], int(k[4]), int(k[5]), *dist), R, t)
    rec = Reconstruction(cameras=cams, points=d["points"], colors=d["colors"])
    names = {int(i): n for i, n in json.loads(str(d["names"])).items()}
    return rec, names
