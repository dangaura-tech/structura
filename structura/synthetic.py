"""Synthetic multi-view scenes with exact ground truth.

Points are sampled on the surfaces of a few coloured primitives (sphere, box, torus) and a
ground plane; cameras sit on a ring looking at the scene centre. Each camera "detects" the
projections of the points it sees (in front, inside the image, not too oblique) with
Gaussian pixel noise, and pairwise matches are generated with a configurable outlier rate,
which exercises every RANSAC stage of the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .camera import Camera, Intrinsics, look_at


@dataclass
class SyntheticScene:
    points: np.ndarray  # (P, 3) ground-truth world points
    colors: np.ndarray  # (P, 3) uint8
    cameras: Dict[int, Camera]  # ground-truth cameras
    intrinsics: Dict[int, Intrinsics]
    keypoints: Dict[int, np.ndarray]  # image -> (K, 2)
    keypoint_point: Dict[int, np.ndarray]  # image -> (K,) ground-truth point id (-1 for clutter)
    keypoint_colors: Dict[int, np.ndarray]
    matches: Dict[Tuple[int, int], np.ndarray]


def _sphere(rng, n, center, r):
    v = rng.normal(size=(n, 3))
    return center + r * v / np.linalg.norm(v, axis=1, keepdims=True)


def _box(rng, n, center, size):
    p = rng.uniform(-0.5, 0.5, (n, 3))
    axis = rng.integers(0, 3, n)
    p[np.arange(n), axis] = np.sign(rng.uniform(-1, 1, n)) * 0.5
    return center + p * np.asarray(size)


def _torus(rng, n, center, R, r):
    u, v = rng.uniform(0, 2 * np.pi, (2, n))
    return center + np.c_[(R + r * np.cos(v)) * np.cos(u), -r * np.sin(v), (R + r * np.cos(v)) * np.sin(u)]


def make_scene(
    num_cameras: int = 12,
    points_per_object: int = 600,
    radius: float = 6.0,
    noise_px: float = 0.5,
    outlier_ratio: float = 0.1,
    image_size: Tuple[int, int] = (1024, 768),
    seed: int = 0,
) -> SyntheticScene:
    rng = np.random.default_rng(seed)
    objs = [
        (_sphere(rng, points_per_object, np.array([0.0, -1.0, 0.0]), 1.0), (230, 90, 70)),
        (_box(rng, points_per_object, np.array([-2.0, -0.6, 1.0]), (1.2, 1.2, 1.2)), (70, 140, 230)),
        (_torus(rng, points_per_object, np.array([2.0, -0.5, -0.5]), 0.8, 0.3), (240, 200, 60)),
        (np.c_[rng.uniform(-4, 4, points_per_object), np.zeros(points_per_object),
               rng.uniform(-4, 4, points_per_object)], (120, 120, 120)),
    ]  # fmt: skip
    points = np.vstack([o[0] for o in objs])
    colors = np.vstack([np.tile(c, (len(o), 1)) for o, c in objs]).astype(np.uint8)
    colors = np.clip(colors + rng.normal(0, 12, colors.shape), 0, 255).astype(np.uint8)

    w, h = image_size
    K = Intrinsics(0.9 * w, 0.9 * w, w / 2, h / 2, w, h)
    cams, intr, kps, kpp, kpc = {}, {}, {}, {}, {}
    for i in range(num_cameras):
        a = 2 * np.pi * i / num_cameras
        center = np.array([radius * np.cos(a), -2.0 - 0.5 * np.sin(3 * a), radius * np.sin(a)])
        R, t = look_at(center, np.array([0.0, -0.6, 0.0]))
        cam = Camera(K, R, t)
        uv = cam.project(points)
        z = cam.depth(points)
        vis = (z > 0.1) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        vis &= rng.random(len(points)) < 0.85  # detector misses
        ids = np.flatnonzero(vis)
        clutter = rng.uniform([0, 0], [w, h], (len(ids) // 10, 2))
        kps[i] = np.vstack([uv[ids] + rng.normal(0, noise_px, (len(ids), 2)), clutter])
        kpp[i] = np.r_[ids, -np.ones(len(clutter), int)]
        kpc[i] = np.vstack([colors[ids], np.full((len(clutter), 3), 128)])
        cams[i], intr[i] = cam, K

    matches = {}
    for i in range(num_cameras):
        for j in range(i + 1, num_cameras):
            common, ia, ja = np.intersect1d(kpp[i][kpp[i] >= 0], kpp[j][kpp[j] >= 0], return_indices=True)
            if len(common) < 30:
                continue
            m = np.c_[ia, ja]
            n_out = int(outlier_ratio * len(m))
            if n_out:
                fake = np.c_[rng.integers(0, len(kps[i]), n_out), rng.integers(0, len(kps[j]), n_out)]
                m = np.vstack([m, fake])
            matches[(i, j)] = m
    return SyntheticScene(points, colors, cams, intr, kps, kpp, kpc, matches)


def procedural_texture(size: int = 512, seed: int = 0) -> np.ndarray:
    """Feature-rich RGB texture: smooth colour field + random glyphs, circles and lines."""
    import cv2

    rng = np.random.default_rng(seed)
    base = cv2.resize(rng.integers(40, 215, (8, 8, 3)).astype(np.uint8), (size, size), interpolation=cv2.INTER_CUBIC)
    img = base.copy()
    for _ in range(140):
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        kind = rng.integers(0, 3)
        x, y = (int(v) for v in rng.integers(0, size, 2))
        if kind == 0:
            cv2.putText(img, chr(int(rng.integers(65, 91))), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        float(rng.uniform(0.6, 1.8)), color, int(rng.integers(1, 4)))  # fmt: skip
        elif kind == 1:
            cv2.circle(img, (x, y), int(rng.integers(4, 22)), color, int(rng.integers(-1, 3)) or -1)
        else:
            x2, y2 = (int(v) for v in rng.integers(0, size, 2))
            cv2.line(img, (x, y), (x2, y2), color, int(rng.integers(1, 3)))
    return img


def render_textured_views(
    num_views: int = 10, image_size: Tuple[int, int] = (800, 600), arc: float = np.pi * 0.9, seed: int = 0
):
    """Render photo-like views of a textured box on a textured floor.

    Each planar face is warped into the view with its homography (exact for planes), faces
    are drawn back-to-front and back faces are culled. Returns ``(images_bgr, cameras)``;
    ``cameras`` are the ground-truth poses (with intrinsics) for evaluation.
    """
    import cv2

    w, h = image_size
    K = Intrinsics(0.9 * w, 0.9 * w, w / 2, h / 2, w, h)
    tex = [procedural_texture(512, seed + k) for k in range(6)]
    s = 1.0  # box half-size, sitting on the floor (y = 0, -y is up)
    faces = [
        (np.array([[-4, 0, -4], [4, 0, -4], [4, 0, 4], [-4, 0, 4]], float), 5),  # floor
        (np.array([[-s, -2 * s, -s], [s, -2 * s, -s], [s, 0, -s], [-s, 0, -s]], float), 0),
        (np.array([[s, -2 * s, -s], [s, -2 * s, s], [s, 0, s], [s, 0, -s]], float), 1),
        (np.array([[s, -2 * s, s], [-s, -2 * s, s], [-s, 0, s], [s, 0, s]], float), 2),
        (np.array([[-s, -2 * s, s], [-s, -2 * s, -s], [-s, 0, -s], [-s, 0, s]], float), 3),
        (np.array([[-s, -2 * s, s], [s, -2 * s, s], [s, -2 * s, -s], [-s, -2 * s, -s]], float), 4),  # top
    ]
    centre = np.array([0.0, -s, 0.0])
    images, cams = [], {}
    for i in range(num_views):
        a = -arc / 2 + arc * i / max(num_views - 1, 1)
        eye = np.array([5.5 * np.sin(a), -3.0, -5.5 * np.cos(a)])
        R, t = look_at(eye, centre)
        cam = Camera(K, R, t)
        img = np.full((h, w, 3), 30, np.uint8)
        order = sorted(faces, key=lambda f: -cam.depth(f[0]).mean())
        for corners, k in order:
            n = np.cross(corners[1] - corners[0], corners[3] - corners[0])
            mid = corners.mean(0)
            # back-face culling: a box face is visible iff its outward normal points towards the camera
            if k != 5 and np.dot(n, mid - centre) * np.dot(n, cam.center - mid) <= 0:
                continue
            if (cam.depth(corners) <= 0.1).any():
                continue
            dst = cam.project(corners).astype(np.float32)
            src = np.float32([[0, 0], [511, 0], [511, 511], [0, 511]])
            H = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(tex[k], H, (w, h), flags=cv2.INTER_LINEAR)
            mask = cv2.warpPerspective(np.full((512, 512), 255, np.uint8), H, (w, h)) > 127
            img[mask] = warped[mask]
        images.append(img)
        cams[i] = cam
    return images, cams
