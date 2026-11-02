"""Dependency-light software renderer for point clouds and camera frustums (NumPy + Pillow).

Used to produce turntable GIFs of reconstructions without OpenGL or matplotlib: points are
projected with a virtual :class:`Camera`, splatted as discs in back-to-front order, and
camera frustums are drawn as wireframes.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from PIL import Image, ImageDraw

from .camera import Camera, Intrinsics, look_at


def frustum_lines(cam: Camera, scale: float = 0.3) -> List[np.ndarray]:
    k = cam.intrinsics
    w, h = k.width or 2 * k.cx, k.height or 2 * k.cy
    corners_px = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
    rays = np.c_[(corners_px[:, 0] - k.cx) / k.fx, (corners_px[:, 1] - k.cy) / k.fy, np.ones(4)] * scale
    world = (rays - cam.t) @ cam.R  # camera -> world: R^T (x - t)
    c = cam.center
    return [np.stack([c, world[i]]) for i in range(4)] + [np.stack([world[i], world[(i + 1) % 4]]) for i in range(4)]


def render_frame(
    points: np.ndarray,
    colors: np.ndarray,
    view: Camera,
    size=(640, 480),
    cameras: Optional[Dict[int, Camera]] = None,
    point_radius: float = 2.0,
    background=(13, 17, 23),
    camera_color=(255, 120, 60),
    frustum_scale: float = 0.4,
) -> Image.Image:
    img = Image.new("RGB", size, background)
    draw = ImageDraw.Draw(img)
    z = view.depth(points)
    ok = z > 0.05
    uv = view.project(points[ok])
    order = np.argsort(-z[ok])  # far to near
    rad = np.clip(point_radius * np.median(z[ok]) / z[ok], 0.8, 2.5 * point_radius)
    for i in order:
        x, y = uv[i]
        if 0 <= x < size[0] and 0 <= y < size[1]:
            r = rad[i]
            draw.ellipse([x - r, y - r, x + r, y + r], fill=tuple(int(c) for c in colors[ok][i]))
    for cam in (cameras or {}).values():
        for seg in frustum_lines(cam, frustum_scale):
            if (view.depth(seg) > 0.05).all():
                p = view.project(seg)
                draw.line([tuple(p[0]), tuple(p[1])], fill=camera_color, width=2)
    return img


def scene_up(cameras: Dict[int, Camera]) -> np.ndarray:
    """World 'up' estimated as the mean camera -y axis (photos are usually taken upright)."""
    up = -np.mean([c.R[1] for c in cameras.values()], axis=0)
    return up / np.linalg.norm(up)


def turntable(
    points: np.ndarray,
    colors: np.ndarray,
    cameras: Optional[Dict[int, Camera]] = None,
    frames: int = 72,
    size=(640, 480),
    elevation: float = 0.45,
    up: Optional[np.ndarray] = None,
) -> List[Image.Image]:
    """Orbit a virtual camera around the scene and render each frame.

    The orbit axis is the scene's up direction and its radius encloses all cameras, so the
    animation works for reconstructions in any (arbitrary) gauge and scale.
    """
    center = np.median(points, axis=0)
    extent = float(np.percentile(np.linalg.norm(points - center, axis=1), 90))
    if up is None:
        up = scene_up(cameras) if cameras else np.array([0.0, -1.0, 0.0])
    e1 = np.cross(up, [1.0, 0.0, 0.0] if abs(up[0]) < 0.9 else [0.0, 0.0, 1.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    cam_dist = max((np.linalg.norm(c.center - center) for c in (cameras or {}).values()), default=extent)
    radius = 1.6 * max(cam_dist, 2.5 * extent)
    K = Intrinsics(0.9 * size[0], 0.9 * size[0], size[0] / 2, size[1] / 2, *size)
    if cameras:  # frame both the structure and the camera ring
        center = 0.6 * center + 0.4 * np.mean([c.center for c in cameras.values()], axis=0)
    out = []
    for f in range(frames):
        a = 2 * np.pi * f / frames
        eye = center + radius * (np.cos(a) * e1 + np.sin(a) * e2 + elevation * up)
        R, t = look_at(eye, center, up=up)
        out.append(render_frame(points, colors, Camera(K, R, t), size, cameras, frustum_scale=0.12 * extent))
    return out


def save_gif(frames: List[Image.Image], path: str, fps: int = 18) -> None:
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0, optimize=True)
