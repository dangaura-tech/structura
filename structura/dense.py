"""Multi-view stereo densification by plane sweeping (Collins, 1996; Gallup et al., 2007).

The sparse SfM model fixes the cameras; this module estimates a depth for (almost) every
pixel of each reference image and fuses the depth maps into a dense, coloured point cloud.

1. **Depth hypotheses.** The depth range of the reference view is taken from the sparse
   points it sees (robust percentiles, slightly widened) and sampled uniformly in *inverse*
   depth, which spaces planes evenly in image disparity.
2. **Plane sweep.** For each fronto-parallel plane ``n^T X = d`` in the reference camera
   frame, every neighbouring image is warped into the reference view with the plane-induced
   homography ``H = K_s (R + t n^T / d) K_r^{-1}``, where ``(R, t)`` maps reference to source
   camera coordinates.
3. **Matching cost.** Zero-mean normalised cross-correlation over a square window, computed
   for all pixels at once with box filters, is robust to exposure changes between photos.
   The per-pixel cost aggregates the best ``k`` neighbours, which tolerates occlusion in
   some views.
4. **Winner-take-all with sub-plane refinement.** A parabola through the best plane and its
   two neighbours refines the inverse depth; low-NCC and low-texture pixels are rejected.
5. **Fusion.** A pixel's 3D point is kept only if it re-projects consistently into at least
   ``min_consistent`` other depth maps (depth agreement within a relative tolerance), which
   removes most mismatches; survivors are back-projected with their reference colour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .camera import Camera


@dataclass
class MVSConfig:
    num_depths: int = 96
    window: int = 7
    num_neighbors: int = 4
    best_k: int = 2
    min_ncc: float = 0.5
    min_texture: float = 4.0  # std of the reference window in grey levels
    depth_margin: float = 0.15  # widen the sparse depth range by this fraction
    min_consistent: int = 1
    consistency_tol: float = 0.01  # relative depth agreement for fusion
    downscale: int = 2  # work on 1 / downscale resolution (speed)


@dataclass
class DepthMap:
    depth: np.ndarray  # (H, W), 0 where invalid
    score: np.ndarray  # (H, W) aggregated NCC
    camera: Camera  # intrinsics scaled to this resolution


def scaled_camera(cam: Camera, s: float) -> Camera:
    from dataclasses import replace

    k = cam.intrinsics
    k2 = replace(k, fx=k.fx * s, fy=k.fy * s, cx=(k.cx + 0.5) * s - 0.5, cy=(k.cy + 0.5) * s - 0.5,
                 width=int(round(k.width * s)), height=int(round(k.height * s)))  # fmt: skip
    return Camera(k2, cam.R, cam.t)


def select_neighbors(cameras: Dict[int, Camera], ref: int, k: int, min_angle: float = 3.0) -> List[int]:
    """Neighbours sorted by viewing-direction similarity, skipping near-identical viewpoints."""
    c_ref, z_ref = cameras[ref].center, cameras[ref].R[2]
    cands = []
    for i, cam in cameras.items():
        if i == ref:
            continue
        cos = float(np.clip(z_ref @ cam.R[2], -1, 1))
        baseline = np.linalg.norm(cam.center - c_ref)
        if np.degrees(np.arccos(cos)) < min_angle and baseline < 1e-6:
            continue
        cands.append((-cos, i))
    return [i for _, i in sorted(cands)[:k]]


def depth_range(cam: Camera, points: np.ndarray, margin: float) -> Optional[Tuple[float, float]]:
    if len(points) == 0:
        return None
    z = cam.depth(points)
    uv = cam.project(points[z > 0]) if (z > 0).any() else np.zeros((0, 2))
    k = cam.intrinsics
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < k.width) & (uv[:, 1] >= 0) & (uv[:, 1] < k.height)
    z = z[z > 0][inside]
    if len(z) < 10:
        return None
    lo, hi = np.percentile(z, [1, 99])
    span = hi - lo
    return max(lo - margin * span, 1e-3), hi + margin * span


def _box(x: np.ndarray, w: int) -> np.ndarray:
    return cv2.boxFilter(x, cv2.CV_32F, (w, w), normalize=True, borderType=cv2.BORDER_REFLECT)


def plane_sweep(
    ref_img: np.ndarray,
    ref_cam: Camera,
    src: Sequence[Tuple[np.ndarray, Camera]],
    depth_lo: float,
    depth_hi: float,
    cfg: MVSConfig,
) -> DepthMap:
    """Depth map of the reference image (grey images, cameras at the same resolution)."""
    h, w = ref_img.shape
    ref = ref_img.astype(np.float32)
    mu_r = _box(ref, cfg.window)
    var_r = np.maximum(_box(ref * ref, cfg.window) - mu_r**2, 0)
    texture = np.sqrt(var_r)
    inv = np.linspace(1 / depth_lo, 1 / depth_hi, cfg.num_depths)  # near -> far
    Kr_inv = np.linalg.inv(ref_cam.intrinsics.K)
    n = np.array([0.0, 0.0, 1.0])
    costs = np.full((cfg.num_depths, h, w), -1.0, np.float32)
    rel = []
    for img, cam in src:
        R = cam.R @ ref_cam.R.T
        t = cam.t - R @ ref_cam.t
        rel.append((img.astype(np.float32), cam.intrinsics.K, R, t))
    for d_i, inv_d in enumerate(inv):
        scores = []
        for img, Ks, R, t in rel:
            H = Ks @ (R + np.outer(t, n) * inv_d) @ Kr_inv  # plane z = d in reference coordinates
            warped = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)  # fmt: skip
            valid = np.isfinite(warped)
            warped = np.where(valid, warped, 0).astype(np.float32)
            mu_s = _box(warped, cfg.window)
            var_s = np.maximum(_box(warped * warped, cfg.window) - mu_s**2, 0)
            cov = _box(ref * warped, cfg.window) - mu_r * mu_s
            ncc = cov / np.sqrt(var_r * var_s + 1e-6)
            ncc[_box(valid.astype(np.float32), cfg.window) < 0.999] = -1.0
            scores.append(ncc)
        s = np.sort(np.stack(scores), axis=0)[::-1][: min(cfg.best_k, len(scores))]
        costs[d_i] = s.mean(0)
    best = np.argmax(costs, axis=0)
    score = np.take_along_axis(costs, best[None], 0)[0]
    # parabolic refinement in inverse depth
    i0 = np.clip(best, 1, cfg.num_depths - 2)
    c_m = np.take_along_axis(costs, (i0 - 1)[None], 0)[0]
    c_0 = np.take_along_axis(costs, i0[None], 0)[0]
    c_p = np.take_along_axis(costs, (i0 + 1)[None], 0)[0]
    denom = c_m - 2 * c_0 + c_p
    offset = np.where((best == i0) & (denom < -1e-6), 0.5 * (c_m - c_p) / np.where(denom < -1e-6, denom, -1), 0.0)
    step = inv[1] - inv[0]
    inv_best = inv[best] + np.clip(offset, -0.5, 0.5) * step
    depth = 1.0 / inv_best
    ok = (score >= cfg.min_ncc) & (texture >= cfg.min_texture) & (best > 0) & (best < cfg.num_depths - 1)
    return DepthMap(np.where(ok, depth, 0.0).astype(np.float64), score, ref_cam)


def backproject(dm: DepthMap) -> Tuple[np.ndarray, np.ndarray]:
    """World points of the valid pixels, and their (row, col) indices."""
    v, u = np.nonzero(dm.depth > 0)
    k = dm.camera.intrinsics
    z = dm.depth[v, u]
    xc = np.c_[(u - k.cx) / k.fx * z, (v - k.cy) / k.fy * z, z]
    return (xc - dm.camera.t) @ dm.camera.R, np.c_[v, u]


def fuse(depth_maps: Dict[int, DepthMap], colors: Dict[int, np.ndarray], cfg: MVSConfig):
    """Geometrically consistent fusion of depth maps into one coloured point cloud."""
    pts_all, col_all = [], []
    for i, dm in depth_maps.items():
        X, vu = backproject(dm)
        if not len(X):
            continue
        support = np.zeros(len(X), int)
        for j, other in depth_maps.items():
            if j == i:
                continue
            cam = other.camera
            z = cam.depth(X)
            uv = np.round(cam.project(X)).astype(int)
            hh, ww = other.depth.shape
            inside = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < ww) & (uv[:, 1] >= 0) & (uv[:, 1] < hh)
            d_other = np.zeros(len(X))
            d_other[inside] = other.depth[uv[inside, 1], uv[inside, 0]]
            agree = inside & (d_other > 0) & (np.abs(d_other - z) <= cfg.consistency_tol * z)
            support += agree
        keep = support >= cfg.min_consistent
        pts_all.append(X[keep])
        col_all.append(colors[i][vu[keep, 0], vu[keep, 1]])
    if not pts_all:
        return np.zeros((0, 3)), np.zeros((0, 3), np.uint8)
    return np.concatenate(pts_all), np.concatenate(col_all).astype(np.uint8)


def densify(
    images_bgr: Dict[int, np.ndarray],
    cameras: Dict[int, Camera],
    sparse_points: np.ndarray,
    cfg: Optional[MVSConfig] = None,
    views: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, DepthMap]]:
    """Dense coloured points from registered images. Returns ``(points, colors_rgb, depth_maps)``."""
    cfg = cfg or MVSConfig()
    s = 1.0 / cfg.downscale
    ids = [i for i in (views or sorted(cameras)) if i in images_bgr]
    small, grey, cams = {}, {}, {}
    for i in cameras:
        if i not in images_bgr:
            continue
        img = images_bgr[i]
        if cfg.downscale > 1:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        small[i] = img[..., ::-1].copy()
        grey[i] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cams[i] = scaled_camera(cameras[i], s)
    maps: Dict[int, DepthMap] = {}
    for r in ids:
        rng = depth_range(cams[r], sparse_points, cfg.depth_margin)
        nbrs = [j for j in select_neighbors(cams, r, cfg.num_neighbors) if j in grey]
        if rng is None or not nbrs:
            continue
        maps[r] = plane_sweep(grey[r], cams[r], [(grey[j], cams[j]) for j in nbrs], *rng, cfg)
    pts, cols = fuse(maps, small, cfg)
    return pts, cols, maps
