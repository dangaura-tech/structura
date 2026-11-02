"""Point-cloud processing: downsampling, normals, outlier removal, ICP registration,
RANSAC plane segmentation and Chamfer distance (NumPy + SciPy KD-trees)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .geometry import umeyama


def voxel_downsample(points: np.ndarray, voxel: float, colors: Optional[np.ndarray] = None):
    """Average all points falling into the same voxel."""
    keys = np.floor(points / voxel).astype(np.int64)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    out = np.zeros((len(counts), 3))
    np.add.at(out, inv, points)
    out /= counts[:, None]
    if colors is None:
        return out
    col = np.zeros((len(counts), 3))
    np.add.at(col, inv, colors.astype(float))
    return out, (col / counts[:, None]).astype(np.uint8)


def estimate_normals(points: np.ndarray, k: int = 16, viewpoint: Optional[np.ndarray] = None) -> np.ndarray:
    """PCA normals from the k nearest neighbours, oriented towards ``viewpoint`` if given."""
    tree = cKDTree(points)
    _, idx = tree.query(points, k=min(k, len(points)))
    nb = points[idx] - points[idx].mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nb, nb)
    _, vecs = np.linalg.eigh(cov)
    normals = vecs[:, :, 0]
    if viewpoint is not None:
        flip = np.sum(normals * (viewpoint - points), axis=1) < 0
        normals[flip] *= -1
    return normals


def remove_statistical_outliers(points: np.ndarray, k: int = 16, std_ratio: float = 2.0) -> np.ndarray:
    """Boolean mask of inliers: mean kNN distance within ``mean + std_ratio * std``."""
    d, _ = cKDTree(points).query(points, k=min(k + 1, len(points)))
    mean_d = d[:, 1:].mean(axis=1)
    return mean_d <= mean_d.mean() + std_ratio * mean_d.std()


@dataclass
class ICPResult:
    R: np.ndarray
    t: np.ndarray
    rmse: float
    fitness: float  # fraction of source points with a correspondence
    iterations: int

    @property
    def T(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = self.R, self.t
        return T


def icp(
    source: np.ndarray,
    target: np.ndarray,
    max_distance: float = 0.5,
    max_iterations: int = 50,
    tolerance: float = 1e-7,
    point_to_plane: bool = False,
    init: Optional[np.ndarray] = None,
) -> ICPResult:
    """Rigid ICP aligning ``source`` onto ``target``.

    Point-to-point uses the closed-form Kabsch/Umeyama step; point-to-plane linearises the
    rotation (small-angle) and solves a 6x6 least-squares system per iteration, which
    typically converges in far fewer iterations on smooth surfaces.
    """
    tree = cKDTree(target)
    normals = estimate_normals(target) if point_to_plane else None
    R = np.eye(3) if init is None else init[:3, :3].copy()
    t = np.zeros(3) if init is None else init[:3, 3].copy()
    prev = np.inf
    rmse, fitness = np.inf, 0.0
    iterations = 0
    for iterations in range(1, max_iterations + 1):  # noqa: B007 - reported in the result
        moved = source @ R.T + t
        d, idx = tree.query(moved, distance_upper_bound=max_distance)
        ok = np.isfinite(d)
        if ok.sum() < 6:
            break
        p, q = moved[ok], target[idx[ok]]
        rmse, fitness = float(np.sqrt(np.mean(d[ok] ** 2))), float(ok.mean())
        if point_to_plane:
            n = normals[idx[ok]]
            A = np.c_[np.cross(p, n), n]
            b = np.sum((q - p) * n, axis=1)
            x, *_ = np.linalg.lstsq(A, b, rcond=None)
            w = x[:3]
            th = np.linalg.norm(w)
            if th > 1e-12:
                k = w / th
                Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
                dR = np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx
            else:
                dR = np.eye(3)
            dt = x[3:]
        else:
            _, dR, dt = umeyama(p, q, with_scale=False)
        R, t = dR @ R, dR @ t + dt
        if abs(prev - rmse) < tolerance:
            break
        prev = rmse
    return ICPResult(R, t, rmse, fitness, iterations)


def segment_plane(
    points: np.ndarray, threshold: float = 0.02, iterations: int = 500, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """RANSAC plane fit. Returns (plane [a, b, c, d] with unit normal, inlier mask)."""
    rng = np.random.default_rng(seed)
    best_mask, best_plane = np.zeros(len(points), bool), np.zeros(4)
    for _ in range(iterations):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue
        n /= norm
        d = -n @ a
        mask = np.abs(points @ n + d) < threshold
        if mask.sum() > best_mask.sum():
            best_mask, best_plane = mask, np.r_[n, d]
    if best_mask.sum() >= 3:  # least-squares refit on the inliers
        P = points[best_mask]
        c = P.mean(0)
        n = np.linalg.svd(P - c)[2][-1]
        best_plane = np.r_[n, -n @ c]
        best_mask = np.abs(points @ n - n @ c) < threshold
    return best_plane, best_mask


def chamfer_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric mean nearest-neighbour distance."""
    return float(cKDTree(b).query(a)[0].mean() + cKDTree(a).query(b)[0].mean()) / 2
