"""Sparse bundle adjustment with SciPy's trust-region least squares.

Parameters: per-camera rotation vector + translation (6 DoF) and per-point XYZ. The first
camera is held fixed to remove the gauge freedom (rotation/translation); scale is
fixed by the initial two-view baseline through the remaining parameters' initialisation.
The Jacobian sparsity pattern (each residual depends on one camera and one point) lets
``least_squares`` use sparse finite differences, so problems with tens of thousands of
observations solve in seconds. A robust Huber loss down-weights outliers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .camera import Camera, matrix_to_rotvec, rotvec_to_matrix


@dataclass
class BAResult:
    initial_rmse: float
    final_rmse: float
    iterations: int
    success: bool


def _rotate(points: np.ndarray, rvecs: np.ndarray) -> np.ndarray:
    """Rodrigues rotation of each point by its own rotation vector (vectorised)."""
    theta = np.linalg.norm(rvecs, axis=1)[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.where(theta > 1e-12, rvecs / theta, 0.0)
    dot = np.sum(points * v, axis=1)[:, None]
    cos, sin = np.cos(theta), np.sin(theta)
    return cos * points + sin * np.cross(v, points) + dot * (1 - cos) * v


def bundle_adjust(
    cameras: Dict[int, Camera],
    points: np.ndarray,
    cam_idx: np.ndarray,
    pt_idx: np.ndarray,
    obs: np.ndarray,
    fixed: Sequence[int] = (),
    loss: str = "huber",
    f_scale: float = 2.0,
    max_nfev: int = 50,
) -> BAResult:
    """Refine ``cameras`` (in place) and ``points`` (in place).

    Args:
        cameras: image id -> Camera (shared intrinsics per camera are kept fixed).
        points: (P, 3) world points.
        cam_idx / pt_idx: (O,) image id and point index of each observation.
        obs: (O, 2) observed pixels.
        fixed: image ids whose pose is not optimised.
    """
    ids = [i for i in cameras if i not in set(fixed)]
    col = {img: k for k, img in enumerate(ids)}
    n_c, n_p = len(ids), len(points)
    fx = np.array([cameras[i].intrinsics.fx for i in cam_idx])
    fy = np.array([cameras[i].intrinsics.fy for i in cam_idx])
    cx = np.array([cameras[i].intrinsics.cx for i in cam_idx])
    cy = np.array([cameras[i].intrinsics.cy for i in cam_idx])
    fixed_pose = {i: (matrix_to_rotvec(cameras[i].R), cameras[i].t) for i in cameras if i not in col}

    def unpack(x):
        cam = x[: 6 * n_c].reshape(n_c, 6)
        pts = x[6 * n_c :].reshape(n_p, 3)
        return cam, pts

    obs_free = np.array([i in col for i in cam_idx])
    obs_col = np.array([col.get(i, 0) for i in cam_idx])
    r_fixed = np.array([fixed_pose[i][0] if i in fixed_pose else np.zeros(3) for i in cam_idx])
    t_fixed = np.array([fixed_pose[i][1] if i in fixed_pose else np.zeros(3) for i in cam_idx])

    def residuals(x):
        cam, pts = unpack(x)
        r = np.where(obs_free[:, None], cam[obs_col, :3], r_fixed)
        t = np.where(obs_free[:, None], cam[obs_col, 3:], t_fixed)
        pc = _rotate(pts[pt_idx], r) + t
        z = np.where(np.abs(pc[:, 2]) < 1e-9, 1e-9, pc[:, 2])
        u = fx * pc[:, 0] / z + cx
        v = fy * pc[:, 1] / z + cy
        return np.c_[u - obs[:, 0], v - obs[:, 1]].ravel()

    sparsity = lil_matrix((2 * len(obs), 6 * n_c + 3 * n_p), dtype=int)
    rows = np.arange(len(obs))
    for k in range(6):
        sel = rows[obs_free]
        sparsity[2 * sel, 6 * obs_col[obs_free] + k] = 1
        sparsity[2 * sel + 1, 6 * obs_col[obs_free] + k] = 1
    for k in range(3):
        sparsity[2 * rows, 6 * n_c + 3 * pt_idx + k] = 1
        sparsity[2 * rows + 1, 6 * n_c + 3 * pt_idx + k] = 1

    x0 = np.concatenate(
        [np.concatenate([np.r_[matrix_to_rotvec(cameras[i].R), cameras[i].t] for i in ids]) if ids else [],
         points.ravel()]
    )  # fmt: skip
    r0 = residuals(x0)
    # Two-stage robust optimisation: when the initialisation is poor, *every* residual is large and a
    # robust loss with a small scale treats them all as outliers, which slows convergence dramatically.
    # Stage 1 therefore adapts the scale to the current median residual; stage 2 uses the target scale.
    x = x0
    nfev = 0
    adaptive = max(f_scale, 3.0 * float(np.median(np.abs(r0))))
    stages = [adaptive, f_scale] if loss != "linear" and adaptive > f_scale else [f_scale]
    for scale in stages:
        res = least_squares(
            residuals, x, jac_sparsity=sparsity, x_scale="jac", loss=loss, f_scale=scale, max_nfev=max_nfev,
            method="trf", ftol=1e-8,
        )  # fmt: skip
        x = res.x
        nfev += res.nfev
    cam, pts = unpack(res.x)
    for img, k in col.items():
        cameras[img].R = rotvec_to_matrix(cam[k, :3])
        cameras[img].t = cam[k, 3:].copy()
    points[:] = pts

    def rmse(r):
        return float(np.sqrt(np.mean(r.reshape(-1, 2) ** 2) * 2))

    return BAResult(rmse(r0), rmse(res.fun), int(nfev), bool(res.success))
