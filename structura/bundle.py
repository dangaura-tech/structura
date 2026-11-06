"""Sparse bundle adjustment with SciPy's trust-region least squares.

Parameters: per-camera rotation vector + translation (6 DoF), per-point XYZ and, optionally,
per-physical-camera intrinsics (focal length, radial distortion k1/k2). The first
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


INTRINSIC_MODES = {"none": 0, "focal": 1, "radial": 3}


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
    refine_intrinsics: str = "none",
) -> BAResult:
    """Refine ``cameras`` (in place) and ``points`` (in place).

    Args:
        cameras: image id -> Camera.
        points: (P, 3) world points.
        cam_idx / pt_idx: (O,) image id and point index of each observation.
        obs: (O, 2) observed (distorted) pixels.
        fixed: image ids whose pose is not optimised.
        refine_intrinsics: ``"none"`` keeps intrinsics fixed, ``"focal"`` refines the focal length and
            ``"radial"`` the focal length plus radial distortion ``k1, k2``. Cameras that share one
            :class:`Intrinsics` object (one physical camera) share one set of intrinsic parameters;
            the principal point stays fixed, as it is poorly constrained without calibration targets.
    """
    if refine_intrinsics not in INTRINSIC_MODES:
        raise ValueError(f"refine_intrinsics must be one of {sorted(INTRINSIC_MODES)}")
    n_k = INTRINSIC_MODES[refine_intrinsics]
    ids = [i for i in cameras if i not in set(fixed)]
    col = {img: k for k, img in enumerate(ids)}
    n_c, n_p = len(ids), len(points)

    # one intrinsic block per distinct Intrinsics object
    groups: list = []
    group_of: Dict[int, int] = {}
    for img in cameras:
        k = cameras[img].intrinsics
        for g, other in enumerate(groups):
            if other is k:
                group_of[img] = g
                break
        else:
            group_of[img] = len(groups)
            groups.append(k)
    n_g = len(groups) if n_k else 0
    obs_group = np.array([group_of[i] for i in cam_idx], int)
    aspect = np.array([g.fy / g.fx for g in groups])[obs_group]
    cx = np.array([g.cx for g in groups])[obs_group]
    cy = np.array([g.cy for g in groups])[obs_group]
    f_fixed = np.array([g.fx for g in groups])[obs_group]
    k_fixed = np.array([[g.k1, g.k2] for g in groups])[obs_group]
    fixed_pose = {i: (matrix_to_rotvec(cameras[i].R), cameras[i].t) for i in cameras if i not in col}
    off_p, off_k = 6 * n_c, 6 * n_c + 3 * n_p

    def unpack(x):
        cam = x[:off_p].reshape(n_c, 6)
        pts = x[off_p:off_k].reshape(n_p, 3)
        intr = x[off_k:].reshape(n_g, n_k) if n_k else np.zeros((0, 0))
        return cam, pts, intr

    obs_free = np.array([i in col for i in cam_idx])
    obs_col = np.array([col.get(i, 0) for i in cam_idx])
    r_fixed = np.array([fixed_pose[i][0] if i in fixed_pose else np.zeros(3) for i in cam_idx])
    t_fixed = np.array([fixed_pose[i][1] if i in fixed_pose else np.zeros(3) for i in cam_idx])

    def residuals(x):
        cam, pts, intr = unpack(x)
        r = np.where(obs_free[:, None], cam[obs_col, :3], r_fixed)
        t = np.where(obs_free[:, None], cam[obs_col, 3:], t_fixed)
        pc = _rotate(pts[pt_idx], r) + t
        z = np.where(np.abs(pc[:, 2]) < 1e-9, 1e-9, pc[:, 2])
        xn, yn = pc[:, 0] / z, pc[:, 1] / z
        f = intr[obs_group, 0] if n_k >= 1 else f_fixed
        k1, k2 = (intr[obs_group, 1], intr[obs_group, 2]) if n_k == 3 else (k_fixed[:, 0], k_fixed[:, 1])
        r2 = xn * xn + yn * yn
        d = 1.0 + k1 * r2 + k2 * r2 * r2
        u = f * xn * d + cx
        v = f * aspect * yn * d + cy
        return np.c_[u - obs[:, 0], v - obs[:, 1]].ravel()

    sparsity = lil_matrix((2 * len(obs), off_k + n_g * n_k), dtype=int)
    rows = np.arange(len(obs))
    sel = rows[obs_free]
    for k in range(6):
        sparsity[2 * sel, 6 * obs_col[obs_free] + k] = 1
        sparsity[2 * sel + 1, 6 * obs_col[obs_free] + k] = 1
    for k in range(3):
        sparsity[2 * rows, off_p + 3 * pt_idx + k] = 1
        sparsity[2 * rows + 1, off_p + 3 * pt_idx + k] = 1
    for k in range(n_k):
        sparsity[2 * rows, off_k + n_k * obs_group + k] = 1
        sparsity[2 * rows + 1, off_k + n_k * obs_group + k] = 1

    intr0 = [[g.fx, g.k1, g.k2][:n_k] for g in groups] if n_k else []
    x0 = np.concatenate(
        [np.concatenate([np.r_[matrix_to_rotvec(cameras[i].R), cameras[i].t] for i in ids]) if ids else [],
         points.ravel(), np.asarray(intr0, float).ravel()]
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
    cam, pts, intr = unpack(res.x)
    for img, k in col.items():
        cameras[img].R = rotvec_to_matrix(cam[k, :3])
        cameras[img].t = cam[k, 3:].copy()
    points[:] = pts
    for g, k in enumerate(groups[:n_g]):
        ratio = k.fy / k.fx
        k.fx = float(intr[g, 0])
        k.fy = float(intr[g, 0] * ratio)
        if n_k == 3:
            k.k1, k.k2 = float(intr[g, 1]), float(intr[g, 2])

    def rmse(r):
        return float(np.sqrt(np.mean(r.reshape(-1, 2) ** 2) * 2))

    return BAResult(rmse(r0), rmse(res.fun), int(nfev), bool(res.success))
