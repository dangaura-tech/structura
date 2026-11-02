"""Two-view and multi-view geometry: triangulation, relative pose, absolute pose (PnP)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .camera import Camera, Intrinsics, rotvec_to_matrix


def triangulate_dlt(P1: np.ndarray, P2: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Linear (DLT) triangulation of N correspondences; returns (N, 3) world points.

    Each point solves the 4x4 homogeneous system built from ``x × (P X) = 0`` via SVD.
    """
    x1 = np.asarray(x1, float)
    x2 = np.asarray(x2, float)
    A = np.stack(
        [
            x1[:, 0, None] * P1[2] - P1[0],
            x1[:, 1, None] * P1[2] - P1[1],
            x2[:, 0, None] * P2[2] - P2[0],
            x2[:, 1, None] * P2[2] - P2[1],
        ],
        axis=1,
    )  # (N, 4, 4)
    _, _, vt = np.linalg.svd(A)
    X = vt[:, -1]
    return X[:, :3] / X[:, 3:4]


def reprojection_errors(cam: Camera, X: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.linalg.norm(cam.project(X) - x, axis=1)


def triangulation_angles(c1: np.ndarray, c2: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Angle (degrees) between the two viewing rays of every point - small = ill-conditioned depth."""
    r1 = X - c1
    r2 = X - c2
    cos = np.sum(r1 * r2, axis=1) / (np.linalg.norm(r1, axis=1) * np.linalg.norm(r2, axis=1) + 1e-12)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


@dataclass
class TwoViewResult:
    R: np.ndarray
    t: np.ndarray
    inliers: np.ndarray  # bool mask over the input correspondences
    points: np.ndarray  # triangulated inlier points in the first camera frame
    num_inliers: int


def relative_pose(
    x1: np.ndarray, x2: np.ndarray, K: Intrinsics, threshold: float = 1.0, confidence: float = 0.999
) -> Optional[TwoViewResult]:
    """Essential matrix (5-point + RANSAC) and cheirality-checked pose of view 2 w.r.t. view 1.

    The translation has unit norm (scale is unobservable from two views).
    """
    if len(x1) < 8:
        return None
    E, mask = cv2.findEssentialMat(
        x1.astype(np.float64), x2.astype(np.float64), K.K, method=cv2.RANSAC, prob=confidence, threshold=threshold
    )
    if E is None or E.shape != (3, 3):
        return None
    n, R, t, mask_pose = cv2.recoverPose(E, x1, x2, K.K, mask=mask.copy())
    inl = mask_pose.ravel() > 0
    if n < 8:
        return None
    cam1 = Camera(K)
    cam2 = Camera(K, R, t.ravel())
    X = triangulate_dlt(cam1.P, cam2.P, x1[inl], x2[inl])
    return TwoViewResult(R, t.ravel(), inl, X, int(inl.sum()))


def absolute_pose(
    X: np.ndarray, x: np.ndarray, K: Intrinsics, threshold: float = 4.0, iterations: int = 1000
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """PnP + RANSAC (EPnP minimal solver, iterative refinement). Returns (R, t, inlier mask)."""
    if len(X) < 6:
        return None
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        X.astype(np.float64),
        x.astype(np.float64),
        K.K,
        None,
        iterationsCount=iterations,
        reprojectionError=threshold,
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or inliers is None or len(inliers) < 6:
        return None
    mask = np.zeros(len(X), bool)
    mask[inliers.ravel()] = True
    ok, rvec, tvec = cv2.solvePnP(
        X[mask], x[mask], K.K, None, rvec, tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE
    )
    return rotvec_to_matrix(rvec.ravel()), tvec.ravel(), mask


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity ``dst ≈ s R src + t`` (Umeyama, 1991). Returns (s, R, t)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s_c, d_c = src - mu_s, dst - mu_d
    cov = d_c.T @ s_c / len(src)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    var_s = (s_c**2).sum() / len(src)
    s = float(np.trace(np.diag(S) @ D) / var_s) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t
