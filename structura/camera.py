"""Pinhole cameras and rotation utilities.

Conventions (same as OpenCV / COLMAP): a world point ``X`` maps to camera coordinates
``x_c = R X + t``; the camera centre is ``C = -R^T t``; pixels are ``K d(x_c / z)``, where
``d`` is the two-parameter polynomial radial distortion of COLMAP's ``RADIAL`` / OpenCV models,

    d(x) = x (1 + k1 r^2 + k2 r^4),   r = |x|   (normalised image coordinates).

With ``k1 = k2 = 0`` (the default) the model is a plain pinhole.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int = 0
    height: int = 0
    k1: float = 0.0
    k2: float = 0.0

    @classmethod
    def from_image_size(cls, width: int, height: int, focal_factor: float = 1.2) -> "Intrinsics":
        """Common prior when EXIF focal length is unknown: f = 1.2 * max(w, h)."""
        f = focal_factor * max(width, height)
        return cls(f, f, width / 2.0, height / 2.0, width, height)

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    @property
    def has_distortion(self) -> bool:
        return self.k1 != 0.0 or self.k2 != 0.0

    def distort(self, xn: np.ndarray) -> np.ndarray:
        """Ideal normalised coordinates (N, 2) -> distorted normalised coordinates."""
        xn = np.asarray(xn, dtype=np.float64)
        r2 = np.sum(xn**2, axis=1, keepdims=True)
        return xn * (1.0 + self.k1 * r2 + self.k2 * r2**2)

    def undistort(self, xd: np.ndarray, iterations: int = 30) -> np.ndarray:
        """Invert :meth:`distort` with Newton's method on the radius.

        The model is radially symmetric, so only ``r_u`` with ``r_u (1 + k1 r_u^2 + k2 r_u^4) = r_d``
        has to be found; for the moderate distortion of real lenses this converges in a few steps.
        """
        xd = np.asarray(xd, dtype=np.float64)
        if not self.has_distortion:
            return xd.copy()
        rd = np.linalg.norm(xd, axis=1)
        ru = rd.copy()
        for _ in range(iterations):
            r2 = ru * ru
            f = ru * (1 + self.k1 * r2 + self.k2 * r2 * r2) - rd
            df = 1 + 3 * self.k1 * r2 + 5 * self.k2 * r2 * r2
            step = f / np.where(np.abs(df) < 1e-9, 1e-9, df)
            ru -= step
            if np.max(np.abs(step), initial=0.0) < 1e-12:
                break
        scale = np.where(rd > 1e-12, ru / np.maximum(rd, 1e-12), 1.0)
        return xd * scale[:, None]

    def normalize(self, pixels: np.ndarray) -> np.ndarray:
        """Pixels (N, 2) -> ideal (undistorted) normalised image coordinates (N, 2)."""
        p = np.asarray(pixels, dtype=np.float64)
        return self.undistort(np.c_[(p[:, 0] - self.cx) / self.fx, (p[:, 1] - self.cy) / self.fy])

    def denormalize(self, xn: np.ndarray, distorted: bool = True) -> np.ndarray:
        """Normalised coordinates -> pixels, applying the lens distortion unless ``distorted=False``."""
        xn = self.distort(xn) if distorted else np.asarray(xn, dtype=np.float64)
        return np.c_[self.fx * xn[:, 0] + self.cx, self.fy * xn[:, 1] + self.cy]

    def undistort_pixels(self, pixels: np.ndarray) -> np.ndarray:
        """Observed pixels -> the pixels an ideal pinhole camera with the same K would see."""
        return self.denormalize(self.normalize(pixels), distorted=False)


def rotvec_to_matrix(rvec: np.ndarray) -> np.ndarray:
    return Rotation.from_rotvec(np.asarray(rvec, dtype=np.float64)).as_matrix()


def matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(R).as_rotvec()


def look_at(center: np.ndarray, target: np.ndarray, up=(0.0, -1.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """World-to-camera (R, t) for a camera at ``center`` looking at ``target``.

    ``up`` is the world up direction (default -y, matching the +y-down image convention);
    the camera frame is +x right, +y down (image rows), +z forward.
    """
    z = np.asarray(target, float) - np.asarray(center, float)
    z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(up, float))  # camera +x = right, so +y = z × x points "down"
    if np.linalg.norm(x) < 1e-8:
        x = np.cross(z, [1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ np.asarray(center, float)


@dataclass
class Camera:
    """A posed camera (extrinsics R, t) with intrinsics."""

    intrinsics: Intrinsics
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    t: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t

    @property
    def P(self) -> np.ndarray:
        """3x4 projection matrix K [R | t]."""
        return self.intrinsics.K @ np.c_[self.R, self.t]

    def to_camera(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, float) @ self.R.T + self.t

    def project(self, X: np.ndarray) -> np.ndarray:
        """World points (N, 3) -> pixels (N, 2), including lens distortion."""
        xc = self.to_camera(X)
        return self.intrinsics.denormalize(xc[:, :2] / xc[:, 2:3])

    def depth(self, X: np.ndarray) -> np.ndarray:
        return self.to_camera(X)[:, 2]
