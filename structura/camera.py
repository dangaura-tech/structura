"""Pinhole cameras and rotation utilities.

Conventions (same as OpenCV / COLMAP): a world point ``X`` maps to camera coordinates
``x_c = R X + t``; the camera centre is ``C = -R^T t``; pixels are ``K x_c / z``.
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

    @classmethod
    def from_image_size(cls, width: int, height: int, focal_factor: float = 1.2) -> "Intrinsics":
        """Common prior when EXIF focal length is unknown: f = 1.2 * max(w, h)."""
        f = focal_factor * max(width, height)
        return cls(f, f, width / 2.0, height / 2.0, width, height)

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    def normalize(self, pixels: np.ndarray) -> np.ndarray:
        """Pixels (N, 2) -> normalised image coordinates (N, 2)."""
        p = np.asarray(pixels, dtype=np.float64)
        return np.c_[(p[:, 0] - self.cx) / self.fx, (p[:, 1] - self.cy) / self.fy]


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
        """World points (N, 3) -> pixels (N, 2)."""
        xc = self.to_camera(X)
        k = self.intrinsics
        return np.c_[k.fx * xc[:, 0] / xc[:, 2] + k.cx, k.fy * xc[:, 1] / xc[:, 2] + k.cy]

    def depth(self, X: np.ndarray) -> np.ndarray:
        return self.to_camera(X)[:, 2]
