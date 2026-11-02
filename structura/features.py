"""Keypoint detection and matching for real images (OpenCV SIFT / ORB)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np


@dataclass
class Features:
    keypoints: np.ndarray  # (K, 2) pixel coordinates
    descriptors: np.ndarray  # (K, D)
    colors: np.ndarray  # (K, 3) RGB at each keypoint


def detect(image_bgr: np.ndarray, method: str = "sift", max_features: int = 8000) -> Features:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if method == "sift":
        det = cv2.SIFT_create(nfeatures=max_features)
    elif method == "orb":
        det = cv2.ORB_create(nfeatures=max_features)
    else:
        raise ValueError("method must be 'sift' or 'orb'")
    kps, desc = det.detectAndCompute(gray, None)
    if desc is None:
        return Features(np.zeros((0, 2)), np.zeros((0, 128), np.float32), np.zeros((0, 3), np.uint8))
    pts = np.array([k.pt for k in kps], dtype=np.float64)
    ij = np.clip(np.round(pts).astype(int), 0, [image_bgr.shape[1] - 1, image_bgr.shape[0] - 1])
    colors = image_bgr[ij[:, 1], ij[:, 0], ::-1].copy()
    return Features(pts, desc, colors)


def match(f1: Features, f2: Features, ratio: float = 0.8, mutual: bool = True) -> np.ndarray:
    """Lowe ratio test + optional mutual nearest-neighbour check. Returns (M, 2) index pairs."""
    if len(f1.descriptors) < 2 or len(f2.descriptors) < 2:
        return np.zeros((0, 2), int)
    norm = cv2.NORM_HAMMING if f1.descriptors.dtype == np.uint8 else cv2.NORM_L2
    bf = cv2.BFMatcher(norm)
    knn = bf.knnMatch(f1.descriptors, f2.descriptors, k=2)
    fwd = {m.queryIdx: m.trainIdx for m, n in (p for p in knn if len(p) == 2) if m.distance < ratio * n.distance}
    if mutual:
        back = bf.match(f2.descriptors, f1.descriptors)
        best_back = {m.queryIdx: m.trainIdx for m in back}
        fwd = {q: t for q, t in fwd.items() if best_back.get(t) == q}
    return np.array(sorted(fwd.items()), dtype=int).reshape(-1, 2)


def match_all(features: Dict[int, Features], pairs: Iterable[Tuple[int, int]] | None = None, **kw):
    """Exhaustive (or user-specified) pairwise matching."""
    ids = sorted(features)
    if pairs is None:
        pairs = [(i, j) for a, i in enumerate(ids) for j in ids[a + 1 :]]
    return {(i, j): m for i, j in pairs if len(m := match(features[i], features[j], **kw))}
