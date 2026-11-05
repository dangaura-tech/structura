"""Fuse dense 2-D vision-language features into a 3-D point cloud and query it with text.

For every registered view, points are projected into the image, occluded points are removed
with a coarse z-buffer, and the view's dense feature map is sampled bilinearly at the
projections. A point's feature is the (visibility-weighted) mean of its L2-normalised
samples over all views, re-normalised - the multi-view fusion used by OpenScene / LERF-style
pipelines, here on a sparse SfM reconstruction.

Relevance of a text query follows LERF (Kerr et al., ICCV 2023): for each canonical negative
phrase n, compute softmax([s(q), s(n)] / τ)[0] with cosine similarities s, and take the
minimum over negatives. This is far more robust than raw cosine similarity, whose scale
varies between prompts.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ..camera import Camera
from .encoders import ImageTextEncoder, _normalize

CANONICAL_NEGATIVES = ("object", "things", "stuff", "texture")


def visible_mask(points: np.ndarray, cam: Camera, cell: int = 4, tolerance: float = 0.02) -> np.ndarray:
    """Points in front of the camera, inside the image and not occluded (coarse z-buffer).

    Image pixels are grouped into ``cell x cell`` bins; a point is visible if its depth is
    within ``(1 + tolerance)`` of the closest depth in its bin.
    """
    k = cam.intrinsics
    w, h = int(k.width or 2 * k.cx), int(k.height or 2 * k.cy)
    z = cam.depth(points)
    vis = z > 1e-6
    uv = np.full((len(points), 2), -1.0)
    uv[vis] = cam.project(points[vis])
    vis &= (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    idx = np.flatnonzero(vis)
    if not len(idx):
        return vis
    cx = (uv[idx, 0] // cell).astype(np.int64)
    cy = (uv[idx, 1] // cell).astype(np.int64)
    key = cy * (w // cell + 1) + cx
    zbuf = np.full(int(key.max()) + 1, np.inf)
    np.minimum.at(zbuf, key, z[idx])
    vis[idx] = z[idx] <= zbuf[key] * (1.0 + tolerance)
    return vis


def bilinear_sample(feature_map: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sample an (h, w, D) map at continuous coordinates given in feature-map pixels (centres at .5)."""
    h, w = feature_map.shape[:2]
    x = np.clip(x - 0.5, 0, w - 1)
    y = np.clip(y - 0.5, 0, h - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    ax, ay = (x - x0)[:, None], (y - y0)[:, None]
    top = feature_map[y0, x0] * (1 - ax) + feature_map[y0, x1] * ax
    bottom = feature_map[y1, x0] * (1 - ax) + feature_map[y1, x1] * ax
    return top * (1 - ay) + bottom * ay


def fuse_point_features(
    points: np.ndarray,
    cameras: Dict[int, Camera],
    images: Dict[int, np.ndarray],
    encoder: ImageTextEncoder,
    cell: int = 4,
    tolerance: float = 0.02,
    feature_maps: Optional[Dict[int, np.ndarray]] = None,
    tracks: Optional[Sequence[Dict[int, int]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(features (P, D) L2-normalised, views (P,) number of observing views)``.

    Visibility: if SfM ``tracks`` are given (one ``{image id: keypoint id}`` dict per point,
    e.g. ``Reconstruction.point_tracks``), a point is sampled exactly in the images that
    observed it - the most reliable visibility signal for a sparse cloud. Otherwise a coarse
    z-buffer over the points themselves is used, which can let points behind sparsely
    sampled surfaces through.

    ``feature_maps`` can pass precomputed dense features (image id -> (h, w, D)); otherwise
    they are computed with ``encoder.encode_image_dense``. Points seen by no view get a zero
    feature (and therefore zero relevance for any query).
    """
    observed = None
    if tracks is not None:
        if len(tracks) != len(points):
            raise ValueError("tracks must have one entry per point")
        observed = {i: np.array([i in t for t in tracks], bool) for i in cameras}
    acc = np.zeros((len(points), encoder.dim))
    views = np.zeros(len(points), int)
    for img_id, cam in cameras.items():
        if img_id not in images and not (feature_maps and img_id in feature_maps):
            continue
        fmap = (
            feature_maps[img_id]
            if feature_maps and img_id in feature_maps
            else encoder.encode_image_dense(images[img_id])
        )
        if observed is not None:
            vis = observed[img_id] & (cam.depth(points) > 1e-6)
        else:
            vis = visible_mask(points, cam, cell, tolerance)
        if not vis.any():
            continue
        k = cam.intrinsics
        w, h = (k.width or 2 * k.cx), (k.height or 2 * k.cy)
        uv = cam.project(points[vis])
        fh, fw = fmap.shape[:2]
        f = bilinear_sample(fmap, uv[:, 0] * fw / w, uv[:, 1] * fh / h)
        acc[vis] += _normalize(f)
        views[vis] += 1
    feats = np.zeros_like(acc)
    seen = views > 0
    feats[seen] = _normalize(acc[seen])
    return feats, views


def relevance(
    point_features: np.ndarray,
    encoder: ImageTextEncoder,
    query: str,
    negatives: Sequence[str] = CANONICAL_NEGATIVES,
    temperature: float = 0.1,
) -> np.ndarray:
    """LERF relevance in [0, 1] of each point for a text query (0.5 = indistinguishable from negatives)."""
    text = encoder.encode_text([query, *negatives])
    sims = point_features @ text.T  # (P, 1 + K)
    q, n = sims[:, :1], sims[:, 1:]
    pair = np.exp(q / temperature) / (np.exp(q / temperature) + np.exp(n / temperature))
    rel = pair.min(axis=1)
    rel[np.linalg.norm(point_features, axis=1) == 0] = 0.0
    return rel


def segment(point_features: np.ndarray, encoder: ImageTextEncoder, labels: Sequence[str]) -> np.ndarray:
    """Open-vocabulary segmentation: index of the most similar label per point (-1 if unseen)."""
    text = encoder.encode_text(list(labels))
    out = (point_features @ text.T).argmax(axis=1)
    out[np.linalg.norm(point_features, axis=1) == 0] = -1
    return out


def heat_colors(scores: np.ndarray, base: Optional[np.ndarray] = None, threshold: float = 0.5) -> np.ndarray:
    """Colour points by relevance: dimmed original colour below ``threshold``, hot colours above."""
    s = np.clip((scores - threshold) / max(1 - threshold, 1e-6), 0, 1)
    hot = np.stack([255 * np.ones_like(s), 80 + 175 * s, 40 * (1 - s)], -1)
    dim = (base.astype(float) * 0.35) if base is not None else np.full((len(s), 3), 60.0)
    out = np.where((scores > threshold)[:, None], hot, dim)
    return out.astype(np.uint8)
