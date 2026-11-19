"""Plane-sweep multi-view stereo on rendered views with exact geometry."""

import numpy as np
import pytest

from structura.camera import Camera, Intrinsics, look_at
from structura.dense import MVSConfig, backproject, densify, depth_range, scaled_camera, select_neighbors
from structura.synthetic import render_textured_views


def _surface_distance(P):
    """Distance to the rendered scene: box [-1,1]x[-2,0]x[-1,1] on the floor y = 0 (|x|,|z| <= 4)."""
    q = np.abs(P - [0, -1, 0]) - 1
    box = np.abs(np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0))
    floor = np.where((np.abs(P[:, 0]) <= 4) & (np.abs(P[:, 2]) <= 4), np.abs(P[:, 1]), np.inf)
    return np.minimum(box, floor)


@pytest.fixture(scope="module")
def dense():
    imgs, cams = render_textured_views(num_views=8, image_size=(480, 360))
    rng = np.random.default_rng(0)
    sparse = np.vstack([np.c_[rng.uniform(-3, 3, 300), np.zeros(300), rng.uniform(-3, 3, 300)],
                        rng.uniform([-1, -2, -1], [1, 0, 1], (300, 3))])  # fmt: skip
    cfg = MVSConfig(num_depths=64, downscale=2)
    return densify(dict(enumerate(imgs)), cams, sparse, cfg), imgs


def test_dense_points_lie_on_the_true_surfaces(dense):
    (pts, cols, maps), _ = dense
    assert len(maps) == 8
    assert len(pts) > 20_000  # far denser than the sparse model
    d = _surface_distance(pts)
    assert np.median(d) < 0.03
    assert (d < 0.08).mean() > 0.85
    assert cols.dtype == np.uint8 and cols.shape == pts.shape


def test_depth_maps_cover_textured_pixels(dense):
    (_, _, maps), _ = dense
    for dm in maps.values():
        assert (dm.depth > 0).mean() > 0.2
        assert dm.depth.shape == (180, 240)


def test_fusion_consistency_removes_points():
    imgs, cams = render_textured_views(num_views=6, image_size=(320, 240))
    sparse = np.random.default_rng(1).uniform([-1, -2, -1], [1, 0, 1], (200, 3))
    loose, _, _ = densify(dict(enumerate(imgs)), cams, sparse, MVSConfig(num_depths=48, min_consistent=0))
    strict, _, _ = densify(dict(enumerate(imgs)), cams, sparse, MVSConfig(num_depths=48, min_consistent=2))
    assert 0 < len(strict) < len(loose)
    assert np.median(_surface_distance(strict)) <= np.median(_surface_distance(loose)) + 1e-9


def test_backproject_inverts_projection():
    K = Intrinsics(100, 100, 32, 24, 64, 48)
    R, t = look_at(np.array([0.0, -1.0, -5.0]), np.zeros(3))
    cam = Camera(K, R, t)
    from structura.dense import DepthMap

    depth = np.full((48, 64), 4.0)
    X, vu = backproject(DepthMap(depth, np.ones_like(depth), cam))
    uv = cam.project(X)
    np.testing.assert_allclose(uv, vu[:, ::-1], atol=1e-9)
    np.testing.assert_allclose(cam.depth(X), 4.0)


def test_helpers():
    K = Intrinsics(200, 200, 99.5, 49.5, 200, 100)
    cams = {}
    for i, a in enumerate(np.linspace(0, 1.2, 5)):
        R, t = look_at(np.array([5 * np.sin(a), -1, -5 * np.cos(a)]), np.zeros(3))
        cams[i] = Camera(K, R, t)
    assert select_neighbors(cams, 0, 2) == [1, 2]
    half = scaled_camera(cams[0], 0.5).intrinsics
    assert (half.width, half.height) == (100, 50) and half.cx == pytest.approx(49.5)
    lo, hi = depth_range(cams[0], np.random.default_rng(0).uniform(-1, 1, (100, 3)), 0.1)
    assert 0 < lo < 5 < hi
    assert depth_range(cams[0], np.zeros((0, 3)), 0.1) is None
