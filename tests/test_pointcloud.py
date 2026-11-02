import numpy as np
import pytest

from structura.camera import rotvec_to_matrix
from structura.pointcloud import (
    chamfer_distance,
    estimate_normals,
    icp,
    remove_statistical_outliers,
    segment_plane,
    voxel_downsample,
)


def sphere(n=2000, seed=0):
    v = np.random.default_rng(seed).normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_voxel_downsample_reduces_and_averages():
    pts = np.array([[0.01, 0.01, 0.01], [0.02, 0.02, 0.02], [1.5, 1.5, 1.5]])
    out, cols = voxel_downsample(pts, 1.0, colors=np.array([[0, 0, 0], [100, 100, 100], [50, 50, 50]]))
    assert len(out) == 2
    assert [50, 50, 50] in cols.tolist()


def test_normals_on_sphere_point_outward():
    pts = sphere()
    n = estimate_normals(pts, k=12, viewpoint=np.array([0.0, 0.0, 0.0]))
    # oriented towards the centre -> anti-parallel to the position vector
    assert np.mean(np.sum(n * -pts, axis=1)) > 0.98


def test_statistical_outlier_removal():
    pts = np.vstack([sphere(), np.random.default_rng(1).uniform(5, 10, (20, 3))])
    mask = remove_statistical_outliers(pts)
    assert mask[:2000].mean() > 0.97 and not mask[2000:].any()


@pytest.mark.parametrize("p2p", [False, True])
def test_icp_recovers_rigid_transform(p2p):
    rng = np.random.default_rng(0)
    target = np.vstack([sphere(1500), np.c_[rng.uniform(-2, 2, (800, 2)), -np.ones(800)]])  # sphere + floor
    R = rotvec_to_matrix([0.05, -0.08, 0.1])
    t = np.array([0.1, -0.05, 0.08])
    source = (target - t) @ R  # inverse transform, so ICP should find (R, t)
    res = icp(source, target, max_distance=0.5, point_to_plane=p2p, max_iterations=100)
    np.testing.assert_allclose(res.R, R, atol=1e-3)
    np.testing.assert_allclose(res.t, t, atol=1e-3)
    assert res.fitness > 0.99


def test_plane_segmentation():
    rng = np.random.default_rng(0)
    plane = np.c_[rng.uniform(-1, 1, (500, 2)), 0.3 + rng.normal(0, 0.002, 500)]
    clutter = rng.uniform(-1, 1, (200, 3))
    model, mask = segment_plane(np.vstack([plane, clutter]), threshold=0.01)
    assert abs(abs(model[2]) - 1) < 1e-3 and abs(abs(model[3]) - 0.3) < 1e-2
    assert mask[:500].mean() > 0.99


def test_chamfer_distance_zero_for_identical():
    pts = sphere(300)
    assert chamfer_distance(pts, pts) == pytest.approx(0.0)
    assert chamfer_distance(pts, pts + [0.1, 0, 0]) > 0.05
