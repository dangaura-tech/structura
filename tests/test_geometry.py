import numpy as np
import pytest

from structura.bundle import bundle_adjust
from structura.camera import Camera, Intrinsics, look_at, matrix_to_rotvec, rotvec_to_matrix
from structura.geometry import absolute_pose, relative_pose, triangulate_dlt, umeyama

K = Intrinsics(800, 800, 320, 240, 640, 480)


def two_cameras():
    c1 = Camera(K, *look_at([0, 0, -5], [0, 0, 0]))
    c2 = Camera(K, *look_at([1.5, -0.3, -4.8], [0, 0, 0]))
    return c1, c2


def cloud(n=200, seed=0):
    return np.random.default_rng(seed).uniform(-1, 1, (n, 3))


def test_projection_and_center():
    c1, _ = two_cameras()
    np.testing.assert_allclose(c1.center, [0, 0, -5], atol=1e-12)
    np.testing.assert_allclose(c1.project(np.zeros((1, 3))), [[320, 240]], atol=1e-9)


def test_rotvec_roundtrip():
    r = np.array([0.1, -0.4, 0.25])
    np.testing.assert_allclose(matrix_to_rotvec(rotvec_to_matrix(r)), r)


def test_triangulation_is_exact_without_noise():
    c1, c2 = two_cameras()
    X = cloud()
    np.testing.assert_allclose(triangulate_dlt(c1.P, c2.P, c1.project(X), c2.project(X)), X, atol=1e-8)


def test_relative_pose_recovers_rotation_and_direction():
    c1, c2 = two_cameras()
    X = cloud(300)
    res = relative_pose(c1.project(X), c2.project(X), K)
    R_rel = c2.R @ c1.R.T
    t_rel = c2.t - R_rel @ c1.t
    np.testing.assert_allclose(res.R, R_rel, atol=1e-5)
    np.testing.assert_allclose(res.t, t_rel / np.linalg.norm(t_rel), atol=1e-4)
    assert res.num_inliers > 290


def test_pnp_with_outliers():
    _, c2 = two_cameras()
    X = cloud(300)
    x = c2.project(X)
    x[:30] += np.random.default_rng(1).uniform(50, 100, (30, 2))  # 10 % gross outliers
    R, t, mask = absolute_pose(X, x, K)
    np.testing.assert_allclose(R, c2.R, atol=1e-6)
    np.testing.assert_allclose(t, c2.t, atol=1e-5)
    assert not mask[:30].any() and mask[30:].all()


def test_umeyama_similarity():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(50, 3))
    R = rotvec_to_matrix([0.3, 0.2, -0.1])
    dst = 2.5 * src @ R.T + [1, 2, 3]
    s, R_est, t = umeyama(src, dst)
    assert s == pytest.approx(2.5)
    np.testing.assert_allclose(R_est, R, atol=1e-10)
    np.testing.assert_allclose(t, [1, 2, 3], atol=1e-10)


def test_bundle_adjustment_reduces_error():
    rng = np.random.default_rng(0)
    cams = {
        i: Camera(K, *look_at([5 * np.sin(a), -1, -5 * np.cos(a)], [0, 0, 0])) for i, a in enumerate([0, 0.3, 0.6, 0.9])
    }
    X = cloud(150)
    cam_idx = np.repeat(list(cams), len(X))
    pt_idx = np.tile(np.arange(len(X)), len(cams))
    obs = np.vstack([cams[i].project(X) for i in cams]) + rng.normal(0, 0.3, (len(cam_idx), 2))
    noisy = {i: Camera(K, c.R.copy(), c.t.copy()) for i, c in cams.items()}
    for i in (2, 3):  # cameras 0 and 1 are held fixed (gauge + scale)
        noisy[i].t = noisy[i].t + rng.normal(0, 0.05, 3)
    pts = X + rng.normal(0, 0.03, X.shape)
    res = bundle_adjust(noisy, pts, cam_idx, pt_idx, obs, fixed=[0, 1])
    assert res.final_rmse < 0.5 < res.initial_rmse
    for i in (2, 3):
        np.testing.assert_allclose(noisy[i].center, cams[i].center, atol=0.02)


def test_look_at_produces_upright_right_handed_camera():
    R, t = look_at([0.0, -3.0, -5.0], [0.0, 0.0, 0.0])
    cam = Camera(K, R, t)
    assert np.linalg.det(R) == pytest.approx(1.0)
    above, below = np.array([[0.0, -1.0, 0.0]]), np.array([[0.0, 1.0, 0.0]])  # -y is up
    assert cam.project(above)[0, 1] < cam.project(below)[0, 1]  # higher in the world = higher in the image
    left, right = np.array([[-1.0, 0.0, 0.0]]), np.array([[1.0, 0.0, 0.0]])
    assert cam.project(left)[0, 0] < cam.project(right)[0, 0]
