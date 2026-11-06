"""Radial distortion model, self-calibrating bundle adjustment and I/O of distorted cameras."""

from dataclasses import replace

import numpy as np
import pytest

from structura import IncrementalSfM, SfMConfig
from structura.camera import Camera, Intrinsics, look_at
from structura.io import export_colmap_text, load_reconstruction, save_reconstruction
from structura.synthetic import make_scene


@pytest.mark.parametrize("k1,k2", [(-0.25, 0.08), (0.15, -0.02), (0.0, 0.0)])
def test_undistort_inverts_distort(k1, k2):
    K = Intrinsics(800, 820, 512, 384, 1024, 768, k1, k2)
    xn = np.random.default_rng(0).uniform(-0.6, 0.6, (500, 2))
    np.testing.assert_allclose(K.undistort(K.distort(xn)), xn, atol=1e-9)
    px = K.denormalize(xn)
    np.testing.assert_allclose(K.normalize(px), xn, atol=1e-9)


def test_projection_applies_distortion_and_undistort_pixels_recovers_pinhole():
    K = Intrinsics(900, 900, 500, 400, 1000, 800, -0.2, 0.05)
    pinhole = replace(K, k1=0.0, k2=0.0)
    R, t = look_at(np.array([0.0, -1.0, -6.0]), np.zeros(3))
    X = np.random.default_rng(1).uniform(-2, 2, (200, 3))
    distorted = Camera(K, R, t).project(X)
    ideal = Camera(pinhole, R, t).project(X)
    assert np.abs(distorted - ideal).max() > 5  # the lens visibly bends the image
    np.testing.assert_allclose(K.undistort_pixels(distorted), ideal, atol=1e-6)


@pytest.fixture(scope="module")
def distorted_runs():
    scene = make_scene(
        num_cameras=10, points_per_object=200, noise_px=0.5, outlier_ratio=0.1, seed=3, distortion=(0.06, 0.0)
    )
    nominal = replace(scene.intrinsics[0], k1=0.0, k2=0.0)  # the lens is unknown to the pipeline
    shared = {i: nominal for i in scene.keypoints}
    runs = {}
    for mode in ("none", "radial"):
        sfm = IncrementalSfM(shared, scene.keypoints, scene.matches, SfMConfig(refine_intrinsics=mode))
        runs[mode] = sfm.run()
    return scene, nominal, runs


def test_self_calibration_reaches_the_noise_floor(distorted_runs):
    scene, _, runs = distorted_runs
    err = {m: r.mean_reprojection_error(scene.keypoints) for m, r in runs.items()}
    # mean |N(0, 0.5^2 I_2)| = 0.627 px: the calibrated model explains the data down to the pixel noise
    assert err["radial"] < 0.7
    assert err["radial"] < err["none"] - 0.1
    assert len(runs["radial"].cameras) == 10


def test_self_calibration_recovers_the_lens(distorted_runs):
    scene, _, runs = distorted_runs
    gt = scene.intrinsics[0]
    est = runs["radial"].cameras[min(runs["radial"].cameras)].intrinsics
    r = np.linspace(0, 0.6, 100)  # radial profile over the image, in pixels
    curve = lambda k: k.fx * r * (1 + k.k1 * r**2 + k.k2 * r**4)  # noqa: E731
    assert np.abs(curve(est) - curve(gt)).max() < 1.0
    assert abs(est.fx - gt.fx) / gt.fx < 0.01


def test_refinement_does_not_mutate_caller_intrinsics_and_keeps_them_shared(distorted_runs):
    _, nominal, runs = distorted_runs
    assert nominal.k1 == 0.0 and nominal.k2 == 0.0
    cams = runs["radial"].cameras
    assert len({id(c.intrinsics) for c in cams.values()}) == 1  # one physical camera, one calibration


def test_distorted_cameras_survive_export_and_npz_roundtrip(tmp_path, distorted_runs):
    scene, _, runs = distorted_runs
    rec = runs["radial"]
    export_colmap_text(rec, scene.keypoints, str(tmp_path / "colmap"))
    line = (tmp_path / "colmap" / "cameras.txt").read_text().splitlines()[1].split()
    assert line[1] == "OPENCV" and len(line) == 12
    save_reconstruction(str(tmp_path / "rec.npz"), rec)
    loaded, _ = load_reconstruction(str(tmp_path / "rec.npz"))
    i = min(rec.cameras)
    assert loaded.cameras[i].intrinsics.k1 == pytest.approx(rec.cameras[i].intrinsics.k1)
