import numpy as np
import pytest

from structura import IncrementalSfM
from structura.geometry import umeyama
from structura.io import export_colmap_text, read_ply, write_ply
from structura.synthetic import make_scene
from structura.tracks import build_tracks


def test_tracks_merge_chains_and_drop_conflicts():
    matches = {(0, 1): np.array([[0, 5], [1, 6]]), (1, 2): np.array([[5, 9], [6, 3]]), (0, 2): np.array([[1, 4]])}
    tracks = build_tracks(matches)
    assert {0: 0, 1: 5, 2: 9} in tracks
    # (0,1)->(1,6)->(2,3) and (0,1)->(2,4) put two keypoints of image 2 in one track -> dropped
    assert all(not (t.get(0) == 1) for t in tracks)


@pytest.fixture(scope="module")
def reconstruction():
    scene = make_scene(num_cameras=8, points_per_object=150, noise_px=0.5, outlier_ratio=0.15, seed=3)
    sfm = IncrementalSfM(scene.intrinsics, scene.keypoints, scene.matches, colors=scene.keypoint_colors)
    return scene, sfm, sfm.run()


def test_incremental_sfm_recovers_cameras(reconstruction):
    scene, _, rec = reconstruction
    assert len(rec.cameras) == 8
    ids = sorted(rec.cameras)
    est = np.array([rec.cameras[i].center for i in ids])
    gt = np.array([scene.cameras[i].center for i in ids])
    s, R, t = umeyama(est, gt)
    err = np.linalg.norm(est @ (s * R).T + t - gt, axis=1)
    assert err.max() < 0.02 * np.linalg.norm(gt, axis=1).mean()
    assert rec.mean_reprojection_error(scene.keypoints) < 1.5


def test_bundle_adjustment_never_increases_error(reconstruction):
    _, sfm, _ = reconstruction
    assert all(r.final_rmse <= r.initial_rmse + 1e-6 for r in sfm.ba_history)


def test_ply_roundtrip(tmp_path):
    pts = np.random.default_rng(0).normal(size=(50, 3))
    cols = np.random.default_rng(1).integers(0, 255, (50, 3)).astype(np.uint8)
    for binary in (True, False):
        path = tmp_path / f"c_{binary}.ply"
        write_ply(str(path), pts, cols, binary=binary)
        p2, c2 = read_ply(str(path))
        np.testing.assert_allclose(p2, pts, atol=1e-5)
        np.testing.assert_array_equal(c2, cols)


def test_colmap_export(reconstruction, tmp_path):
    scene, _, rec = reconstruction
    export_colmap_text(rec, scene.keypoints, str(tmp_path))
    lines = [ln for ln in (tmp_path / "points3D.txt").read_text().splitlines() if not ln.startswith("#")]
    assert len(lines) == rec.num_points
    img_lines = [ln for ln in (tmp_path / "images.txt").read_text().splitlines() if not ln.startswith("#")]
    assert len(img_lines) == 2 * len(rec.cameras)
    # POINT2D_IDX in points3D.txt must index the per-image 2-D point list
    first = lines[0].split()
    img_id, p2d_idx = int(first[8]), int(first[9])
    pts2d = img_lines[2 * sorted(rec.cameras).index(img_id - 1) + 1].split()
    assert int(pts2d[3 * p2d_idx + 2]) == 1


def test_image_based_pipeline_on_textured_renders():
    """Pixels -> SIFT -> matching -> verification -> SfM, on photo-like renders."""
    from structura import Intrinsics
    from structura.features import detect, match_all
    from structura.synthetic import render_textured_views

    images, gt = render_textured_views(num_views=6, arc=np.pi * 0.5)
    feats = {i: detect(im, max_features=4000) for i, im in enumerate(images)}
    intr = {i: Intrinsics.from_image_size(im.shape[1], im.shape[0], 0.9) for i, im in enumerate(images)}
    rec = IncrementalSfM(intr, {i: f.keypoints for i, f in feats.items()}, match_all(feats)).run()
    assert len(rec.cameras) == 6 and rec.num_points > 200
    ids = sorted(rec.cameras)
    est = np.array([rec.cameras[i].center for i in ids])
    ref = np.array([gt[i].center for i in ids])
    s, R, t = umeyama(est, ref)
    assert np.linalg.norm(est @ (s * R).T + t - ref, axis=1).max() < 0.1  # scene radius is ~6
