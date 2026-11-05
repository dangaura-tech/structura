import numpy as np
import pytest

from structura.camera import Camera, Intrinsics, look_at
from structura.io import load_reconstruction, save_reconstruction
from structura.reconstruction import Reconstruction
from structura.render import render_frame
from structura.semantics import MockColorEncoder, fuse_point_features, relevance, segment, visible_mask
from structura.semantics.fusion import bilinear_sample, heat_colors
from structura.synthetic import make_scene


def test_mock_encoder_aligns_colours_and_words():
    enc = MockColorEncoder(patch=4)
    img = np.zeros((8, 8, 3), np.uint8)
    img[:, :4] = (70, 90, 230)  # BGR red
    img[:, 4:] = (230, 140, 70)  # BGR blue
    fmap = enc.encode_image_dense(img)
    text = enc.encode_text(["a red thing", "the blue box"])
    sims = fmap.reshape(-1, enc.dim) @ text.T
    assert fmap.shape == (2, 2, enc.dim)
    assert (sims[[0, 2], 0] > sims[[0, 2], 1]).all() and (sims[[1, 3], 1] > sims[[1, 3], 0]).all()
    np.testing.assert_allclose(np.linalg.norm(fmap, axis=-1), 1.0, atol=1e-6)


def test_bilinear_sample_hits_pixel_centres_and_interpolates():
    fmap = np.arange(12, dtype=float).reshape(3, 4, 1)
    np.testing.assert_allclose(bilinear_sample(fmap, np.array([0.5, 3.5]), np.array([0.5, 2.5]))[:, 0], [0, 11])
    assert bilinear_sample(fmap, np.array([1.0]), np.array([0.5]))[0, 0] == pytest.approx(0.5)


def test_visible_mask_handles_occlusion_and_frustum():
    K = Intrinsics(100, 100, 50, 50, 100, 100)
    cam = Camera(K, *look_at([0, 0, -5], [0, 0, 0]))
    pts = np.array(
        [[0, 0, 0], [0, 0, 2], [0, 0, -10], [10, 0, 0]], float
    )  # front, behind front, behind camera, outside
    vis = visible_mask(pts, cam)
    assert vis.tolist() == [True, False, False, False]


@pytest.fixture(scope="module")
def fused_scene():
    scene = make_scene(num_cameras=8, points_per_object=500, seed=2)
    k = next(iter(scene.intrinsics.values()))
    images = {
        i: np.asarray(render_frame(scene.points, scene.colors, cam, (k.width, k.height), point_radius=3.0))[:, :, ::-1]
        for i, cam in scene.cameras.items()
    }
    enc = MockColorEncoder()
    feats, views = fuse_point_features(scene.points, scene.cameras, images, enc)
    return scene, enc, feats, views


def test_query_selects_the_right_object(fused_scene):
    scene, enc, feats, views = fused_scene
    n = 500  # objects are stacked: sphere (red), box (blue), torus (yellow), floor (gray)
    seen = views > 0
    assert seen.mean() > 0.9
    rel = relevance(feats, enc, "the red sphere")
    sphere, rest = np.arange(n), np.arange(n, 4 * n)
    assert (rel[sphere][seen[sphere]] > 0.5).mean() > 0.9
    assert (rel[rest][seen[rest]] > 0.5).mean() < 0.05


def test_open_vocabulary_segmentation(fused_scene):
    scene, enc, feats, views = fused_scene
    labels = segment(feats, enc, ["red", "blue", "yellow", "gray floor"])
    for k in range(4):
        obj = labels[k * 500 : (k + 1) * 500]
        obj = obj[obj >= 0]
        assert (obj == k).mean() > 0.8


def test_tracks_restrict_fusion_to_observing_views(fused_scene):
    scene, enc, _, _ = fused_scene
    pts = scene.points[:3]
    tracks = [{0: 0}, {0: 0, 1: 0}, {}]
    images = {i: np.zeros((768, 1024, 3), np.uint8) for i in scene.cameras}
    _, views = fuse_point_features(pts, scene.cameras, images, enc, tracks=tracks)
    assert views.tolist() == [1, 2, 0]
    with pytest.raises(ValueError):
        fuse_point_features(pts, scene.cameras, images, enc, tracks=tracks[:2])


def test_unseen_points_have_zero_relevance():
    enc = MockColorEncoder()
    feats = np.zeros((2, enc.dim))
    feats[1, 0] = 1.0
    rel = relevance(feats, enc, "red")
    assert rel[0] == 0.0 and rel[1] > 0.5
    assert segment(feats, enc, ["red", "blue"]).tolist() == [-1, 0]


def test_heat_colors_highlight_above_threshold():
    cols = heat_colors(np.array([0.2, 0.9]), np.array([[100, 100, 100], [100, 100, 100]]), threshold=0.5)
    assert cols[0].tolist() == [35, 35, 35] and cols[1, 0] == 255


def test_reconstruction_npz_roundtrip(tmp_path):
    K = Intrinsics(500, 500, 320, 240, 640, 480)
    rec = Reconstruction(
        cameras={3: Camera(K, *look_at([1, 0, -4], [0, 0, 0]))},
        points=np.random.default_rng(0).normal(size=(10, 3)),
        colors=np.full((10, 3), 7, np.uint8),
    )
    save_reconstruction(str(tmp_path / "r.npz"), rec, {3: "/data/img_003.jpg"})
    rec2, names = load_reconstruction(str(tmp_path / "r.npz"))
    assert names == {3: "/data/img_003.jpg"}
    np.testing.assert_allclose(rec2.points, rec.points)
    np.testing.assert_allclose(rec2.cameras[3].R, rec.cameras[3].R)
    assert rec2.cameras[3].intrinsics.width == 640


# ------------------------------------------------------------------- real OpenCLIP graph
open_clip = pytest.importorskip("open_clip")
torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def vit():
    torch.manual_seed(0)
    return open_clip.create_model("ViT-B-16", pretrained=None).eval()


def test_maskclip_prefix_matches_open_clip_forward(vit):
    from structura.semantics.encoders import maskclip_tokens_before_last_block

    captured = {}
    handle = vit.visual.transformer.resblocks[-1].register_forward_pre_hook(lambda m, a: captured.setdefault("x", a[0]))
    img = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        vit.visual(img)
        ours, gh, gw = maskclip_tokens_before_last_block(vit.visual, img)
    handle.remove()
    ref = captured["x"] if vit.visual.transformer.batch_first else captured["x"].permute(1, 0, 2)
    assert (gh, gw) == (14, 14)
    torch.testing.assert_close(ours, ref, atol=1e-5, rtol=1e-5)


def test_maskclip_dense_features_any_resolution(vit):
    from structura.semantics.encoders import maskclip_dense_features

    with torch.no_grad():
        f = maskclip_dense_features(vit.visual, torch.randn(2, 3, 320, 448))
    assert f.shape == (2, 20, 28, 512)
    torch.testing.assert_close(f.norm(dim=-1), torch.ones(2, 20, 28), atol=1e-4, rtol=1e-4)


def test_openclip_encoder_interface_with_random_weights():
    from structura.semantics import ImageTextEncoder, OpenCLIPEncoder

    enc = OpenCLIPEncoder("ViT-B-16", pretrained=None, short_side=224, templates=("a photo of a {}.",))
    assert isinstance(enc, ImageTextEncoder)
    fmap = enc.encode_image_dense(np.zeros((240, 320, 3), np.uint8))
    assert fmap.shape == (14, 19, enc.dim)
    text = enc.encode_text(["a red chair", "a table"])
    assert text.shape == (2, enc.dim)
    np.testing.assert_allclose(np.linalg.norm(text, axis=1), 1.0, atol=1e-5)
