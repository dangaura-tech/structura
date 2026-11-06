"""Pair selection by image retrieval."""

import numpy as np
import pytest

from structura.features import detect
from structura.retrieval import (
    BagOfVisualWords,
    DinoV2Descriptor,
    exhaustive_pairs,
    gem,
    pair_recall,
    retrieve_pairs,
    sequential_pairs,
)
from structura.synthetic import render_textured_views


def test_retrieve_pairs_is_symmetric_and_respects_k():
    rng = np.random.default_rng(0)
    d = rng.normal(size=(12, 16))
    pairs = retrieve_pairs(d, top_k=2)
    assert all(i < j for i, j in pairs)
    assert len(pairs) == len(set(pairs))
    assert 12 <= len(pairs) <= 24
    # every image appears in at least top_k pairs
    deg = np.bincount(np.array(pairs).ravel(), minlength=12)
    assert deg.min() >= 2


def test_retrieve_pairs_threshold_window_and_ids():
    d = np.eye(4)  # orthogonal: similarity 0 everywhere
    assert retrieve_pairs(d, top_k=3, min_similarity=0.5) == []
    assert retrieve_pairs(d, top_k=3, min_similarity=0.5, window=1) == [(0, 1), (1, 2), (2, 3)]
    assert retrieve_pairs(d, top_k=1, window=1, ids=[10, 20, 30, 40])[0] == (10, 20)
    assert sequential_pairs(5, 2) == {(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)}
    assert len(exhaustive_pairs(6)) == 15


@pytest.fixture(scope="module")
def views():
    images, _ = render_textured_views(num_views=14, arc=np.pi * 1.6)
    return images


@pytest.mark.parametrize("method", ["sift", "orb"])
def test_bag_of_words_retrieves_overlapping_views(views, method):
    feats = [detect(im, method, 1500) for im in views]
    desc = BagOfVisualWords(num_words=256).fit_transform([f.descriptors for f in feats])
    np.testing.assert_allclose(np.linalg.norm(desc, axis=1), 1.0, atol=1e-6)
    neighbours = [(i, i + 1) for i in range(len(views) - 1)]
    pairs = retrieve_pairs(desc, top_k=3)
    assert pair_recall(pairs, neighbours) >= 0.9
    assert len(pairs) < len(exhaustive_pairs(len(views))) / 2  # far fewer pairs than exhaustive matching


def test_bag_of_words_requires_fit():
    with pytest.raises(RuntimeError):
        BagOfVisualWords().transform([np.zeros((3, 128), np.float32)])


torch = pytest.importorskip("torch")


class _FakeViT(torch.nn.Module):
    """Stands in for a Hugging Face DINOv2 model: same call signature and output field."""

    def __init__(self, dim=24, patch=16):
        super().__init__()
        self.embed = torch.nn.Conv2d(3, dim, patch, patch)
        self.cls = torch.nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, pixel_values):
        tok = self.embed(pixel_values).flatten(2).transpose(1, 2)
        hidden = torch.cat([self.cls.expand(len(tok), -1, -1), tok], dim=1)
        return type("Out", (), {"last_hidden_state": hidden})()


def test_gem_interpolates_between_mean_and_max():
    x = torch.rand(2, 50, 8) + 0.1
    torch.testing.assert_close(gem(x, p=1.0), x.mean(1))
    assert torch.all(gem(x, p=50.0) <= x.max(1).values + 1e-5)
    assert torch.all(gem(x, p=50.0) > gem(x, p=1.0))


def test_dinov2_descriptor_with_injected_model(views):
    torch.manual_seed(0)
    enc = DinoV2Descriptor(model=_FakeViT(), size=64, device="cpu", batch_size=4)
    d = enc.describe(views[:6])
    assert d.shape == (6, 48)  # CLS (24) + GeM patches (24)
    np.testing.assert_allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(enc.describe(views[:6]), d, atol=1e-6)  # deterministic in eval mode
