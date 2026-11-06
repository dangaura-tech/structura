"""Image retrieval for choosing which image pairs to match.

Exhaustive matching costs ``O(N^2)`` pairs, most of which do not overlap in large unordered
collections. Following the retrieval stage of modern SfM pipelines (e.g. hloc), each image is
summarised by a single L2-normalised global descriptor; every image is then matched only
against its ``k`` most similar images (optionally plus its temporal neighbours for video).

Two descriptors are provided:

* :class:`BagOfVisualWords`, training-free: a TF-IDF bag of visual words over the SIFT/ORB
  descriptors the pipeline extracts anyway (a flat version of the vocabulary-tree retrieval
  of Nister and Stewenius, 2006).
* :class:`DinoV2Descriptor`, a self-supervised ViT (Oquab et al., 2024) loaded lazily from
  the Hugging Face hub (``pip install structura[retrieval]``). Its patch tokens are
  aggregated with generalised-mean (GeM) pooling, which is markedly more robust to
  viewpoint and illumination changes for landmark-style collections.

Any (N, D) array of global descriptors can be passed to :func:`retrieve_pairs`.
"""

from __future__ import annotations

from typing import Iterable, List, Protocol, Sequence, Set, Tuple

import cv2
import numpy as np


class GlobalDescriptor(Protocol):
    def describe(self, images_bgr: Sequence[np.ndarray]) -> np.ndarray: ...


def _l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


class BagOfVisualWords:
    """Training-free retrieval from the local features SfM extracts anyway (Sivic and Zisserman, 2003).

    A visual vocabulary is learnt by k-means on a sample of the collection's descriptors; each
    image becomes a TF-IDF weighted histogram of visual words (with square-root / Hellinger
    normalisation, which suppresses bursty repeated texture), and images are compared by cosine
    similarity. No images are needed, only the per-image descriptor arrays.
    """

    def __init__(self, num_words: int = 512, sample: int = 50_000, iterations: int = 20, seed: int = 0):
        self.num_words, self.sample, self.iterations, self.seed = num_words, sample, iterations, seed
        self.vocabulary: np.ndarray | None = None
        self.idf: np.ndarray | None = None

    @staticmethod
    def _as_float(desc: np.ndarray) -> np.ndarray:
        if desc.dtype == np.uint8:  # binary (ORB) descriptors -> bits, so L2 approximates Hamming
            return np.unpackbits(desc, axis=1).astype(np.float32)
        d = desc.astype(np.float32)
        return np.sqrt(np.abs(d) / (np.abs(d).sum(1, keepdims=True) + 1e-6))  # RootSIFT

    def _assign(self, desc: np.ndarray) -> np.ndarray:
        vocab = self.vocabulary
        d2 = (desc**2).sum(1)[:, None] - 2 * desc @ vocab.T + (vocab**2).sum(1)[None]
        return np.argmin(d2, axis=1)

    def fit(self, descriptors: Sequence[np.ndarray]) -> "BagOfVisualWords":
        rng = np.random.default_rng(self.seed)
        pool = np.concatenate([self._as_float(d) for d in descriptors if len(d)])
        if len(pool) > self.sample:
            pool = pool[rng.choice(len(pool), self.sample, replace=False)]
        k = min(self.num_words, len(pool))
        cv2.setRNGSeed(self.seed)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, self.iterations, 1e-3)
        _, _, centers = cv2.kmeans(pool.astype(np.float32), k, None, criteria, 1, cv2.KMEANS_PP_CENTERS)
        self.vocabulary = centers
        words = [np.unique(self._assign(self._as_float(d))) if len(d) else np.zeros(0, int) for d in descriptors]
        df = np.bincount(np.concatenate(words), minlength=k) if words else np.zeros(k)
        self.idf = np.log((len(descriptors) + 1) / (df + 1)) + 1.0
        return self

    def transform(self, descriptors: Sequence[np.ndarray]) -> np.ndarray:
        if self.vocabulary is None:
            raise RuntimeError("call fit() first")
        k = len(self.vocabulary)
        out = np.zeros((len(descriptors), k))
        for i, d in enumerate(descriptors):
            if len(d):
                tf = np.bincount(self._assign(self._as_float(d)), minlength=k).astype(float)
                out[i] = np.sqrt(tf / tf.sum()) * self.idf
        return _l2(out)

    def fit_transform(self, descriptors: Sequence[np.ndarray]) -> np.ndarray:
        return self.fit(descriptors).transform(descriptors)


def gem(tokens, p: float = 3.0, eps: float = 1e-6):
    """Generalised-mean pooling over the token axis of a (B, T, D) tensor."""
    return tokens.clamp(min=eps).pow(p).mean(dim=1).pow(1.0 / p)


class DinoV2Descriptor:
    """DINOv2 global descriptor (CLS token concatenated with GeM-pooled patch tokens).

    The model is loaded on first use, so importing this module never needs PyTorch.
    """

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self, model_name: str = "facebook/dinov2-small", size: int = 224, device: str | None = None,
                 batch_size: int = 16, gem_p: float = 3.0, model=None):  # fmt: skip
        self.model_name, self.size, self.batch_size, self.gem_p = model_name, size, batch_size, gem_p
        self._model, self._device = model, device

    def _load(self):
        import torch

        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        if self._model is None:
            from transformers import AutoModel

            self._model = AutoModel.from_pretrained(self.model_name)
        self._model = self._model.to(self._device).eval()
        return torch

    def _preprocess(self, images_bgr, torch):
        batch = []
        for im in images_bgr:
            rgb = cv2.cvtColor(cv2.resize(im, (self.size, self.size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            x = (rgb.astype(np.float32) / 255.0 - self.MEAN) / self.STD
            batch.append(x.transpose(2, 0, 1))
        return torch.from_numpy(np.stack(batch).astype(np.float32)).to(self._device)

    def describe(self, images_bgr: Sequence[np.ndarray]) -> np.ndarray:
        torch = self._load()
        out = []
        with torch.no_grad():
            for s in range(0, len(images_bgr), self.batch_size):
                x = self._preprocess(images_bgr[s : s + self.batch_size], torch)
                tokens = self._model(pixel_values=x).last_hidden_state  # (B, 1 + T, D)
                cls = torch.nn.functional.normalize(tokens[:, 0], dim=-1)
                patches = torch.nn.functional.normalize(gem(tokens[:, 1:], self.gem_p), dim=-1)
                out.append(torch.cat([cls, patches], dim=-1).cpu().numpy())
        return _l2(np.concatenate(out))


def sequential_pairs(n: int, window: int) -> Set[Tuple[int, int]]:
    """Pairs of frames at most ``window`` apart (video or ordered captures)."""
    return {(i, j) for i in range(n) for j in range(i + 1, min(n, i + window + 1))}


def exhaustive_pairs(n: int) -> Set[Tuple[int, int]]:
    return {(i, j) for i in range(n) for j in range(i + 1, n)}


def retrieve_pairs(
    descriptors: np.ndarray,
    top_k: int = 10,
    min_similarity: float = -1.0,
    window: int = 0,
    ids: Iterable[int] | None = None,
) -> List[Tuple[int, int]]:
    """Image pairs to match: each image with its ``top_k`` nearest neighbours by cosine similarity.

    The relation is symmetrised (a pair is kept if either image retrieves the other), pairs below
    ``min_similarity`` are dropped, and ``window > 0`` additionally adds sequential neighbours.
    Returned pairs are ``(i, j)`` with ``i < j`` in terms of ``ids`` (default ``0..N-1``).
    """
    d = _l2(np.asarray(descriptors, dtype=np.float64))
    n = len(d)
    ids = list(range(n)) if ids is None else list(ids)
    if len(ids) != n:
        raise ValueError("ids and descriptors differ in length")
    sim = d @ d.T
    np.fill_diagonal(sim, -np.inf)
    k = min(top_k, n - 1)
    pairs: Set[Tuple[int, int]] = set()
    if k > 0:
        nn = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        for a in range(n):
            for b in nn[a]:
                if sim[a, b] >= min_similarity:
                    pairs.add((min(a, b), max(a, b)))
    if window > 0:
        pairs |= sequential_pairs(n, window)
    return sorted((min(ids[a], ids[b]), max(ids[a], ids[b])) for a, b in pairs)


def pair_recall(pairs: Iterable[Tuple[int, int]], positives: Iterable[Tuple[int, int]]) -> float:
    """Fraction of ground-truth overlapping pairs that the selection keeps."""
    pos = {tuple(sorted(p)) for p in positives}
    sel = {tuple(sorted(p)) for p in pairs}
    return len(pos & sel) / max(len(pos), 1)
