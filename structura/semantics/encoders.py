"""Vision-language encoders producing *dense* (per-patch) image features aligned with text.

``OpenCLIPEncoder`` extracts dense features from a CLIP ViT with the MaskCLIP trick
(Zhou et al., ECCV 2022): in the last transformer block the query-key attention is dropped
and the *value* embedding of every patch token is passed through the output projection,
``ln_post`` and the image projection. The result lives in CLIP's joint image-text space,
so each patch can be compared with text embeddings without any training.

``MockColorEncoder`` implements the same interface with a tiny deterministic colour
vocabulary. It is used in tests and in the synthetic demo, where no pretrained weights are
available.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np

DEFAULT_TEMPLATES = (
    "a photo of a {}.",
    "a photo of the {}.",
    "a close-up photo of a {}.",
    "a rendering of a {}.",
    "there is a {} in the scene.",
)


@runtime_checkable
class ImageTextEncoder(Protocol):
    """Anything that maps images to dense features and text to the same embedding space."""

    dim: int

    def encode_image_dense(self, image_bgr: np.ndarray) -> np.ndarray:
        """(H, W, 3) uint8 BGR image -> (h, w, dim) L2-normalised feature map covering the image."""
        ...

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        """List of strings -> (T, dim) L2-normalised text embeddings."""
        ...


def _normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-12)


# --------------------------------------------------------------------------- OpenCLIP
def interpolate_pos_embed(pos, grid_h: int, grid_w: int):
    """Resize a ViT positional embedding ``(1 + g*g, C)`` to a ``grid_h x grid_w`` patch grid."""
    import torch
    import torch.nn.functional as F

    cls, patches = pos[:1], pos[1:]
    g = int(round(patches.shape[0] ** 0.5))
    if (g, g) == (grid_h, grid_w):
        return pos
    grid = patches.reshape(1, g, g, -1).permute(0, 3, 1, 2)
    grid = F.interpolate(grid, size=(grid_h, grid_w), mode="bicubic", align_corners=False)
    return torch.cat([cls, grid.permute(0, 2, 3, 1).reshape(grid_h * grid_w, -1)], dim=0)


def maskclip_tokens_before_last_block(visual, images):
    """Run an open_clip ``VisionTransformer`` up to (excluding) its last block.

    Returns ``(tokens, grid_h, grid_w)`` with tokens in batch-first layout ``(B, 1 + N, C)``.
    Supports arbitrary input sizes (multiples of the patch size) via positional-embedding
    interpolation.
    """
    import torch

    x = visual.conv1(images)  # (B, C, gh, gw)
    b, c, gh, gw = x.shape
    x = x.reshape(b, c, gh * gw).permute(0, 2, 1)
    cls = visual.class_embedding.to(x.dtype) + torch.zeros(b, 1, c, dtype=x.dtype, device=x.device)
    x = torch.cat([cls, x], dim=1)
    x = x + interpolate_pos_embed(visual.positional_embedding.to(x.dtype), gh, gw)
    x = visual.ln_pre(x)
    batch_first = getattr(visual.transformer, "batch_first", True)
    if not batch_first:
        x = x.permute(1, 0, 2)
    for block in visual.transformer.resblocks[:-1]:
        x = block(x)
    if not batch_first:
        x = x.permute(1, 0, 2)
    return x, gh, gw


def maskclip_dense_features(visual, images):
    """Dense CLIP features ``(B, gh, gw, D)`` (L2-normalised) for a batch of normalised images."""
    import torch.nn.functional as F

    x, gh, gw = maskclip_tokens_before_last_block(visual, images)
    last = visual.transformer.resblocks[-1]
    attn = last.attn
    if not hasattr(attn, "in_proj_weight") or attn.in_proj_weight is None:
        raise NotImplementedError("MaskCLIP needs a packed-QKV nn.MultiheadAttention in the last block")
    c = x.shape[-1]
    y = last.ln_1(x)
    v = F.linear(y, attn.in_proj_weight[2 * c :], attn.in_proj_bias[2 * c :] if attn.in_proj_bias is not None else None)
    v = attn.out_proj(v)
    if hasattr(last, "ls_1"):
        v = last.ls_1(v)
    patches = visual.ln_post(v[:, 1:]) @ visual.proj  # value path only: no residual, no FFN (MaskCLIP)
    patches = F.normalize(patches, dim=-1)
    return patches.reshape(x.shape[0], gh, gw, -1).contiguous()


class OpenCLIPEncoder:
    """Dense OpenCLIP features (MaskCLIP) + prompt-ensembled text embeddings.

    Requires ``pip install -e ".[clip]"``; weights are downloaded by open_clip on first use, e.g.
    ``OpenCLIPEncoder("ViT-B-16", "laion2b_s34b_b88k")``.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-16",
        pretrained: str | None = "laion2b_s34b_b88k",
        device: str = "cpu",
        short_side: int = 448,
        templates: Sequence[str] = DEFAULT_TEMPLATES,
    ) -> None:
        try:
            import open_clip
            import torch
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError('OpenCLIPEncoder requires `pip install -e ".[clip]"`') from e
        self.torch = torch
        self.model = open_clip.create_model(model_name, pretrained=pretrained, device=device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.device = device
        self.short_side = short_side
        self.templates = tuple(templates)
        self.patch = int(self.model.visual.conv1.kernel_size[0])
        self.dim = int(self.model.visual.proj.shape[1])
        cfg = getattr(self.model.visual, "preprocess_cfg", {}) or {}
        self.mean = np.array(cfg.get("mean", (0.48145466, 0.4578275, 0.40821073)), np.float32)
        self.std = np.array(cfg.get("std", (0.26862954, 0.26130258, 0.27577711)), np.float32)

    def _prepare(self, image_bgr: np.ndarray):
        import cv2

        h, w = image_bgr.shape[:2]
        s = self.short_side / min(h, w)
        nh = max(self.patch, int(round(h * s / self.patch)) * self.patch)
        nw = max(self.patch, int(round(w * s / self.patch)) * self.patch)
        rgb = cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_AREA)[:, :, ::-1].astype(np.float32) / 255.0
        rgb = (rgb - self.mean) / self.std
        return self.torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))[None].to(self.device)

    def encode_image_dense(self, image_bgr: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            feats = maskclip_dense_features(self.model.visual, self._prepare(image_bgr))
        return feats[0].float().cpu().numpy()

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        out = []
        with self.torch.no_grad():
            for t in texts:
                prompts = [tpl.format(t) for tpl in self.templates] if self.templates else [t]
                emb = self.model.encode_text(self.tokenizer(prompts).to(self.device)).float()
                emb = self.torch.nn.functional.normalize(emb, dim=-1).mean(0)
                out.append(self.torch.nn.functional.normalize(emb, dim=-1).cpu().numpy())
        return np.stack(out)


# ------------------------------------------------------------------------------ mock
class MockColorEncoder:
    """Deterministic stand-in for a vision-language model, for tests and synthetic demos.

    Image patches are embedded by their soft assignment to a small colour vocabulary; text is
    embedded by the colour words it mentions. A query such as "the red sphere" therefore
    selects red points - which makes fusion and querying verifiable without pretrained weights.
    It does **not** understand shapes or semantics.
    """

    PALETTE = {
        "red": (230, 90, 70),
        "blue": (70, 140, 230),
        "yellow": (240, 200, 60),
        "gray": (120, 120, 120),
        "green": (70, 190, 90),
    }

    def __init__(self, patch: int = 8, sigma: float = 40.0, background_level: float = 45.0) -> None:
        self.patch = patch
        self.sigma = sigma
        self.background_level = background_level
        self.names = list(self.PALETTE)
        self.colors = np.array([self.PALETTE[n] for n in self.names], np.float32)  # RGB
        self.dim = len(self.names) + 1  # + one "generic" dimension

    def _embed_colors(self, rgb: np.ndarray) -> np.ndarray:
        d2 = ((rgb[..., None, :] - self.colors) ** 2).sum(-1)
        w = np.exp(-d2 / (2 * self.sigma**2))
        return _normalize(np.concatenate([w, np.full(w.shape[:-1] + (1,), 0.05)], -1))

    def encode_image_dense(self, image_bgr: np.ndarray) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        gh, gw = max(1, h // self.patch), max(1, w // self.patch)
        crop = image_bgr[: gh * self.patch, : gw * self.patch, ::-1].astype(np.float32)
        blocks = crop.reshape(gh, self.patch, gw, self.patch, 3)
        # average only foreground pixels so that dark background between sparse splats
        # does not pull a patch's colour towards black; empty patches become "generic"
        fg = (blocks.max(axis=-1) > self.background_level)[..., None]
        count = fg.sum(axis=(1, 3))
        mean = (blocks * fg).sum(axis=(1, 3)) / np.maximum(count, 1)
        feats = self._embed_colors(mean)
        generic = _normalize(np.r_[np.zeros(len(self.names)), 1.0])
        feats[count[..., 0] == 0] = generic
        return feats

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        out = []
        for t in texts:
            v = np.zeros(self.dim, np.float32)
            words = t.lower().replace("grey", "gray")
            for k, name in enumerate(self.names):
                if name in words:
                    v[k] = 1.0
            if not v.any():
                v[:] = 1.0  # generic phrases ("object", "stuff") are equally similar to everything
            out.append(_normalize(v))
        return np.stack(out)
