"""Language-driven 3-D scene understanding: fuse dense vision-language features into points."""

from .encoders import ImageTextEncoder, MockColorEncoder, OpenCLIPEncoder
from .fusion import fuse_point_features, relevance, segment, visible_mask

__all__ = [
    "ImageTextEncoder",
    "OpenCLIPEncoder",
    "MockColorEncoder",
    "fuse_point_features",
    "visible_mask",
    "relevance",
    "segment",
]
