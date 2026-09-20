"""`generate()` is the documented Python API, so it must get the wrapper's protections.

`WeeImageToImagePipeline.generate` already routes through `self(...)`;
`WeeTextToImagePipeline.generate` used to call `self._pipeline(...)` directly, which
skipped every kwarg default, the VRAM overhead estimate and the OOM recovery path.
"""

from __future__ import annotations

import pytest
import torch

from weellm.pipelines.image.weeimagetoimagepipeline import WeeImageToImagePipeline
from weellm.pipelines.image.weetexttoimagepipeline import WeeTextToImagePipeline


class _Output:
    def __init__(self):
        self.images = ["image"]


class _KvCachePipeline:
    """A stand-in for QwenImage21Pipeline: accepts use_kv_cache, records what it got."""

    vae_scale_factor = 16

    def __init__(self):
        self.received = None

    def __call__(self, prompt=None, image=None, generator=None, height=None, width=None,
                 use_kv_cache=True, **kwargs):
        self.received = dict(
            prompt=prompt, image=image, height=height, width=width,
            use_kv_cache=use_kv_cache, generator=generator,
        )
        return _Output()


def test_text_to_image_generate_applies_kv_cache_default():
    inner = _KvCachePipeline()
    pipe = WeeTextToImagePipeline(inner)

    result = pipe.generate("a cat", height=64, width=64)

    assert result == "image"
    assert inner.received["use_kv_cache"] is False, "generate() bypassed the wrapper"


def test_text_to_image_generate_honours_explicit_opt_in():
    inner = _KvCachePipeline()
    pipe = WeeTextToImagePipeline(inner)

    pipe.generate("a cat", height=64, width=64, use_kv_cache=True)

    assert inner.received["use_kv_cache"] is True


def test_image_to_image_generate_applies_kv_cache_default():
    inner = _KvCachePipeline()
    pipe = WeeImageToImagePipeline(inner)

    from PIL import Image

    image = Image.new("RGB", (64, 64))
    pipe.generate("a cat", image=image, height=64, width=64)

    assert inner.received["use_kv_cache"] is False


def test_text_to_image_generate_still_rejects_an_image():
    """Routing through __call__ must keep the T2I/I2I guard rail intact."""
    pipe = WeeTextToImagePipeline(_KvCachePipeline())

    from PIL import Image

    with pytest.raises(ValueError, match="text-to-image pipeline"):
        pipe.generate("a cat", image=Image.new("RGB", (64, 64)))


def test_text_to_image_generate_still_passes_a_seeded_generator():
    inner = _KvCachePipeline()
    pipe = WeeTextToImagePipeline(inner)

    pipe.generate("a cat", height=64, width=64, seed=1234)

    generator = inner.received["generator"]
    assert isinstance(generator, torch.Generator)
    assert generator.initial_seed() == 1234
