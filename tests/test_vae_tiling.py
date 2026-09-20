"""T-6: VAE tiling configuration must match whichever attribute convention the VAE uses.

diffusers has two: the legacy ``AutoencoderKL`` exposes a single ``tile_sample_min_size``,
while Wan-style 3D VAEs (including ``AutoencoderKLQwenImage21``) expose separate
``tile_sample_min_height`` / ``tile_sample_min_width`` plus strides. Setting only the
legacy attribute on the latter silently does nothing.
"""

from __future__ import annotations

import pytest

from weellm.pipelines.weebasepipeline import WeeBasePipeline

TILE = 256


class _Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _BaseVae:
    def __init__(self):
        self.tiling_enabled = False

    def enable_tiling(self):
        self.tiling_enabled = True

    def encode(self, *args, **kwargs):
        return "encoded"

    def decode(self, *args, **kwargs):
        return "decoded"


class _LegacyVae(_BaseVae):
    """SD / SDXL style."""

    def __init__(self):
        super().__init__()
        self.tile_sample_min_size = 512
        self.config = _Config(block_out_channels=[128, 256, 512, 512])


class _QwenImage21StyleVae(_BaseVae):
    """Wan / Qwen-Image 2.1 style: height/width plus strides, no tile_sample_min_size."""

    def __init__(self):
        super().__init__()
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256
        self.tile_sample_stride_height = 192
        self.tile_sample_stride_width = 192
        self.config = _Config(dim_mult=[1, 2, 4, 4])


class _VideoVae(_BaseVae):
    """Detected by the presence of use_framewise_decoding."""

    def __init__(self):
        super().__init__()
        self.use_framewise_decoding = False
        self.tile_sample_min_num_frames = 1
        self.tile_sample_stride_num_frames = 1
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256

    def enable_slicing(self):
        self.slicing_enabled = True


class _StubPipeline:
    """Just enough surface for _apply_optimizations; no unet/transformer on purpose."""

    def __init__(self, vae):
        self.vae = vae


def _apply(vae, tile_size=TILE):
    pipeline = _StubPipeline(vae)
    WeeBasePipeline._apply_optimizations(
        pipeline,
        device="cpu",
        cache_to_ram=False,
        te_streamers={},
        transformer_key="transformer",
        vae_tile_size=tile_size,
    )
    return pipeline


def test_legacy_vae_keeps_existing_behaviour():
    vae = _LegacyVae()
    _apply(vae)

    assert vae.tiling_enabled
    assert vae.tile_sample_min_size == TILE
    # 4 block_out_channels -> downscale factor 8
    assert vae.tile_latent_min_size == TILE // 8
    assert not hasattr(vae, "tile_sample_min_height")


def test_qwenimage21_style_vae_gets_height_and_width():
    vae = _QwenImage21StyleVae()
    _apply(vae)

    assert vae.tiling_enabled
    assert vae.tile_sample_min_height == TILE
    assert vae.tile_sample_min_width == TILE
    # The 0.75 stride/tile ratio these VAEs ship with (192/256) is preserved.
    assert vae.tile_sample_stride_height == int(TILE * 0.75)
    assert vae.tile_sample_stride_width == int(TILE * 0.75)
    # The legacy attribute must not be invented on a VAE that has no use for it.
    assert not hasattr(vae, "tile_sample_min_size")
    assert not hasattr(vae, "tile_latent_min_size")


def test_video_vae_still_uses_temporal_decoding():
    vae = _VideoVae()
    _apply(vae)

    assert vae.use_framewise_decoding is True
    assert vae.tile_sample_min_num_frames == 9
    assert vae.tile_sample_stride_num_frames == 8
    # Spatial tiling is deliberately pushed out of range to avoid seam artifacts.
    assert vae.tile_sample_min_height == 10_000
    assert vae.tile_sample_min_width == 10_000
    assert not vae.tiling_enabled, "the video branch must not call enable_tiling()"


def test_vae_without_any_tile_attribute_does_not_crash(caplog):
    class _Bare(_BaseVae):
        pass

    vae = _Bare()
    with caplog.at_level("INFO", logger="weellm"):
        _apply(vae)

    assert vae.tiling_enabled
    assert any("built-in defaults" in record.message for record in caplog.records)


@pytest.mark.parametrize("tile_size", [64, 256, 1024])
def test_stride_never_collapses_to_zero(tile_size):
    vae = _QwenImage21StyleVae()
    _apply(vae, tile_size=tile_size)

    assert vae.tile_sample_stride_height >= 8
    assert vae.tile_sample_stride_width >= 8
    assert vae.tile_sample_stride_height <= vae.tile_sample_min_height
