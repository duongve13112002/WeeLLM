"""T-4: WeeBasePipeline must default use_kv_cache to off, but honour an explicit choice.

The prefix KV cache holds per-layer keys/values for the whole denoising loop. That is
activation memory the layer streamer cannot evict and the VRAM calibration pass cannot
see, so it must not be on by default under a tight VRAM budget.
"""

from __future__ import annotations

import pytest

from weellm.pipelines.weebasepipeline import WeeBasePipeline


class _RecordingPipeline:
    """Minimal stand-in for a diffusers pipeline that accepts use_kv_cache."""

    def __init__(self):
        self.received = None
        self.vae_scale_factor = 16

    def __call__(self, prompt=None, height=None, width=None, use_kv_cache=True, **kwargs):
        self.received = dict(
            prompt=prompt, height=height, width=width, use_kv_cache=use_kv_cache, **kwargs
        )
        return "result"


class _PipelineWithoutKvCache:
    """A pipeline whose signature has no use_kv_cache at all."""

    def __init__(self):
        self.received = None
        self.vae_scale_factor = 8

    def __call__(self, prompt=None, height=None, width=None, **kwargs):
        self.received = dict(prompt=prompt, height=height, width=width, **kwargs)
        return "result"


@pytest.fixture
def wrapped_recording():
    inner = _RecordingPipeline()
    return WeeBasePipeline(inner), inner


def test_defaults_to_off(wrapped_recording):
    wrapper, inner = wrapped_recording

    wrapper(prompt="a cat", height=64, width=64)

    assert inner.received["use_kv_cache"] is False


def test_explicit_true_is_honoured(wrapped_recording):
    wrapper, inner = wrapped_recording

    wrapper(prompt="a cat", height=64, width=64, use_kv_cache=True)

    assert inner.received["use_kv_cache"] is True


def test_explicit_false_is_honoured(wrapped_recording):
    wrapper, inner = wrapped_recording

    wrapper(prompt="a cat", height=64, width=64, use_kv_cache=False)

    assert inner.received["use_kv_cache"] is False


def test_not_injected_when_pipeline_does_not_accept_it():
    inner = _PipelineWithoutKvCache()
    wrapper = WeeBasePipeline(inner)

    wrapper(prompt="a cat", height=64, width=64)

    assert "use_kv_cache" not in inner.received


def test_dropped_when_pipeline_does_not_accept_it():
    """--use_kv_cache is a global CLI flag, so it reaches pipelines that reject it.

    Forwarding it there raises TypeError, which is how a FLUX run would die.
    """

    class _StrictPipeline:
        vae_scale_factor = 8

        def __init__(self):
            self.received = None

        def __call__(self, prompt=None, height=None, width=None):
            self.received = dict(prompt=prompt, height=height, width=width)
            return "result"

    inner = _StrictPipeline()
    wrapper = WeeBasePipeline(inner)

    wrapper(prompt="a cat", height=64, width=64, use_kv_cache=True)

    assert inner.received == {"prompt": "a cat", "height": 64, "width": 64}


def test_logs_the_reason(wrapped_recording, caplog):
    wrapper, _ = wrapped_recording

    with caplog.at_level("INFO", logger="weellm"):
        wrapper(prompt="a cat", height=64, width=64)

    assert any("use_kv_cache=False" in record.message for record in caplog.records)
