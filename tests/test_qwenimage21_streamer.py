"""Tests for QwenImage21Transformer2DModelStreamer (T-1, T-2, T-3).

These run the real streaming machinery on CPU against a tiny synthetic checkpoint:
weights live on disk, blocks are pulled in by the forward pre-hook and pushed back to
the meta device by the post-hook.
"""

from __future__ import annotations

import pytest
import torch

from weellm.models.transformers.qwen_image_21_transformer_2d_model import (
    QwenImage21Transformer2DModelStreamer,
)

pytestmark = pytest.mark.needs_diffusers_main


def _build_streamer(model_dir, prefetch=False, dtype=torch.float32):
    return QwenImage21Transformer2DModelStreamer.from_pretrained(
        model_dir, device="cpu", dtype=dtype, prefetch=prefetch
    )


# --------------------------------------------------------------------------- T-1

def test_shard_order_covers_every_block(tiny_qwenimage21):
    model_dir, reference = tiny_qwenimage21
    streamer = _build_streamer(model_dir)

    order = streamer._get_shard_order()
    names = [name for name, _ in order]

    assert names == ["transformer_blocks.0", "transformer_blocks.1"]
    assert len(order) == len(reference.transformer_blocks)
    # The second element must be the live module, not a copy.
    for idx, (_, module) in enumerate(order):
        assert module is streamer.model.transformer_blocks[idx]


def test_resident_keys_exclude_streamed_blocks(tiny_qwenimage21):
    model_dir, _ = tiny_qwenimage21
    streamer = _build_streamer(model_dir)

    resident = streamer._get_resident_keys()

    assert resident, "resident set must not be empty"
    assert not [k for k in resident if k.startswith("transformer_blocks.")]

    # The small always-needed submodules have to stay resident.
    for prefix in ("img_in.", "txt_in.", "modulation.", "norm_out.", "proj_out.", "time_text_embed."):
        assert any(k.startswith(prefix) for k in resident), f"missing resident weights for {prefix}"


def test_resident_and_streamed_partition_the_checkpoint(tiny_qwenimage21):
    """Every checkpoint key is either resident or belongs to exactly one shard."""
    model_dir, _ = tiny_qwenimage21
    streamer = _build_streamer(model_dir)

    resident = set(streamer._get_resident_keys())
    streamed = set()
    for name, _ in streamer._get_shard_order():
        keys = set(streamer._get_layer_keys(name))
        assert keys, f"shard {name} resolved to no weights"
        assert not (streamed & keys), "a weight key is claimed by two shards"
        streamed |= keys

    assert not (resident & streamed)
    assert resident | streamed == set(streamer.seeker.weight_map)


# --------------------------------------------------------------------------- T-2

def test_streamed_forward_matches_reference(tiny_qwenimage21, tiny_qwenimage21_inputs):
    model_dir, reference = tiny_qwenimage21
    streamer = _build_streamer(model_dir)

    with torch.no_grad():
        expected = reference(**tiny_qwenimage21_inputs)[0]
        actual = streamer.model(**tiny_qwenimage21_inputs)[0]

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("prefetch", [False, True])
def test_streamed_forward_matches_reference_with_prefetch(
    tiny_qwenimage21, tiny_qwenimage21_inputs, prefetch
):
    """Prefetch changes the scheduling, never the numbers."""
    model_dir, reference = tiny_qwenimage21
    streamer = _build_streamer(model_dir, prefetch=prefetch)

    with torch.no_grad():
        expected = reference(**tiny_qwenimage21_inputs)[0]
        actual = streamer.model(**tiny_qwenimage21_inputs)[0]

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


# --------------------------------------------------------------------------- T-3

def _block_param_devices(streamer):
    devices = set()
    for _, block in streamer._get_shard_order():
        for param in block.parameters():
            devices.add(param.device.type)
    return devices


def test_blocks_return_to_meta_after_forward(tiny_qwenimage21, tiny_qwenimage21_inputs):
    model_dir, _ = tiny_qwenimage21
    streamer = _build_streamer(model_dir)

    # Before any forward the streamed blocks have never been materialised.
    assert _block_param_devices(streamer) == {"meta"}

    with torch.no_grad():
        streamer.model(**tiny_qwenimage21_inputs)

    assert _block_param_devices(streamer) == {"meta"}, "streamed blocks were left in memory"

    # Resident weights must NOT have been evicted.
    resident_keys = streamer._get_resident_keys()
    state = dict(streamer.model.state_dict())
    for key in resident_keys:
        assert state[key].device.type != "meta", f"resident weight {key} was evicted"


def test_second_pass_is_still_correct(tiny_qwenimage21, tiny_qwenimage21_inputs):
    """The second pass takes the post-calibration path; results must not drift."""
    model_dir, reference = tiny_qwenimage21
    streamer = _build_streamer(model_dir)

    with torch.no_grad():
        expected = reference(**tiny_qwenimage21_inputs)[0]
        first = streamer.model(**tiny_qwenimage21_inputs)[0]
        second = streamer.model(**tiny_qwenimage21_inputs)[0]

    torch.testing.assert_close(first, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(second, expected, rtol=1e-5, atol=1e-6)


# ------------------------------------------------------- edit mode & KV cache

def _edit_mode_inputs():
    """Condition image + target image, the layout WeeImageToImagePipeline drives.

    The vision-language sequence is 3 text tokens plus 1 condition-image slot; the
    target image contributes ``target_tokens // 4`` further slots. Packed latents hold
    the condition image's 2x2 tokens followed by the target's 2x2.
    """
    torch.manual_seed(1)
    return dict(
        hidden_states=torch.randn(1, 8, 8),
        encoder_hidden_states=torch.randn(1, 4, 32),
        timestep=torch.tensor([1.0]),
        img_shapes=[[(1, 2, 2), (1, 2, 2)]],
        img_mask=torch.tensor([[False, False, False, True, True]]),
        return_dict=False,
    )


def test_edit_mode_shapes_stream_correctly(tiny_qwenimage21):
    """Multiple image blocks exercise the block-causal path that plain T2I skips."""
    model_dir, reference = tiny_qwenimage21
    streamer = _build_streamer(model_dir)
    inputs = _edit_mode_inputs()

    with torch.no_grad():
        expected = reference(**inputs)[0]
        actual = streamer.model(**inputs)[0]

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert _block_param_devices(streamer) == {"meta"}


def test_kv_cache_survives_block_eviction(tiny_qwenimage21):
    """Opting into use_kv_cache must still be numerically correct under streaming.

    The caches live on the caller's QwenImage21KVCache, not on the blocks, so evicting
    a block to meta between steps must not disturb them. If this ever regresses the
    failure is silent — wrong pixels, not an exception.
    """
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache

    model_dir, _ = tiny_qwenimage21
    streamer = _build_streamer(model_dir)
    inputs = _edit_mode_inputs()
    num_layers = len(streamer.model.transformer_blocks)

    with torch.no_grad():
        uncached = streamer.model(**inputs)[0]

        cache = QwenImage21KVCache(num_layers)
        prefill = streamer.model(**inputs, kv_cache=cache, kv_cache_mode="extract")[0]
        decoded = streamer.model(**inputs, kv_cache=cache, kv_cache_mode="cached")[0]

    torch.testing.assert_close(prefill, uncached, rtol=1e-5, atol=1e-6)
    # The cached pass returns only the target rows.
    target_rows = uncached[:, -decoded.shape[1]:]
    torch.testing.assert_close(decoded, target_rows, rtol=1e-4, atol=1e-5)
    assert _block_param_devices(streamer) == {"meta"}


def test_cache_context_is_forwarded(tiny_qwenimage21):
    """QwenImage21Pipeline wraps each transformer call in transformer.cache_context(...)."""
    model_dir, _ = tiny_qwenimage21
    streamer = _build_streamer(model_dir)

    with streamer.cache_context("cond"):
        pass  # must not raise


def test_import_error_mentions_diffusers_version(monkeypatch):
    """A too-old diffusers must fail with an actionable message, not a bare ImportError.

    Simulates the real failure by swapping in a diffusers module that lacks the class.
    """
    import sys
    import types

    from weellm.models.transformers.qwen_image_21_transformer_2d_model import (
        _import_transformer_cls,
    )

    stub = types.ModuleType("diffusers")
    stub.__version__ = "0.40.0"  # no QwenImage21Transformer2DModel attribute
    monkeypatch.setitem(sys.modules, "diffusers", stub)

    with pytest.raises(ImportError) as excinfo:
        _import_transformer_cls()

    message = str(excinfo.value)
    assert "0.41.0" in message
    assert "0.40.0" in message, "the installed version should be reported back to the user"
    assert "git+https://github.com/huggingface/diffusers" in message
