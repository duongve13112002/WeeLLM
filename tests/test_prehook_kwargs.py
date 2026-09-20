"""T-5: the streaming pre-hook must work for blocks called with keyword arguments.

Qwen-Image 2.1 calls each transformer block with keyword arguments only. A
positional-only ``register_forward_pre_hook`` sees an empty ``args`` there, which used
to make the float16 overflow clamp a silent no-op. These tests pin down both calling
conventions.
"""

from __future__ import annotations

from typing import List, Tuple

import pytest
import torch
import torch.nn as nn

from weellm.io.memory import evict_module
from weellm.models.transformers.base_transformer_streamer import BaseTransformerStreamer


class _DictSeeker:
    """In-memory stand-in for a safetensors seeker."""

    def __init__(self, state_dict):
        self._state = {k: v.clone() for k, v in state_dict.items()}
        self.weight_map = {k: "memory" for k in self._state}

    def get_tensors(self, keys, device="cpu", dtype=None):
        out = {}
        for key in keys:
            tensor = self._state[key].clone()
            if dtype is not None and tensor.is_floating_point():
                tensor = tensor.to(dtype)
            out[key] = tensor.to(device)
        return out

    def get_block_bytes(self, keys):
        return sum(self._state[k].numel() * self._state[k].element_size() for k in keys)


class _KwargsBlock(nn.Module):
    """A block that can only be called with keyword arguments."""

    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        self.seen_kwargs = None

    def forward(self, *, hidden_states, scale):
        self.seen_kwargs = dict(hidden_states=hidden_states, scale=scale)
        return self.linear(hidden_states) * scale


class _PositionalBlock(nn.Module):
    """A classic block taking positional arguments, like the older architectures."""

    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        self.seen_args = None

    def forward(self, hidden_states):
        self.seen_args = hidden_states
        return self.linear(hidden_states)


class _Toy(nn.Module):
    def __init__(self, block_cls, dim, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([block_cls(dim) for _ in range(num_blocks)])


class _ToyStreamer(BaseTransformerStreamer):
    def _get_shard_order(self) -> List[Tuple[str, nn.Module]]:
        return [(f"blocks.{i}", block) for i, block in enumerate(self.model.blocks)]

    def _get_resident_keys(self) -> List[str]:
        return []


def _make_streamer(block_cls, dim=4, dtype=torch.float32):
    torch.manual_seed(0)
    reference = _Toy(block_cls, dim)
    seeker = _DictSeeker(reference.state_dict())

    streamed = _Toy(block_cls, dim)
    # Same helper production code uses to push weights back to the meta device.
    evict_module(streamed)

    streamer = _ToyStreamer(
        model=streamed, seeker=seeker, device="cpu", dtype=dtype, prefetch=False
    )
    return streamer, reference


def test_kwargs_only_block_gets_its_weights_streamed():
    streamer, reference = _make_streamer(_KwargsBlock)

    hidden = torch.randn(1, 3, 4)
    scale = 2.0

    out = hidden
    for block in streamer.model.blocks:
        out = block(hidden_states=out, scale=scale)

    expected = hidden
    for block in reference.blocks:
        expected = block(hidden_states=expected, scale=scale)

    torch.testing.assert_close(out, expected)
    # And everything went back to meta afterwards.
    assert all(p.device.type == "meta" for p in streamer.model.parameters())


def test_positional_block_still_works():
    """Regression guard for the ~20 architectures that call blocks positionally."""
    streamer, reference = _make_streamer(_PositionalBlock)

    hidden = torch.randn(1, 3, 4)

    out = hidden
    for block in streamer.model.blocks:
        out = block(out)

    expected = hidden
    for block in reference.blocks:
        expected = block(expected)

    torch.testing.assert_close(out, expected)
    assert all(p.device.type == "meta" for p in streamer.model.parameters())


def test_float16_clamp_applies_to_kwargs():
    """The overflow clamp must reach tensors passed as keyword arguments."""
    streamer, _ = _make_streamer(_KwargsBlock, dtype=torch.float16)

    huge = torch.full((1, 2, 4), 65000.0, dtype=torch.float16)
    block = streamer.model.blocks[0]
    block(hidden_states=huge, scale=1.0)

    seen = block.seen_kwargs["hidden_states"]
    assert seen.max().item() <= 60000.0, "kwargs tensor was not clamped"


def test_float16_clamp_applies_to_positional_args():
    streamer, _ = _make_streamer(_PositionalBlock, dtype=torch.float16)

    huge = torch.full((1, 2, 4), 65000.0, dtype=torch.float16)
    block = streamer.model.blocks[0]
    block(huge)

    assert block.seen_args.max().item() <= 60000.0, "positional tensor was not clamped"


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_no_clamp_outside_float16(dtype):
    """Clamping is a float16 overflow workaround; other dtypes must pass through."""
    streamer, _ = _make_streamer(_PositionalBlock, dtype=dtype)

    value = torch.full((1, 2, 4), 70000.0, dtype=dtype)
    block = streamer.model.blocks[0]
    block(value)

    assert block.seen_args.max().item() == pytest.approx(70000.0, rel=1e-2)
