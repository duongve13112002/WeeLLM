"""Shared fixtures for the WeeLLM test suite.

Everything here runs on CPU with tiny synthetic checkpoints generated at runtime —
no GPU, no model downloads. Fixture files are written into pytest's ``tmp_path``
because ``.gitignore`` deliberately excludes ``*.safetensors`` / ``*.pt``.
"""

from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

# Tiny config: axes_dims_rope must sum to attention_head_dim and each axis must be even.
TINY_QWEN21_CONFIG = dict(
    patch_size=1,
    in_channels=8,
    out_channels=8,
    num_layers=2,
    attention_head_dim=16,
    num_attention_heads=2,
    context_in_dim=32,
    mlp_ratio=2,
    axes_dims_rope=(4, 6, 6),
    eps=1e-6,
    causal_condition=True,
)


def pytest_collection_modifyitems(config, items):
    """Make the declared markers actually skip, instead of being decorative."""
    has_cuda = torch.cuda.is_available()
    try:
        from diffusers import QwenImage21Transformer2DModel  # noqa: F401

        has_qwenimage21 = True
    except ImportError:
        has_qwenimage21 = False

    for item in items:
        if "gpu" in item.keywords and not has_cuda:
            item.add_marker(pytest.mark.skip(reason="no CUDA device"))
        if "needs_diffusers_main" in item.keywords and not has_qwenimage21:
            item.add_marker(pytest.mark.skip(reason="needs diffusers >= 0.41.0"))


def qwenimage21_cls():
    """Return QwenImage21Transformer2DModel, skipping the test if diffusers is too old."""
    try:
        from diffusers import QwenImage21Transformer2DModel
    except ImportError:  # pragma: no cover - environment dependent
        pytest.skip("diffusers >= 0.41.0 (with QwenImage21Transformer2DModel) is not installed")
    return QwenImage21Transformer2DModel


def write_checkpoint(model: torch.nn.Module, model_dir) -> None:
    """Persist *model* as a single-shard safetensors checkpoint plus config.json."""
    model_dir.mkdir(parents=True, exist_ok=True)
    state_dict = {k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}
    save_file(state_dict, str(model_dir / "diffusion_pytorch_model.safetensors"))
    # ConfigMixin writes config.json in exactly the shape load_config() expects.
    model.save_config(str(model_dir))


@pytest.fixture
def tiny_qwenimage21(tmp_path):
    """A tiny QwenImage21Transformer2DModel saved to disk.

    Returns ``(model_dir, reference_model)`` where *reference_model* holds the very
    same weights in ordinary (non-streamed) form, so streamed output can be compared
    against it.
    """
    torch.manual_seed(0)
    model_cls = qwenimage21_cls()

    model = model_cls(**TINY_QWEN21_CONFIG)
    model.eval()

    model_dir = tmp_path / "transformer"
    write_checkpoint(model, model_dir)
    return model_dir, model


@pytest.fixture
def tiny_qwenimage21_inputs():
    """Deterministic forward inputs matching ``TINY_QWEN21_CONFIG``.

    The vision-language sequence is 4 text tokens plus 1 image slot; each slot stands
    for a 2x2 group of latent tokens, so the target image is 2x2 = 4 latent tokens.
    That is the smallest shape that still exercises the block-causal path.
    """
    torch.manual_seed(1)
    height = width = 2
    num_latent_tokens = height * width          # 4
    num_image_slots = num_latent_tokens // 4    # 1
    num_text_tokens = 4

    img_mask = torch.tensor(
        [[False] * num_text_tokens + [True] * num_image_slots],
        dtype=torch.bool,
    )

    return dict(
        hidden_states=torch.randn(1, num_latent_tokens, TINY_QWEN21_CONFIG["in_channels"]),
        encoder_hidden_states=torch.randn(1, num_text_tokens, TINY_QWEN21_CONFIG["context_in_dim"]),
        timestep=torch.tensor([1.0]),
        img_shapes=[[(1, height, width)]],
        img_mask=img_mask,
        return_dict=False,
    )
