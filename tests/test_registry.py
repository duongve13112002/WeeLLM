"""T-7: the class-name -> module registries must stay importable and consistent.

WeeLLM dispatches purely on the diffusers class name found in model_index.json, so a
typo in a registry entry only surfaces at load time for that one model. These tests
walk every entry.
"""

from __future__ import annotations

import importlib

import pytest

from weellm.pipelines.image.weeimagetoimagepipeline import IMG2IMG_MAPPING
from weellm.pipelines.weebasepipeline import _TE_MAP, _TR_MAP


def test_qwenimage21_transformer_is_registered():
    assert (
        _TR_MAP["QwenImage21Transformer2DModel"]
        == "weellm.models.transformers.qwen_image_21_transformer_2d_model"
    )


def test_qwenimage21_pipeline_supports_edit_mode():
    # Qwen-Image 2.1 serves text-to-image and editing from the same class, so the
    # img2img mapping points at itself rather than a separate *EditPipeline.
    assert IMG2IMG_MAPPING["QwenImage21Pipeline"] == "QwenImage21Pipeline"


def test_qwenimage21_is_distinct_from_v1():
    """Qwen-Image and Qwen-Image 2.1 are different architectures and must not collide."""
    assert _TR_MAP["QwenImageTransformer2DModel"] != _TR_MAP["QwenImage21Transformer2DModel"]


@pytest.mark.parametrize("class_name,module_path", sorted(_TR_MAP.items()))
def test_every_transformer_entry_resolves(class_name, module_path):
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        if exc.name == module_path:
            pytest.skip(f"{module_path} is not implemented yet")
        raise

    streamer_name = class_name + "Streamer"
    assert hasattr(module, streamer_name), f"{module_path} has no {streamer_name}"


@pytest.mark.parametrize("class_name,module_path", sorted(set(_TE_MAP.items())))
def test_every_text_encoder_entry_resolves(class_name, module_path):
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        if exc.name == module_path:
            pytest.skip(f"{module_path} is not implemented yet")
        raise

    streamer_name = "CLIPTextModelStreamer" if "CLIP" in class_name else class_name + "Streamer"
    assert hasattr(module, streamer_name), f"{module_path} has no {streamer_name}"


def test_new_streamer_is_exported_from_package():
    import weellm

    assert "QwenImage21Transformer2DModelStreamer" in weellm.__all__
    assert hasattr(weellm, "QwenImage21Transformer2DModelStreamer")


def test_package_exports_match_all():
    """Everything advertised in __all__ must actually be importable from the package."""
    import weellm

    missing = [name for name in weellm.__all__ if not hasattr(weellm, name)]
    assert not missing, f"__all__ advertises names that do not exist: {missing}"
