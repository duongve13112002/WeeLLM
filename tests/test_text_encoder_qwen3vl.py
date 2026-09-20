"""T-8: the shared Qwen3-VL text-encoder streamer must serve both checkpoint shapes.

MiniMax-H3 ships a pruned Qwen3-VL with no final-norm weight and fewer layers than the
config declares; Qwen-Image 2.1 ships a full checkpoint whose ``norm.weight`` lands in
the resident set. Replacing the norm with Identity unconditionally (the old behaviour)
breaks the second case, because placing that tensor then fails.
"""

from __future__ import annotations

import torch.nn as nn

from weellm.models.text_encoders import qwen3_vl_for_conditional_generation as mod


class _FakeSeeker:
    def __init__(self, keys):
        self.weight_map = {k: "shard" for k in keys}


class _FakeLanguageModel:
    def __init__(self, num_layers):
        self.layers = nn.ModuleList([nn.Identity() for _ in range(num_layers)])
        self.norm = nn.LayerNorm(4)


class _FakeInner:
    def __init__(self, num_layers):
        self.language_model = _FakeLanguageModel(num_layers)


class _FakeLMHead(nn.Linear):
    def __init__(self):
        super().__init__(4, 8, bias=False)


class _FakeModel:
    def __init__(self, num_layers):
        self.model = _FakeInner(num_layers)
        self.lm_head = _FakeLMHead()

    def eval(self):
        return self


def _run_skeleton(keys, num_layers=4, monkeypatch=None):
    """Drive _load_model_skeleton with a stubbed config load and model construction."""
    streamer = mod.Qwen3VLForConditionalGenerationStreamer.__new__(
        mod.Qwen3VLForConditionalGenerationStreamer
    )
    streamer.text_encoder_dir = "unused"
    streamer.dtype = None
    streamer._seeker = _FakeSeeker(keys)
    streamer._model = None

    fake_model = _FakeModel(num_layers)

    monkeypatch.setattr(mod.AutoConfig, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(mod, "default_dtype", lambda _dtype: _nullcontext())
    monkeypatch.setattr(mod, "init_empty_weights", _nullcontext)
    monkeypatch.setattr(mod, "Qwen3VLForConditionalGeneration", lambda _cfg: fake_model)

    streamer._load_model_skeleton()
    return streamer


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def test_full_checkpoint_keeps_the_real_norm(monkeypatch):
    """Qwen-Image 2.1: norm.weight is in the checkpoint, so the module must survive."""
    keys = [
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.1.self_attn.q_proj.weight",
        "model.language_model.layers.2.self_attn.q_proj.weight",
        "model.language_model.layers.3.self_attn.q_proj.weight",
        "model.language_model.norm.weight",
        "model.language_model.embed_tokens.weight",
    ]
    streamer = _run_skeleton(keys, num_layers=4, monkeypatch=monkeypatch)

    norm = streamer._model.model.language_model.norm
    assert not isinstance(norm, nn.Identity), "the real norm was replaced, resident load would fail"


def test_pruned_checkpoint_still_gets_identity(monkeypatch):
    """MiniMax-H3: no norm weight shipped, so Identity avoids a meta-device crash."""
    keys = [
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.1.self_attn.q_proj.weight",
        "model.language_model.embed_tokens.weight",
    ]
    streamer = _run_skeleton(keys, num_layers=4, monkeypatch=monkeypatch)

    assert isinstance(streamer._model.model.language_model.norm, nn.Identity)


def test_pruned_checkpoint_truncates_layers(monkeypatch):
    """Fewer layer weights than the config declares -> the module list is truncated."""
    keys = [
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.1.self_attn.q_proj.weight",
    ]
    streamer = _run_skeleton(keys, num_layers=8, monkeypatch=monkeypatch)

    assert len(streamer._model.model.language_model.layers) == 2


def test_full_checkpoint_is_not_truncated(monkeypatch):
    keys = [f"model.language_model.layers.{i}.self_attn.q_proj.weight" for i in range(4)]
    streamer = _run_skeleton(keys, num_layers=4, monkeypatch=monkeypatch)

    assert len(streamer._model.model.language_model.layers) == 4


def test_no_layer_keys_means_no_truncation(monkeypatch):
    """The old code guessed 50 layers here and silently truncated. It must not."""
    keys = ["model.language_model.embed_tokens.weight", "model.language_model.norm.weight"]
    streamer = _run_skeleton(keys, num_layers=8, monkeypatch=monkeypatch)

    assert len(streamer._model.model.language_model.layers) == 8


def test_lm_head_is_replaced_with_a_passthrough(monkeypatch):
    """lm_head is excluded from the resident set, so it must not stay a real Linear.

    Qwen3VLForConditionalGeneration.forward projects through lm_head unconditionally.
    With bias=False and a meta weight that does NOT raise — it silently returns an
    uninitialised (batch, seq, vocab) tensor.
    """
    import torch

    keys = [
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.norm.weight",
        "lm_head.weight",  # present in the checkpoint but skipped by _get_resident_keys
    ]
    streamer = _run_skeleton(keys, num_layers=1, monkeypatch=monkeypatch)

    head = streamer._model.lm_head
    assert isinstance(head, mod._PassThroughLMHead)
    assert not list(head.parameters()), "the pass-through must hold no weights"

    hidden = torch.randn(1, 3, 4)
    assert head(hidden) is hidden


def test_lm_head_is_kept_when_it_would_be_resident(monkeypatch):
    """If the resident policy ever starts loading lm_head, stop replacing it."""
    monkeypatch.setattr(mod, "_get_resident_keys", lambda _seeker: ["lm_head.weight"])

    keys = ["model.language_model.layers.0.self_attn.q_proj.weight", "lm_head.weight"]
    streamer = _run_skeleton(keys, num_layers=1, monkeypatch=monkeypatch)

    assert not isinstance(streamer._model.lm_head, mod._PassThroughLMHead)


def test_module_emits_no_prints(capsys, monkeypatch):
    """Debug output belongs in the logger, not on the user's stdout."""
    keys = [f"model.language_model.layers.{i}.self_attn.q_proj.weight" for i in range(2)]
    _run_skeleton(keys, num_layers=2, monkeypatch=monkeypatch)

    captured = capsys.readouterr()
    assert captured.out == ""
