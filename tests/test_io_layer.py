"""T-9: regression tests for the pure-Python IO layer the streamers sit on.

No torch tensors of consequence, no GPU, no downloads — just header parsing, shard
selection and the thread-local weights-path override.
"""

from __future__ import annotations

import json
import struct
import threading
from pathlib import Path

import numpy as np
import pytest

from weellm.io.safetensors.safetensors_base import DTYPE_MAP, SafetensorsBase
from weellm.io.seeker import _extract_hub_repo_from_cache_path, override_weights_path


def _write_safetensors(path: Path, tensors: dict) -> None:
    """Write a minimal but valid safetensors file."""
    header = {}
    blobs = []
    offset = 0
    for name, (dtype_str, shape) in tensors.items():
        count = int(np.prod(shape)) if shape else 1
        nbytes = count * np.dtype(DTYPE_MAP[dtype_str]).itemsize
        header[name] = {"dtype": dtype_str, "shape": list(shape), "data_offsets": [offset, offset + nbytes]}
        blobs.append(b"\x00" * nbytes)
        offset += nbytes

    header_bytes = json.dumps(header).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(header_bytes)))
        fh.write(header_bytes)
        for blob in blobs:
            fh.write(blob)


# --------------------------------------------------------------------- shard selection

def test_prefers_canonical_shard_name(tmp_path):
    (tmp_path / "model.safetensors").touch()
    (tmp_path / "zzz-other.safetensors").touch()

    assert SafetensorsBase(tmp_path)._find_single_shard() == "model.safetensors"


def test_single_glob_fallback(tmp_path):
    (tmp_path / "custom-weights.safetensors").touch()

    assert SafetensorsBase(tmp_path)._find_single_shard() == "custom-weights.safetensors"


def test_ambiguous_shards_raise(tmp_path):
    (tmp_path / "a.safetensors").touch()
    (tmp_path / "b.safetensors").touch()

    with pytest.raises(FileNotFoundError, match="Multiple .safetensors files"):
        SafetensorsBase(tmp_path)._find_single_shard()


def test_no_shard_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Could not find a safetensors index"):
        SafetensorsBase(tmp_path)._find_single_shard()


# --------------------------------------------------------------------- header parsing

def test_header_is_parsed_and_memoised(tmp_path):
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, {"a.weight": ("F32", [2, 3])})

    base = SafetensorsBase(tmp_path)
    header, data_base = base._read_header(path)

    assert header["a.weight"]["shape"] == [2, 3]
    assert data_base == 8 + len(json.dumps(header).encode("utf-8"))
    assert "model.safetensors" in base._parsed_headers

    # A second call must hit the memo rather than re-reading.
    assert base._read_header(path) == (header, data_base)


def test_truncated_file_raises(tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"123")

    with pytest.raises(ValueError, match="too small"):
        SafetensorsBase(tmp_path)._read_header(path)


# --------------------------------------------------------------------- block sizing

def test_get_block_bytes_sums_only_requested_keys(tmp_path):
    path = tmp_path / "model.safetensors"
    _write_safetensors(
        path,
        {
            "blocks.0.weight": ("F32", [4, 4]),   # 64 bytes
            "blocks.1.weight": ("BF16", [4, 4]),  # 32 bytes
        },
    )

    base = SafetensorsBase(tmp_path)
    base._parse_index()

    assert base.get_block_bytes(["blocks.0.weight"]) == 64
    assert base.get_block_bytes(["blocks.1.weight"]) == 32
    assert base.get_block_bytes(["blocks.0.weight", "blocks.1.weight"]) == 96


def test_get_block_bytes_ignores_unknown_keys(tmp_path):
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, {"blocks.0.weight": ("F32", [2, 2])})

    base = SafetensorsBase(tmp_path)
    base._parse_index()

    assert base.get_block_bytes(["nope"]) == 0
    assert base.get_block_bytes(["blocks.0.weight", "nope"]) == 16


def test_parse_index_reads_sharded_index(tmp_path):
    _write_safetensors(tmp_path / "shard-1.safetensors", {"a": ("F32", [1])})
    index = {"weight_map": {"a": "shard-1.safetensors"}}
    (tmp_path / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps(index))

    base = SafetensorsBase(tmp_path)
    base._parse_index()

    assert base.weight_map == {"a": "shard-1.safetensors"}


def test_parse_index_drops_metadata_key(tmp_path):
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, {"a": ("F32", [1])})

    base = SafetensorsBase(tmp_path)
    base._parse_index()

    assert "__metadata__" not in base.weight_map


# --------------------------------------------------------------------- hub cache paths

@pytest.mark.parametrize(
    "path,expected",
    [
        (
            "/root/.cache/huggingface/hub/models--Qwen--Qwen-Image-2.1/snapshots/abc123/transformer",
            ("Qwen/Qwen-Image-2.1", "transformer"),
        ),
        (
            "/root/.cache/huggingface/hub/models--Qwen--Qwen-Image-2.1/snapshots/abc123",
            ("Qwen/Qwen-Image-2.1", None),
        ),
        ("/home/user/my-local-model/transformer", (None, None)),
    ],
)
def test_extract_hub_repo_from_cache_path(path, expected):
    assert _extract_hub_repo_from_cache_path(Path(path)) == expected


def test_repo_id_only_splits_the_first_separator():
    repo_id, _ = _extract_hub_repo_from_cache_path(
        Path("/c/models--org--repo--with--dashes/snapshots/h/sub")
    )
    assert repo_id == "org/repo--with--dashes"


# --------------------------------------------------------------------- override context

def test_override_restores_previous_value():
    with override_weights_path("/tmp/outer", subfolder="transformer"):
        with override_weights_path("/tmp/inner", subfolder="vae"):
            pass
        from weellm.io.seeker import _override_local

        assert _override_local.weights_path == "/tmp/outer"
        assert _override_local.subfolder == "transformer"

    from weellm.io.seeker import _override_local

    assert getattr(_override_local, "weights_path", None) is None


def test_override_is_thread_local():
    seen = {}

    def worker():
        from weellm.io.seeker import _override_local

        seen["other_thread"] = getattr(_override_local, "weights_path", None)

    with override_weights_path("/tmp/main-thread"):
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

    assert seen["other_thread"] is None, "override leaked into another thread"
