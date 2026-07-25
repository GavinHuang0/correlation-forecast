from __future__ import annotations

import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "cache_flan_t5_xl.py"
SPEC = importlib.util.spec_from_file_location("cache_flan_t5_xl_for_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
cache = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cache
SPEC.loader.exec_module(cache)


def write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    header: dict[str, object] = {}
    payload = bytearray()
    for name, (dtype, shape, data) in tensors.items():
        start = len(payload)
        payload.extend(data)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    padding = (-len(encoded)) % 8
    encoded += b" " * padding
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


class FlanT5XlCacheTests(unittest.TestCase):
    def build_snapshot(self, root: Path) -> Path:
        snapshot = root / "snapshot"
        snapshot.mkdir()
        shared_data = bytes(range(24))
        other_data = bytes(range(8))
        write_safetensors(
            snapshot / "model-00001-of-00002.safetensors",
            {
                "shared.weight": ("F32", [3, 2], shared_data),
                "encoder.block.weight": ("F32", [2], other_data),
            },
        )
        write_safetensors(
            snapshot / "model-00002-of-00002.safetensors",
            {"lm_head.weight": ("F32", [3, 2], shared_data)},
        )
        weight_map = {
            "shared.weight": "model-00001-of-00002.safetensors",
            "encoder.embed_tokens.weight": "model-00001-of-00002.safetensors",
            "decoder.embed_tokens.weight": "model-00001-of-00002.safetensors",
            "encoder.block.weight": "model-00001-of-00002.safetensors",
            "lm_head.weight": "model-00002-of-00002.safetensors",
        }
        # Physical bytes: 24 + 8 + 24. Logical aliases add 2 * 24.
        index = {"metadata": {"total_size": 104}, "weight_map": weight_map}
        (snapshot / "model.safetensors.index.json").write_text(
            json.dumps(index), encoding="utf-8"
        )
        (snapshot / "config.json").write_text("{}", encoding="utf-8")
        (snapshot / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        (snapshot / "spiece.model").write_bytes(b"test")
        return snapshot

    def test_accepts_exact_shared_embedding_alias_accounting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = self.build_snapshot(Path(temporary))
            manifest = cache.snapshot_file_manifest(snapshot)
            self.assertIn("model-00001-of-00002.safetensors", manifest)
            self.assertIn("model-00002-of-00002.safetensors", manifest)

    def test_rejects_truncated_shard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = self.build_snapshot(Path(temporary))
            shard = snapshot / "model-00002-of-00002.safetensors"
            shard.write_bytes(shard.read_bytes()[:-1])
            with self.assertRaisesRegex(RuntimeError, "Out-of-range|truncated"):
                cache.snapshot_file_manifest(snapshot)

    def test_rejects_unexpected_missing_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = self.build_snapshot(Path(temporary))
            index_path = snapshot / "model.safetensors.index.json"
            index = json.loads(index_path.read_text(encoding="utf-8"))
            index["weight_map"]["missing.weight"] = (
                "model-00002-of-00002.safetensors"
            )
            index_path.write_text(json.dumps(index), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Unexpected indexed tensors"):
                cache.snapshot_file_manifest(snapshot)


if __name__ == "__main__":
    unittest.main()
