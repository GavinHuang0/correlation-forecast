from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "cache_llama_2.py"
SPEC = importlib.util.spec_from_file_location("cache_llama_2", SCRIPT)
assert SPEC and SPEC.loader
cache = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cache)


def fake_safetensors(payload: bytes) -> bytes:
    header = b"{}"
    return len(header).to_bytes(8, byteorder="little") + header + payload


class CacheLlama2Tests(unittest.TestCase):
    def test_model_and_revision_are_frozen(self) -> None:
        self.assertEqual(cache.MODEL_ID, "meta-llama/Llama-2-7b-chat-hf")
        self.assertEqual(
            cache.MODEL_REVISION, "f5db02db724555f92da89c216ac04704f23d4590"
        )
        self.assertRegex(cache.MODEL_REVISION, r"^[0-9a-f]{40}$")

    def test_allow_patterns_include_all_weight_shards_and_tokenizer(self) -> None:
        self.assertIn("model*.safetensors", cache.ALLOW_PATTERNS)
        self.assertIn("model.safetensors.index.json", cache.ALLOW_PATTERNS)
        self.assertIn("tokenizer.json", cache.ALLOW_PATTERNS)
        self.assertIn("tokenizer_config.json", cache.ALLOW_PATTERNS)
        self.assertIn("tokenizer.model", cache.ALLOW_PATTERNS)

    def test_snapshot_manifest_hashes_files_without_loading_a_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contents = {
                "config.json": b"{}",
                "tokenizer_config.json": b"{}",
                "tokenizer.model": b"vocabulary",
                "model-00001-of-00001.safetensors": b"weights",
            }
            for name, payload in contents.items():
                (root / name).write_bytes(payload)
            manifest = cache.snapshot_file_manifest(root)
            self.assertEqual(set(manifest), set(contents))
            self.assertEqual(
                manifest["model-00001-of-00001.safetensors"]["sha256"],
                hashlib.sha256(b"weights").hexdigest(),
            )

    def test_snapshot_manifest_rejects_missing_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("config.json", "tokenizer_config.json", "tokenizer.model"):
                (root / name).write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "no safetensors"):
                cache.snapshot_file_manifest(root)

    def test_snapshot_manifest_requires_every_indexed_shard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("config.json", "tokenizer_config.json", "tokenizer.model"):
                (root / name).write_text("{}", encoding="utf-8")
            (root / "model-00002-of-00002.safetensors").write_bytes(b"second")
            index = {
                "metadata": {"total_size": len(b"first") + len(b"second")},
                "weight_map": {
                    "layer.0": "model-00001-of-00002.safetensors",
                    "layer.1": "model-00002-of-00002.safetensors",
                },
            }
            (root / "model.safetensors.index.json").write_text(
                json.dumps(index), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "missing indexed"):
                cache.snapshot_file_manifest(root)

    def test_snapshot_manifest_checks_indexed_weight_byte_total(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("config.json", "tokenizer_config.json", "tokenizer.model"):
                (root / name).write_text("{}", encoding="utf-8")
            (root / "model-00001-of-00001.safetensors").write_bytes(
                fake_safetensors(b"weights")
            )
            index = {
                "metadata": {"total_size": 999},
                "weight_map": {
                    "layer.0": "model-00001-of-00001.safetensors",
                },
            }
            (root / "model.safetensors.index.json").write_text(
                json.dumps(index), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "do not match"):
                cache.snapshot_file_manifest(root)

    def test_snapshot_manifest_compares_payload_not_header_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("config.json", "tokenizer_config.json", "tokenizer.model"):
                (root / name).write_text("{}", encoding="utf-8")
            payloads = {
                "model-00001-of-00002.safetensors": b"first",
                "model-00002-of-00002.safetensors": b"second",
            }
            for name, payload in payloads.items():
                (root / name).write_bytes(fake_safetensors(payload))
            index = {
                "metadata": {"total_size": sum(map(len, payloads.values()))},
                "weight_map": {
                    "layer.0": "model-00001-of-00002.safetensors",
                    "layer.1": "model-00002-of-00002.safetensors",
                },
            }
            (root / "model.safetensors.index.json").write_text(
                json.dumps(index), encoding="utf-8"
            )
            manifest = cache.snapshot_file_manifest(root)
            self.assertEqual(
                manifest["model-00001-of-00002.safetensors"]["size_bytes"],
                len(fake_safetensors(b"first")),
            )


if __name__ == "__main__":
    unittest.main()
