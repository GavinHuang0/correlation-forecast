from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "run_flan_t5_xl_coarse.py"
SPEC = importlib.util.spec_from_file_location("run_flan_t5_xl_for_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class FlanT5XlRunnerTests(unittest.TestCase):
    def test_injects_pinned_model_revision_and_offline_mode(self) -> None:
        arguments = runner.inject_pinned_arguments(
            ["--input", "in.jsonl", "--output", "out.jsonl"]
        )
        self.assertEqual(
            runner.option_value(arguments, "--model-id"), runner.MODEL_ID
        )
        self.assertEqual(
            runner.option_value(arguments, "--revision"), runner.MODEL_REVISION
        )
        self.assertIn("--local-files-only", arguments)

    def test_rejects_a_different_model_or_revision(self) -> None:
        with self.assertRaisesRegex(ValueError, "model-id is pinned"):
            runner.inject_pinned_arguments(["--model-id", "other/model"])
        with self.assertRaisesRegex(ValueError, "revision is pinned"):
            runner.inject_pinned_arguments(["--revision", "0" * 40])

    def test_sharded_hash_verifier_uses_manifest_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            names = (
                "config.json",
                "model.safetensors.index.json",
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
                "spiece.model",
                "tokenizer_config.json",
            )
            files = {}
            for index, name in enumerate(names):
                path = snapshot / name
                path.write_bytes(f"file-{index}".encode("ascii"))
                files[name] = {
                    "size_bytes": path.stat().st_size,
                    "sha256": runner.base.sha256_file(path),
                }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "manifest_version": "flan-t5-xl-local-snapshot-v1",
                        "model_id": runner.MODEL_ID,
                        "model_revision": runner.MODEL_REVISION,
                        "snapshot_path": str(snapshot.resolve()),
                        "files": files,
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.object(runner, "SNAPSHOT_MANIFEST", manifest_path):
                hashes = runner.sharded_snapshot_file_hashes(snapshot)
            self.assertEqual(set(hashes), set(names))

    def test_sharded_hash_verifier_rejects_changed_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            names = (
                "config.json",
                "model.safetensors.index.json",
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
                "spiece.model",
                "tokenizer_config.json",
            )
            files = {}
            for index, name in enumerate(names):
                path = snapshot / name
                path.write_bytes(f"file-{index}".encode("ascii"))
                files[name] = {
                    "size_bytes": path.stat().st_size,
                    "sha256": runner.base.sha256_file(path),
                }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "manifest_version": "flan-t5-xl-local-snapshot-v1",
                        "model_id": runner.MODEL_ID,
                        "model_revision": runner.MODEL_REVISION,
                        "snapshot_path": str(snapshot.resolve()),
                        "files": files,
                    }
                ),
                encoding="utf-8",
            )
            changed = snapshot / "config.json"
            changed.write_bytes(b"changed")
            with mock.patch.object(runner, "SNAPSHOT_MANIFEST", manifest_path):
                with self.assertRaisesRegex(ValueError, "size changed"):
                    runner.sharded_snapshot_file_hashes(snapshot)

    def test_sharded_hash_verifier_rejects_unsafe_manifest_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            files = {
                name: {"size_bytes": 1, "sha256": "0" * 64}
                for name in (
                    "config.json",
                    "model.safetensors.index.json",
                    "model-00001-of-00002.safetensors",
                    "model-00002-of-00002.safetensors",
                    "spiece.model",
                    "tokenizer_config.json",
                )
            }
            files["../outside"] = {"size_bytes": 1, "sha256": "0" * 64}
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "manifest_version": "flan-t5-xl-local-snapshot-v1",
                        "model_id": runner.MODEL_ID,
                        "model_revision": runner.MODEL_REVISION,
                        "snapshot_path": str(snapshot.resolve()),
                        "files": files,
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.object(runner, "SNAPSHOT_MANIFEST", manifest_path):
                with self.assertRaisesRegex(ValueError, "Unsafe path"):
                    runner.sharded_snapshot_file_hashes(snapshot)


if __name__ == "__main__":
    unittest.main()
