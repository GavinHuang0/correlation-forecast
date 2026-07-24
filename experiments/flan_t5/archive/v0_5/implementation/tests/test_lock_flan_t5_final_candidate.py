from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "lock_flan_t5_final_candidate.py"
SPEC = importlib.util.spec_from_file_location("flan_candidate_lock_for_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
lock = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lock)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def write_jsonl(path: Path, ids: list[str]) -> None:
    path.write_text(
        "".join(json.dumps({"article_id": article_id}) + "\n" for article_id in ids),
        encoding="utf-8",
    )


class FinalCandidateLockTests(unittest.TestCase):
    def make_args(self, directory: Path) -> list[str]:
        protocol = directory / "PROTOCOL.md"
        protocol.write_text("locked protocol\n", encoding="utf-8")
        promotion = directory / "promotion.json"
        write_json(promotion, {"rule_version": "flan-t5-final-promotion-v1"})
        config = directory / "config.json"
        write_json(
            config,
            {
                "protocol_version": "0.5.0",
                "prompt_version": "flan-stock-sector-news-binary-v0.5.0",
            },
        )
        plain_files: dict[str, Path] = {}
        for name in (
            "extractor",
            "selector",
            "schema",
            "deterministic",
            "dev_coarse",
            "dev_fine",
            "eval_baseline",
        ):
            path = directory / f"{name}.txt"
            path.write_text(name + "\n", encoding="utf-8")
            plain_files[name] = path
        development = directory / "development.jsonl"
        evaluation = directory / "evaluation.jsonl"
        reference = directory / "reference.jsonl"
        write_jsonl(development, ["a", "b"])
        write_jsonl(evaluation, ["c"])
        write_jsonl(reference, ["a", "b", "c"])
        output = directory / "lock.json"
        return [
            "lock_flan_t5_final_candidate.py",
            "--protocol",
            str(protocol),
            "--promotion-rule",
            str(promotion),
            "--config",
            str(config),
            "--extractor",
            str(plain_files["extractor"]),
            "--selector",
            str(plain_files["selector"]),
            "--schema",
            str(plain_files["schema"]),
            "--deterministic-module",
            str(plain_files["deterministic"]),
            "--development-input",
            str(development),
            "--evaluation-input",
            str(evaluation),
            "--reference",
            str(reference),
            "--development-coarse-v0-4",
            str(plain_files["dev_coarse"]),
            "--development-fine-v0-2",
            str(plain_files["dev_fine"]),
            "--evaluation-v0-4-hybrid",
            str(plain_files["eval_baseline"]),
            "--output",
            str(output),
        ]

    def test_lock_records_hashes_and_split_invariants(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            arguments = self.make_args(directory)
            with mock.patch.object(sys, "argv", arguments):
                self.assertEqual(lock.main(), 0)
            output = directory / "lock.json"
            manifest = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["status"], "locked_before_v0_5_model_scoring"
            )
            self.assertEqual(
                manifest["model"]["revision"], lock.MODEL_REVISION
            )
            self.assertEqual(
                manifest["split_invariants"]["development_count"], 2
            )
            self.assertEqual(
                manifest["split_invariants"]["evaluation_count"], 1
            )
            self.assertEqual(
                len(manifest["tracked_sources"]["extractor"]["sha256"]), 64
            )
            with mock.patch.object(sys, "argv", arguments):
                with self.assertRaises(FileExistsError):
                    lock.main()

    def test_lock_rejects_overlapping_or_incomplete_partitions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            arguments = self.make_args(directory)
            evaluation_index = arguments.index("--evaluation-input") + 1
            write_jsonl(Path(arguments[evaluation_index]), ["b", "c"])
            with mock.patch.object(sys, "argv", arguments):
                with self.assertRaisesRegex(ValueError, "overlap"):
                    lock.main()


if __name__ == "__main__":
    unittest.main()
