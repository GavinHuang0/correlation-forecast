from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "relocate_flan_t5_artifacts.py"
SPEC = importlib.util.spec_from_file_location(
    "relocate_flan_t5_artifacts_for_test", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
relocator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(relocator)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@contextlib.contextmanager
def isolated_repository():
    with tempfile.TemporaryDirectory() as temporary_directory:
        repository_root = Path(temporary_directory).resolve() / "repository"
        outputs_root = repository_root / "outputs"
        outputs_root.mkdir(parents=True)
        with (
            mock.patch.object(relocator, "REPOSITORY_ROOT", repository_root),
            mock.patch.object(relocator, "OUTPUTS_ROOT", outputs_root),
        ):
            yield repository_root, outputs_root


def write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


class RelocationPlanningTests(unittest.TestCase):
    def test_version_and_shared_categorization_is_exact(self) -> None:
        with isolated_repository() as (_, outputs_root):
            source_root = outputs_root / "flat_run"
            destination_root = outputs_root / "flan_t5"
            files = {
                "flan_t5_large_predictions.jsonl": b"v0.1",
                "flan_t5_v0_2_core.jsonl": b"v0.2",
                "flan_t5_v0_3_1_development.jsonl": b"v0.3.1",
                "flan_t5_v0_4_evaluation.json": b"v0.4",
                "reference_validation_report.json": b"shared",
                "flan_t5_v0_30_not_a_v0_3_artifact.json": b"unrelated",
                "flan_t5_larger_not_a_v0_1_artifact.json": b"unrelated",
                "unrecognized.txt": b"unrelated",
            }
            for filename, content in files.items():
                write(source_root / filename, content)
            write(source_root / "annotation_batches" / "all_inputs.jsonl", b"batch")
            write(source_root / "previews" / "source_workbook_readme.png", b"preview")

            plan = relocator.build_plan(source_root, destination_root)
            category_by_filename = {
                Path(item["old_path"]).name: item["category"] for item in plan
            }

            self.assertEqual(
                category_by_filename,
                {
                    "flan_t5_large_predictions.jsonl": "legacy/v0_1",
                    "flan_t5_v0_2_core.jsonl": "legacy/v0_2",
                    "flan_t5_v0_3_1_development.jsonl": "legacy/v0_3",
                    "flan_t5_v0_4_evaluation.json": "legacy/v0_4",
                    "reference_validation_report.json": "shared/benchmark_300",
                    "all_inputs.jsonl": "shared/benchmark_300/annotation_batches",
                    "source_workbook_readme.png": "shared/benchmark_300/previews",
                },
            )
            self.assertEqual(len(plan), 7)

    def test_main_dry_run_has_no_filesystem_side_effects(self) -> None:
        with isolated_repository() as (_, outputs_root):
            source_root = outputs_root / "flat_run"
            destination_root = outputs_root / "flan_t5"
            source = source_root / "flan_t5_v0_4_evaluation.json"
            write(source, b"evaluation")
            stdout = io.StringIO()

            with (
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        str(SCRIPT),
                        "--source-root",
                        str(source_root),
                        "--destination-root",
                        str(destination_root),
                    ],
                ),
                contextlib.redirect_stdout(stdout),
            ):
                result = relocator.main()

            report = json.loads(stdout.getvalue())
            self.assertEqual(result, 0)
            self.assertEqual(report["mode"], "dry_run")
            self.assertEqual(report["file_count"], 1)
            self.assertTrue(source.is_file())
            self.assertFalse(destination_root.exists())

    def test_build_plan_rejects_roots_outside_repository_outputs(self) -> None:
        with isolated_repository() as (repository_root, outputs_root):
            source_root = outputs_root / "flat_run"
            destination_root = outputs_root / "flan_t5"
            write(source_root / "flan_t5_v0_2_core.jsonl", b"inside")
            external_root = repository_root / "not_outputs"
            write(external_root / "flan_t5_v0_2_core.jsonl", b"outside")

            with self.assertRaisesRegex(ValueError, "source root must stay inside"):
                relocator.build_plan(external_root, destination_root)
            with self.assertRaisesRegex(
                ValueError, "destination root must stay inside"
            ):
                relocator.build_plan(source_root, external_root / "destination")


class RelocationApplicationTests(unittest.TestCase):
    def test_apply_moves_files_and_writes_verified_hash_manifest(self) -> None:
        with isolated_repository() as (repository_root, outputs_root):
            source_root = outputs_root / "flat_run"
            destination_root = outputs_root / "flan_t5"
            payloads = {
                source_root / "flan_t5_large_predictions.jsonl": b"old model",
                source_root / "flan_t5_v0_4_evaluation.json": b"latest legacy",
                source_root / "reference_validation_report.json": b"reference",
                source_root
                / "annotation_batches"
                / "coarse_evaluation_inputs.jsonl": b"shared inputs",
                source_root
                / "previews"
                / "source_workbook_readme.png": b"preview",
            }
            for path, content in payloads.items():
                write(path, content)
            unrelated = source_root / "leave_me.txt"
            write(unrelated, b"unrelated")

            plan = relocator.build_plan(source_root, destination_root)
            manifest_path = relocator.apply_plan(plan, destination_root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(
                manifest["manifest_version"], "flan-t5-artifact-relocation-v1"
            )
            self.assertEqual(manifest["file_count"], len(payloads))
            self.assertEqual(
                manifest["total_size_bytes"],
                sum(len(content) for content in payloads.values()),
            )
            self.assertEqual(manifest["repository_root"], str(repository_root))
            self.assertEqual(manifest["outputs_root"], str(outputs_root))
            self.assertTrue(manifest["created_at_utc"].endswith("+00:00"))

            manifest_by_old_path = {
                item["old_path"]: item for item in manifest["moves"]
            }
            for old_path, content in payloads.items():
                old_relative = str(old_path.relative_to(repository_root))
                item = manifest_by_old_path[old_relative]
                destination = repository_root / item["new_path"]
                self.assertFalse(old_path.exists())
                self.assertEqual(destination.read_bytes(), content)
                self.assertEqual(item["size_bytes"], len(content))
                self.assertEqual(item["sha256"], digest(content))
                self.assertEqual(relocator.sha256_file(destination), digest(content))

            self.assertEqual(unrelated.read_bytes(), b"unrelated")

    def test_collision_is_refused_during_planning_and_after_planning(self) -> None:
        with isolated_repository() as (_, outputs_root):
            source_root = outputs_root / "flat_run"
            destination_root = outputs_root / "flan_t5"
            source = source_root / "flan_t5_v0_2_core.jsonl"
            destination = (
                destination_root / "legacy" / "v0_2" / source.name
            )
            write(source, b"source")
            write(destination, b"existing")

            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                relocator.build_plan(source_root, destination_root)
            self.assertEqual(source.read_bytes(), b"source")
            self.assertEqual(destination.read_bytes(), b"existing")

            destination.unlink()
            plan = relocator.build_plan(source_root, destination_root)
            write(destination, b"raced")
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                relocator.apply_plan(plan, destination_root)
            self.assertEqual(source.read_bytes(), b"source")
            self.assertEqual(destination.read_bytes(), b"raced")
            self.assertFalse(
                (destination_root / "relocation_manifest.json").exists()
            )

    def test_source_hash_change_after_planning_is_refused_before_any_move(self) -> None:
        with isolated_repository() as (_, outputs_root):
            source_root = outputs_root / "flat_run"
            destination_root = outputs_root / "flan_t5"
            first = source_root / "flan_t5_v0_2_first.jsonl"
            second = source_root / "flan_t5_v0_3_second.jsonl"
            write(first, b"first")
            write(second, b"second")
            plan = relocator.build_plan(source_root, destination_root)
            second.write_bytes(b"changed")

            with self.assertRaisesRegex(ValueError, "changed after planning"):
                relocator.apply_plan(plan, destination_root)

            self.assertEqual(first.read_bytes(), b"first")
            self.assertEqual(second.read_bytes(), b"changed")
            self.assertFalse(destination_root.exists())

    def test_apply_rejects_plan_paths_that_escape_outputs(self) -> None:
        with isolated_repository() as (repository_root, outputs_root):
            destination_root = outputs_root / "flan_t5"
            external_source = repository_root / "external.json"
            write(external_source, b"external")
            malicious_plan = [
                {
                    "category": "legacy/v0_1",
                    "old_path": "external.json",
                    "new_path": "outputs/flan_t5/legacy/v0_1/external.json",
                    "size_bytes": len(b"external"),
                    "sha256": digest(b"external"),
                }
            ]

            with self.assertRaisesRegex(ValueError, "source must stay inside"):
                relocator.apply_plan(malicious_plan, destination_root)
            self.assertEqual(external_source.read_bytes(), b"external")
            self.assertFalse(destination_root.exists())


if __name__ == "__main__":
    unittest.main()
