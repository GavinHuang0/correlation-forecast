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
SCRIPT = ROOT / "scripts" / "finalize_flan_t5_candidate.py"
SPEC = importlib.util.spec_from_file_location(
    "finalize_flan_t5_candidate_for_test", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
finalizer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(finalizer)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@contextlib.contextmanager
def isolated_repository():
    with tempfile.TemporaryDirectory() as temporary_directory:
        repository_root = Path(temporary_directory).resolve() / "repository"
        flan_outputs_root = repository_root / "outputs" / "flan_t5"
        flan_outputs_root.mkdir(parents=True)
        with (
            mock.patch.object(finalizer, "REPOSITORY_ROOT", repository_root),
            mock.patch.object(finalizer, "FLAN_OUTPUTS_ROOT", flan_outputs_root),
        ):
            yield repository_root, flan_outputs_root


def write_candidate(
    flan_outputs_root: Path, *, promoted: bool
) -> tuple[Path, Path, dict[str, bytes]]:
    candidate = flan_outputs_root / "candidates" / "v0_5"
    report = candidate / "final_evaluation_report.json"
    payloads = {
        candidate / "raw" / "scores.jsonl": b'{"score": 1}\n',
        candidate / "calibration.json": b'{"locked": true}\n',
    }
    for path, content in payloads.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps({"evaluation": {"promotion": {"promoted": promoted}}}) + "\n",
        encoding="utf-8",
    )
    payloads[report] = report.read_bytes()
    return candidate, report, payloads


class FinalizerPlanningTests(unittest.TestCase):
    def test_dry_run_reports_promotion_without_mutating(self) -> None:
        with isolated_repository() as (_, flan_outputs_root):
            candidate, report, payloads = write_candidate(
                flan_outputs_root, promoted=True
            )
            stdout = io.StringIO()
            with (
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        str(SCRIPT),
                        "--candidate-dir",
                        str(candidate),
                        "--evaluation-report",
                        str(report),
                    ],
                ),
                contextlib.redirect_stdout(stdout),
            ):
                result = finalizer.main()

            plan = json.loads(stdout.getvalue())
            self.assertEqual(result, 0)
            self.assertEqual(plan["mode"], "promotion")
            self.assertTrue(plan["promoted"])
            self.assertEqual(plan["file_count"], len(payloads))
            self.assertTrue(candidate.is_dir())
            self.assertFalse((flan_outputs_root / "frozen" / "v1_0").exists())
            self.assertFalse((flan_outputs_root / "active_frozen.json").exists())

    def test_report_boolean_is_strictly_required(self) -> None:
        with isolated_repository() as (_, flan_outputs_root):
            candidate, report, _ = write_candidate(
                flan_outputs_root, promoted=True
            )
            report.write_text(
                '{"evaluation":{"promotion":{"promoted":"true"}}}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "must be a JSON boolean"):
                finalizer.build_plan(candidate, report)

            report.write_text('{"evaluation":{}}\n', encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "evaluation.promotion.promoted"
            ):
                finalizer.build_plan(candidate, report)

    def test_candidate_and_report_must_be_in_exact_safe_namespace(self) -> None:
        with isolated_repository() as (repository_root, flan_outputs_root):
            candidate, report, _ = write_candidate(
                flan_outputs_root, promoted=False
            )
            wrong_candidate = flan_outputs_root / "candidates" / "v9"
            wrong_report = wrong_candidate / "report.json"
            wrong_report.parent.mkdir(parents=True)
            wrong_report.write_text(
                '{"evaluation":{"promotion":{"promoted":false}}}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "must be exactly"):
                finalizer.build_plan(wrong_candidate, wrong_report)

            external_report = repository_root / "outside.json"
            external_report.write_text(
                '{"evaluation":{"promotion":{"promoted":false}}}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "must stay inside"):
                finalizer.build_plan(candidate, external_report)

            other_inside_report = flan_outputs_root / "shared" / "report.json"
            other_inside_report.parent.mkdir(parents=True)
            other_inside_report.write_text(
                '{"evaluation":{"promotion":{"promoted":false}}}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "evaluation report must stay inside"
            ):
                finalizer.build_plan(candidate, other_inside_report)


class FinalizerApplicationTests(unittest.TestCase):
    def test_promotion_moves_and_verifies_tree_then_creates_pointer(self) -> None:
        with isolated_repository() as (repository_root, flan_outputs_root):
            candidate, report, payloads = write_candidate(
                flan_outputs_root, promoted=True
            )
            plan = finalizer.build_plan(candidate, report)
            manifest_path, pointer_path = finalizer.apply_plan(plan)

            destination = flan_outputs_root / "frozen" / "v1_0"
            self.assertFalse(candidate.exists())
            self.assertEqual(manifest_path, destination / "artifact_manifest.json")
            self.assertEqual(pointer_path, flan_outputs_root / "active_frozen.json")

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["manifest_version"], "flan-t5-finalized-artifact-v1"
            )
            self.assertTrue(manifest["promoted"])
            self.assertEqual(manifest["outcome"], "promoted")
            self.assertEqual(manifest["file_count_before_manifest"], len(payloads))
            manifest_files = {item["path"]: item for item in manifest["files"]}
            for original, content in payloads.items():
                relative = original.relative_to(candidate).as_posix()
                moved = destination / relative
                self.assertEqual(moved.read_bytes(), content)
                self.assertEqual(manifest_files[relative]["size_bytes"], len(content))
                self.assertEqual(manifest_files[relative]["sha256"], digest(content))

            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            self.assertEqual(pointer["active_version"], "v1_0")
            self.assertEqual(
                pointer["artifact_directory"], "outputs/flan_t5/frozen/v1_0"
            )
            self.assertEqual(
                pointer["artifact_manifest_sha256"],
                finalizer.sha256_file(manifest_path),
            )
            self.assertTrue(
                (repository_root / pointer["evaluation_report"]).is_file()
            )

    def test_rejection_moves_to_rejected_and_does_not_create_pointer(self) -> None:
        with isolated_repository() as (_, flan_outputs_root):
            candidate, report, _ = write_candidate(
                flan_outputs_root, promoted=False
            )
            plan = finalizer.build_plan(candidate, report)
            manifest_path, pointer_path = finalizer.apply_plan(plan)

            self.assertEqual(
                manifest_path,
                flan_outputs_root
                / "rejected"
                / "v0_5"
                / "artifact_manifest.json",
            )
            self.assertIsNone(pointer_path)
            self.assertFalse((flan_outputs_root / "active_frozen.json").exists())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertFalse(manifest["promoted"])
            self.assertEqual(manifest["outcome"], "rejected")

    def test_existing_destination_or_active_pointer_is_refused_before_move(self) -> None:
        with isolated_repository() as (_, flan_outputs_root):
            candidate, report, _ = write_candidate(
                flan_outputs_root, promoted=True
            )
            destination = flan_outputs_root / "frozen" / "v1_0"
            destination.mkdir(parents=True)
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                finalizer.build_plan(candidate, report)
            self.assertTrue(candidate.is_dir())

            destination.rmdir()
            pointer = flan_outputs_root / "active_frozen.json"
            pointer.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "active frozen pointer"):
                finalizer.build_plan(candidate, report)
            self.assertTrue(candidate.is_dir())

    def test_source_change_after_planning_is_refused_before_move(self) -> None:
        with isolated_repository() as (_, flan_outputs_root):
            candidate, report, _ = write_candidate(
                flan_outputs_root, promoted=False
            )
            plan = finalizer.build_plan(candidate, report)
            (candidate / "calibration.json").write_bytes(b"changed")

            with self.assertRaisesRegex(ValueError, "changed during"):
                finalizer.apply_plan(plan)
            self.assertTrue(candidate.is_dir())
            self.assertFalse((flan_outputs_root / "rejected" / "v0_5").exists())

    def test_added_source_file_after_planning_is_refused_before_move(self) -> None:
        with isolated_repository() as (_, flan_outputs_root):
            candidate, report, _ = write_candidate(
                flan_outputs_root, promoted=False
            )
            plan = finalizer.build_plan(candidate, report)
            (candidate / "late.txt").write_text("late", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "file set changed"):
                finalizer.apply_plan(plan)
            self.assertTrue(candidate.is_dir())


if __name__ == "__main__":
    unittest.main()
