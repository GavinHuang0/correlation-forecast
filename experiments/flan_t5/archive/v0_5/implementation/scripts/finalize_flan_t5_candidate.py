from __future__ import annotations

"""Safely archive the v0.5 FLAN candidate according to its locked evaluation.

The command is a dry run unless ``--apply`` is supplied.  It moves only the
ignored run artifacts under ``outputs/flan_t5``; tracked experiment sources and
documentation are never modified.
"""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FLAN_OUTPUTS_ROOT = (REPOSITORY_ROOT / "outputs" / "flan_t5").resolve()
CANDIDATE_RELATIVE_PATH = Path("candidates") / "v0_5"
PROMOTED_RELATIVE_PATH = Path("frozen") / "v1_0"
REJECTED_RELATIVE_PATH = Path("rejected") / "v0_5"
ARTIFACT_MANIFEST_NAME = "artifact_manifest.json"
ACTIVE_POINTER_NAME = "active_frozen.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=FLAN_OUTPUTS_ROOT / CANDIDATE_RELATIVE_PATH,
        help="The v0.5 candidate artifact directory.",
    )
    parser.add_argument(
        "--evaluation-report",
        required=True,
        type=Path,
        help="Final JSON report containing evaluation.promotion.promoted.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform the verified move. Without this flag, only print the plan.",
    )
    return parser.parse_args()


def require_within(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside {root}: {resolved}") from exc
    return resolved


def repository_relative(path: Path) -> str:
    return require_within(path, REPOSITORY_ROOT, "artifact path").relative_to(
        REPOSITORY_ROOT
    ).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_promotion(report_path: Path) -> bool:
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Evaluation report is not valid JSON: {report_path}") from exc
    try:
        promoted = report["evaluation"]["promotion"]["promoted"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Evaluation report must contain evaluation.promotion.promoted"
        ) from exc
    if not isinstance(promoted, bool):
        raise ValueError("evaluation.promotion.promoted must be a JSON boolean")
    return promoted


def reject_symlinks_and_list_files(candidate_dir: Path) -> list[Path]:
    """Return ordinary files without following or accepting any symlink."""

    files: list[Path] = []
    for directory, directory_names, file_names in os.walk(
        candidate_dir, topdown=True, followlinks=False
    ):
        current = Path(directory)
        for name in directory_names:
            path = current / name
            if path.is_symlink():
                raise ValueError(f"Candidate directory contains a symlink: {path}")
        for name in file_names:
            path = current / name
            if path.is_symlink():
                raise ValueError(f"Candidate directory contains a symlink: {path}")
            resolved = require_within(path, candidate_dir, "candidate file")
            if not resolved.is_file():
                raise ValueError(f"Unexpected non-file candidate entry: {path}")
            files.append(resolved)
    return sorted(files, key=lambda path: path.relative_to(candidate_dir).as_posix())


def snapshot_candidate(candidate_dir: Path) -> list[dict[str, Any]]:
    files = reject_symlinks_and_list_files(candidate_dir)
    if not files:
        raise ValueError(f"Candidate directory contains no files: {candidate_dir}")
    return [
        {
            "path": path.relative_to(candidate_dir).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]


def verify_snapshot(root: Path, files: list[dict[str, Any]], phase: str) -> None:
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in reject_symlinks_and_list_files(root)
    }
    expected_paths = {item["path"] for item in files}
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths)
        extra = sorted(actual_paths - expected_paths)
        raise ValueError(
            f"Candidate file set changed during {phase}; "
            f"missing={missing}, extra={extra}"
        )
    for item in files:
        path = require_within(root / item["path"], root, "candidate file")
        if (
            path.stat().st_size != item["size_bytes"]
            or sha256_file(path) != item["sha256"]
        ):
            raise ValueError(f"Candidate file changed during {phase}: {path}")


def build_plan(candidate_dir: Path, evaluation_report: Path) -> dict[str, Any]:
    candidate_input = candidate_dir.absolute()
    if candidate_input.is_symlink():
        raise ValueError(f"Candidate directory may not be a symlink: {candidate_input}")
    candidate_dir = require_within(
        candidate_dir, FLAN_OUTPUTS_ROOT, "candidate directory"
    )
    expected_candidate = (FLAN_OUTPUTS_ROOT / CANDIDATE_RELATIVE_PATH).resolve()
    if candidate_dir != expected_candidate:
        raise ValueError(
            "Candidate directory must be exactly "
            f"{expected_candidate}, not {candidate_dir}"
        )
    if candidate_dir.is_symlink() or not candidate_dir.is_dir():
        raise FileNotFoundError(f"Candidate directory does not exist: {candidate_dir}")

    report_input = evaluation_report.absolute()
    if report_input.is_symlink():
        raise ValueError(f"Evaluation report may not be a symlink: {report_input}")
    evaluation_report = require_within(
        evaluation_report, FLAN_OUTPUTS_ROOT, "evaluation report"
    )
    evaluation_report = require_within(
        evaluation_report, candidate_dir, "evaluation report"
    )
    if evaluation_report.is_symlink() or not evaluation_report.is_file():
        raise FileNotFoundError(f"Evaluation report does not exist: {evaluation_report}")

    promoted = read_promotion(evaluation_report)
    destination_relative = (
        PROMOTED_RELATIVE_PATH if promoted else REJECTED_RELATIVE_PATH
    )
    destination = require_within(
        FLAN_OUTPUTS_ROOT / destination_relative,
        FLAN_OUTPUTS_ROOT,
        "destination",
    )
    manifest_path = require_within(
        destination / ARTIFACT_MANIFEST_NAME,
        FLAN_OUTPUTS_ROOT,
        "artifact manifest",
    )
    active_pointer = require_within(
        FLAN_OUTPUTS_ROOT / ACTIVE_POINTER_NAME,
        FLAN_OUTPUTS_ROOT,
        "active frozen pointer",
    )

    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite finalization target: {destination}")
    if (candidate_dir / ARTIFACT_MANIFEST_NAME).exists():
        raise FileExistsError(
            "Candidate already contains the reserved artifact manifest name: "
            f"{candidate_dir / ARTIFACT_MANIFEST_NAME}"
        )
    if promoted and active_pointer.exists():
        raise FileExistsError(
            f"Refusing to replace existing active frozen pointer: {active_pointer}"
        )

    files = snapshot_candidate(candidate_dir)
    report_relative = evaluation_report.relative_to(candidate_dir).as_posix()
    report_item = next((item for item in files if item["path"] == report_relative), None)
    if report_item is None:
        raise ValueError("Evaluation report is not present in the candidate snapshot")

    return {
        "mode": "promotion" if promoted else "rejection",
        "promoted": promoted,
        "candidate_directory": repository_relative(candidate_dir),
        "destination_directory": repository_relative(destination),
        "evaluation_report": repository_relative(evaluation_report),
        "evaluation_report_relative_to_candidate": report_relative,
        "evaluation_report_sha256": report_item["sha256"],
        "artifact_manifest": repository_relative(manifest_path),
        "active_frozen_pointer": (
            repository_relative(active_pointer) if promoted else None
        ),
        "file_count": len(files),
        "total_size_bytes": sum(item["size_bytes"] for item in files),
        "candidate_tree_sha256": sha256_json(files),
        "files": files,
    }


def write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def apply_plan(plan: dict[str, Any]) -> tuple[Path, Path | None]:
    source_input = REPOSITORY_ROOT / plan["candidate_directory"]
    if source_input.is_symlink():
        raise ValueError(f"Candidate directory may not be a symlink: {source_input}")
    source = require_within(
        source_input,
        FLAN_OUTPUTS_ROOT,
        "candidate directory",
    )
    expected_source = (FLAN_OUTPUTS_ROOT / CANDIDATE_RELATIVE_PATH).resolve()
    if source != expected_source:
        raise ValueError(f"Plan source is not the v0.5 candidate: {source}")
    destination = require_within(
        REPOSITORY_ROOT / plan["destination_directory"],
        FLAN_OUTPUTS_ROOT,
        "destination",
    )
    expected_destination = (
        FLAN_OUTPUTS_ROOT
        / (PROMOTED_RELATIVE_PATH if plan["promoted"] else REJECTED_RELATIVE_PATH)
    ).resolve()
    if destination != expected_destination:
        raise ValueError(f"Plan destination does not match its outcome: {destination}")
    manifest_path = require_within(
        destination / ARTIFACT_MANIFEST_NAME,
        FLAN_OUTPUTS_ROOT,
        "artifact manifest",
    )
    active_pointer = require_within(
        FLAN_OUTPUTS_ROOT / ACTIVE_POINTER_NAME,
        FLAN_OUTPUTS_ROOT,
        "active frozen pointer",
    )

    if not source.is_dir():
        raise FileNotFoundError(f"Candidate directory disappeared: {source}")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite finalization target: {destination}")
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite artifact manifest: {manifest_path}")
    if plan["promoted"] and active_pointer.exists():
        raise FileExistsError(
            f"Refusing to replace existing active frozen pointer: {active_pointer}"
        )

    verify_snapshot(source, plan["files"], "pre-move verification")
    report = require_within(
        source / plan["evaluation_report_relative_to_candidate"],
        source,
        "evaluation report",
    )
    if read_promotion(report) is not plan["promoted"]:
        raise ValueError("Evaluation promotion decision changed after planning")
    if sha256_file(report) != plan["evaluation_report_sha256"]:
        raise ValueError("Evaluation report changed after planning")

    destination.parent.mkdir(parents=True, exist_ok=True)
    # Both paths are in the same repository output tree.  A direct rename avoids
    # shutil.move's surprising "move inside the existing directory" behavior if
    # a destination appears after the collision check.
    source.rename(destination)
    if source.exists() or not destination.is_dir():
        raise RuntimeError("Candidate directory move did not complete")
    verify_snapshot(destination, plan["files"], "post-move verification")

    created_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "manifest_version": "flan-t5-finalized-artifact-v1",
        "created_at_utc": created_at,
        "outcome": "promoted" if plan["promoted"] else "rejected",
        "promoted": plan["promoted"],
        "original_candidate_directory": plan["candidate_directory"],
        "artifact_directory": plan["destination_directory"],
        "evaluation_report": (
            Path(plan["destination_directory"])
            / plan["evaluation_report_relative_to_candidate"]
        ).as_posix(),
        "evaluation_report_sha256": plan["evaluation_report_sha256"],
        "file_count_before_manifest": plan["file_count"],
        "total_size_bytes_before_manifest": plan["total_size_bytes"],
        "candidate_tree_sha256": plan["candidate_tree_sha256"],
        "files": plan["files"],
    }
    write_json_exclusive(manifest_path, manifest)

    pointer_path: Path | None = None
    if plan["promoted"]:
        pointer = {
            "pointer_version": "flan-t5-active-frozen-v1",
            "created_at_utc": created_at,
            "active_version": "v1_0",
            "artifact_directory": repository_relative(destination),
            "artifact_manifest": repository_relative(manifest_path),
            "artifact_manifest_sha256": sha256_file(manifest_path),
            "evaluation_report": manifest["evaluation_report"],
            "evaluation_report_sha256": plan["evaluation_report_sha256"],
        }
        write_json_exclusive(active_pointer, pointer)
        pointer_path = active_pointer
    return manifest_path, pointer_path


def main() -> int:
    args = parse_args()
    plan = build_plan(args.candidate_dir, args.evaluation_report)
    if not args.apply:
        print(json.dumps({"mode": "dry_run", **plan}, indent=2, sort_keys=True))
        return 0
    manifest_path, pointer_path = apply_plan(plan)
    outcome = "promoted" if plan["promoted"] else "rejected"
    print(
        f"Verified and {outcome} {plan['file_count']} candidate files; "
        f"manifest: {manifest_path}"
    )
    if pointer_path is not None:
        print(f"Created active frozen pointer: {pointer_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
