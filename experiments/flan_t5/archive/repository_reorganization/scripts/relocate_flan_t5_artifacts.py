from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = (REPOSITORY_ROOT / "outputs").resolve()

SHARED_FILENAMES = {
    "alpha_vantage_news_annotation_benchmark.xlsx",
    "alpha_vantage_news_annotation_benchmark.xlsx.inspect.ndjson",
    "chatgpt_reference_news_benchmark.xlsx",
    "chatgpt_reference_news_benchmark.xlsx.inspect.ndjson",
    "chatgpt_reference_news_benchmark.xlsx.summary.png",
    "reference_validation_report.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Relocate the original flat FLAN output directory into shared and "
            "versioned legacy namespaces, with a verified SHA-256 manifest."
        )
    )
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument(
        "--destination-root",
        default=Path("outputs/flan_t5"),
        type=Path,
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform the moves. Without this flag, print a dry-run plan.",
    )
    return parser.parse_args()


def require_within(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside {root}: {resolved}") from exc
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def legacy_version(filename: str) -> str | None:
    def matches_artifact_prefix(prefix: str) -> bool:
        return filename == prefix or filename.startswith((f"{prefix}_", f"{prefix}."))

    if matches_artifact_prefix("flan_t5_large"):
        return "v0_1"
    for version in ("v0_2", "v0_3", "v0_4"):
        if matches_artifact_prefix(f"flan_t5_{version}"):
            return version
    return None


def build_plan(source_root: Path, destination_root: Path) -> list[dict[str, Any]]:
    source_root = require_within(source_root, OUTPUTS_ROOT, "source root")
    destination_root = require_within(
        destination_root, OUTPUTS_ROOT, "destination root"
    )
    if source_root == destination_root:
        raise ValueError("Source and destination roots must differ")
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")

    moves: list[tuple[Path, Path, str]] = []
    for source in sorted(source_root.iterdir(), key=lambda path: path.name):
        if not source.is_file():
            continue
        version = legacy_version(source.name)
        if version is not None:
            destination = destination_root / "legacy" / version / source.name
            category = f"legacy/{version}"
        elif source.name in SHARED_FILENAMES:
            destination = destination_root / "shared" / "benchmark_300" / source.name
            category = "shared/benchmark_300"
        else:
            continue
        moves.append((source, destination, category))

    annotation_batches = source_root / "annotation_batches"
    if annotation_batches.exists():
        if not annotation_batches.is_dir():
            raise ValueError(f"Expected a directory: {annotation_batches}")
        for source in sorted(annotation_batches.iterdir(), key=lambda path: path.name):
            if not source.is_file():
                raise ValueError(f"Unexpected nested entry in annotation_batches: {source}")
            destination = (
                destination_root
                / "shared"
                / "benchmark_300"
                / "annotation_batches"
                / source.name
            )
            moves.append((source, destination, "shared/benchmark_300/annotation_batches"))

    previews = source_root / "previews"
    if previews.exists():
        if not previews.is_dir():
            raise ValueError(f"Expected a directory: {previews}")
        for source in sorted(previews.iterdir(), key=lambda path: path.name):
            if not source.is_file():
                raise ValueError(f"Unexpected nested entry in previews: {source}")
            destination = (
                destination_root
                / "shared"
                / "benchmark_300"
                / "previews"
                / source.name
            )
            moves.append((source, destination, "shared/benchmark_300/previews"))

    plan: list[dict[str, Any]] = []
    seen_destinations: set[Path] = set()
    for source, destination, category in moves:
        source = require_within(source, OUTPUTS_ROOT, "source")
        destination = require_within(destination, OUTPUTS_ROOT, "destination")
        if destination in seen_destinations:
            raise ValueError(f"Duplicate destination in relocation plan: {destination}")
        seen_destinations.add(destination)
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite relocation target: {destination}")
        plan.append(
            {
                "category": category,
                "old_path": str(source.relative_to(REPOSITORY_ROOT)),
                "new_path": str(destination.relative_to(REPOSITORY_ROOT)),
                "size_bytes": source.stat().st_size,
                "sha256": sha256_file(source),
            }
        )
    if not plan:
        raise ValueError(f"No recognized FLAN artifacts found under {source_root}")
    return plan


def apply_plan(plan: list[dict[str, Any]], destination_root: Path) -> Path:
    destination_root = require_within(
        destination_root, OUTPUTS_ROOT, "destination root"
    )
    manifest_path = destination_root / "relocation_manifest.json"
    manifest_path = require_within(manifest_path, OUTPUTS_ROOT, "manifest")
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite relocation manifest: {manifest_path}")

    for item in plan:
        source = require_within(REPOSITORY_ROOT / item["old_path"], OUTPUTS_ROOT, "source")
        destination = require_within(
            REPOSITORY_ROOT / item["new_path"], OUTPUTS_ROOT, "destination"
        )
        if not source.is_file():
            raise FileNotFoundError(f"Relocation source disappeared: {source}")
        if source.stat().st_size != item["size_bytes"] or sha256_file(source) != item["sha256"]:
            raise ValueError(f"Relocation source changed after planning: {source}")
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite relocation target: {destination}")

    for item in plan:
        source = require_within(
            REPOSITORY_ROOT / item["old_path"], OUTPUTS_ROOT, "source"
        )
        destination = require_within(
            REPOSITORY_ROOT / item["new_path"], OUTPUTS_ROOT, "destination"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        if (
            not destination.is_file()
            or destination.stat().st_size != item["size_bytes"]
            or sha256_file(destination) != item["sha256"]
        ):
            raise RuntimeError(f"Post-move verification failed: {destination}")

    manifest = {
        "manifest_version": "flan-t5-artifact-relocation-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(REPOSITORY_ROOT),
        "outputs_root": str(OUTPUTS_ROOT),
        "file_count": len(plan),
        "total_size_bytes": sum(item["size_bytes"] for item in plan),
        "moves": plan,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def main() -> int:
    args = parse_args()
    source_root = require_within(args.source_root, OUTPUTS_ROOT, "source root")
    destination_root = require_within(
        args.destination_root, OUTPUTS_ROOT, "destination root"
    )
    if source_root == destination_root:
        raise ValueError("Source and destination roots must differ")
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")
    plan = build_plan(source_root, destination_root)
    if not args.apply:
        print(json.dumps({"mode": "dry_run", "file_count": len(plan), "moves": plan}, indent=2))
        return 0
    manifest_path = apply_plan(plan, destination_root)
    print(f"Relocated and verified {len(plan)} files; manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
