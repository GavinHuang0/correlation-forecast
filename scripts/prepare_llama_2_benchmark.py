"""Prepare the fixed FLAN benchmark for a leakage-safe Llama 2 comparison.

This script does not call, download, or import an LLM.  It validates the
existing 300-article benchmark and byte-copies only the two extractor-input
splits.  GPT silver labels remain in their separate annotation file and are
never copied into either extractor input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import coarse_news_features as coarse
import extract_flan_t5 as base


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = (
    REPOSITORY_ROOT
    / "outputs"
    / "flan_t5"
    / "shared"
    / "benchmark_300"
    / "annotation_batches"
)
DEFAULT_OUTPUT_ROOT = (
    REPOSITORY_ROOT / "outputs" / "llama_2" / "shared" / "benchmark_300"
)
DEFAULT_ALL_INPUTS = SOURCE_ROOT / "all_inputs.jsonl"
DEFAULT_DEVELOPMENT_INPUTS = SOURCE_ROOT / "coarse_v0_3_development_inputs.jsonl"
DEFAULT_EVALUATION_INPUTS = SOURCE_ROOT / "coarse_v0_3_evaluation_inputs.jsonl"
DEFAULT_FINE_REFERENCE = REPOSITORY_ROOT / "annotations" / "chatgpt_5_6_sol_reference.jsonl"
DEFAULT_COARSE_REFERENCE = (
    REPOSITORY_ROOT / "annotations" / "chatgpt_5_6_sol_reference_coarse_v0_2.jsonl"
)
DEFAULT_SCHEMA = REPOSITORY_ROOT / "config" / "news_feature_schema_coarse.json"

KNOWLEDGE_CUTOFF = date(2023, 7, 31)
MANIFEST_VERSION = "llama-2-benchmark-v1"

EXPECTED_SOURCE_HASHES = {
    "all_inputs": "7254ae0efc78e6698161f1b9a638c232c1f511a160492b77810fb8949b7842d6",
    "development_inputs": "44920068fa6806f2397609b9181d96e506ee0f729c8b1a42dd950e85278b4014",
    "evaluation_inputs": "92cf5da0dcb26d186f416885528f6453fac81b26fa5b277025063b4ae070ac83",
    "fine_reference": "94ae2481611fc2a88ce3ff2a16c6c2945c4083be2ef787810c1d1bb1f50967b7",
    "coarse_reference": "d765d236c423a7f3ccebcc1c9b3336462458b40862c69f554e01a6fa4bd97ce1",
    "schema": "201e19a0f2b9723e690c3a62eb145213b7e6d8c106a09e0da7374b71830d05fa",
}


@dataclass(frozen=True)
class BenchmarkPaths:
    all_inputs: Path
    development_inputs: Path
    evaluation_inputs: Path
    fine_reference: Path
    coarse_reference: Path
    schema: Path


@dataclass(frozen=True)
class BenchmarkContract:
    expected_hashes: Mapping[str, str]
    all_count: int = 300
    development_count: int = 72
    evaluation_count: int = 228
    knowledge_cutoff: date = KNOWLEDGE_CUTOFF


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and byte-copy the fixed 72/228 FLAN benchmark splits for "
            "Llama 2 extraction. No model is loaded."
        )
    )
    parser.add_argument("--all-inputs", type=Path, default=DEFAULT_ALL_INPUTS)
    parser.add_argument(
        "--development-inputs", type=Path, default=DEFAULT_DEVELOPMENT_INPUTS
    )
    parser.add_argument("--evaluation-inputs", type=Path, default=DEFAULT_EVALUATION_INPUTS)
    parser.add_argument("--fine-reference", type=Path, default=DEFAULT_FINE_REFERENCE)
    parser.add_argument("--coarse-reference", type=Path, default=DEFAULT_COARSE_REFERENCE)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run every source, split, cutoff, and reference check without writing files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing prepared benchmark.",
    )
    return parser.parse_args()


def _path_for_manifest(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _id_order_sha256(records: list[dict[str, Any]]) -> str:
    return base.sha256_text("\n".join(record["article_id"] for record in records))


def _records_by_id(
    records: list[dict[str, Any]], *, source_name: str
) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records, start=1):
        article_id = record.get("article_id")
        if not isinstance(article_id, str) or not article_id:
            raise ValueError(f"{source_name} row {index} has no valid article_id")
        if article_id in by_id:
            raise ValueError(f"{source_name} contains duplicate article_id {article_id!r}")
        by_id[article_id] = record
    return by_id


def _parse_utc_timestamp(value: Any, *, article_id: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{article_id}: time_published_utc must be a nonempty string")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(
            f"{article_id}: invalid time_published_utc {value!r}"
        ) from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{article_id}: time_published_utc must include a timezone")
    return parsed.astimezone(timezone.utc)


def _contains_annotation_labels(value: Any) -> bool:
    if isinstance(value, dict):
        if "labels" in value:
            return True
        return any(_contains_annotation_labels(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_annotation_labels(item) for item in value)
    return False


def _validate_hashes(
    paths: BenchmarkPaths, expected_hashes: Mapping[str, str]
) -> dict[str, str]:
    actual = {
        "all_inputs": base.sha256_file(paths.all_inputs),
        "development_inputs": base.sha256_file(paths.development_inputs),
        "evaluation_inputs": base.sha256_file(paths.evaluation_inputs),
        "fine_reference": base.sha256_file(paths.fine_reference),
        "coarse_reference": base.sha256_file(paths.coarse_reference),
        "schema": base.sha256_file(paths.schema),
    }
    missing = set(actual) - set(expected_hashes)
    if missing:
        raise ValueError(f"Expected-hash contract is missing {sorted(missing)!r}")
    mismatches = {
        name: {"expected": expected_hashes[name], "actual": digest}
        for name, digest in actual.items()
        if digest.lower() != expected_hashes[name].lower()
    }
    if mismatches:
        raise ValueError(
            "Benchmark source hash mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return actual


def validate_benchmark_sources(
    paths: BenchmarkPaths,
    *,
    contract: BenchmarkContract | None = None,
) -> dict[str, Any]:
    """Validate the immutable source benchmark and return manifest-ready facts."""

    selected_contract = contract or BenchmarkContract(EXPECTED_SOURCE_HASHES)
    for name, path in vars(paths).items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing benchmark source {name}: {path}")
    hashes = _validate_hashes(paths, selected_contract.expected_hashes)

    schema = coarse.load_schema(paths.schema)
    all_records = base.read_jsonl(paths.all_inputs)
    development = base.read_jsonl(paths.development_inputs)
    evaluation = base.read_jsonl(paths.evaluation_inputs)
    fine_reference = base.read_jsonl(paths.fine_reference)
    coarse_reference = base.read_jsonl(paths.coarse_reference)

    expected_counts = {
        "all": selected_contract.all_count,
        "development": selected_contract.development_count,
        "evaluation": selected_contract.evaluation_count,
    }
    actual_counts = {
        "all": len(all_records),
        "development": len(development),
        "evaluation": len(evaluation),
    }
    if actual_counts != expected_counts:
        raise ValueError(
            f"Benchmark counts differ from the fixed contract: "
            f"expected={expected_counts!r}, actual={actual_counts!r}"
        )

    all_by_id = _records_by_id(all_records, source_name="all inputs")
    development_by_id = _records_by_id(development, source_name="development inputs")
    evaluation_by_id = _records_by_id(evaluation, source_name="evaluation inputs")
    fine_by_id = _records_by_id(fine_reference, source_name="fine reference")
    coarse_by_id = _records_by_id(coarse_reference, source_name="coarse reference")

    development_ids = set(development_by_id)
    evaluation_ids = set(evaluation_by_id)
    all_ids = set(all_by_id)
    overlap = development_ids & evaluation_ids
    if overlap:
        raise ValueError(
            f"Development/evaluation split overlap contains {len(overlap)} article IDs"
        )
    if development_ids | evaluation_ids != all_ids:
        missing = all_ids - (development_ids | evaluation_ids)
        unexpected = (development_ids | evaluation_ids) - all_ids
        raise ValueError(
            "Development/evaluation IDs are not exhaustive: "
            f"missing={sorted(missing)[:10]!r}, unexpected={sorted(unexpected)[:10]!r}"
        )
    if set(fine_by_id) != all_ids or set(coarse_by_id) != all_ids:
        raise ValueError("Fine/coarse reference article IDs must exactly match all inputs")

    ordered_all_ids = [record["article_id"] for record in all_records]
    expected_development_order = [
        article_id for article_id in ordered_all_ids if article_id in development_ids
    ]
    expected_evaluation_order = [
        article_id for article_id in ordered_all_ids if article_id in evaluation_ids
    ]
    if [record["article_id"] for record in development] != expected_development_order:
        raise ValueError("Development split order differs from the fixed all-input order")
    if [record["article_id"] for record in evaluation] != expected_evaluation_order:
        raise ValueError("Evaluation split order differs from the fixed all-input order")

    for split_name, split_by_id in (
        ("development", development_by_id),
        ("evaluation", evaluation_by_id),
    ):
        for article_id, record in split_by_id.items():
            if record != all_by_id[article_id]:
                raise ValueError(
                    f"{split_name} input {article_id} differs from its all-input record"
                )
            if _contains_annotation_labels(record):
                raise ValueError(
                    f"{split_name} extractor input {article_id} contains annotation labels"
                )

    timestamps: list[datetime] = []
    for article_id, input_record in all_by_id.items():
        coarse.validate_input_record(input_record)
        timestamp = _parse_utc_timestamp(
            input_record.get("time_published_utc"), article_id=article_id
        )
        if timestamp.date() <= selected_contract.knowledge_cutoff:
            raise ValueError(
                f"{article_id}: publication date {timestamp.date()} is not after "
                f"the declared cutoff {selected_contract.knowledge_cutoff}"
            )
        timestamps.append(timestamp)

        target_ticker = input_record["target"]["ticker"]
        fine = fine_by_id[article_id]
        mapped = coarse_by_id[article_id]
        if fine.get("target_ticker") != target_ticker:
            raise ValueError(f"{article_id}: fine-reference target ticker mismatch")
        if mapped.get("target_ticker") != target_ticker:
            raise ValueError(f"{article_id}: coarse-reference target ticker mismatch")
        expected_labels = coarse.map_fine_labels(fine.get("labels", {}))
        if mapped.get("labels") != expected_labels:
            raise ValueError(f"{article_id}: coarse reference does not match fine mapping")
        expected_applicable = coarse.semantic_applicable(fine["labels"])
        if mapped.get("semantic_applicable") is not expected_applicable:
            raise ValueError(f"{article_id}: coarse reference applicability mismatch")
        coarse.validate_coarse_labels(mapped["labels"], schema)

    return {
        "manifest_version": MANIFEST_VERSION,
        "dataset_name": "Llama 2 reuse of fixed FLAN 300-article benchmark",
        "reference_type": "deterministically coarsened GPT-5.6 Sol silver annotations",
        "warning": (
            "These are silver labels, not human ground truth. The 72-record development "
            "split may be used for prompt development; the 228-record evaluation split "
            "must remain untouched until the prompt and model configuration are frozen."
        ),
        "model_eligibility_basis": {
            "model_family": "Llama 2",
            "latest_reported_training_data": "July 2023",
            "conservative_cutoff": selected_contract.knowledge_cutoff.isoformat(),
        },
        "knowledge_cutoff": selected_contract.knowledge_cutoff.isoformat(),
        "eligibility_rule": "time_published_utc calendar date must be after knowledge_cutoff",
        "minimum_time_published_utc": min(timestamps).isoformat().replace("+00:00", "Z"),
        "maximum_time_published_utc": max(timestamps).isoformat().replace("+00:00", "Z"),
        "all_articles_after_cutoff": True,
        "labels_present_in_extractor_inputs": False,
        "counts": {
            "all": len(all_records),
            "development": len(development),
            "evaluation": len(evaluation),
        },
        "split_integrity": {
            "development_evaluation_disjoint": True,
            "development_evaluation_exhaustive": True,
            "input_records_identical_to_all_inputs": True,
            "order_preserved_from_all_inputs": True,
        },
        "source_files": {
            name: {
                "path": _path_for_manifest(getattr(paths, name)),
                "sha256": hashes[name],
            }
            for name in (
                "all_inputs",
                "development_inputs",
                "evaluation_inputs",
                "fine_reference",
                "coarse_reference",
                "schema",
            )
        },
        "schema": {
            "name": schema["schema_name"],
            "version": schema["schema_version"],
            "coarse_fields": list(coarse.COARSE_FIELDS),
        },
        "article_id_order_sha256": {
            "all": _id_order_sha256(all_records),
            "development": _id_order_sha256(development),
            "evaluation": _id_order_sha256(evaluation),
        },
    }


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_benchmark(
    paths: BenchmarkPaths,
    output_root: Path,
    *,
    contract: BenchmarkContract | None = None,
    validate_only: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Validate sources and optionally materialize byte-identical input copies."""

    manifest = validate_benchmark_sources(paths, contract=contract)
    development_output = output_root / "development_inputs.jsonl"
    evaluation_output = output_root / "evaluation_inputs.jsonl"
    manifest_output = output_root / "manifest.json"
    outputs = (development_output, evaluation_output, manifest_output)
    if validate_only:
        return {**manifest, "mode": "validate_only"}

    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Prepared Llama benchmark already exists; pass --overwrite to replace: "
            + ", ".join(str(path) for path in existing)
        )

    _atomic_copy(paths.development_inputs, development_output)
    _atomic_copy(paths.evaluation_inputs, evaluation_output)
    if development_output.read_bytes() != paths.development_inputs.read_bytes():
        raise RuntimeError("Development input byte-copy verification failed")
    if evaluation_output.read_bytes() != paths.evaluation_inputs.read_bytes():
        raise RuntimeError("Evaluation input byte-copy verification failed")

    completed_manifest = {
        **manifest,
        "mode": "prepared",
        "prepared_outputs": {
            "development_inputs": {
                "path": _path_for_manifest(development_output),
                "sha256": base.sha256_file(development_output),
                "byte_identical_to_source": True,
            },
            "evaluation_inputs": {
                "path": _path_for_manifest(evaluation_output),
                "sha256": base.sha256_file(evaluation_output),
                "byte_identical_to_source": True,
            },
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(
        json.dumps(completed_manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return completed_manifest


def main() -> int:
    args = parse_args()
    paths = BenchmarkPaths(
        all_inputs=args.all_inputs,
        development_inputs=args.development_inputs,
        evaluation_inputs=args.evaluation_inputs,
        fine_reference=args.fine_reference,
        coarse_reference=args.coarse_reference,
        schema=args.schema,
    )
    manifest = prepare_benchmark(
        paths,
        args.output_root,
        validate_only=args.validate_only,
        overwrite=args.overwrite,
    )
    if args.validate_only:
        print(json.dumps(manifest, indent=2))
    else:
        print(
            "Prepared byte-identical Llama benchmark inputs at "
            f"{args.output_root} ({manifest['counts']['development']} development, "
            f"{manifest['counts']['evaluation']} evaluation)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
