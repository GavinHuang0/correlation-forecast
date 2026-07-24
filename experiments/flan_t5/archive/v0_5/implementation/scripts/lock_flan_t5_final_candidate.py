from __future__ import annotations

"""Create the immutable pre-score manifest for the final FLAN candidate.

The lock contains hashes and ID assignments, but never copies article text or
reference labels into the tracked experiment directory.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


MODEL_ID = "google/flan-t5-large"
MODEL_REVISION = "0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--promotion-rule", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--extractor", required=True, type=Path)
    parser.add_argument("--selector", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--deterministic-module", required=True, type=Path)
    parser.add_argument("--development-input", required=True, type=Path)
    parser.add_argument("--evaluation-input", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--development-coarse-v0-4", required=True, type=Path)
    parser.add_argument("--development-fine-v0-2", required=True, type=Path)
    parser.add_argument("--evaluation-v0-4-hybrid", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(value)
    return records


def unique_ids(records: Iterable[dict[str, Any]], description: str) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for record in records:
        article_id = record.get("article_id")
        if not isinstance(article_id, str) or not article_id:
            raise ValueError(f"{description} has a missing article_id")
        if article_id in seen:
            raise ValueError(f"{description} has duplicate article_id {article_id!r}")
        seen.add(article_id)
        result.append(article_id)
    return result


def source_entry(path: Path, *, jsonl_ids: list[str] | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if jsonl_ids is not None:
        entry["record_count"] = len(jsonl_ids)
        entry["ordered_article_ids_sha256"] = sha256_text("\n".join(jsonl_ids))
        entry["article_id_set_sha256"] = sha256_text("\n".join(sorted(jsonl_ids)))
    return entry


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(
            f"{args.output} already exists; a pre-score lock is never overwritten"
        )

    tracked_sources = {
        "protocol": args.protocol,
        "promotion_rule": args.promotion_rule,
        "protocol_config": args.config,
        "extractor": args.extractor,
        "selector": args.selector,
        "schema": args.schema,
        "deterministic_module": args.deterministic_module,
    }
    data_sources = {
        "reference": args.reference,
        "development_coarse_v0_4": args.development_coarse_v0_4,
        "development_fine_v0_2": args.development_fine_v0_2,
        "evaluation_v0_4_hybrid": args.evaluation_v0_4_hybrid,
    }
    for name, path in {**tracked_sources, **data_sources}.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")

    development_records = read_jsonl(args.development_input)
    evaluation_records = read_jsonl(args.evaluation_input)
    reference_records = read_jsonl(args.reference)
    development_ids = unique_ids(development_records, "development input")
    evaluation_ids = unique_ids(evaluation_records, "evaluation input")
    reference_ids = unique_ids(reference_records, "reference")
    overlap = set(development_ids) & set(evaluation_ids)
    if overlap:
        raise ValueError(
            "Development and evaluation assignments overlap: "
            + ", ".join(sorted(overlap)[:10])
        )
    if set(development_ids) | set(evaluation_ids) != set(reference_ids):
        raise ValueError(
            "Development and evaluation assignments must partition the reference IDs"
        )

    protocol_config = json.loads(args.config.read_text(encoding="utf-8"))
    if protocol_config.get("protocol_version") != "0.5.0":
        raise ValueError("Expected final candidate protocol_version 0.5.0")
    promotion_rule = json.loads(args.promotion_rule.read_text(encoding="utf-8"))
    if promotion_rule.get("rule_version") != "flan-t5-final-promotion-v1":
        raise ValueError("Unexpected promotion rule version")

    manifest = {
        "status": "locked_before_v0_5_model_scoring",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "runtime_target": "cuda_float16",
            "local_files_only": True,
        },
        "experiment": {
            "candidate_protocol_version": protocol_config["protocol_version"],
            "prompt_version": protocol_config["prompt_version"],
            "promotion_rule_version": promotion_rule["rule_version"],
            "calibration_policy": "development_only_then_hash_lock",
            "evaluation_policy": "one_materialization_after_calibration_lock",
            "interpretation": (
                "engineering comparison because the existing evaluation set "
                "informed diagnosis; not a pristine scientific holdout"
            ),
        },
        "tracked_sources": {
            name: source_entry(path) for name, path in tracked_sources.items()
        },
        "data_sources": {
            "development_input": source_entry(
                args.development_input, jsonl_ids=development_ids
            ),
            "evaluation_input": source_entry(
                args.evaluation_input, jsonl_ids=evaluation_ids
            ),
            "reference": source_entry(args.reference, jsonl_ids=reference_ids),
            **{
                name: source_entry(path)
                for name, path in data_sources.items()
                if name != "reference"
            },
        },
        "split_invariants": {
            "development_count": len(development_ids),
            "evaluation_count": len(evaluation_ids),
            "reference_count": len(reference_ids),
            "development_evaluation_overlap_count": 0,
            "assignments_partition_reference": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote immutable pre-score lock: {args.output}")
    print(f"Lock SHA-256: {sha256_file(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
