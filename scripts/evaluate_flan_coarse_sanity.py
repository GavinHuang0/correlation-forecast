from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import coarse_news_features as coarse
import extract_flan_t5 as base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FLAN v0.3 on the checked-in synthetic sanity set.")
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Exit nonzero unless accuracy is at least 90%, all gate checks pass, and no input was truncated.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = base.read_jsonl(args.inputs)
    predictions = base.read_jsonl(args.predictions)
    manifest_path = args.manifest or args.predictions.with_suffix(
        args.predictions.suffix + ".manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("output_sha256") != base.sha256_file(
        args.predictions
    ):
        raise ValueError("Sanity predictions failed manifest/hash validation")
    if manifest.get("prompt_version") != "flan-stock-sector-news-v0.4.0":
        raise ValueError("Sanity predictions were not produced by v0.4.0")
    prediction_by_id = {record["article_id"]: record for record in predictions}
    if len(prediction_by_id) != len(predictions):
        raise ValueError("Duplicate prediction article IDs")
    field_correct = Counter()
    field_total = Counter()
    mismatches: list[dict[str, Any]] = []
    gate_checks: list[dict[str, Any]] = []
    for record in inputs:
        article_id = record["article_id"]
        if article_id not in prediction_by_id:
            raise ValueError(f"Missing prediction for {article_id}")
        prediction = prediction_by_id[article_id]
        expected = record.get("expected")
        if expected is None:
            gate_checks.append(
                {
                    "article_id": article_id,
                    "expected_semantic_applicable": False,
                    "predicted_semantic_applicable": prediction["semantic_applicable"],
                    "correct": prediction["semantic_applicable"] is False,
                }
            )
            continue
        for field in coarse.COARSE_FIELDS:
            if field not in expected:
                continue
            truth = expected[field]
            guess = prediction["labels"].get(field)
            field_total[field] += 1
            field_correct[field] += int(truth == guess)
            if truth != guess:
                mismatches.append(
                    {
                        "article_id": article_id,
                        "field": field,
                        "expected": truth,
                        "predicted": guess,
                        "origin": prediction.get("label_origins", {}).get(field),
                    }
                )
        if "gate_route" in expected:
            gate_checks.append(
                {
                    "article_id": article_id,
                    "expected_gate_route": expected["gate_route"],
                    "predicted_gate_route": prediction["deterministic_features"]["gate_route"],
                    "correct": expected["gate_route"]
                    == prediction["deterministic_features"]["gate_route"],
                }
            )
    truncation_count = sum(
        int(not prediction.get("validity", {}).get("no_input_truncation", False))
        for prediction in predictions
    )
    total_accuracy = sum(field_correct.values()) / sum(field_total.values()) if field_total else None
    gate_correct_count = sum(int(check["correct"]) for check in gate_checks)
    per_field_pass = all(
        field_total[field] and field_correct[field] / field_total[field] >= 0.80
        for field in coarse.COARSE_FIELDS
    )
    passed = bool(
        total_accuracy is not None
        and total_accuracy >= 0.90
        and per_field_pass
        and gate_correct_count == len(gate_checks)
        and truncation_count == 0
    )
    report = {
        "record_count": len(inputs),
        "field_accuracy": {
            field: {
                "correct": field_correct[field],
                "count": field_total[field],
                "accuracy": field_correct[field] / field_total[field] if field_total[field] else None,
            }
            for field in coarse.COARSE_FIELDS
        },
        "all_scored_fields_accuracy": total_accuracy,
        "mismatches": mismatches,
        "gate_checks": gate_checks,
        "gate_correct_count": gate_correct_count,
        "gate_check_count": len(gate_checks),
        "input_truncation_count": truncation_count,
        "pass_criteria": {
            "all_scored_fields_accuracy_at_least": 0.90,
            "each_field_accuracy_at_least": 0.80,
            "all_gate_checks_correct": True,
            "input_truncation_count": 0,
        },
        "passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if args.require_pass and not passed else 0


if __name__ == "__main__":
    raise SystemExit(main())
