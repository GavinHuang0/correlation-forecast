from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CORE_FIELDS = (
    "shock_scope",
    "event_family",
    "information_status",
    "directional_alignment",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a compact, checked-in summary from locked FLAN v0.4 reports."
    )
    parser.add_argument("--hybrid-report", required=True, type=Path)
    parser.add_argument("--pure-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "record_count",
        "reference_relevant_count",
        "relevance_gate_metrics",
        "deterministic_explicit_surprise_metrics",
        "end_to_end_semantic_metrics_on_reference_relevant",
        "old_v0_2_predictions_mapped_to_same_coarse_task",
        "aggregate_comparison_vs_v0_2",
        "threshold_results",
        "prediction_sha256",
    }
    missing = required - set(report)
    if missing:
        raise ValueError(f"{path} is missing report keys: {sorted(missing)}")
    if set(report["end_to_end_semantic_metrics_on_reference_relevant"]) != set(
        CORE_FIELDS
    ):
        raise ValueError(f"{path} does not contain exactly the expected core fields")
    if report.get("split") != "evaluation":
        raise ValueError(f"{path} must be a fixed evaluation-split report")
    return report


def field_metrics(report: dict[str, Any]) -> dict[str, dict[str, float]]:
    return {
        field: {
            "accuracy": report[
                "end_to_end_semantic_metrics_on_reference_relevant"
            ][field]["accuracy"],
            "macro_f1": report[
                "end_to_end_semantic_metrics_on_reference_relevant"
            ][field]["macro_f1"],
        }
        for field in CORE_FIELDS
    }


def main() -> int:
    args = parse_args()
    hybrid = read_report(args.hybrid_report)
    pure = read_report(args.pure_report)
    identity_keys = (
        "record_count",
        "reference_relevant_count",
        "reference_sha256",
        "input_sha256",
        "old_v0_2_prediction_sha256",
    )
    for key in identity_keys:
        if hybrid.get(key) != pure.get(key):
            raise ValueError(f"Hybrid and pure reports disagree on {key}")

    summary = {
        "title": "FLAN-T5-Large coarse extraction evaluation",
        "reference_type": hybrid["reference_type"],
        "warning": hybrid["warning"],
        "split": "fixed reserved evaluation split",
        "record_count": hybrid["record_count"],
        "reference_relevant_count": hybrid["reference_relevant_count"],
        "model": {
            "model_id": hybrid["extraction_manifest"]["model_id"],
            "model_revision": hybrid["extraction_manifest"]["model_revision"],
            "device": hybrid["extraction_manifest"]["device"],
            "precision": hybrid["extraction_manifest"]["precision"],
        },
        "hybrid_protocol": {
            "prompt_version": hybrid["extraction_manifest"]["prompt_version"],
            "selection_basis": "fixed 72-article development split",
            "field_metrics": field_metrics(hybrid),
            "aggregate_comparison_vs_v0_2": hybrid[
                "aggregate_comparison_vs_v0_2"
            ],
            "relevance_gate": hybrid["relevance_gate_metrics"],
            "deterministic_explicit_surprise": hybrid[
                "deterministic_explicit_surprise_metrics"
            ],
            "threshold_results": hybrid["threshold_results"],
        },
        "pure_v0_4_protocol": {
            "prompt_version": pure["extraction_manifest"]["prompt_version"],
            "decoding": pure["extraction_manifest"]["decoding"],
            "prompt_profile": pure["extraction_manifest"]["prompt_profile"],
            "field_metrics": field_metrics(pure),
            "aggregate_comparison_vs_v0_2": pure[
                "aggregate_comparison_vs_v0_2"
            ],
        },
        "interpretation": (
            "The development-selected FLAN-only hybrid materially improves mean "
            "agreement over v0.2, especially for event family and information status, "
            "but no semantic field reaches its preregistered go/no-go threshold. "
            "FLAN-T5-Large remains a research baseline rather than a production extractor."
        ),
        "source_reports": {
            "hybrid_path": str(args.hybrid_report),
            "hybrid_sha256": sha256_file(args.hybrid_report),
            "pure_path": str(args.pure_report),
            "pure_sha256": sha256_file(args.pure_report),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote compact evaluation summary to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
