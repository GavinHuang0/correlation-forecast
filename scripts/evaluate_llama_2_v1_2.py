from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import coarse_news_features as coarse
import build_llama_2_v1_2_hybrid as hybrid_builder
import evaluate_llama_2_v1_1 as dev_eval
import extract_flan_t5 as base
import extract_llama_2_coarse as v1_0


DEFAULT_REFERENCE = Path(
    "annotations/chatgpt_5_6_sol_reference_coarse_v0_2.jsonl"
)
DEFAULT_SCHEMA = Path("config/news_feature_schema_coarse.json")
DEFAULT_FLAN_SUMMARY = Path(
    "experiments/flan_t5/v0_4/evaluation_summary.json"
)
SPLIT_DEFAULTS = {
    "development": {
        "inputs": Path(
            "outputs/llama_2/shared/benchmark_300/development_inputs.jsonl"
        ),
        "v1_0": Path("outputs/llama_2/v1_0/development_predictions.jsonl"),
        "count": 72,
        "relevant": 56,
    },
    "posthoc_evaluation": {
        "inputs": Path(
            "outputs/llama_2/shared/benchmark_300/evaluation_inputs.jsonl"
        ),
        "v1_0": Path("outputs/llama_2/v1_0/evaluation_predictions.jsonl"),
        "count": 228,
        "relevant": 174,
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the development-selected Llama 2 v1.2 fieldwise hybrid."
        )
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=tuple(SPLIT_DEFAULTS),
    )
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--v1-0-predictions", type=Path)
    parser.add_argument("--reference", default=DEFAULT_REFERENCE, type=Path)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA, type=Path)
    parser.add_argument("--flan-summary", default=DEFAULT_FLAN_SUMMARY, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def aggregate_delta(
    candidate: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, Any]:
    fieldwise: dict[str, Any] = {}
    for field in coarse.COARSE_FIELDS:
        candidate_metrics = candidate["field_metrics"][field]
        baseline_metrics = baseline["field_metrics"][field]
        fieldwise[field] = {
            "candidate_accuracy": candidate_metrics["accuracy"],
            "baseline_accuracy": baseline_metrics["accuracy"],
            "accuracy_delta": (
                candidate_metrics["accuracy"] - baseline_metrics["accuracy"]
            ),
            "candidate_macro_f1": candidate_metrics["macro_f1"],
            "baseline_macro_f1": baseline_metrics["macro_f1"],
            "macro_f1_delta": (
                candidate_metrics["macro_f1"] - baseline_metrics["macro_f1"]
            ),
        }
    return {
        "fieldwise": fieldwise,
        "candidate_mean_field_accuracy": candidate["mean_field_accuracy"],
        "baseline_mean_field_accuracy": baseline["mean_field_accuracy"],
        "mean_field_accuracy_delta": (
            candidate["mean_field_accuracy"] - baseline["mean_field_accuracy"]
        ),
        "candidate_mean_macro_f1": candidate["mean_macro_f1"],
        "baseline_mean_macro_f1": baseline["mean_macro_f1"],
        "mean_macro_f1_delta": (
            candidate["mean_macro_f1"] - baseline["mean_macro_f1"]
        ),
    }


def flan_metrics(summary: Mapping[str, Any]) -> dict[str, Any]:
    fields = summary["hybrid_protocol"]["field_metrics"]
    return {
        "field_metrics": fields,
        "mean_field_accuracy": sum(
            fields[field]["accuracy"] for field in coarse.COARSE_FIELDS
        )
        / len(coarse.COARSE_FIELDS),
        "mean_macro_f1": sum(
            fields[field]["macro_f1"] for field in coarse.COARSE_FIELDS
        )
        / len(coarse.COARSE_FIELDS),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite")
    split = SPLIT_DEFAULTS[args.split]
    inputs_path = args.inputs or split["inputs"]
    v1_path = args.v1_0_predictions or split["v1_0"]
    manifest_path = args.predictions.with_suffix(
        args.predictions.suffix + ".manifest.json"
    )
    required = [
        args.predictions,
        manifest_path,
        inputs_path,
        v1_path,
        args.reference,
        args.schema,
    ]
    if args.split == "posthoc_evaluation":
        required.append(args.flan_summary)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    inputs = base.read_jsonl(inputs_path)
    candidate_rows = base.read_jsonl(args.predictions)
    baseline_rows = base.read_jsonl(v1_path)
    reference = base.read_jsonl(args.reference)
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    v1_0.validate_schema(schema)
    if len(inputs) != split["count"]:
        raise ValueError(
            f"Expected {split['count']} {args.split} records, found {len(inputs)}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("Hybrid manifest is not complete")
    if manifest.get("hybrid_version") != hybrid_builder.HYBRID_VERSION:
        raise ValueError("Hybrid manifest version is not the frozen v1.2 contract")
    if manifest.get("model_id") != v1_0.MODEL_DEFAULT:
        raise ValueError("Hybrid manifest model ID mismatch")
    if manifest.get("model_revision") != v1_0.REVISION_DEFAULT:
        raise ValueError("Hybrid manifest model revision mismatch")
    if manifest.get("field_source") != hybrid_builder.FIELD_SOURCE:
        raise ValueError("Hybrid manifest field mapping mismatch")
    if manifest.get("output_sha256") != base.sha256_file(args.predictions):
        raise ValueError("Hybrid output hash does not match its manifest")
    if manifest.get("record_count") != split["count"]:
        raise ValueError("Hybrid manifest record count does not match split")
    selected_ids_hash = base.sha256_text(
        "\n".join(record["article_id"] for record in inputs)
    )
    if manifest.get("selected_article_ids_sha256") != selected_ids_hash:
        raise ValueError("Hybrid manifest selected IDs do not match inputs")
    if manifest.get("input_sha256") != base.sha256_file(inputs_path):
        raise ValueError("Hybrid manifest input hash does not match split")
    if manifest.get("schema_sha256") != base.sha256_file(args.schema):
        raise ValueError("Hybrid manifest schema hash mismatch")
    if manifest.get("source_v1_0_sha256") != base.sha256_file(v1_path):
        raise ValueError(
            "The comparison v1.0 predictions are not the hybrid's v1.0 source"
        )
    for path_key, hash_key in (
        ("source_v1_0_path", "source_v1_0_sha256"),
        ("source_v1_1_path", "source_v1_1_sha256"),
        ("source_v1_0_manifest_path", "source_v1_0_manifest_sha256"),
        ("source_v1_1_manifest_path", "source_v1_1_manifest_sha256"),
    ):
        source_path = Path(manifest[path_key])
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        if manifest.get(hash_key) != base.sha256_file(source_path):
            raise ValueError(f"Hybrid source integrity mismatch for {path_key}")

    candidate_metrics = dev_eval.evaluate_predictions(
        inputs=inputs,
        reference=reference,
        predictions=candidate_rows,
        schema=schema,
        expected_reference_relevant_count=split["relevant"],
    )
    v1_metrics = dev_eval.evaluate_predictions(
        inputs=inputs,
        reference=reference,
        predictions=baseline_rows,
        schema=schema,
        expected_reference_relevant_count=split["relevant"],
    )
    report: dict[str, Any] = {
        "title": "Llama 2 v1.2 fieldwise hybrid evaluation",
        "split": args.split,
        "confirmatory": False,
        "warning": (
            "The field assignment was selected on the development labels. The "
            "228-document evaluation was previously inspected for v1.0 and is "
            "reported only as a post-hoc engineering comparison."
        ),
        "reference_type": (
            "deterministically coarsened GPT-5.6 Sol silver annotations"
        ),
        "candidate_metrics": candidate_metrics,
        "comparison_vs_llama_v1_0": aggregate_delta(
            candidate_metrics, v1_metrics
        ),
        "integrity": {
            "inputs_sha256": base.sha256_file(inputs_path),
            "reference_sha256": base.sha256_file(args.reference),
            "schema_sha256": base.sha256_file(args.schema),
            "v1_0_predictions_sha256": base.sha256_file(v1_path),
            "candidate_predictions_sha256": base.sha256_file(args.predictions),
            "candidate_manifest_sha256": base.sha256_file(manifest_path),
        },
    }
    if args.split == "posthoc_evaluation":
        flan_summary = json.loads(
            args.flan_summary.read_text(encoding="utf-8")
        )
        if flan_summary.get("record_count") != 228:
            raise ValueError("FLAN summary record count is not 228")
        if flan_summary.get("reference_relevant_count") != 174:
            raise ValueError("FLAN summary relevant count is not 174")
        if (
            flan_summary.get("hybrid_protocol", {}).get("prompt_version")
            != "flan-stock-sector-news-hybrid-v0.4.0"
        ):
            raise ValueError("FLAN summary is not the frozen v0.4 hybrid")
        expected_flan_model = {
            "model_id": "google/flan-t5-large",
            "model_revision": "0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a",
            "device": "cuda",
            "precision": "float16",
        }
        if flan_summary.get("model") != expected_flan_model:
            raise ValueError("FLAN summary model contract mismatch")
        source_reports = flan_summary.get("source_reports", {})
        for path_key, hash_key in (
            ("hybrid_path", "hybrid_sha256"),
            ("pure_path", "pure_sha256"),
        ):
            source_path = Path(source_reports.get(path_key, ""))
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            if source_reports.get(hash_key) != base.sha256_file(source_path):
                raise ValueError(
                    f"FLAN source report integrity mismatch for {path_key}"
                )
        flan = flan_metrics(flan_summary)
        report["comparison_vs_flan_t5_v0_4"] = aggregate_delta(
            candidate_metrics, flan
        )
        report["integrity"]["flan_summary_sha256"] = base.sha256_file(
            args.flan_summary
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "split": args.split,
                "mean_accuracy": candidate_metrics["mean_field_accuracy"],
                "mean_macro_f1": candidate_metrics["mean_macro_f1"],
                "accuracy_delta_vs_v1_0": report[
                    "comparison_vs_llama_v1_0"
                ]["mean_field_accuracy_delta"],
                "macro_f1_delta_vs_v1_0": report[
                    "comparison_vs_llama_v1_0"
                ]["mean_macro_f1_delta"],
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
