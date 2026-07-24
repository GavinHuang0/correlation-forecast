from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import coarse_news_features as coarse
import evaluate_flan_agreement as metrics_lib
import experiment_llama_2_v1_1 as experiment
import extract_flan_t5 as base
import extract_llama_2_coarse as v1


EXPECTED_DEVELOPMENT_COUNT = 72
EXPECTED_REFERENCE_RELEVANT_COUNT = 56
DEFAULT_INPUTS = Path(
    "outputs/llama_2/shared/benchmark_300/development_inputs.jsonl"
)
DEFAULT_REFERENCE = Path(
    "annotations/chatgpt_5_6_sol_reference_coarse_v0_2.jsonl"
)
DEFAULT_SCHEMA = Path("config/news_feature_schema_coarse.json")
DEFAULT_V1_PREDICTIONS = Path(
    "outputs/llama_2/v1_0/development_predictions.jsonl"
)
FALLBACK_BY_FIELD = {
    "shock_scope": "unclear",
    "event_family": "other_or_unclear",
    "information_status": "unclear",
    "directional_alignment": "unclear",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the Llama 2 v1.1 decoder on the fixed development split "
            "and apply the promotion rule frozen before the run."
        )
    )
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--inputs", default=DEFAULT_INPUTS, type=Path)
    parser.add_argument("--reference", default=DEFAULT_REFERENCE, type=Path)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA, type=Path)
    parser.add_argument(
        "--prediction-manifest",
        type=Path,
        help="Defaults to <predictions>.manifest.json.",
    )
    parser.add_argument(
        "--v1-predictions", default=DEFAULT_V1_PREDICTIONS, type=Path
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _by_id(
    records: Sequence[Mapping[str, Any]], *, source: str
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for record in records:
        article_id = record.get("article_id")
        if not isinstance(article_id, str) or not article_id:
            raise ValueError(f"{source} contains a record without article_id")
        if article_id in result:
            raise ValueError(f"{source} contains duplicate article_id {article_id}")
        result[article_id] = record
    return result


def evaluate_predictions(
    *,
    inputs: Sequence[Mapping[str, Any]],
    reference: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    schema: Mapping[str, Any],
    expected_reference_relevant_count: int = EXPECTED_REFERENCE_RELEVANT_COUNT,
) -> dict[str, Any]:
    reference_by_id = _by_id(reference, source="reference")
    prediction_by_id = _by_id(predictions, source="predictions")
    selected_ids = [str(record["article_id"]) for record in inputs]
    if set(prediction_by_id) != set(selected_ids):
        raise ValueError("Prediction IDs must exactly match development input IDs")
    if [record["article_id"] for record in predictions] != selected_ids:
        raise ValueError("Prediction order must match development input order")
    if not set(selected_ids) <= set(reference_by_id):
        raise ValueError("Reference is missing development input IDs")

    relevant_ids = [
        article_id
        for article_id in selected_ids
        if bool(reference_by_id[article_id]["semantic_applicable"])
    ]
    if len(relevant_ids) != expected_reference_relevant_count:
        raise ValueError(
            f"Expected {expected_reference_relevant_count} relevant records, "
            f"found {len(relevant_ids)}"
        )

    field_metrics: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    for field in coarse.COARSE_FIELDS:
        allowed = list(schema["closed_label_fields"][field])
        truths = [
            reference_by_id[article_id]["labels"][field]
            for article_id in relevant_ids
        ]
        guesses = [
            prediction_by_id[article_id]["labels"].get(field)
            or FALLBACK_BY_FIELD[field]
            for article_id in relevant_ids
        ]
        field_metrics[field] = metrics_lib.closed_label_metrics(
            truths, guesses, allowed
        )
        distribution = Counter(guesses)
        model_guesses = [
            prediction_by_id[article_id]["labels"].get(field)
            or FALLBACK_BY_FIELD[field]
            for article_id in relevant_ids
            if prediction_by_id[article_id]
            .get("passes", {})
            .get(field, {})
            .get("origin")
            == "model"
        ]
        model_distribution = Counter(model_guesses)
        model_passes = [
            prediction_by_id[article_id].get("passes", {}).get(field, {})
            for article_id in relevant_ids
            if prediction_by_id[article_id]
            .get("passes", {})
            .get(field, {})
            .get("origin")
            == "model"
        ]
        rotation_agreements = [
            float(pass_record["rotation_prediction_agreement_rate"])
            for pass_record in model_passes
            if "rotation_prediction_agreement_rate" in pass_record
        ]
        diagnostics[field] = {
            "prediction_distribution": dict(sorted(distribution.items())),
            "unique_prediction_count": len(distribution),
            "model_origin_prediction_distribution": dict(
                sorted(model_distribution.items())
            ),
            "model_origin_unique_prediction_count": len(model_distribution),
            "single_class_collapse": (
                len(model_guesses) >= 2 and len(model_distribution) == 1
            ),
            "dominant_prediction_share": (
                max(distribution.values()) / len(guesses) if guesses else 0.0
            ),
            "model_origin_dominant_prediction_share": (
                max(model_distribution.values()) / len(model_guesses)
                if model_guesses
                else None
            ),
            "model_routed_record_count": len(model_passes),
            "mean_rotation_prediction_agreement_rate": (
                sum(rotation_agreements) / len(rotation_agreements)
                if rotation_agreements
                else None
            ),
        }

    mean_accuracy = sum(
        field_metrics[field]["accuracy"] for field in coarse.COARSE_FIELDS
    ) / len(coarse.COARSE_FIELDS)
    mean_macro_f1 = sum(
        field_metrics[field]["macro_f1"] for field in coarse.COARSE_FIELDS
    ) / len(coarse.COARSE_FIELDS)
    return {
        "record_count": len(selected_ids),
        "reference_relevant_count": len(relevant_ids),
        "field_metrics": field_metrics,
        "decoder_diagnostics": diagnostics,
        "mean_field_accuracy": mean_accuracy,
        "mean_macro_f1": mean_macro_f1,
    }


def promotion_decision(
    v1_0: Mapping[str, Any], v1_1: Mapping[str, Any]
) -> dict[str, Any]:
    macro_delta = float(v1_1["mean_macro_f1"]) - float(v1_0["mean_macro_f1"])
    accuracy_delta = float(v1_1["mean_field_accuracy"]) - float(
        v1_0["mean_field_accuracy"]
    )
    checks = {
        "mean_macro_f1_improves_by_at_least_0_05": macro_delta >= 0.05,
        "mean_accuracy_declines_by_no_more_than_0_02": accuracy_delta >= -0.02,
        "shock_scope_macro_f1_improves": (
            v1_1["field_metrics"]["shock_scope"]["macro_f1"]
            > v1_0["field_metrics"]["shock_scope"]["macro_f1"]
        ),
        "event_family_macro_f1_improves": (
            v1_1["field_metrics"]["event_family"]["macro_f1"]
            > v1_0["field_metrics"]["event_family"]["macro_f1"]
        ),
        "no_v1_1_single_class_collapse": not any(
            diagnostic["single_class_collapse"]
            for diagnostic in v1_1["decoder_diagnostics"].values()
        ),
    }
    passes = all(checks.values())
    return {
        "checks": checks,
        "all_checks_pass": passes,
        "mean_macro_f1_delta": macro_delta,
        "mean_field_accuracy_delta": accuracy_delta,
        "decision": (
            "promote_decoder_for_fresh_holdout"
            if passes
            else "reject_decoder_and_keep_v1_0_as_documented_baseline"
        ),
        "important_limit": (
            "Passing selects a decoder for a fresh holdout only. The previously "
            "inspected 228-document evaluation split cannot provide a new "
            "confirmatory claim."
        ),
    }


def validate_manifest(
    manifest: Mapping[str, Any],
    *,
    predictions_path: Path,
    inputs_path: Path,
    schema_path: Path,
) -> None:
    expected = {
        "status": "complete",
        "development_only": True,
        "prompt_version": experiment.PROMPT_VERSION,
        "model_id": v1.MODEL_DEFAULT,
        "model_revision": v1.REVISION_DEFAULT,
        "decoding": experiment.DECODING_METHOD,
        "extractor_source_sha256": base.sha256_file(
            Path(experiment.__file__).resolve()
        ),
        "frozen_v1_source_sha256": base.sha256_file(
            Path(v1.__file__).resolve()
        ),
        "deterministic_rule_version": coarse.DETERMINISTIC_RULE_VERSION,
        "deterministic_rules_payload_sha256": (
            coarse.DETERMINISTIC_RULES_SHA256
        ),
        "device": "cuda",
        "precision": "float16",
        "input_sha256": base.sha256_file(inputs_path),
        "schema_sha256": base.sha256_file(schema_path),
        "output_sha256": base.sha256_file(predictions_path),
        "total_prediction_count": EXPECTED_DEVELOPMENT_COUNT,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"Prediction manifest mismatch for {key}: "
                f"{manifest.get(key)!r} != {value!r}"
            )
    if manifest.get("option_orders") != "all_cyclic_rotations":
        raise ValueError("Prediction manifest did not use all cyclic rotations")
    quantization = manifest.get("quantization")
    if not isinstance(quantization, dict):
        raise ValueError("Prediction manifest has no quantization contract")
    expected_quantization = {
        "method": "nf4",
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "compute_dtype": "float16",
        "hf_device_map": {"": 0},
    }
    if quantization != expected_quantization:
        raise ValueError(
            f"Prediction manifest quantization mismatch: {quantization!r}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite")
    manifest_path = args.prediction_manifest or args.predictions.with_suffix(
        args.predictions.suffix + ".manifest.json"
    )
    v1_manifest_path = args.v1_predictions.with_suffix(
        args.v1_predictions.suffix + ".manifest.json"
    )
    for path in (
        args.predictions,
        manifest_path,
        args.inputs,
        args.reference,
        args.schema,
        args.v1_predictions,
        v1_manifest_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    inputs = base.read_jsonl(args.inputs)
    predictions = base.read_jsonl(args.predictions)
    v1_predictions = base.read_jsonl(args.v1_predictions)
    reference = base.read_jsonl(args.reference)
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    v1.validate_schema(schema)
    if len(inputs) != EXPECTED_DEVELOPMENT_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_DEVELOPMENT_COUNT} development inputs, "
            f"found {len(inputs)}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(
        manifest,
        predictions_path=args.predictions,
        inputs_path=args.inputs,
        schema_path=args.schema,
    )
    v1_manifest = json.loads(v1_manifest_path.read_text(encoding="utf-8"))
    expected_v1_manifest = {
        "status": "complete",
        "prompt_version": v1.PROMPT_VERSION,
        "model_id": v1.MODEL_DEFAULT,
        "model_revision": v1.REVISION_DEFAULT,
        "input_sha256": base.sha256_file(args.inputs),
        "schema_sha256": base.sha256_file(args.schema),
        "output_sha256": base.sha256_file(args.v1_predictions),
        "total_prediction_count": EXPECTED_DEVELOPMENT_COUNT,
        "device": "cuda",
        "precision": "float16",
    }
    for key, value in expected_v1_manifest.items():
        if v1_manifest.get(key) != value:
            raise ValueError(
                f"v1.0 baseline manifest mismatch for {key}: "
                f"{v1_manifest.get(key)!r} != {value!r}"
            )

    v1_0_metrics = evaluate_predictions(
        inputs=inputs,
        reference=reference,
        predictions=v1_predictions,
        schema=schema,
    )
    v1_1_metrics = evaluate_predictions(
        inputs=inputs,
        reference=reference,
        predictions=predictions,
        schema=schema,
    )
    decision = promotion_decision(v1_0_metrics, v1_1_metrics)
    report = {
        "title": "Llama 2 v1.1 answer-boundary development experiment",
        "reference_type": (
            "deterministically coarsened GPT-5.6 Sol silver annotations"
        ),
        "warning": (
            "This is development-set agreement, not human-ground-truth accuracy "
            "or confirmatory out-of-sample performance."
        ),
        "experimental_change": {
            "from": v1.DECODING_METHOD,
            "to": experiment.DECODING_METHOD,
            "assistant_reply_template": experiment.ASSISTANT_REPLY_TEMPLATE,
            "unchanged": (
                "model revision, NF4 weights, schema, inputs, deterministic gate, "
                "and hierarchical routing"
            ),
        },
        "v1_0_development_baseline": v1_0_metrics,
        "v1_1_development_result": v1_1_metrics,
        "promotion_rule_result": decision,
        "integrity": {
            "inputs_sha256": base.sha256_file(args.inputs),
            "reference_sha256": base.sha256_file(args.reference),
            "schema_sha256": base.sha256_file(args.schema),
            "v1_0_predictions_sha256": base.sha256_file(args.v1_predictions),
            "v1_0_manifest_sha256": base.sha256_file(v1_manifest_path),
            "v1_1_predictions_sha256": base.sha256_file(args.predictions),
            "v1_1_manifest_sha256": base.sha256_file(manifest_path),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "v1_0_mean_accuracy": v1_0_metrics["mean_field_accuracy"],
                "v1_1_mean_accuracy": v1_1_metrics["mean_field_accuracy"],
                "v1_0_mean_macro_f1": v1_0_metrics["mean_macro_f1"],
                "v1_1_mean_macro_f1": v1_1_metrics["mean_macro_f1"],
                "decision": decision["decision"],
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
