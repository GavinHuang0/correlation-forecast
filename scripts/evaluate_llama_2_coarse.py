"""Evaluate the frozen Llama 2 coarse extractor against the fixed silver set.

The evaluator loads no model.  It validates the prepared benchmark, prediction
manifest, deterministic-gate replay, and row contract before calculating the
same end-to-end four-field metrics used for FLAN-T5 v0.4.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import coarse_news_features as coarse
import evaluate_flan_agreement as legacy_eval
import evaluate_flan_coarse as flan_eval
import extract_flan_t5 as base
import extract_llama_2_coarse as llama_extractor


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
MODEL_REVISION = "f5db02db724555f92da89c216ac04704f23d4590"
PROMPT_VERSION = "llama-2-stock-sector-news-v1.0.0"
EXPECTED_EVALUATION_INPUT_SHA256 = (
    "92cf5da0dcb26d186f416885528f6453fac81b26fa5b277025063b4ae070ac83"
)
EXPECTED_FINE_REFERENCE_SHA256 = (
    "94ae2481611fc2a88ce3ff2a16c6c2945c4083be2ef787810c1d1bb1f50967b7"
)
EXPECTED_COARSE_REFERENCE_SHA256 = (
    "d765d236c423a7f3ccebcc1c9b3336462458b40862c69f554e01a6fa4bd97ce1"
)
EXPECTED_SCHEMA_SHA256 = (
    "201e19a0f2b9723e690c3a62eb145213b7e6d8c106a09e0da7374b71830d05fa"
)
EXPECTED_EVALUATION_COUNT = 228
EXPECTED_REFERENCE_RELEVANT_COUNT = 174

DEFAULT_INPUTS = (
    REPOSITORY_ROOT
    / "outputs"
    / "llama_2"
    / "shared"
    / "benchmark_300"
    / "evaluation_inputs.jsonl"
)
DEFAULT_BENCHMARK_MANIFEST = DEFAULT_INPUTS.parent / "manifest.json"
DEFAULT_COARSE_REFERENCE = (
    REPOSITORY_ROOT / "annotations" / "chatgpt_5_6_sol_reference_coarse_v0_2.jsonl"
)
DEFAULT_FINE_REFERENCE = REPOSITORY_ROOT / "annotations" / "chatgpt_5_6_sol_reference.jsonl"
DEFAULT_SCHEMA = REPOSITORY_ROOT / "config" / "news_feature_schema_coarse.json"
DEFAULT_FLAN_BASELINE = (
    REPOSITORY_ROOT
    / "experiments"
    / "flan_t5"
    / "v0_4"
    / "evaluation_summary.json"
)

FALLBACK_BY_FIELD = {
    "shock_scope": "unclear",
    "event_family": "other_or_unclear",
    "information_status": "unclear",
    "directional_alignment": "unclear",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen Llama 2 predictions on the fixed 228-article "
            "silver-label evaluation split and compare with FLAN-T5 v0.4."
        )
    )
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument(
        "--benchmark-manifest", type=Path, default=DEFAULT_BENCHMARK_MANIFEST
    )
    parser.add_argument("--coarse-reference", type=Path, default=DEFAULT_COARSE_REFERENCE)
    parser.add_argument("--fine-reference", type=Path, default=DEFAULT_FINE_REFERENCE)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--flan-baseline", type=Path, default=DEFAULT_FLAN_BASELINE)
    parser.add_argument("--prediction-manifest", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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


def _selected_ids_sha256(records: list[dict[str, Any]]) -> str:
    return base.sha256_text("\n".join(record["article_id"] for record in records))


def _mean_field_metric(metrics: Mapping[str, Mapping[str, Any]], name: str) -> float:
    return sum(float(metrics[field][name]) for field in coarse.COARSE_FIELDS) / len(
        coarse.COARSE_FIELDS
    )


def validate_benchmark_manifest(
    manifest: Mapping[str, Any],
    *,
    inputs_path: Path,
    fine_reference_path: Path | None = None,
    coarse_reference_path: Path | None = None,
    schema_path: Path | None = None,
    expected_input_sha256: str = EXPECTED_EVALUATION_INPUT_SHA256,
    expected_count: int = EXPECTED_EVALUATION_COUNT,
) -> None:
    if manifest.get("manifest_version") != "llama-2-benchmark-v1":
        raise ValueError("Unexpected Llama benchmark manifest version")
    if manifest.get("mode") != "prepared":
        raise ValueError("Llama benchmark manifest is not in prepared mode")
    if manifest.get("counts", {}).get("evaluation") != expected_count:
        raise ValueError("Llama benchmark evaluation count differs from the fixed contract")
    if manifest.get("knowledge_cutoff") != "2023-07-31":
        raise ValueError("Llama benchmark knowledge cutoff differs from the frozen contract")
    if manifest.get("all_articles_after_cutoff") is not True:
        raise ValueError("Llama benchmark did not pass the post-cutoff check")
    if manifest.get("labels_present_in_extractor_inputs") is not False:
        raise ValueError("Llama extractor inputs must not contain silver labels")
    prepared = manifest.get("prepared_outputs", {}).get("evaluation_inputs", {})
    actual_hash = base.sha256_file(inputs_path)
    if actual_hash != expected_input_sha256:
        raise ValueError("Evaluation input hash differs from the fixed FLAN split")
    if prepared.get("sha256") != actual_hash:
        raise ValueError("Evaluation input hash differs from the benchmark manifest")
    if prepared.get("byte_identical_to_source") is not True:
        raise ValueError("Benchmark manifest does not attest to a byte-identical copy")
    source_checks = (
        ("fine_reference", fine_reference_path, EXPECTED_FINE_REFERENCE_SHA256),
        ("coarse_reference", coarse_reference_path, EXPECTED_COARSE_REFERENCE_SHA256),
        ("schema", schema_path, EXPECTED_SCHEMA_SHA256),
    )
    for name, path, expected_hash in source_checks:
        if path is None:
            continue
        actual = base.sha256_file(path)
        recorded = manifest.get("source_files", {}).get(name, {}).get("sha256")
        if actual != expected_hash or recorded != actual:
            raise ValueError(
                f"{name} hash differs from the frozen benchmark contract or manifest"
            )


def validate_prediction_manifest(
    manifest: Mapping[str, Any],
    *,
    predictions_path: Path,
    inputs_path: Path,
    schema_path: Path,
    predictions: list[dict[str, Any]],
    expected_model_id: str = MODEL_ID,
    expected_model_revision: str = MODEL_REVISION,
    expected_prompt_version: str = PROMPT_VERSION,
) -> None:
    if manifest.get("manifest_version") != llama_extractor.MANIFEST_VERSION:
        raise ValueError("Unexpected Llama 2 extraction manifest version")
    if manifest.get("status") != "complete":
        raise ValueError("Prediction manifest is not complete")
    if manifest.get("model_id") != expected_model_id:
        raise ValueError("Predictions were not produced by the frozen Llama model ID")
    if manifest.get("model_revision") != expected_model_revision:
        raise ValueError("Predictions were not produced by the frozen Llama revision")
    if not re.fullmatch(r"[0-9a-f]{40}", str(manifest.get("model_revision", ""))):
        raise ValueError("Llama 2 model revision is not an immutable 40-character commit")
    if manifest.get("prompt_version") != expected_prompt_version:
        raise ValueError("Predictions were not produced by the frozen Llama prompt")
    if (
        manifest.get("conservative_model_data_cutoff")
        != llama_extractor.CONSERVATIVE_DATA_CUTOFF
    ):
        raise ValueError("Predictions used a different Llama 2 data cutoff")
    if manifest.get("decoding") != "order_averaged_letter_score":
        raise ValueError("Predictions used a different Llama decoding protocol")
    if manifest.get("prompt_profile") != "zero_shot":
        raise ValueError("Predictions used a different Llama prompt profile")
    if manifest.get("device") != "cuda" or manifest.get("precision") != "float16":
        raise ValueError("Predictions used a different frozen device or precision")
    if (
        manifest.get("batch_size") != 1
        or manifest.get("max_input_tokens")
        != llama_extractor.MODEL_CONTEXT_WINDOW
    ):
        raise ValueError("Predictions used a different frozen batch or input limit")
    if manifest.get("attention_implementation") != "eager":
        raise ValueError("Predictions used a different attention implementation")
    quantization = manifest.get("quantization")
    if not isinstance(quantization, Mapping) or any(
        (
            quantization.get("method") != "nf4",
            quantization.get("bnb_4bit_quant_type") != "nf4",
            quantization.get("bnb_4bit_use_double_quant") is not True,
            quantization.get("compute_dtype") != "float16",
        )
    ):
        raise ValueError("Predictions used a different frozen NF4 configuration")
    if (
        manifest.get("chat_template_source")
        != "frozen official Llama 2 single-turn [INST] format"
        or manifest.get("chat_template_sha256")
        != llama_extractor._canonical_json_sha256(
            llama_extractor.LLAMA_2_SINGLE_TURN_CHAT_TEMPLATE
        )
    ):
        raise ValueError("Predictions used a different frozen Llama 2 chat template")
    if manifest.get("extractor_source_sha256") != base.sha256_file(
        Path(llama_extractor.__file__).resolve()
    ):
        raise ValueError("Current Llama 2 extractor source differs from extraction")
    if manifest.get("output_sha256") != base.sha256_file(predictions_path):
        raise ValueError("Prediction hash differs from its manifest")
    if manifest.get("input_sha256") != base.sha256_file(inputs_path):
        raise ValueError("Prediction manifest input hash differs from evaluation inputs")
    if manifest.get("schema_sha256") != base.sha256_file(schema_path):
        raise ValueError("Prediction schema hash differs from the evaluation schema")
    if manifest.get("deterministic_module_sha256") != base.sha256_file(
        Path(coarse.__file__).resolve()
    ):
        raise ValueError("Current deterministic module differs from extraction")
    if manifest.get("hierarchy_module_sha256") != base.sha256_file(
        Path(llama_extractor.flan_coarse.__file__).resolve()
    ):
        raise ValueError("Current coarse hierarchy module differs from extraction")
    if manifest.get("total_prediction_count") != len(predictions):
        raise ValueError("Prediction count differs from its manifest")
    if manifest.get("selected_article_ids_sha256") != _selected_ids_sha256(predictions):
        raise ValueError("Prediction ID/order hash differs from its manifest")


def validate_prediction_rows(
    predictions: list[dict[str, Any]],
    *,
    inputs_by_id: Mapping[str, dict[str, Any]],
    schema: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    errors: list[str] = []
    for prediction in predictions:
        article_id = prediction["article_id"]
        input_record = inputs_by_id[article_id]
        if prediction.get("target_ticker") != input_record["target"]["ticker"]:
            errors.append(f"{article_id}: target_ticker mismatch")
        if prediction.get("extractor") not in (None, manifest["model_id"]):
            errors.append(f"{article_id}: extractor mismatch")
        if prediction.get("model_revision") != manifest["model_revision"]:
            errors.append(f"{article_id}: model_revision mismatch")
        if prediction.get("prompt_version") != manifest["prompt_version"]:
            errors.append(f"{article_id}: prompt_version mismatch")

        replay = coarse.deterministic_features(input_record)
        if prediction.get("deterministic_features") != replay:
            errors.append(f"{article_id}: deterministic feature replay mismatch")
        if not isinstance(prediction.get("semantic_applicable"), bool):
            errors.append(f"{article_id}: semantic_applicable must be boolean")
        applicable = prediction.get("semantic_applicable") is True
        if applicable is not bool(replay["semantic_applicable"]):
            errors.append(f"{article_id}: semantic_applicable mismatch")

        labels = prediction.get("labels")
        origins = prediction.get("label_origins")
        passes = prediction.get("passes")
        if not isinstance(labels, dict) or set(labels) != set(coarse.COARSE_FIELDS):
            errors.append(f"{article_id}: label fields mismatch")
            continue
        if not isinstance(origins, dict) or set(origins) != set(coarse.COARSE_FIELDS):
            errors.append(f"{article_id}: label-origin fields mismatch")
            continue
        if not isinstance(passes, dict) or set(passes) != set(coarse.COARSE_FIELDS):
            errors.append(f"{article_id}: pass fields mismatch")
            continue

        for field in coarse.COARSE_FIELDS:
            value = labels[field]
            if value is not None and value not in schema["closed_label_fields"][field]:
                errors.append(f"{article_id}: invalid {field}={value!r}")
            if not isinstance(origins[field], str) or not origins[field]:
                errors.append(f"{article_id}: invalid {field} label origin")
            pass_record = passes[field]
            if not isinstance(pass_record, dict):
                errors.append(f"{article_id}: invalid {field} pass")
                continue
            if pass_record.get("value") != value:
                errors.append(f"{article_id}: {field} pass/label mismatch")
            if pass_record.get("origin") != origins[field]:
                errors.append(f"{article_id}: {field} pass/origin mismatch")
            if pass_record.get("valid") is not True:
                errors.append(f"{article_id}: invalid {field} pass")
            if pass_record.get("input_truncated") is not False:
                errors.append(f"{article_id}: truncated {field} pass or flag missing")

        if applicable and any(labels[field] is None for field in coarse.COARSE_FIELDS):
            errors.append(f"{article_id}: applicable row has unresolved labels")
        if not applicable and any(labels[field] is not None for field in coarse.COARSE_FIELDS):
            errors.append(f"{article_id}: gated row has fabricated semantic labels")
        validity = prediction.get("validity")
        if not isinstance(validity, dict):
            errors.append(f"{article_id}: missing validity object")

    if errors:
        raise ValueError(
            "Llama prediction row contract validation failed: "
            + "; ".join(errors[:20])
        )


def _majority_baseline(truths: list[str], allowed: list[str]) -> dict[str, Any]:
    counts = Counter(truths)
    majority = max(allowed, key=lambda label: (counts[label], -allowed.index(label)))
    return {
        "majority_label": majority,
        **legacy_eval.closed_label_metrics(
            truths, [majority] * len(truths), allowed
        ),
    }


def build_evaluation_report(
    *,
    inputs: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    coarse_reference: list[dict[str, Any]],
    fine_reference: list[dict[str, Any]],
    schema: Mapping[str, Any],
    flan_baseline_summary: Mapping[str, Any],
    prediction_manifest: Mapping[str, Any],
    expected_reference_relevant_count: int | None = EXPECTED_REFERENCE_RELEVANT_COUNT,
) -> dict[str, Any]:
    input_by_id = _records_by_id(inputs, source_name="evaluation inputs")
    prediction_by_id = _records_by_id(predictions, source_name="Llama predictions")
    coarse_by_id = _records_by_id(coarse_reference, source_name="coarse reference")
    fine_by_id = _records_by_id(fine_reference, source_name="fine reference")
    input_ids = set(input_by_id)
    if set(prediction_by_id) != input_ids:
        missing = input_ids - set(prediction_by_id)
        unexpected = set(prediction_by_id) - input_ids
        raise ValueError(
            "Prediction IDs must exactly match evaluation inputs: "
            f"missing={sorted(missing)[:10]!r}, unexpected={sorted(unexpected)[:10]!r}"
        )
    if not input_ids <= set(coarse_by_id) or not input_ids <= set(fine_by_id):
        raise ValueError("References are missing evaluation input article IDs")

    selected_ids = [record["article_id"] for record in inputs]
    for article_id in selected_ids:
        ticker = input_by_id[article_id]["target"]["ticker"]
        if coarse_by_id[article_id].get("target_ticker") != ticker:
            raise ValueError(f"{article_id}: coarse-reference target mismatch")
        if fine_by_id[article_id].get("target_ticker") != ticker:
            raise ValueError(f"{article_id}: fine-reference target mismatch")
        coarse.validate_coarse_labels(coarse_by_id[article_id]["labels"], schema)

    reference_relevant_ids = [
        article_id
        for article_id in selected_ids
        if coarse_by_id[article_id]["semantic_applicable"]
    ]
    if (
        expected_reference_relevant_count is not None
        and len(reference_relevant_ids) != expected_reference_relevant_count
    ):
        raise ValueError(
            f"Expected {expected_reference_relevant_count} reference-relevant records, "
            f"found {len(reference_relevant_ids)}"
        )

    reference_gate = [
        bool(coarse_by_id[article_id]["semantic_applicable"]) for article_id in selected_ids
    ]
    predicted_gate = [
        bool(prediction_by_id[article_id]["semantic_applicable"])
        for article_id in selected_ids
    ]
    gate_metrics = flan_eval.binary_metrics(reference_gate, predicted_gate)
    surprise_truths = [
        fine_by_id[article_id]["labels"]["explicit_surprise"]
        for article_id in selected_ids
    ]
    surprise_guesses = [
        prediction_by_id[article_id]["deterministic_features"][
            "explicit_surprise_rule"
        ]
        for article_id in selected_ids
    ]
    surprise_metrics = legacy_eval.closed_label_metrics(
        surprise_truths,
        surprise_guesses,
        ["positive", "negative", "mixed", "none", "unknown"],
    )

    gate_true_positive_ids = [
        article_id
        for article_id in reference_relevant_ids
        if prediction_by_id[article_id]["semantic_applicable"]
    ]
    conditional_metrics: dict[str, Any] = {}
    end_to_end_metrics: dict[str, Any] = {}
    majority_baselines: dict[str, Any] = {}
    label_origin_counts: dict[str, Any] = {}
    for field in coarse.COARSE_FIELDS:
        allowed = schema["closed_label_fields"][field]
        conditional_truths = [
            coarse_by_id[article_id]["labels"][field]
            for article_id in gate_true_positive_ids
        ]
        conditional_guesses = [
            prediction_by_id[article_id]["labels"][field]
            for article_id in gate_true_positive_ids
        ]
        conditional_metrics[field] = legacy_eval.closed_label_metrics(
            conditional_truths, conditional_guesses, allowed
        )
        truths = [
            coarse_by_id[article_id]["labels"][field]
            for article_id in reference_relevant_ids
        ]
        guesses = [
            prediction_by_id[article_id]["labels"].get(field)
            or FALLBACK_BY_FIELD[field]
            for article_id in reference_relevant_ids
        ]
        end_to_end_metrics[field] = legacy_eval.closed_label_metrics(
            truths, guesses, allowed
        )
        majority_baselines[field] = _majority_baseline(truths, allowed)
        label_origin_counts[field] = dict(
            sorted(
                Counter(
                    prediction_by_id[article_id]["label_origins"][field]
                    for article_id in selected_ids
                ).items()
            )
        )

    baseline_protocol = flan_baseline_summary.get("hybrid_protocol", {})
    baseline_field_metrics = baseline_protocol.get("field_metrics")
    if not isinstance(baseline_field_metrics, dict) or set(
        coarse.COARSE_FIELDS
    ) - set(baseline_field_metrics):
        raise ValueError("FLAN v0.4 baseline summary is missing field metrics")
    if flan_baseline_summary.get("record_count") != len(selected_ids):
        raise ValueError("FLAN v0.4 baseline used a different evaluation record count")
    if flan_baseline_summary.get("reference_relevant_count") != len(
        reference_relevant_ids
    ):
        raise ValueError("FLAN v0.4 baseline used a different relevant-record count")

    fieldwise_comparison = {
        field: {
            "llama_2": {
                "accuracy": end_to_end_metrics[field]["accuracy"],
                "macro_f1": end_to_end_metrics[field]["macro_f1"],
            },
            "flan_t5_v0_4": {
                "accuracy": float(baseline_field_metrics[field]["accuracy"]),
                "macro_f1": float(baseline_field_metrics[field]["macro_f1"]),
            },
            "accuracy_delta": end_to_end_metrics[field]["accuracy"]
            - float(baseline_field_metrics[field]["accuracy"]),
            "macro_f1_delta": end_to_end_metrics[field]["macro_f1"]
            - float(baseline_field_metrics[field]["macro_f1"]),
        }
        for field in coarse.COARSE_FIELDS
    }
    llama_mean_accuracy = _mean_field_metric(end_to_end_metrics, "accuracy")
    llama_mean_macro_f1 = _mean_field_metric(end_to_end_metrics, "macro_f1")
    flan_mean_accuracy = _mean_field_metric(baseline_field_metrics, "accuracy")
    flan_mean_macro_f1 = _mean_field_metric(baseline_field_metrics, "macro_f1")

    return {
        "title": "Llama 2 7B Chat coarse extraction evaluation",
        "reference_type": "deterministically coarsened GPT-5.6 Sol silver annotations",
        "warning": (
            "Agreement with a GPT-5.6 silver reference is not objective human-label "
            "accuracy. The fixed evaluation set was not used to tune this Llama prompt."
        ),
        "split": "fixed reserved evaluation split",
        "record_count": len(selected_ids),
        "reference_relevant_count": len(reference_relevant_ids),
        "gate_true_positive_semantic_count": len(gate_true_positive_ids),
        "semantic_coverage_on_reference_relevant": flan_eval.safe_divide(
            len(gate_true_positive_ids), len(reference_relevant_ids)
        ),
        "model": {
            "model_id": prediction_manifest["model_id"],
            "model_revision": prediction_manifest["model_revision"],
            "prompt_version": prediction_manifest["prompt_version"],
            "decoding": prediction_manifest.get("decoding"),
            "quantization": prediction_manifest.get("quantization"),
            "device": prediction_manifest.get("device"),
            "precision": prediction_manifest.get("precision"),
        },
        "relevance_gate_metrics": gate_metrics,
        "deterministic_explicit_surprise_metrics": surprise_metrics,
        "semantic_metrics_conditional_on_reference_relevant_and_gate_pass": (
            conditional_metrics
        ),
        "end_to_end_semantic_metrics_on_reference_relevant": end_to_end_metrics,
        "majority_baselines_on_reference_relevant": majority_baselines,
        "label_origin_counts": label_origin_counts,
        "fieldwise_comparison_vs_flan_t5_v0_4": fieldwise_comparison,
        "aggregate_comparison_vs_flan_t5_v0_4": {
            "llama_2_mean_field_accuracy": llama_mean_accuracy,
            "flan_t5_v0_4_mean_field_accuracy": flan_mean_accuracy,
            "mean_field_accuracy_delta": llama_mean_accuracy - flan_mean_accuracy,
            "llama_2_mean_macro_f1": llama_mean_macro_f1,
            "flan_t5_v0_4_mean_macro_f1": flan_mean_macro_f1,
            "mean_macro_f1_delta": llama_mean_macro_f1 - flan_mean_macro_f1,
            "llama_improves_both_aggregate_metrics": (
                llama_mean_accuracy > flan_mean_accuracy
                and llama_mean_macro_f1 > flan_mean_macro_f1
            ),
        },
    }


def main() -> int:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Evaluation output already exists; pass --overwrite to replace: {args.output}"
        )
    prediction_manifest_path = args.prediction_manifest or args.predictions.with_suffix(
        args.predictions.suffix + ".manifest.json"
    )
    for path in (
        args.predictions,
        prediction_manifest_path,
        args.inputs,
        args.benchmark_manifest,
        args.coarse_reference,
        args.fine_reference,
        args.schema,
        args.flan_baseline,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    schema = coarse.load_schema(args.schema)
    inputs = base.read_jsonl(args.inputs)
    predictions = base.read_jsonl(args.predictions)
    coarse_reference = base.read_jsonl(args.coarse_reference)
    fine_reference = base.read_jsonl(args.fine_reference)
    input_by_id = _records_by_id(inputs, source_name="evaluation inputs")
    prediction_by_id = _records_by_id(predictions, source_name="Llama predictions")
    if len(inputs) != EXPECTED_EVALUATION_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_EVALUATION_COUNT} evaluation inputs, found {len(inputs)}"
        )
    if [record["article_id"] for record in predictions] != [
        record["article_id"] for record in inputs
    ]:
        raise ValueError("Prediction IDs and order must exactly match evaluation inputs")

    benchmark_manifest = json.loads(args.benchmark_manifest.read_text(encoding="utf-8"))
    prediction_manifest = json.loads(
        prediction_manifest_path.read_text(encoding="utf-8")
    )
    validate_benchmark_manifest(
        benchmark_manifest,
        inputs_path=args.inputs,
        fine_reference_path=args.fine_reference,
        coarse_reference_path=args.coarse_reference,
        schema_path=args.schema,
    )
    validate_prediction_manifest(
        prediction_manifest,
        predictions_path=args.predictions,
        inputs_path=args.inputs,
        schema_path=args.schema,
        predictions=predictions,
    )
    validate_prediction_rows(
        predictions,
        inputs_by_id=input_by_id,
        schema=schema,
        manifest=prediction_manifest,
    )
    baseline = json.loads(args.flan_baseline.read_text(encoding="utf-8"))
    report = build_evaluation_report(
        inputs=inputs,
        predictions=predictions,
        coarse_reference=coarse_reference,
        fine_reference=fine_reference,
        schema=schema,
        flan_baseline_summary=baseline,
        prediction_manifest=prediction_manifest,
    )
    report["integrity"] = {
        "benchmark_manifest_path": str(args.benchmark_manifest),
        "benchmark_manifest_sha256": base.sha256_file(args.benchmark_manifest),
        "prediction_manifest_path": str(prediction_manifest_path),
        "prediction_manifest_sha256": base.sha256_file(prediction_manifest_path),
        "inputs_sha256": base.sha256_file(args.inputs),
        "predictions_sha256": base.sha256_file(args.predictions),
        "coarse_reference_sha256": base.sha256_file(args.coarse_reference),
        "fine_reference_sha256": base.sha256_file(args.fine_reference),
        "schema_sha256": base.sha256_file(args.schema),
        "flan_v0_4_baseline_sha256": base.sha256_file(args.flan_baseline),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        f"Wrote Llama 2 agreement report for {len(inputs)} records to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
