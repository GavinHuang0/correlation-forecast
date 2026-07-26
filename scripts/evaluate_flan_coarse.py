from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import coarse_news_features as coarse
import evaluate_flan_agreement as legacy_eval
import extract_flan_t5 as base


FALLBACK_BY_FIELD = {
    "shock_scope": "unclear",
    "event_family": "other_or_unclear",
    "information_status": "unclear",
    "directional_alignment": "unclear",
}
ACCEPTED_PROMPT_VERSIONS = frozenset(
    {
        "flan-stock-sector-news-v0.4.0",
        "flan-stock-sector-news-hybrid-v0.4.0",
        "llama-3.1-stock-sector-news-v1.0.0",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the v0.3 FLAN coarse pipeline against deterministically coarsened GPT-5.6 silver labels."
        )
    )
    parser.add_argument("--reference", required=True, type=Path, help="Original fine-label reference JSONL")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument(
        "--schema", default=Path("config/news_feature_schema_coarse.json"), type=Path
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--split",
        choices=("all", "development", "evaluation"),
        default="all",
        help="Use the fixed, stratified v0.3 development/evaluation assignment.",
    )
    parser.add_argument(
        "--old-fine-predictions",
        type=Path,
        help="Optional v0.2 predictions to map onto the same coarse task as a fair baseline.",
    )
    return parser.parse_args()


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def binary_metrics(truths: list[bool], guesses: list[bool]) -> dict[str, Any]:
    tp = sum(t and g for t, g in zip(truths, guesses))
    tn = sum((not t) and (not g) for t, g in zip(truths, guesses))
    fp = sum((not t) and g for t, g in zip(truths, guesses))
    fn = sum(t and (not g) for t, g in zip(truths, guesses))
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    return {
        "count": len(truths),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "accuracy": safe_divide(tp + tn, len(truths)),
        "precision": precision,
        "recall": recall,
        "f1": safe_divide(2 * precision * recall, precision + recall),
    }


def stable_development_ids(
    coarse_reference_by_id: dict[str, dict[str, Any]], fraction: float = 0.24
) -> set[str]:
    split = coarse.stratified_split(
        list(coarse_reference_by_id.values()), development_fraction=fraction
    )
    return {record["article_id"] for record in split["development"]}


def majority_baseline(truths: list[str], allowed: list[str]) -> dict[str, Any]:
    counts = Counter(truths)
    majority = max(allowed, key=lambda label: (counts[label], -allowed.index(label)))
    metrics = legacy_eval.closed_label_metrics(truths, [majority] * len(truths), allowed)
    return {"majority_label": majority, **metrics}


def map_reference_records(
    reference: list[dict[str, Any]], inputs_by_id: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    mapped: dict[str, dict[str, Any]] = {}
    for record in reference:
        article_id = record["article_id"]
        if article_id not in inputs_by_id:
            raise ValueError(f"Reference article {article_id} is absent from the input file")
        input_ticker = inputs_by_id[article_id]["target"]["ticker"]
        if record.get("target_ticker") != input_ticker:
            raise ValueError(
                f"Reference/input target mismatch for {article_id}: "
                f"{record.get('target_ticker')!r} != {input_ticker!r}"
            )
        fine = record["labels"]
        labels = coarse.map_fine_labels(fine)
        coarse.validate_coarse_labels(labels)
        mapped[article_id] = {
            "article_id": article_id,
            "row_number": record.get("row_number", inputs_by_id[article_id]["row_number"]),
            "target_ticker": input_ticker,
            "semantic_applicable": coarse.semantic_applicable(fine),
            "labels": labels,
        }
    return mapped


def mapped_old_predictions(
    path: Path, selected_ids: list[str]
) -> dict[str, dict[str, str]]:
    records = base.read_jsonl(path)
    by_id = {record["article_id"]: record for record in records}
    missing = set(selected_ids) - set(by_id)
    if missing:
        raise ValueError(f"Old prediction file is missing {len(missing)} selected articles")
    return {
        article_id: coarse.map_fine_labels(by_id[article_id]["labels"])
        for article_id in selected_ids
    }


def metrics_for_prediction_source(
    *,
    selected_ids: list[str],
    reference_by_id: dict[str, dict[str, Any]],
    prediction_labels_by_id: dict[str, dict[str, str | None]],
    schema: dict[str, Any],
    include_only_reference_applicable: bool = True,
) -> dict[str, Any]:
    ids = [
        article_id
        for article_id in selected_ids
        if not include_only_reference_applicable
        or reference_by_id[article_id]["semantic_applicable"]
    ]
    result: dict[str, Any] = {}
    for field in coarse.COARSE_FIELDS:
        allowed = schema["closed_label_fields"][field]
        truths = [reference_by_id[article_id]["labels"][field] for article_id in ids]
        guesses = [prediction_labels_by_id[article_id].get(field) for article_id in ids]
        result[field] = legacy_eval.closed_label_metrics(truths, guesses, allowed)
    return result


def main() -> int:
    args = parse_args()
    schema_text = args.schema.read_text(encoding="utf-8")
    schema = json.loads(schema_text)
    predictions = base.read_jsonl(args.predictions)
    reference = base.read_jsonl(args.reference)
    inputs = base.read_jsonl(args.inputs)
    prediction_by_id = {record["article_id"]: record for record in predictions}
    reference_by_id_fine = {record["article_id"]: record for record in reference}
    inputs_by_id = {record["article_id"]: record for record in inputs}
    for name, records, mapping in (
        ("prediction", predictions, prediction_by_id),
        ("reference", reference, reference_by_id_fine),
        ("input", inputs, inputs_by_id),
    ):
        if len(records) != len(mapping):
            raise ValueError(f"Duplicate article_id values in {name} file")
    if not set(prediction_by_id) <= set(reference_by_id_fine) or not set(prediction_by_id) <= set(inputs_by_id):
        raise ValueError("Prediction IDs must be a subset of reference and input IDs")

    manifest_path = args.manifest or args.predictions.with_suffix(
        args.predictions.suffix + ".manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("Prediction manifest is not complete")
    if manifest.get("output_sha256") != base.sha256_file(args.predictions):
        raise ValueError("Prediction hash differs from its manifest")
    if manifest.get("schema_sha256") != hashlib.sha256(schema_text.encode("utf-8")).hexdigest():
        raise ValueError("Evaluation schema differs from the extraction schema")
    if manifest.get("prompt_version") not in ACCEPTED_PROMPT_VERSIONS:
        raise ValueError(
            "Predictions were not produced by an accepted coarse extraction protocol"
        )
    if manifest.get("total_prediction_count") != len(predictions):
        raise ValueError("Prediction count differs from the extraction manifest")
    prediction_order_hash = base.sha256_text(
        "\n".join(record["article_id"] for record in predictions)
    )
    if manifest.get("selected_article_ids_sha256") != prediction_order_hash:
        raise ValueError("Prediction order/IDs differ from the extraction manifest")
    if manifest.get("deterministic_module_sha256") != base.sha256_file(
        Path(coarse.__file__).resolve()
    ):
        raise ValueError("Current deterministic module differs from the extraction manifest")

    row_contract_errors: list[str] = []
    for prediction in predictions:
        article_id = prediction["article_id"]
        expected_ticker = inputs_by_id[article_id]["target"]["ticker"]
        if prediction.get("target_ticker") != expected_ticker:
            row_contract_errors.append(f"{article_id}: target_ticker mismatch")
        if prediction.get("model_revision") != manifest.get("model_revision"):
            row_contract_errors.append(f"{article_id}: model_revision mismatch")
        if prediction.get("prompt_version") != manifest.get("prompt_version"):
            row_contract_errors.append(f"{article_id}: prompt_version mismatch")
        labels = prediction.get("labels")
        origins = prediction.get("label_origins")
        passes = prediction.get("passes")
        if not isinstance(labels, dict) or set(labels) != set(coarse.COARSE_FIELDS):
            row_contract_errors.append(f"{article_id}: label fields mismatch")
            continue
        if not isinstance(origins, dict) or set(origins) != set(coarse.COARSE_FIELDS):
            row_contract_errors.append(f"{article_id}: label origin fields mismatch")
            continue
        if not isinstance(passes, dict) or set(passes) != set(coarse.COARSE_FIELDS):
            row_contract_errors.append(f"{article_id}: pass fields mismatch")
            continue
        for field in coarse.COARSE_FIELDS:
            value = labels[field]
            if value is not None and value not in schema["closed_label_fields"][field]:
                row_contract_errors.append(f"{article_id}: invalid {field}={value!r}")
            pass_record = passes[field]
            if pass_record.get("value") != value:
                row_contract_errors.append(f"{article_id}: {field} pass/label mismatch")
            if pass_record.get("origin") != origins[field]:
                row_contract_errors.append(f"{article_id}: {field} pass/origin mismatch")
            if pass_record.get("valid") is not True or pass_record.get("input_truncated"):
                row_contract_errors.append(f"{article_id}: invalid or truncated {field} pass")
        applicable = bool(prediction.get("semantic_applicable"))
        if applicable and any(labels[field] is None for field in coarse.COARSE_FIELDS):
            row_contract_errors.append(f"{article_id}: applicable row has unresolved labels")
        if not applicable and any(labels[field] is not None for field in coarse.COARSE_FIELDS):
            row_contract_errors.append(f"{article_id}: gated row has fabricated semantic labels")
    if row_contract_errors:
        raise ValueError(
            "Prediction row contract validation failed: " + "; ".join(row_contract_errors[:20])
        )

    coarse_reference_by_id = map_reference_records(reference, inputs_by_id)
    development_ids = stable_development_ids(coarse_reference_by_id)
    available_ids = set(prediction_by_id)
    if args.split == "development":
        chosen = available_ids & development_ids
    elif args.split == "evaluation":
        chosen = available_ids - development_ids
    else:
        chosen = available_ids
    selected_ids = sorted(chosen, key=lambda article_id: inputs_by_id[article_id]["row_number"])
    if not selected_ids:
        raise ValueError(f"No predictions available for split {args.split!r}")

    replay_errors: list[str] = []
    for article_id in selected_ids:
        replay = coarse.deterministic_features(inputs_by_id[article_id])
        stored = prediction_by_id[article_id].get("deterministic_features")
        if replay != stored:
            replay_errors.append(article_id)
        if prediction_by_id[article_id].get("semantic_applicable") != replay[
            "semantic_applicable"
        ]:
            replay_errors.append(f"{article_id}:semantic_applicable")
    if replay_errors:
        raise ValueError(
            f"Stored deterministic features failed replay for {len(replay_errors)} records: "
            + ", ".join(replay_errors[:10])
        )

    reference_gate = [
        coarse_reference_by_id[article_id]["semantic_applicable"] for article_id in selected_ids
    ]
    predicted_gate = [
        bool(prediction_by_id[article_id].get("semantic_applicable")) for article_id in selected_ids
    ]
    gate_metrics = binary_metrics(reference_gate, predicted_gate)
    surprise_truths = [
        reference_by_id_fine[article_id]["labels"]["explicit_surprise"]
        for article_id in selected_ids
    ]
    surprise_guesses = [
        prediction_by_id[article_id]["deterministic_features"]["explicit_surprise_rule"]
        for article_id in selected_ids
    ]
    surprise_metrics = legacy_eval.closed_label_metrics(
        surprise_truths,
        surprise_guesses,
        ["positive", "negative", "mixed", "none", "unknown"],
    )
    reference_relevant_ids = [
        article_id
        for article_id in selected_ids
        if coarse_reference_by_id[article_id]["semantic_applicable"]
    ]
    gate_true_positive_ids = [
        article_id
        for article_id in reference_relevant_ids
        if prediction_by_id[article_id].get("semantic_applicable")
    ]

    pipeline_labels = {
        article_id: prediction_by_id[article_id].get("labels", {}) for article_id in selected_ids
    }
    end_to_end_labels: dict[str, dict[str, str]] = {}
    for article_id in selected_ids:
        labels = pipeline_labels[article_id]
        end_to_end_labels[article_id] = {
            field: labels.get(field) or FALLBACK_BY_FIELD[field]
            for field in coarse.COARSE_FIELDS
        }

    conditional_metrics: dict[str, Any] = {}
    end_to_end_metrics: dict[str, Any] = {}
    majority: dict[str, Any] = {}
    origin_counts: dict[str, Any] = {}
    for field in coarse.COARSE_FIELDS:
        allowed = schema["closed_label_fields"][field]
        conditional_truths = [
            coarse_reference_by_id[article_id]["labels"][field]
            for article_id in gate_true_positive_ids
        ]
        conditional_guesses = [
            pipeline_labels[article_id].get(field) for article_id in gate_true_positive_ids
        ]
        conditional_metrics[field] = legacy_eval.closed_label_metrics(
            conditional_truths, conditional_guesses, allowed
        )
        truths = [
            coarse_reference_by_id[article_id]["labels"][field]
            for article_id in reference_relevant_ids
        ]
        guesses = [end_to_end_labels[article_id][field] for article_id in reference_relevant_ids]
        end_to_end_metrics[field] = legacy_eval.closed_label_metrics(truths, guesses, allowed)
        majority[field] = majority_baseline(truths, allowed)
        origin_counts[field] = dict(
            sorted(
                Counter(
                    prediction_by_id[article_id].get("label_origins", {}).get(field, "missing")
                    for article_id in selected_ids
                ).items()
            )
        )

    decoder_diagnostics: dict[str, Any] = {}
    if manifest.get("decoding") == "order_averaged_letter_score":
        for alternative_key in ("canonical_prediction", "reversed_prediction"):
            alternative_labels: dict[str, dict[str, str | None]] = {}
            for article_id in selected_ids:
                result = prediction_by_id[article_id]
                labels: dict[str, str | None] = {}
                for field in coarse.COARSE_FIELDS:
                    pass_record = result.get("passes", {}).get(field, {})
                    labels[field] = (
                        pass_record.get(alternative_key)
                        if pass_record.get("origin") == "model"
                        else result.get("labels", {}).get(field)
                    )
                alternative_labels[article_id] = labels
            decoder_diagnostics[alternative_key] = metrics_for_prediction_source(
                selected_ids=selected_ids,
                reference_by_id=coarse_reference_by_id,
                prediction_labels_by_id=alternative_labels,
                schema=schema,
            )

    old_baseline: dict[str, Any] | None = None
    if args.old_fine_predictions:
        old_manifest_path = args.old_fine_predictions.with_suffix(
            args.old_fine_predictions.suffix + ".manifest.json"
        )
        if not old_manifest_path.exists():
            raise FileNotFoundError(f"Missing old-baseline manifest: {old_manifest_path}")
        old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8"))
        if old_manifest.get("status") != "complete" or old_manifest.get(
            "output_sha256"
        ) != base.sha256_file(args.old_fine_predictions):
            raise ValueError("Old v0.2 baseline failed manifest/hash validation")
        if old_manifest.get("prompt_version") != "flan-stock-sector-news-v0.2.0":
            raise ValueError("Old baseline is not a v0.2 extraction run")
        old_labels = mapped_old_predictions(args.old_fine_predictions, selected_ids)
        old_baseline = metrics_for_prediction_source(
            selected_ids=selected_ids,
            reference_by_id=coarse_reference_by_id,
            prediction_labels_by_id=old_labels,
            schema=schema,
        )

    fieldwise_delta_vs_v0_2: dict[str, Any] | None = None
    aggregate_comparison: dict[str, Any] | None = None
    if old_baseline is not None:
        fieldwise_delta_vs_v0_2 = {
            field: {
                "accuracy_delta": end_to_end_metrics[field]["accuracy"]
                - old_baseline[field]["accuracy"],
                "macro_f1_delta": end_to_end_metrics[field]["macro_f1"]
                - old_baseline[field]["macro_f1"],
            }
            for field in coarse.COARSE_FIELDS
        }
        new_mean_accuracy = sum(
            end_to_end_metrics[field]["accuracy"] for field in coarse.COARSE_FIELDS
        ) / len(coarse.COARSE_FIELDS)
        old_mean_accuracy = sum(
            old_baseline[field]["accuracy"] for field in coarse.COARSE_FIELDS
        ) / len(coarse.COARSE_FIELDS)
        new_mean_macro_f1 = sum(
            end_to_end_metrics[field]["macro_f1"] for field in coarse.COARSE_FIELDS
        ) / len(coarse.COARSE_FIELDS)
        old_mean_macro_f1 = sum(
            old_baseline[field]["macro_f1"] for field in coarse.COARSE_FIELDS
        ) / len(coarse.COARSE_FIELDS)
        aggregate_comparison = {
            "new_mean_field_accuracy": new_mean_accuracy,
            "old_v0_2_mean_field_accuracy": old_mean_accuracy,
            "mean_field_accuracy_delta": new_mean_accuracy - old_mean_accuracy,
            "new_mean_macro_f1": new_mean_macro_f1,
            "old_v0_2_mean_macro_f1": old_mean_macro_f1,
            "mean_macro_f1_delta": new_mean_macro_f1 - old_mean_macro_f1,
            "mean_macro_f1_relative_change": safe_divide(
                new_mean_macro_f1 - old_mean_macro_f1, old_mean_macro_f1
            ),
        }

    thresholds = {
        "relevance_gate_f1": 0.90,
        **schema.get("suggested_go_no_go_thresholds", {}),
    }
    threshold_results = {
        "relevance_gate_f1": gate_metrics["f1"] >= thresholds.get("relevance_gate_f1", 0.0),
        **{
            f"{field}_macro_f1": end_to_end_metrics[field]["macro_f1"]
            >= thresholds.get(f"{field}_macro_f1", 0.0)
            for field in coarse.COARSE_FIELDS
        },
    }
    report = {
        "reference_type": "deterministically coarsened GPT-5.6 Sol silver annotations",
        "warning": (
            "The coarse reference is not new human ground truth. This 300-record dataset has already informed "
            "protocol design and remains a development benchmark."
        ),
        "split": args.split,
        "record_count": len(selected_ids),
        "reference_relevant_count": len(reference_relevant_ids),
        "development_assignment_count_full_reference": len(development_ids),
        "gate_true_positive_semantic_count": len(gate_true_positive_ids),
        "semantic_coverage_on_reference_relevant": safe_divide(
            len(gate_true_positive_ids), len(reference_relevant_ids)
        ),
        "relevance_gate_metrics": gate_metrics,
        "deterministic_explicit_surprise_metrics": surprise_metrics,
        "semantic_metrics_conditional_on_reference_relevant_and_gate_pass": conditional_metrics,
        "end_to_end_semantic_metrics_on_reference_relevant": end_to_end_metrics,
        "majority_baselines_on_reference_relevant": majority,
        "old_v0_2_predictions_mapped_to_same_coarse_task": old_baseline,
        "fieldwise_delta_vs_v0_2": fieldwise_delta_vs_v0_2,
        "aggregate_comparison_vs_v0_2": aggregate_comparison,
        "choice_order_diagnostics": decoder_diagnostics,
        "label_origin_counts": origin_counts,
        "go_no_go_thresholds": thresholds,
        "threshold_results": threshold_results,
        "extraction_manifest": {
            "path": str(manifest_path),
            "model_id": manifest.get("model_id"),
            "model_revision": manifest.get("model_revision"),
            "prompt_version": manifest.get("prompt_version"),
            "decoding": manifest.get("decoding"),
            "prompt_profile": manifest.get("prompt_profile"),
            "deterministic_rule_version": manifest.get("deterministic_rule_version"),
            "device": manifest.get("device"),
            "precision": manifest.get("precision"),
        },
        "input_sha256": base.sha256_file(args.inputs),
        "reference_sha256": base.sha256_file(args.reference),
        "prediction_sha256": base.sha256_file(args.predictions),
        "old_v0_2_prediction_sha256": (
            base.sha256_file(args.old_fine_predictions) if args.old_fine_predictions else None
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote coarse agreement report for {len(selected_ids)} records to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
