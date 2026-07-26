from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import evaluate_flan_agreement as agreement
import extract_flan_t5 as base


MODEL_ID = "google/flan-t5-xl"
MODEL_REVISION = "7d6315df2c2fb742f0f5b556879d730926ca9001"
CONFIG_VERSION = "flan-t5-xl-score-calibration-v1"
FIELDS = (
    "shock_scope",
    "event_family",
    "information_status",
    "directional_alignment",
)
SCORE_SOURCES = {
    "averaged": "order_averaged_mean_log_probabilities",
    "canonical": "canonical_candidate_mean_log_probabilities",
    "reversed": "reversed_candidate_mean_log_probabilities",
}
SOURCE_TIE_PRIORITY = {"averaged": 2, "canonical": 1, "reversed": 0}
TAU_GRID = (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)
PRIOR_SMOOTHING = 0.5
FALLBACK_BY_FIELD = {
    "shock_scope": "unclear",
    "event_family": "other_or_unclear",
    "information_status": "unclear",
    "directional_alignment": "unclear",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select or apply the frozen FLAN-T5-XL v1.1 fieldwise score-source "
            "and log-prior calibration rule."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    select = subparsers.add_parser(
        "select",
        help="Select rules using development labels, then write calibrated predictions.",
    )
    select.add_argument("--predictions", required=True, type=Path)
    select.add_argument("--reference", required=True, type=Path)
    select.add_argument("--schema", required=True, type=Path)
    select.add_argument("--output", required=True, type=Path)
    select.add_argument("--config-output", required=True, type=Path)
    select.add_argument("--report-output", required=True, type=Path)
    select.add_argument("--overwrite", action="store_true")

    apply = subparsers.add_parser(
        "apply",
        help="Apply a previously selected rule without reading any labels.",
    )
    apply.add_argument("--predictions", required=True, type=Path)
    apply.add_argument("--schema", required=True, type=Path)
    apply.add_argument("--config", required=True, type=Path)
    apply.add_argument("--output", required=True, type=Path)
    apply.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_prediction_contract(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = base.read_jsonl(path)
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("Source prediction manifest is not complete")
    if manifest.get("output_sha256") != base.sha256_file(path):
        raise ValueError("Source predictions differ from their manifest")
    if manifest.get("model_id") != MODEL_ID:
        raise ValueError(f"Expected model_id {MODEL_ID}")
    if manifest.get("model_revision") != MODEL_REVISION:
        raise ValueError(f"Expected model revision {MODEL_REVISION}")
    if manifest.get("total_prediction_count") != len(records):
        raise ValueError("Source prediction count differs from its manifest")
    ids = [record.get("article_id") for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Source predictions contain duplicate article_id values")
    expected_order_hash = base.sha256_text("\n".join(ids))
    if manifest.get("selected_article_ids_sha256") != expected_order_hash:
        raise ValueError("Source prediction IDs/order differ from their manifest")
    return records, manifest


def load_schema(path: Path) -> dict[str, Any]:
    schema = json.loads(path.read_text(encoding="utf-8"))
    if tuple(schema.get("flan_core_fields", ())) != FIELDS:
        raise ValueError("Schema fields differ from the frozen XL calibration fields")
    return schema


def class_priors(
    reference_records: list[dict[str, Any]],
    allowed: list[str],
    field: str,
    smoothing: float = PRIOR_SMOOTHING,
) -> dict[str, float]:
    relevant = [
        record for record in reference_records if record.get("semantic_applicable")
    ]
    counts = {label: 0 for label in allowed}
    for record in relevant:
        value = record["labels"][field]
        if value not in counts:
            raise ValueError(f"Reference has invalid {field} label {value!r}")
        counts[value] += 1
    denominator = len(relevant) + smoothing * len(allowed)
    return {
        label: (counts[label] + smoothing) / denominator for label in allowed
    }


def source_scores(pass_record: dict[str, Any], source: str) -> dict[str, float]:
    key = SCORE_SOURCES[source]
    scores = pass_record.get(key)
    if not isinstance(scores, dict) or len(scores) < 2:
        raise ValueError(f"Model pass has no valid {key}")
    if not all(
        isinstance(label, str) and isinstance(score, (int, float))
        for label, score in scores.items()
    ):
        raise ValueError(f"Model pass contains invalid values in {key}")
    return {label: float(score) for label, score in scores.items()}


def calibrated_choice(
    pass_record: dict[str, Any],
    *,
    source: str,
    tau: float,
    priors: dict[str, float],
    schema_order: list[str],
) -> tuple[str, dict[str, float]]:
    scores = source_scores(pass_record, source)
    if not set(scores) <= set(priors):
        raise ValueError("Candidate scores contain labels missing from calibrated priors")
    adjusted = {
        label: score - tau * math.log(priors[label])
        for label, score in scores.items()
    }
    available_in_schema_order = [label for label in schema_order if label in adjusted]
    selected = max(available_in_schema_order, key=lambda label: adjusted[label])
    return selected, adjusted


def guess_for_record(
    record: dict[str, Any],
    field: str,
    *,
    source: str,
    tau: float,
    priors: dict[str, float],
    schema_order: list[str],
) -> str:
    if not record.get("semantic_applicable"):
        return FALLBACK_BY_FIELD[field]
    pass_record = record["passes"][field]
    if pass_record.get("origin") != "model":
        value = record["labels"][field]
        if value not in schema_order:
            raise ValueError(f"Derived {field} label is invalid: {value!r}")
        return value
    value, _ = calibrated_choice(
        pass_record,
        source=source,
        tau=tau,
        priors=priors,
        schema_order=schema_order,
    )
    return value


def candidate_metrics(
    prediction_by_id: dict[str, dict[str, Any]],
    reference_records: list[dict[str, Any]],
    field: str,
    *,
    source: str,
    tau: float,
    priors: dict[str, float],
    schema_order: list[str],
) -> dict[str, Any]:
    selected_reference = [
        record
        for record in reference_records
        if record.get("semantic_applicable")
        and record["article_id"] in prediction_by_id
    ]
    truths = [record["labels"][field] for record in selected_reference]
    guesses = [
        guess_for_record(
            prediction_by_id[record["article_id"]],
            field,
            source=source,
            tau=tau,
            priors=priors,
            schema_order=schema_order,
        )
        for record in selected_reference
    ]
    return agreement.closed_label_metrics(truths, guesses, schema_order)


def selection_key(candidate: dict[str, Any]) -> tuple[float, float, float, float, int]:
    metrics = candidate["metrics"]
    return (
        metrics["macro_f1"],
        metrics["accuracy"],
        metrics["cohens_kappa"],
        -abs(candidate["tau"]),
        SOURCE_TIE_PRIORITY[candidate["source"]],
    )


def select_rules(
    predictions: list[dict[str, Any]],
    reference: list[dict[str, Any]],
    schema: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    prediction_by_id = {record["article_id"]: record for record in predictions}
    reference_ids = {record["article_id"] for record in reference}
    if not set(prediction_by_id) <= reference_ids:
        raise ValueError("Development predictions are not a subset of the reference")
    rules: dict[str, Any] = {}
    candidates_by_field: dict[str, list[dict[str, Any]]] = {}
    for field in FIELDS:
        allowed = list(schema["closed_label_fields"][field])
        priors = class_priors(reference, allowed, field)
        candidates: list[dict[str, Any]] = []
        for source in SCORE_SOURCES:
            for tau in TAU_GRID:
                metrics = candidate_metrics(
                    prediction_by_id,
                    reference,
                    field,
                    source=source,
                    tau=tau,
                    priors=priors,
                    schema_order=allowed,
                )
                candidates.append(
                    {
                        "source": source,
                        "tau": tau,
                        "metrics": {
                            "count": metrics["count"],
                            "accuracy": metrics["accuracy"],
                            "macro_f1": metrics["macro_f1"],
                            "cohens_kappa": metrics["cohens_kappa"],
                            "prediction_distribution": metrics[
                                "prediction_distribution"
                            ],
                        },
                    }
                )
        selected = max(candidates, key=selection_key)
        rules[field] = {
            "source": selected["source"],
            "tau": selected["tau"],
            "priors": priors,
            "development_metrics": selected["metrics"],
        }
        candidates_by_field[field] = sorted(
            candidates, key=selection_key, reverse=True
        )
    return rules, candidates_by_field


def apply_rules(
    predictions: list[dict[str, Any]],
    rules: dict[str, Any],
    schema: dict[str, Any],
) -> list[dict[str, Any]]:
    result = copy.deepcopy(predictions)
    for record in result:
        for field in FIELDS:
            pass_record = record["passes"][field]
            if pass_record.get("origin") != "model":
                continue
            rule = rules[field]
            old_value = record["labels"][field]
            new_value, adjusted = calibrated_choice(
                pass_record,
                source=rule["source"],
                tau=float(rule["tau"]),
                priors={key: float(value) for key, value in rule["priors"].items()},
                schema_order=list(schema["closed_label_fields"][field]),
            )
            record["labels"][field] = new_value
            record["label_origins"][field] = "model_calibrated"
            pass_record["origin"] = "model_calibrated"
            pass_record["precalibration_value"] = old_value
            pass_record["value"] = new_value
            pass_record["calibration"] = {
                "config_version": CONFIG_VERSION,
                "source": rule["source"],
                "tau": rule["tau"],
                "adjusted_scores": adjusted,
            }
    return result


def write_predictions_and_manifest(
    *,
    output: Path,
    records: list[dict[str, Any]],
    source_path: Path,
    source_manifest: dict[str, Any],
    config_path: Path,
    config: dict[str, Any],
    overwrite: bool,
) -> None:
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    for path in (output, manifest_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    manifest = copy.deepcopy(source_manifest)
    manifest.update(
        {
            "status": "complete",
            "experiment_id": "flan-t5-xl-v1.1-calibrated",
            "decoding": "development_selected_source_log_prior_adjustment",
            "postprocessor": {
                "config_version": CONFIG_VERSION,
                "source_prediction_path": str(source_path),
                "source_prediction_sha256": base.sha256_file(source_path),
                "config_path": str(config_path),
                "config_sha256": base.sha256_file(config_path),
                "rules_sha256": sha256_json(config["field_rules"]),
                "implementation_sha256": base.sha256_file(Path(__file__).resolve()),
            },
            "output_sha256": base.sha256_file(output),
            "total_prediction_count": len(records),
        }
    )
    base.write_manifest(manifest_path, manifest)


def config_contract(
    *,
    rules: dict[str, Any],
    candidates: dict[str, list[dict[str, Any]]],
    predictions_path: Path,
    reference_path: Path,
    schema_path: Path,
) -> dict[str, Any]:
    selected_metrics = [rules[field]["development_metrics"] for field in FIELDS]
    return {
        "config_version": CONFIG_VERSION,
        "status": "frozen_on_development",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "selection_split": "fixed 72-document development split",
        "selection_objective": [
            "field macro_f1",
            "field accuracy",
            "field cohens_kappa",
            "smaller absolute tau",
            "averaged_then_canonical_then_reversed",
        ],
        "tau_grid": list(TAU_GRID),
        "prior_smoothing": PRIOR_SMOOTHING,
        "source_prediction_path": str(predictions_path),
        "source_prediction_sha256": base.sha256_file(predictions_path),
        "reference_path": str(reference_path),
        "reference_sha256": base.sha256_file(reference_path),
        "schema_path": str(schema_path),
        "schema_sha256": base.sha256_file(schema_path),
        "field_rules": rules,
        "development_aggregate": {
            "mean_accuracy": sum(
                metric["accuracy"] for metric in selected_metrics
            )
            / len(selected_metrics),
            "mean_macro_f1": sum(
                metric["macro_f1"] for metric in selected_metrics
            )
            / len(selected_metrics),
        },
        "candidate_rankings": candidates,
        "claim_limit": (
            "Supervised postprocessing selected against GPT-5.6 silver labels; "
            "requires a single locked evaluation on the untouched 228-document split."
        ),
    }


def validate_apply_config(config: dict[str, Any]) -> None:
    if config.get("config_version") != CONFIG_VERSION:
        raise ValueError("Unsupported XL calibration config version")
    if config.get("status") != "frozen_on_development":
        raise ValueError("XL calibration config is not frozen")
    if config.get("model_id") != MODEL_ID or config.get("model_revision") != MODEL_REVISION:
        raise ValueError("XL calibration config targets a different model")
    rules = config.get("field_rules")
    if not isinstance(rules, dict) or set(rules) != set(FIELDS):
        raise ValueError("XL calibration config has invalid field rules")
    for field, rule in rules.items():
        if rule.get("source") not in SCORE_SOURCES:
            raise ValueError(f"Invalid score source for {field}")
        if float(rule.get("tau")) not in TAU_GRID:
            raise ValueError(f"Invalid tau for {field}")


def main() -> int:
    args = parse_args()
    schema = load_schema(args.schema)
    predictions, source_manifest = load_prediction_contract(args.predictions)
    if args.command == "select":
        for path in (args.output, args.config_output, args.report_output):
            if path.exists() and not args.overwrite:
                raise FileExistsError(f"{path} exists; pass --overwrite")
        reference = base.read_jsonl(args.reference)
        rules, candidates = select_rules(predictions, reference, schema)
        config = config_contract(
            rules=rules,
            candidates=candidates,
            predictions_path=args.predictions,
            reference_path=args.reference,
            schema_path=args.schema,
        )
        args.config_output.parent.mkdir(parents=True, exist_ok=True)
        base.write_manifest(args.config_output, config)
        calibrated = apply_rules(predictions, rules, schema)
        write_predictions_and_manifest(
            output=args.output,
            records=calibrated,
            source_path=args.predictions,
            source_manifest=source_manifest,
            config_path=args.config_output,
            config=config,
            overwrite=args.overwrite,
        )
        report = {
            "title": "FLAN-T5-XL v1.1 development selection",
            "config_path": str(args.config_output),
            "config_sha256": base.sha256_file(args.config_output),
            "development_aggregate": config["development_aggregate"],
            "field_rules": rules,
            "candidate_count_per_field": len(TAU_GRID) * len(SCORE_SOURCES),
        }
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        base.write_manifest(args.report_output, report)
        print(f"Wrote frozen calibration config to {args.config_output}")
        print(f"Wrote calibrated predictions to {args.output}")
        return 0

    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_apply_config(config)
    calibrated = apply_rules(predictions, config["field_rules"], schema)
    write_predictions_and_manifest(
        output=args.output,
        records=calibrated,
        source_path=args.predictions,
        source_manifest=source_manifest,
        config_path=args.config,
        config=config,
        overwrite=args.overwrite,
    )
    print(f"Applied frozen calibration config to {len(calibrated)} predictions")
    print(f"Wrote calibrated predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
