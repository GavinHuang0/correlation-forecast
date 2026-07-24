from __future__ import annotations

"""Calibrate, materialize, and decide whether to promote FLAN-T5 v0.5.

This module deliberately does not modify or call the legacy extraction
pipelines.  It consumes immutable *raw* yes/no score records from the v0.5
extractor and applies a development-fitted decision layer.

Expected raw score components
-----------------------------

``scope_firm``, ``scope_common``, the six ``event_<family>`` components,
``alignment_same``, and ``alignment_opposite``.  Every applicable component
must contain:

* ``order_averaged_yes_no_log_odds`` (finite number);
* ``strict_consensus_yes`` (boolean);
* ``valid`` (boolean);
* ``input_truncated`` (boolean).

The development baseline is reconstructed from two already-frozen sources:
the coarse v0.4 development run supplies ``event_family`` and
``information_status`` while the fine v0.2 development-covering run supplies
the mapped ``shock_scope`` and ``directional_alignment`` fields.  The
evaluation baseline is the already-materialized v0.4 hybrid.

Threshold/tie conventions are explicit and deterministic:

* threshold decisions use ``score >= threshold``;
* candidates are the two finite outer sentinels and every adjacent midpoint;
* joint scope ties prefer macro-F1, accuracy, mixed F1, then earlier
  lexicographically enumerated threshold candidates;
* event one-vs-rest ties prefer positive-class F1, accuracy, precision, then
  the higher (more conservative) threshold;
* multiple positive event families are resolved by the largest calibrated
  margin ``score - threshold``, then the fixed family order.

The existing 300-record silver benchmark has already informed protocol
development.  Promotion here is therefore an engineering freeze decision, not
an uncontaminated scientific validation claim.
"""

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import coarse_news_features as coarse
import evaluate_flan_agreement as legacy_eval
import extract_flan_t5 as base


SELECTOR_VERSION = "flan-stock-sector-news-selector-v0.5.0"
PREDICTION_VERSION = "flan-stock-sector-news-final-candidate-v0.5.0"
BOOTSTRAP_SEED = 20260723
BOOTSTRAP_SAMPLES = 10_000

SCOPE_COMPONENTS = ("scope_firm", "scope_common")
EVENT_FAMILIES = (
    "earnings_guidance",
    "product_demand",
    "supply_capacity",
    "regulation_legal",
    "corporate_analyst",
    "macro_market",
)
EVENT_COMPONENTS = tuple(f"event_{family}" for family in EVENT_FAMILIES)
ALIGNMENT_COMPONENTS = ("alignment_same", "alignment_opposite")
EXPECTED_COMPONENTS = SCOPE_COMPONENTS + EVENT_COMPONENTS + ALIGNMENT_COMPONENTS
TARGET_FIELDS = ("shock_scope", "event_family", "directional_alignment")
EXTRACTOR_IDENTITY_FIELDS = (
    "model_id",
    "model_revision",
    "prompt_version",
    "protocol_version",
    "protocol_config_sha256",
    "prompt_builder_sha256",
    "source_files_sha256",
    "candidate_component_order",
    "binary_semantics",
    "max_input_tokens",
    "precision",
    "model_files_sha256",
)

FALLBACK_BY_FIELD = {
    "shock_scope": "unclear",
    "event_family": "other_or_unclear",
    "information_status": "unclear",
    "directional_alignment": "unclear",
}

PROMOTION_THRESHOLDS = {
    "mean_macro_f1_delta": 0.02,
    "mean_accuracy_delta": 0.01,
    "maximum_field_macro_f1_drop": 0.02,
    "target_field_macro_f1_gain": 0.05,
    "valid_nontruncated_rate": 1.0,
    "bootstrap_positive_delta_fraction": 0.95,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Two-stage FLAN v0.5 decision layer. Calibrate and hash-lock using "
            "development data before running evaluation inference."
        )
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    calibrate = subparsers.add_parser(
        "calibrate", help="Fit thresholds using development sources only"
    )
    calibrate.add_argument("--development-scores", required=True, type=Path)
    calibrate.add_argument(
        "--reference", required=True, type=Path, help="Original fine silver JSONL"
    )
    calibrate.add_argument(
        "--inputs", required=True, type=Path, help="Original benchmark inputs JSONL"
    )
    calibrate.add_argument(
        "--development-coarse-v0-4",
        required=True,
        type=Path,
        help="Frozen development coarse source for event/status fields",
    )
    calibrate.add_argument(
        "--development-fine-v0-2",
        required=True,
        type=Path,
        help="Frozen fine v0.2 source covering development IDs for scope/alignment",
    )
    calibrate.add_argument(
        "--schema", default=Path("config/news_feature_schema_coarse.json"), type=Path
    )
    calibrate.add_argument("--calibration-output", required=True, type=Path)
    calibrate.add_argument("--overwrite", action="store_true")

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Apply an already hash-locked calibration without refitting",
    )
    evaluate.add_argument("--calibration", required=True, type=Path)
    evaluate.add_argument(
        "--calibration-sha256",
        required=True,
        help="Expected SHA-256 printed by the completed calibrate stage",
    )
    evaluate.add_argument("--evaluation-scores", required=True, type=Path)
    evaluate.add_argument(
        "--reference", required=True, type=Path, help="Original fine silver JSONL"
    )
    evaluate.add_argument(
        "--inputs", required=True, type=Path, help="Original benchmark inputs JSONL"
    )
    evaluate.add_argument(
        "--evaluation-v0-4-hybrid",
        required=True,
        type=Path,
        help="Frozen materialized v0.4 evaluation hybrid and status source",
    )
    evaluate.add_argument(
        "--schema", default=Path("config/news_feature_schema_coarse.json"), type=Path
    )
    evaluate.add_argument("--predictions-output", required=True, type=Path)
    evaluate.add_argument("--report-output", required=True, type=Path)
    evaluate.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_object(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, records: Sequence[Mapping[str, Any]], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(canonical_json(record) + "\n")


def index_unique(records: Sequence[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        article_id = record.get("article_id")
        if not isinstance(article_id, str) or not article_id:
            raise ValueError(f"{name} contains a record without a valid article_id")
        if article_id in result:
            raise ValueError(f"{name} contains duplicate article_id {article_id!r}")
        result[article_id] = record
    return result


def require_complete_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing source manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"Source manifest is not complete: {manifest_path}")
    if manifest.get("output_sha256") != base.sha256_file(path):
        raise ValueError(f"Source hash differs from manifest: {path}")
    return manifest


def finite_number(value: Any, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{description} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{description} must be finite")
    return result


def validate_raw_score_record(record: Mapping[str, Any]) -> None:
    article_id = record.get("article_id")
    ticker = record.get("target_ticker")
    if not isinstance(article_id, str) or not article_id:
        raise ValueError("Raw score row needs a nonempty article_id")
    if not isinstance(ticker, str) or not ticker:
        raise ValueError(f"{article_id}: raw score row needs target_ticker")
    if not isinstance(record.get("semantic_applicable"), bool):
        raise ValueError(f"{article_id}: semantic_applicable must be boolean")
    if not isinstance(record.get("deterministic_features"), dict):
        raise ValueError(f"{article_id}: deterministic_features must be an object")
    components = record.get("components")
    if not isinstance(components, dict):
        raise ValueError(f"{article_id}: components must be an object")

    if not record["semantic_applicable"]:
        if components:
            raise ValueError(f"{article_id}: gated row must have no model components")
        return

    if set(components) != set(EXPECTED_COMPONENTS):
        missing = sorted(set(EXPECTED_COMPONENTS) - set(components))
        extra = sorted(set(components) - set(EXPECTED_COMPONENTS))
        raise ValueError(
            f"{article_id}: component contract mismatch; missing={missing}, extra={extra}"
        )
    for name in EXPECTED_COMPONENTS:
        component = components[name]
        if not isinstance(component, dict):
            raise ValueError(f"{article_id}: component {name} must be an object")
        finite_number(
            component.get("order_averaged_yes_no_log_odds"),
            f"{article_id}.{name}.order_averaged_yes_no_log_odds",
        )
        if not isinstance(component.get("strict_consensus_yes"), bool):
            raise ValueError(f"{article_id}.{name}.strict_consensus_yes must be boolean")
        if not isinstance(component.get("valid"), bool):
            raise ValueError(f"{article_id}.{name}.valid must be boolean")
        if not isinstance(component.get("input_truncated"), bool):
            raise ValueError(f"{article_id}.{name}.input_truncated must be boolean")


def component_score(record: Mapping[str, Any], component: str) -> float:
    return float(record["components"][component]["order_averaged_yes_no_log_odds"])


def strict_consensus(record: Mapping[str, Any], component: str) -> bool:
    return bool(record["components"][component]["strict_consensus_yes"])


def threshold_candidates(values: Iterable[float]) -> list[float]:
    """Return finite outer sentinels plus all adjacent midpoints."""

    unique = sorted({finite_number(value, "threshold score") for value in values})
    if not unique:
        raise ValueError("Cannot derive thresholds from an empty score collection")
    lower = math.nextafter(unique[0], -math.inf)
    upper = math.nextafter(unique[-1], math.inf)
    if not math.isfinite(lower):
        lower = unique[0] - max(1.0, abs(unique[0]))
    if not math.isfinite(upper):
        upper = unique[-1] + max(1.0, abs(unique[-1]))
    midpoints = [
        left + (right - left) / 2.0
        for left, right in zip(unique, unique[1:])
    ]
    result = [lower, *midpoints, upper]
    if any(not math.isfinite(value) for value in result):
        raise ValueError("Threshold candidate construction produced a nonfinite value")
    return result


def predict_scope(record: Mapping[str, Any], firm_threshold: float, common_threshold: float) -> str:
    firm = component_score(record, "scope_firm") >= firm_threshold
    common = component_score(record, "scope_common") >= common_threshold
    if firm and common:
        return "mixed"
    if firm:
        return "idiosyncratic"
    if common:
        return "common"
    return "unclear"


def fit_scope_thresholds(
    records: Sequence[Mapping[str, Any]],
    truth_by_id: Mapping[str, Mapping[str, str]],
    allowed_labels: Sequence[str],
) -> dict[str, Any]:
    if not records:
        raise ValueError("Scope calibration needs at least one applicable development row")
    firm_candidates = threshold_candidates(component_score(row, "scope_firm") for row in records)
    common_candidates = threshold_candidates(
        component_score(row, "scope_common") for row in records
    )
    truths = [truth_by_id[row["article_id"]]["shock_scope"] for row in records]

    best: dict[str, Any] | None = None
    best_key: tuple[float, ...] | None = None
    for firm_index, firm_threshold in enumerate(firm_candidates):
        for common_index, common_threshold in enumerate(common_candidates):
            guesses = [
                predict_scope(row, firm_threshold, common_threshold) for row in records
            ]
            metrics = legacy_eval.closed_label_metrics(
                truths, guesses, list(allowed_labels)
            )
            key = (
                metrics["macro_f1"],
                metrics["accuracy"],
                metrics["per_class"]["mixed"]["f1"],
                -float(firm_index),
                -float(common_index),
            )
            if best_key is None or key > best_key:
                best_key = key
                best = {
                    "firm_threshold": firm_threshold,
                    "common_threshold": common_threshold,
                    "development_metrics": metrics,
                    "candidate_count": len(firm_candidates) * len(common_candidates),
                    "firm_candidate_count": len(firm_candidates),
                    "common_candidate_count": len(common_candidates),
                    "tie_break": (
                        "macro_f1, accuracy, mixed_f1, then earlier ascending "
                        "firm/common threshold candidate"
                    ),
                }
    assert best is not None
    return best


def binary_positive_metrics(truths: Sequence[bool], guesses: Sequence[bool]) -> dict[str, Any]:
    tp = sum(truth and guess for truth, guess in zip(truths, guesses))
    tn = sum((not truth) and (not guess) for truth, guess in zip(truths, guesses))
    fp = sum((not truth) and guess for truth, guess in zip(truths, guesses))
    fn = sum(truth and (not guess) for truth, guess in zip(truths, guesses))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "count": len(truths),
        "positive_support": sum(truths),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "accuracy": (tp + tn) / len(truths) if truths else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def fit_event_thresholds(
    records: Sequence[Mapping[str, Any]],
    truth_by_id: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    if not records:
        raise ValueError("Event calibration needs at least one applicable development row")
    result: dict[str, Any] = {}
    for family in EVENT_FAMILIES:
        component = f"event_{family}"
        values = [component_score(row, component) for row in records]
        candidates = threshold_candidates(values)
        truths = [
            truth_by_id[row["article_id"]]["event_family"] == family for row in records
        ]
        best: dict[str, Any] | None = None
        best_key: tuple[float, ...] | None = None
        for threshold in candidates:
            guesses = [value >= threshold for value in values]
            metrics = binary_positive_metrics(truths, guesses)
            key = (
                metrics["f1"],
                metrics["accuracy"],
                metrics["precision"],
                threshold,
            )
            if best_key is None or key > best_key:
                best_key = key
                best = {
                    "threshold": threshold,
                    "development_one_vs_rest_metrics": metrics,
                    "candidate_count": len(candidates),
                    "tie_break": (
                        "positive_f1, accuracy, precision, then higher conservative threshold"
                    ),
                }
        assert best is not None
        result[family] = best
    return result


def predict_event(record: Mapping[str, Any], event_calibration: Mapping[str, Any]) -> str:
    eligible: list[tuple[float, int, str]] = []
    for index, family in enumerate(EVENT_FAMILIES):
        score = component_score(record, f"event_{family}")
        threshold = float(event_calibration[family]["threshold"])
        if score >= threshold:
            eligible.append((score - threshold, -index, family))
    if not eligible:
        return "other_or_unclear"
    return max(eligible)[2]


def predict_alignment(record: Mapping[str, Any], candidate_scope: str) -> str:
    """Conservatively map scope plus strict two-order yes consensus."""

    if candidate_scope == "idiosyncratic":
        return "single_firm_only"
    if candidate_scope == "unclear":
        return "unclear"
    same = strict_consensus(record, "alignment_same")
    opposite = strict_consensus(record, "alignment_opposite")
    if same and not opposite:
        return "same_direction"
    if opposite and not same:
        return "opposite_direction"
    return "common_direction_unclear"


def applicable_component_validity(record: Mapping[str, Any]) -> tuple[bool, bool]:
    if not record["semantic_applicable"]:
        validity = record.get("validity", {})
        return (
            bool(validity.get("all_components_valid", True)),
            bool(validity.get("no_input_truncation", True)),
        )
    components = record["components"]
    all_valid = all(bool(components[name].get("valid")) for name in EXPECTED_COMPONENTS)
    no_truncation = not any(
        bool(components[name].get("input_truncated")) for name in EXPECTED_COMPONENTS
    )
    top_level = record.get("validity", {})
    if "all_components_valid" in top_level:
        all_valid = all_valid and bool(top_level["all_components_valid"])
    if "expected_component_set" in top_level:
        all_valid = all_valid and bool(top_level["expected_component_set"])
    if "no_input_truncation" in top_level:
        no_truncation = no_truncation and bool(top_level["no_input_truncation"])
    return all_valid, no_truncation


def materialize_candidate_row(
    raw: Mapping[str, Any],
    input_record: Mapping[str, Any],
    status_source: Mapping[str, Any],
    calibration: Mapping[str, Any],
    calibration_sha256: str,
) -> dict[str, Any]:
    validate_raw_score_record(raw)
    article_id = raw["article_id"]
    expected_ticker = input_record["target"]["ticker"]
    if raw["target_ticker"] != expected_ticker:
        raise ValueError(f"{article_id}: raw/input target ticker mismatch")
    if status_source.get("target_ticker") != expected_ticker:
        raise ValueError(f"{article_id}: status/input target ticker mismatch")
    if raw["semantic_applicable"] != raw["deterministic_features"].get(
        "semantic_applicable"
    ):
        raise ValueError(f"{article_id}: raw gate differs from deterministic features")

    all_components_valid, no_input_truncation = applicable_component_validity(raw)
    applicable = bool(raw["semantic_applicable"])
    if applicable:
        scope = predict_scope(
            raw,
            float(calibration["scope"]["firm_threshold"]),
            float(calibration["scope"]["common_threshold"]),
        )
        event = predict_event(raw, calibration["event"])
        status = status_source.get("labels", {}).get("information_status")
        alignment = predict_alignment(raw, scope)
        labels: dict[str, str | None] = {
            "shock_scope": scope,
            "event_family": event,
            "information_status": status,
            "directional_alignment": alignment,
        }
    else:
        labels = {field: None for field in coarse.COARSE_FIELDS}

    origins = {
        "shock_scope": "v0.5_development_calibrated_binary_composition",
        "event_family": "v0.5_development_calibrated_one_vs_rest",
        "information_status": "frozen_v0.4_hybrid",
        "directional_alignment": "v0.5_scope_plus_strict_consensus",
    }
    if not applicable:
        origins = {field: "deterministic_relevance_gate" for field in coarse.COARSE_FIELDS}

    resolved = (not applicable) or all(labels[field] is not None for field in coarse.COARSE_FIELDS)
    return {
        "row_number": input_record.get("row_number"),
        "article_id": article_id,
        "target_ticker": expected_ticker,
        "extractor": raw.get("extractor", "google/flan-t5-large"),
        "model_revision": raw.get("model_revision"),
        "prompt_version": PREDICTION_VERSION,
        "selector_version": SELECTOR_VERSION,
        "calibration_sha256": calibration_sha256,
        "semantic_applicable": applicable,
        "deterministic_features": raw["deterministic_features"],
        "labels": labels,
        "label_origins": origins,
        "component_decisions": (
            {
                "scope_firm": {
                    "score": component_score(raw, "scope_firm"),
                    "threshold": calibration["scope"]["firm_threshold"],
                },
                "scope_common": {
                    "score": component_score(raw, "scope_common"),
                    "threshold": calibration["scope"]["common_threshold"],
                },
                **{
                    f"event_{family}": {
                        "score": component_score(raw, f"event_{family}"),
                        "threshold": calibration["event"][family]["threshold"],
                    }
                    for family in EVENT_FAMILIES
                },
                "alignment_same": {
                    "strict_consensus_yes": strict_consensus(raw, "alignment_same")
                },
                "alignment_opposite": {
                    "strict_consensus_yes": strict_consensus(raw, "alignment_opposite")
                },
            }
            if applicable
            else {}
        ),
        "raw_score_record_sha256": sha256_object(raw),
        "validity": {
            "all_fields_resolved_when_applicable": resolved,
            "all_components_valid": all_components_valid,
            "no_input_truncation": no_input_truncation,
            "expected_component_set": (
                set(raw["components"]) == set(EXPECTED_COMPONENTS)
                if applicable
                else not raw["components"]
            ),
        },
    }


def end_to_end_labels(record: Mapping[str, Any]) -> dict[str, str]:
    labels = record.get("labels", {})
    return {
        field: labels.get(field) or FALLBACK_BY_FIELD[field]
        for field in coarse.COARSE_FIELDS
    }


def field_metrics(
    selected_ids: Sequence[str],
    truth_by_id: Mapping[str, Mapping[str, str]],
    prediction_by_id: Mapping[str, Mapping[str, Any]],
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in coarse.COARSE_FIELDS:
        truths = [truth_by_id[article_id][field] for article_id in selected_ids]
        guesses = [
            end_to_end_labels(prediction_by_id[article_id])[field]
            for article_id in selected_ids
        ]
        result[field] = legacy_eval.closed_label_metrics(
            truths, guesses, schema["closed_label_fields"][field]
        )
    return result


def aggregate_metrics(metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    return {
        "mean_accuracy": sum(metrics[field]["accuracy"] for field in coarse.COARSE_FIELDS)
        / len(coarse.COARSE_FIELDS),
        "mean_macro_f1": sum(metrics[field]["macro_f1"] for field in coarse.COARSE_FIELDS)
        / len(coarse.COARSE_FIELDS),
    }


def macro_f1_for_sample(
    indices: Sequence[int],
    truths: Sequence[str],
    guesses: Sequence[str],
    allowed: Sequence[str],
) -> float:
    observed: list[float] = []
    for label in allowed:
        support = sum(truths[index] == label for index in indices)
        if not support:
            continue
        tp = sum(
            truths[index] == label and guesses[index] == label for index in indices
        )
        fp = sum(
            truths[index] != label and guesses[index] == label for index in indices
        )
        fn = support - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        observed.append(
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return sum(observed) / len(observed) if observed else 0.0


def paired_bootstrap_positive_fraction(
    selected_ids: Sequence[str],
    truth_by_id: Mapping[str, Mapping[str, str]],
    candidate_by_id: Mapping[str, Mapping[str, Any]],
    baseline_by_id: Mapping[str, Mapping[str, Any]],
    schema: Mapping[str, Any],
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if not selected_ids:
        raise ValueError("Paired bootstrap requires at least one evaluation record")
    if samples <= 0:
        raise ValueError("Paired bootstrap sample count must be positive")
    truths_by_field = {
        field: [truth_by_id[article_id][field] for article_id in selected_ids]
        for field in coarse.COARSE_FIELDS
    }
    candidate_by_field = {
        field: [
            end_to_end_labels(candidate_by_id[article_id])[field]
            for article_id in selected_ids
        ]
        for field in coarse.COARSE_FIELDS
    }
    baseline_by_field = {
        field: [
            end_to_end_labels(baseline_by_id[article_id])[field]
            for article_id in selected_ids
        ]
        for field in coarse.COARSE_FIELDS
    }
    rng = random.Random(seed)
    positive = 0
    zero = 0
    deltas: list[float] = []
    count = len(selected_ids)
    for _ in range(samples):
        indices = [rng.randrange(count) for _ in range(count)]
        candidate_mean = 0.0
        baseline_mean = 0.0
        for field in coarse.COARSE_FIELDS:
            allowed = schema["closed_label_fields"][field]
            candidate_mean += macro_f1_for_sample(
                indices,
                truths_by_field[field],
                candidate_by_field[field],
                allowed,
            )
            baseline_mean += macro_f1_for_sample(
                indices,
                truths_by_field[field],
                baseline_by_field[field],
                allowed,
            )
        delta = (candidate_mean - baseline_mean) / len(coarse.COARSE_FIELDS)
        deltas.append(delta)
        if delta > 0.0:
            positive += 1
        elif delta == 0.0:
            zero += 1
    ordered = sorted(deltas)
    return {
        "samples": samples,
        "seed": seed,
        "metric": "paired delta in mean field macro_f1",
        "positive_delta_count": positive,
        "zero_delta_count": zero,
        "positive_delta_fraction": positive / samples,
        "mean_delta": sum(deltas) / samples,
        "percentile_2_5": ordered[int(0.025 * (samples - 1))],
        "percentile_97_5": ordered[int(0.975 * (samples - 1))],
    }


def promotion_decision(
    candidate_metrics: Mapping[str, Mapping[str, Any]],
    baseline_metrics: Mapping[str, Mapping[str, Any]],
    *,
    valid_nontruncated_rate: float,
    bootstrap_positive_fraction: float,
) -> dict[str, Any]:
    candidate_aggregate = aggregate_metrics(candidate_metrics)
    baseline_aggregate = aggregate_metrics(baseline_metrics)
    field_deltas = {
        field: {
            "accuracy_delta": candidate_metrics[field]["accuracy"]
            - baseline_metrics[field]["accuracy"],
            "macro_f1_delta": candidate_metrics[field]["macro_f1"]
            - baseline_metrics[field]["macro_f1"],
        }
        for field in coarse.COARSE_FIELDS
    }
    mean_macro_delta = (
        candidate_aggregate["mean_macro_f1"] - baseline_aggregate["mean_macro_f1"]
    )
    mean_accuracy_delta = (
        candidate_aggregate["mean_accuracy"] - baseline_aggregate["mean_accuracy"]
    )
    minimum_field_delta = min(
        field_deltas[field]["macro_f1_delta"] for field in coarse.COARSE_FIELDS
    )
    maximum_target_gain = max(
        field_deltas[field]["macro_f1_delta"] for field in TARGET_FIELDS
    )
    mixed_f1 = candidate_metrics["shock_scope"]["per_class"]["mixed"]["f1"]
    checks = {
        "mean_macro_f1_delta_at_least_0_02": mean_macro_delta
        >= PROMOTION_THRESHOLDS["mean_macro_f1_delta"],
        "mean_accuracy_delta_at_least_0_01": mean_accuracy_delta
        >= PROMOTION_THRESHOLDS["mean_accuracy_delta"],
        "no_field_macro_f1_drop_exceeds_0_02": minimum_field_delta
        >= -PROMOTION_THRESHOLDS["maximum_field_macro_f1_drop"],
        "mixed_scope_f1_is_nonzero": mixed_f1 > 0.0,
        "one_target_field_macro_f1_gain_at_least_0_05": maximum_target_gain
        >= PROMOTION_THRESHOLDS["target_field_macro_f1_gain"],
        "valid_and_nontruncated_rate_is_100_percent": valid_nontruncated_rate
        >= PROMOTION_THRESHOLDS["valid_nontruncated_rate"],
        "paired_bootstrap_positive_delta_fraction_at_least_0_95": (
            bootstrap_positive_fraction
            >= PROMOTION_THRESHOLDS["bootstrap_positive_delta_fraction"]
        ),
    }
    return {
        "promoted": all(checks.values()),
        "checks": checks,
        "thresholds": PROMOTION_THRESHOLDS,
        "candidate_aggregate": candidate_aggregate,
        "baseline_aggregate": baseline_aggregate,
        "mean_macro_f1_delta": mean_macro_delta,
        "mean_accuracy_delta": mean_accuracy_delta,
        "minimum_field_macro_f1_delta": minimum_field_delta,
        "maximum_target_field_macro_f1_delta": maximum_target_gain,
        "mixed_scope_f1": mixed_f1,
        "valid_nontruncated_rate": valid_nontruncated_rate,
        "bootstrap_positive_delta_fraction": bootstrap_positive_fraction,
        "field_deltas": field_deltas,
    }


def reconstruct_development_baseline(
    ids: Sequence[str],
    coarse_source_by_id: Mapping[str, Mapping[str, Any]],
    fine_source_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for article_id in ids:
        if article_id not in coarse_source_by_id or article_id not in fine_source_by_id:
            raise ValueError(f"Development baseline sources do not cover {article_id}")
        coarse_source = coarse_source_by_id[article_id]
        fine_source = fine_source_by_id[article_id]
        mapped = coarse.map_fine_labels(fine_source["labels"])
        applicable = bool(coarse_source.get("semantic_applicable"))
        labels = (
            {
                "shock_scope": mapped["shock_scope"],
                "event_family": coarse_source["labels"]["event_family"],
                "information_status": coarse_source["labels"]["information_status"],
                "directional_alignment": mapped["directional_alignment"],
            }
            if applicable
            else {field: None for field in coarse.COARSE_FIELDS}
        )
        result[article_id] = {
            "article_id": article_id,
            "target_ticker": coarse_source["target_ticker"],
            "semantic_applicable": applicable,
            "labels": labels,
        }
    return result


def source_description(path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    return {
        "path": str(path),
        "sha256": base.sha256_file(path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": base.sha256_file(manifest_path),
        "prompt_version": manifest.get("prompt_version"),
        "model_id": manifest.get("model_id"),
        "model_revision": manifest.get("model_revision"),
    }


def plain_source_description(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": base.sha256_file(path)}


def mapped_truth_for_ids(
    ids: Iterable[str],
    reference_by_id: Mapping[str, Mapping[str, Any]],
    inputs_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, str]], dict[str, bool]]:
    truth_by_id: dict[str, dict[str, str]] = {}
    applicable_by_id: dict[str, bool] = {}
    for article_id in ids:
        if article_id not in reference_by_id or article_id not in inputs_by_id:
            raise ValueError(f"Reference/inputs do not cover {article_id}")
        reference_record = reference_by_id[article_id]
        expected_ticker = inputs_by_id[article_id]["target"]["ticker"]
        if reference_record.get("target_ticker") != expected_ticker:
            raise ValueError(f"{article_id}: reference/input target mismatch")
        truth_by_id[article_id] = coarse.map_fine_labels(reference_record["labels"])
        applicable_by_id[article_id] = coarse.semantic_applicable(
            reference_record["labels"]
        )
    return truth_by_id, applicable_by_id


def validate_and_replay_raw_scores(
    records: Sequence[Mapping[str, Any]],
    inputs_by_id: Mapping[str, Mapping[str, Any]],
) -> None:
    for record in records:
        validate_raw_score_record(record)
        article_id = record["article_id"]
        if article_id not in inputs_by_id:
            raise ValueError(f"Inputs do not cover {article_id}")
        replay = coarse.deterministic_features(inputs_by_id[article_id])
        if record["deterministic_features"] != replay:
            raise ValueError(f"{article_id}: deterministic feature replay failed")
        if record["semantic_applicable"] != replay["semantic_applicable"]:
            raise ValueError(f"{article_id}: deterministic gate replay failed")


def validate_locked_calibration(
    path: Path,
    expected_file_sha256: str,
    schema_text: str,
) -> dict[str, Any]:
    expected = expected_file_sha256.strip().lower()
    if not expected or len(expected) != 64 or any(
        character not in "0123456789abcdef" for character in expected
    ):
        raise ValueError("--calibration-sha256 must be a 64-character hexadecimal digest")
    actual = base.sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"Calibration file hash mismatch: expected {expected}, observed {actual}"
        )
    calibration = json.loads(path.read_text(encoding="utf-8"))
    if calibration.get("selector_version") != SELECTOR_VERSION:
        raise ValueError("Calibration selector version is not supported")
    current_selector_hash = base.sha256_file(Path(__file__).resolve())
    if calibration.get("selector_source_sha256") != current_selector_hash:
        raise ValueError(
            "Current selector source differs from the development-locked selector"
        )
    stored_payload_hash = calibration.get("calibration_payload_sha256")
    payload = dict(calibration)
    payload.pop("calibration_payload_sha256", None)
    if stored_payload_hash != sha256_object(payload):
        raise ValueError("Calibration self/payload hash is invalid")
    if calibration.get("schema", {}).get("sha256") != hashlib.sha256(
        schema_text.encode("utf-8")
    ).hexdigest():
        raise ValueError("Calibration was fit against a different schema")
    development_ids = calibration.get("development_article_ids")
    if (
        not isinstance(development_ids, list)
        or not all(isinstance(article_id, str) and article_id for article_id in development_ids)
        or len(set(development_ids)) != len(development_ids)
        or calibration.get("development_record_count") != len(development_ids)
        or calibration.get("development_article_ids_sha256")
        != base.sha256_text("\n".join(development_ids))
    ):
        raise ValueError("Calibration development ID lock is invalid")
    sources = calibration.get("sources")
    if not isinstance(sources, dict) or calibration.get(
        "source_bundle_sha256"
    ) != sha256_object(sources):
        raise ValueError("Calibration development-source bundle hash is invalid")
    for name, description in sources.items():
        if not isinstance(description, dict):
            raise ValueError(f"Invalid calibration source description: {name}")
        required_hash_fields = ["sha256"]
        if "manifest_path" in description or "manifest_sha256" in description:
            required_hash_fields.append("manifest_sha256")
        for hash_field in required_hash_fields:
            digest = description.get(hash_field)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(
                    f"Calibration source {name}.{hash_field} is not a SHA-256 digest"
                )
    return calibration


def load_locked_development_score_manifest(
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay the locked development score file and manifest provenance."""

    description = calibration.get("sources", {}).get("development_scores")
    if not isinstance(description, Mapping):
        raise ValueError("Calibration lacks a locked development score source")
    score_path = Path(str(description.get("path", "")))
    manifest_path = Path(str(description.get("manifest_path", "")))
    if not score_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            "Locked development score file or manifest is no longer available"
        )
    score_hash = base.sha256_file(score_path)
    manifest_hash = base.sha256_file(manifest_path)
    if score_hash != description.get("sha256"):
        raise ValueError("Locked development score file hash has changed")
    if manifest_hash != description.get("manifest_sha256"):
        raise ValueError("Locked development score manifest hash has changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("Locked development score manifest is not complete")
    if manifest.get("output_sha256") != score_hash:
        raise ValueError(
            "Locked development score manifest no longer authenticates its output"
        )
    return manifest


def verify_extractor_identity(
    development_manifest: Mapping[str, Any],
    evaluation_manifest: Mapping[str, Any],
) -> None:
    """Require exact development/evaluation extractor identity where declared."""

    mismatches: list[str] = []
    for field in EXTRACTOR_IDENTITY_FIELDS:
        development_has = field in development_manifest
        evaluation_has = field in evaluation_manifest
        if development_has != evaluation_has or (
            development_has
            and development_manifest[field] != evaluation_manifest[field]
        ):
            mismatches.append(field)
    if mismatches:
        raise ValueError(
            "Evaluation scores differ from the locked development extractor on: "
            + ", ".join(mismatches)
        )


def calibrate_main(args: argparse.Namespace) -> int:
    if args.calibration_output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.calibration_output} exists; pass --overwrite")
    schema_text = args.schema.read_text(encoding="utf-8")
    schema = json.loads(schema_text)
    coarse.validate_schema(schema)
    source_paths = {
        "development_scores": args.development_scores,
        "development_coarse_v0_4": args.development_coarse_v0_4,
        "development_fine_v0_2": args.development_fine_v0_2,
    }
    manifests = {
        name: require_complete_manifest(path) for name, path in source_paths.items()
    }
    development_scores = base.read_jsonl(args.development_scores)
    reference_by_id = index_unique(base.read_jsonl(args.reference), "reference")
    inputs_by_id = index_unique(base.read_jsonl(args.inputs), "inputs")
    dev_coarse_by_id = index_unique(
        base.read_jsonl(args.development_coarse_v0_4),
        "development coarse source",
    )
    dev_fine_by_id = index_unique(
        base.read_jsonl(args.development_fine_v0_2), "development fine source"
    )
    dev_by_id = index_unique(development_scores, "development scores")
    validate_and_replay_raw_scores(development_scores, inputs_by_id)
    development_ids = sorted(
        dev_by_id, key=lambda article_id: inputs_by_id[article_id]["row_number"]
    )
    truth_by_id, reference_applicable = mapped_truth_for_ids(
        development_ids, reference_by_id, inputs_by_id
    )
    calibration_ids = [
        article_id
        for article_id in development_ids
        if reference_applicable[article_id] and dev_by_id[article_id]["semantic_applicable"]
    ]
    calibration_records = [dev_by_id[article_id] for article_id in calibration_ids]
    scope_calibration = fit_scope_thresholds(
        calibration_records,
        truth_by_id,
        schema["closed_label_fields"]["shock_scope"],
    )
    event_calibration = fit_event_thresholds(calibration_records, truth_by_id)
    sources = {
        name: source_description(source_paths[name], manifests[name])
        for name in source_paths
    }
    sources["reference"] = plain_source_description(args.reference)
    sources["inputs"] = plain_source_description(args.inputs)
    calibration: dict[str, Any] = {
        "status": "calibrated_on_development_only",
        "selector_version": SELECTOR_VERSION,
        "prediction_version": PREDICTION_VERSION,
        "warning": (
            "The existing 300-record GPT-5.6 silver benchmark has already informed "
            "protocol design; this is an engineering freeze decision, not a clean final claim."
        ),
        "decision_rule": {
            "threshold_operator": "score >= threshold",
            "scope_composition": {
                "firm_only": "idiosyncratic",
                "common_only": "common",
                "firm_and_common": "mixed",
                "neither": "unclear",
            },
            "event_resolution": (
                "eligible if score >= family threshold; choose largest score-threshold "
                "margin, then fixed family order; none => other_or_unclear"
            ),
            "alignment": (
                "idiosyncratic=>single_firm_only; unclear=>unclear; for common/mixed, "
                "exactly one strict same/opposite consensus wins; otherwise "
                "common_direction_unclear"
            ),
            "information_status": "copied unchanged from frozen v0.4 source",
        },
        "development_record_count": len(development_ids),
        "development_article_ids": development_ids,
        "development_reference_and_gate_applicable_count": len(calibration_ids),
        "development_article_ids_sha256": base.sha256_text("\n".join(development_ids)),
        "calibration_article_ids_sha256": base.sha256_text("\n".join(calibration_ids)),
        "scope": scope_calibration,
        "event": event_calibration,
        "bootstrap_policy": {
            "seed": BOOTSTRAP_SEED,
            "samples": BOOTSTRAP_SAMPLES,
            "positive_delta_definition": "candidate mean macro_f1 - baseline mean macro_f1 > 0",
        },
        "promotion_thresholds": PROMOTION_THRESHOLDS,
        "schema": {
            "path": str(args.schema),
            "sha256": hashlib.sha256(schema_text.encode("utf-8")).hexdigest(),
            "name": schema["schema_name"],
            "version": schema["schema_version"],
        },
        "sources": sources,
        "source_bundle_sha256": sha256_object(sources),
        "selector_source_sha256": base.sha256_file(Path(__file__).resolve()),
    }

    # Development metrics are recorded before the calibration is hash-locked.
    # The placeholder is provenance-only and never used to validate development
    # predictions.
    dev_baseline_by_id = reconstruct_development_baseline(
        development_ids, dev_coarse_by_id, dev_fine_by_id
    )
    dev_candidate_rows = [
        materialize_candidate_row(
            dev_by_id[article_id],
            inputs_by_id[article_id],
            dev_coarse_by_id[article_id],
            calibration,
            "development-calibration-not-yet-hash-locked",
        )
        for article_id in development_ids
    ]
    dev_candidate_by_id = index_unique(dev_candidate_rows, "development candidate")
    relevant_ids = [
        article_id for article_id in development_ids if reference_applicable[article_id]
    ]
    candidate_metrics = field_metrics(
        relevant_ids, truth_by_id, dev_candidate_by_id, schema
    )
    baseline_metrics = field_metrics(
        relevant_ids, truth_by_id, dev_baseline_by_id, schema
    )
    calibration["development_diagnostics"] = {
        "reference_relevant_count": len(relevant_ids),
        "candidate_field_metrics": candidate_metrics,
        "baseline_field_metrics": baseline_metrics,
        "candidate_aggregate": aggregate_metrics(candidate_metrics),
        "baseline_aggregate": aggregate_metrics(baseline_metrics),
    }
    calibration["calibration_payload_sha256"] = sha256_object(calibration)
    write_json(args.calibration_output, calibration, overwrite=args.overwrite)
    file_hash = base.sha256_file(args.calibration_output)
    print(f"Wrote development-only locked calibration to {args.calibration_output}")
    print(f"CALIBRATION_SHA256={file_hash}")
    return 0


def evaluate_main(args: argparse.Namespace) -> int:
    output_paths = (
        args.predictions_output,
        args.predictions_output.with_suffix(args.predictions_output.suffix + ".manifest.json"),
        args.report_output,
    )
    if not args.overwrite:
        existing = [str(path) for path in output_paths if path.exists()]
        if existing:
            raise FileExistsError("Output files already exist: " + ", ".join(existing))
    schema_text = args.schema.read_text(encoding="utf-8")
    schema = json.loads(schema_text)
    coarse.validate_schema(schema)
    calibration = validate_locked_calibration(
        args.calibration, args.calibration_sha256, schema_text
    )
    calibration_file_sha256 = base.sha256_file(args.calibration)

    source_paths = {
        "evaluation_scores": args.evaluation_scores,
        "evaluation_v0_4_hybrid": args.evaluation_v0_4_hybrid,
    }
    manifests = {
        name: require_complete_manifest(path) for name, path in source_paths.items()
    }
    eval_score_manifest = manifests["evaluation_scores"]
    locked_development_manifest = load_locked_development_score_manifest(calibration)
    verify_extractor_identity(locked_development_manifest, eval_score_manifest)
    for name, path in (("reference", args.reference), ("inputs", args.inputs)):
        if base.sha256_file(path) != calibration["sources"][name]["sha256"]:
            raise ValueError(
                f"Evaluation {name} file differs from the development-locked source"
            )

    evaluation_scores = base.read_jsonl(args.evaluation_scores)
    evaluation_baseline = base.read_jsonl(args.evaluation_v0_4_hybrid)
    reference_by_id = index_unique(base.read_jsonl(args.reference), "reference")
    inputs_by_id = index_unique(base.read_jsonl(args.inputs), "inputs")
    eval_by_id = index_unique(evaluation_scores, "evaluation scores")
    eval_baseline_by_id = index_unique(evaluation_baseline, "evaluation baseline")
    if set(eval_by_id) != set(eval_baseline_by_id):
        raise ValueError("Evaluation raw scores and v0.4 hybrid must have identical IDs")
    validate_and_replay_raw_scores(evaluation_scores, inputs_by_id)
    evaluation_ids = sorted(
        eval_by_id, key=lambda article_id: inputs_by_id[article_id]["row_number"]
    )
    development_id_set = set(calibration.get("development_article_ids", []))
    if len(development_id_set) != calibration.get("development_record_count"):
        raise ValueError("Locked development ID list is absent or internally inconsistent")
    overlap = development_id_set & set(evaluation_ids)
    if overlap:
        raise ValueError(
            f"Evaluation overlaps locked development set by {len(overlap)} records"
        )
    truth_by_id, reference_applicable = mapped_truth_for_ids(
        evaluation_ids, reference_by_id, inputs_by_id
    )
    candidate_rows = [
        materialize_candidate_row(
            eval_by_id[article_id],
            inputs_by_id[article_id],
            eval_baseline_by_id[article_id],
            calibration,
            calibration_file_sha256,
        )
        for article_id in evaluation_ids
    ]
    candidate_by_id = index_unique(candidate_rows, "evaluation candidate")
    relevant_ids = [
        article_id for article_id in evaluation_ids if reference_applicable[article_id]
    ]
    candidate_metrics = field_metrics(
        relevant_ids, truth_by_id, candidate_by_id, schema
    )
    baseline_metrics = field_metrics(
        relevant_ids, truth_by_id, eval_baseline_by_id, schema
    )
    valid_rows = sum(
        all(
            (
                row["validity"]["all_fields_resolved_when_applicable"],
                row["validity"]["all_components_valid"],
                row["validity"]["no_input_truncation"],
                row["validity"]["expected_component_set"],
            )
        )
        for row in candidate_rows
    )
    valid_nontruncated_rate = valid_rows / len(candidate_rows)
    bootstrap_policy = calibration["bootstrap_policy"]
    if (
        bootstrap_policy.get("seed") != BOOTSTRAP_SEED
        or bootstrap_policy.get("samples") != BOOTSTRAP_SAMPLES
    ):
        raise ValueError("Locked calibration does not use the required bootstrap policy")
    bootstrap = paired_bootstrap_positive_fraction(
        relevant_ids,
        truth_by_id,
        candidate_by_id,
        eval_baseline_by_id,
        schema,
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    decision = promotion_decision(
        candidate_metrics,
        baseline_metrics,
        valid_nontruncated_rate=valid_nontruncated_rate,
        bootstrap_positive_fraction=bootstrap["positive_delta_fraction"],
    )

    write_jsonl(args.predictions_output, candidate_rows, overwrite=args.overwrite)
    prediction_manifest = {
        "status": "complete",
        "selector_version": SELECTOR_VERSION,
        "prompt_version": PREDICTION_VERSION,
        "promotion_decision": "promoted" if decision["promoted"] else "rejected",
        "calibration_path": str(args.calibration),
        "calibration_sha256": calibration_file_sha256,
        "calibration_payload_sha256": calibration["calibration_payload_sha256"],
        "schema_sha256": hashlib.sha256(schema_text.encode("utf-8")).hexdigest(),
        "model_id": eval_score_manifest.get("model_id"),
        "model_revision": eval_score_manifest.get("model_revision"),
        "source_files": {
            name: source_description(source_paths[name], manifests[name])
            for name in source_paths
        },
        "total_prediction_count": len(candidate_rows),
        "semantic_applicable_count": sum(
            bool(row["semantic_applicable"]) for row in candidate_rows
        ),
        "valid_nontruncated_rate": valid_nontruncated_rate,
        "selected_article_ids_sha256": base.sha256_text("\n".join(evaluation_ids)),
        "selector_source_sha256": base.sha256_file(Path(__file__).resolve()),
        "output_sha256": base.sha256_file(args.predictions_output),
    }
    write_json(
        args.predictions_output.with_suffix(
            args.predictions_output.suffix + ".manifest.json"
        ),
        prediction_manifest,
        overwrite=args.overwrite,
    )
    report = {
        "title": "FLAN-T5-Large v0.5 final-candidate promotion evaluation",
        "reference_type": "deterministically coarsened GPT-5.6 Sol silver annotations",
        "warning": calibration["warning"],
        "selector_version": SELECTOR_VERSION,
        "candidate_prediction_version": PREDICTION_VERSION,
        "calibration_lock": {
            "path": str(args.calibration),
            "sha256": calibration_file_sha256,
            "payload_sha256": calibration["calibration_payload_sha256"],
            "development_article_ids_sha256": calibration[
                "development_article_ids_sha256"
            ],
            "source_bundle_sha256": calibration["source_bundle_sha256"],
        },
        "evaluation": {
            "record_count": len(evaluation_ids),
            "reference_relevant_count": len(relevant_ids),
            "candidate_field_metrics": candidate_metrics,
            "baseline_v0_4_field_metrics": baseline_metrics,
            "paired_bootstrap": bootstrap,
            "promotion": decision,
        },
        "artifacts": {
            "predictions_path": str(args.predictions_output),
            "predictions_sha256": base.sha256_file(args.predictions_output),
            "prediction_manifest_path": str(
                args.predictions_output.with_suffix(
                    args.predictions_output.suffix + ".manifest.json"
                )
            ),
        },
    }
    write_json(args.report_output, report, overwrite=args.overwrite)
    outcome = "PROMOTED" if decision["promoted"] else "REJECTED"
    print(
        f"{outcome}: mean macro-F1 delta={decision['mean_macro_f1_delta']:.6f}; "
        f"mean accuracy delta={decision['mean_accuracy_delta']:.6f}; "
        f"bootstrap positive fraction={bootstrap['positive_delta_fraction']:.4f}"
    )
    print(f"Wrote evaluation predictions to {args.predictions_output}")
    print(f"Wrote promotion report to {args.report_output}")
    return 0


def main() -> int:
    args = parse_args()
    if args.mode == "calibrate":
        return calibrate_main(args)
    if args.mode == "evaluate":
        return evaluate_main(args)
    raise AssertionError(f"Unhandled mode: {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())
