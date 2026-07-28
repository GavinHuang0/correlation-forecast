"""Analyze frozen FLAN-T5-XL behavior by Massive description length.

There are intentionally no reference-label metrics in this report. It
summarizes operational validity, calibrated label behavior, score margins,
and paired native-description versus headline-only sensitivity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA = ROOT / "config" / "news_feature_schema_coarse.json"
REPORT_VERSION = "flan-length-stratified-analysis-v1"
STRATA = (
    "headline_only",
    "description_001_149",
    "description_150_299",
    "description_300_599",
    "description_600_plus",
)
ABSTENTIONS = {
    "shock_scope": {"unclear", None},
    "event_family": {"other_or_unclear", None},
    "information_status": {"unclear", None},
    "directional_alignment": {
        "unclear",
        "common_direction_unclear",
        None,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--benchmark-manifest", required=True, type=Path)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(value)
    return rows


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def model_scores(pass_record: Mapping[str, Any]) -> dict[str, float] | None:
    calibration = pass_record.get("calibration")
    if isinstance(calibration, Mapping):
        adjusted = calibration.get("adjusted_scores")
        if isinstance(adjusted, Mapping) and adjusted:
            return {str(key): float(value) for key, value in adjusted.items()}
    averaged = pass_record.get("order_averaged_mean_log_probabilities")
    if isinstance(averaged, Mapping) and averaged:
        return {str(key): float(value) for key, value in averaged.items()}
    return None


def top_margin(pass_record: Mapping[str, Any]) -> float | None:
    scores = model_scores(pass_record)
    if not scores:
        return None
    ordered = sorted(scores.values(), reverse=True)
    return ordered[0] - ordered[1] if len(ordered) > 1 else 0.0


def precalibration_label(
    record: Mapping[str, Any], field: str
) -> Any:
    pass_record = record.get("passes", {}).get(field, {})
    if "precalibration_value" in pass_record:
        return pass_record["precalibration_value"]
    return record.get("labels", {}).get(field)


def normalized_entropy(counts: Counter[Any]) -> float | None:
    total = sum(counts.values())
    observed = [count for count in counts.values() if count]
    if not total or len(observed) <= 1:
        return 0.0 if total else None
    entropy = -sum(
        (count / total) * math.log(count / total) for count in observed
    )
    return entropy / math.log(len(observed))


def field_summary(
    records: Sequence[Mapping[str, Any]], field: str
) -> dict[str, Any]:
    labels = Counter(record["labels"].get(field) for record in records)
    raw_labels = Counter(
        precalibration_label(record, field) for record in records
    )
    margins: list[float] = []
    order_agreements: list[bool] = []
    model_count = 0
    for record in records:
        pass_record = record.get("passes", {}).get(field, {})
        if pass_record.get("origin") in {"model", "model_calibrated"}:
            model_count += 1
            margin = top_margin(pass_record)
            if margin is not None:
                margins.append(margin)
            canonical = pass_record.get("canonical_prediction")
            reversed_value = pass_record.get("reversed_prediction")
            if canonical is not None and reversed_value is not None:
                order_agreements.append(canonical == reversed_value)
    abstained = sum(
        label in ABSTENTIONS[field] for label in labels.elements()
    )
    most_common = labels.most_common(1)
    raw_abstained = sum(
        label in ABSTENTIONS[field] for label in raw_labels.elements()
    )
    calibration_overrides = sum(
        precalibration_label(record, field)
        != record["labels"].get(field)
        for record in records
    )
    return {
        "record_count": len(records),
        "model_evaluated_count": model_count,
        "label_counts": {
            "__not_applicable__" if key is None else str(key): value
            for key, value in sorted(
                labels.items(), key=lambda item: str(item[0])
            )
        },
        "abstention_count": abstained,
        "abstention_rate": safe_divide(abstained, len(records)),
        "precalibration_label_counts": {
            "__not_applicable__" if key is None else str(key): value
            for key, value in sorted(
                raw_labels.items(), key=lambda item: str(item[0])
            )
        },
        "precalibration_abstention_count": raw_abstained,
        "precalibration_abstention_rate": safe_divide(
            raw_abstained, len(records)
        ),
        "calibration_override_count": calibration_overrides,
        "calibration_override_rate": safe_divide(
            calibration_overrides, len(records)
        ),
        "largest_label_share": (
            safe_divide(most_common[0][1], len(records))
            if most_common
            else None
        ),
        "observed_label_entropy_normalized": normalized_entropy(labels),
        "calibrated_margin_count": len(margins),
        "calibrated_margin_mean": mean(margins),
        "calibrated_margin_median": median(margins),
        "raw_order_agreement_count": len(order_agreements),
        "raw_order_agreement_rate": (
            safe_divide(sum(order_agreements), len(order_agreements))
            if order_agreements
            else None
        ),
    }


def group_summary(
    records: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> dict[str, Any]:
    semantically_applicable = sum(
        bool(record.get("semantic_applicable")) for record in records
    )
    no_truncation = sum(
        bool(record.get("validity", {}).get("no_input_truncation"))
        for record in records
    )
    resolved = sum(
        (
            not record.get("semantic_applicable")
            or bool(
                record.get("validity", {}).get(
                    "all_fields_resolved_when_applicable"
                )
            )
        )
        for record in records
    )
    gate_routes = Counter(
        record.get("deterministic_features", {}).get("gate_route")
        for record in records
    )
    return {
        "record_count": len(records),
        "semantic_applicable_count": semantically_applicable,
        "semantic_applicable_rate": safe_divide(
            semantically_applicable, len(records)
        ),
        "no_input_truncation_count": no_truncation,
        "no_input_truncation_rate": safe_divide(no_truncation, len(records)),
        "operationally_resolved_count": resolved,
        "operationally_resolved_rate": safe_divide(resolved, len(records)),
        "gate_route_counts": {
            "__missing__" if key is None else str(key): value
            for key, value in sorted(
                gate_routes.items(), key=lambda item: str(item[0])
            )
        },
        "fields": {
            field: field_summary(records, field) for field in fields
        },
    }


def pair_summary(
    pairs: Sequence[
        tuple[Mapping[str, Any], Mapping[str, Any]]
    ],
    fields: Sequence[str],
) -> dict[str, Any]:
    exact = sum(
        all(native["labels"].get(field) == headline["labels"].get(field)
            for field in fields)
        for native, headline in pairs
    )
    result: dict[str, Any] = {
        "pair_count": len(pairs),
        "all_field_exact_agreement_count": exact,
        "all_field_exact_agreement_rate": safe_divide(exact, len(pairs)),
        "fields": {},
    }
    for field in fields:
        agreement = 0
        native_abstention = 0
        headline_abstention = 0
        raw_agreement = 0
        raw_native_abstention = 0
        raw_headline_abstention = 0
        changed_to_abstention = 0
        changed_from_abstention = 0
        native_margins: list[float] = []
        headline_margins: list[float] = []
        margin_deltas: list[float] = []
        for native, headline in pairs:
            native_label = native["labels"].get(field)
            headline_label = headline["labels"].get(field)
            agreement += native_label == headline_label
            native_is_abstention = native_label in ABSTENTIONS[field]
            headline_is_abstention = headline_label in ABSTENTIONS[field]
            native_raw = precalibration_label(native, field)
            headline_raw = precalibration_label(headline, field)
            raw_agreement += native_raw == headline_raw
            raw_native_abstention += native_raw in ABSTENTIONS[field]
            raw_headline_abstention += headline_raw in ABSTENTIONS[field]
            native_abstention += native_is_abstention
            headline_abstention += headline_is_abstention
            changed_to_abstention += (
                not native_is_abstention and headline_is_abstention
            )
            changed_from_abstention += (
                native_is_abstention and not headline_is_abstention
            )
            native_margin = top_margin(
                native.get("passes", {}).get(field, {})
            )
            headline_margin = top_margin(
                headline.get("passes", {}).get(field, {})
            )
            if native_margin is not None:
                native_margins.append(native_margin)
            if headline_margin is not None:
                headline_margins.append(headline_margin)
            if native_margin is not None and headline_margin is not None:
                margin_deltas.append(native_margin - headline_margin)
        result["fields"][field] = {
            "label_agreement_count": agreement,
            "label_agreement_rate": safe_divide(agreement, len(pairs)),
            "precalibration_label_agreement_rate": safe_divide(
                raw_agreement, len(pairs)
            ),
            "native_abstention_rate": safe_divide(
                native_abstention, len(pairs)
            ),
            "headline_only_abstention_rate": safe_divide(
                headline_abstention, len(pairs)
            ),
            "precalibration_native_abstention_rate": safe_divide(
                raw_native_abstention, len(pairs)
            ),
            "precalibration_headline_only_abstention_rate": safe_divide(
                raw_headline_abstention, len(pairs)
            ),
            "changed_to_abstention_when_description_removed": (
                changed_to_abstention
            ),
            "changed_from_abstention_when_description_removed": (
                changed_from_abstention
            ),
            "native_margin_mean": mean(native_margins),
            "headline_only_margin_mean": mean(headline_margins),
            "native_minus_headline_margin_mean": mean(margin_deltas),
        }
    return result


def main() -> int:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite")
    inputs = read_jsonl(args.inputs)
    predictions = read_jsonl(args.predictions)
    benchmark = json.loads(args.benchmark_manifest.read_text(encoding="utf-8"))
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    fields = tuple(schema["flan_core_fields"])

    input_by_id = {record["article_id"]: record for record in inputs}
    prediction_by_id = {
        record["article_id"]: record for record in predictions
    }
    if len(input_by_id) != len(inputs) or len(prediction_by_id) != len(
        predictions
    ):
        raise ValueError("Inputs or predictions contain duplicate article IDs")
    if set(input_by_id) != set(prediction_by_id):
        missing = sorted(set(input_by_id) - set(prediction_by_id))
        extra = sorted(set(prediction_by_id) - set(input_by_id))
        raise ValueError(
            f"Input/prediction IDs differ: missing={len(missing)}, extra={len(extra)}"
        )
    expected_input_hash = benchmark["files"]["model_inputs"]["sha256"]
    if sha256_file(args.inputs) != expected_input_hash:
        raise ValueError("Input hash does not match the benchmark manifest")

    native_by_stratum: dict[str, list[dict[str, Any]]] = {
        stratum: [] for stratum in STRATA
    }
    native_by_parent: dict[str, dict[str, Any]] = {}
    headline_by_parent: dict[str, dict[str, Any]] = {}
    for article_id, input_record in input_by_id.items():
        prediction = prediction_by_id[article_id]
        parent_id = input_record["parent_article_id"]
        variant = input_record["text_variant"]
        stratum = input_record["native_length_stratum"]
        if variant == "native_description":
            native_by_stratum[stratum].append(prediction)
            native_by_parent[parent_id] = prediction
        elif variant == "headline_only_ablation":
            headline_by_parent[parent_id] = prediction
        else:
            raise ValueError(f"Unexpected text variant {variant!r}")

    paired_by_stratum: dict[
        str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]
    ] = {stratum: [] for stratum in STRATA}
    for parent_id, headline_prediction in headline_by_parent.items():
        native_prediction = native_by_parent[parent_id]
        native_input = input_by_id[native_prediction["article_id"]]
        paired_by_stratum[native_input["native_length_stratum"]].append(
            (native_prediction, headline_prediction)
        )

    all_records = [prediction_by_id[record["article_id"]] for record in inputs]
    all_native = [
        prediction
        for rows in native_by_stratum.values()
        for prediction in rows
    ]
    all_pairs = [
        pair for rows in paired_by_stratum.values() for pair in rows
    ]
    operational_pass = all(
        (
            not record.get("semantic_applicable")
            or record.get("validity", {}).get(
                "all_fields_resolved_when_applicable"
            )
        )
        and record.get("validity", {}).get("no_input_truncation")
        for record in all_records
    )
    report = {
        "report_version": REPORT_VERSION,
        "status": (
            "operational_validation_passed"
            if operational_pass
            else "operational_validation_failed"
        ),
        "extractor": {
            "id": "flan-t5-xl-v1.1",
            "model": "google/flan-t5-xl",
            "result_type": (
                "unlabeled operational, distributional, and paired-ablation "
                "robustness analysis"
            ),
        },
        "claim_limit": (
            "No human or independent model reference labels were used. Label "
            "agreement between text views is stability, not accuracy."
        ),
        "benchmark": {
            "manifest_path": str(args.benchmark_manifest),
            "manifest_sha256": sha256_file(args.benchmark_manifest),
            "parent_article_count": len(all_native),
            "model_input_count": len(all_records),
            "paired_article_count": len(all_pairs),
            "coverage_by_stratum": benchmark.get(
                "coverage_by_stratum", {}
            ),
        },
        "interpretation_guardrails": [
            (
                "Provider/source and calendar-period composition differ by "
                "native length stratum; direct cross-stratum label differences "
                "are descriptive, not causal length effects."
            ),
            (
                "Within-article native-versus-headline-only comparisons remove "
                "article, source, target, and date confounding, but they measure "
                "prediction sensitivity rather than accuracy."
            ),
            (
                "Candidate-score margins are score separation, not calibrated "
                "probabilities or empirical correctness confidence."
            ),
        ],
        "operational_acceptance": {
            "criterion": (
                "Every model input is untruncated and every semantically "
                "applicable input resolves all frozen-schema fields."
            ),
            "passed": operational_pass,
        },
        "all_model_inputs": group_summary(all_records, fields),
        "all_native_views": group_summary(all_native, fields),
        "native_views_by_length_stratum": {
            stratum: group_summary(native_by_stratum[stratum], fields)
            for stratum in STRATA
        },
        "paired_native_vs_headline_only": {
            "overall": pair_summary(all_pairs, fields),
            "by_native_length_stratum": {
                stratum: pair_summary(paired_by_stratum[stratum], fields)
                for stratum in STRATA
                if paired_by_stratum[stratum]
            },
        },
        "requested_focus": {
            "headline_only_native": group_summary(
                native_by_stratum["headline_only"], fields
            ),
            "sub_150_native": group_summary(
                native_by_stratum["description_001_149"], fields
            ),
            "sub_150_native_vs_headline_only": pair_summary(
                paired_by_stratum["description_001_149"], fields
            ),
        },
        "files": {
            "inputs": {
                "path": str(args.inputs),
                "sha256": sha256_file(args.inputs),
            },
            "predictions": {
                "path": str(args.predictions),
                "sha256": sha256_file(args.predictions),
            },
            "schema": {
                "path": str(args.schema),
                "sha256": sha256_file(args.schema),
            },
            "analyzer": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "parent_articles": len(all_native),
                "model_inputs": len(all_records),
                "paired_articles": len(all_pairs),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
