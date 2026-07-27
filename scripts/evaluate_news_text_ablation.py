"""Evaluate article-text ablations against the coarse GPT silver reference.

This utility consumes already-produced prediction JSONL.  It never imports or
runs a model.  Full-text chunk predictions are reduced to one article-level
label per field using a declared score reducer when present, otherwise
deterministic plurality with schema-order tie breaking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import coarse_news_features as coarse


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA = REPOSITORY_ROOT / "config" / "news_feature_schema_coarse.json"
DEFAULT_BENCHMARK_MANIFEST = (
    REPOSITORY_ROOT
    / "data"
    / "benchmarks"
    / "news_text_ablation_300"
    / "v1"
    / "manifest.json"
)
VARIANT_ORDER = (
    "massive_description",
    "alpha_summary",
    "fulltext_evidence_chunks",
)
INVALID = "__invalid__"
REPORT_VERSION = "news-text-ablation-evaluation-v1.1"
CHUNK_AGGREGATIONS = (
    "mean_score",
    "max_score",
    "best_margin",
    "plurality",
)
ASSIGNMENT_FILTERS = (
    "direct_target_tag",
    "peer_only_tag",
    "expanded_sector_or_market_only",
)
PRIMARY_SELECTION_FIELDS = ("shock_scope", "directional_alignment")
BOOTSTRAP_REPLICATES = 2000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate cached text-variant predictions and evaluate agreement "
            "with the coarse GPT silver reference. No inference is run."
        )
    )
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument(
        "--predictions",
        action="append",
        required=True,
        metavar="VARIANT=PATH",
        help=(
            "Prediction JSONL for massive_description, alpha_summary, or "
            "fulltext_evidence_chunks; repeat for each variant."
        ),
    )
    parser.add_argument(
        "--benchmark-manifest",
        type=Path,
        default=DEFAULT_BENCHMARK_MANIFEST,
    )
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument(
        "--split",
        choices=("all", "development", "evaluation"),
        default="evaluation",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Evaluate available rows instead of requiring manifest-complete predictions.",
    )
    parser.add_argument(
        "--chunk-aggregation",
        choices=CHUNK_AGGREGATIONS,
        default="mean_score",
        help=(
            "Article reducer for full-text chunks. Select this on the "
            "development split before reading evaluation results."
        ),
    )
    parser.add_argument(
        "--aggregation-lock",
        type=Path,
        help=(
            "Development-selected aggregation lock. When supplied, evaluation "
            "is restricted to the holdout split and all locked artifact hashes "
            "are revalidated before metrics are computed."
        ),
    )
    parser.add_argument(
        "--reference-filter",
        choices=("all", "non_abstained"),
        default="all",
        help="Optional sensitivity filter applied after the benchmark split.",
    )
    parser.add_argument(
        "--source-filter",
        help=(
            "Optional exact publisher/source stratum from benchmark manifest "
            "v1.1."
        ),
    )
    parser.add_argument(
        "--assignment-filter",
        choices=ASSIGNMENT_FILTERS,
        help=(
            "Optional assignment-provenance stratum from benchmark manifest "
            "v1.1."
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: prediction must be an object")
            records.append(record)
    return records


def parse_prediction_specs(values: Sequence[str]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"Prediction specification must be VARIANT=PATH, got {value!r}"
            )
        variant, raw_path = value.split("=", 1)
        variant = variant.strip()
        if variant not in VARIANT_ORDER:
            raise ValueError(
                f"Unknown variant {variant!r}; expected one of {VARIANT_ORDER!r}"
            )
        if variant in paths:
            raise ValueError(f"Duplicate prediction variant {variant!r}")
        if not raw_path.strip():
            raise ValueError(f"Prediction path is empty for variant {variant!r}")
        paths[variant] = Path(raw_path)
    return paths


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def closed_label_metrics(
    truths: Sequence[str],
    guesses: Sequence[str | None],
    allowed: Sequence[str],
) -> dict[str, Any]:
    if len(truths) != len(guesses):
        raise ValueError("Truth and prediction lengths differ")
    allowed_list = list(allowed)
    if any(truth not in allowed_list for truth in truths):
        raise ValueError("Reference contains a label outside the evaluation schema")
    normalized = [guess if guess in allowed_list else INVALID for guess in guesses]
    total = len(truths)
    correct = sum(truth == guess for truth, guess in zip(truths, normalized))
    reference_counts = Counter(truths)
    prediction_counts = Counter(normalized)
    accuracy = safe_divide(correct, total)
    expected_agreement = sum(
        safe_divide(reference_counts[label], total)
        * safe_divide(prediction_counts[label], total)
        for label in allowed_list
    )
    kappa_denominator = 1 - expected_agreement
    cohens_kappa = (
        safe_divide(accuracy - expected_agreement, kappa_denominator)
        if kappa_denominator
        else (1.0 if accuracy == 1.0 else 0.0)
    )
    per_class: dict[str, Any] = {}
    observed_f1: list[float] = []
    for label in allowed_list:
        tp = sum(
            truth == label and guess == label
            for truth, guess in zip(truths, normalized)
        )
        fp = sum(
            truth != label and guess == label
            for truth, guess in zip(truths, normalized)
        )
        fn = sum(
            truth == label and guess != label
            for truth, guess in zip(truths, normalized)
        )
        support = sum(truth == label for truth in truths)
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        if support:
            observed_f1.append(f1)
    return {
        "count": total,
        "accuracy": accuracy,
        "majority_class_baseline_accuracy": safe_divide(
            max(reference_counts.values(), default=0), total
        ),
        "cohens_kappa": cohens_kappa,
        "macro_f1": safe_divide(sum(observed_f1), len(observed_f1)),
        "macro_f1_supported_class_count": len(observed_f1),
        "schema_class_count": len(allowed_list),
        "invalid_prediction_count": sum(
            guess == INVALID for guess in normalized
        ),
        "reference_distribution": dict(sorted(reference_counts.items())),
        "prediction_distribution": dict(sorted(prediction_counts.items())),
        "per_class": per_class,
    }


def _coarse_reference_labels(record: Mapping[str, Any]) -> dict[str, str]:
    labels = record.get("labels")
    if not isinstance(labels, Mapping):
        raise ValueError(f"Reference {record.get('article_id')!r} has no labels object")
    if all(field in labels for field in coarse.COARSE_FIELDS):
        result = {field: str(labels[field]) for field in coarse.COARSE_FIELDS}
        coarse.validate_coarse_labels(result)
        return result
    return coarse.map_fine_labels(labels)


def reference_by_id(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row_number, record in enumerate(records, start=1):
        article_id = record.get("article_id")
        if not isinstance(article_id, str) or not article_id:
            raise ValueError(f"Reference row {row_number} has no article_id")
        if article_id in result:
            raise ValueError(f"Duplicate reference article_id {article_id!r}")
        result[article_id] = {
            "labels": _coarse_reference_labels(record),
            "target_ticker": record.get("target_ticker"),
            "abstained": bool(record.get("abstained", False)),
        }
    return result


def _label_for_field(record: Mapping[str, Any], field: str) -> str | None:
    labels = record.get("labels")
    if isinstance(labels, Mapping):
        value = labels.get(field)
        if isinstance(value, str):
            return value
    passes = record.get("passes")
    if isinstance(passes, Mapping) and isinstance(passes.get(field), Mapping):
        value = passes[field].get("value")
        if isinstance(value, str):
            return value
    value = record.get(field)
    return value if isinstance(value, str) else None


def _complete_scores(
    value: Any, allowed: Sequence[str]
) -> dict[str, float] | None:
    if not isinstance(value, Mapping) or not all(label in value for label in allowed):
        return None
    scores: dict[str, float] = {}
    for label in allowed:
        raw = value[label]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            return None
        score = float(raw)
        if not math.isfinite(score):
            return None
        scores[label] = score
    return scores


def scores_for_field(
    record: Mapping[str, Any], field: str, allowed: Sequence[str]
) -> dict[str, float] | None:
    # The active FLAN-T5-XL v1.1 output retains the raw score maps but changes
    # its selected label with a frozen development-selected calibration.  Use
    # the calibrated score map first so aggregation reproduces the active
    # extractor instead of silently reverting to its raw v1.0 decision.
    passes = record.get("passes")
    if isinstance(passes, Mapping) and isinstance(passes.get(field), Mapping):
        pass_record = passes[field]
        calibration = pass_record.get("calibration")
        if isinstance(calibration, Mapping):
            if complete := _complete_scores(
                calibration.get("adjusted_scores"), allowed
            ):
                return complete
    for container_name in (
        "scores",
        "candidate_scores",
        "label_scores",
        "candidate_mean_log_probabilities",
        "order_averaged_mean_log_probabilities",
        "all_rotation_mean_log_probabilities",
    ):
        container = record.get(container_name)
        if isinstance(container, Mapping):
            candidate = container.get(field)
            if complete := _complete_scores(candidate, allowed):
                return complete
    if isinstance(passes, Mapping) and isinstance(passes.get(field), Mapping):
        pass_record = passes[field]
        for name in (
            "all_rotation_mean_log_probabilities",
            "order_averaged_mean_log_probabilities",
            "candidate_mean_log_probabilities",
            "candidate_scores",
            "scores",
            "label_scores",
        ):
            if complete := _complete_scores(pass_record.get(name), allowed):
                return complete
    return None


def _schema_argmax(scores: Mapping[str, float], allowed: Sequence[str]) -> str:
    return max(allowed, key=lambda label: (scores[label], -allowed.index(label)))


def aggregate_field(
    records: Sequence[Mapping[str, Any]],
    field: str,
    allowed: Sequence[str],
    *,
    score_aggregation: str = "mean_score",
) -> tuple[str | None, str]:
    if score_aggregation not in CHUNK_AGGREGATIONS:
        raise ValueError(
            f"Unknown score aggregation {score_aggregation!r}"
        )
    complete_scores = [
        scores
        for record in records
        if (scores := scores_for_field(record, field, allowed)) is not None
    ]
    if complete_scores and score_aggregation != "plurality":
        if score_aggregation == "mean_score":
            reduced = {
                label: sum(scores[label] for scores in complete_scores)
                / len(complete_scores)
                for label in allowed
            }
            method = "mean_candidate_score"
        elif score_aggregation == "max_score":
            reduced = {
                label: max(scores[label] for scores in complete_scores)
                for label in allowed
            }
            method = "max_candidate_score"
        else:
            def margin(scores: Mapping[str, float]) -> float:
                ordered = sorted(
                    (scores[label] for label in allowed),
                    reverse=True,
                )
                return ordered[0] - ordered[1] if len(ordered) > 1 else math.inf

            reduced = max(
                enumerate(complete_scores),
                key=lambda item: (margin(item[1]), -item[0]),
            )[1]
            method = "best_margin_candidate_score"
        return _schema_argmax(reduced, allowed), method
    labels = [
        label
        for record in records
        if (label := _label_for_field(record, field)) in allowed
    ]
    if not labels:
        return None, "no_valid_vote"
    counts = Counter(labels)
    return (
        max(allowed, key=lambda label: (counts[label], -allowed.index(label))),
        "plurality",
    )


def _chunk_map(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    chunking = manifest.get("fulltext_chunking")
    if not isinstance(chunking, Mapping):
        raise ValueError("Benchmark manifest has no fulltext_chunking object")
    rows = chunking.get("chunk_id_to_article_id")
    if not isinstance(rows, list):
        raise ValueError("Benchmark manifest has no chunk ID mapping")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Chunk mapping rows must be objects")
        chunk_id = row.get("chunk_id")
        article_id = row.get("article_id")
        if not isinstance(chunk_id, str) or not isinstance(article_id, str):
            raise ValueError("Chunk mapping row has invalid IDs")
        if chunk_id in result:
            raise ValueError(f"Duplicate manifest chunk_id {chunk_id!r}")
        result[chunk_id] = dict(row)
    return result


def split_article_ids(manifest: Mapping[str, Any], split: str) -> set[str]:
    split_record = manifest.get("split")
    if not isinstance(split_record, Mapping):
        raise ValueError("Benchmark manifest has no split object")
    development = split_record.get("development_article_ids")
    evaluation = split_record.get("evaluation_article_ids")
    if not isinstance(development, list) or not isinstance(evaluation, list):
        raise ValueError("Benchmark manifest is missing split article ID lists")
    development_set = set(development)
    evaluation_set = set(evaluation)
    if development_set & evaluation_set:
        raise ValueError("Benchmark manifest split IDs overlap")
    if split == "development":
        return development_set
    if split == "evaluation":
        return evaluation_set
    return development_set | evaluation_set


def _variant_article_ids(
    manifest: Mapping[str, Any], variant: str
) -> set[str]:
    variants = manifest.get("variants")
    if not isinstance(variants, Mapping) or not isinstance(
        variants.get(variant), Mapping
    ):
        raise ValueError(f"Benchmark manifest has no {variant!r} variant metadata")
    values = variants[variant].get("article_ids")
    if not isinstance(values, list) or not all(
        isinstance(value, str) for value in values
    ):
        raise ValueError(f"Variant {variant!r} has no article ID list")
    return set(values)


def _article_targets(manifest: Mapping[str, Any]) -> dict[str, str]:
    target_metadata = manifest.get("target_metadata")
    if not isinstance(target_metadata, Mapping):
        return {}
    values = target_metadata.get("article_target_tickers")
    if not isinstance(values, Mapping):
        return {}
    return {
        str(article_id): str(ticker).upper()
        for article_id, ticker in values.items()
        if isinstance(article_id, str) and isinstance(ticker, str)
    }


def _article_sources(manifest: Mapping[str, Any]) -> dict[str, str]:
    source_metadata = manifest.get("source_metadata")
    if not isinstance(source_metadata, Mapping):
        return {}
    values = source_metadata.get("article_sources")
    if not isinstance(values, Mapping):
        return {}
    return {
        str(article_id): str(source)
        for article_id, source in values.items()
        if isinstance(article_id, str) and isinstance(source, str)
    }


def _article_assignment_bases(
    manifest: Mapping[str, Any],
) -> dict[str, str]:
    provenance = manifest.get("ticker_provenance_contract")
    if not isinstance(provenance, Mapping):
        return {}
    values = provenance.get("article_assignment_basis")
    if not isinstance(values, Mapping):
        return {}
    result = {
        str(article_id): str(basis)
        for article_id, basis in values.items()
        if (
            isinstance(article_id, str)
            and isinstance(basis, str)
            and basis in ASSIGNMENT_FILTERS
        )
    }
    if len(result) != len(values):
        raise ValueError(
            "Benchmark assignment-provenance mapping contains invalid values"
        )
    return result


def aggregate_variant_predictions(
    *,
    variant: str,
    predictions: Sequence[Mapping[str, Any]],
    schema: Mapping[str, Any],
    manifest: Mapping[str, Any],
    selected_article_ids: set[str],
    strict: bool = True,
    chunk_aggregation: str = "mean_score",
) -> tuple[dict[str, dict[str, str | None]], dict[str, Any]]:
    if variant not in VARIANT_ORDER:
        raise ValueError(f"Unknown variant {variant!r}")
    expected_articles = _variant_article_ids(manifest, variant) & selected_article_ids
    chunk_by_id = _chunk_map(manifest) if variant == "fulltext_evidence_chunks" else {}
    expected_targets = _article_targets(manifest)
    records_by_unit: dict[str, Mapping[str, Any]] = {}
    article_by_unit: dict[str, str] = {}

    for row_number, record in enumerate(predictions, start=1):
        unit_id = record.get("chunk_id") or record.get("article_id")
        if not isinstance(unit_id, str) or not unit_id:
            raise ValueError(f"{variant} prediction row {row_number} has no unit ID")
        if unit_id in records_by_unit:
            raise ValueError(f"Duplicate {variant} prediction unit {unit_id!r}")
        if variant == "fulltext_evidence_chunks":
            mapping = chunk_by_id.get(unit_id)
            if mapping is None:
                raise ValueError(f"Unknown full-text chunk prediction {unit_id!r}")
            article_id = str(mapping["article_id"])
            supplied_parent = record.get("parent_article_id")
            if supplied_parent not in (None, article_id):
                raise ValueError(
                    f"Chunk {unit_id!r} parent differs from benchmark manifest"
                )
        else:
            article_id = unit_id
            if article_id not in _variant_article_ids(manifest, variant):
                raise ValueError(
                    f"Unknown {variant} prediction article {article_id!r}"
                )
        supplied_target = record.get("target_ticker")
        target = record.get("target")
        if supplied_target is None and isinstance(target, Mapping):
            supplied_target = target.get("ticker")
        expected_target = expected_targets.get(article_id)
        if (
            expected_target is not None
            and supplied_target is not None
            and str(supplied_target).upper() != expected_target
        ):
            raise ValueError(
                f"{variant} prediction target mismatch for {article_id!r}"
            )
        records_by_unit[unit_id] = record
        article_by_unit[unit_id] = article_id

    relevant_units = {
        unit_id
        for unit_id, article_id in article_by_unit.items()
        if article_id in selected_article_ids
    }
    if variant == "fulltext_evidence_chunks":
        expected_units = {
            chunk_id
            for chunk_id, mapping in chunk_by_id.items()
            if mapping["article_id"] in expected_articles
        }
    else:
        expected_units = expected_articles
    if strict and relevant_units != expected_units:
        missing = expected_units - relevant_units
        unexpected = relevant_units - expected_units
        raise ValueError(
            f"{variant} prediction units differ from benchmark manifest: "
            f"missing={sorted(missing)[:10]!r}, unexpected={sorted(unexpected)[:10]!r}"
        )

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for unit_id in sorted(relevant_units):
        grouped[article_by_unit[unit_id]].append(records_by_unit[unit_id])
    aggregated: dict[str, dict[str, str | None]] = {}
    method_counts = {
        field: Counter() for field in coarse.COARSE_FIELDS
    }
    score_coverage = {
        field: {
            "all_chunks_scored_article_count": 0,
            "partial_chunks_scored_article_count": 0,
            "no_chunks_scored_article_count": 0,
            "chunks_with_complete_scores": 0,
            "total_chunks": 0,
        }
        for field in coarse.COARSE_FIELDS
    }
    closed = schema["closed_label_fields"]
    for article_id in sorted(grouped):
        aggregated[article_id] = {}
        records = grouped[article_id]
        for field in coarse.COARSE_FIELDS:
            scored_chunk_count = sum(
                scores_for_field(record, field, closed[field]) is not None
                for record in records
            )
            coverage = score_coverage[field]
            coverage["chunks_with_complete_scores"] += scored_chunk_count
            coverage["total_chunks"] += len(records)
            if scored_chunk_count == len(records):
                coverage["all_chunks_scored_article_count"] += 1
            elif scored_chunk_count:
                coverage["partial_chunks_scored_article_count"] += 1
            else:
                coverage["no_chunks_scored_article_count"] += 1
            label, method = aggregate_field(
                records,
                field,
                closed[field],
                score_aggregation=(
                    chunk_aggregation
                    if variant == "fulltext_evidence_chunks"
                    else "mean_score"
                ),
            )
            aggregated[article_id][field] = label
            method_counts[field][method] += 1
    metadata = {
        "input_prediction_unit_count": len(predictions),
        "selected_prediction_unit_count": len(relevant_units),
        "aggregated_article_count": len(aggregated),
        "expected_article_count": len(expected_articles),
        "complete_for_selected_split": (
            relevant_units == expected_units and set(aggregated) == expected_articles
        ),
        "requested_chunk_aggregation": (
            chunk_aggregation
            if variant == "fulltext_evidence_chunks"
            else "not_applicable"
        ),
        "field_aggregation_method_counts": {
            field: dict(sorted(counts.items()))
            for field, counts in method_counts.items()
        },
        "field_score_coverage": {
            field: {
                **coverage,
                "complete_score_chunk_rate": safe_divide(
                    coverage["chunks_with_complete_scores"],
                    coverage["total_chunks"],
                ),
                "partial_scores_are_used_when_available": True,
            }
            for field, coverage in score_coverage.items()
        },
    }
    return aggregated, metadata


def metrics_for_ids(
    *,
    article_ids: Sequence[str],
    reference: Mapping[str, Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, str | None]],
    schema: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for field in coarse.COARSE_FIELDS:
        truths = [str(reference[article_id]["labels"][field]) for article_id in article_ids]
        guesses = [predictions[article_id].get(field) for article_id in article_ids]
        result[field] = closed_label_metrics(
            truths, guesses, schema["closed_label_fields"][field]
        )
    return result


def _mean_field(metrics: Mapping[str, Mapping[str, Any]], key: str) -> float:
    return safe_divide(
        sum(float(metrics[field][key]) for field in coarse.COARSE_FIELDS),
        len(coarse.COARSE_FIELDS),
    )


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _mcnemar_exact_p(baseline_only: int, comparison_only: int) -> float:
    discordant = baseline_only + comparison_only
    if not discordant:
        return 1.0
    tail = min(baseline_only, comparison_only)
    probability = sum(
        math.comb(discordant, value)
        for value in range(tail + 1)
    ) / (2**discordant)
    return min(1.0, 2 * probability)


def paired_field_uncertainty(
    *,
    article_ids: Sequence[str],
    reference: Mapping[str, Mapping[str, Any]],
    baseline: Mapping[str, Mapping[str, str | None]],
    comparison: Mapping[str, Mapping[str, str | None]],
    field: str,
    allowed: Sequence[str],
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    truths = [
        str(reference[article_id]["labels"][field])
        for article_id in article_ids
    ]
    baseline_guesses = [
        baseline[article_id].get(field) for article_id in article_ids
    ]
    comparison_guesses = [
        comparison[article_id].get(field) for article_id in article_ids
    ]
    baseline_correct = [
        truth == guess
        for truth, guess in zip(truths, baseline_guesses)
    ]
    comparison_correct = [
        truth == guess
        for truth, guess in zip(truths, comparison_guesses)
    ]
    baseline_only = sum(
        before and not after
        for before, after in zip(baseline_correct, comparison_correct)
    )
    comparison_only = sum(
        after and not before
        for before, after in zip(baseline_correct, comparison_correct)
    )
    both_correct = sum(
        before and after
        for before, after in zip(baseline_correct, comparison_correct)
    )
    both_wrong = len(article_ids) - baseline_only - comparison_only - both_correct

    rng_seed = int.from_bytes(
        hashlib.sha256(
            (
                "news-text-ablation-paired-bootstrap-v1"
                f"\x1f{field}\x1f{len(article_ids)}"
            ).encode("utf-8")
        ).digest()[:8],
        "big",
    )
    rng = random.Random(rng_seed)
    accuracy_deltas: list[float] = []
    macro_f1_deltas: list[float] = []
    for _ in range(replicates):
        indices = [
            rng.randrange(len(article_ids)) for _ in article_ids
        ]
        sampled_truths = [truths[index] for index in indices]
        sampled_baseline = [baseline_guesses[index] for index in indices]
        sampled_comparison = [
            comparison_guesses[index] for index in indices
        ]
        baseline_metrics = closed_label_metrics(
            sampled_truths, sampled_baseline, allowed
        )
        comparison_metrics = closed_label_metrics(
            sampled_truths, sampled_comparison, allowed
        )
        accuracy_deltas.append(
            comparison_metrics["accuracy"] - baseline_metrics["accuracy"]
        )
        macro_f1_deltas.append(
            comparison_metrics["macro_f1"]
            - baseline_metrics["macro_f1"]
        )

    return {
        "paired_article_bootstrap_replicates": replicates,
        "accuracy_delta_bootstrap_95pct": [
            _percentile(accuracy_deltas, 0.025),
            _percentile(accuracy_deltas, 0.975),
        ],
        "macro_f1_delta_bootstrap_95pct": [
            _percentile(macro_f1_deltas, 0.025),
            _percentile(macro_f1_deltas, 0.975),
        ],
        "mcnemar": {
            "both_correct": both_correct,
            "baseline_only_correct": baseline_only,
            "comparison_only_correct": comparison_only,
            "both_wrong": both_wrong,
            "two_sided_exact_p": _mcnemar_exact_p(
                baseline_only, comparison_only
            ),
        },
    }


def paired_aggregate_uncertainty(
    *,
    article_ids: Sequence[str],
    reference: Mapping[str, Mapping[str, Any]],
    baseline: Mapping[str, Mapping[str, str | None]],
    comparison: Mapping[str, Mapping[str, str | None]],
    schema: Mapping[str, Any],
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    fields = tuple(coarse.COARSE_FIELDS)
    truth_by_field = {
        field: [
            str(reference[article_id]["labels"][field])
            for article_id in article_ids
        ]
        for field in fields
    }
    baseline_by_field = {
        field: [
            baseline[article_id].get(field) for article_id in article_ids
        ]
        for field in fields
    }
    comparison_by_field = {
        field: [
            comparison[article_id].get(field) for article_id in article_ids
        ]
        for field in fields
    }
    rng_seed = int.from_bytes(
        hashlib.sha256(
            (
                "news-text-ablation-paired-aggregate-bootstrap-v1"
                f"\x1f{len(article_ids)}"
            ).encode("utf-8")
        ).digest()[:8],
        "big",
    )
    rng = random.Random(rng_seed)
    mean_accuracy_deltas: list[float] = []
    mean_macro_f1_deltas: list[float] = []
    primary_macro_f1_deltas: list[float] = []
    for _ in range(replicates):
        indices = [rng.randrange(len(article_ids)) for _ in article_ids]
        accuracy_deltas: dict[str, float] = {}
        macro_f1_deltas: dict[str, float] = {}
        for field in fields:
            sampled_truths = [
                truth_by_field[field][index] for index in indices
            ]
            sampled_baseline = [
                baseline_by_field[field][index] for index in indices
            ]
            sampled_comparison = [
                comparison_by_field[field][index] for index in indices
            ]
            baseline_metrics = closed_label_metrics(
                sampled_truths,
                sampled_baseline,
                schema["closed_label_fields"][field],
            )
            comparison_metrics = closed_label_metrics(
                sampled_truths,
                sampled_comparison,
                schema["closed_label_fields"][field],
            )
            accuracy_deltas[field] = (
                comparison_metrics["accuracy"]
                - baseline_metrics["accuracy"]
            )
            macro_f1_deltas[field] = (
                comparison_metrics["macro_f1"]
                - baseline_metrics["macro_f1"]
            )
        mean_accuracy_deltas.append(
            sum(accuracy_deltas.values()) / len(fields)
        )
        mean_macro_f1_deltas.append(
            sum(macro_f1_deltas.values()) / len(fields)
        )
        primary_macro_f1_deltas.append(
            sum(
                macro_f1_deltas[field]
                for field in PRIMARY_SELECTION_FIELDS
            )
            / len(PRIMARY_SELECTION_FIELDS)
        )
    return {
        "paired_article_bootstrap_replicates": replicates,
        "mean_field_accuracy_delta_bootstrap_95pct": [
            _percentile(mean_accuracy_deltas, 0.025),
            _percentile(mean_accuracy_deltas, 0.975),
        ],
        "mean_field_macro_f1_delta_bootstrap_95pct": [
            _percentile(mean_macro_f1_deltas, 0.025),
            _percentile(mean_macro_f1_deltas, 0.975),
        ],
        "primary_two_field_macro_f1_delta_bootstrap_95pct": [
            _percentile(primary_macro_f1_deltas, 0.025),
            _percentile(primary_macro_f1_deltas, 0.975),
        ],
        "primary_fields": list(PRIMARY_SELECTION_FIELDS),
    }


def build_evaluation_report(
    *,
    reference_records: Sequence[Mapping[str, Any]],
    predictions_by_variant: Mapping[str, Sequence[Mapping[str, Any]]],
    manifest: Mapping[str, Any],
    schema: Mapping[str, Any],
    split: str = "evaluation",
    strict: bool = True,
    input_hashes: Mapping[str, str] | None = None,
    chunk_aggregation: str = "mean_score",
    reference_filter: str = "all",
    source_filter: str | None = None,
    assignment_filter: str | None = None,
) -> dict[str, Any]:
    if reference_filter not in {"all", "non_abstained"}:
        raise ValueError(f"Unknown reference filter {reference_filter!r}")
    if (
        assignment_filter is not None
        and assignment_filter not in ASSIGNMENT_FILTERS
    ):
        raise ValueError(f"Unknown assignment filter {assignment_filter!r}")
    split_ids = split_article_ids(manifest, split)
    references = reference_by_id(reference_records)
    missing_reference = split_ids - set(references)
    if missing_reference:
        raise ValueError(
            f"Reference is missing {len(missing_reference)} selected benchmark articles"
        )
    selected_ids = (
        split_ids
        if reference_filter == "all"
        else {
            article_id
            for article_id in split_ids
            if not references[article_id]["abstained"]
        }
    )
    if source_filter is not None:
        sources = _article_sources(manifest)
        if not sources:
            raise ValueError(
                "Source filtering requires benchmark source_metadata"
            )
        selected_ids = {
            article_id
            for article_id in selected_ids
            if sources.get(article_id) == source_filter
        }
        if not selected_ids:
            raise ValueError(
                f"No selected benchmark articles have source {source_filter!r}"
            )
    if assignment_filter is not None:
        assignment_bases = _article_assignment_bases(manifest)
        if not assignment_bases:
            raise ValueError(
                "Assignment filtering requires benchmark assignment provenance"
            )
        selected_ids = {
            article_id
            for article_id in selected_ids
            if assignment_bases.get(article_id) == assignment_filter
        }
        if not selected_ids:
            raise ValueError(
                "No selected benchmark articles have assignment basis "
                f"{assignment_filter!r}"
            )
    expected_targets = _article_targets(manifest)
    target_mismatches = [
        article_id
        for article_id in selected_ids
        if expected_targets.get(article_id) is not None
        and references[article_id].get("target_ticker") is not None
        and str(references[article_id]["target_ticker"]).upper()
        != expected_targets[article_id]
    ]
    if target_mismatches:
        raise ValueError(
            "Reference target ticker differs from benchmark manifest for "
            + ", ".join(sorted(target_mismatches)[:10])
        )

    aggregated_by_variant: dict[str, dict[str, dict[str, str | None]]] = {}
    variant_reports: dict[str, Any] = {}
    for variant in VARIANT_ORDER:
        if variant not in predictions_by_variant:
            continue
        aggregated, aggregation_metadata = aggregate_variant_predictions(
            variant=variant,
            predictions=predictions_by_variant[variant],
            schema=schema,
            manifest=manifest,
            selected_article_ids=selected_ids,
            strict=strict,
            chunk_aggregation=chunk_aggregation,
        )
        article_ids = sorted(aggregated)
        metrics = metrics_for_ids(
            article_ids=article_ids,
            reference=references,
            predictions=aggregated,
            schema=schema,
        )
        aggregated_by_variant[variant] = aggregated
        variant_reports[variant] = {
            "article_count": len(article_ids),
            "article_coverage_of_selected_split": safe_divide(
                len(article_ids), len(selected_ids)
            ),
            "aggregation": aggregation_metadata,
            "field_metrics": metrics,
            "mean_field_accuracy": _mean_field(metrics, "accuracy"),
            "mean_field_macro_f1": _mean_field(metrics, "macro_f1"),
        }

    paired: dict[str, Any] = {}
    available = [variant for variant in VARIANT_ORDER if variant in aggregated_by_variant]
    for baseline, comparison in combinations(available, 2):
        shared = sorted(
            set(aggregated_by_variant[baseline])
            & set(aggregated_by_variant[comparison])
        )
        baseline_metrics = metrics_for_ids(
            article_ids=shared,
            reference=references,
            predictions=aggregated_by_variant[baseline],
            schema=schema,
        )
        comparison_metrics = metrics_for_ids(
            article_ids=shared,
            reference=references,
            predictions=aggregated_by_variant[comparison],
            schema=schema,
        )
        fields: dict[str, Any] = {}
        for field in coarse.COARSE_FIELDS:
            fields[field] = {
                "baseline_accuracy": baseline_metrics[field]["accuracy"],
                "comparison_accuracy": comparison_metrics[field]["accuracy"],
                "accuracy_delta": (
                    comparison_metrics[field]["accuracy"]
                    - baseline_metrics[field]["accuracy"]
                ),
                "baseline_macro_f1": baseline_metrics[field]["macro_f1"],
                "comparison_macro_f1": comparison_metrics[field]["macro_f1"],
                "macro_f1_delta": (
                    comparison_metrics[field]["macro_f1"]
                    - baseline_metrics[field]["macro_f1"]
                ),
                "paired_uncertainty": paired_field_uncertainty(
                    article_ids=shared,
                    reference=references,
                    baseline=aggregated_by_variant[baseline],
                    comparison=aggregated_by_variant[comparison],
                    field=field,
                    allowed=schema["closed_label_fields"][field],
                ),
            }
        name = f"{comparison}_minus_{baseline}"
        paired[name] = {
            "baseline_variant": baseline,
            "comparison_variant": comparison,
            "paired_article_count": len(shared),
            "fields": fields,
            "mean_field_accuracy_delta": safe_divide(
                sum(fields[field]["accuracy_delta"] for field in coarse.COARSE_FIELDS),
                len(coarse.COARSE_FIELDS),
            ),
            "mean_field_macro_f1_delta": safe_divide(
                sum(fields[field]["macro_f1_delta"] for field in coarse.COARSE_FIELDS),
                len(coarse.COARSE_FIELDS),
            ),
            "aggregate_uncertainty": paired_aggregate_uncertainty(
                article_ids=shared,
                reference=references,
                baseline=aggregated_by_variant[baseline],
                comparison=aggregated_by_variant[comparison],
                schema=schema,
            ),
        }

    return {
        "report_version": REPORT_VERSION,
        "reference_type": "coarse GPT silver reference, not human ground truth",
        "split": split,
        "reference_filter": reference_filter,
        "source_filter": source_filter,
        "assignment_filter": assignment_filter,
        "split_article_count_before_reference_filter": len(split_ids),
        "selected_reference_article_count": len(selected_ids),
        "strict_manifest_completeness": strict,
        "aggregation_contract": {
            "requested_fulltext_chunk_aggregation": chunk_aggregation,
            "score_methods": {
                "mean_score": "arithmetic mean of each label score",
                "max_score": "maximum observed score for each label",
                "best_margin": "all scores from the chunk with the largest top-two margin",
                "plurality": "ignore score maps and vote valid chunk labels",
            },
            "fallback_method": "plurality of valid chunk labels",
            "tie_break": "schema label order",
        },
        "metric_contract": {
            "accuracy": "correct article labels divided by paired article count",
            "macro_f1": (
                "unweighted mean of per-class F1 over classes with reference "
                "support in the evaluated article set"
            ),
            "cohens_kappa": (
                "chance-adjusted agreement over the closed labels; invalid "
                "predictions have zero reference prevalence"
            ),
            "majority_class_baseline_accuracy": (
                "reference prevalence of the most-supported class"
            ),
            "invalid_or_missing_article_label": "counted as an error",
            "paired_uncertainty": (
                "2,000 deterministic paired article bootstrap replicates plus "
                "two-sided exact McNemar; articles are treated as independent "
                "and the interval is not publisher- or date-clustered"
            ),
        },
        "variants": variant_reports,
        "paired_deltas": paired,
        "input_sha256": dict(input_hashes or {}),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    prediction_paths = parse_prediction_specs(args.predictions)
    aggregation_lock_metadata: dict[str, Any] | None = None
    if args.aggregation_lock is not None:
        if args.split != "evaluation":
            raise ValueError(
                "--aggregation-lock is only valid with --split evaluation"
            )
        import lock_news_text_ablation_aggregation as aggregation_locker

        aggregation_lock_metadata = (
            aggregation_locker.validate_lock_for_evaluation(
                lock_path=args.aggregation_lock,
                benchmark_manifest_path=args.benchmark_manifest,
                reference_path=args.reference,
                schema_path=args.schema,
                prediction_paths=prediction_paths,
                chunk_aggregation=args.chunk_aggregation,
            )
        )
    manifest = json.loads(args.benchmark_manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("Benchmark manifest is not complete")
    schema = coarse.load_schema(args.schema)
    schema_metadata = manifest.get("evaluation_schema")
    if isinstance(schema_metadata, Mapping) and schema_metadata.get(
        "sha256"
    ) != sha256_file(args.schema):
        raise ValueError("Evaluation schema differs from the benchmark manifest")
    reference_records = read_jsonl(args.reference)
    predictions_by_variant = {
        variant: read_jsonl(path) for variant, path in prediction_paths.items()
    }
    hashes = {
        "benchmark_manifest": sha256_file(args.benchmark_manifest),
        "reference": sha256_file(args.reference),
        "schema": sha256_file(args.schema),
        **{
            f"predictions:{variant}": sha256_file(path)
            for variant, path in prediction_paths.items()
        },
    }
    report = build_evaluation_report(
        reference_records=reference_records,
        predictions_by_variant=predictions_by_variant,
        manifest=manifest,
        schema=schema,
        split=args.split,
        strict=not args.allow_partial,
        input_hashes=hashes,
        chunk_aggregation=args.chunk_aggregation,
        reference_filter=args.reference_filter,
        source_filter=args.source_filter,
        assignment_filter=args.assignment_filter,
    )
    if aggregation_lock_metadata is not None:
        report["aggregation_lock"] = aggregation_lock_metadata
    _write_json(args.output, report)
    print(
        f"Wrote {args.split} ablation evaluation for "
        f"{len(report['variants'])} variant(s) to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
