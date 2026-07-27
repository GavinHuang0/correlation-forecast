"""Freeze development-selected full-text aggregation with artifact provenance.

This script is deliberately CPU-only.  It consumes the four development
evaluation reports produced with the supported chunk reducers, validates the
prediction sidecars and benchmark inputs, applies the predeclared
lexicographic selection rule, and writes an immutable-input lock for holdout
evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import evaluate_news_text_ablation as evaluator


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
LOCK_VERSION = "news-text-ablation-aggregation-lock-v1"
REQUIRED_VARIANTS = ("massive_description", "fulltext_evidence_chunks")
PRIMARY_FIELDS = evaluator.PRIMARY_SELECTION_FIELDS
TIE_BREAK_ORDER = (
    "mean_score",
    "plurality",
    "best_margin",
    "max_score",
)


class AggregationLockError(ValueError):
    """Raised when an aggregation artifact violates the frozen contract."""


def parse_named_paths(
    values: Sequence[str],
    *,
    label: str,
    allowed: Sequence[str] | None = None,
) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise AggregationLockError(
                f"{label} must be NAME=PATH, got {value!r}"
            )
        name, raw_path = value.split("=", 1)
        name = name.strip()
        raw_path = raw_path.strip()
        if not name or not raw_path:
            raise AggregationLockError(f"Invalid {label} specification {value!r}")
        if allowed is not None and name not in allowed:
            raise AggregationLockError(
                f"Unknown {label} name {name!r}; expected one of {tuple(allowed)!r}"
            )
        if name in result:
            raise AggregationLockError(f"Duplicate {label} name {name!r}")
        result[name] = Path(raw_path)
    return result


def read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AggregationLockError(
            f"{description} is invalid JSON: {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise AggregationLockError(f"{description} must be a JSON object: {path}")
    return value


def jsonl_record_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig") as handle:
        return sum(1 for line in handle if line.strip())


def prediction_sidecar_path(prediction_path: Path) -> Path:
    return prediction_path.with_suffix(
        prediction_path.suffix + ".manifest.json"
    )


def _manifest_variant_record(
    manifest: Mapping[str, Any], variant: str
) -> Mapping[str, Any]:
    variants = manifest.get("variants")
    if not isinstance(variants, Mapping):
        raise AggregationLockError("Benchmark manifest has no variants object")
    record = variants.get(variant)
    if not isinstance(record, Mapping):
        raise AggregationLockError(
            f"Benchmark manifest has no {variant!r} variant"
        )
    return record


def validate_benchmark_variant(
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    variant: str,
) -> dict[str, Any]:
    record = _manifest_variant_record(manifest, variant)
    filename = record.get("file")
    declared_sha256 = record.get("sha256")
    if not isinstance(filename, str) or not filename:
        raise AggregationLockError(
            f"Benchmark variant {variant!r} has no file"
        )
    if not isinstance(declared_sha256, str) or not declared_sha256:
        raise AggregationLockError(
            f"Benchmark variant {variant!r} has no SHA-256"
        )
    path = (manifest_path.parent / filename).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_sha256 = evaluator.sha256_file(path)
    if actual_sha256 != declared_sha256:
        raise AggregationLockError(
            f"Benchmark variant {variant!r} differs from its manifest SHA-256"
        )
    return {
        "path": str(path),
        "sha256": actual_sha256,
        "declared_sha256": declared_sha256,
    }


def validate_prediction_artifact(
    *,
    variant: str,
    prediction_path: Path,
    sidecar_path: Path,
    benchmark_variant: Mapping[str, Any],
    schema_sha256: str,
) -> dict[str, Any]:
    prediction_path = prediction_path.resolve()
    sidecar_path = sidecar_path.resolve()
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)
    if not sidecar_path.is_file():
        raise FileNotFoundError(sidecar_path)
    sidecar = read_json(sidecar_path, f"{variant} prediction sidecar")
    if sidecar.get("status") != "complete":
        raise AggregationLockError(
            f"{variant} prediction sidecar status is not complete"
        )
    output_sha256 = evaluator.sha256_file(prediction_path)
    if sidecar.get("output_sha256") != output_sha256:
        raise AggregationLockError(
            f"{variant} prediction output differs from its sidecar SHA-256"
        )
    expected_input_sha256 = benchmark_variant["sha256"]
    if sidecar.get("input_sha256") != expected_input_sha256:
        raise AggregationLockError(
            f"{variant} sidecar input SHA-256 differs from the current "
            "benchmark variant"
        )
    input_path_value = sidecar.get("input_path")
    if not isinstance(input_path_value, str) or not input_path_value:
        raise AggregationLockError(f"{variant} sidecar has no input_path")
    if Path(input_path_value).resolve() != Path(
        str(benchmark_variant["path"])
    ).resolve():
        raise AggregationLockError(
            f"{variant} sidecar input_path differs from the current "
            "benchmark variant"
        )
    if sidecar.get("schema_sha256") != schema_sha256:
        raise AggregationLockError(
            f"{variant} sidecar schema SHA-256 differs from the current schema"
        )
    record_count = jsonl_record_count(prediction_path)
    declared_count = sidecar.get("total_prediction_count")
    if (
        isinstance(declared_count, bool)
        or not isinstance(declared_count, int)
        or declared_count != record_count
    ):
        raise AggregationLockError(
            f"{variant} prediction count differs from its sidecar"
        )
    return {
        "path": str(prediction_path),
        "sha256": output_sha256,
        "record_count": record_count,
        "sidecar": {
            "path": str(sidecar_path),
            "sha256": evaluator.sha256_file(sidecar_path),
            "status": sidecar["status"],
            "output_sha256": sidecar["output_sha256"],
            "input_path": str(Path(input_path_value).resolve()),
            "input_sha256": sidecar["input_sha256"],
            "schema_sha256": sidecar["schema_sha256"],
            "model_id": sidecar.get("model_id"),
            "model_revision": sidecar.get("model_revision"),
            "experiment_id": sidecar.get("experiment_id"),
            "prompt_version": sidecar.get("prompt_version"),
            "extractor_source_sha256": sidecar.get(
                "extractor_source_sha256"
            ),
            "deterministic_rule_version": sidecar.get(
                "deterministic_rule_version"
            ),
        },
    }


def _finite_metric(value: Any, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AggregationLockError(f"{description} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise AggregationLockError(f"{description} is not finite")
    return result


def validate_development_report(
    *,
    reducer: str,
    report_path: Path,
    expected_hashes: Mapping[str, str],
    expected_development_count: int,
) -> tuple[dict[str, Any], dict[str, float]]:
    report_path = report_path.resolve()
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = read_json(report_path, f"{reducer} development report")
    if report.get("report_version") != evaluator.REPORT_VERSION:
        raise AggregationLockError(
            f"{reducer} report has an unsupported report_version"
        )
    if report.get("split") != "development":
        raise AggregationLockError(
            f"{reducer} report is not a development-split report"
        )
    if report.get("reference_filter") != "all":
        raise AggregationLockError(
            f"{reducer} development report must use reference_filter=all"
        )
    if report.get("source_filter") is not None:
        raise AggregationLockError(
            f"{reducer} development report must not use a source filter"
        )
    if report.get("assignment_filter") is not None:
        raise AggregationLockError(
            f"{reducer} development report must not use an assignment filter"
        )
    if report.get("strict_manifest_completeness") is not True:
        raise AggregationLockError(
            f"{reducer} development report is not manifest-complete"
        )
    if (
        report.get("selected_reference_article_count")
        != expected_development_count
    ):
        raise AggregationLockError(
            f"{reducer} development report has the wrong selected count"
        )
    aggregation = report.get("aggregation_contract")
    if not isinstance(aggregation, Mapping) or aggregation.get(
        "requested_fulltext_chunk_aggregation"
    ) != reducer:
        raise AggregationLockError(
            f"{reducer} report records a different chunk reducer"
        )
    input_hashes = report.get("input_sha256")
    if not isinstance(input_hashes, Mapping):
        raise AggregationLockError(f"{reducer} report has no input hashes")
    for name, expected in expected_hashes.items():
        if input_hashes.get(name) != expected:
            raise AggregationLockError(
                f"{reducer} report input hash differs for {name}"
            )
    variants = report.get("variants")
    if not isinstance(variants, Mapping):
        raise AggregationLockError(f"{reducer} report has no variants")
    for variant in REQUIRED_VARIANTS:
        if not isinstance(variants.get(variant), Mapping):
            raise AggregationLockError(
                f"{reducer} report is missing {variant}"
            )
        if variants[variant].get("article_count") != expected_development_count:
            raise AggregationLockError(
                f"{reducer} report has incomplete {variant} article coverage"
            )
        if variants[variant].get(
            "article_coverage_of_selected_split"
        ) != 1.0:
            raise AggregationLockError(
                f"{reducer} report has partial {variant} article coverage"
            )
    fulltext = variants["fulltext_evidence_chunks"]
    fulltext_aggregation = fulltext.get("aggregation")
    if not isinstance(
        fulltext_aggregation, Mapping
    ) or fulltext_aggregation.get("requested_chunk_aggregation") != reducer:
        raise AggregationLockError(
            f"{reducer} full-text variant records a different reducer"
        )
    field_metrics = fulltext.get("field_metrics")
    if not isinstance(field_metrics, Mapping):
        raise AggregationLockError(
            f"{reducer} full-text variant has no field metrics"
        )
    primary_values: list[float] = []
    for field in PRIMARY_FIELDS:
        metrics = field_metrics.get(field)
        if not isinstance(metrics, Mapping):
            raise AggregationLockError(
                f"{reducer} report lacks {field} metrics"
            )
        primary_values.append(
            _finite_metric(
                metrics.get("macro_f1"),
                f"{reducer} {field} macro-F1",
            )
        )
    selection_metrics = {
        "primary_two_field_macro_f1": sum(primary_values)
        / len(primary_values),
        "fulltext_mean_field_macro_f1": _finite_metric(
            fulltext.get("mean_field_macro_f1"),
            f"{reducer} full-text mean-field macro-F1",
        ),
        "fulltext_mean_field_accuracy": _finite_metric(
            fulltext.get("mean_field_accuracy"),
            f"{reducer} full-text mean-field accuracy",
        ),
    }
    provenance = {
        "path": str(report_path),
        "sha256": evaluator.sha256_file(report_path),
        "report_version": report["report_version"],
        "split": report["split"],
        "selected_reference_article_count": expected_development_count,
        "selection_metrics": selection_metrics,
    }
    return provenance, selection_metrics


def select_reducer(
    candidates: Mapping[str, Mapping[str, float]]
) -> str:
    if set(candidates) != set(evaluator.CHUNK_AGGREGATIONS):
        raise AggregationLockError(
            "Reducer candidates must contain exactly "
            + ", ".join(evaluator.CHUNK_AGGREGATIONS)
        )
    tie_rank = {name: index for index, name in enumerate(TIE_BREAK_ORDER)}
    return max(
        candidates,
        key=lambda reducer: (
            float(candidates[reducer]["primary_two_field_macro_f1"]),
            float(candidates[reducer]["fulltext_mean_field_macro_f1"]),
            float(candidates[reducer]["fulltext_mean_field_accuracy"]),
            -tie_rank[reducer],
        ),
    )


def build_aggregation_lock(
    *,
    model_name: str,
    benchmark_manifest_path: Path,
    reference_path: Path,
    schema_path: Path,
    prediction_paths: Mapping[str, Path],
    sidecar_paths: Mapping[str, Path],
    development_report_paths: Mapping[str, Path],
) -> dict[str, Any]:
    if not model_name.strip():
        raise AggregationLockError("model_name must not be empty")
    if set(prediction_paths) != set(REQUIRED_VARIANTS):
        raise AggregationLockError(
            "Predictions must contain exactly " + ", ".join(REQUIRED_VARIANTS)
        )
    if set(sidecar_paths) - set(prediction_paths):
        raise AggregationLockError(
            "A sidecar override was supplied for an unknown prediction variant"
        )
    if set(development_report_paths) != set(evaluator.CHUNK_AGGREGATIONS):
        raise AggregationLockError(
            "Development reports must contain exactly "
            + ", ".join(evaluator.CHUNK_AGGREGATIONS)
        )

    benchmark_manifest_path = benchmark_manifest_path.resolve()
    reference_path = reference_path.resolve()
    schema_path = schema_path.resolve()
    for path in (benchmark_manifest_path, reference_path, schema_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = read_json(benchmark_manifest_path, "benchmark manifest")
    if manifest.get("status") != "complete":
        raise AggregationLockError("Benchmark manifest status is not complete")
    schema_sha256 = evaluator.sha256_file(schema_path)
    schema_metadata = manifest.get("evaluation_schema")
    if not isinstance(schema_metadata, Mapping) or schema_metadata.get(
        "sha256"
    ) != schema_sha256:
        raise AggregationLockError(
            "Current schema differs from the benchmark manifest"
        )
    split = manifest.get("split")
    development_ids = (
        split.get("development_article_ids")
        if isinstance(split, Mapping)
        else None
    )
    if not isinstance(development_ids, list) or not all(
        isinstance(value, str) for value in development_ids
    ):
        raise AggregationLockError(
            "Benchmark manifest has no development article IDs"
        )
    if len(development_ids) != len(set(development_ids)):
        raise AggregationLockError(
            "Benchmark development article IDs are not unique"
        )

    benchmark_variants = {
        variant: validate_benchmark_variant(
            manifest_path=benchmark_manifest_path,
            manifest=manifest,
            variant=variant,
        )
        for variant in REQUIRED_VARIANTS
    }
    predictions: dict[str, Any] = {}
    for variant, prediction_path in prediction_paths.items():
        sidecar_path = sidecar_paths.get(
            variant, prediction_sidecar_path(prediction_path)
        )
        predictions[variant] = validate_prediction_artifact(
            variant=variant,
            prediction_path=prediction_path,
            sidecar_path=sidecar_path,
            benchmark_variant=benchmark_variants[variant],
            schema_sha256=schema_sha256,
        )
    identities = {
        (
            record["sidecar"].get("model_id"),
            record["sidecar"].get("model_revision"),
            record["sidecar"].get("prompt_version"),
            record["sidecar"].get("extractor_source_sha256"),
            record["sidecar"].get("deterministic_rule_version"),
        )
        for record in predictions.values()
    }
    identity = next(iter(identities)) if len(identities) == 1 else None
    if identity is None or any(value is None for value in identity):
        raise AggregationLockError(
            "Prediction sidecars do not identify one consistent model and "
            "extractor protocol"
        )

    benchmark_sha256 = evaluator.sha256_file(benchmark_manifest_path)
    reference_sha256 = evaluator.sha256_file(reference_path)
    expected_report_hashes = {
        "benchmark_manifest": benchmark_sha256,
        "reference": reference_sha256,
        "schema": schema_sha256,
        **{
            f"predictions:{variant}": record["sha256"]
            for variant, record in predictions.items()
        },
    }
    report_provenance: dict[str, Any] = {}
    candidate_metrics: dict[str, dict[str, float]] = {}
    for reducer, report_path in development_report_paths.items():
        provenance, metrics = validate_development_report(
            reducer=reducer,
            report_path=report_path,
            expected_hashes=expected_report_hashes,
            expected_development_count=len(development_ids),
        )
        report_provenance[reducer] = provenance
        candidate_metrics[reducer] = metrics
    selected = select_reducer(candidate_metrics)

    evaluator_path = Path(evaluator.__file__).resolve()
    lock_script_path = Path(__file__).resolve()
    return {
        "lock_version": LOCK_VERSION,
        "status": "frozen_on_development",
        "model_name": model_name.strip(),
        "model_identity": {
            "model_id": identity[0],
            "model_revision": identity[1],
            "prompt_version": identity[2],
            "extractor_source_sha256": identity[3],
            "deterministic_rule_version": identity[4],
        },
        "selection_split": "development",
        "development_article_count": len(development_ids),
        "selection_rule": {
            "direction": "maximize_lexicographically",
            "criteria": [
                "mean full-text macro-F1 over shock_scope and directional_alignment",
                "full-text mean-field macro-F1",
                "full-text mean-field accuracy",
                "fixed reducer preference",
            ],
            "primary_fields": list(PRIMARY_FIELDS),
            "fixed_reducer_preference": list(TIE_BREAK_ORDER),
        },
        "selected_chunk_aggregation": selected,
        "selected_metrics": candidate_metrics[selected],
        "candidate_metrics": {
            reducer: candidate_metrics[reducer]
            for reducer in TIE_BREAK_ORDER
        },
        "artifacts": {
            "benchmark_manifest": {
                "path": str(benchmark_manifest_path),
                "sha256": benchmark_sha256,
            },
            "benchmark_variants": benchmark_variants,
            "reference": {
                "path": str(reference_path),
                "sha256": reference_sha256,
            },
            "schema": {
                "path": str(schema_path),
                "sha256": schema_sha256,
            },
            "predictions": predictions,
            "development_reports": {
                reducer: report_provenance[reducer]
                for reducer in TIE_BREAK_ORDER
            },
        },
        "implementation": {
            "lock_script": {
                "path": str(lock_script_path),
                "sha256": evaluator.sha256_file(lock_script_path),
            },
            "evaluator": {
                "path": str(evaluator_path),
                "sha256": evaluator.sha256_file(evaluator_path),
                "report_version": evaluator.REPORT_VERSION,
            },
        },
        "holdout_contract": {
            "required_split": "evaluation",
            "required_chunk_aggregation": selected,
            "artifact_hashes_must_match": True,
            "prediction_sidecars_must_remain_complete": True,
        },
    }


def validate_lock_for_evaluation(
    *,
    lock_path: Path,
    benchmark_manifest_path: Path,
    reference_path: Path,
    schema_path: Path,
    prediction_paths: Mapping[str, Path],
    chunk_aggregation: str,
) -> dict[str, Any]:
    lock_path = lock_path.resolve()
    lock = read_json(lock_path, "aggregation lock")
    if lock.get("lock_version") != LOCK_VERSION:
        raise AggregationLockError("Unsupported aggregation lock version")
    if lock.get("status") != "frozen_on_development":
        raise AggregationLockError("Aggregation lock is not frozen on development")
    if lock.get("selected_chunk_aggregation") != chunk_aggregation:
        raise AggregationLockError(
            "Requested chunk aggregation differs from the development lock"
        )
    artifacts = lock.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise AggregationLockError("Aggregation lock has no artifact contract")
    current_core = {
        "benchmark_manifest": evaluator.sha256_file(
            benchmark_manifest_path.resolve()
        ),
        "reference": evaluator.sha256_file(reference_path.resolve()),
        "schema": evaluator.sha256_file(schema_path.resolve()),
    }
    for name, actual_sha256 in current_core.items():
        record = artifacts.get(name)
        if not isinstance(record, Mapping) or record.get(
            "sha256"
        ) != actual_sha256:
            raise AggregationLockError(
                f"Current {name} differs from the aggregation lock"
            )

    manifest = read_json(
        benchmark_manifest_path.resolve(), "benchmark manifest"
    )
    locked_variants = artifacts.get("benchmark_variants")
    if not isinstance(locked_variants, Mapping):
        raise AggregationLockError(
            "Aggregation lock has no benchmark variant contract"
        )
    for variant in REQUIRED_VARIANTS:
        current = validate_benchmark_variant(
            manifest_path=benchmark_manifest_path.resolve(),
            manifest=manifest,
            variant=variant,
        )
        locked = locked_variants.get(variant)
        if not isinstance(locked, Mapping) or locked.get(
            "sha256"
        ) != current["sha256"]:
            raise AggregationLockError(
                f"Current benchmark {variant} differs from the aggregation lock"
            )

    locked_predictions = artifacts.get("predictions")
    if not isinstance(locked_predictions, Mapping):
        raise AggregationLockError("Aggregation lock has no predictions")
    if set(prediction_paths) != set(locked_predictions):
        raise AggregationLockError(
            "Evaluation predictions differ from the aggregation lock variants"
        )
    for variant, path in prediction_paths.items():
        locked = locked_predictions.get(variant)
        if not isinstance(locked, Mapping):
            raise AggregationLockError(
                f"Aggregation lock has no {variant} prediction"
            )
        if evaluator.sha256_file(path.resolve()) != locked.get("sha256"):
            raise AggregationLockError(
                f"Current {variant} predictions differ from the aggregation lock"
            )
        sidecar = locked.get("sidecar")
        if not isinstance(sidecar, Mapping):
            raise AggregationLockError(
                f"Aggregation lock has no {variant} sidecar"
            )
        sidecar_path = Path(str(sidecar.get("path", ""))).resolve()
        if not sidecar_path.is_file() or evaluator.sha256_file(
            sidecar_path
        ) != sidecar.get("sha256"):
            raise AggregationLockError(
                f"Current {variant} sidecar differs from the aggregation lock"
            )
        sidecar_payload = read_json(
            sidecar_path, f"{variant} prediction sidecar"
        )
        if sidecar_payload.get("status") != "complete":
            raise AggregationLockError(
                f"Current {variant} sidecar status is not complete"
            )
        if sidecar_payload.get("output_sha256") != locked.get("sha256"):
            raise AggregationLockError(
                f"Current {variant} sidecar output hash is inconsistent"
            )

    implementation = lock.get("implementation")
    lock_script_record = (
        implementation.get("lock_script")
        if isinstance(implementation, Mapping)
        else None
    )
    lock_script_path = Path(__file__).resolve()
    if not isinstance(lock_script_record, Mapping) or lock_script_record.get(
        "sha256"
    ) != evaluator.sha256_file(lock_script_path):
        raise AggregationLockError(
            "Current lock implementation differs from the aggregation lock"
        )
    evaluator_record = (
        implementation.get("evaluator")
        if isinstance(implementation, Mapping)
        else None
    )
    evaluator_path = Path(evaluator.__file__).resolve()
    if not isinstance(evaluator_record, Mapping) or evaluator_record.get(
        "sha256"
    ) != evaluator.sha256_file(evaluator_path):
        raise AggregationLockError(
            "Current evaluator implementation differs from the aggregation lock"
        )
    return {
        "path": str(lock_path),
        "sha256": evaluator.sha256_file(lock_path),
        "lock_version": lock["lock_version"],
        "status": lock["status"],
        "model_name": lock.get("model_name"),
        "selected_chunk_aggregation": lock["selected_chunk_aggregation"],
        "selected_metrics": lock.get("selected_metrics"),
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate text-ablation prediction provenance and freeze the "
            "development-selected full-text chunk reducer."
        )
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument(
        "--benchmark-manifest",
        type=Path,
        default=DEFAULT_BENCHMARK_MANIFEST,
    )
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        metavar="VARIANT=PATH",
    )
    parser.add_argument(
        "--sidecar",
        action="append",
        default=[],
        metavar="VARIANT=PATH",
        help="Optional prediction-sidecar override; default is PATH.manifest.json.",
    )
    parser.add_argument(
        "--development-report",
        action="append",
        required=True,
        metavar="REDUCER=PATH",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    predictions = parse_named_paths(
        args.prediction,
        label="prediction",
        allowed=evaluator.VARIANT_ORDER,
    )
    sidecars = parse_named_paths(
        args.sidecar,
        label="sidecar",
        allowed=evaluator.VARIANT_ORDER,
    )
    development_reports = parse_named_paths(
        args.development_report,
        label="development report",
        allowed=evaluator.CHUNK_AGGREGATIONS,
    )
    lock = build_aggregation_lock(
        model_name=args.model_name,
        benchmark_manifest_path=args.benchmark_manifest,
        reference_path=args.reference,
        schema_path=args.schema,
        prediction_paths=predictions,
        sidecar_paths=sidecars,
        development_report_paths=development_reports,
    )
    write_json(args.output, lock)
    print(
        f"Locked {args.model_name} chunk aggregation to "
        f"{lock['selected_chunk_aggregation']} at {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
