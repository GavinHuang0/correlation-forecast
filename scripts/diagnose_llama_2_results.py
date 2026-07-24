"""Diagnose saved Llama 2 predictions without loading either model.

The diagnostic is intentionally independent of the frozen extractor and
evaluator.  It uses only Python's standard library and reads immutable saved
artifacts to:

* expose option-order collapse in model-origin classification passes;
* audit saved truncation flags and extraction-manifest prompt maxima; and
* bootstrap paired Llama-minus-FLAN aggregate metric differences by article.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LLAMA_PREDICTIONS = (
    REPOSITORY_ROOT / "outputs" / "llama_2" / "v1_0" / "evaluation_predictions.jsonl"
)
DEFAULT_FLAN_PREDICTIONS = (
    REPOSITORY_ROOT
    / "outputs"
    / "flan_t5"
    / "v0_4"
    / "flan_t5_v0_4_evaluation_hybrid.jsonl"
)
DEFAULT_COARSE_REFERENCE = (
    REPOSITORY_ROOT / "annotations" / "chatgpt_5_6_sol_reference_coarse_v0_2.jsonl"
)
DEFAULT_SCHEMA = REPOSITORY_ROOT / "config" / "news_feature_schema_coarse.json"
DEFAULT_EVALUATION_INPUTS = (
    REPOSITORY_ROOT
    / "outputs"
    / "llama_2"
    / "shared"
    / "benchmark_300"
    / "evaluation_inputs.jsonl"
)
DEFAULT_BOOTSTRAP_REPS = 20_000
DEFAULT_BOOTSTRAP_SEED = 20_260_724
DEFAULT_EVALUATION_COUNT = 228
DEFAULT_RELEVANT_COUNT = 174

FALLBACK_BY_FIELD = {
    "shock_scope": "unclear",
    "event_family": "other_or_unclear",
    "information_status": "unclear",
    "directional_alignment": "unclear",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose saved Llama 2 option-order behavior and compute paired "
            "bootstrap intervals against the FLAN-T5 v0.4 hybrid."
        )
    )
    parser.add_argument(
        "--llama-predictions", type=Path, default=DEFAULT_LLAMA_PREDICTIONS
    )
    parser.add_argument(
        "--flan-predictions", type=Path, default=DEFAULT_FLAN_PREDICTIONS
    )
    parser.add_argument(
        "--coarse-reference", type=Path, default=DEFAULT_COARSE_REFERENCE
    )
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument(
        "--evaluation-inputs", type=Path, default=DEFAULT_EVALUATION_INPUTS
    )
    parser.add_argument(
        "--llama-manifest",
        type=Path,
        help="Defaults to <llama-predictions>.manifest.json when that file exists.",
    )
    parser.add_argument(
        "--flan-manifest",
        type=Path,
        help="Defaults to <flan-predictions>.manifest.json when that file exists.",
    )
    parser.add_argument(
        "--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--expected-evaluation-count", type=int, default=DEFAULT_EVALUATION_COUNT
    )
    parser.add_argument(
        "--expected-relevant-count", type=int, default=DEFAULT_RELEVANT_COUNT
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.bootstrap_reps < 1:
        parser.error("--bootstrap-reps must be at least 1")
    if args.expected_evaluation_count < 1:
        parser.error("--expected-evaluation-count must be at least 1")
    if args.expected_relevant_count < 1:
        parser.error("--expected-relevant-count must be at least 1")
    return args


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(value)
    return records


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def records_by_id(
    records: Sequence[Mapping[str, Any]], *, source_name: str
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(records, start=1):
        article_id = record.get("article_id")
        if not isinstance(article_id, str) or not article_id:
            raise ValueError(f"{source_name} row {index} has no valid article_id")
        if article_id in result:
            raise ValueError(f"{source_name} contains duplicate ID {article_id!r}")
        result[article_id] = record
    return result


def semantic_fields(schema: Mapping[str, Any]) -> list[str]:
    fields = schema.get("flan_core_fields")
    closed = schema.get("closed_label_fields")
    if not isinstance(fields, list) or not fields:
        raise ValueError("Schema has no nonempty flan_core_fields list")
    if not isinstance(closed, dict):
        raise ValueError("Schema has no closed_label_fields object")
    for field in fields:
        labels = closed.get(field)
        if (
            not isinstance(field, str)
            or not isinstance(labels, list)
            or len(labels) < 2
            or len(labels) != len(set(labels))
            or not all(isinstance(label, str) and label for label in labels)
        ):
            raise ValueError(f"Schema has invalid closed labels for {field!r}")
    return list(fields)


def ordered_distribution(
    values: Sequence[str], allowed_labels: Sequence[str]
) -> dict[str, int]:
    counts = Counter(values)
    unexpected = set(counts) - set(allowed_labels)
    if unexpected:
        raise ValueError(f"Predictions contain labels outside the schema: {unexpected}")
    return {label: counts[label] for label in allowed_labels}


def decoder_diagnostics(
    predictions: Sequence[Mapping[str, Any]],
    schema: Mapping[str, Any],
    fields: Sequence[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in fields:
        allowed = schema["closed_label_fields"][field]
        current: list[str] = []
        canonical: list[str] = []
        reversed_predictions: list[str] = []
        for prediction in predictions:
            origins = prediction.get("label_origins")
            if not isinstance(origins, dict) or origins.get(field) != "model":
                continue
            labels = prediction.get("labels")
            passes = prediction.get("passes")
            if not isinstance(labels, dict) or not isinstance(passes, dict):
                raise ValueError(
                    f"{prediction.get('article_id')}: missing labels/passes for {field}"
                )
            pass_record = passes.get(field)
            if not isinstance(pass_record, dict):
                raise ValueError(
                    f"{prediction.get('article_id')}: missing model pass for {field}"
                )
            values = (
                labels.get(field),
                pass_record.get("canonical_prediction"),
                pass_record.get("reversed_prediction"),
            )
            if not all(isinstance(value, str) and value in allowed for value in values):
                raise ValueError(
                    f"{prediction.get('article_id')}: incomplete/invalid decoder "
                    f"diagnostics for model-origin {field}"
                )
            current.append(values[0])
            canonical.append(values[1])
            reversed_predictions.append(values[2])

        agreements = sum(
            first == second
            for first, second in zip(canonical, reversed_predictions)
        )
        count = len(current)
        result[field] = {
            "model_origin_pass_count": count,
            "current_prediction_distribution": ordered_distribution(current, allowed),
            "canonical_prediction_distribution": ordered_distribution(
                canonical, allowed
            ),
            "reversed_prediction_distribution": ordered_distribution(
                reversed_predictions, allowed
            ),
            "canonical_reversed_agreement_count": agreements,
            "canonical_reversed_disagreement_count": count - agreements,
            "canonical_reversed_agreement_rate": (
                agreements / count if count else None
            ),
        }
    return result


def truncation_diagnostics(
    predictions: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
    manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    pass_count = 0
    truncated_count = 0
    missing_flag_count = 0
    rows_with_truncation: set[str] = set()
    for prediction in predictions:
        article_id = str(prediction.get("article_id"))
        passes = prediction.get("passes")
        if not isinstance(passes, dict):
            raise ValueError(f"{article_id}: missing passes object")
        for field in fields:
            pass_record = passes.get(field)
            if not isinstance(pass_record, dict):
                raise ValueError(f"{article_id}: missing pass for {field}")
            pass_count += 1
            flag = pass_record.get("input_truncated")
            if flag is True:
                truncated_count += 1
                rows_with_truncation.add(article_id)
            elif flag is not False:
                missing_flag_count += 1

    manifest_summary: dict[str, Any] | None = None
    if manifest is not None:
        maxima = manifest.get("preflight_max_prompt_tokens_by_field")
        if not isinstance(maxima, dict):
            maxima = None
        maximum = (
            max(
                (
                    int(value)
                    for value in maxima.values()
                    if isinstance(value, int) and not isinstance(value, bool)
                ),
                default=None,
            )
            if maxima is not None
            else None
        )
        max_input_tokens = manifest.get("max_input_tokens")
        manifest_summary = {
            "status": manifest.get("status"),
            "max_input_tokens": max_input_tokens,
            "preflight_max_prompt_tokens_by_field": maxima,
            "maximum_preflight_prompt_tokens": maximum,
            "minimum_prompt_headroom_tokens": (
                max_input_tokens - maximum
                if isinstance(max_input_tokens, int) and maximum is not None
                else None
            ),
        }

    return {
        "record_count": len(predictions),
        "pass_count": pass_count,
        "truncated_pass_count": truncated_count,
        "missing_input_truncated_flag_count": missing_flag_count,
        "rows_with_truncation": sorted(rows_with_truncation),
        "all_recorded_passes_untruncated": (
            truncated_count == 0 and missing_flag_count == 0
        ),
        "extraction_manifest": manifest_summary,
    }


def _fallback_label(field: str, allowed: Sequence[str]) -> str:
    preferred = FALLBACK_BY_FIELD.get(field)
    if preferred in allowed:
        return preferred
    return allowed[-1]


def _resolved_prediction_label(
    prediction: Mapping[str, Any], field: str, allowed: Sequence[str]
) -> str:
    labels = prediction.get("labels")
    if not isinstance(labels, dict):
        raise ValueError(f"{prediction.get('article_id')}: missing labels object")
    value = labels.get(field)
    if value is None:
        return _fallback_label(field, allowed)
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(
            f"{prediction.get('article_id')}: invalid prediction {field}={value!r}"
        )
    return value


def encode_metric_rows(
    *,
    relevant_ids: Sequence[str],
    references_by_id: Mapping[str, Mapping[str, Any]],
    llama_by_id: Mapping[str, Mapping[str, Any]],
    flan_by_id: Mapping[str, Mapping[str, Any]],
    schema: Mapping[str, Any],
    fields: Sequence[str],
) -> list[tuple[list[int], list[int], list[int], int]]:
    encoded: list[tuple[list[int], list[int], list[int], int]] = []
    for field in fields:
        allowed = schema["closed_label_fields"][field]
        label_to_index = {label: index for index, label in enumerate(allowed)}
        truths: list[int] = []
        llama: list[int] = []
        flan: list[int] = []
        for article_id in relevant_ids:
            reference_labels = references_by_id[article_id].get("labels")
            if not isinstance(reference_labels, dict):
                raise ValueError(f"{article_id}: coarse reference has no labels")
            truth = reference_labels.get(field)
            if truth not in label_to_index:
                raise ValueError(
                    f"{article_id}: invalid coarse-reference {field}={truth!r}"
                )
            truths.append(label_to_index[truth])
            llama.append(
                label_to_index[
                    _resolved_prediction_label(
                        llama_by_id[article_id], field, allowed
                    )
                ]
            )
            flan.append(
                label_to_index[
                    _resolved_prediction_label(
                        flan_by_id[article_id], field, allowed
                    )
                ]
            )
        encoded.append((truths, llama, flan, len(allowed)))
    return encoded


def paired_metrics_for_indices(
    encoded_fields: Sequence[tuple[list[int], list[int], list[int], int]],
    indices: Sequence[int],
) -> tuple[float, float, float, float]:
    if not indices:
        raise ValueError("Metric sample must contain at least one article")
    llama_accuracy_sum = 0.0
    flan_accuracy_sum = 0.0
    llama_macro_f1_sum = 0.0
    flan_macro_f1_sum = 0.0
    sample_size = len(indices)

    for truths, llama, flan, label_count in encoded_fields:
        truth_counts = [0] * label_count
        llama_counts = [0] * label_count
        flan_counts = [0] * label_count
        llama_true_positive = [0] * label_count
        flan_true_positive = [0] * label_count
        llama_correct = 0
        flan_correct = 0
        for index in indices:
            truth = truths[index]
            llama_guess = llama[index]
            flan_guess = flan[index]
            truth_counts[truth] += 1
            llama_counts[llama_guess] += 1
            flan_counts[flan_guess] += 1
            if llama_guess == truth:
                llama_correct += 1
                llama_true_positive[truth] += 1
            if flan_guess == truth:
                flan_correct += 1
                flan_true_positive[truth] += 1

        llama_accuracy_sum += llama_correct / sample_size
        flan_accuracy_sum += flan_correct / sample_size
        observed_labels = [
            label for label in range(label_count) if truth_counts[label] > 0
        ]
        if not observed_labels:
            raise ValueError("Metric field has no observed reference labels")
        llama_macro_f1_sum += sum(
            (
                2.0 * llama_true_positive[label]
                / (truth_counts[label] + llama_counts[label])
                if truth_counts[label] + llama_counts[label]
                else 0.0
            )
            for label in observed_labels
        ) / len(observed_labels)
        flan_macro_f1_sum += sum(
            (
                2.0 * flan_true_positive[label]
                / (truth_counts[label] + flan_counts[label])
                if truth_counts[label] + flan_counts[label]
                else 0.0
            )
            for label in observed_labels
        ) / len(observed_labels)

    field_count = len(encoded_fields)
    if not field_count:
        raise ValueError("No semantic fields were supplied")
    return (
        llama_accuracy_sum / field_count,
        flan_accuracy_sum / field_count,
        llama_macro_f1_sum / field_count,
        flan_macro_f1_sum / field_count,
    )


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile of an empty sample")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("Percentile probability must be between zero and one")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def paired_bootstrap(
    encoded_fields: Sequence[tuple[list[int], list[int], list[int], int]],
    *,
    article_count: int,
    reps: int,
    seed: int,
) -> dict[str, Any]:
    if article_count < 1:
        raise ValueError("Bootstrap requires at least one article")
    point = paired_metrics_for_indices(encoded_fields, list(range(article_count)))
    llama_accuracy, flan_accuracy, llama_macro_f1, flan_macro_f1 = point
    accuracy_deltas: list[float] = []
    macro_f1_deltas: list[float] = []
    rng = random.Random(seed)
    for _ in range(reps):
        indices = [rng.randrange(article_count) for _ in range(article_count)]
        (
            replicate_llama_accuracy,
            replicate_flan_accuracy,
            replicate_llama_macro_f1,
            replicate_flan_macro_f1,
        ) = paired_metrics_for_indices(encoded_fields, indices)
        accuracy_deltas.append(
            replicate_llama_accuracy - replicate_flan_accuracy
        )
        macro_f1_deltas.append(
            replicate_llama_macro_f1 - replicate_flan_macro_f1
        )

    def interval(values: Sequence[float], point_estimate: float) -> dict[str, Any]:
        return {
            "point_estimate": point_estimate,
            "ci_95": [
                percentile(values, 0.025),
                percentile(values, 0.975),
            ],
            "bootstrap_mean": sum(values) / len(values),
            "replicates_below_zero_fraction": (
                sum(value < 0.0 for value in values) / len(values)
            ),
        }

    return {
        "method": "paired article bootstrap with percentile interval",
        "sampling_unit": "reference-relevant article",
        "seed": seed,
        "replicates": reps,
        "article_count": article_count,
        "point_estimates": {
            "llama_2": {
                "mean_field_accuracy": llama_accuracy,
                "mean_macro_f1": llama_macro_f1,
            },
            "flan_t5_v0_4_hybrid": {
                "mean_field_accuracy": flan_accuracy,
                "mean_macro_f1": flan_macro_f1,
            },
        },
        "llama_minus_flan": {
            "mean_field_accuracy": interval(
                accuracy_deltas, llama_accuracy - flan_accuracy
            ),
            "mean_macro_f1": interval(
                macro_f1_deltas, llama_macro_f1 - flan_macro_f1
            ),
        },
    }


def build_report(
    *,
    evaluation_inputs: Sequence[Mapping[str, Any]],
    llama_predictions: Sequence[Mapping[str, Any]],
    flan_predictions: Sequence[Mapping[str, Any]],
    coarse_reference: Sequence[Mapping[str, Any]],
    schema: Mapping[str, Any],
    bootstrap_reps: int,
    seed: int,
    llama_manifest: Mapping[str, Any] | None = None,
    flan_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    fields = semantic_fields(schema)
    inputs_by_id = records_by_id(evaluation_inputs, source_name="evaluation inputs")
    llama_by_id = records_by_id(llama_predictions, source_name="Llama predictions")
    flan_by_id = records_by_id(flan_predictions, source_name="FLAN predictions")
    reference_by_id = records_by_id(coarse_reference, source_name="coarse reference")
    ordered_ids = [str(record["article_id"]) for record in evaluation_inputs]
    input_ids = set(ordered_ids)
    for name, values in (
        ("Llama predictions", set(llama_by_id)),
        ("FLAN predictions", set(flan_by_id)),
    ):
        if values != input_ids:
            raise ValueError(
                f"{name} IDs must exactly match evaluation inputs: "
                f"missing={sorted(input_ids - values)[:10]!r}, "
                f"unexpected={sorted(values - input_ids)[:10]!r}"
            )
    if not input_ids <= set(reference_by_id):
        raise ValueError("Coarse reference is missing evaluation input IDs")

    relevant_ids = [
        article_id
        for article_id in ordered_ids
        if reference_by_id[article_id].get("semantic_applicable") is True
    ]
    encoded = encode_metric_rows(
        relevant_ids=relevant_ids,
        references_by_id=reference_by_id,
        llama_by_id=llama_by_id,
        flan_by_id=flan_by_id,
        schema=schema,
        fields=fields,
    )
    return {
        "title": "Saved Llama 2 decoder and paired-performance diagnostic",
        "reference_type": "deterministically coarsened GPT-5.6 Sol silver annotations",
        "warning": (
            "This measures agreement with silver labels, not objective "
            "human-ground-truth accuracy."
        ),
        "evaluation_record_count": len(evaluation_inputs),
        "reference_relevant_count": len(relevant_ids),
        "semantic_fields": fields,
        "decoder_diagnostics_on_model_origin_passes": {
            "llama_2": decoder_diagnostics(llama_predictions, schema, fields),
            "flan_t5_v0_4_hybrid": decoder_diagnostics(
                flan_predictions, schema, fields
            ),
        },
        "truncation_diagnostics": {
            "llama_2": truncation_diagnostics(
                llama_predictions, fields, llama_manifest
            ),
            "flan_t5_v0_4_hybrid": truncation_diagnostics(
                flan_predictions, fields, flan_manifest
            ),
        },
        "paired_bootstrap_95_percent_intervals": paired_bootstrap(
            encoded,
            article_count=len(relevant_ids),
            reps=bootstrap_reps,
            seed=seed,
        ),
    }


def _adjacent_manifest(predictions_path: Path) -> Path:
    return predictions_path.with_suffix(predictions_path.suffix + ".manifest.json")


def _load_optional_manifest(
    explicit_path: Path | None, predictions_path: Path
) -> tuple[Mapping[str, Any] | None, Path | None]:
    path = explicit_path
    if path is None:
        candidate = _adjacent_manifest(predictions_path)
        path = candidate if candidate.is_file() else None
    if path is None:
        return None, None
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: manifest must be a JSON object")
    return value, path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite")
    for path in (
        args.llama_predictions,
        args.flan_predictions,
        args.coarse_reference,
        args.schema,
        args.evaluation_inputs,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    evaluation_inputs = read_jsonl(args.evaluation_inputs)
    llama_predictions = read_jsonl(args.llama_predictions)
    flan_predictions = read_jsonl(args.flan_predictions)
    coarse_reference = read_jsonl(args.coarse_reference)
    schema = read_json(args.schema)
    if len(evaluation_inputs) != args.expected_evaluation_count:
        raise ValueError(
            f"Expected {args.expected_evaluation_count} evaluation records, "
            f"found {len(evaluation_inputs)}"
        )

    llama_manifest, llama_manifest_path = _load_optional_manifest(
        args.llama_manifest, args.llama_predictions
    )
    flan_manifest, flan_manifest_path = _load_optional_manifest(
        args.flan_manifest, args.flan_predictions
    )
    report = build_report(
        evaluation_inputs=evaluation_inputs,
        llama_predictions=llama_predictions,
        flan_predictions=flan_predictions,
        coarse_reference=coarse_reference,
        schema=schema,
        bootstrap_reps=args.bootstrap_reps,
        seed=args.seed,
        llama_manifest=llama_manifest,
        flan_manifest=flan_manifest,
    )
    if report["reference_relevant_count"] != args.expected_relevant_count:
        raise ValueError(
            f"Expected {args.expected_relevant_count} reference-relevant records, "
            f"found {report['reference_relevant_count']}"
        )
    report["artifacts"] = {
        "evaluation_inputs": {
            "path": str(args.evaluation_inputs),
            "sha256": sha256_file(args.evaluation_inputs),
        },
        "llama_predictions": {
            "path": str(args.llama_predictions),
            "sha256": sha256_file(args.llama_predictions),
        },
        "llama_manifest": (
            {
                "path": str(llama_manifest_path),
                "sha256": sha256_file(llama_manifest_path),
            }
            if llama_manifest_path is not None
            else None
        ),
        "flan_predictions": {
            "path": str(args.flan_predictions),
            "sha256": sha256_file(args.flan_predictions),
        },
        "flan_manifest": (
            {
                "path": str(flan_manifest_path),
                "sha256": sha256_file(flan_manifest_path),
            }
            if flan_manifest_path is not None
            else None
        ),
        "coarse_reference": {
            "path": str(args.coarse_reference),
            "sha256": sha256_file(args.coarse_reference),
        },
        "schema": {
            "path": str(args.schema),
            "sha256": sha256_file(args.schema),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"Wrote saved-result diagnostic for {len(evaluation_inputs)} records "
        f"to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
