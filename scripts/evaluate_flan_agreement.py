from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


INVALID = "__invalid__"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FLAN-T5 agreement with GPT-5.6 Sol silver labels.")
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--inputs", required=True, type=Path, help="Blinded inputs for evidence validation")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Extractor manifest; defaults to <predictions>.manifest.json",
    )
    parser.add_argument("--schema", default=Path("config/news_feature_schema.json"), type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--allow-subset",
        action="store_true",
        help="Evaluate an explicit prediction subset, such as a --limit smoke run.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            records.append(record)
    return records


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def closed_label_metrics(
    reference: list[str], predictions: list[str | None], allowed_classes: list[str]
) -> dict[str, Any]:
    predicted = [value if value in allowed_classes else INVALID for value in predictions]
    all_classes = allowed_classes + ([INVALID] if INVALID in predicted else [])
    matrix = {
        truth: {guess: 0 for guess in all_classes}
        for truth in allowed_classes
    }
    for truth, guess in zip(reference, predicted):
        matrix[truth][guess] += 1

    per_class: dict[str, Any] = {}
    observed_f1: list[float] = []
    observed_precision: list[float] = []
    observed_recall: list[float] = []
    for label in allowed_classes:
        tp = matrix[label][label]
        fp = sum(matrix[truth][label] for truth in allowed_classes if truth != label)
        fn = sum(matrix[label][guess] for guess in all_classes if guess != label)
        support = sum(matrix[label].values())
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
            observed_precision.append(precision)
            observed_recall.append(recall)
            observed_f1.append(f1)

    total = len(reference)
    correct = sum(matrix[label][label] for label in allowed_classes)
    reference_counts = Counter(reference)
    prediction_counts = Counter(predicted)
    expected = sum(
        safe_divide(reference_counts[label], total) * safe_divide(prediction_counts[label], total)
        for label in all_classes
    )
    observed = safe_divide(correct, total)
    kappa = safe_divide(observed - expected, 1 - expected)
    return {
        "count": total,
        "accuracy": observed,
        "macro_precision": safe_divide(sum(observed_precision), len(observed_precision)),
        "macro_recall": safe_divide(sum(observed_recall), len(observed_recall)),
        "macro_f1": safe_divide(sum(observed_f1), len(observed_f1)),
        "cohens_kappa": kappa,
        "invalid_prediction_count": prediction_counts[INVALID],
        "reference_distribution": dict(sorted(reference_counts.items())),
        "prediction_distribution": dict(sorted(prediction_counts.items())),
        "per_class": per_class,
        "confusion_matrix": {
            "row_labels_reference": allowed_classes,
            "column_labels_prediction": all_classes,
            "values": [[matrix[truth][guess] for guess in all_classes] for truth in allowed_classes],
        },
    }


def normalize_entity(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join("".join(character if character.isalnum() else " " for character in normalized).split())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_extractor_module() -> Any:
    path = Path(__file__).resolve().with_name("extract_flan_t5.py")
    spec = importlib.util.spec_from_file_location("flan_extractor_for_evaluation", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load extractor module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def multilabel_metrics(
    reference_sets: Iterable[set[str]], prediction_sets: Iterable[set[str]]
) -> dict[str, float | int]:
    tp = fp = fn = exact = 0
    jaccards: list[float] = []
    count = 0
    for reference, prediction in zip(reference_sets, prediction_sets):
        count += 1
        tp += len(reference & prediction)
        fp += len(prediction - reference)
        fn += len(reference - prediction)
        exact += int(reference == prediction)
        union = reference | prediction
        jaccards.append(safe_divide(len(reference & prediction), len(union)) if union else 1.0)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    return {
        "count": count,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": safe_divide(2 * precision * recall, precision + recall),
        "mean_jaccard": safe_divide(sum(jaccards), len(jaccards)),
        "exact_match_rate": safe_divide(exact, count),
        "true_positive_items": tp,
        "false_positive_items": fp,
        "false_negative_items": fn,
    }


def main() -> int:
    args = parse_args()
    schema_text = args.schema.read_text(encoding="utf-8")
    schema = json.loads(schema_text)
    reference = read_jsonl(args.reference)
    predictions = read_jsonl(args.predictions)
    inputs = read_jsonl(args.inputs)
    manifest_path = args.manifest or args.predictions.with_suffix(args.predictions.suffix + ".manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing extraction manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("Extraction manifest is not complete")
    if manifest.get("output_sha256") != sha256_file(args.predictions):
        raise ValueError("Prediction file hash does not match the extraction manifest")
    if manifest.get("schema_sha256") != hashlib.sha256(schema_text.encode("utf-8")).hexdigest():
        raise ValueError("Evaluation schema does not match the extraction manifest")
    if manifest.get("total_prediction_count") != len(predictions):
        raise ValueError("Prediction count does not match the extraction manifest")

    reference_by_id = {record["article_id"]: record for record in reference}
    prediction_by_id = {record["article_id"]: record for record in predictions}
    input_by_id = {record["article_id"]: record for record in inputs}
    for name, records, mapping in (
        ("reference", reference, reference_by_id),
        ("predictions", predictions, prediction_by_id),
        ("inputs", inputs, input_by_id),
    ):
        if len(mapping) != len(records):
            raise ValueError(f"Duplicate article_id values detected in {name}")
    reference_ids = set(reference_by_id)
    prediction_ids = set(prediction_by_id)
    input_ids = set(input_by_id)
    if args.allow_subset:
        if not prediction_ids or not prediction_ids <= reference_ids or not prediction_ids <= input_ids:
            raise ValueError("Prediction article_ids must be a non-empty subset of reference and input IDs")
    elif reference_ids != prediction_ids or reference_ids != input_ids:
        raise ValueError("Reference, prediction, and input article_id sets must match exactly")
    prediction_file_order_ids = [record["article_id"] for record in predictions]
    selected_ids_hash = hashlib.sha256(
        "\n".join(prediction_file_order_ids).encode("utf-8")
    ).hexdigest()
    if manifest.get("selected_article_ids_sha256") and manifest.get(
        "selected_article_ids_sha256"
    ) != selected_ids_hash:
        raise ValueError("Prediction file order does not match the manifest's selected article hash")
    ordered_ids = [
        record["article_id"] for record in sorted(predictions, key=lambda item: item["row_number"])
    ]

    extractor = load_extractor_module()
    prompt_version = manifest.get("prompt_version")
    if not isinstance(prompt_version, str):
        raise ValueError("Extraction manifest is missing prompt_version")
    passes = tuple(manifest.get("passes", []))
    expected_passes = extractor.passes_for(prompt_version, manifest.get("mode"))
    core_passes = extractor.core_passes_for(prompt_version)
    closed_label_decoding = manifest.get("closed_label_decoding", "generate")
    if prompt_version == extractor.LEGACY_PROMPT_VERSION and closed_label_decoding != "generate":
        raise ValueError("Legacy v0.1 predictions must use ordinary generation")
    if prompt_version == extractor.PROMPT_VERSION and closed_label_decoding not in {
        "score",
        "constrained",
        "generate",
    }:
        raise ValueError(f"Unsupported v0.2 closed-label decoder {closed_label_decoding!r}")
    if passes != expected_passes:
        raise ValueError(f"Manifest pass list {passes} is inconsistent with mode {manifest.get('mode')!r}")
    recomputed_validity: dict[str, dict[str, bool]] = {}
    validation_errors: list[str] = []
    for article_id in ordered_ids:
        input_record = input_by_id[article_id]
        reference_record = reference_by_id[article_id]
        prediction = prediction_by_id[article_id]
        if prediction.get("row_number") != input_record.get("row_number") or reference_record.get(
            "row_number"
        ) != input_record.get("row_number"):
            validation_errors.append(f"{article_id}: row_number alignment mismatch")
        expected_ticker = input_record["target"]["ticker"]
        if prediction.get("target_ticker") != expected_ticker or reference_record.get(
            "target_ticker"
        ) != expected_ticker:
            validation_errors.append(f"{article_id}: target_ticker alignment mismatch")
        if prediction.get("model_revision") != manifest.get("model_revision"):
            validation_errors.append(f"{article_id}: model revision differs from manifest")
        if prediction.get("extractor") != manifest.get("model_id"):
            validation_errors.append(f"{article_id}: model ID differs from manifest")
        if prediction.get("protocol_version") != schema["schema_version"]:
            validation_errors.append(f"{article_id}: protocol version differs from schema")

        expected_labels = extractor.empty_labels()
        parsed_passes: dict[str, dict[str, Any]] = {}
        for pass_name in passes:
            stored_pass = prediction.get("passes", {}).get(pass_name)
            if not isinstance(stored_pass, dict) or not isinstance(stored_pass.get("raw_output"), str):
                validation_errors.append(f"{article_id}: missing raw output for pass {pass_name}")
                continue
            prompt = extractor.build_prompt(
                pass_name,
                input_record,
                schema,
                expected_labels,
                prompt_version,
            )
            prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            if stored_pass.get("prompt_sha256") != prompt_hash:
                validation_errors.append(f"{article_id}: prompt hash mismatch for pass {pass_name}")
            parsed = extractor.parse_pass(
                pass_name,
                stored_pass["raw_output"],
                input_record,
                schema,
                expected_labels,
                prompt_version,
            )
            extractor.update_labels(
                expected_labels,
                pass_name,
                parsed["values"],
                prompt_version,
            )
            parsed_passes[pass_name] = parsed
            if prompt_version == extractor.PROMPT_VERSION and pass_name in core_passes:
                candidate_scores = stored_pass.get("candidate_mean_log_probabilities")
                if closed_label_decoding == "score":
                    allowed = schema["closed_label_fields"][pass_name]
                    if not isinstance(candidate_scores, dict) or set(candidate_scores) != set(allowed):
                        validation_errors.append(
                            f"{article_id}: incomplete candidate scores for pass {pass_name}"
                        )
                    elif stored_pass["raw_output"] != max(
                        allowed, key=lambda value: candidate_scores[value]
                    ):
                        validation_errors.append(
                            f"{article_id}: scored output is not the recorded candidate argmax for pass {pass_name}"
                        )
                elif candidate_scores is not None:
                    validation_errors.append(
                        f"{article_id}: unexpected candidate scores for decoder {closed_label_decoding} pass {pass_name}"
                    )
            if bool(stored_pass.get("valid")) != bool(parsed["valid"]):
                validation_errors.append(f"{article_id}: stored validity mismatch for pass {pass_name}")
            if bool(stored_pass.get("strict_format_valid")) != bool(parsed["strict_format_valid"]):
                validation_errors.append(f"{article_id}: stored strict-format mismatch for pass {pass_name}")
            if stored_pass.get("input_truncated"):
                validation_errors.append(f"{article_id}: truncated inputs are not evaluation eligible")
        if expected_labels["relevance"] == "insufficient":
            expected_labels["abstain_reason"] = "FLAN-T5 classified the supplied text as insufficient."
        if prediction.get("labels") != expected_labels:
            validation_errors.append(f"{article_id}: stored labels do not reproduce from raw pass outputs")

        primary_valid = all(parsed_passes.get(name, {}).get("valid", False) for name in core_passes)
        strict_primary = primary_valid and all(
            parsed_passes.get(name, {}).get("strict_format_valid", False)
            for name in core_passes
        )
        all_requested = all(parsed_passes.get(name, {}).get("valid", False) for name in passes)
        recomputed = {
            "primary_closed_labels_valid": primary_valid,
            "primary_strict_format_valid": strict_primary,
            "all_requested_passes_valid": all_requested,
        }
        if prediction.get("validity") != recomputed:
            validation_errors.append(f"{article_id}: stored record validity does not match recomputation")
        recomputed_validity[article_id] = recomputed

    if validation_errors:
        raise ValueError(
            f"Prediction integrity validation failed with {len(validation_errors)} error(s): "
            + "; ".join(validation_errors[:20])
        )

    closed_metrics: dict[str, Any] = {}
    for field, allowed in schema["closed_label_fields"].items():
        truths = [reference_by_id[article_id]["labels"][field] for article_id in ordered_ids]
        guesses = [prediction_by_id[article_id].get("labels", {}).get(field) for article_id in ordered_ids]
        closed_metrics[field] = closed_label_metrics(truths, guesses, allowed)

    extended_available = manifest.get("mode") == "full"
    channel_metrics: dict[str, Any]
    entity_metrics: dict[str, Any]
    if extended_available:
        channel_reference = [
            set(reference_by_id[article_id]["labels"]["transmission_channels"])
            for article_id in ordered_ids
        ]
        channel_predictions = [
            set(prediction_by_id[article_id].get("labels", {}).get("transmission_channels") or [])
            for article_id in ordered_ids
        ]
        channel_metrics = {"available": True, **multilabel_metrics(channel_reference, channel_predictions)}
        entity_metrics = {}
        for field in ("affected_companies", "affected_sectors"):
            truth_sets = []
            guess_sets = []
            for article_id in ordered_ids:
                truth = {
                    normalized
                    for value in reference_by_id[article_id]["labels"][field]
                    if (normalized := normalize_entity(value))
                }
                guess = {
                    normalized
                    for value in (prediction_by_id[article_id].get("labels", {}).get(field) or [])
                    if isinstance(value, str) and (normalized := normalize_entity(value))
                }
                truth_sets.append(truth)
                guess_sets.append(guess)
            entity_metrics[field] = {"available": True, **multilabel_metrics(truth_sets, guess_sets)}
    else:
        unavailable = {"available": False, "reason": "Extractor was run in core mode"}
        channel_metrics = unavailable
        entity_metrics = {
            "affected_companies": unavailable,
            "affected_sectors": unavailable,
        }

    valid_primary = 0
    strict_primary = 0
    valid_full = 0
    evidence_nonempty = 0
    evidence_exact = 0
    required_evidence = 0
    required_evidence_present = 0
    required_evidence_exact = 0
    for article_id in ordered_ids:
        prediction = prediction_by_id[article_id]
        validity = recomputed_validity[article_id]
        valid_primary += int(bool(validity.get("primary_closed_labels_valid")))
        strict_primary += int(bool(validity.get("primary_strict_format_valid")))
        valid_full += int(bool(validity.get("all_requested_passes_valid")))
        source = f"{input_by_id[article_id]['headline']}\n{input_by_id[article_id]['article_text']}"
        labels = prediction.get("labels", {})
        evidence = labels.get("evidence") or {}
        for snippet in evidence.values():
            if snippet:
                evidence_nonempty += 1
                evidence_exact += int(snippet in source)
        required_fields = []
        if labels.get("event_scope") not in {None, "unclear"}:
            required_fields.append("scope")
        if {labels.get("target_direction"), labels.get("sector_direction")} & {
            "positive",
            "negative",
            "neutral",
            "mixed",
        }:
            required_fields.append("direction")
        if labels.get("explicit_surprise") in {"positive", "negative", "mixed"}:
            required_fields.append("surprise")
        for field in required_fields:
            required_evidence += 1
            snippet = evidence.get(field, "")
            required_evidence_present += int(bool(snippet))
            required_evidence_exact += int(bool(snippet) and snippet in source)

    total = len(ordered_ids)
    report = {
        "reference_type": "GPT-5.6 Sol silver annotations",
        "record_count": total,
        "closed_label_metrics": closed_metrics,
        "extraction_manifest": {
            "path": str(manifest_path),
            "model_id": manifest.get("model_id"),
            "model_revision": manifest.get("model_revision"),
            "model_files_sha256": manifest.get("model_files_sha256"),
            "prompt_version": prompt_version,
            "closed_label_decoding": closed_label_decoding,
            "mode": manifest.get("mode"),
            "device": manifest.get("device"),
            "precision": manifest.get("precision"),
        },
        "transmission_channel_metrics": channel_metrics,
        "entity_metrics": entity_metrics,
        "output_validity": {
            "format_validity_enforced_by_decoder": closed_label_decoding in {"score", "constrained"},
            "primary_closed_labels_valid_rate": safe_divide(valid_primary, total),
            "primary_strict_format_rate": safe_divide(strict_primary, total),
            "all_requested_passes_valid_rate": safe_divide(valid_full, total),
            "evidence_metrics_available": extended_available,
            "nonempty_evidence_exact_substring_rate": (
                safe_divide(evidence_exact, evidence_nonempty) if extended_available else None
            ),
            "nonempty_evidence_count": evidence_nonempty if extended_available else None,
            "required_evidence_coverage": (
                safe_divide(required_evidence_present, required_evidence) if extended_available else None
            ),
            "required_evidence_exact_rate": (
                safe_divide(required_evidence_exact, required_evidence) if extended_available else None
            ),
            "required_evidence_slot_count": required_evidence if extended_available else None,
        },
        "go_no_go_thresholds": {
            "primary_valid_output_rate": 0.98,
            "relevance_macro_f1": 0.85,
            "event_scope_macro_f1": 0.80,
            "target_direction_macro_f1": 0.75,
            "sector_direction_macro_f1": 0.75,
            "peer_effect_macro_f1": 0.70,
            "event_type_macro_f1": 0.70,
        },
    }
    report["threshold_results"] = {
        "primary_valid_output_rate": report["output_validity"]["primary_closed_labels_valid_rate"] >= 0.98,
        "relevance_macro_f1": closed_metrics["relevance"]["macro_f1"] >= 0.85,
        "event_scope_macro_f1": closed_metrics["event_scope"]["macro_f1"] >= 0.80,
        "target_direction_macro_f1": closed_metrics["target_direction"]["macro_f1"] >= 0.75,
        "sector_direction_macro_f1": closed_metrics["sector_direction"]["macro_f1"] >= 0.75,
        "peer_effect_macro_f1": closed_metrics["peer_effect"]["macro_f1"] >= 0.70,
        "event_type_macro_f1": closed_metrics["event_type"]["macro_f1"] >= 0.70,
    }
    report["threshold_notes"] = {
        "primary_valid_output_rate": (
            "Schema validity is enforced by the configured closed-label decoder and is not an independent "
            "measure of instruction following."
            if closed_label_decoding in {"score", "constrained"}
            else "Schema validity is observed from unconstrained model generation."
        ),
        "semantic_thresholds": "Semantic agreement thresholds remain applicable under every decoder.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote agreement report for {total} articles to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
