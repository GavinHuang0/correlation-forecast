from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

SPEC = importlib.util.spec_from_file_location(
    "evaluate_llama_2_coarse_for_test",
    SCRIPTS / "evaluate_llama_2_coarse.py",
)
assert SPEC is not None and SPEC.loader is not None
evaluate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluate)

import coarse_news_features as coarse  # noqa: E402
import extract_flan_t5 as base  # noqa: E402


def schema() -> dict:
    return coarse.load_schema(ROOT / "config" / "news_feature_schema_coarse.json")


def input_record(index: int) -> dict:
    return {
        "row_number": index,
        "article_id": f"article-{index}",
        "time_published_utc": "2024-05-01T12:00:00Z",
        "source": "Example Wire",
        "headline": f"AMD announced update {index}",
        "article_text": "Advanced Micro Devices announced an update.",
        "vendor_tickers": ["AMD"],
        "target": {
            "company": "Advanced Micro Devices",
            "ticker": "AMD",
            "sector": "Semiconductors",
            "sector_benchmark": "SOXX",
            "known_sector_peers": ["NVDA", "INTC"],
        },
    }


LABELS_ONE = {
    "shock_scope": "idiosyncratic",
    "event_family": "product_demand",
    "information_status": "confirmed",
    "directional_alignment": "single_firm_only",
}
LABELS_TWO = {
    "shock_scope": "common",
    "event_family": "macro_market",
    "information_status": "anticipated",
    "directional_alignment": "common_direction_unclear",
}


def prediction(record: dict, labels: dict | None = None) -> dict:
    deterministic = coarse.deterministic_features(record)
    selected_labels = dict(labels) if labels is not None else {
        field: None for field in coarse.COARSE_FIELDS
    }
    applicable = labels is not None
    origins = {
        field: ("model" if applicable else "deterministic_gate")
        for field in coarse.COARSE_FIELDS
    }
    return {
        "row_number": record["row_number"],
        "article_id": record["article_id"],
        "target_ticker": "AMD",
        "extractor": evaluate.MODEL_ID,
        "model_revision": evaluate.MODEL_REVISION,
        "prompt_version": evaluate.PROMPT_VERSION,
        "semantic_applicable": applicable,
        "deterministic_features": deterministic,
        "labels": selected_labels,
        "label_origins": origins,
        "passes": {
            field: {
                "value": selected_labels[field],
                "origin": origins[field],
                "valid": True,
                "input_truncated": False,
            }
            for field in coarse.COARSE_FIELDS
        },
        "validity": {"all_fields_resolved_when_applicable": True},
    }


def coarse_reference(record: dict, labels: dict) -> dict:
    return {
        "row_number": record["row_number"],
        "article_id": record["article_id"],
        "target_ticker": "AMD",
        "semantic_applicable": True,
        "labels": dict(labels),
    }


def fine_reference(record: dict) -> dict:
    return {
        "row_number": record["row_number"],
        "article_id": record["article_id"],
        "target_ticker": "AMD",
        "labels": {"explicit_surprise": "none"},
    }


def baseline_summary(count: int) -> dict:
    return {
        "record_count": count,
        "reference_relevant_count": count,
        "hybrid_protocol": {
            "field_metrics": {
                field: {"accuracy": 0.5, "macro_f1": 0.5}
                for field in coarse.COARSE_FIELDS
            }
        },
    }


def prediction_manifest() -> dict:
    return {
        "manifest_version": evaluate.llama_extractor.MANIFEST_VERSION,
        "status": "complete",
        "model_id": evaluate.MODEL_ID,
        "model_revision": evaluate.MODEL_REVISION,
        "prompt_version": evaluate.PROMPT_VERSION,
        "conservative_model_data_cutoff": (
            evaluate.llama_extractor.CONSERVATIVE_DATA_CUTOFF
        ),
        "decoding": "order_averaged_letter_score",
        "prompt_profile": "zero_shot",
        "device": "cuda",
        "precision": "float16",
        "batch_size": 1,
        "max_input_tokens": evaluate.llama_extractor.MODEL_CONTEXT_WINDOW,
        "attention_implementation": "eager",
        "quantization": {
            "method": "nf4",
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "compute_dtype": "float16",
        },
        "chat_template_source": (
            "frozen official Llama 2 single-turn [INST] format"
        ),
        "chat_template_sha256": evaluate.llama_extractor._canonical_json_sha256(
            evaluate.llama_extractor.LLAMA_2_SINGLE_TURN_CHAT_TEMPLATE
        ),
        "extractor_source_sha256": base.sha256_file(
            Path(evaluate.llama_extractor.__file__).resolve()
        ),
        "hierarchy_module_sha256": base.sha256_file(
            Path(evaluate.llama_extractor.flan_coarse.__file__).resolve()
        ),
    }


class LlamaEvaluationMetricsTests(unittest.TestCase):
    def test_perfect_predictions_beat_flan_baseline(self) -> None:
        inputs = [input_record(1), input_record(2)]
        predictions = [
            prediction(inputs[0], LABELS_ONE),
            prediction(inputs[1], LABELS_TWO),
        ]
        report = evaluate.build_evaluation_report(
            inputs=inputs,
            predictions=predictions,
            coarse_reference=[
                coarse_reference(inputs[0], LABELS_ONE),
                coarse_reference(inputs[1], LABELS_TWO),
            ],
            fine_reference=[fine_reference(record) for record in inputs],
            schema=schema(),
            flan_baseline_summary=baseline_summary(2),
            prediction_manifest=prediction_manifest(),
            expected_reference_relevant_count=2,
        )
        aggregate = report["aggregate_comparison_vs_flan_t5_v0_4"]
        self.assertEqual(aggregate["llama_2_mean_field_accuracy"], 1.0)
        self.assertEqual(aggregate["llama_2_mean_macro_f1"], 1.0)
        self.assertEqual(aggregate["mean_field_accuracy_delta"], 0.5)
        self.assertTrue(aggregate["llama_improves_both_aggregate_metrics"])
        self.assertEqual(report["reference_relevant_count"], 2)

    def test_gate_miss_uses_frozen_end_to_end_fallback_labels(self) -> None:
        inputs = [input_record(1), input_record(2)]
        missed = prediction(inputs[1], None)
        report = evaluate.build_evaluation_report(
            inputs=inputs,
            predictions=[prediction(inputs[0], LABELS_ONE), missed],
            coarse_reference=[
                coarse_reference(inputs[0], LABELS_ONE),
                coarse_reference(inputs[1], LABELS_TWO),
            ],
            fine_reference=[fine_reference(record) for record in inputs],
            schema=schema(),
            flan_baseline_summary=baseline_summary(2),
            prediction_manifest=prediction_manifest(),
            expected_reference_relevant_count=2,
        )
        metrics = report["end_to_end_semantic_metrics_on_reference_relevant"]
        for field in coarse.COARSE_FIELDS:
            self.assertEqual(metrics[field]["accuracy"], 0.5)
        self.assertEqual(report["semantic_coverage_on_reference_relevant"], 0.5)


class LlamaEvaluationIntegrityTests(unittest.TestCase):
    def test_prediction_manifest_hashes_and_frozen_identity_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "inputs.jsonl"
            prediction_path = root / "predictions.jsonl"
            schema_path = ROOT / "config" / "news_feature_schema_coarse.json"
            record = input_record(1)
            predicted = prediction(record, LABELS_ONE)
            input_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            prediction_path.write_text(json.dumps(predicted) + "\n", encoding="utf-8")
            manifest = {
                **prediction_manifest(),
                "output_sha256": base.sha256_file(prediction_path),
                "input_sha256": base.sha256_file(input_path),
                "schema_sha256": base.sha256_file(schema_path),
                "deterministic_module_sha256": base.sha256_file(
                    Path(coarse.__file__).resolve()
                ),
                "total_prediction_count": 1,
                "selected_article_ids_sha256": base.sha256_text(record["article_id"]),
            }
            evaluate.validate_prediction_manifest(
                manifest,
                predictions_path=prediction_path,
                inputs_path=input_path,
                schema_path=schema_path,
                predictions=[predicted],
            )

            wrong_prompt = dict(manifest)
            wrong_prompt["prompt_version"] = "changed"
            with self.assertRaisesRegex(ValueError, "frozen Llama prompt"):
                evaluate.validate_prediction_manifest(
                    wrong_prompt,
                    predictions_path=prediction_path,
                    inputs_path=input_path,
                    schema_path=schema_path,
                    predictions=[predicted],
                )

            wrong_revision = dict(manifest)
            wrong_revision["model_revision"] = "f" * 40
            with self.assertRaisesRegex(ValueError, "frozen Llama revision"):
                evaluate.validate_prediction_manifest(
                    wrong_revision,
                    predictions_path=prediction_path,
                    inputs_path=input_path,
                    schema_path=schema_path,
                    predictions=[predicted],
                )

    def test_row_contract_replays_deterministic_gate_and_rejects_truncation(self) -> None:
        record = input_record(1)
        predicted = prediction(record, LABELS_ONE)
        manifest = prediction_manifest()
        evaluate.validate_prediction_rows(
            [predicted],
            inputs_by_id={record["article_id"]: record},
            schema=schema(),
            manifest=manifest,
        )
        predicted["passes"]["shock_scope"]["input_truncated"] = True
        with self.assertRaisesRegex(ValueError, "truncated shock_scope"):
            evaluate.validate_prediction_rows(
                [predicted],
                inputs_by_id={record["article_id"]: record},
                schema=schema(),
                manifest=manifest,
            )


if __name__ == "__main__":
    unittest.main()
