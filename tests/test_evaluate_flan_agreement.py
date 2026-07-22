from __future__ import annotations

import importlib.util
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "evaluate_flan_agreement.py"
SPEC = importlib.util.spec_from_file_location("evaluate_flan_agreement", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
EXTRACTOR_PATH = REPOSITORY_ROOT / "scripts" / "extract_flan_t5.py"
EXTRACTOR_SPEC = importlib.util.spec_from_file_location("extractor_for_test", EXTRACTOR_PATH)
assert EXTRACTOR_SPEC and EXTRACTOR_SPEC.loader
EXTRACTOR = importlib.util.module_from_spec(EXTRACTOR_SPEC)
EXTRACTOR_SPEC.loader.exec_module(EXTRACTOR)


class ClosedMetricsTests(unittest.TestCase):
    def test_perfect_predictions(self) -> None:
        metrics = MODULE.closed_label_metrics(
            ["firm_specific", "sector_wide"],
            ["firm_specific", "sector_wide"],
            ["firm_specific", "sector_wide", "unclear"],
        )
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["cohens_kappa"], 1.0)

    def test_invalid_output_counts_as_error(self) -> None:
        metrics = MODULE.closed_label_metrics(
            ["firm_specific", "sector_wide"],
            [None, "sector_wide"],
            ["firm_specific", "sector_wide", "unclear"],
        )
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["invalid_prediction_count"], 1)


class MultilabelMetricsTests(unittest.TestCase):
    def test_multilabel_micro_scores(self) -> None:
        metrics = MODULE.multilabel_metrics(
            [{"demand", "competition"}, set()],
            [{"demand"}, set()],
        )
        self.assertEqual(metrics["micro_precision"], 1.0)
        self.assertEqual(metrics["micro_recall"], 0.5)
        self.assertAlmostEqual(metrics["micro_f1"], 2 / 3)


class EndToEndIntegrityTests(unittest.TestCase):
    def test_core_prediction_report_without_loading_model(self) -> None:
        schema_path = REPOSITORY_ROOT / "config" / "news_feature_schema.json"
        schema_text = schema_path.read_text(encoding="utf-8")
        schema = json.loads(schema_text)
        input_record = {
            "row_number": 1,
            "article_id": "example",
            "time_published_utc": "2024-01-01T12:00:00Z",
            "source": "Example Wire",
            "headline": "AMD raises guidance above expectations",
            "article_text": "Advanced Micro Devices raised guidance above expectations.",
            "vendor_tickers": ["AMD"],
            "target": {
                "company": "Advanced Micro Devices",
                "ticker": "AMD",
                "sector": "Semiconductors",
                "sector_benchmark": "SOXX",
                "known_sector_peers": ["NVDA", "INTC"],
            },
        }
        raw_by_pass = {
            "scope": "direct_target | firm_specific | single_firm",
            "event": "guidance | confirmed | positive",
            "direction": "positive | not_applicable | not_applicable",
        }
        labels = EXTRACTOR.empty_labels()
        pass_records = {}
        for pass_name in EXTRACTOR.CORE_PASSES:
            prompt = EXTRACTOR.build_prompt(pass_name, input_record, schema, labels)
            parsed = EXTRACTOR.parse_pass(pass_name, raw_by_pass[pass_name], input_record, schema, labels)
            EXTRACTOR.update_labels(labels, pass_name, parsed["values"])
            pass_records[pass_name] = {
                "raw_output": raw_by_pass[pass_name],
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "valid": parsed["valid"],
                "strict_format_valid": parsed["strict_format_valid"],
                "input_truncated": False,
                "errors": [],
            }
        prediction = {
            "row_number": 1,
            "article_id": "example",
            "target_ticker": "AMD",
            "extractor": "google/flan-t5-large",
            "model_revision": "0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a",
            "protocol_version": schema["schema_version"],
            "labels": labels,
            "quality_flags": [],
            "validity": {
                "primary_closed_labels_valid": True,
                "primary_strict_format_valid": True,
                "all_requested_passes_valid": True,
            },
            "passes": pass_records,
        }
        reference_labels = dict(labels)
        reference_labels["transmission_channels"] = ["demand"]
        reference_labels["affected_companies"] = ["Advanced Micro Devices"]
        reference_labels["evidence"] = {
            "scope": "Advanced Micro Devices",
            "direction": "raised guidance",
            "surprise": "above expectations",
        }
        reference = {
            "row_number": 1,
            "article_id": "example",
            "target_ticker": "AMD",
            "annotator": "gpt-5.6-sol",
            "protocol_version": schema["schema_version"],
            "labels": reference_labels,
            "quality_flags": [],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "inputs.jsonl"
            reference_path = root / "reference.jsonl"
            prediction_path = root / "predictions.jsonl"
            report_path = root / "report.json"
            input_path.write_text(json.dumps(input_record) + "\n", encoding="utf-8")
            reference_path.write_text(json.dumps(reference) + "\n", encoding="utf-8")
            prediction_path.write_text(json.dumps(prediction) + "\n", encoding="utf-8")
            manifest = {
                "status": "complete",
                "output_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
                "schema_sha256": hashlib.sha256(schema_text.encode("utf-8")).hexdigest(),
                "total_prediction_count": 1,
                "model_id": "google/flan-t5-large",
                "model_revision": "0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a",
                "mode": "core",
                "passes": list(EXTRACTOR.CORE_PASSES),
                "model_files_sha256": {"model.safetensors": "test"},
                "device": "cpu",
                "precision": "float32",
            }
            prediction_path.with_suffix(".jsonl.manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--reference",
                    str(reference_path),
                    "--predictions",
                    str(prediction_path),
                    "--inputs",
                    str(input_path),
                    "--schema",
                    str(schema_path),
                    "--output",
                    str(report_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["closed_label_metrics"]["relevance"]["macro_f1"], 1.0)
            self.assertFalse(report["transmission_channel_metrics"]["available"])


if __name__ == "__main__":
    unittest.main()
