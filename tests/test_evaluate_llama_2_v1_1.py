from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def load_module(filename: str, name: str):
    path = SCRIPTS / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluator = load_module(
    "evaluate_llama_2_v1_1.py", "llama_2_v1_1_evaluator_for_test"
)


def metric(value: float) -> dict:
    return {"accuracy": value, "macro_f1": value}


class LlamaV11EvaluatorTests(unittest.TestCase):
    def test_deterministic_rows_cannot_hide_model_origin_collapse(self) -> None:
        schema = {
            "closed_label_fields": {
                "shock_scope": ["idiosyncratic", "common", "mixed", "unclear"],
                "event_family": [
                    "earnings_guidance",
                    "product_demand",
                    "other_or_unclear",
                ],
                "information_status": [
                    "confirmed",
                    "anticipated",
                    "unclear",
                ],
                "directional_alignment": [
                    "single_firm_only",
                    "same_direction",
                    "unclear",
                ],
            }
        }
        inputs = [{"article_id": value} for value in ("a", "b", "c")]
        reference_labels = {
            "shock_scope": "idiosyncratic",
            "event_family": "product_demand",
            "information_status": "confirmed",
            "directional_alignment": "single_firm_only",
        }
        reference = [
            {
                "article_id": value,
                "semantic_applicable": True,
                "labels": dict(reference_labels),
            }
            for value in ("a", "b", "c")
        ]
        model_labels = dict(reference_labels)
        gate_labels = {
            "shock_scope": "unclear",
            "event_family": "other_or_unclear",
            "information_status": "unclear",
            "directional_alignment": "unclear",
        }
        predictions = []
        for value in ("a", "b", "c"):
            is_gate = value == "c"
            labels = gate_labels if is_gate else model_labels
            predictions.append(
                {
                    "article_id": value,
                    "labels": dict(labels),
                    "passes": {
                        field: {
                            "origin": (
                                "deterministic_gate" if is_gate else "model"
                            )
                        }
                        for field in labels
                    },
                }
            )
        result = evaluator.evaluate_predictions(
            inputs=inputs,
            reference=reference,
            predictions=predictions,
            schema=schema,
            expected_reference_relevant_count=3,
        )
        diagnostic = result["decoder_diagnostics"]["shock_scope"]
        self.assertEqual(diagnostic["unique_prediction_count"], 2)
        self.assertEqual(diagnostic["model_origin_unique_prediction_count"], 1)
        self.assertTrue(diagnostic["single_class_collapse"])

    def test_promotion_requires_every_predeclared_check(self) -> None:
        baseline = {
            "mean_field_accuracy": 0.42,
            "mean_macro_f1": 0.17,
            "field_metrics": {
                "shock_scope": metric(0.18),
                "event_family": metric(0.06),
            },
            "decoder_diagnostics": {},
        }
        candidate = {
            "mean_field_accuracy": 0.43,
            "mean_macro_f1": 0.24,
            "field_metrics": {
                "shock_scope": metric(0.23),
                "event_family": metric(0.10),
            },
            "decoder_diagnostics": {
                field: {"single_class_collapse": False}
                for field in evaluator.coarse.COARSE_FIELDS
            },
        }
        decision = evaluator.promotion_decision(baseline, candidate)
        self.assertTrue(decision["all_checks_pass"])
        self.assertEqual(
            decision["decision"], "promote_decoder_for_fresh_holdout"
        )

        candidate["decoder_diagnostics"]["event_family"][
            "single_class_collapse"
        ] = True
        decision = evaluator.promotion_decision(baseline, candidate)
        self.assertFalse(decision["all_checks_pass"])

    def test_accuracy_cannot_rescue_insufficient_macro_f1_gain(self) -> None:
        baseline = {
            "mean_field_accuracy": 0.42,
            "mean_macro_f1": 0.17,
            "field_metrics": {
                "shock_scope": metric(0.18),
                "event_family": metric(0.06),
            },
            "decoder_diagnostics": {},
        }
        candidate = {
            "mean_field_accuracy": 0.60,
            "mean_macro_f1": 0.21,
            "field_metrics": {
                "shock_scope": metric(0.20),
                "event_family": metric(0.07),
            },
            "decoder_diagnostics": {
                field: {"single_class_collapse": False}
                for field in evaluator.coarse.COARSE_FIELDS
            },
        }
        decision = evaluator.promotion_decision(baseline, candidate)
        self.assertFalse(
            decision["checks"]["mean_macro_f1_improves_by_at_least_0_05"]
        )
        self.assertFalse(decision["all_checks_pass"])


if __name__ == "__main__":
    unittest.main()
