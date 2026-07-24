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
    "evaluate_llama_2_v1_2.py", "llama_2_v1_2_evaluator_for_test"
)


class LlamaV12EvaluatorTests(unittest.TestCase):
    def test_aggregate_delta_is_fieldwise_and_signed(self) -> None:
        candidate_fields = {}
        baseline_fields = {}
        for index, field in enumerate(evaluator.coarse.COARSE_FIELDS, start=1):
            candidate_fields[field] = {
                "accuracy": 0.4 + index / 100,
                "macro_f1": 0.3 + index / 100,
            }
            baseline_fields[field] = {
                "accuracy": 0.4,
                "macro_f1": 0.3,
            }
        candidate = {
            "field_metrics": candidate_fields,
            "mean_field_accuracy": 0.425,
            "mean_macro_f1": 0.325,
        }
        baseline = {
            "field_metrics": baseline_fields,
            "mean_field_accuracy": 0.4,
            "mean_macro_f1": 0.3,
        }
        result = evaluator.aggregate_delta(candidate, baseline)
        self.assertAlmostEqual(result["mean_field_accuracy_delta"], 0.025)
        self.assertAlmostEqual(result["mean_macro_f1_delta"], 0.025)
        self.assertAlmostEqual(
            result["fieldwise"]["shock_scope"]["accuracy_delta"], 0.01
        )


if __name__ == "__main__":
    unittest.main()
