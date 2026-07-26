from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "experiment_flan_t5_xl_v1_1.py"
SPEC = importlib.util.spec_from_file_location("flan_xl_v1_1_for_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
experiment = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = experiment
SPEC.loader.exec_module(experiment)


class FlanT5XlV11Tests(unittest.TestCase):
    def test_positive_tau_boosts_the_rare_label(self) -> None:
        pass_record = {
            "order_averaged_mean_log_probabilities": {
                "common": -1.0,
                "mixed": -1.1,
            }
        }
        selected, adjusted = experiment.calibrated_choice(
            pass_record,
            source="averaged",
            tau=0.5,
            priors={"common": 0.8, "mixed": 0.2},
            schema_order=["common", "mixed"],
        )
        self.assertEqual(selected, "mixed")
        self.assertGreater(adjusted["mixed"], adjusted["common"])

    def test_zero_tau_preserves_source_argmax(self) -> None:
        pass_record = {
            "canonical_candidate_mean_log_probabilities": {
                "first": -0.3,
                "second": -0.2,
            }
        }
        selected, adjusted = experiment.calibrated_choice(
            pass_record,
            source="canonical",
            tau=0.0,
            priors={"first": 0.5, "second": 0.5},
            schema_order=["first", "second"],
        )
        self.assertEqual(selected, "second")
        self.assertEqual(adjusted, {"first": -0.3, "second": -0.2})

    def test_non_model_origin_remains_unchanged(self) -> None:
        record = {
            "semantic_applicable": True,
            "labels": {"shock_scope": "idiosyncratic"},
            "passes": {
                "shock_scope": {
                    "origin": "hierarchical_derivation",
                    "value": "idiosyncratic",
                }
            },
        }
        value = experiment.guess_for_record(
            record,
            "shock_scope",
            source="averaged",
            tau=1.0,
            priors={
                "idiosyncratic": 0.5,
                "common": 0.2,
                "mixed": 0.2,
                "unclear": 0.1,
            },
            schema_order=["idiosyncratic", "common", "mixed", "unclear"],
        )
        self.assertEqual(value, "idiosyncratic")

    def test_selection_prefers_macro_f1_before_accuracy(self) -> None:
        high_accuracy = {
            "source": "averaged",
            "tau": 0.0,
            "metrics": {
                "macro_f1": 0.3,
                "accuracy": 0.8,
                "cohens_kappa": 0.4,
            },
        }
        balanced = {
            "source": "canonical",
            "tau": 0.25,
            "metrics": {
                "macro_f1": 0.5,
                "accuracy": 0.6,
                "cohens_kappa": 0.3,
            },
        }
        self.assertGreater(
            experiment.selection_key(balanced),
            experiment.selection_key(high_accuracy),
        )


if __name__ == "__main__":
    unittest.main()
