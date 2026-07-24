from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "diagnose_llama_2_results.py"
SPEC = importlib.util.spec_from_file_location("diagnose_llama_2_results", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
diagnose = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnose)


def schema() -> dict:
    return diagnose.read_json(ROOT / "config" / "news_feature_schema_coarse.json")


def input_record(index: int) -> dict:
    return {
        "article_id": f"article-{index}",
        "row_number": index,
        "target": {"ticker": "AMD"},
    }


def reference_record(index: int, labels: dict[str, str], relevant: bool) -> dict:
    return {
        "article_id": f"article-{index}",
        "row_number": index,
        "semantic_applicable": relevant,
        "labels": dict(labels),
    }


def prediction_record(
    index: int,
    labels: dict[str, str],
    *,
    canonical: dict[str, str],
    reversed_labels: dict[str, str],
    no_truncation: bool = True,
) -> dict:
    return {
        "article_id": f"article-{index}",
        "row_number": index,
        "labels": dict(labels),
        "label_origins": {field: "model" for field in labels},
        "passes": {
            field: {
                "origin": "model",
                "value": value,
                "canonical_prediction": canonical[field],
                "reversed_prediction": reversed_labels[field],
                "input_truncated": not no_truncation,
            }
            for field, value in labels.items()
        },
        "validity": {"no_input_truncation": no_truncation},
    }


class SavedResultDiagnosticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = schema()
        self.allowed = self.schema["closed_label_fields"]
        self.truth = {field: values[0] for field, values in self.allowed.items()}
        self.wrong = {field: values[1] for field, values in self.allowed.items()}
        self.inputs = [input_record(index) for index in range(1, 5)]
        self.references = [
            reference_record(index, self.truth, relevant=index < 4)
            for index in range(1, 5)
        ]
        self.llama = [
            prediction_record(
                index,
                self.wrong,
                canonical=self.truth,
                reversed_labels=self.wrong,
            )
            for index in range(1, 5)
        ]
        self.flan = [
            prediction_record(
                index,
                self.truth,
                canonical=self.truth,
                reversed_labels=self.truth,
            )
            for index in range(1, 5)
        ]

    def test_reports_order_sensitivity_truncation_and_paired_cis(self) -> None:
        report = diagnose.build_report(
            evaluation_inputs=self.inputs,
            llama_predictions=self.llama,
            flan_predictions=self.flan,
            coarse_reference=self.references,
            schema=self.schema,
            bootstrap_reps=100,
            seed=20260724,
        )

        self.assertEqual(report["evaluation_record_count"], 4)
        self.assertEqual(report["reference_relevant_count"], 3)
        shock = report["decoder_diagnostics_on_model_origin_passes"]["llama_2"][
            "shock_scope"
        ]
        self.assertEqual(shock["model_origin_pass_count"], 4)
        self.assertEqual(
            shock["current_prediction_distribution"][self.wrong["shock_scope"]],
            4,
        )
        self.assertEqual(
            shock["canonical_prediction_distribution"][self.truth["shock_scope"]],
            4,
        )
        self.assertEqual(shock["canonical_reversed_agreement_rate"], 0.0)

        truncation = report["truncation_diagnostics"]["llama_2"]
        self.assertTrue(truncation["all_recorded_passes_untruncated"])

        bootstrap = report["paired_bootstrap_95_percent_intervals"]
        self.assertEqual(bootstrap["seed"], 20260724)
        self.assertEqual(bootstrap["article_count"], 3)
        for metric in ("mean_field_accuracy", "mean_macro_f1"):
            difference = bootstrap["llama_minus_flan"][metric]
            self.assertEqual(difference["point_estimate"], -1.0)
            self.assertEqual(difference["ci_95"], [-1.0, -1.0])

    def test_bootstrap_is_reproducible_and_flags_truncated_pass(self) -> None:
        self.llama[0]["passes"]["shock_scope"]["input_truncated"] = True
        self.llama[0]["validity"]["no_input_truncation"] = False
        first = diagnose.build_report(
            evaluation_inputs=self.inputs,
            llama_predictions=self.llama,
            flan_predictions=self.flan,
            coarse_reference=self.references,
            schema=self.schema,
            bootstrap_reps=37,
            seed=20260724,
        )
        second = diagnose.build_report(
            evaluation_inputs=self.inputs,
            llama_predictions=self.llama,
            flan_predictions=self.flan,
            coarse_reference=self.references,
            schema=self.schema,
            bootstrap_reps=37,
            seed=20260724,
        )
        self.assertEqual(
            first["paired_bootstrap_95_percent_intervals"],
            second["paired_bootstrap_95_percent_intervals"],
        )
        truncation = first["truncation_diagnostics"]["llama_2"]
        self.assertEqual(truncation["truncated_pass_count"], 1)
        self.assertFalse(truncation["all_recorded_passes_untruncated"])

    def test_requires_exact_prediction_id_alignment(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "FLAN predictions IDs must exactly match"
        ):
            diagnose.build_report(
                evaluation_inputs=self.inputs,
                llama_predictions=self.llama,
                flan_predictions=self.flan[:-1],
                coarse_reference=self.references,
                schema=self.schema,
                bootstrap_reps=5,
                seed=20260724,
            )


if __name__ == "__main__":
    unittest.main()
