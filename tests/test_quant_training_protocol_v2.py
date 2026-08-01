from __future__ import annotations

import json
import unittest
from datetime import date, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "config" / "quant_training_protocol_v2.json"
V1_PROTOCOL_PATH = ROOT / "config" / "quant_training_protocol_v1.json"


class QuantTrainingProtocolV2Tests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))

    def test_full_history_artifacts_are_separate_from_v1(self):
        paths = self.protocol["artifact_paths"]
        self.assertEqual(
            paths["panel"],
            "data/features/quant/training_v2/modeling_panel.parquet",
        )
        self.assertEqual(
            paths["experiment_root"], "experiments/quant_training/v2"
        )
        self.assertEqual(paths["output_root"], "outputs/quant_training/v2")
        self.assertNotIn("/v1", paths["panel"])
        self.assertNotIn("/v1", paths["experiment_root"])
        self.assertNotIn("/v1", paths["output_root"])
        self.assertEqual(
            paths["dcc_history_panel"],
            "data/features/quant/training_v1/"
            "correlation_targets_and_loo_features.parquet",
        )
        self.assertEqual(
            paths["dcc_calendar"],
            "data/prices/alpaca/calendar/2016-01-01_2026-06-30.json",
        )

    def test_thirteen_expanding_half_year_folds_are_contiguous(self):
        folds = self.protocol["folds"]
        self.assertEqual(len(folds), 13)
        self.assertEqual(folds[0]["train_start"], "2017-12-28")
        self.assertEqual(folds[0]["test_start"], "2020-01-01")
        self.assertEqual(folds[-1]["test_end"], "2026-06-30")

        test_ranges = []
        for position, fold in enumerate(folds, start=1):
            self.assertEqual(fold["name"], f"fold_{position:02d}")
            train_end = date.fromisoformat(fold["train_end"])
            validation_start = date.fromisoformat(fold["validation_start"])
            validation_end = date.fromisoformat(fold["validation_end"])
            test_start = date.fromisoformat(fold["test_start"])
            test_end = date.fromisoformat(fold["test_end"])
            self.assertEqual(validation_start, train_end + timedelta(days=1))
            self.assertEqual(test_start, validation_end + timedelta(days=1))
            self.assertLess(test_start, test_end)
            test_ranges.append((test_start, test_end))

            if position > 1:
                previous = folds[position - 2]
                self.assertEqual(
                    fold["train_end"], previous["validation_end"]
                )
                self.assertEqual(
                    fold["validation_start"], previous["test_start"]
                )
                self.assertEqual(
                    fold["validation_end"], previous["test_end"]
                )

        for left, right in zip(
            test_ranges[:-1], test_ranges[1:], strict=True
        ):
            self.assertEqual(right[0], left[1] + timedelta(days=1))

    def test_2016_is_warmup_history_not_a_claimed_model_row(self):
        self.assertEqual(self.protocol["price_history_start"], "2016-01-01")
        self.assertEqual(
            self.protocol["feature_warmup"]["first_model_ready_date"],
            self.protocol["modeling_period_start"],
        )
        self.assertGreater(
            date.fromisoformat(self.protocol["modeling_period_start"]),
            date.fromisoformat(self.protocol["price_history_start"]),
        )

    def test_model_ladder_and_tuning_contract_match_v1(self):
        v1 = json.loads(V1_PROTOCOL_PATH.read_text(encoding="utf-8"))
        same_design_keys = [
            "target_contract",
            "linear_alpha_grid",
            "linear_alpha_boundary_rule",
            "elastic_net_l1_ratio_grid",
            "rung_1",
            "rung_2",
            "rung_3",
            "rung_4",
            "primary_metrics",
        ]
        for key in same_design_keys:
            self.assertEqual(self.protocol[key], v1[key], key)


if __name__ == "__main__":
    unittest.main()
