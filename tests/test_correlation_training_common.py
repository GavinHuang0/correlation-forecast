from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.correlation_training import training_common as module


class CorrelationTrainingCommonTests(unittest.TestCase):
    def test_t2_split_purges_target_crossing_boundary(self):
        spec = next(item for item in module.target_specs() if item.name == "t2_etf")
        dates = pd.bdate_range("2024-06-24", periods=10)
        panel = pd.DataFrame(
            {
                "forecast_date": dates,
                spec.response_column: 0.1,
                spec.target_end_column: dates + pd.offsets.BDay(4),
            }
        )
        fold = {
            "train_start": "2024-06-01",
            "train_end": "2024-06-30",
            "validation_start": "2024-07-01",
            "validation_end": "2024-07-31",
            "test_start": "2024-08-01",
            "test_end": "2024-08-31",
        }
        masks = module.split_masks(panel, spec, fold)
        train = panel.loc[masks["train"]]
        self.assertTrue(
            (train[spec.target_end_column] <= pd.Timestamp("2024-06-30")).all()
        )
        self.assertNotIn(pd.Timestamp("2024-06-27"), set(train["forecast_date"]))

    def test_leakage_columns_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "leakage"):
            module.validate_feature_columns(["target_etf_t1_correlation"])

    def test_fisher_roundtrip_is_bounded(self):
        correlations = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        reconstructed = module.fisher_to_correlation(
            module.correlation_to_fisher(correlations)
        )
        self.assertTrue(np.all(reconstructed <= 0.995))
        self.assertTrue(np.all(reconstructed >= -0.995))

    def test_log1p_transform_is_fixed_and_preserves_missing(self):
        frame = pd.DataFrame(
            {
                "premarket_volume": [0.0, 9.0, np.nan],
                "vix_lag1": [10.0, 11.0, 12.0],
            }
        )
        result = module.apply_fixed_log1p_transforms(
            frame, ["premarket_volume", "vix_lag1"]
        )
        self.assertAlmostEqual(result.loc[1, "premarket_volume"], np.log(10))
        self.assertTrue(np.isnan(result.loc[2, "premarket_volume"]))
        self.assertEqual(result.loc[1, "vix_lag1"], 11.0)

    def test_path_selector_returns_only_estimator_parameters(self):
        rng = np.random.default_rng(4)
        train = pd.DataFrame(
            {
                "x1": rng.normal(size=60),
                "x2": rng.normal(size=60),
            }
        )
        train["y"] = 0.5 * train["x1"] + rng.normal(scale=0.1, size=60)
        validation = pd.DataFrame(
            {
                "x1": rng.normal(size=20),
                "x2": rng.normal(size=20),
            }
        )
        validation["y"] = (
            0.5 * validation["x1"] + rng.normal(scale=0.1, size=20)
        )
        best, candidates = module.select_linear_hyperparameters(
            "elastic_net",
            train,
            validation,
            ["x1", "x2"],
            "y",
            alpha_grid=[0.001, 0.01],
            l1_ratios=[0.1, 0.9],
        )
        self.assertEqual(set(best), {"alpha", "l1_ratio"})
        self.assertGreaterEqual(len(candidates), 4)
        self.assertIn("path_iterations", candidates[0])
        self.assertIn("boundary_expansion", candidates[0])


if __name__ == "__main__":
    unittest.main()
