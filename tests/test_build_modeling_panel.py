from __future__ import annotations

import unittest

import pandas as pd

from scripts import build_bollerslev_core_features as core
from scripts.correlation_training import build_modeling_panel as module


class ModelingPanelTests(unittest.TestCase):
    def test_join_is_one_to_one_and_feature_groups_are_target_specific(self):
        key = {
            "sector": "Test",
            "stock": "A",
            "benchmark": "ETF",
            "forecast_date": pd.Timestamp("2024-01-02"),
        }
        core_row = {
            **key,
            "asof_session": pd.Timestamp("2023-12-29"),
            **{column: 0.1 for column in core.FEATURE_COLUMNS},
        }
        target_row = {
            **key,
            **{
                f"target_{target}_{horizon}_correlation": 0.5
                for target in ("etf", "loo")
                for horizon in ("t1", "t2")
            },
            **{
                column: 0.1
                for column in module.LOO_PAIR_FEATURES
            },
        }
        context_columns = [
            *module.DENSE_CONTEXT_FEATURES,
            *module.VOLATILITY_FEATURES,
            *module.EXTENDED_FEATURES,
        ]
        context_row = {**key, **{column: 1.0 for column in context_columns}}
        panel, groups = module.build_modeling_panel(
            pd.DataFrame([core_row]),
            pd.DataFrame([context_row]),
            pd.DataFrame([target_row]),
            start=pd.Timestamp("2024-01-01"),
            end=pd.Timestamp("2024-12-31"),
        )
        self.assertEqual(len(panel), 1)
        self.assertIn("etf_rc_d", panel)
        self.assertNotIn("rc_d", panel)
        self.assertIn("loo_rc_d", groups["loo_core_22"])
        self.assertIn("sector_exp_rc_d", groups["etf_core_22"])
        self.assertIn("sector_exp_rc_d", groups["loo_core_22"])

    def test_duplicate_keys_are_rejected(self):
        frame = pd.DataFrame(
            [
                {
                    "sector": "Test",
                    "stock": "A",
                    "benchmark": "ETF",
                    "forecast_date": pd.Timestamp("2024-01-02"),
                }
            ]
            * 2
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            module.assert_unique(frame, "test")


if __name__ == "__main__":
    unittest.main()
