from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from scripts import build_bollerslev_core_features as core
from scripts.correlation_training import build_targets as module


def returns(symbol: str, dates: pd.DatetimeIndex, values: list[list[float]]):
    rows = []
    for trade_date, day_values in zip(dates, values, strict=True):
        for position, value in enumerate(day_values):
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": trade_date,
                    "interval_key": f"key_{position}",
                    "log_return": value,
                }
            )
    return pd.DataFrame(rows)


class CorrelationTargetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dates = pd.bdate_range("2024-01-02", periods=8)
        self.sector = core.Sector(
            "Test", "ETF", ("A", "B", "C", "D", "E", "F")
        )
        base = {
            "A": [[0.01, -0.02]] * len(self.dates),
            "ETF": [[0.02, -0.01]] * len(self.dates),
            "B": [[0.01, 0.00]] * len(self.dates),
            "C": [[0.02, -0.01]] * len(self.dates),
            "D": [[0.00, -0.02]] * len(self.dates),
            "E": [[0.03, 0.01]] * len(self.dates),
            "F": [[-0.01, -0.01]] * len(self.dates),
        }
        self.frames = {
            symbol: returns(symbol, self.dates, values)
            for symbol, values in base.items()
        }
        self.expected = {
            date: frozenset({"key_0", "key_1"}) for date in self.dates
        }

    def test_equal_weight_bucket_averages_simple_not_log_returns(self) -> None:
        values = np.array([[math.log(1.10), math.log(0.90)]])
        actual = module.exact_equal_weight_log_return(values)[0]
        self.assertAlmostEqual(actual, 0.0)
        self.assertNotAlmostEqual(actual, values.mean())

    def test_etf_and_loo_use_same_complete_interval_set(self) -> None:
        etf, loo = module.build_common_rth_components(
            self.frames, self.sector, "A", self.expected
        )
        first = etf.iloc[0]
        x = np.array([0.01, -0.02])
        y = np.array([0.02, -0.01])
        self.assertAlmostEqual(first["realized_covariance"], float(x @ y))
        self.assertTrue(
            etf[
                [
                    "trade_date",
                    "aligned_return_count",
                    "complete_official_schedule",
                ]
            ].equals(
                loo[
                    [
                        "trade_date",
                        "aligned_return_count",
                        "complete_official_schedule",
                    ]
                ]
            )
        )

        missing = dict(self.frames)
        missing_peer = missing["F"].copy()
        missing_peer = missing_peer.drop(
            missing_peer[
                (missing_peer["trade_date"].eq(self.dates[2]))
                & (missing_peer["interval_key"].eq("key_1"))
            ].index
        )
        missing["F"] = missing_peer
        etf_missing, loo_missing = module.build_common_rth_components(
            missing, self.sector, "A", self.expected
        )
        for frame in (etf_missing, loo_missing):
            row = frame[frame["trade_date"].eq(self.dates[2])].iloc[0]
            self.assertFalse(row["complete_official_schedule"])
            self.assertTrue(np.isnan(row["realized_covariance"]))

    def test_t2_sums_components_and_lags_end_before_forecast(self) -> None:
        components = pd.DataFrame(
            {
                "trade_date": self.dates,
                "realized_covariance": [1, 0, 1, 0, 1, 2, 2, 2],
                "stock_realized_variance": [1] * 8,
                "benchmark_realized_variance": [1] * 8,
                "negative_realized_covariance": [0] * 8,
                "stock_negative_realized_variance": [0] * 8,
                "benchmark_negative_realized_variance": [0] * 8,
                "stock_rth_log_return": [0.01] * 8,
                "benchmark_rth_log_return": [0.02] * 8,
                "aligned_return_count": [26] * 8,
                "expected_return_count": [26] * 8,
            }
        )
        result = module.target_and_lag_frame(
            components, self.dates, prefix="etf"
        ).set_index("forecast_date")
        # (1 + 0 + 1 + 0 + 1) / sqrt(5 * 5), not mean daily corr
        self.assertAlmostEqual(
            result.loc[self.dates[0], "target_etf_t2_correlation"], 3 / 5
        )
        self.assertEqual(
            result.loc[self.dates[0], "target_etf_t2_end_date"],
            self.dates[4],
        )
        self.assertTrue(np.isnan(result.loc[self.dates[0], "etf_rth_rc_d"]))
        self.assertAlmostEqual(
            result.loc[self.dates[1], "etf_rth_rc_d"], 1.0
        )
        self.assertTrue(
            np.isnan(result.loc[self.dates[4], "etf_rth_rc_w"])
        )
        self.assertAlmostEqual(
            result.loc[self.dates[5], "etf_rth_rc_w"], 3 / 5
        )
        self.assertTrue(
            result.iloc[-4:]["target_etf_t2_correlation"].isna().all()
        )

    def test_missing_one_daily_component_invalidates_t2(self) -> None:
        components = pd.DataFrame(
            {
                "trade_date": self.dates[:5],
                "realized_covariance": [1, 1, np.nan, 1, 1],
                "stock_realized_variance": [1, 1, np.nan, 1, 1],
                "benchmark_realized_variance": [1, 1, np.nan, 1, 1],
                "negative_realized_covariance": [0] * 5,
                "stock_negative_realized_variance": [0] * 5,
                "benchmark_negative_realized_variance": [0] * 5,
                "stock_rth_log_return": [0.01] * 5,
                "benchmark_rth_log_return": [0.02] * 5,
                "aligned_return_count": [26] * 5,
                "expected_return_count": [26] * 5,
            }
        )
        result = module.target_and_lag_frame(
            components, self.dates[:5], prefix="loo"
        )
        self.assertTrue(np.isnan(result.iloc[0]["target_loo_t2_correlation"]))


if __name__ == "__main__":
    unittest.main()
