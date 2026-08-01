from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import stock_etf_spread_backtest as module


class StockEtfSpreadBacktestTests(unittest.TestCase):
    def test_quant_v2_run_reuses_the_v1_strategy_contract(self) -> None:
        original = module.load_protocol(
            Path("config/stock_etf_spread_backtest_v1.json")
        )
        extended = module.load_protocol(
            Path("config/stock_etf_spread_backtest_quant_v2.json")
        )
        for section in ("signal", "portfolio", "execution", "inference"):
            self.assertEqual(extended[section], original[section])
        self.assertEqual(
            extended["inputs"]["prediction_model"],
            "elastic_xgboost_ensemble",
        )
        self.assertEqual(
            extended["inputs"]["prediction_target"], "t2_etf"
        )
        self.assertNotEqual(
            extended["outputs"]["directory"],
            original["outputs"]["directory"],
        )
        self.assertEqual(
            extended["reporting_subsamples"]["matched_recent"],
            ["fold_11", "fold_12", "fold_13"],
        )

    def test_daily_returns_require_exact_official_interval_set(self) -> None:
        date_1, date_2 = pd.to_datetime(["2025-01-02", "2025-01-03"])
        returns = pd.DataFrame(
            {
                "symbol": ["A", "A", "A"],
                "trade_date": [date_1, date_1, date_2],
                "interval_key": ["09:30", "09:45", "09:30"],
                "log_return": [0.01, -0.02, 0.03],
            }
        )
        expected = {
            date_1: frozenset({"09:30", "09:45"}),
            date_2: frozenset({"09:30", "09:45"}),
        }
        result = module.complete_daily_rth_returns(returns, expected)
        self.assertEqual(result["trade_date"].tolist(), [date_1])
        self.assertAlmostEqual(result.iloc[0]["rth_log_return"], -0.01)
        self.assertAlmostEqual(
            result.iloc[0]["rth_simple_return"], np.expm1(-0.01)
        )

    def test_beta_and_divergence_are_shifted_before_forecast_date(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=8)
        stock = pd.Series(
            [0.01, 0.02, 0.03, 0.04, 0.05, 99.0, 0.07, 0.08],
            index=dates,
        )
        benchmark = pd.Series(
            [0.01, 0.01, 0.02, 0.02, 0.03, 99.0, 0.04, 0.04],
            index=dates,
        )
        first = module.rolling_beta_and_divergence(
            stock,
            benchmark,
            beta_window=5,
            beta_minimum=3,
            divergence_window=3,
            beta_lower=0.0,
            beta_upper=3.0,
        )
        changed = stock.copy()
        changed.loc[dates[5]] = -99.0
        second = module.rolling_beta_and_divergence(
            changed,
            benchmark,
            beta_window=5,
            beta_minimum=3,
            divergence_window=3,
            beta_lower=0.0,
            beta_upper=3.0,
        )
        self.assertAlmostEqual(
            first.loc[dates[5], "return_divergence"],
            second.loc[dates[5], "return_divergence"],
        )
        self.assertNotAlmostEqual(
            first.loc[dates[6], "return_divergence"],
            second.loc[dates[6], "return_divergence"],
        )

    def test_missing_session_is_not_compressed_out_of_divergence(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=7)
        stock = pd.Series(
            [0.01, 0.02, 0.015, 0.03, 0.025, 0.04, 0.035],
            index=dates,
        )
        benchmark = pd.Series(
            [0.008, 0.012, 0.009, 0.016, 0.011, 0.018, 0.014],
            index=dates,
        )
        stock.loc[dates[3]] = np.nan
        output = module.rolling_beta_and_divergence(
            stock,
            benchmark,
            beta_window=3,
            beta_minimum=2,
            divergence_window=3,
            beta_lower=0.0,
            beta_upper=3.0,
        )
        self.assertTrue(np.isnan(output.loc[dates[4], "return_divergence"]))
        self.assertFalse(np.isnan(output.loc[dates[3], "return_divergence"]))

    def _signals(self) -> pd.DataFrame:
        records = []
        for sector, benchmark, divergences in (
            ("S1", "E1", [-0.03, -0.01, 0.01, 0.03]),
            ("S2", "E2", [-0.04, -0.02, 0.02, 0.04]),
        ):
            for index, divergence in enumerate(divergences):
                records.append(
                    {
                        "fold": "fold_1",
                        "forecast_date": pd.Timestamp("2025-01-02"),
                        "sector": sector,
                        "stock": f"{sector}_{index}",
                        "benchmark": benchmark,
                        "rolling_beta": 1.0 + index / 10,
                        "return_divergence": divergence,
                        "predicted_correlation": 0.6 + index / 20,
                        "persistence_correlation": 0.55,
                        "predicted_correlation_change": 0.05 + index / 20,
                    }
                )
        return pd.DataFrame(records)

    def test_sleeves_are_sector_neutral_and_gross_normalized(self) -> None:
        _, positions = module.construct_sleeve_weights(
            self._signals(),
            variant="ml_strengthening",
            correlation_floor=0.5,
            minimum_names=2,
            sleeve_gross=1.0,
        )
        stock = positions[positions["leg"].eq("stock")]
        sector_net = stock.groupby("sector")["weight"].sum()
        np.testing.assert_allclose(sector_net, 0.0, atol=1e-12)
        self.assertAlmostEqual(float(positions["weight"].abs().sum()), 1.0)
        self.assertTrue((positions["leg"] == "etf").any())

    def test_divergence_direction_and_etf_beta_hedge_are_correct(self) -> None:
        signals = self._signals()
        _, positions = module.construct_sleeve_weights(
            signals,
            variant="unconditional",
            correlation_floor=0.5,
            minimum_names=2,
            sleeve_gross=1.0,
        )
        stock_positions = positions[positions["leg"].eq("stock")]
        weights = stock_positions.set_index("symbol")["weight"]
        self.assertLess(weights["S1_0"], 0.0)
        self.assertGreater(weights["S1_3"], 0.0)

        stock_beta = stock_positions.merge(
            signals[["sector", "stock", "rolling_beta"]],
            left_on=["sector", "symbol"],
            right_on=["sector", "stock"],
            validate="one_to_one",
        )
        stock_beta["beta_exposure"] = (
            stock_beta["weight"] * stock_beta["rolling_beta"]
        )
        combined_beta = stock_beta.groupby("sector")[
            "beta_exposure"
        ].sum() + positions[positions["leg"].eq("etf")].groupby("sector")[
            "weight"
        ].sum()
        np.testing.assert_allclose(combined_beta, 0.0, atol=1e-12)

    def test_actual_targets_cannot_change_weights(self) -> None:
        signals = self._signals()
        signals["actual_correlation"] = np.linspace(-1, 1, len(signals))
        _, first = module.construct_sleeve_weights(
            signals,
            variant="ml_strengthening",
            correlation_floor=0.5,
            minimum_names=2,
            sleeve_gross=1.0,
        )
        signals["actual_correlation"] *= -1
        _, second = module.construct_sleeve_weights(
            signals,
            variant="ml_strengthening",
            correlation_floor=0.5,
            minimum_names=2,
            sleeve_gross=1.0,
        )
        pd.testing.assert_frame_equal(first, second)

    def test_overlapping_sleeve_contributes_one_fifth_for_five_days(self) -> None:
        sessions = pd.bdate_range("2025-01-02", periods=6)
        sleeves = pd.DataFrame(
            {
                "fold": ["fold_1"],
                "forecast_date": [sessions[0]],
                "variant": ["ml_strengthening"],
                "sector": ["S"],
                "leg": ["stock"],
                "symbol": ["A"],
                "weight": [1.0],
            }
        )
        output = module.expand_overlapping_sleeves(
            sleeves, sessions, holding_sessions=5
        )
        self.assertEqual(output["trade_date"].tolist(), list(sessions[:5]))
        np.testing.assert_allclose(output["weight"], 0.2)

    def test_intraday_cost_charges_open_and_close(self) -> None:
        date = pd.Timestamp("2025-01-02")
        positions = pd.DataFrame(
            {
                "fold": ["fold_1", "fold_1"],
                "trade_date": [date, date],
                "variant": ["unconditional", "unconditional"],
                "sector": ["S", "S"],
                "leg": ["stock", "etf"],
                "symbol": ["A", "E"],
                "weight": [0.5, -0.5],
            }
        )
        returns = pd.DataFrame(
            {
                "trade_date": [date, date],
                "symbol": ["A", "E"],
                "rth_simple_return": [0.01, 0.0],
            }
        )
        calendar = pd.DataFrame({"fold": ["fold_1"], "trade_date": [date]})
        output = module.daily_strategy_returns(
            positions, returns, calendar, ["unconditional"], [2.0]
        )
        row = output.iloc[0]
        self.assertAlmostEqual(row["gross_return"], 0.005)
        self.assertAlmostEqual(row["turnover"], 2.0)
        self.assertAlmostEqual(row["transaction_cost"], 0.0004)
        self.assertAlmostEqual(row["net_return"], 0.0046)

    def test_drawdown_includes_loss_from_initial_capital(self) -> None:
        result = module._maximum_drawdown(np.array([-0.10, 0.05]))
        self.assertAlmostEqual(result, -0.10)

    def test_reporting_subsample_uses_only_requested_folds(self) -> None:
        rows = []
        for fold, dates in (
            ("fold_01", pd.bdate_range("2025-01-02", periods=2)),
            ("fold_02", pd.bdate_range("2025-07-01", periods=2)),
        ):
            for date in dates:
                for variant, daily_return in (
                    ("ml_strengthening", 0.001),
                    ("unconditional", 0.0),
                ):
                    rows.append(
                        {
                            "fold": fold,
                            "trade_date": date,
                            "variant": variant,
                            "cost_bps_per_side": 2.0,
                            "net_return": daily_return,
                            "gross_exposure": 1.0,
                            "turnover": 2.0,
                            "transaction_cost": 0.0004,
                        }
                    )
        protocol = {
            "signal": {"primary_variant": "ml_strengthening"},
            "execution": {"primary_cost_bps_per_side": 2.0},
            "inference": {
                "annualization_sessions": 252,
                "moving_block_sessions": 1,
                "bootstrap_resamples": 20,
                "bootstrap_seed": 1729,
                "comparators": ["unconditional"],
            },
            "reporting_subsamples": {"second": ["fold_02"]},
        }
        metrics, comparisons = module.reporting_subsample_results(
            pd.DataFrame(rows), protocol
        )
        self.assertEqual(set(metrics["sample"]), {"second"})
        self.assertEqual(set(metrics["dates"]), {2})
        self.assertEqual(set(comparisons["sample"]), {"second"})
        self.assertEqual(set(comparisons["dates"]), {2})

    def test_json_records_replace_missing_values_with_null(self) -> None:
        frame = pd.DataFrame(
            {
                "value": [1.0, np.nan],
                "label": ["observed", None],
            }
        )
        records = module._json_records(frame)
        self.assertEqual(records[0], {"value": 1.0, "label": "observed"})
        self.assertEqual(records[1], {"value": None, "label": None})


if __name__ == "__main__":
    unittest.main()
