from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_additional_quant_features.py"
SPEC = importlib.util.spec_from_file_location("build_additional_quant_features", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def bars(symbol: str, dates: list[str], closes: list[tuple[float, float]], volume=100):
    rows = []
    for day, (first, last) in zip(dates, closes, strict=True):
        rows.extend(
            [
                {
                    "symbol": symbol,
                    "timestamp_utc": f"{day}T14:30:00Z",
                    "trade_date": day,
                    "bar_start_et": "09:30:00",
                    "open": first,
                    "close": first,
                    "volume": volume,
                },
                {
                    "symbol": symbol,
                    "timestamp_utc": f"{day}T14:45:00Z",
                    "trade_date": day,
                    "bar_start_et": "09:45:00",
                    "open": first,
                    "close": last,
                    "volume": volume,
                },
            ]
        )
    frame = pd.DataFrame(rows)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    return frame


class AdditionalQuantFeatureTests(unittest.TestCase):
    def test_regular_aggregation_lags_close_only_features(self) -> None:
        dates = [f"2024-01-{day:02d}" for day in range(2, 17)]
        frame = bars(
            "AMD",
            dates,
            [(100 + index, 101 + index) for index in range(len(dates))],
        )
        daily = MODULE.aggregate_regular_bars(frame)

        expected = abs(np.log(101 / 100))
        self.assertAlmostEqual(daily.iloc[0]["realized_volatility"], expected)
        self.assertTrue(np.isnan(daily.iloc[0]["lagged_realized_volatility"]))
        self.assertAlmostEqual(
            daily.iloc[1]["lagged_realized_volatility"],
            daily.iloc[0]["realized_volatility"],
        )
        self.assertTrue(
            daily.iloc[-1]["lagged_relative_daily_volume_20d"] > 0
        )

    def test_premarket_aggregation_uses_current_cutoff_window(self) -> None:
        frame = bars(
            "AMD",
            ["2024-01-02"],
            [(100, 102)],
            volume=50,
        )
        daily = MODULE.aggregate_extended_bars(frame, "premarket")

        self.assertAlmostEqual(daily.iloc[0]["premarket_return"], 0.02)
        self.assertEqual(daily.iloc[0]["premarket_volume"], 100)
        self.assertEqual(daily.iloc[0]["premarket_bar_count"], 2)

    def test_factor_implied_correlation_recovers_common_factor_signal(self) -> None:
        rng = np.random.default_rng(7)
        count = 150
        dates = pd.bdate_range("2023-01-02", periods=count)
        market = rng.normal(0, 0.01, count)
        factors = pd.DataFrame(
            {
                "date": dates,
                "mkt_rf": market,
                "smb": rng.normal(0, 0.003, count),
                "hml": rng.normal(0, 0.003, count),
                "mom": rng.normal(0, 0.003, count),
                "rf": np.zeros(count),
            }
        )
        daily = pd.DataFrame(
            {
                "trade_date": np.tile(dates, 2),
                "symbol": np.repeat(["AMD", "SOXX"], count),
                "daily_close_return": np.concatenate(
                    [
                        1.2 * market + rng.normal(0, 0.002, count),
                        0.9 * market + rng.normal(0, 0.002, count),
                    ]
                ),
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "factors.csv"
            factors.to_csv(path, index=False)
            result = MODULE.rolling_factor_implied_correlations(
                daily,
                path,
                [MODULE.Pair("Semiconductors", "AMD", "SOXX")],
                window=100,
                minimum=60,
                factor_lag_sessions=2,
            )

        self.assertGreater(len(result), 50)
        self.assertGreater(result["factor_implied_correlation"].median(), 0.5)
        self.assertTrue(result["factor_implied_correlation"].between(-1, 1).all())

    def test_panel_builds_stock_sector_and_premarket_features(self) -> None:
        dates = [f"2024-01-{day:02d}" for day in range(2, 17)]
        regular = pd.concat(
            [
                bars(
                    "AMD",
                    dates,
                    [(100 + index, 101 + index) for index in range(len(dates))],
                ),
                bars(
                    "SOXX",
                    dates,
                    [(200 + index, 201 + index) for index in range(len(dates))],
                ),
            ],
            ignore_index=True,
        )
        premarket = pd.concat(
            [
                bars(
                    "AMD",
                    dates[1:],
                    [(102 + index, 103 + index) for index in range(len(dates) - 1)],
                ),
                bars(
                    "SOXX",
                    dates[1:],
                    [(202 + index, 203 + index) for index in range(len(dates) - 1)],
                ),
            ],
            ignore_index=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            panel = MODULE.build_panel(
                regular,
                premarket,
                pd.DataFrame(),
                pairs=[MODULE.Pair("Semiconductors", "AMD", "SOXX")],
                official_dir=Path(directory),
                include_factor_feature=False,
            )

        self.assertEqual(len(panel), len(dates))
        self.assertNotIn("rth_close", panel.columns)
        self.assertNotIn("daily_volume", panel.columns)
        available = panel.dropna(subset=["stock_overnight_return"])
        self.assertGreater(len(available), 10)
        self.assertTrue(
            np.allclose(
                available["stock_minus_sector_overnight_return"],
                available["stock_overnight_return"]
                - available["sector_overnight_return"],
            )
        )


if __name__ == "__main__":
    unittest.main()
