from __future__ import annotations

import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_bollerslev_core_features.py"
SPEC = importlib.util.spec_from_file_location(
    "build_bollerslev_core_features", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def make_bars(
    symbol: str,
    sessions: list[tuple[str, list[str], float, list[float]]],
) -> pd.DataFrame:
    """Build bars from (date, times, first_open, exact log returns)."""

    rows: list[dict[str, object]] = []
    for day, times, first_open, returns in sessions:
        prior_close = first_open
        for position, (bar_time, value) in enumerate(
            zip(times, returns, strict=True)
        ):
            open_price = first_open if position == 0 else prior_close
            close_price = open_price * math.exp(value)
            hour, minute = map(int, bar_time.split(":")[:2])
            timestamp = (
                pd.Timestamp(day, tz="UTC")
                + pd.Timedelta(hours=hour + 5, minutes=minute)
            )
            rows.append(
                {
                    "symbol": symbol,
                    "timestamp_utc": timestamp,
                    "trade_date": pd.Timestamp(day),
                    "bar_start_et": f"{hour:02d}:{minute:02d}:00",
                    "open": open_price,
                    "close": close_price,
                }
            )
            prior_close = close_price
    return pd.DataFrame(rows)


def interval_frame(
    symbol: str, dates: pd.DatetimeIndex, values: list[list[float]]
) -> pd.DataFrame:
    rows = []
    for day, day_values in zip(dates, values, strict=True):
        for index, value in enumerate(day_values):
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": day,
                    "interval_key": f"i{index}",
                    "log_return": value,
                }
            )
    return pd.DataFrame(rows)


class BollerslevCoreFeatureTests(unittest.TestCase):
    def test_feature_schema_has_exactly_twenty_two_core_columns(self) -> None:
        self.assertEqual(len(MODULE.FEATURE_COLUMNS), 22)
        self.assertEqual(len(set(MODULE.FEATURE_COLUMNS)), 22)

    def test_full_day_returns_include_overnight_exactly_once(self) -> None:
        prior_stock_close = 100.0
        prior_benchmark_close = 200.0
        stock_open = prior_stock_close * math.exp(0.02)
        benchmark_open = prior_benchmark_close * math.exp(-0.01)
        stock_bars = make_bars(
            "STOCK",
            [
                ("2024-01-02", ["09:30"], prior_stock_close, [0.0]),
                ("2024-01-03", ["09:30", "09:45"], stock_open, [0.01, -0.03]),
            ],
        )
        benchmark_bars = make_bars(
            "ETF",
            [
                ("2024-01-02", ["09:30"], prior_benchmark_close, [0.0]),
                (
                    "2024-01-03",
                    ["09:30", "09:45"],
                    benchmark_open,
                    [0.02, 0.01],
                ),
            ],
        )
        stock_returns = MODULE.build_interval_returns(stock_bars)
        benchmark_returns = MODULE.build_interval_returns(benchmark_bars)
        day = pd.Timestamp("2024-01-03")
        actual_stock = stock_returns.loc[
            stock_returns["trade_date"].eq(day), "log_return"
        ].to_numpy()
        actual_benchmark = benchmark_returns.loc[
            benchmark_returns["trade_date"].eq(day), "log_return"
        ].to_numpy()

        np.testing.assert_allclose(actual_stock, [0.02, 0.01, -0.03])
        np.testing.assert_allclose(actual_benchmark, [-0.01, 0.02, 0.01])
        components = MODULE.pair_daily_components(
            stock_returns,
            benchmark_returns,
            min_aligned_returns=3,
            min_alignment_ratio=1.0,
        )
        row = components[components["trade_date"].eq(day)].iloc[0]
        self.assertAlmostEqual(row["realized_covariance"], -0.0003)
        self.assertAlmostEqual(row["left_realized_variance"], 0.0014)
        self.assertAlmostEqual(row["right_realized_variance"], 0.0006)
        correlation = MODULE.component_correlation(
            [row["realized_covariance"]],
            [row["left_realized_variance"]],
            [row["right_realized_variance"]],
        )[0]
        self.assertAlmostEqual(correlation, -0.327326835354, places=11)
        self.assertAlmostEqual(
            actual_stock.sum(),
            math.log(
                stock_bars.iloc[-1]["close"] / stock_bars.iloc[0]["close"]
            ),
        )

    def test_multiday_correlation_aggregates_components_not_correlations(self) -> None:
        dates = pd.date_range("2024-01-02", periods=2, freq="D")
        components = pd.DataFrame(
            {
                "realized_covariance": [0.0008, -0.0002],
                "left_realized_variance": [0.0008, 0.0002],
                "right_realized_variance": [0.0008, 0.0002],
                "negative_realized_covariance": [0.0004, 0.0001],
                "left_negative_realized_variance": [0.0004, 0.0001],
                "right_negative_realized_variance": [0.0004, 0.0001],
            },
            index=dates,
        )
        features = MODULE.build_har_features(
            components, horizons={"two": 2}
        )
        self.assertAlmostEqual(features.iloc[-1]["rc_two"], 0.6)
        self.assertNotAlmostEqual(features.iloc[-1]["rc_two"], 0.0)

    def test_negative_semicorrelation_uses_own_semivariances(self) -> None:
        day = pd.Timestamp("2024-01-02")
        left = interval_frame(
            "A", pd.DatetimeIndex([day]), [[-0.02, -0.01, 0.03, -0.04]]
        )
        right = interval_frame(
            "B", pd.DatetimeIndex([day]), [[-0.01, 0.02, -0.02, -0.02]]
        )
        components = MODULE.pair_daily_components(
            left,
            right,
            min_aligned_returns=1,
            min_alignment_ratio=1.0,
            require_overnight=False,
        )
        row = components.iloc[0]
        self.assertAlmostEqual(row["negative_realized_covariance"], 0.001)
        self.assertAlmostEqual(row["left_negative_realized_variance"], 0.0021)
        self.assertAlmostEqual(row["right_negative_realized_variance"], 0.0009)
        value = MODULE.component_correlation(
            [row["negative_realized_covariance"]],
            [row["left_negative_realized_variance"]],
            [row["right_negative_realized_variance"]],
        )[0]
        self.assertAlmostEqual(value, 0.727392967453, places=11)
        zero_denominator = MODULE.component_correlation([0], [0], [1])[0]
        self.assertTrue(np.isnan(zero_denominator))

    def test_exponential_weights_match_paper_center_of_mass(self) -> None:
        result = MODULE.finite_exponential_average(
            np.array([0.0, 0.5, 1.0]),
            center_of_mass=1,
            window=3,
        )
        self.assertAlmostEqual(MODULE.center_of_mass_decay(1), 0.5)
        self.assertAlmostEqual(result[-1], 5 / 7)
        self.assertAlmostEqual(
            -math.log(MODULE.center_of_mass_decay(5)),
            math.log(1 + 1 / 5),
        )

    def test_sector_state_averages_unique_pair_correlations(self) -> None:
        dates = pd.DatetimeIndex([pd.Timestamp("2024-01-02")])
        frames = []
        ordinary = [0.8, 0.2, -0.1]
        downside = [0.9, 0.3, 0.0]
        for regular, negative in zip(ordinary, downside, strict=True):
            frame = pd.DataFrame(index=dates)
            for label in MODULE.EXPONENTIAL_CENTERS:
                frame[f"exp_rc_{label}"] = regular
                frame[f"exp_rc_negative_{label}"] = negative
            frames.append(frame)
        state, coverage = MODULE.build_sector_state(
            frames, minimum_pair_fraction=1.0
        )
        self.assertAlmostEqual(state.iloc[0]["sector_exp_rc_d"], 0.3)
        self.assertAlmostEqual(
            state.iloc[0]["sector_exp_rc_negative_d"], 0.4
        )
        self.assertAlmostEqual(coverage.iloc[0].min(), 1.0)

    def test_forecast_date_uses_only_previous_session(self) -> None:
        dates = pd.date_range("2024-01-02", periods=5, freq="D")
        base_values = [
            [0.02, -0.01],
            [0.01, -0.02],
            [0.03, -0.01],
            [0.02, -0.03],
            [0.01, -0.01],
        ]
        returns = {
            "A": interval_frame("A", dates, base_values),
            "B": interval_frame(
                "B", dates, [[0.01, -0.02]] * len(dates)
            ),
            "C": interval_frame(
                "C", dates, [[0.03, -0.01]] * len(dates)
            ),
            "ETF": interval_frame(
                "ETF", dates, [[0.02, -0.02]] * len(dates)
            ),
        }
        sector = MODULE.Sector("Test", "ETF", ("A", "B", "C"))
        first, _ = MODULE.build_core_panel(
            returns,
            [sector],
            dates,
            output_stocks=["A"],
            min_aligned_returns=1,
            min_alignment_ratio=1.0,
            require_overnight=False,
            exponential_window=3,
            min_exponential_valid=3,
            minimum_sector_pair_fraction=1.0,
            require_complete=False,
        )
        changed_returns = dict(returns)
        changed = returns["A"].copy()
        changed.loc[
            changed["trade_date"].eq(dates[3]), "log_return"
        ] = [0.5, -0.01]
        changed_returns["A"] = changed
        second, _ = MODULE.build_core_panel(
            changed_returns,
            [sector],
            dates,
            output_stocks=["A"],
            min_aligned_returns=1,
            min_alignment_ratio=1.0,
            require_overnight=False,
            exponential_window=3,
            min_exponential_valid=3,
            minimum_sector_pair_fraction=1.0,
            require_complete=False,
        )
        forecast_d4_first = first[first["forecast_date"].eq(dates[3])].iloc[0]
        forecast_d4_second = second[second["forecast_date"].eq(dates[3])].iloc[0]
        np.testing.assert_allclose(
            forecast_d4_first[MODULE.FEATURE_COLUMNS].astype(float),
            forecast_d4_second[MODULE.FEATURE_COLUMNS].astype(float),
            equal_nan=True,
        )
        forecast_d5_first = first[first["forecast_date"].eq(dates[4])].iloc[0]
        forecast_d5_second = second[second["forecast_date"].eq(dates[4])].iloc[0]
        self.assertNotAlmostEqual(
            forecast_d5_first["rc_d"], forecast_d5_second["rc_d"]
        )
        self.assertEqual(forecast_d4_first["asof_session"], dates[2])

    def test_missing_bar_does_not_create_mismatched_return(self) -> None:
        stock = make_bars(
            "A",
            [
                ("2024-01-02", ["09:30"], 100.0, [0.0]),
                (
                    "2024-01-03",
                    ["09:30", "09:45", "10:15"],
                    100.0,
                    [0.01, -0.01, 0.5],
                ),
            ],
        )
        benchmark = make_bars(
            "B",
            [
                ("2024-01-02", ["09:30"], 100.0, [0.0]),
                (
                    "2024-01-03",
                    ["09:30", "09:45", "10:00", "10:15"],
                    100.0,
                    [0.01, -0.01, 0.7, -0.8],
                ),
            ],
        )
        stock_returns = MODULE.build_interval_returns(stock)
        benchmark_returns = MODULE.build_interval_returns(benchmark)
        day = pd.Timestamp("2024-01-03")
        stock_day = stock_returns[stock_returns["trade_date"].eq(day)]
        self.assertNotIn("10:15:00", set(stock_day["interval_key"]))
        components = MODULE.pair_daily_components(
            stock_returns,
            benchmark_returns,
            min_aligned_returns=3,
            min_alignment_ratio=0.5,
        )
        row = components[components["trade_date"].eq(day)].iloc[0]
        self.assertEqual(row["aligned_return_count"], 3)
        correlation = MODULE.component_correlation(
            [row["realized_covariance"]],
            [row["left_realized_variance"]],
            [row["right_realized_variance"]],
        )[0]
        self.assertAlmostEqual(correlation, 1.0)

    def test_early_close_session_is_valid(self) -> None:
        bars = make_bars(
            "A",
            [
                ("2024-01-02", ["09:30"], 100.0, [0.0]),
                (
                    "2024-01-03",
                    ["09:30", "09:45", "10:00"],
                    100.0,
                    [0.01, -0.02, 0.01],
                ),
            ],
        )
        returns = MODULE.build_interval_returns(bars)
        day = returns[returns["trade_date"].eq(pd.Timestamp("2024-01-03"))]
        self.assertEqual(
            list(day["interval_key"]),
            ["overnight", "09:30:00", "09:45:00", "10:00:00"],
        )
        components = MODULE.pair_daily_components(
            returns,
            returns,
            min_aligned_returns=4,
            min_alignment_ratio=1.0,
        )
        row = components[
            components["trade_date"].eq(pd.Timestamp("2024-01-03"))
        ].iloc[0]
        self.assertTrue(np.isfinite(row["realized_covariance"]))

    def test_missing_session_is_not_compressed_out_of_windows(self) -> None:
        sessions = pd.date_range("2024-01-02", periods=3, freq="D")
        components = pd.DataFrame(
            {
                column: [1.0, 1.0] for column in MODULE.PAIR_COMPONENT_COLUMNS
            },
            index=[sessions[0], sessions[2]],
        )
        components.index.name = "trade_date"
        reindexed = components.reindex(sessions)
        har = MODULE.build_har_features(reindexed, horizons={"two": 2})
        exponential = MODULE.build_exponential_features(
            reindexed,
            centers={"two": 2},
            window=2,
            min_valid=2,
        )
        self.assertTrue(np.isnan(har.loc[sessions[2], "rc_two"]))
        self.assertTrue(np.isnan(exponential.loc[sessions[2], "exp_rc_two"]))

    def test_missing_whole_session_does_not_create_multiday_overnight(self) -> None:
        sessions = pd.date_range("2024-01-02", periods=3, freq="D")
        stock_bars = make_bars(
            "A",
            [
                ("2024-01-02", ["09:30"], 100.0, [0.0]),
                ("2024-01-04", ["09:30"], 102.0, [0.01]),
            ],
        )
        benchmark_bars = make_bars(
            "B",
            [
                ("2024-01-02", ["09:30"], 100.0, [0.0]),
                ("2024-01-03", ["09:30"], 101.0, [0.01]),
                ("2024-01-04", ["09:30"], 102.0, [0.01]),
            ],
        )
        stock_returns = MODULE.build_interval_returns(
            stock_bars, official_sessions=sessions
        )
        benchmark_returns = MODULE.build_interval_returns(
            benchmark_bars, official_sessions=sessions
        )
        stock_day = stock_returns[
            stock_returns["trade_date"].eq(sessions[2])
        ]
        self.assertNotIn("overnight", set(stock_day["interval_key"]))
        components = MODULE.pair_daily_components(
            stock_returns,
            benchmark_returns,
            min_aligned_returns=1,
            min_alignment_ratio=0.5,
            require_overnight=True,
        )
        row = components[components["trade_date"].eq(sessions[2])].iloc[0]
        self.assertTrue(np.isnan(row["realized_covariance"]))

    def test_sparse_leg_coverage_uses_fuller_leg(self) -> None:
        day = pd.DatetimeIndex([pd.Timestamp("2024-01-02")])
        left = interval_frame("A", day, [[0.01] * 10])
        right = interval_frame("B", day, [[0.01] * 27])
        components = MODULE.pair_daily_components(
            left,
            right,
            min_aligned_returns=10,
            min_alignment_ratio=0.8,
            require_overnight=False,
        )
        self.assertAlmostEqual(components.iloc[0]["alignment_ratio"], 10 / 27)
        self.assertTrue(np.isnan(components.iloc[0]["realized_covariance"]))

    def test_official_schedule_is_alignment_denominator(self) -> None:
        day = pd.Timestamp("2024-01-02")
        left = interval_frame("A", pd.DatetimeIndex([day]), [[0.01] * 20])
        right = interval_frame("B", pd.DatetimeIndex([day]), [[0.01] * 20])
        expected = {day: frozenset(f"i{index}" for index in range(27))}
        components = MODULE.pair_daily_components(
            left,
            right,
            min_aligned_returns=15,
            min_alignment_ratio=0.8,
            require_overnight=False,
            expected_keys=expected,
        )
        self.assertAlmostEqual(components.iloc[0]["alignment_ratio"], 20 / 27)
        self.assertTrue(np.isnan(components.iloc[0]["realized_covariance"]))

    def test_relaxed_count_still_requires_recent_exponential_weight(self) -> None:
        recent_missing = np.ones(500)
        recent_missing[-2] = np.nan
        oldest_missing = np.ones(500)
        oldest_missing[0] = np.nan
        rejected = MODULE.finite_exponential_average(
            recent_missing,
            center_of_mass=1,
            window=500,
            min_valid=490,
            minimum_weight_fraction=0.99,
        )
        accepted = MODULE.finite_exponential_average(
            oldest_missing,
            center_of_mass=1,
            window=500,
            min_valid=490,
            minimum_weight_fraction=0.99,
        )
        self.assertTrue(np.isnan(rejected[-1]))
        self.assertAlmostEqual(accepted[-1], 1.0)

    def test_snapshot_ignores_orphan_and_incomplete_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            symbol_dir = base / "A"
            symbol_dir.mkdir()
            complete = symbol_dir / "2024.csv.gz"
            incomplete = symbol_dir / "2025.csv.gz"
            orphan = symbol_dir / "2026.csv.gz"
            complete.write_bytes(b"complete")
            incomplete.write_bytes(b"incomplete")
            orphan.write_bytes(b"orphan")
            complete_manifest = {
                "status": "complete",
                "download_schema_version": 2,
                "symbol": "A",
                "request": {
                    "timeframe": "15Min",
                    "feed": "sip",
                    "adjustment": "all",
                    "regular_session_only": True,
                },
                "csv_sha256": MODULE.sha256_file(complete),
                "row_count": 1,
            }
            incomplete_manifest = dict(complete_manifest)
            incomplete_manifest["status"] = "downloading"
            MODULE.write_json_atomic(
                MODULE.manifest_path_for_csv(complete), complete_manifest
            )
            MODULE.write_json_atomic(
                MODULE.manifest_path_for_csv(incomplete), incomplete_manifest
            )
            snapshot = MODULE.snapshot_completed_chunks(base, ["A"])
        self.assertEqual(len(snapshot["A"]), 1)
        self.assertEqual(snapshot["A"][0].csv_path.name, "2024.csv.gz")

    def test_snapshot_validates_calendar_identity_over_chunk_range(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "bars"
            symbol_dir = base / "A"
            symbol_dir.mkdir(parents=True)
            csv_path = symbol_dir / "2024.csv.gz"
            csv_path.write_bytes(b"complete")
            source_calendar = root / "source_calendar.json"
            build_calendar = root / "build_calendar.json"
            source_payload = {
                "status": "complete",
                "sessions": [
                    {"date": "2024-01-02", "open": "09:30", "close": "16:00"}
                ],
            }
            build_payload = {
                "status": "complete",
                "sessions": [
                    {"date": "2024-01-02", "open": "09:30", "close": "13:00"}
                ],
            }
            MODULE.write_json_atomic(source_calendar, source_payload)
            MODULE.write_json_atomic(build_calendar, build_payload)
            manifest = {
                "status": "complete",
                "download_schema_version": 2,
                "symbol": "A",
                "chunk": {"start": "2024-01-01", "end": "2024-12-31"},
                "request": {
                    "timeframe": "15Min",
                    "feed": "sip",
                    "adjustment": "all",
                    "regular_session_only": True,
                },
                "market_calendar": {
                    "path": str(source_calendar),
                    "sha256": MODULE.sha256_file(source_calendar),
                },
                "csv_sha256": MODULE.sha256_file(csv_path),
                "row_count": 1,
            }
            MODULE.write_json_atomic(
                MODULE.manifest_path_for_csv(csv_path), manifest
            )
            with self.assertRaisesRegex(
                ValueError, "Input and build calendars disagree"
            ):
                MODULE.snapshot_completed_chunks(
                    base,
                    ["A"],
                    build_calendar_path=build_calendar,
                    project_root=root,
                )

    def test_snapshot_accepts_calendar_differences_outside_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "bars"
            symbol_dir = base / "A"
            symbol_dir.mkdir(parents=True)
            csv_path = symbol_dir / "2024.csv.gz"
            csv_path.write_bytes(b"complete")
            source_calendar = root / "source_calendar.json"
            build_calendar = root / "build_calendar.json"
            shared = {
                "date": "2024-01-02",
                "open": "09:30",
                "close": "16:00",
            }
            MODULE.write_json_atomic(
                source_calendar,
                {
                    "status": "complete",
                    "sessions": [
                        {
                            "date": "2023-12-29",
                            "open": "09:30",
                            "close": "16:00",
                        },
                        shared,
                    ],
                },
            )
            MODULE.write_json_atomic(
                build_calendar,
                {
                    "status": "complete",
                    "sessions": [
                        shared,
                        {
                            "date": "2025-01-02",
                            "open": "09:30",
                            "close": "16:00",
                        },
                    ],
                },
            )
            manifest = {
                "status": "complete",
                "download_schema_version": 2,
                "symbol": "A",
                "chunk": {"start": "2024-01-01", "end": "2024-12-31"},
                "request": {
                    "timeframe": "15Min",
                    "feed": "sip",
                    "adjustment": "all",
                    "regular_session_only": True,
                },
                "market_calendar": {
                    "path": str(source_calendar),
                    "sha256": MODULE.sha256_file(source_calendar),
                },
                "csv_sha256": MODULE.sha256_file(csv_path),
                "row_count": 1,
            }
            MODULE.write_json_atomic(
                MODULE.manifest_path_for_csv(csv_path), manifest
            )
            snapshot = MODULE.snapshot_completed_chunks(
                base,
                ["A"],
                build_calendar_path=build_calendar,
                project_root=root,
            )
        self.assertEqual(len(snapshot["A"]), 1)

    def test_snapshot_rejects_calendar_hash_mismatch_and_empty_range(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "bars"
            symbol_dir = base / "A"
            symbol_dir.mkdir(parents=True)
            csv_path = symbol_dir / "2030.csv.gz"
            csv_path.write_bytes(b"complete")
            calendar = root / "calendar.json"
            MODULE.write_json_atomic(
                calendar,
                {
                    "status": "complete",
                    "sessions": [
                        {
                            "date": "2024-01-02",
                            "open": "09:30",
                            "close": "16:00",
                        }
                    ],
                },
            )
            manifest = {
                "status": "complete",
                "download_schema_version": 2,
                "symbol": "A",
                "chunk": {"start": "2030-01-01", "end": "2030-12-31"},
                "request": {
                    "timeframe": "15Min",
                    "feed": "sip",
                    "adjustment": "all",
                    "regular_session_only": True,
                },
                "market_calendar": {
                    "path": str(calendar),
                    "sha256": "0" * 64,
                },
                "csv_sha256": MODULE.sha256_file(csv_path),
                "row_count": 1,
            }
            manifest_path = MODULE.manifest_path_for_csv(csv_path)
            MODULE.write_json_atomic(manifest_path, manifest)
            with self.assertRaisesRegex(
                ValueError, "Input calendar hash mismatch"
            ):
                MODULE.snapshot_completed_chunks(
                    base,
                    ["A"],
                    build_calendar_path=calendar,
                    project_root=root,
                )
            manifest["market_calendar"]["sha256"] = MODULE.sha256_file(calendar)
            MODULE.write_json_atomic(manifest_path, manifest)
            with self.assertRaisesRegex(
                ValueError, "Calendar has no sessions in chunk range"
            ):
                MODULE.snapshot_completed_chunks(
                    base,
                    ["A"],
                    build_calendar_path=calendar,
                    project_root=root,
                )

    def test_atomic_csv_output_and_manifest_naming(self) -> None:
        frame = pd.DataFrame({"value": [1.0, 2.0]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.csv.gz"
            MODULE.write_frame_atomic(path, frame)
            restored = pd.read_csv(path)
            manifest_path = MODULE.output_manifest_path(path)
        pd.testing.assert_frame_equal(restored, frame)
        self.assertEqual(manifest_path.name, "features.manifest.json")


if __name__ == "__main__":
    unittest.main()
