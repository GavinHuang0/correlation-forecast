from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "fetch_alpaca_bars.py"
SPEC = importlib.util.spec_from_file_location("fetch_alpaca_bars", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FetchAlpacaBarsTests(unittest.TestCase):
    def test_price_universe_has_30_unique_stocks_and_five_benchmarks(self) -> None:
        universe = MODULE.load_universe(
            REPOSITORY_ROOT / "config" / "price_universe.json"
        )
        stocks = [
            symbol for sector in universe["sectors"] for symbol in sector["stocks"]
        ]
        benchmarks = [sector["benchmark"] for sector in universe["sectors"]]

        self.assertEqual(len(stocks), 30)
        self.assertEqual(len(set(stocks)), 30)
        self.assertEqual(len(benchmarks), 5)
        self.assertEqual(len(set(benchmarks)), 5)
        self.assertEqual(universe["feed"], "sip")
        self.assertEqual(universe["timeframe"], "15Min")

    def test_symbols_from_universe_are_unique_and_include_control(self) -> None:
        universe = json.loads(
            (REPOSITORY_ROOT / "config" / "price_universe.json").read_text(
                encoding="utf-8"
            )
        )
        symbols = MODULE.symbols_from_universe(universe)

        self.assertEqual(len(symbols), 36)
        self.assertEqual(len(symbols), len(set(symbols)))
        self.assertTrue({"AMD", "SOXX", "SPY"}.issubset(symbols))

    def test_yearly_chunks_cover_inclusive_range(self) -> None:
        chunks = MODULE.yearly_chunks(date(2023, 8, 1), date(2025, 2, 3))

        self.assertEqual(
            chunks,
            [
                MODULE.DateChunk(date(2023, 8, 1), date(2023, 12, 31)),
                MODULE.DateChunk(date(2024, 1, 1), date(2024, 12, 31)),
                MODULE.DateChunk(date(2025, 1, 1), date(2025, 2, 3)),
            ],
        )
        self.assertEqual(
            [chunk.label for chunk in chunks],
            [
                "2023-08-01_2023-12-31",
                "2024",
                "2025-01-01_2025-02-03",
            ],
        )

    def test_regular_session_filter_handles_standard_and_daylight_time(self) -> None:
        self.assertTrue(
            MODULE.is_regular_session_bar(
                MODULE.parse_timestamp("2024-01-02T14:30:00Z")
            )
        )
        self.assertTrue(
            MODULE.is_regular_session_bar(
                MODULE.parse_timestamp("2024-07-01T13:30:00Z")
            )
        )
        self.assertFalse(
            MODULE.is_regular_session_bar(
                MODULE.parse_timestamp("2024-01-02T14:15:00Z")
            )
        )
        self.assertFalse(
            MODULE.is_regular_session_bar(
                MODULE.parse_timestamp("2024-07-01T20:00:00Z")
            )
        )

    def test_market_calendar_filter_excludes_early_close_bars(self) -> None:
        sessions = MODULE.normalize_market_calendar(
            [{"date": "2016-11-25", "open": "09:30", "close": "13:00"}]
        )

        self.assertTrue(
            MODULE.is_regular_session_bar(
                MODULE.parse_timestamp("2016-11-25T17:45:00Z"), sessions
            )
        )
        self.assertFalse(
            MODULE.is_regular_session_bar(
                MODULE.parse_timestamp("2016-11-25T18:00:00Z"), sessions
            )
        )

    def test_normalize_bar_preserves_utc_and_eastern_timestamps(self) -> None:
        row = MODULE.normalize_bar(
            "AMD",
            {
                "t": "2024-01-02T14:30:00Z",
                "o": 1.0,
                "h": 2.0,
                "l": 0.5,
                "c": 1.5,
                "v": 100,
                "n": 20,
                "vw": 1.4,
            },
        )

        self.assertEqual(row["symbol"], "AMD")
        self.assertEqual(row["timestamp_utc"], "2024-01-02T14:30:00Z")
        self.assertTrue(row["timestamp_et"].startswith("2024-01-02T09:30:00"))
        self.assertEqual(row["trade_date"], "2024-01-02")
        self.assertEqual(row["bar_start_et"], "09:30:00")

    def test_dotenv_parser_does_not_modify_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text(
                "ALPACA_PUBLIC_KEY='public-value'\n"
                'export ALPACA_SECRET_KEY="secret-value"\n',
                encoding="utf-8",
            )

            values = MODULE.load_dotenv_values(env_file)

        self.assertEqual(
            values,
            {
                "ALPACA_PUBLIC_KEY": "public-value",
                "ALPACA_SECRET_KEY": "secret-value",
            },
        )


if __name__ == "__main__":
    unittest.main()
