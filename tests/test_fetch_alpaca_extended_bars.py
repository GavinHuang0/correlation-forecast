from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from datetime import date, time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "fetch_alpaca_extended_bars", SCRIPTS / "fetch_alpaca_extended_bars.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakeClient:
    def __init__(self, pages):
        self.pages = list(pages)
        self.calls = []

    def get_json(self, params):
        self.calls.append(dict(params))
        return self.pages.pop(0)


class FetchExtendedBarsTests(unittest.TestCase):
    def test_window_bounds_respect_early_close(self) -> None:
        regular = MODULE.MarketSession(time(9, 30), time(16, 0))
        early = MODULE.MarketSession(time(9, 30), time(13, 0))

        self.assertEqual(
            MODULE.request_window(
                date(2024, 1, 2),
                regular,
                "premarket",
                premarket_start=time(4),
                premarket_cutoff=time(9),
                aftermarket_end=time(20),
            ),
            (time(4), time(9)),
        )
        self.assertEqual(
            MODULE.request_window(
                date(2024, 11, 29),
                early,
                "aftermarket",
                premarket_start=time(4),
                premarket_cutoff=time(9),
                aftermarket_end=time(20),
            ),
            (time(13), time(20)),
        )

    def test_utc_bounds_follow_daylight_saving_time(self) -> None:
        self.assertEqual(
            MODULE.utc_bound(date(2024, 1, 2), time(4)),
            "2024-01-02T09:00:00Z",
        )
        self.assertEqual(
            MODULE.utc_bound(date(2024, 7, 1), time(4)),
            "2024-07-01T08:00:00Z",
        )

    def test_multi_symbol_fetch_uses_narrow_window_and_filters_rows(self) -> None:
        client = FakeClient(
            [
                (
                    {
                        "bars": {
                            "AMD": [
                                {
                                    "t": "2024-01-02T09:00:00Z",
                                    "o": 1,
                                    "h": 2,
                                    "l": 1,
                                    "c": 2,
                                    "v": 10,
                                    "n": 2,
                                    "vw": 1.5,
                                },
                                {
                                    "t": "2024-01-02T14:00:00Z",
                                    "o": 2,
                                    "h": 2,
                                    "l": 2,
                                    "c": 2,
                                    "v": 1,
                                    "n": 1,
                                    "vw": 2,
                                },
                            ],
                            "SOXX": [
                                {
                                    "t": "2024-01-02T13:45:00Z",
                                    "o": 3,
                                    "h": 3,
                                    "l": 3,
                                    "c": 3,
                                    "v": 5,
                                    "n": 1,
                                    "vw": 3,
                                }
                            ],
                        }
                    },
                    "request-1",
                )
            ]
        )

        rows, request_ids = MODULE.fetch_window(
            client,
            symbols=["AMD", "SOXX"],
            trade_date=date(2024, 1, 2),
            session_name="premarket",
            start_time=time(4),
            end_time=time(9),
            timeframe="15Min",
            feed="sip",
            adjustment="all",
        )

        self.assertEqual([row["symbol"] for row in rows], ["AMD", "SOXX"])
        self.assertEqual(request_ids, ["request-1"])
        self.assertEqual(client.calls[0]["symbols"], "AMD,SOXX")
        self.assertEqual(client.calls[0]["start"], "2024-01-02T09:00:00Z")
        self.assertEqual(client.calls[0]["end"], "2024-01-02T13:59:59.999999Z")
        self.assertEqual({row["session"] for row in rows}, {"premarket"})

    def test_calendar_cache_is_read_only_and_subsets_requested_dates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calendar.json"
            path.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "sessions": [
                            {"date": "2024-01-02", "open": "09:30", "close": "16:00"},
                            {"date": "2024-01-03", "open": "09:30", "close": "16:00"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            before = path.read_bytes()
            sessions = MODULE.load_calendar_cache(
                path, start=date(2024, 1, 3), end=date(2024, 1, 3)
            )

            self.assertEqual(list(sessions), [date(2024, 1, 3)])
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
