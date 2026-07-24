from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SCRIPT = ROOT / "scripts" / "fetch_alpha_vantage_earnings.py"
SPEC = importlib.util.spec_from_file_location("fetch_alpha_vantage_earnings", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class AlphaVantageEarningsTests(unittest.TestCase):
    def test_parser_filters_dates_and_marks_timing_unknown(self) -> None:
        raw = json.dumps(
            {
                "symbol": "AMD",
                "quarterlyEarnings": [
                    {
                        "fiscalDateEnding": "2024-03-31",
                        "reportedDate": "2024-04-30",
                        "reportedEPS": "0.62",
                        "estimatedEPS": "0.61",
                        "surprise": "0.01",
                        "surprisePercentage": "1.64",
                    },
                    {
                        "fiscalDateEnding": "2023-12-31",
                        "reportedDate": "2024-01-30",
                    },
                ],
            }
        ).encode()
        rows = MODULE.parse_earnings(
            raw,
            symbol="AMD",
            start=date(2024, 4, 1),
            end=date(2024, 12, 31),
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["reported_date"], "2024-04-30")
        self.assertEqual(rows[0]["announcement_time_known"], "false")
        self.assertEqual(rows[0]["point_in_time_schedule_available"], "false")

    def test_provider_rate_limit_message_is_not_accepted_as_data(self) -> None:
        raw = b'{"Information":"Thank you for using Alpha Vantage."}'
        with self.assertRaises(RuntimeError):
            MODULE.parse_earnings(
                raw,
                symbol="AMD",
                start=date(2024, 1, 1),
                end=date(2024, 12, 31),
            )

    def test_complete_cache_requires_matching_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            raw_path, manifest_path = MODULE.symbol_paths(output, "AMD")
            raw_path.parent.mkdir(parents=True)
            raw_path.write_bytes(b'{"quarterlyEarnings":[]}')
            manifest_path.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "raw_sha256": MODULE.sha256_file(raw_path),
                    }
                ),
                encoding="utf-8",
            )

            self.assertTrue(MODULE.is_complete(raw_path, manifest_path))
            raw_path.write_bytes(b"changed")
            self.assertFalse(MODULE.is_complete(raw_path, manifest_path))


if __name__ == "__main__":
    unittest.main()
