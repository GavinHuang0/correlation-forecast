from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_deterministic_news_features.py"
SPEC = importlib.util.spec_from_file_location(
    "build_deterministic_news_features",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def article(
    *,
    title: str,
    url: str,
    timestamp: str,
    summary: str,
    tickers: list[str],
    topics: list[str] | None = None,
    source: str = "Test Wire",
) -> dict[str, object]:
    return {
        "title": title,
        "url": url,
        "time_published": timestamp,
        "authors": [],
        "summary": summary,
        "banner_image": None,
        "source": source,
        "category_within_source": "General",
        "source_domain": source,
        "topics": [
            {"topic": topic, "relevance_score": "1.0"}
            for topic in (topics or [])
        ],
        "overall_sentiment_score": 0.0,
        "overall_sentiment_label": "Neutral",
        "ticker_sentiment": [
            {
                "ticker": ticker,
                "relevance_score": "1.0",
                "ticker_sentiment_score": "0.0",
                "ticker_sentiment_label": "Neutral",
            }
            for ticker in tickers
        ],
    }


class DeterministicNewsFeatureTests(unittest.TestCase):
    def test_midnight_timestamp_is_delayed_and_timing_ineligible(self) -> None:
        published, precision = MODULE.parse_alpha_timestamp("20240102T000000")
        self.assertEqual(precision, "date_only_proxy")
        available = MODULE.conservative_available_at(published, precision)
        self.assertEqual(
            available,
            datetime(2024, 1, 3, 0, 0, tzinfo=timezone.utc),
        )

    def test_forecast_assignment_obeys_nine_am_cutoff(self) -> None:
        sessions = [
            MODULE.Session(
                date=date(2024, 1, 2),
                open_time=time(9, 30),
                close_time=time(16, 0),
                cutoff_utc=datetime(2024, 1, 2, 14, 0, tzinfo=timezone.utc),
            ),
            MODULE.Session(
                date=date(2024, 1, 3),
                open_time=time(9, 30),
                close_time=time(16, 0),
                cutoff_utc=datetime(2024, 1, 3, 14, 0, tzinfo=timezone.utc),
            ),
        ]
        before = datetime(2024, 1, 2, 13, 59, tzinfo=timezone.utc)
        after = datetime(2024, 1, 2, 14, 1, tzinfo=timezone.utc)
        self.assertEqual(MODULE.assign_forecast_session(before, sessions).date, date(2024, 1, 2))
        self.assertEqual(MODULE.assign_forecast_session(after, sessions).date, date(2024, 1, 3))

    def test_end_to_end_pilot_deduplicates_and_marks_not_training_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw = root / "raw"
            raw.mkdir()
            direct = article(
                title="AMD raises guidance above expectations",
                url="https://example.com/amd-guidance",
                timestamp="20240102T130000",
                summary="AMD raised guidance above analyst expectations.",
                tickers=["AMD"],
                topics=["earnings"],
            )
            peer = article(
                title="Nvidia launches new product",
                url="https://example.com/nvidia-product",
                timestamp="20240102T150000",
                summary="Nvidia launched a new product.",
                tickers=["NVDA"],
                topics=["technology"],
            )
            macro = article(
                title="Federal Reserve policy outlook",
                url="https://example.com/fed",
                timestamp="20240102T000000",
                summary="Federal Reserve officials discussed interest rates.",
                tickers=[],
                topics=["economy_monetary"],
            )
            (raw / "news_2024_ticker_AMD.json").write_text(
                json.dumps({"items": "2", "feed": [direct, peer]}),
                encoding="utf-8",
            )
            (raw / "news_2024_topic_economy_monetary.json").write_text(
                json.dumps({"items": "2", "feed": [direct, macro]}),
                encoding="utf-8",
            )
            universe = root / "universe.json"
            universe.write_text(
                json.dumps(
                    {
                        "sector": "Semiconductors",
                        "sector_benchmark": "SOXX",
                        "targets": {
                            "AMD": {
                                "company": "Advanced Micro Devices",
                                "peers": ["NVDA"],
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            calendar = root / "calendar.json"
            calendar.write_text(
                json.dumps(
                    {
                        "sessions": [
                            {
                                "date": "2024-01-02",
                                "open": "09:30",
                                "close": "16:00",
                            },
                            {
                                "date": "2024-01-03",
                                "open": "09:30",
                                "close": "16:00",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            output = root / "output"
            manifest = MODULE.build(
                raw_directory=raw,
                universe_path=universe,
                calendar_path=calendar,
                output_directory=output,
                start=date(2024, 1, 2),
                end=date(2024, 1, 3),
            )

            self.assertEqual(manifest["counts"]["raw_query_rows"], 4)
            self.assertEqual(manifest["counts"]["unique_exact_url_articles"], 3)
            self.assertEqual(
                manifest["counts"]["cross_query_duplicate_rows_removed"], 1
            )
            self.assertEqual(
                manifest["coarse_rule_version"],
                MODULE.coarse.DETERMINISTIC_RULE_VERSION,
            )
            self.assertEqual(len(manifest["builder_script_sha256"]), 64)
            self.assertEqual(len(manifest["coarse_module_sha256"]), 64)
            with gzip.open(
                output / "stock_day_features.csv.gz",
                "rt",
                encoding="utf-8",
                newline="",
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            first, second = rows
            self.assertEqual(first["forecast_date"], "2024-01-02")
            self.assertEqual(first["primary_training_eligible"], "0")
            self.assertEqual(first["observed_direct_target_article_count"], "1")
            self.assertEqual(first["positive_surprise_cue_article_count"], "1")
            self.assertEqual(second["forecast_date"], "2024-01-03")
            self.assertGreaterEqual(
                int(second["observed_peer_specific_article_count"]),
                1,
            )
            self.assertGreaterEqual(
                int(second["timing_imprecise_article_count"]),
                1,
            )
            original_hash = hashlib.sha256(
                (output / "stock_day_features.csv.gz").read_bytes()
            ).hexdigest()
            MODULE.build(
                raw_directory=raw,
                universe_path=universe,
                calendar_path=calendar,
                output_directory=output,
                start=date(2024, 1, 2),
                end=date(2024, 1, 3),
                overwrite=True,
            )
            rebuilt_hash = hashlib.sha256(
                (output / "stock_day_features.csv.gz").read_bytes()
            ).hexdigest()
            self.assertEqual(original_hash, rebuilt_hash)
            dry_run_manifest = MODULE.build(
                raw_directory=raw,
                universe_path=universe,
                calendar_path=calendar,
                output_directory=output,
                start=date(2024, 1, 2),
                end=date(2024, 1, 3),
                dry_run=True,
            )
            self.assertTrue(dry_run_manifest["dry_run"])

    def test_peer_only_article_is_not_a_common_shock(self) -> None:
        target = MODULE.Target(
            ticker="AMD",
            company="Advanced Micro Devices",
            sector="Semiconductors",
            benchmark="SOXX",
            peers=("NVDA",),
        )
        session = MODULE.Session(
            date=date(2024, 1, 2),
            open_time=time(9, 30),
            close_time=time(16, 0),
            cutoff_utc=datetime(2024, 1, 2, 14, 0, tzinfo=timezone.utc),
        )
        raw_article = {
            "article_id": "peer-only",
            "url": "https://example.com/peer-only",
            "title": "Nvidia launches a new product",
            "summary": "Nvidia launched a new product.",
            "text": "Nvidia launches a new product\nNvidia launched a new product.",
            "published_at": datetime(2024, 1, 2, 13, 0, tzinfo=timezone.utc),
            "published_at_utc": "2024-01-02T13:00:00Z",
            "available_at": datetime(2024, 1, 2, 13, 0, tzinfo=timezone.utc),
            "conservative_available_at_utc": "2024-01-02T13:00:00Z",
            "timestamp_precision": "second",
            "source": "Test",
            "publisher_host": "example.com",
            "topics": ["technology"],
            "tickers": ["NVDA"],
            "query_files": ["test.json"],
            "query_copy_count": 1,
            "url_sha256": "a" * 64,
            "text_sha256": "b" * 64,
            "normalized_title_cluster_id": "title_test",
            "positive_surprise_cue": False,
            "negative_surprise_cue": False,
            "family_cues": {
                family: family == "product_demand"
                for family in MODULE.EVENT_FAMILY_PATTERNS
            },
            "government_primary_source": False,
            "press_release_wire_source": False,
        }
        rows, _ = MODULE.build_pair_rows(
            [raw_article],
            [target],
            [session],
            date(2024, 1, 2),
            date(2024, 1, 2),
        )
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["peer_specific_proxy"])
        self.assertFalse(rows[0]["common_proxy"])

    def test_burst_requires_exactly_sixty_prior_sessions(self) -> None:
        rows = []
        start = date(2024, 1, 1)
        for index in range(62):
            rows.append(
                {
                    "stock": "AMD",
                    "forecast_date": (start + timedelta(days=index)).isoformat(),
                    "observed_direct_target_article_count": index % 2,
                    "observed_target_news_burst_60_session": None,
                    "target_news_burst_history_count": 0,
                }
            )
        MODULE.apply_target_news_burst(rows)
        self.assertIsNone(rows[59]["observed_target_news_burst_60_session"])
        self.assertEqual(rows[59]["target_news_burst_history_count"], 59)
        self.assertIsNotNone(rows[60]["observed_target_news_burst_60_session"])
        self.assertEqual(rows[60]["target_news_burst_history_count"], 60)

    def test_rejects_flan_output_destination(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.validate_output_directory(
                ROOT / "outputs" / "flan_t5_xl" / "deterministic"
            )


if __name__ == "__main__":
    unittest.main()
