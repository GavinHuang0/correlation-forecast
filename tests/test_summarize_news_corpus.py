from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from scripts import summarize_news_corpus as summary


class SummarizeNewsCorpusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _write_gzip(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump(value, handle)

    def test_massive_deduplicates_cross_ticker_results(self) -> None:
        record = {
            "id": "same",
            "published_utc": "2026-03-02T12:00:00Z",
            "description": "Description",
            "article_url": "https://example.com/same",
            "tickers": ["AMD", "NVDA"],
            "publisher": {"name": "Example"},
            "keywords": ["chips"],
            "insights": [{"ticker": "AMD"}],
        }
        self._write_gzip(
            self.root / "pages" / "AMD" / "page_000001.json.gz",
            {"results": [record]},
        )
        self._write_gzip(
            self.root / "pages" / "NVDA" / "page_000001.json.gz",
            {"results": [record]},
        )
        report = summary.summarize_massive(
            self.root, datetime(2026, 3, 1, tzinfo=timezone.utc)
        )
        self.assertEqual(report["query_result_rows"], 2)
        self.assertEqual(report["unique_documents"], 1)
        self.assertEqual(report["duplicate_query_rows"], 1)
        self.assertEqual(report["post_cutoff"]["by_query_ticker"], {"AMD": 1, "NVDA": 1})
        self.assertEqual(report["field_coverage"]["description"]["rate"], 1.0)

    def test_alpha_ignores_ceiling_pages_for_unique_coverage(self) -> None:
        ceiling = [
            {
                "url": f"https://example.com/{index}",
                "time_published": "20260302T120000",
                "summary": "Text",
                "source": "Example",
            }
            for index in range(1_000)
        ]
        accepted = [ceiling[0]]
        self._write_gzip(
            self.root / "responses" / "AMD" / "wide.json.gz",
            {"feed": ceiling},
        )
        self._write_gzip(
            self.root / "responses" / "AMD" / "narrow.json.gz",
            {"feed": accepted},
        )
        report = summary.summarize_alpha(
            self.root, datetime(2026, 3, 1, tzinfo=timezone.utc)
        )
        self.assertEqual(report["query_result_rows"], 1_001)
        self.assertEqual(report["accepted_query_result_rows"], 1)
        self.assertEqual(report["unique_documents"], 1)
        self.assertEqual(report["ceiling_response_count"], 1)
        self.assertEqual(report["field_coverage"]["summary"]["rate"], 1.0)

    def test_alpha_honors_reported_ceiling_when_feed_is_short(self) -> None:
        record = {
            "url": "https://example.com/truncated",
            "time_published": "20260302T120000",
            "summary": "Text",
            "source": "Example",
        }
        self._write_gzip(
            self.root / "responses" / "AMD" / "truncated.json.gz",
            {"items": "1000", "feed": [record]},
        )

        report = summary.summarize_alpha(
            self.root, datetime(2026, 3, 1, tzinfo=timezone.utc)
        )

        self.assertEqual(report["query_result_rows"], 1)
        self.assertEqual(report["accepted_query_result_rows"], 0)
        self.assertEqual(report["unique_documents"], 0)
        self.assertEqual(report["ceiling_response_count"], 1)

    def test_post_cutoff_relation_is_strict(self) -> None:
        exact = {
            "id": "exact",
            "published_utc": "2026-03-01T00:00:00Z",
        }
        after = {
            "id": "after",
            "published_utc": "2026-03-01T00:00:01Z",
        }
        self._write_gzip(
            self.root / "pages" / "AMD" / "page_000001.json.gz",
            {"results": [exact, after]},
        )

        report = summary.summarize_massive(
            self.root, datetime(2026, 3, 1, tzinfo=timezone.utc)
        )

        self.assertEqual(report["post_cutoff"]["unique_documents"], 1)
        self.assertEqual(
            report["post_cutoff"]["cutoff_relation"], "strictly_after"
        )


if __name__ == "__main__":
    unittest.main()
