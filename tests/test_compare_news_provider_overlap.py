from __future__ import annotations

import gzip
import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from scripts import compare_news_provider_overlap as overlap


class CompareNewsProviderOverlapTests(unittest.TestCase):
    def test_input_root_fingerprint_hashes_file_content_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            response = root / "pages" / "AMD" / "page_000001.json.gz"
            response.parent.mkdir(parents=True)
            response.write_bytes(b"first")
            manifest = root / "manifest.json"
            manifest.write_text('{"status":"complete"}\n', encoding="utf-8")

            first = overlap.input_root_fingerprint(root, "pages")
            self.assertEqual(first["cached_file_count"], 1)
            self.assertEqual(
                first["manifest_sha256"],
                hashlib.sha256(manifest.read_bytes()).hexdigest(),
            )

            response.write_bytes(b"other")
            second = overlap.input_root_fingerprint(root, "pages")
            self.assertNotEqual(
                first["cached_files_sha256"],
                second["cached_files_sha256"],
            )

    def test_matches_normalized_url_then_title_and_time(self) -> None:
        massive = {
            "one": {
                "article_url": "https://example.com/a?utm_source=x",
                "title": "First story",
                "published_utc": "2026-03-02T12:00:00Z",
                "description": "short",
                "tickers": ["AMD"],
            },
            "two": {
                "article_url": "https://other.example/b",
                "title": "Second: Story!",
                "published_utc": "2026-03-02T13:00:00Z",
                "description": "longer text",
                "tickers": ["NVDA"],
            },
        }
        alpha = {
            "https://example.com/a": {
                "url": "https://example.com/a",
                "title": "Different syndicated title",
                "time_published": "20260302T120000",
                "summary": "alpha one",
            },
            "https://alpha.example/b": {
                "url": "https://alpha.example/b",
                "title": "second story",
                "time_published": "20260302T130000",
                "summary": "alpha two",
            },
        }
        report = overlap.compare(massive, alpha)
        self.assertEqual(report["matched_unique_documents"], 2)
        self.assertEqual(report["match_methods"]["normalized_url"], 1)
        self.assertEqual(
            report["match_methods"]["normalized_title_and_exact_second"], 1
        )
        self.assertEqual(
            report["matched_by_massive_ticker_tag"], {"AMD": 1, "NVDA": 1}
        )

    def test_alpha_loader_excludes_reported_ceiling_with_short_feed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "responses" / "AMD" / "truncated.json.gz"
            path.parent.mkdir(parents=True)
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                json.dump(
                    {
                        "items": "1000",
                        "feed": [
                            {
                                "url": "https://example.com/truncated",
                                "title": "Truncated",
                                "time_published": "20260302T120000",
                            }
                        ],
                    },
                    handle,
                )

            records = overlap.load_alpha(
                root, datetime(2026, 3, 1, tzinfo=timezone.utc)
            )

        self.assertEqual(records, {})


if __name__ == "__main__":
    unittest.main()
