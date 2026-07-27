from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from scripts import build_fulltext_candidate_queue as queue


class BuildFulltextCandidateQueueTests(unittest.TestCase):
    def test_filters_cutoff_deduplicates_and_round_robins(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            common = {
                "id": "common",
                "article_url": "https://example.com/common",
                "title": "Common",
                "description": "A sufficiently present description.",
                "published_utc": "2026-03-02T00:00:00Z",
                "tickers": ["AAA", "BBB"],
            }
            old = {
                **common,
                "id": "old",
                "published_utc": "2026-03-01T00:00:00Z",
            }
            aaa = {
                **common,
                "id": "aaa",
                "article_url": "https://example.com/aaa",
                "title": "AAA",
                "tickers": ["AAA"],
            }
            bbb = {
                **common,
                "id": "bbb",
                "article_url": "https://example.com/bbb",
                "title": "BBB",
                "tickers": ["BBB"],
            }
            for ticker, rows in (
                ("AAA", [common, old, aaa]),
                ("BBB", [common, bbb]),
            ):
                path = root / "pages" / ticker / "page_000001.json.gz"
                path.parent.mkdir(parents=True, exist_ok=True)
                with gzip.open(path, "wt", encoding="utf-8") as handle:
                    json.dump({"results": rows}, handle)

            candidates = queue.read_candidates(
                root,
                tickers=["AAA", "BBB"],
                cutoff=datetime(2026, 3, 1, tzinfo=timezone.utc),
            )
            self.assertEqual({row["id"] for row in candidates}, {"common", "aaa", "bbb"})
            ordered = queue.balanced_order(candidates, ["AAA", "BBB"])
            self.assertEqual(len(ordered), 3)
            self.assertEqual(
                {row["queue_primary_ticker"] for row in ordered[:2]},
                {"AAA", "BBB"},
            )
            self.assertEqual(len({row["id"] for row in ordered}), 3)

    def test_multiple_roots_expand_economically_valid_targets_with_provenance(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stock_root = root / "stocks"
            benchmark_root = root / "benchmarks"

            def write(
                collection: Path, query: str, rows: list[dict[str, object]]
            ) -> None:
                path = (
                    collection
                    / "pages"
                    / query
                    / "page_000001.json.gz"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                with gzip.open(path, "wt", encoding="utf-8") as handle:
                    json.dump({"results": rows}, handle)

            def article(
                article_id: str, *, tickers: list[str]
            ) -> dict[str, object]:
                return {
                    "id": article_id,
                    "article_url": f"https://example.com/{article_id}",
                    "title": f"Article {article_id}",
                    "description": "A complete synthetic description.",
                    "published_utc": "2026-03-02T00:00:00Z",
                    "tickers": tickers,
                }

            direct = article("direct", tickers=["AAA"])
            duplicate = article("duplicate", tickers=["AAA"])
            write(stock_root, "AAA", [direct, duplicate])
            # The duplicate verifies that query provenance merges across roots.
            write(benchmark_root, "SXA", [duplicate, article("sector", tickers=["SXA"])])
            write(benchmark_root, "SPY", [article("market", tickers=["SPY"])])

            candidates = queue.read_candidates(
                [stock_root, benchmark_root],
                tickers=["AAA", "AAB", "BBB", "BBC"],
                cutoff=datetime(2026, 3, 1, tzinfo=timezone.utc),
                benchmark_by_ticker={
                    "AAA": "SXA",
                    "AAB": "SXA",
                    "BBB": "SXB",
                    "BBC": "SXB",
                },
                stocks_by_benchmark={
                    "SXA": ["AAA", "AAB"],
                    "SXB": ["BBB", "BBC"],
                },
                controls=["SPY"],
            )
            by_id = {row["id"]: row for row in candidates}

            self.assertEqual(
                by_id["direct"]["eligible_target_tickers"], ["AAA", "AAB"]
            )
            self.assertNotIn("BBB", by_id["direct"]["eligible_target_tickers"])
            self.assertIn(
                "direct_stock_query",
                {
                    item["reason"]
                    for item in by_id["direct"]["eligibility_provenance"]["AAA"]
                },
            )
            self.assertIn(
                "same_sector_peer",
                {
                    item["reason"]
                    for item in by_id["direct"]["eligibility_provenance"]["AAB"]
                },
            )
            self.assertEqual(
                by_id["sector"]["eligible_target_tickers"], ["AAA", "AAB"]
            )
            self.assertTrue(
                all(
                    any(
                        item["reason"] == "sector_benchmark_query"
                        for item in by_id["sector"]["eligibility_provenance"][
                            target
                        ]
                    )
                    for target in ("AAA", "AAB")
                )
            )
            self.assertEqual(
                by_id["market"]["eligible_target_tickers"],
                ["AAA", "AAB", "BBB", "BBC"],
            )
            self.assertTrue(
                all(
                    by_id["market"]["eligibility_provenance"][target][0][
                        "reason"
                    ]
                    == "market_control_query"
                    for target in ("AAA", "AAB", "BBB", "BBC")
                )
            )
            self.assertEqual(
                by_id["duplicate"]["query_tickers"], ["AAA", "SXA"]
            )
            self.assertEqual(
                {
                    item["query_ticker"]
                    for item in by_id["duplicate"]["query_provenance"]
                },
                {"AAA", "SXA"},
            )

    def test_load_universe_preserves_sector_and_control_structure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "universe.json"
            path.write_text(
                json.dumps(
                    {
                        "controls": ["SPY"],
                        "sectors": [
                            {
                                "name": "One",
                                "benchmark": "SXA",
                                "stocks": ["AAA", "AAB"],
                            },
                            {
                                "name": "Two",
                                "benchmark": "SXB",
                                "stocks": ["BBB", "BBC"],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            universe = queue.load_universe(path)
            self.assertEqual(
                universe["benchmark_by_ticker"]["AAB"], "SXA"
            )
            self.assertEqual(
                universe["stocks_by_benchmark"]["SXB"], ["BBB", "BBC"]
            )
            self.assertEqual(universe["controls"], ["SPY"])


if __name__ == "__main__":
    unittest.main()
