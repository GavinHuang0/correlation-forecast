from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_massive_deterministic_news_features.py"
SPEC = importlib.util.spec_from_file_location(
    "build_massive_deterministic_news_features", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "massive_ordinary_news"
EMPTY_PAGE = (
    json.dumps(
        {"request_id": "empty-fixture", "status": "OK", "results": []},
        sort_keys=True,
        separators=(",", ":"),
    )
    + "\n"
).encode("utf-8")
SPY_PAGE = (
    json.dumps(
        {
            "request_id": "spy-fixture",
            "status": "OK",
            "results": [
                {
                    "id": "massive-fixture-004",
                    "publisher": {
                        "name": "Macro News",
                        "homepage_url": "https://macro.example.com/",
                    },
                    "title": "Federal Reserve discusses interest rates",
                    "published_utc": "2024-01-02T13:30:00Z",
                    "article_url": "https://example.com/fed-rates",
                    "tickers": ["SPY"],
                    "description": (
                        "Federal Reserve officials discussed monetary policy."
                    ),
                    "keywords": ["Federal Reserve", "interest rates"],
                    "insights": [],
                }
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    + "\n"
).encode("utf-8")

BANK_COMMON_PAGE = (
    json.dumps(
        {
            "request_id": "bank-common-fixture",
            "status": "OK",
            "results": [
                {
                    "id": "massive-fixture-bank-common",
                    "publisher": {
                        "name": "Bank News",
                        "homepage_url": "https://bank.example.com/",
                    },
                    "title": "JPMorgan, Bank of America, and Wells Fargo update",
                    "published_utc": "2024-01-02T13:30:00Z",
                    "article_url": "https://example.com/bank-common",
                    "tickers": ["JPM", "BAC", "WFC"],
                    "description": (
                        "JPMorgan, Bank of America, and Wells Fargo issued "
                        "updates for bank investors."
                    ),
                    "keywords": ["banks"],
                    "insights": [],
                }
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    + "\n"
).encode("utf-8")


def write_collector_root(
    root: Path,
    *,
    tickers: list[str],
    payloads: dict[str, bytes],
) -> None:
    page_records = []
    for ticker in tickers:
        raw = payloads.get(ticker, EMPTY_PAGE)
        compressed = gzip.compress(raw, compresslevel=9, mtime=0)
        relative = f"pages/{ticker}/page_000001.json.gz"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(compressed)
        payload = json.loads(raw)
        page_records.append(
            {
                "ticker": ticker,
                "page_number": 1,
                "path": relative,
                "gzip_sha256": hashlib.sha256(compressed).hexdigest(),
                "json_sha256": hashlib.sha256(raw).hexdigest(),
                "result_count": len(payload["results"]),
                "request_cursor_sha256": None,
                "request_id": payload.get("request_id"),
                "has_next_page": False,
            }
        )
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "provider": "Massive",
        "dataset": "ordinary_news",
        "endpoint": "https://api.massive.com/v2/reference/news",
        "scope": {
            "tickers": tickers,
            "start": "2016-06-22",
            "end": "2026-06-30",
            "limit": 1000,
            "sort": "published_utc",
            "order": "asc",
        },
        "completed_tickers": tickers,
        "pages": page_records,
        "credential_values_recorded": False,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_calendar(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "sessions": [
                    {"date": "2024-01-02", "open": "09:30", "close": "16:00"},
                    {"date": "2024-01-03", "open": "09:30", "close": "16:00"},
                ]
            }
        ),
        encoding="utf-8",
    )


def read_csv_gz(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class MassiveDeterministicNewsFeatureTests(unittest.TestCase):
    def test_dedicated_universe_covers_all_30_stocks_and_five_sectors(
        self,
    ) -> None:
        targets = MODULE.load_targets(MODULE.DEFAULT_UNIVERSE)
        price_universe = json.loads(
            (ROOT / "config" / "price_universe.json").read_text(encoding="utf-8")
        )
        price_tickers = {
            ticker
            for sector in price_universe["sectors"]
            for ticker in sector["stocks"]
        }

        self.assertEqual(len(targets), 30)
        self.assertEqual({target.ticker for target in targets}, price_tickers)
        self.assertEqual(len({target.sector for target in targets}), 5)
        qcom = next(target for target in targets if target.ticker == "QCOM")
        self.assertEqual(len(qcom.peers), 5)
        self.assertEqual(qcom.benchmark, "SOXX")

    def test_native_pages_deduplicate_provider_id_and_preserve_provenance(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stock_root = root / "stocks"
            auxiliary_root = root / "auxiliary"
            write_collector_root(
                stock_root,
                tickers=["AMD"],
                payloads={
                    "AMD": (FIXTURE_ROOT / "amd_page.json").read_bytes()
                },
            )
            write_collector_root(
                auxiliary_root,
                tickers=["SOXX"],
                payloads={
                    "SOXX": (FIXTURE_ROOT / "nvda_page.json").read_bytes()
                },
            )

            articles, coverage, paths, sources = MODULE.load_massive_articles(
                [stock_root, auxiliary_root]
            )

        self.assertEqual(len(articles), 3)
        self.assertEqual(len(coverage), 2)
        self.assertEqual(len(paths), 2)
        self.assertEqual(len(sources), 2)
        duplicate = next(
            article
            for article in articles
            if article["provider_article_id"] == "massive-fixture-001"
        )
        self.assertEqual(duplicate["query_copy_count"], 2)
        self.assertEqual(duplicate["query_tickers"], ["AMD", "SOXX"])
        self.assertEqual(duplicate["tickers"], ["AMD", "NVDA"])
        self.assertNotIn("SOXX", duplicate["tickers"])
        self.assertEqual(
            duplicate["published_at_utc"], "2024-01-02T13:59:59Z"
        )
        self.assertEqual(duplicate["timestamp_precision"], "second")
        self.assertEqual(duplicate["publisher_name"], "Business Wire")
        self.assertEqual(duplicate["keywords"], ["earnings", "guidance"])
        self.assertEqual(duplicate["insight_tickers"], ["AMD"])
        self.assertTrue(
            all("::pages/" in path for path in duplicate["query_files"])
        )

    def test_end_to_end_multi_root_build_is_deterministic_and_not_version_safe(
        self,
    ) -> None:
        targets = MODULE.load_targets(MODULE.DEFAULT_UNIVERSE)
        stock_tickers = [target.ticker for target in targets]
        auxiliary_tickers = sorted(
            MODULE.OPTIONAL_BENCHMARK_TICKERS
            | MODULE.OPTIONAL_CONTROL_TICKERS
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stock_root = root / "stocks"
            auxiliary_root = root / "benchmarks"
            write_collector_root(
                stock_root,
                tickers=stock_tickers,
                payloads={
                    "AMD": (FIXTURE_ROOT / "amd_page.json").read_bytes(),
                    "NVDA": (FIXTURE_ROOT / "nvda_page.json").read_bytes(),
                },
            )
            write_collector_root(
                auxiliary_root,
                tickers=auxiliary_tickers,
                payloads={
                    "SOXX": (FIXTURE_ROOT / "nvda_page.json").read_bytes(),
                    "XLF": BANK_COMMON_PAGE,
                    "SPY": SPY_PAGE,
                },
            )
            calendar = root / "calendar.json"
            write_calendar(calendar)
            output = root / "output"

            manifest = MODULE.build(
                raw_roots=[stock_root, auxiliary_root],
                universe_path=MODULE.DEFAULT_UNIVERSE,
                calendar_path=calendar,
                output_directory=output,
                start=date(2024, 1, 2),
                end=date(2024, 1, 3),
            )

            normalized = read_csv_gz(output / "normalized_articles.csv.gz")
            pairs = read_csv_gz(output / "article_target_features.csv.gz")
            stock_days = read_csv_gz(output / "stock_day_features.csv.gz")
            first_hashes = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in MODULE.existing_output_paths(output)
            }
            MODULE.build(
                raw_roots=[stock_root, auxiliary_root],
                universe_path=MODULE.DEFAULT_UNIVERSE,
                calendar_path=calendar,
                output_directory=output,
                start=date(2024, 1, 2),
                end=date(2024, 1, 3),
                overwrite=True,
            )
            second_hashes = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in MODULE.existing_output_paths(output)
            }

        self.assertEqual(first_hashes, second_hashes)
        self.assertEqual(len(normalized), 5)
        self.assertEqual(len(stock_days), 60)
        duplicate = next(
            row
            for row in normalized
            if row["provider_article_id"] == "massive-fixture-001"
        )
        self.assertEqual(duplicate["query_copy_count"], "3")
        self.assertEqual(duplicate["query_tickers"], "AMD;NVDA;SOXX")
        self.assertEqual(duplicate["vendor_tickers"], "AMD;NVDA")
        exact_cutoff = next(
            row
            for row in pairs
            if row["article_id"] == "massive_massive-fixture-002"
            and row["target_ticker"] == "AMD"
        )
        after_cutoff = next(
            row
            for row in pairs
            if row["article_id"] == "massive_massive-fixture-003"
            and row["target_ticker"] == "AMD"
        )
        self.assertEqual(exact_cutoff["forecast_date"], "2024-01-02")
        self.assertEqual(after_cutoff["forecast_date"], "2024-01-03")
        bank_article_id = "massive_massive-fixture-bank-common"
        self.assertFalse(
            any(
                row["article_id"] == bank_article_id
                and row["target_ticker"] == "AMD"
                for row in pairs
            )
        )
        self.assertTrue(
            any(
                row["article_id"] == bank_article_id
                and row["target_ticker"] == "JPM"
                and row["common_proxy"] == "1"
                for row in pairs
            )
        )
        amd_day = next(
            row
            for row in stock_days
            if row["stock"] == "AMD" and row["forecast_date"] == "2024-01-02"
        )
        self.assertEqual(amd_day["ordinary_ticker_news_collection_complete"], "1")
        self.assertEqual(amd_day["target_ticker_query_complete"], "1")
        self.assertEqual(amd_day["sector_benchmark_query_complete"], "1")
        self.assertEqual(amd_day["common_news_coverage_complete"], "1")
        self.assertEqual(amd_day["primary_training_eligible"], "0")
        jpm_day = next(
            row
            for row in stock_days
            if row["stock"] == "JPM" and row["forecast_date"] == "2024-01-02"
        )
        self.assertAlmostEqual(
            float(amd_day["observed_sector_firms_with_news_share"]),
            2 / 6,
        )
        self.assertAlmostEqual(
            float(jpm_day["observed_sector_firms_with_news_share"]),
            3 / 6,
        )
        self.assertTrue(
            manifest["collection_completeness"][
                "ordinary_ticker_collection_complete"
            ]
        )
        self.assertTrue(
            manifest["collection_completeness"][
                "benchmark_and_control_collection_complete"
            ]
        )
        self.assertFalse(manifest["data_limitations"]["point_in_time_version_safe"])
        self.assertFalse(manifest["data_limitations"]["primary_training_eligible"])
        self.assertEqual(
            manifest["constructable_q_plus_d_features"],
            list(MODULE.CONSTRUCTABLE_Q_PLUS_D_FEATURES),
        )
        self.assertIn(
            "causal_event_count", manifest["strictly_unconstructable_features"]
        )

    def test_stock_only_collection_is_complete_for_targets_not_common_queries(
        self,
    ) -> None:
        targets = MODULE.load_targets(MODULE.DEFAULT_UNIVERSE)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stock_root = root / "stocks"
            write_collector_root(
                stock_root,
                tickers=[target.ticker for target in targets],
                payloads={
                    "AMD": (FIXTURE_ROOT / "amd_page.json").read_bytes()
                },
            )
            calendar = root / "calendar.json"
            write_calendar(calendar)
            manifest = MODULE.build(
                raw_roots=[stock_root],
                universe_path=MODULE.DEFAULT_UNIVERSE,
                calendar_path=calendar,
                output_directory=root / "unused",
                start=date(2024, 1, 2),
                end=date(2024, 1, 3),
                dry_run=True,
            )

        completeness = manifest["collection_completeness"]
        self.assertTrue(completeness["ordinary_ticker_collection_complete"])
        self.assertFalse(completeness["sector_benchmark_collection_complete"])
        self.assertFalse(completeness["control_collection_complete"])
        self.assertFalse(manifest["data_limitations"]["primary_training_eligible"])

    def test_rejects_flan_and_llama_output_destinations(self) -> None:
        for path in (
            ROOT / "outputs" / "flan_t5_xl" / "massive",
            ROOT / "experiments" / "llama_3_1" / "massive",
        ):
            with self.subTest(path=path), self.assertRaises(ValueError):
                MODULE.validate_output_directory(path)


if __name__ == "__main__":
    unittest.main()
