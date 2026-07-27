from __future__ import annotations

import importlib.util
import gzip
import hashlib
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
FIXTURES = ROOT / "tests" / "fixtures" / "news_text_ablation"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

SPEC = importlib.util.spec_from_file_location(
    "build_news_text_ablation_benchmark_for_test",
    SCRIPTS / "build_news_text_ablation_benchmark.py",
)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)

import coarse_news_features as coarse  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class NewsTextAblationBuilderTests(unittest.TestCase):
    def test_candidate_queue_eligibility_never_becomes_vendor_metadata(
        self,
    ) -> None:
        self.assertEqual(
            builder.provider_ticker_values(
                {
                    "tickers": ["NOT_IN_UNIVERSE"],
                    "eligible_target_tickers": ["AAA", "AAB"],
                },
            ),
            {"NOT_IN_UNIVERSE"},
        )
        self.assertEqual(
            builder.eligible_target_values(
                {
                    "tickers": ["NOT_IN_UNIVERSE"],
                    "eligible_target_tickers": ["AAA", "AAB"],
                },
                {"AAA", "AAB", "BBB"},
            ),
            {"AAA", "AAB"},
        )
        raw = {
            "id": "shared",
            "title": "Synthetic shared article",
            "description": "Canonical Massive description.",
            "published_utc": "2026-03-02T00:00:00Z",
            "tickers": ["AAA"],
        }
        queue_copy = {
            **raw,
            "eligible_target_tickers": ["AAA", "AAB"],
        }
        deduplicated, duplicate_count = builder.deduplicate_massive(
            [raw, queue_copy], {"AAA", "AAB", "BBB"}
        )
        self.assertEqual(duplicate_count, 1)
        self.assertEqual(deduplicated[0]["provider_tickers"], ("AAA",))
        self.assertEqual(
            deduplicated[0]["eligible_target_tickers"], ("AAA", "AAB")
        )
        self.assertEqual(
            deduplicated[0]["assignment_tickers"], ("AAA", "AAB")
        )
        common = builder._common_record(
            {
                "article_id": "synthetic-id",
                "published": builder.parse_timestamp(
                    "2026-03-02T00:00:00Z"
                ),
                "source": "Synthetic",
                "headline": "Synthetic shared article",
                "provider_tickers": deduplicated[0]["provider_tickers"],
            },
            row_number=1,
            target={
                "company": "AAB Corp",
                "ticker": "AAB",
                "sector": "Synthetic",
                "sector_benchmark": "SYN",
                "known_sector_peers": ["AAA"],
            },
            split="evaluation",
            variant="massive_description",
            article_text="Canonical Massive description.",
            match_method="synthetic",
        )
        self.assertEqual(common["vendor_tickers"], ["AAA"])

    def test_fulltext_quality_gate_is_conservative_and_auditable(self) -> None:
        minor_mojibake = (
            "The companyâ€™s quarterly revenue increased while management "
            "reaffirmed its outlook. "
        ) * 8
        accepted, reason, _ = builder.assess_fulltext_quality(minor_mojibake)
        self.assertTrue(accepted)
        self.assertIsNone(reason)

        chinese = (
            "公司宣布季度收入增长，并重申全年展望。投资者关注需求和供应情况。"
        ) * 15
        accepted, reason, metrics = builder.assess_fulltext_quality(chinese)
        self.assertFalse(accepted)
        self.assertEqual(reason, "clearly_non_english_fulltext")
        self.assertGreater(metrics["non_latin_letter_ratio"], 0.9)

        garbled = "Company revenue " + ("\ufffd\x00" * 30)
        accepted, reason, _ = builder.assess_fulltext_quality(garbled)
        self.assertFalse(accepted)
        self.assertEqual(reason, "severely_garbled_fulltext")

        massive_record = {
            "id": "non-english-1",
            "article_url": "https://example.com/zh-hans/article",
            "title": "English discovery title",
            "description": "An English provider description.",
            "published_utc": "2026-03-02T00:00:00Z",
            "tickers": ["AAA"],
        }
        deduplicated, _ = builder.deduplicate_massive(
            [massive_record], {"AAA"}
        )
        fulltext_index = builder.MatchIndex(
            [{"id": "non-english-1", "full_text": chinese}]
        )
        eligible, counts, exclusions = builder.eligible_candidates(
            deduplicated,
            fulltext_index,
            cutoff=builder.parse_timestamp("2026-03-01T00:00:00Z"),
        )
        self.assertEqual(eligible, [])
        self.assertEqual(counts["clearly_non_english_fulltext"], 1)
        self.assertEqual(len(exclusions), 1)
        self.assertEqual(
            exclusions[0]["reason"], "clearly_non_english_fulltext"
        )
        self.assertNotIn("full_text", exclusions[0])

    def test_fixture_build_matches_sources_balances_targets_and_chunks_text(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "private-benchmark"
            manifest = builder.build_benchmark(
                massive_paths=[FIXTURES / "massive.json"],
                fulltext_paths=[FIXTURES / "fulltext.jsonl"],
                alpha_paths=[FIXTURES / "alpha.json"],
                universe_path=FIXTURES / "universe.json",
                schema_path=ROOT / "config" / "news_feature_schema_coarse.json",
                output_root=output,
                cutoff="2026-03-01T00:00:00Z",
                per_stock=1,
                development_count=2,
                max_chunk_characters=100,
            )

            massive = read_jsonl(output / "massive_description.jsonl")
            alpha = read_jsonl(output / "alpha_summary.jsonl")
            chunks = read_jsonl(output / "fulltext_evidence_chunks.jsonl")
            self.assertEqual(len(massive), 4)
            self.assertEqual(len(alpha), 3)
            self.assertEqual(
                {row["target"]["ticker"] for row in massive},
                {"AAA", "AAB", "BBB", "BBC"},
            )
            self.assertEqual(
                Counter(row["target"]["sector"] for row in massive),
                Counter({"Synthetic Sector One": 2, "Synthetic Sector Two": 2}),
            )
            self.assertEqual(
                Counter(row["benchmark_split"] for row in massive),
                Counter({"development": 2, "evaluation": 2}),
            )
            self.assertTrue(all(len(row["article_text"]) <= 100 for row in chunks))
            self.assertTrue(
                all(row["article_id"] == row["chunk_id"] for row in chunks)
            )
            self.assertEqual(
                {row["parent_article_id"] for row in chunks},
                {row["article_id"] for row in massive},
            )
            self.assertTrue(
                all(
                    set(row["target"])
                    == {
                        "company",
                        "ticker",
                        "sector",
                        "sector_benchmark",
                        "known_sector_peers",
                    }
                    for row in chunks
                )
            )
            for row in massive + alpha + chunks:
                coarse.validate_input_record(row)
            self.assertEqual(
                manifest["matching"]["fulltext_method_counts"],
                {
                    "exact_id": 2,
                    "exact_url": 1,
                    "normalized_title_timestamp": 1,
                },
            )
            self.assertEqual(
                manifest["matching"]["alpha_method_counts"],
                {
                    "exact_id": 1,
                    "exact_url": 1,
                    "normalized_title_timestamp": 1,
                    "unmatched": 1,
                },
            )
            self.assertEqual(manifest["coverage"]["fulltext_rate"], 1.0)
            self.assertEqual(
                manifest["fulltext_quality_gate"]["excluded_count"], 0
            )
            self.assertFalse(
                manifest["fulltext_quality_gate"]["raw_retrieval_modified"]
            )
            self.assertEqual(manifest["coverage"]["alpha_summary_rate"], 0.75)
            self.assertEqual(
                set(
                    manifest["target_metadata"][
                        "article_target_tickers"
                    ].values()
                ),
                {"AAA", "AAB", "BBB", "BBC"},
            )
            assignment_bases = manifest["ticker_provenance_contract"][
                "article_assignment_basis"
            ]
            self.assertEqual(
                set(assignment_bases),
                set(
                    manifest["target_metadata"][
                        "article_target_tickers"
                    ]
                ),
            )
            self.assertTrue(
                set(assignment_bases.values())
                <= set(builder.ASSIGNMENT_BASIS_VALUES)
            )
            self.assertEqual(
                sum(
                    manifest["ticker_provenance_contract"][
                        "selected_assignment_basis_counts"
                    ].values()
                ),
                4,
            )
            self.assertEqual(
                len(
                    manifest["fulltext_chunking"][
                        "chunk_id_to_article_id"
                    ]
                ),
                len(chunks),
            )
            self.assertEqual(
                manifest["fulltext_chunking"]["aggregation"]["tie_break"],
                "evaluation schema label order",
            )
            stored = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(stored, manifest)
            self.assertEqual(
                stored["builder_script"]["sha256"],
                builder.sha256_file(
                    SCRIPTS / "build_news_text_ablation_benchmark.py"
                ),
            )
            self.assertTrue(
                all(
                    variant["sha256"]
                    == builder.sha256_file(output / variant["file"])
                    for variant in stored["variants"].values()
                )
            )
            self.assertTrue(
                all("labels" not in row for row in massive + alpha + chunks)
            )

    def test_paragraph_chunking_is_bounded_and_deterministic(self) -> None:
        text = (
            "First short paragraph.\n\n"
            + " ".join(f"synthetic{index}" for index in range(60))
            + "\n\nLast paragraph."
        )
        first = builder.paragraph_chunks(text, maximum=90)
        second = builder.paragraph_chunks(text, maximum=90)
        self.assertEqual(first, second)
        self.assertGreater(len(first), 2)
        self.assertTrue(all(0 < len(chunk) <= 90 for chunk in first))

    def test_fixed_assignment_rebuild_preserves_selected_documents(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "private-benchmark"
            first = builder.build_benchmark(
                massive_paths=[FIXTURES / "massive.json"],
                fulltext_paths=[FIXTURES / "fulltext.jsonl"],
                alpha_paths=[FIXTURES / "alpha.json"],
                universe_path=FIXTURES / "universe.json",
                schema_path=ROOT / "config" / "news_feature_schema_coarse.json",
                output_root=output,
                cutoff="2026-03-01T00:00:00Z",
                per_stock=1,
                development_count=2,
                max_chunk_characters=100,
            )
            selected_before = dict(
                first["target_metadata"]["article_target_tickers"]
            )
            preserved_manifest = (
                Path(temporary_directory) / "fixed-assignment-source.json"
            )
            preserved_manifest.write_text(
                json.dumps(first, indent=2) + "\n",
                encoding="utf-8",
            )
            second = builder.build_benchmark(
                massive_paths=[FIXTURES / "massive.json"],
                fulltext_paths=[FIXTURES / "fulltext.jsonl"],
                alpha_paths=[FIXTURES / "alpha.json"],
                universe_path=FIXTURES / "universe.json",
                schema_path=ROOT / "config" / "news_feature_schema_coarse.json",
                output_root=output,
                cutoff="2026-03-01T00:00:00Z",
                per_stock=1,
                development_count=2,
                max_chunk_characters=70,
                fixed_assignment_manifest=preserved_manifest,
                overwrite=True,
            )
            self.assertEqual(
                second["target_metadata"]["article_target_tickers"],
                selected_before,
            )
            self.assertEqual(
                second["selection"]["method"],
                "fixed article-to-ticker assignment from prior manifest",
            )
            self.assertIsNotNone(
                second["selection"]["fixed_assignment_source"]
            )
            self.assertTrue(
                all(
                    len(row["article_text"]) <= 70
                    for row in read_jsonl(
                        output / "fulltext_evidence_chunks.jsonl"
                    )
                )
            )
            fixed_source = second["selection"]["fixed_assignment_source"]
            self.assertEqual(
                fixed_source["sha256"],
                builder.sha256_file(preserved_manifest),
            )

    def test_fixed_assignment_rejects_output_manifest_as_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "private-benchmark"
            builder.build_benchmark(
                massive_paths=[FIXTURES / "massive.json"],
                fulltext_paths=[FIXTURES / "fulltext.jsonl"],
                alpha_paths=[FIXTURES / "alpha.json"],
                universe_path=FIXTURES / "universe.json",
                schema_path=ROOT / "config" / "news_feature_schema_coarse.json",
                output_root=output,
                cutoff="2026-03-01T00:00:00Z",
                per_stock=1,
                development_count=2,
                max_chunk_characters=100,
            )
            with self.assertRaisesRegex(
                ValueError,
                "preserved source copy",
            ):
                builder.build_benchmark(
                    massive_paths=[FIXTURES / "massive.json"],
                    fulltext_paths=[FIXTURES / "fulltext.jsonl"],
                    alpha_paths=[FIXTURES / "alpha.json"],
                    universe_path=FIXTURES / "universe.json",
                    schema_path=(
                        ROOT / "config" / "news_feature_schema_coarse.json"
                    ),
                    output_root=output,
                    cutoff="2026-03-01T00:00:00Z",
                    per_stock=1,
                    development_count=2,
                    max_chunk_characters=70,
                    fixed_assignment_manifest=output / "manifest.json",
                    overwrite=True,
                )

    def test_fixed_assignment_rejects_cross_sector_ticker_reassignment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output = root / "private-benchmark"
            first = builder.build_benchmark(
                massive_paths=[FIXTURES / "massive.json"],
                fulltext_paths=[FIXTURES / "fulltext.jsonl"],
                alpha_paths=[FIXTURES / "alpha.json"],
                universe_path=FIXTURES / "universe.json",
                schema_path=ROOT / "config" / "news_feature_schema_coarse.json",
                output_root=output,
                cutoff="2026-03-01T00:00:00Z",
                per_stock=1,
                development_count=2,
                max_chunk_characters=100,
            )
            invalid_manifest = dict(first)
            invalid_target_metadata = dict(first["target_metadata"])
            invalid_assignment = dict(
                first["target_metadata"]["article_target_tickers"]
            )
            first_sector_id = next(
                article_id
                for article_id, ticker in invalid_assignment.items()
                if ticker in {"AAA", "AAB"}
            )
            second_sector_id = next(
                article_id
                for article_id, ticker in invalid_assignment.items()
                if ticker in {"BBB", "BBC"}
            )
            (
                invalid_assignment[first_sector_id],
                invalid_assignment[second_sector_id],
            ) = (
                invalid_assignment[second_sector_id],
                invalid_assignment[first_sector_id],
            )
            invalid_target_metadata["article_target_tickers"] = invalid_assignment
            invalid_manifest["target_metadata"] = invalid_target_metadata
            invalid_path = root / "invalid-fixed-assignment.json"
            invalid_path.write_text(
                json.dumps(invalid_manifest, indent=2) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "no longer eligible",
            ):
                builder.build_benchmark(
                    massive_paths=[FIXTURES / "massive.json"],
                    fulltext_paths=[FIXTURES / "fulltext.jsonl"],
                    alpha_paths=[FIXTURES / "alpha.json"],
                    universe_path=FIXTURES / "universe.json",
                    schema_path=ROOT
                    / "config"
                    / "news_feature_schema_coarse.json",
                    output_root=output,
                    cutoff="2026-03-01T00:00:00Z",
                    per_stock=1,
                    development_count=2,
                    max_chunk_characters=100,
                    fixed_assignment_manifest=invalid_path,
                    overwrite=True,
                )

    def test_alpha_required_mode_filters_before_assignment_and_fails_closed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with self.assertRaisesRegex(
                ValueError, "alpha_matched_eligible_by_ticker=.*'BBC': 0"
            ):
                builder.build_benchmark(
                    massive_paths=[FIXTURES / "massive.json"],
                    fulltext_paths=[FIXTURES / "fulltext.jsonl"],
                    alpha_paths=[FIXTURES / "alpha.json"],
                    universe_path=FIXTURES / "universe.json",
                    schema_path=(
                        ROOT / "config" / "news_feature_schema_coarse.json"
                    ),
                    output_root=root / "comparison-incomplete",
                    cutoff="2026-03-01T00:00:00Z",
                    per_stock=1,
                    development_count=2,
                    max_chunk_characters=100,
                    require_alpha_match=True,
                )
            self.assertFalse((root / "comparison-incomplete").exists())

            alpha_payload = json.loads(
                (FIXTURES / "alpha.json").read_text(encoding="utf-8")
            )
            alpha_payload["feed"].append(
                {
                    "title": "BBC schedules a prototype review",
                    "url": "https://news.example/bbc-review",
                    "time_published": "20260305T150000",
                    "summary": "Synthetic Alpha-style summary for BBC.",
                }
            )
            complete_alpha = root / "alpha-complete.json"
            complete_alpha.write_text(
                json.dumps(alpha_payload), encoding="utf-8"
            )
            output = root / "provider-comparison"
            manifest = builder.build_benchmark(
                massive_paths=[FIXTURES / "massive.json"],
                fulltext_paths=[FIXTURES / "fulltext.jsonl"],
                alpha_paths=[complete_alpha],
                universe_path=FIXTURES / "universe.json",
                schema_path=ROOT / "config" / "news_feature_schema_coarse.json",
                output_root=output,
                cutoff="2026-03-01T00:00:00Z",
                per_stock=1,
                development_count=2,
                max_chunk_characters=100,
                require_alpha_match=True,
            )
            self.assertTrue(manifest["selection"]["require_alpha_match"])
            self.assertTrue(
                manifest["provider_comparison"][
                    "nonempty_alpha_summary_required_before_assignment"
                ]
            )
            self.assertEqual(
                manifest["provider_comparison"][
                    "eligible_with_nonempty_alpha_summary"
                ],
                4,
            )
            self.assertEqual(
                manifest["provider_comparison"][
                    "alpha_matched_eligible_by_ticker"
                ],
                {"AAA": 1, "AAB": 1, "BBB": 1, "BBC": 1},
            )
            self.assertEqual(
                {
                    variant: metadata["article_count"]
                    for variant, metadata in manifest["variants"].items()
                },
                {
                    "massive_description": 4,
                    "alpha_summary": 4,
                    "fulltext_evidence_chunks": 4,
                },
            )
            self.assertEqual(manifest["coverage"]["alpha_summary_rate"], 1.0)

    def test_balanced_assignment_fails_closed_when_a_ticker_is_short(self) -> None:
        candidates = [
            {"article_id": "one", "assignment_tickers": ["AAA"]},
            {"article_id": "two", "assignment_tickers": ["AAA"]},
        ]
        with self.assertRaisesRegex(ValueError, "unmatched_slots"):
            builder.balanced_assignment(
                candidates, ["AAA", "BBB"], per_stock=1
            )

    def test_default_universe_contract_is_five_sectors_and_thirty_stocks(self) -> None:
        tickers, targets, metadata = builder.load_universe(
            ROOT / "config" / "price_universe.json"
        )
        self.assertEqual(len(tickers), 30)
        self.assertEqual(len(metadata["sectors"]), 5)
        self.assertEqual(len(tickers) * builder.DEFAULT_PER_STOCK, 300)
        self.assertTrue(all(targets[ticker]["company"] != "" for ticker in tickers))

    def test_native_collector_gzip_pages_and_retriever_layout_are_read_directly(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            collector = root / "collector"
            collector.mkdir()
            payload = (FIXTURES / "massive.json").read_bytes()
            with gzip.open(collector / "page_000001.json.gz", "wb") as handle:
                handle.write(payload)
            (collector / "state.json").write_text(
                json.dumps({"pages": []}), encoding="utf-8"
            )
            records, files = builder.read_input_files([collector])
            self.assertEqual(len(records), 4)
            self.assertEqual(
                [path.name for path in files], ["page_000001.json.gz"]
            )

            retrieval = root / "fulltext"
            articles = retrieval / "articles"
            articles.mkdir(parents=True)
            body = "Entirely synthetic retrieved body."
            body_path = articles / "body.txt"
            body_path.write_text(body, encoding="utf-8")
            entry = {
                "article_id": "massive-1",
                "published": "2026-03-02T09:00:00Z",
                "status": "retrieved",
                "content_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
                "text_file": "articles/body.txt",
            }
            (retrieval / "manifest.jsonl").write_text(
                json.dumps(entry) + "\n", encoding="utf-8"
            )
            fulltext, source_files = builder.read_fulltext_inputs([retrieval])
            self.assertEqual(fulltext[0]["full_text"], body)
            self.assertEqual(
                {path.name for path in source_files},
                {"manifest.jsonl", "body.txt"},
            )
            materialized = {
                "article_id": "massive-1",
                "article_url": "https://news.example/aaa-component",
                "title": "AAA launches a new component",
                "published": "2026-03-02T09:00:00Z",
                "full_text": body,
                "content_sha256": entry["content_sha256"],
                "text_file": "articles/body.txt",
            }
            (retrieval / "records.jsonl").write_text(
                json.dumps(materialized) + "\n", encoding="utf-8"
            )
            with (retrieval / "manifest.jsonl").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write("{interrupted")
            fulltext, source_files = builder.read_fulltext_inputs([retrieval])
            self.assertEqual(fulltext, [materialized])
            self.assertEqual(
                {path.name for path in source_files},
                {"manifest.jsonl", "records.jsonl", "body.txt"},
            )


if __name__ == "__main__":
    unittest.main()
