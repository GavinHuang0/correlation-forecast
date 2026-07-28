from __future__ import annotations

import unittest
from datetime import date, datetime, timezone

import pandas as pd

from scripts import build_v2_deterministic_news_features as d2
from scripts import build_v2_semantic_corpus as semantic


class SemanticCorpusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.targets = [
            d2.Target(
                ticker="AAA",
                company="Alpha Corporation",
                sector="Test Sector",
                benchmark="TST",
                peers=("BBB",),
            ),
            d2.Target(
                ticker="BBB",
                company="Beta Corporation",
                sector="Test Sector",
                benchmark="TST",
                peers=("AAA",),
            ),
        ]
        self.rules = {
            "aliases": {
                "AAA": ["Alpha"],
                "BBB": ["Beta"],
            },
            "sector_phrases": {
                "Test Sector": ["test sector"],
            },
            "macro_provider_keywords": ["inflation"],
        }
        self.sessions = [
            d2.Session(
                session_date=date(2025, 1, 2),
                cutoff_utc=datetime(
                    2025, 1, 2, 14, 0, tzinfo=timezone.utc
                ),
            ),
            d2.Session(
                session_date=date(2025, 1, 3),
                cutoff_utc=datetime(
                    2025, 1, 3, 14, 0, tzinfo=timezone.utc
                ),
            ),
            d2.Session(
                session_date=date(2025, 1, 6),
                cutoff_utc=datetime(
                    2025, 1, 6, 14, 0, tzinfo=timezone.utc
                ),
            ),
        ]
        self.universe = pd.DataFrame(
            [
                {
                    "forecast_date": "2025-01-03",
                    "sector": "Test Sector",
                    "stock": ticker,
                    "benchmark": "TST",
                }
                for ticker in ("AAA", "BBB")
            ]
        )

    def article(
        self,
        article_id: str,
        title: str,
        *,
        description: str | None = "",
        published: str = "2025-01-03T13:00:00Z",
        provider_tickers: str = "",
        keywords: str = "",
    ) -> dict:
        return {
            "provider_article_id": article_id,
            "published_at_utc": published,
            "title": title,
            "description": description,
            "provider_tickers": provider_tickers,
            "keywords": keywords,
            "text_sha256": f"source-{article_id}",
        }

    def test_exact_cip_routing_shared_text_and_stock_day_hashes(self) -> None:
        articles = pd.DataFrame(
            [
                self.article(
                    "direct",
                    "Alpha wins a contract",
                    description=None,
                    provider_tickers="AAA",
                ),
                self.article(
                    "sector",
                    "The test sector expands",
                    description="Alpha and Beta are both affected.",
                ),
                self.article(
                    "macro",
                    "Inflation surprises markets",
                    keywords="inflation",
                ),
                self.article(
                    "at-cutoff",
                    "Alpha update exactly at cutoff",
                    published="2025-01-03T14:00:00Z",
                    provider_tickers="AAA",
                ),
            ]
        )
        texts, assignments, scopes, audit = semantic.build_semantic_corpus(
            articles=articles,
            targets=self.targets,
            static_rules=self.rules,
            sessions=self.sessions,
            row_universe=self.universe,
        )
        self.assertEqual(len(assignments), 6)
        self.assertEqual(audit["role_counts"], {"C": 4, "I": 1, "P": 1})
        self.assertNotIn("at-cutoff", set(assignments["provider_article_id"]))
        direct = assignments[
            assignments["provider_article_id"].eq("direct")
        ].set_index("target_ticker")
        self.assertEqual(direct.loc["AAA", "role"], "I")
        self.assertEqual(direct.loc["BBB", "role"], "P")
        self.assertEqual(
            direct.loc["AAA", "model_text"], "Alpha wins a contract"
        )
        self.assertEqual(
            direct.loc["AAA", "text_sha256"],
            semantic.sha256_text("Alpha wins a contract"),
        )
        self.assertEqual(len(texts), 3)
        self.assertEqual(len(scopes), 2)
        for row in scopes.to_dict(orient="records"):
            observed = assignments[
                assignments["target_ticker"].eq(row["stock"])
            ]["assignment_id"].tolist()
            self.assertEqual(row["expected_assignment_count"], 3)
            self.assertEqual(
                row["expected_assignment_ids_sha256"],
                semantic.assignment_ids_sha256(observed),
            )
            self.assertTrue(row["source_query_scope_complete"])
            self.assertTrue(row["candidate_assignment_complete"])

    def test_row_universe_requires_the_same_target_set_each_date(self) -> None:
        incomplete = self.universe.iloc[:1].copy()
        with self.assertRaisesRegex(ValueError, "differs from"):
            semantic.normalize_row_universe(
                incomplete, self.targets, self.sessions
            )

    def test_description_bound_is_explicit_but_routing_uses_full_text(self) -> None:
        long_prefix = "ordinary words " * 60
        source_description = long_prefix + " Alpha"
        articles = pd.DataFrame(
            [
                self.article(
                    "late-target",
                    "Company update",
                    description=source_description,
                )
            ]
        )
        texts, assignments, _scopes, audit = semantic.build_semantic_corpus(
            articles=articles,
            targets=self.targets,
            static_rules=self.rules,
            sessions=self.sessions,
            row_universe=self.universe,
        )
        self.assertEqual(
            set(assignments["target_ticker"]), {"AAA", "BBB"}
        )
        self.assertTrue(assignments["description_was_bounded"].all())
        self.assertLessEqual(
            int(assignments["retained_description_char_count"].max()),
            semantic.DESCRIPTION_EXCERPT_MAX_CHARS,
        )
        self.assertNotIn("Alpha", assignments.iloc[0]["description"])
        routed = assignments.set_index("target_ticker")
        self.assertTrue(routed.loc["AAA", "candidate_roles"]["direct"])
        self.assertFalse(routed.loc["AAA", "roles"]["direct"])
        self.assertNotIn("AAA", routed.loc["AAA", "detected_entities"])
        self.assertIn(
            "AAA", routed.loc["AAA", "routing_detected_entities"]
        )
        self.assertEqual(
            assignments.iloc[0]["source_description_sha256"],
            semantic.sha256_text(source_description),
        )
        self.assertEqual(audit["bounded_description_unique_article_count"], 1)
        self.assertTrue(texts.iloc[0]["description_was_bounded"])


if __name__ == "__main__":
    unittest.main()
