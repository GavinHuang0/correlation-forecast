from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts import join_quant_deterministic_news as module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class QuantDeterministicNewsJoinTests(unittest.TestCase):
    def quant_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "forecast_date": pd.Timestamp("2024-01-02"),
                    "sector": "Semiconductors",
                    "stock": "AMD",
                    "benchmark": "SOXX",
                    "quant_feature": 1.25,
                },
                {
                    "forecast_date": pd.Timestamp("2024-01-03"),
                    "sector": "Semiconductors",
                    "stock": "AMD",
                    "benchmark": "SOXX",
                    "quant_feature": 2.50,
                },
            ]
        )

    def news_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "forecast_date": "2024-01-02",
                    "sector": "Semiconductors",
                    "stock": "AMD",
                    "benchmark": "SOXX",
                    "observed_relevant_article_count": 3,
                    "point_in_time_version_safe": 0,
                    "primary_training_eligible": 0,
                }
            ]
        )

    def test_left_join_preserves_quant_rows_values_and_order(self) -> None:
        quant = self.quant_frame()
        panel, stats, audit = module.build_joined_panel(
            quant,
            self.news_frame(),
            declared_news_features=["observed_relevant_article_count"],
            start=pd.Timestamp("2024-01-01"),
            end=pd.Timestamp("2024-01-31"),
            require_full_match=False,
        )

        self.assertEqual(len(panel), 2)
        self.assertEqual(panel["quant_feature"].tolist(), [1.25, 2.50])
        self.assertEqual(panel["news_row_matched"].tolist(), [1, 0])
        self.assertEqual(stats["unmatched_quant_rows"], 1)
        self.assertEqual(stats["quant_rows_in_requested_range"], 2)
        self.assertIn("point_in_time_version_safe", audit)
        self.assertTrue(
            pd.isna(
                panel.loc[
                    panel["news_row_matched"].eq(0),
                    "observed_relevant_article_count",
                ].iloc[0]
            )
        )

    def test_full_match_mode_fails_instead_of_zero_imputing(self) -> None:
        with self.assertRaisesRegex(ValueError, "no deterministic-news row"):
            module.build_joined_panel(
                self.quant_frame(),
                self.news_frame(),
                declared_news_features=["observed_relevant_article_count"],
                start=pd.Timestamp("2024-01-01"),
                end=pd.Timestamp("2024-01-31"),
                require_full_match=True,
            )

    def test_duplicate_and_colliding_payload_columns_are_rejected(self) -> None:
        duplicated = pd.concat(
            [self.news_frame(), self.news_frame()], ignore_index=True
        )
        with self.assertRaisesRegex(ValueError, "duplicate join keys"):
            module.build_joined_panel(
                self.quant_frame(),
                duplicated,
                declared_news_features=["observed_relevant_article_count"],
                start=pd.Timestamp("2024-01-01"),
                end=pd.Timestamp("2024-01-31"),
                require_full_match=False,
            )

        collision = self.news_frame().copy()
        collision["quant_feature"] = 999.0
        with self.assertRaisesRegex(ValueError, "collides"):
            module.build_joined_panel(
                self.quant_frame(),
                collision,
                declared_news_features=["observed_relevant_article_count"],
                start=pd.Timestamp("2024-01-01"),
                end=pd.Timestamp("2024-01-31"),
                require_full_match=False,
            )

    def test_feature_preflight_removes_constant_and_exact_redundancy(self) -> None:
        frame = pd.DataFrame(
            {
                "observed_relevant_article_count": [0, 2, 1],
                "observed_no_relevant_news": [1, 0, 0],
                "timing_eligible_article_count": [0, 2, 1],
                "timing_eligible_share": [0.0, 1.0, 1.0],
                "timing_imprecise_article_count": [0, 0, 0],
                "independent_signal": [0.1, None, 0.2],
            }
        )
        features = list(frame.columns)
        report = module.deterministic_feature_preflight(frame, features)
        self.assertEqual(
            report["constant_features"],
            ["timing_imprecise_article_count"],
        )
        self.assertEqual(report["recommended_feature_count"], 3)
        self.assertEqual(
            report["recommended_feature_columns"],
            [
                "observed_relevant_article_count",
                "observed_no_relevant_news",
                "independent_signal",
            ],
        )
        self.assertEqual(
            {
                item["feature"]
                for item in report["verified_exact_redundancies"]
            },
            {"timing_eligible_article_count", "timing_eligible_share"},
        )
        self.assertEqual(
            report["recommended_feature_missingness"],
            {
                "independent_signal": {
                    "missing_count": 1,
                    "missing_fraction": 1 / 3,
                }
            },
        )

    def test_cli_writes_bound_manifest_and_warning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            quant_path = root / "quant.parquet"
            news_path = root / "news.csv.gz"
            quant_manifest_path = root / "quant.manifest.json"
            news_manifest_path = root / "news.manifest.json"
            output_root = root / "output"

            quant = self.quant_frame().iloc[[0]].copy()
            news = self.news_frame().copy()
            quant.to_parquet(quant_path, index=False)
            news.to_csv(news_path, index=False, compression="gzip")
            quant_manifest_path.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "output": {"sha256": sha256(quant_path)},
                    }
                ),
                encoding="utf-8",
            )
            news_manifest_path.write_text(
                json.dumps(
                    {
                        "status": (
                            "ordinary_collection_complete_but_not_version_safe"
                        ),
                        "collection_completeness": {
                            "ordinary_ticker_collection_complete": True,
                            "sector_benchmark_collection_complete": True,
                            "control_collection_complete": True,
                        },
                        "data_limitations": {
                            "point_in_time_version_safe": False,
                            "primary_training_eligible": False,
                            "first_seen_available": False,
                            "last_updated_available": False,
                            "historical_version_history_available": False,
                            "full_article_body_available": False,
                        },
                        "constructable_q_plus_d_features": [
                            "observed_relevant_article_count"
                        ],
                        "generated_files": {
                            "stock_day_features.csv.gz": {
                                "sha256": sha256(news_path)
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = module.main(
                [
                    "--quant",
                    str(quant_path),
                    "--quant-manifest",
                    str(quant_manifest_path),
                    "--news",
                    str(news_path),
                    "--news-manifest",
                    str(news_manifest_path),
                    "--output-root",
                    str(output_root),
                    "--start",
                    "2024-01-02",
                    "--end",
                    "2024-01-02",
                ]
            )

            self.assertEqual(result, 0)
            output = output_root / module.DEFAULT_OUTPUT_NAME
            manifest_path = output_root / module.MANIFEST_NAME
            hash_path = output_root / module.MANIFEST_HASH_NAME
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            written = pd.read_parquet(output)
            pd.testing.assert_frame_equal(
                written[list(quant.columns)],
                quant,
                check_dtype=True,
            )
            self.assertEqual(
                manifest["status"],
                "complete_exploratory_non_version_safe",
            )
            self.assertTrue(manifest["claim_scope"]["exploratory_only"])
            self.assertFalse(
                manifest["claim_scope"]["point_in_time_version_safe"]
            )
            self.assertEqual(manifest["counts"]["matched_news_rows"], 1)
            self.assertEqual(
                manifest["modeling_preflight"][
                    "materialized_feature_count"
                ],
                1,
            )
            self.assertEqual(manifest["output"]["sha256"], sha256(output))
            expected_sidecar = f"{sha256(manifest_path)}  manifest.json\n"
            self.assertEqual(
                hash_path.read_text(encoding="utf-8"), expected_sidecar
            )

    def test_protected_extractor_output_paths_are_rejected(self) -> None:
        for parts in module.PROTECTED_OUTPUT_PARTS:
            with self.subTest(parts=parts), self.assertRaisesRegex(
                ValueError, "active extractor"
            ):
                module.validate_output_root(module.ROOT.joinpath(*parts, "x"))


if __name__ == "__main__":
    unittest.main()
