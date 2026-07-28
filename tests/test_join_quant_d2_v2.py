from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import join_quant_d2_v2 as module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class QuantD2V2JoinTests(unittest.TestCase):
    def quant_frame(self) -> pd.DataFrame:
        rows = []
        for day, value in (
            ("2024-01-02", 1.25),
            ("2024-01-03", 2.50),
        ):
            row = {
                "forecast_date": pd.Timestamp(day),
                "sector": "Semiconductors",
                "stock": "AMD",
                "benchmark": "SOXX",
                "quant_marker": value,
            }
            for name in self.quant_groups()["etf_core_22"]:
                row[name] = 0.1
            for name in self.quant_groups()["loo_core_22"]:
                row[name] = 0.2
            for name in self.quant_groups()["dense_context"]:
                row[name] = 0.3
            for name in self.quant_groups()["volatility"]:
                row[name] = 0.4
            for name in self.quant_groups()["extended_hours"]:
                row[name] = 0.5
            rows.append(row)
        return pd.DataFrame(rows)

    def quant_groups(self) -> dict[str, list[str]]:
        sector = [f"sector_state_{index}" for index in range(8)]
        return {
            "etf_core_22": [
                *[f"etf_pair_{index}" for index in range(14)],
                *sector,
            ],
            "loo_core_22": [
                *[f"loo_pair_{index}" for index in range(14)],
                *sector,
            ],
            "dense_context": [f"dense_{index}" for index in range(15)],
            "volatility": [f"volatility_{index}" for index in range(2)],
            "extended_hours": [f"extended_{index}" for index in range(17)],
        }

    def news_frame(self) -> pd.DataFrame:
        rows = []
        for day, base in (("2024-01-02", 0.1), ("2024-01-03", 0.2)):
            row = {
                "forecast_date": pd.Timestamp(day),
                "sector": "Semiconductors",
                "stock": "AMD",
                "benchmark": "SOXX",
                "source_profile": "ordinary_massive_retrospective",
            }
            row.update(
                {
                    name: base + index / 1000
                    for index, name in enumerate(module.D2_NORMALIZED_30)
                }
            )
            row.update(
                {
                    name: base + index / 100
                    for index, name in enumerate(module.D2_LEVELS_5)
                }
            )
            rows.append(row)
        return pd.DataFrame(rows)

    def test_exact_join_preserves_quant_and_complete_d2(self) -> None:
        quant = self.quant_frame()
        joined, stats, audits = module.build_joined_panel(
            quant,
            self.news_frame().iloc[::-1].reset_index(drop=True),
            d2_features=module.D2_NORMALIZED_30,
            d2_level_features=module.D2_LEVELS_5,
            start=pd.Timestamp("2024-01-01"),
            end=pd.Timestamp("2024-01-31"),
            expected_rows=2,
        )
        pd.testing.assert_frame_equal(
            joined[list(quant.columns)], quant, check_dtype=True
        )
        self.assertEqual(stats["matched_rows"], 2)
        self.assertEqual(stats["d2_missing_value_count"], 0)
        self.assertEqual(joined["d2_row_matched"].tolist(), [1, 1])
        self.assertEqual(audits, ["source_profile"])

    def test_rejects_key_mismatch_missing_and_nonfinite_features(self) -> None:
        news = self.news_frame()
        news.loc[1, "stock"] = "NVDA"
        with self.assertRaisesRegex(ValueError, "key sets differ"):
            module.build_joined_panel(
                self.quant_frame(),
                news,
                d2_features=module.D2_NORMALIZED_30,
                d2_level_features=module.D2_LEVELS_5,
                start=pd.Timestamp("2024-01-01"),
                end=pd.Timestamp("2024-01-31"),
                expected_rows=2,
            )

        for invalid in (np.nan, np.inf):
            with self.subTest(invalid=invalid):
                news = self.news_frame()
                news.loc[0, module.D2_NORMALIZED_30[0]] = invalid
                with self.assertRaisesRegex(
                    ValueError, "missing feature|not finite"
                ):
                    module.build_joined_panel(
                        self.quant_frame(),
                        news,
                        d2_features=module.D2_NORMALIZED_30,
                        d2_level_features=module.D2_LEVELS_5,
                        start=pd.Timestamp("2024-01-01"),
                        end=pd.Timestamp("2024-01-31"),
                        expected_rows=2,
                    )

    def test_protocol_has_exact_feature_lists_and_folds(self) -> None:
        quant_features = {
            target: [f"q_{target}_{i}" for i in range(56)]
            for target in module.TARGETS
        }
        quant_features["t2_etf"] = list(quant_features["t1_etf"])
        quant_features["t2_loo"] = list(quant_features["t1_loo"])
        protocol = module.build_locked_protocol(
            quant_features=quant_features,
            d2_features=module.D2_NORMALIZED_30,
            d2_level_features=module.D2_LEVELS_5,
            d43_features=None,
            joined_path=Path("joined.parquet"),
            joined_manifest_path=Path("manifest.json"),
            joined_sha256="a" * 64,
            joined_manifest_sha256="b" * 64,
            row_count=2,
            date_count=2,
            stock_count=1,
            sector_count=1,
            start="2024-01-02",
            end="2024-01-03",
        )
        self.assertEqual(protocol["status"], "locked_before_training")
        self.assertEqual(protocol["folds"], list(module.FOLDS))
        self.assertEqual(
            protocol["feature_blocks"]["d2_normalized_30"],
            list(module.D2_NORMALIZED_30),
        )
        by_name = {
            rung["name"]: rung for rung in protocol["deterministic_rungs"]
        }
        self.assertTrue(by_name["V2-D3"]["construction_available"])
        self.assertFalse(by_name["V2-D2"]["construction_available"])
        self.assertEqual(
            protocol["feature_blocks"]["d43_recomputed_43"], []
        )
        self.assertFalse(
            protocol["bundle_availability"]["D2"]["available"]
        )

    def test_cli_binds_panel_manifest_and_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            quant_path = root / "quant.parquet"
            news_path = root / "news.parquet"
            quant_manifest_path = root / "quant.manifest.json"
            news_manifest_path = root / "news.manifest.json"
            output_root = root / "output"
            protocol_path = root / "protocol.json"
            protocol_hash_path = root / "protocol.sha256"

            quant = self.quant_frame()
            news = self.news_frame()
            quant.to_parquet(quant_path, index=False)
            news.to_parquet(news_path, index=False)
            quant_manifest_path.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "row_count": 2,
                        "output": {"sha256": sha256(quant_path)},
                        "feature_groups": self.quant_groups(),
                    }
                ),
                encoding="utf-8",
            )
            news_manifest_path.write_text(
                json.dumps(
                    {
                        "status": (
                            "complete_exploratory_non_version_safe"
                        ),
                        "generated_files": {
                            "stock_day_features.parquet": {
                                "sha256": sha256(news_path)
                            }
                        },
                        "ordered_feature_lists": {
                            "d2_normalized_30": list(
                                module.D2_NORMALIZED_30
                            ),
                            "d2_levels_5": list(module.D2_LEVELS_5),
                        },
                        "source_completeness": {
                            "all_configured_query_roots_complete": True,
                            "all_model_rows_eligible": True,
                            "expected_model_row_count": 2,
                        },
                        "source_limitations": {
                            "point_in_time_version_safe": False,
                            "primary_training_eligible": False,
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
                    "--protocol-output",
                    str(protocol_path),
                    "--protocol-hash-output",
                    str(protocol_hash_path),
                    "--start",
                    "2024-01-02",
                    "--end",
                    "2024-01-03",
                    "--expected-row-count",
                    "2",
                ]
            )
            self.assertEqual(result, 0)
            output = output_root / module.DEFAULT_OUTPUT_NAME
            manifest_path = output_root / module.MANIFEST_NAME
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["counts"]["match_fraction"], 1.0)
            self.assertEqual(manifest["output"]["sha256"], sha256(output))
            self.assertEqual(
                protocol["source_artifacts"]["panel_sha256"],
                sha256(output),
            )
            self.assertEqual(
                protocol_hash_path.read_text(encoding="utf-8"),
                f"{sha256(protocol_path)}  protocol.json\n",
            )

    def test_feature_manifest_order_is_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            quant_path = root / "quant"
            news_path = root / "news"
            quant_path.write_bytes(b"quant")
            news_path.write_bytes(b"news")
            quant_manifest = root / "quant.json"
            news_manifest = root / "news.json"
            quant_manifest.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "row_count": 2,
                        "output": {"sha256": sha256(quant_path)},
                    }
                ),
                encoding="utf-8",
            )
            reversed_d2 = list(reversed(module.D2_NORMALIZED_30))
            news_manifest.write_text(
                json.dumps(
                    {
                        "status": (
                            "complete_exploratory_non_version_safe"
                        ),
                        "generated_files": {
                            "stock_day_features.parquet": {
                                "sha256": sha256(news_path)
                            }
                        },
                        "ordered_feature_lists": {
                            "d2_normalized_30": reversed_d2,
                            "d2_levels_5": list(module.D2_LEVELS_5),
                        },
                        "source_completeness": {
                            "all_configured_query_roots_complete": True,
                            "all_model_rows_eligible": True,
                            "expected_model_row_count": 2,
                        },
                        "source_limitations": {
                            "point_in_time_version_safe": False,
                            "primary_training_eligible": False,
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "feature order"):
                module.validate_source_contracts(
                    quant_path=quant_path,
                    quant_manifest_path=quant_manifest,
                    news_path=news_path,
                    news_manifest_path=news_manifest,
                    expected_rows=2,
                )


if __name__ == "__main__":
    unittest.main()
