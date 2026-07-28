from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.correlation_training import training_common as quant_common
from scripts.correlation_training import build_modeling_panel as quant_schema
from scripts.quant_deterministic_news_training_v2 import common
from scripts.quant_deterministic_news_training_v2 import run_controls
from scripts.quant_deterministic_news_training_v2 import run_final_comparison
from scripts.quant_deterministic_news_training_v2 import run_xgboost


class QuantDeterministicNewsTrainingV2Tests(unittest.TestCase):
    def _feature_blocks(self, *, include_d43: bool = True) -> dict[str, list[str]]:
        return {
            "q56_etf": [f"q_etf_{index:02d}" for index in range(56)],
            "q56_loo": [f"q_loo_{index:02d}" for index in range(56)],
            "d2_normalized_30": [
                f"d2_normalized_{index:02d}" for index in range(30)
            ],
            "d43_recomputed_43": (
                [f"d43_recomputed_{index:02d}" for index in range(43)]
                if include_d43
                else []
            ),
            "d2_levels_5": [
                f"d2_level_{index:02d}" for index in range(5)
            ],
        }

    def _folds(self) -> list[dict[str, str]]:
        return [
            {
                "name": "fold_1",
                "train_start": "2024-01-01",
                "train_end": "2024-01-01",
                "validation_start": "2024-01-02",
                "validation_end": "2024-01-02",
                "test_start": "2024-01-03",
                "test_end": "2024-01-03",
            },
            {
                "name": "fold_2",
                "train_start": "2024-01-01",
                "train_end": "2024-01-02",
                "validation_start": "2024-01-03",
                "validation_end": "2024-01-03",
                "test_start": "2024-01-04",
                "test_end": "2024-01-04",
            },
            {
                "name": "fold_3",
                "train_start": "2024-01-01",
                "train_end": "2024-01-03",
                "validation_start": "2024-01-04",
                "validation_end": "2024-01-04",
                "test_start": "2024-01-05",
                "test_end": "2024-01-05",
            },
        ]

    def _protocol(
        self,
        panel_path: Path,
        manifest_path: Path,
        *,
        include_d43: bool = True,
    ) -> dict[str, object]:
        availability = {
            "D2": {
                "available": include_d43,
                "reason": (
                    None
                    if include_d43
                    else "D43 cannot be recomputed under the v2 cutoff contract"
                ),
            }
        }
        return {
            "experiment_id": "quant-deterministic-news-v2",
            "status": "locked_before_training",
            "claim_scope": "exploratory development only",
            "source_profile": "ordinary_massive_retrospective",
            "targets": list(common.TARGETS),
            "folds": self._folds(),
            "source_artifacts": {
                "panel_path": panel_path.as_posix(),
                "panel_sha256": (
                    common.sha256_file(panel_path)
                    if panel_path.exists()
                    else "0" * 64
                ),
                "panel_manifest_path": manifest_path.as_posix(),
                "panel_manifest_sha256": (
                    common.sha256_file(manifest_path)
                    if manifest_path.exists()
                    else "1" * 64
                ),
                "row_count": 5,
                "date_count": 5,
                "stock_count": 1,
                "sector_count": 1,
            },
            "feature_blocks": self._feature_blocks(
                include_d43=include_d43
            ),
            "model_tuning": {
                "linear_alpha_grid": [0.001, 0.01],
                "elastic_net_l1_ratio_grid": [0.5, 1.0],
            },
            "preprocessing": {
                "fixed_log1p_features": list(quant_schema.LOG1P_FEATURES)
            },
            "row_eligibility": {
                "required_equalities": {"d2_row_eligible": 1}
            },
            "target_row_eligibility": {
                "expected_rows": {
                    "fold_1": {
                        "t1_train_validation_test": [1, 1, 1],
                        "t2_train_validation_test": [1, 1, 1],
                    },
                    "fold_2": {
                        "t1_train_validation_test": [2, 1, 1],
                        "t2_train_validation_test": [2, 1, 1],
                    },
                    "fold_3": {
                        "t1_train_validation_test": [3, 1, 1],
                        "t2_train_validation_test": [3, 1, 1],
                    },
                }
            },
            "bundle_availability": availability,
        }

    def _panel(self) -> pd.DataFrame:
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        frame = pd.DataFrame(
            {
                "forecast_date": dates,
                "asof_session": dates - pd.Timedelta(days=1),
                "sector": "test_sector",
                "stock": "TEST",
                "benchmark": "TEST_ETF",
                "d2_row_eligible": 1,
                "etf_rth_rc_d": 0.1,
                "etf_rth_rc_w": 0.1,
                "loo_rth_rc_d": 0.1,
                "loo_rth_rc_w": 0.1,
            }
        )
        for benchmark in ("etf", "loo"):
            for horizon in ("t1", "t2"):
                frame[f"target_{benchmark}_{horizon}_fisher_z"] = 0.2
                frame[f"target_{benchmark}_{horizon}_correlation"] = np.tanh(
                    0.2
                )
            frame[f"target_{benchmark}_t2_end_date"] = dates
        blocks = self._feature_blocks(include_d43=False)
        feature_values: dict[str, float] = {}
        for key in ("q56_etf", "q56_loo", "d2_normalized_30", "d2_levels_5"):
            for position, feature in enumerate(blocks[key]):
                feature_values[feature] = float(position + 1)
        return pd.concat(
            [
                frame,
                pd.DataFrame(
                    {
                        feature: np.full(len(frame), value)
                        for feature, value in feature_values.items()
                    }
                ),
            ],
            axis=1,
        )

    def test_protocol_validation_and_feature_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            panel_path = root / "panel.parquet"
            manifest_path = root / "manifest.json"
            protocol_path = root / "protocol.json"
            protocol = self._protocol(panel_path, manifest_path)
            protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
            with patch.object(common, "PANEL_PATH", panel_path):
                loaded = common.load_protocol(protocol_path)
            for spec in quant_common.target_specs():
                observed = {
                    key: len(common.bundle_features(key, spec, loaded))
                    for key in common.BUNDLE_BY_KEY
                }
                self.assertEqual(
                    observed,
                    {
                        "D0": 56,
                        "D1": 30,
                        "D2": 99,
                        "D3": 86,
                        "D4": 91,
                        "D5": 86,
                        "C-D3-L20": 86,
                        "C-D3-WS": 86,
                    },
                )

            invalid = dict(protocol)
            invalid["status"] = "draft"
            protocol_path.write_text(json.dumps(invalid), encoding="utf-8")
            with patch.object(common, "PANEL_PATH", panel_path):
                with self.assertRaisesRegex(ValueError, "locked"):
                    common.load_protocol(protocol_path)

    def test_protocol_allows_predeclared_unavailable_d43(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            panel_path = root / "panel.parquet"
            manifest_path = root / "manifest.json"
            protocol_path = root / "protocol.json"
            protocol = self._protocol(
                panel_path, manifest_path, include_d43=False
            )
            protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
            with patch.object(common, "PANEL_PATH", panel_path):
                loaded = common.load_protocol(protocol_path)
            available, reason = common.bundle_availability(loaded, "D2")
            self.assertFalse(available)
            self.assertIn("cannot be recomputed", reason)
            for spec in quant_common.target_specs():
                self.assertEqual(
                    len(common.bundle_features("D3", spec, loaded)), 86
                )
            with self.assertRaises(ValueError):
                common.bundle_features(
                    "D2", quant_common.target_specs()[0], loaded
                )

    def test_panel_preflight_and_split_counts_without_d43(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            panel_path = root / "panel.parquet"
            manifest_path = root / "manifest.json"
            protocol_path = root / "protocol.json"
            self._panel().to_parquet(panel_path, index=False)
            manifest_path.write_text(
                json.dumps({"status": "complete"}), encoding="utf-8"
            )
            protocol = self._protocol(
                panel_path, manifest_path, include_d43=False
            )
            protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
            with (
                patch.object(common, "PANEL_PATH", panel_path),
                patch.object(common, "PROTOCOL_PATH", protocol_path),
            ):
                loaded = common.load_protocol(protocol_path)
                panel, audit = common.load_panel_and_preflight(
                    loaded,
                    allow_exploratory=True,
                    required_bundle_keys=("D3",),
                )
            self.assertEqual(len(panel), 5)
            self.assertEqual(audit["required_bundles"], ["D3"])
            self.assertEqual(
                audit["split_counts"]["t1_etf"]["fold_3"],
                {"train": 3, "validation": 1, "test": 1},
            )
            self.assertEqual(
                audit["split_counts"]["t2_loo"]["fold_1"],
                {"train": 1, "validation": 1, "test": 1},
            )

    def test_exact_prediction_comparison(self) -> None:
        rows = []
        for fold, actual, base, candidate in (
            ("fold_1", 1.0, 0.0, 0.75),
            ("fold_2", 0.5, 0.0, 0.4),
            ("fold_3", -0.5, 0.0, -0.4),
        ):
            rows.append(
                {
                    "fold": fold,
                    "target": "t1_etf",
                    "forecast_date": pd.Timestamp("2025-01-01"),
                    "sector": "test_sector",
                    "stock": fold,
                    "benchmark": "TEST_ETF",
                    "actual_fisher_z": actual,
                    "base_prediction": base,
                    "candidate_prediction": candidate,
                }
            )
        source = pd.DataFrame(rows)
        base = source[
            [*common.PREDICTION_KEYS, "actual_fisher_z"]
        ].copy()
        candidate = base.copy()
        base["predicted_fisher_z"] = source["base_prediction"]
        candidate["predicted_fisher_z"] = source["candidate_prediction"]
        comparison = common.compare_predictions(candidate, base)
        self.assertEqual(len(comparison), 1)
        self.assertGreater(comparison[0]["incremental_r2"], 0)
        self.assertEqual(comparison[0]["folds_model_better"], 3)

        mismatched = base.iloc[:-1].copy()
        with self.assertRaisesRegex(ValueError, "keys differ"):
            common.compare_predictions(candidate, mismatched)

    def test_xgboost_gate_is_target_specific_and_validation_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = self._protocol(
                root / "panel.parquet",
                root / "manifest.json",
                include_d43=False,
            )
        rows: list[dict[str, object]] = []
        for target in common.TARGETS:
            for fold_index, fold in enumerate(
                ("fold_1", "fold_2", "fold_3")
            ):
                d3_prediction = (
                    0.5
                    if target == "t1_etf" and fold_index < 2
                    else (0.5 if fold_index == 0 else 0.0)
                )
                rows.append(
                    {
                        "fold": fold,
                        "target": target,
                        "forecast_date": pd.Timestamp("2024-01-02"),
                        "sector": "test_sector",
                        "stock": f"{target}_{fold}",
                        "benchmark": "TEST_ETF",
                        "actual_fisher_z": 1.0,
                        "predicted_fisher_z": d3_prediction,
                        "predicted_correlation": np.tanh(d3_prediction),
                    }
                )
        d3 = pd.DataFrame(rows)
        d0 = d3.copy()
        d0["predicted_fisher_z"] = 0.0
        d0["predicted_correlation"] = 0.0
        gate = run_xgboost.validation_gate_records(d3, d0, protocol)
        by_target = {record["target"]: record for record in gate}
        self.assertTrue(by_target["t1_etf"]["eligible"])
        self.assertEqual(
            by_target["t1_etf"]["improved_validation_folds"], 2
        )
        for target in ("t1_loo", "t2_etf", "t2_loo"):
            self.assertFalse(by_target[target]["eligible"])
            self.assertEqual(by_target[target]["improved_validation_folds"], 1)

    def test_xgboost_budget_reuses_exact_quant_v1_schema(self) -> None:
        protocol = {
            "model_tuning": {
                "reuse_quant_v1_shallow_xgboost_candidates": True
            }
        }
        candidate = {
            "max_depth": 2,
            "learning_rate": 0.05,
            "min_child_weight": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 10,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "quant_protocol.json"
            path.write_text(
                json.dumps(
                    {
                        "rung_3": {
                            "candidate_configs": [
                                candidate,
                                {**candidate, "max_depth": 3},
                                {**candidate, "learning_rate": 0.02},
                                {**candidate, "reg_lambda": 1},
                            ],
                            "n_estimators": 2000,
                            "early_stopping_rounds": 75,
                        }
                    }
                ),
                encoding="utf-8",
            )
            budget = run_xgboost.load_xgboost_budget(protocol, path)
        self.assertEqual(len(budget["candidate_configs"]), 4)
        self.assertEqual(budget["max_estimators"], 2000)
        self.assertEqual(budget["early_stopping_rounds"], 75)

    def test_d3_controls_replace_only_d2_and_preserve_all_rows(self) -> None:
        blocks = self._feature_blocks(include_d43=False)
        d2 = blocks["d2_normalized_30"]
        dates = pd.date_range("2023-01-02", periods=25, freq="B")
        source_rows: list[dict[str, object]] = []
        for stock_index, stock in enumerate(("AAA", "BBB")):
            for date_index, forecast_date in enumerate(dates):
                row: dict[str, object] = {
                    "forecast_date": forecast_date,
                    "sector": "test_sector",
                    "stock": stock,
                    "source_profile": "ordinary_massive_retrospective",
                }
                for feature_index, feature in enumerate(d2):
                    row[feature] = (
                        stock_index * 10_000
                        + date_index * 100
                        + feature_index
                    )
                source_rows.append(row)
        full_d2 = pd.DataFrame(source_rows)
        current = full_d2[full_d2["forecast_date"].isin(dates[20:])].copy()
        current["benchmark"] = "TEST_ETF"
        current["asof_session"] = current["forecast_date"] - pd.Timedelta(days=1)
        current["q_sentinel"] = [
            float(index) for index in range(len(current))
        ]
        protocol = {
            "source_profile": "ordinary_massive_retrospective",
            "source_artifacts": {"stock_count": 2, "sector_count": 1},
            "feature_blocks": blocks,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "full_d2.parquet"
            full_d2.to_parquet(path, index=False)
            stale, stale_audit = run_controls.prepare_control_panel(
                current, protocol, "lag20", full_d2_path=path
            )
        self.assertEqual(len(stale), len(current))
        self.assertTrue(stale_audit["all_rows_available"])
        aaa_first = stale[
            stale["stock"].eq("AAA")
            & stale["forecast_date"].eq(dates[20])
        ].iloc[0]
        self.assertEqual(aaa_first[d2[0]], 0)
        self.assertEqual(
            aaa_first["_control_source_date"], dates[0]
        )
        self.assertEqual(
            stale["q_sentinel"].sort_values().tolist(),
            current["q_sentinel"].sort_values().tolist(),
        )

        wrong, wrong_audit = run_controls.prepare_control_panel(
            current, protocol, "wrong_stock"
        )
        self.assertEqual(len(wrong), len(current))
        self.assertEqual(
            wrong_audit["mapping"]["test_sector"],
            {"AAA": "BBB", "BBB": "AAA"},
        )
        aaa_first = wrong[
            wrong["stock"].eq("AAA")
            & wrong["forecast_date"].eq(dates[20])
        ].iloc[0]
        self.assertEqual(aaa_first[d2[0]], 12_000)
        self.assertEqual(
            wrong["q_sentinel"].sort_values().tolist(),
            current["q_sentinel"].sort_values().tolist(),
        )

    def test_final_comparison_ladder_limits_d5_targets(self) -> None:
        without_d5 = run_final_comparison.comparison_ladder()
        self.assertFalse(
            any(item.candidate == "D5" for item in without_d5)
        )
        with_d5 = run_final_comparison.comparison_ladder(("t1_etf",))
        d5_rows = [item for item in with_d5 if item.candidate == "D5"]
        self.assertEqual(
            [item.key for item in d5_rows],
            ["D5_vs_D0", "D5_vs_D3"],
        )
        self.assertTrue(
            all(item.targets == ("t1_etf",) for item in d5_rows)
        )

    def test_final_comparison_merge_requires_exact_keys_and_actuals(
        self,
    ) -> None:
        keys = {
            "fold": "fold_1",
            "target": "t1_etf",
            "forecast_date": pd.Timestamp("2025-01-02"),
            "sector": "test_sector",
            "stock": "AAA",
            "benchmark": "TEST_ETF",
        }
        base = pd.DataFrame(
            [
                {
                    **keys,
                    "actual_fisher_z": 1.0,
                    "actual_correlation": np.tanh(1.0),
                    "predicted_fisher_z": 0.0,
                    "predicted_correlation": 0.0,
                }
            ]
        )
        candidate = base.copy()
        candidate["predicted_fisher_z"] = 0.5
        candidate["predicted_correlation"] = np.tanh(0.5)
        comparison = run_final_comparison.Comparison(
            "candidate_vs_base",
            "candidate",
            "base",
            "test",
        )
        merged = run_final_comparison._merge_pair(
            comparison, {"candidate": candidate, "base": base}
        )
        self.assertAlmostEqual(
            float(merged["squared_loss_delta"].iloc[0]), -0.75
        )

        missing = base.iloc[0:0].copy()
        with self.assertRaisesRegex(ValueError, "keys differ"):
            run_final_comparison._merge_pair(
                comparison, {"candidate": candidate, "base": missing}
            )
        wrong_actual = base.copy()
        wrong_actual["actual_fisher_z"] = 2.0
        with self.assertRaisesRegex(ValueError, "actual targets differ"):
            run_final_comparison._merge_pair(
                comparison,
                {"candidate": candidate, "base": wrong_actual},
            )

    def test_final_bootstrap_is_fold_contained_and_row_weighted(self) -> None:
        rows: list[dict[str, object]] = []
        for fold, row_count, delta in (
            ("fold_1", 2, -2.0),
            ("fold_2", 1, 4.0),
        ):
            for date in pd.date_range("2025-01-02", periods=2, freq="B"):
                for stock_index in range(row_count):
                    rows.append(
                        {
                            "fold": fold,
                            "forecast_date": date,
                            "stock": f"{fold}_{stock_index}",
                            "squared_loss_delta": delta,
                        }
                    )
        frame = pd.DataFrame(rows)
        samples = run_final_comparison._moving_block_samples(
            frame,
            block_sessions=1,
            resamples=100,
            rng=np.random.default_rng(1729),
        )
        np.testing.assert_allclose(samples, 0.0)

    def test_final_useful_news_gate_requires_all_three_ci_checks(
        self,
    ) -> None:
        records: list[dict[str, object]] = []
        for comparison in (
            "D3_vs_D0",
            "D3_vs_C-D3-L20",
            "D3_vs_C-D3-WS",
        ):
            for target in common.TARGETS:
                records.append(
                    {
                        "comparison": comparison,
                        "target": target,
                        "mean_squared_loss_delta": -0.01,
                        "bootstrap_ci_upper": -0.001,
                        "folds_candidate_better": 2,
                    }
                )
        comparisons = pd.DataFrame(records)
        passed = run_final_comparison._decision_gate(comparisons)
        self.assertTrue(passed["all_required_gates_passed"].all())

        mask = (
            comparisons["comparison"].eq("D3_vs_C-D3-WS")
            & comparisons["target"].eq("t2_loo")
        )
        comparisons.loc[mask, "bootstrap_ci_upper"] = 0.001
        failed = run_final_comparison._decision_gate(comparisons)
        row = failed[failed["target"].eq("t2_loo")].iloc[0]
        self.assertFalse(row["beats_wrong_stock_d2_ci"])
        self.assertFalse(row["all_required_gates_passed"])

    def test_elastic_net_selection_stability_is_fold_specific(self) -> None:
        d2_feature = "d2_normalized_00"
        level_feature = "d2_level_00"
        features = ["q_feature", d2_feature, level_feature]
        fits: list[dict[str, object]] = []
        coefficients = {
            "fold_1": [1.0, 0.5, 0.0],
            "fold_2": [1.0, 0.0, 0.2],
            "fold_3": [1.0, -0.5, 0.0],
        }
        for target in common.TARGETS:
            for fold in ("fold_1", "fold_2", "fold_3"):
                fits.append(
                    {
                        "target": target,
                        "fold": fold,
                        "feature_count": len(features),
                        "features": features,
                        "coefficients": [
                            {
                                "feature": feature,
                                "coefficient_standardized": coefficient,
                            }
                            for feature, coefficient in zip(
                                features, coefficients[fold], strict=True
                            )
                        ],
                    }
                )
        protocol = {
            "folds": self._folds(),
            "feature_blocks": self._feature_blocks(include_d43=False),
        }
        stability, summary = run_final_comparison._selection_stability(
            "D4", fits, protocol
        )
        q_row = stability[
            stability["target"].eq("t1_etf")
            & stability["feature"].eq("q_feature")
        ].iloc[0]
        self.assertEqual(q_row["folds_nonzero"], 3)
        self.assertTrue(q_row["sign_consistent_when_selected"])
        d2_row = stability[
            stability["target"].eq("t1_etf")
            & stability["feature"].eq(d2_feature)
        ].iloc[0]
        self.assertEqual(d2_row["folds_nonzero"], 2)
        self.assertFalse(d2_row["sign_consistent_when_selected"])
        by_target = {
            record["target"]: record
            for record in summary
        }
        self.assertEqual(
            by_target["t1_etf"]["stable_all_folds_count"], 1
        )
        self.assertEqual(
            by_target["t1_etf"]["d2_union_selected_count"], 1
        )


if __name__ == "__main__":
    unittest.main()
