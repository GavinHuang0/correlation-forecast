from __future__ import annotations

import numpy as np
import pandas as pd
import tempfile
import unittest
from pathlib import Path

from scripts.quant_deterministic_news_training_v4 import common, contract
from scripts.quant_deterministic_news_training_v4 import run_final_comparison
from scripts.quant_deterministic_news_training_v4 import run_long
from scripts.quant_deterministic_news_training_v4 import run_short


def _long_split_fixture() -> pd.DataFrame:
    rows = []
    for number in range(6, 14):
        rows.append(
            {
                "target": "t1_etf",
                "fold": f"fold_{number:02d}",
                "forecast_date": pd.Timestamp(2020 + number, 1, 1),
            }
        )
    return pd.DataFrame(rows)


def _prediction_pair() -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_rows = []
    base_rows = []
    for date in pd.date_range("2025-01-02", periods=20, freq="B"):
        for stock in ("AAA", "BBB"):
            row = {
                "fold": "fold_1",
                "target": "t1_etf",
                "forecast_date": date,
                "sector": "Test",
                "stock": stock,
                "benchmark": "ETF",
                "actual_fisher_z": 1.0,
            }
            candidate_rows.append(
                {**row, "predicted_fisher_z": 1.0}
            )
            base_rows.append({**row, "predicted_fisher_z": 0.0})
    return pd.DataFrame(candidate_rows), pd.DataFrame(base_rows)


class V4TrainingTests(unittest.TestCase):
    def test_soft_route_feature_contract_is_exact_and_disjoint(self) -> None:
        self.assertEqual(len(contract.CURRENT_9), 9)
        self.assertEqual(len(contract.INNOVATION_9), 9)
        self.assertEqual(len(contract.SOFT_ROUTE_19), 19)
        self.assertEqual(len(contract.COUPLING_6), 6)
        self.assertEqual(len(contract.QUALITY_4), 4)
        self.assertEqual(len(contract.QUALITY_ONLY_5), 5)
        self.assertEqual(
            contract.SOFT_ROUTE_19[-1],
            "lsoft_observed_no_selected_article",
        )
        self.assertTrue(
            set(contract.CURRENT_9).isdisjoint(contract.INNOVATION_9)
        )
        self.assertTrue(
            set(contract.SOFT_ROUTE_19).isdisjoint(contract.COUPLING_6)
        )
        self.assertTrue(
            set(contract.SOFT_ROUTE_19).isdisjoint(contract.QUALITY_4)
        )
        self.assertEqual(
            contract.QUALITY_ONLY_5[-1],
            "lsoft_observed_no_selected_article",
        )

    def test_control_panel_routing_and_matched_bases_are_explicit(self) -> None:
        self.assertEqual(
            common.panel_source_key("C-J1-PERM"), "permuted_panel"
        )
        self.assertEqual(
            common.panel_source_key("C-RRES-L19-PERM"), "permuted_panel"
        )
        self.assertEqual(
            common.panel_source_key("C-J1-QUALITY"), "canonical_panel"
        )
        self.assertEqual(run_short.matched_base_key("C-J1-QUALITY"), "S0")
        self.assertEqual(run_short.matched_base_key("C-J2-QUALITY"), "S1")
        self.assertEqual(run_short.matched_base_key("C-J2-PERM"), "S1")

    def test_quality_only_feature_architectures_exclude_event_identity(self) -> None:
        protocol = {
            "feature_blocks": {
                "q56_by_target": {
                    target: [f"q_{index}" for index in range(56)]
                    for target in contract.TARGETS
                },
                "d2_normalized_30": [f"d_{index}" for index in range(30)],
            }
        }
        short = common.model_features("C-J1-QUALITY", "t1_etf", protocol)
        short_d2 = common.model_features(
            "C-J2-QUALITY", "t1_etf", protocol
        )
        long = common.model_features(
            "C-RRES-L19-QUALITY", "t1_etf", protocol
        )
        self.assertEqual(short[-5:], contract.QUALITY_ONLY_5)
        self.assertEqual(short_d2[-5:], contract.QUALITY_ONLY_5)
        self.assertEqual(long, contract.QUALITY_ONLY_5)
        self.assertTrue(set(short).isdisjoint(contract.CURRENT_9))

    def test_elastic_net_selection_uses_exact_locked_grid(self) -> None:
        values = np.linspace(-2, 2, 80)
        train = pd.DataFrame({"signal": values, "response": values})
        validation = pd.DataFrame(
            {"signal": values[::2], "response": values[::2]}
        )
        alphas = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]
        ratios = [0.1, 0.5, 0.9, 1.0]
        selected, candidates = common.select_elastic_net(
            train,
            validation,
            ("signal",),
            "response",
            {
                "model_tuning": {
                    "alpha_grid": alphas,
                    "l1_ratio_grid": ratios,
                }
            },
        )
        self.assertEqual(len(candidates), 28)
        self.assertEqual(
            {(row["alpha"], row["l1_ratio"]) for row in candidates},
            {(alpha, ratio) for alpha in alphas for ratio in ratios},
        )
        self.assertTrue(all(not row["boundary_expansion"] for row in candidates))
        self.assertIn(selected["alpha"], alphas)
        self.assertIn(selected["l1_ratio"], ratios)

    def test_long_split_uses_only_earlier_oos_folds(self) -> None:
        train, validation, test, audit = common.long_split(
            _long_split_fixture(), "t1_etf", "fold_09"
        )
        self.assertEqual(train["fold"].tolist(), ["fold_06", "fold_07"])
        self.assertEqual(validation["fold"].tolist(), ["fold_08"])
        self.assertEqual(test["fold"].tolist(), ["fold_09"])
        self.assertEqual(audit["training_folds"], ["fold_06", "fold_07"])
        self.assertEqual(audit["validation_fold"], "fold_08")

    def test_long_split_expands_prequential_training_history(self) -> None:
        train, validation, test, audit = common.long_split(
            _long_split_fixture(), "t1_etf", "fold_13"
        )
        self.assertEqual(
            train["fold"].tolist(),
            [
                "fold_06",
                "fold_07",
                "fold_08",
                "fold_09",
                "fold_10",
                "fold_11",
            ],
        )
        self.assertEqual(validation["fold"].tolist(), ["fold_12"])
        self.assertEqual(test["fold"].tolist(), ["fold_13"])
        self.assertEqual(audit["test_rows"], 1)

    def test_residual_selection_tunes_shrinkage_on_validation(self) -> None:
        train_x = np.linspace(-2, 2, 80)
        validation_x = np.linspace(-1.8, 1.8, 40)
        train = pd.DataFrame(
            {
                "signal": train_x,
                "_residual_fisher_z": train_x,
                "_base_fisher_z": np.zeros_like(train_x),
                "_actual_fisher_z": train_x,
            }
        )
        validation = pd.DataFrame(
            {
                "signal": validation_x,
                "_residual_fisher_z": validation_x,
                "_base_fisher_z": np.zeros_like(validation_x),
                "_actual_fisher_z": validation_x,
            }
        )
        protocol = {
            "model_tuning": {
                "alpha_grid": [0.0001],
                "l1_ratio_grid": [1.0],
                "residual_correction_shrinkage_grid": [0.0, 0.5, 1.0],
            }
        }
        selected, candidates = run_long.select_residual_parameters(
            train, validation, ("signal",), protocol
        )
        self.assertEqual(selected["correction_shrinkage"], 1.0)
        self.assertEqual(len(candidates), 3)
        self.assertLess(
            candidates[0]["validation_mse"],
            candidates[-1]["validation_mse"],
        )

    def test_whole_date_bootstrap_is_deterministic_and_paired(self) -> None:
        candidate, base = _prediction_pair()
        daily = run_final_comparison.paired_date_losses(
            candidate, base, target="t1_etf"
        )
        self.assertEqual(len(daily), 20)
        self.assertTrue(daily["rows"].eq(2).all())
        first = run_final_comparison.paired_bootstrap(
            daily, resamples=50, block_sessions=5, seed=11
        )
        second = run_final_comparison.paired_bootstrap(
            daily, resamples=50, block_sessions=5, seed=11
        )
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["incremental_mse_r2"], 1.0)
        self.assertEqual(
            first["incremental_mse_r2_bootstrap_95pct"], [1.0, 1.0]
        )

    def test_moving_blocks_reject_a_block_longer_than_sample(self) -> None:
        with self.assertRaisesRegex(ValueError, "block_sessions"):
            run_final_comparison.moving_block_positions(
                4, block_sessions=5, rng=np.random.default_rng(1)
            )

    def test_bootstrap_blocks_are_fold_contained_with_discontinuous_folds(self) -> None:
        candidate_rows = []
        base_rows = []
        for fold, start in (("fold_1", "2025-01-02"), ("fold_2", "2025-10-01")):
            for date in pd.date_range(start, periods=12, freq="B"):
                row = {
                    "fold": fold,
                    "target": "t1_etf",
                    "forecast_date": date,
                    "sector": "Test",
                    "stock": "AAA",
                    "benchmark": "ETF",
                    "actual_fisher_z": 1.0,
                }
                candidate_rows.append({**row, "predicted_fisher_z": 0.8})
                base_rows.append({**row, "predicted_fisher_z": 0.0})
        daily = run_final_comparison.paired_date_losses(
            pd.DataFrame(candidate_rows),
            pd.DataFrame(base_rows),
            target="t1_etf",
        )
        self.assertEqual(daily.groupby("fold").size().to_dict(), {"fold_1": 12, "fold_2": 12})
        first = run_final_comparison.paired_bootstrap(
            daily, resamples=40, block_sessions=3, seed=19
        )
        second = run_final_comparison.paired_bootstrap(
            daily, resamples=40, block_sessions=3, seed=19
        )
        self.assertEqual(first, second)
        self.assertTrue(first["bootstrap_blocks_fold_contained"])

    def test_useful_gate_requires_all_controls_and_marks_stack_incomplete(self) -> None:
        rows = []
        for candidate, requirements in run_final_comparison.USEFUL_GATE_REQUIREMENTS.items():
            for target in contract.TARGETS:
                for comparison in requirements:
                    rows.append(
                        {
                            "comparison": comparison,
                            "target": target,
                            "comparison_gate_pass": True,
                            "point_loss_delta_gate_pass": True,
                            "loss_delta_ci_upper_gate_pass": True,
                            "fold_improvement_gate_pass": True,
                        }
                    )
        gates = run_final_comparison.build_useful_semantic_gates(
            pd.DataFrame(rows)
        )
        primary = gates[gates["candidate"].eq("J1")]
        stack = gates[gates["candidate"].eq("RSTACK")]
        self.assertTrue(primary["useful_semantic_gate_pass"].all())
        self.assertFalse(stack["useful_semantic_gate_evaluable"].any())
        self.assertFalse(stack["useful_semantic_gate_pass"].any())

    def test_matched_base_validation_fails_on_missing_row(self) -> None:
        candidate, base = _prediction_pair()
        with self.assertRaisesRegex(ValueError, "keys differ"):
            common.validate_matched_prediction_rows(candidate, base.iloc[:-1])

    def test_long_anchor_target_parity_is_bitwise(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "base.parquet"
            panel_rows = {
                "forecast_date": [pd.Timestamp("2022-11-01")],
                "sector": ["Test"],
                "stock": ["AAA"],
                "benchmark": ["ETF"],
            }
            for name in (
                *contract.SOFT_ROUTE_19,
                *contract.COUPLING_6,
                *contract.QUALITY_4,
            ):
                panel_rows[name] = [0.0]
            response_by_target = {
                spec.name: spec.response_column
                for spec in common.quant_common.target_specs()
            }
            for response in response_by_target.values():
                panel_rows[response] = [0.25]
            panel = pd.DataFrame(panel_rows)
            base_rows = []
            for target in contract.TARGETS:
                base_rows.append(
                    {
                        "fold": "fold_06",
                        "target": target,
                        "model": "xgboost",
                        "forecast_date": pd.Timestamp("2022-11-01"),
                        "sector": "Test",
                        "stock": "AAA",
                        "benchmark": "ETF",
                        "actual_fisher_z": 0.25,
                        "actual_correlation": np.tanh(0.25),
                        "predicted_fisher_z": 0.2,
                        "predicted_correlation": np.tanh(0.2),
                        "persistence_correlation": 0.1,
                        "persistence_fisher_z": np.arctanh(0.1),
                    }
                )
            pd.DataFrame(base_rows).to_parquet(path, index=False)
            protocol = {
                "source_artifacts": {
                    "quant_v2_rung03_predictions": {
                        **common.artifact_record(path),
                    }
                }
            }
            _, audit = common.load_long_base(protocol, panel)
            self.assertTrue(
                all(
                    row["bitwise_equal"]
                    for row in audit["target_parity"].values()
                )
            )
            changed = pd.DataFrame(base_rows)
            changed.loc[changed["target"].eq("t1_etf"), "actual_fisher_z"] = 0.3
            changed.to_parquet(path, index=False)
            protocol["source_artifacts"]["quant_v2_rung03_predictions"] = {
                **common.artifact_record(path)
            }
            with self.assertRaisesRegex(ValueError, "differs"):
                common.load_long_base(protocol, panel)


if __name__ == "__main__":
    unittest.main()
