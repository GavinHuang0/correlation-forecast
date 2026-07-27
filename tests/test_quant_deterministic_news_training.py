from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from scripts.correlation_training import training_common as quant_common
from scripts.quant_deterministic_news_training import common
from scripts.quant_deterministic_news_training import run_final_comparison
from scripts.quant_deterministic_news_training import run_joint_linear
from scripts.quant_deterministic_news_training import run_residual


class QuantDeterministicNewsTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = common.load_protocol()

    def test_feature_ladder_counts_and_allowlisted_target_words(self) -> None:
        expected = {
            "A0-L": 56,
            "A0-T": 56,
            "A1": 43,
            "A2": 74,
            "A3": 82,
            "A4": 90,
            "A5": 99,
            "A6": 99,
        }
        for spec in quant_common.target_specs():
            for bundle, count in expected.items():
                features = common.joint_features(
                    bundle, spec, self.protocol
                )
                self.assertEqual(len(features), count)
                self.assertEqual(len(features), len(set(features)))
            self.assertIn(
                "observed_direct_target_article_count",
                common.joint_features("A5", spec, self.protocol),
            )

    def test_xgboost_candidates_resolve_to_frozen_quant_v1_configs(self) -> None:
        candidates = common.xgboost_candidate_configs(self.protocol)
        quant_protocol = common.load_json(
            Path(self.protocol["isolation"]["quant_protocol"])
        )
        self.assertEqual(len(candidates), 4)
        self.assertEqual(
            candidates,
            quant_protocol["rung_3"]["candidate_configs"],
        )

    def test_local_linear_wrappers_accept_target_named_news_feature(self) -> None:
        feature = "observed_direct_target_article_count"
        response = "response"
        train = pd.DataFrame(
            {
                feature: np.arange(12, dtype=float),
                response: np.linspace(-0.4, 0.5, 12),
            }
        )
        validation = pd.DataFrame(
            {
                feature: np.arange(12, 18, dtype=float),
                response: np.linspace(0.6, 0.9, 6),
            }
        )
        selected, candidates = common.select_linear_hyperparameters(
            train,
            validation,
            [feature],
            response,
            alpha_grid=[0.001, 0.01],
            l1_ratios=[0.5, 1.0],
        )
        predicted, _ = common.fit_predict_linear(
            "elastic_net",
            pd.concat([train, validation], ignore_index=True),
            validation,
            [feature],
            response,
            selected,
        )
        self.assertTrue(candidates)
        self.assertEqual(len(predicted), len(validation))
        self.assertTrue(np.isfinite(predicted).all())

    def test_runners_use_local_linear_wrappers(self) -> None:
        for module in (run_joint_linear, run_residual):
            source = inspect.getsource(module)
            self.assertNotIn(
                "quant_common.select_linear_hyperparameters(",
                source,
            )
            self.assertNotIn(
                "quant_common.fit_predict_linear(",
                source,
            )
            self.assertIn("common.fit_predict_linear(", source)
        self.assertIn(
            "common.select_linear_hyperparameters(",
            inspect.getsource(run_joint_linear),
        )
        self.assertIn(
            "common.select_linear_hyperparameters(",
            inspect.getsource(run_residual),
        )

    def test_missingness_flags_are_created_from_original_nulls(self) -> None:
        panel = pd.DataFrame(
            {
                "hours_since_latest_precise_target_article": [np.nan, 2.0],
                "hours_since_latest_precise_common_article": [1.0, np.nan],
                "observed_target_news_burst_60_session": [np.nan, 0.0],
            }
        )
        output = common.add_missingness_indicators(panel, self.protocol)
        self.assertEqual(
            output[
                "missing_hours_since_latest_precise_target_article"
            ].tolist(),
            [1, 0],
        )
        self.assertEqual(
            output[
                "missing_hours_since_latest_precise_common_article"
            ].tolist(),
            [0, 1],
        )
        self.assertEqual(
            output[
                "missing_observed_target_news_burst_60_session"
            ].tolist(),
            [1, 0],
        )

    def _control_fixture(self) -> pd.DataFrame:
        d43 = common.deterministic_features(self.protocol)
        dates = pd.date_range("2024-01-02", periods=25, freq="B")
        rows = []
        stocks = ["A", "B", "C", "D", "E", "F"]
        for date_position, date in enumerate(dates):
            for stock_position, stock in enumerate(stocks):
                row = {
                    "forecast_date": date,
                    "sector": "sector",
                    "stock": stock,
                    "benchmark": "ETF",
                }
                for feature_position, feature in enumerate(d43):
                    row[feature] = (
                        date_position * 10_000
                        + stock_position * 100
                        + feature_position
                    )
                rows.append(row)
        return pd.DataFrame(rows)

    def test_lag20_control_shifts_by_stock_without_wrap(self) -> None:
        panel = self._control_fixture()
        output, audit = common.prepare_control_panel(
            panel, self.protocol, "lag20"
        )
        feature = common.deterministic_features(self.protocol)[0]
        stock_a = output[output["stock"].eq("A")].sort_values("forecast_date")
        self.assertFalse(stock_a["_control_available"].iloc[:20].any())
        self.assertTrue(stock_a["_control_available"].iloc[20:].all())
        self.assertTrue(stock_a[feature].iloc[:20].isna().all())
        self.assertEqual(stock_a[feature].iloc[20], 0)
        self.assertFalse(audit["wrapped"])

    def test_wrong_stock_control_rotates_entire_d43_within_sector(self) -> None:
        panel = self._control_fixture()
        output, audit = common.prepare_control_panel(
            panel, self.protocol, "wrong_stock"
        )
        d43 = list(common.deterministic_features(self.protocol))
        date = output["forecast_date"].min()
        target = output[
            output["forecast_date"].eq(date) & output["stock"].eq("A")
        ].iloc[0]
        donor = panel[
            panel["forecast_date"].eq(date) & panel["stock"].eq("B")
        ].iloc[0]
        self.assertEqual(target[d43].tolist(), donor[d43].tolist())
        self.assertEqual(audit["mapping"]["sector:A"], "B")

    def test_t2_internal_development_mask_purges_crossing_labels(self) -> None:
        spec = next(
            item
            for item in quant_common.target_specs()
            if item.name == "t2_etf"
        )
        frame = pd.DataFrame(
            {
                "fold": ["fold_1"] * 4,
                "forecast_date": pd.to_datetime(
                    [
                        "2025-03-27",
                        "2025-03-28",
                        "2025-04-01",
                        "2025-06-27",
                    ]
                ),
                "target_etf_t2_end_date": pd.to_datetime(
                    [
                        "2025-03-31",
                        "2025-04-01",
                        "2025-04-07",
                        "2025-07-03",
                    ]
                ),
                "_control_available": [True] * 4,
            }
        )
        train, validation = run_residual._development_masks(frame, spec)
        self.assertEqual(train.tolist(), [True, False, False, False])
        self.assertEqual(validation.tolist(), [False, False, True, False])

    def test_parity_statistics_fail_closed_on_key_difference(self) -> None:
        base = pd.DataFrame(
            {
                "fold": ["fold_1"],
                "target": ["t1_etf"],
                "forecast_date": pd.to_datetime(["2025-01-02"]),
                "sector": ["s"],
                "stock": ["x"],
                "benchmark": ["e"],
                "predicted_fisher_z": [0.1],
                "predicted_correlation": [np.tanh(0.1)],
            }
        )
        stats = common.parity_statistics(base, base)
        self.assertEqual(stats["maximum_absolute_fisher_z_difference"], 0.0)
        changed = base.copy()
        changed["stock"] = "y"
        with self.assertRaises(ValueError):
            common.parity_statistics(changed, base)

    def test_final_pair_merge_is_exactly_key_aligned(self) -> None:
        comparison = run_final_comparison.Comparison(
            key="candidate_vs_base",
            candidate="candidate",
            base="base",
            family="test",
        )
        keys = {
            "fold": ["fold_1"],
            "target": ["t1_etf"],
            "forecast_date": pd.to_datetime(["2025-01-02"]),
            "sector": ["sector"],
            "stock": ["stock"],
            "benchmark": ["ETF"],
        }
        candidate = pd.DataFrame(
            {
                **keys,
                "actual_fisher_z": [0.5],
                "actual_correlation": [np.tanh(0.5)],
                "predicted_fisher_z": [0.4],
                "predicted_correlation": [np.tanh(0.4)],
            }
        )
        base = candidate.copy()
        base["predicted_fisher_z"] = 0.2
        base["predicted_correlation"] = np.tanh(0.2)
        merged = run_final_comparison._merge_pair(
            comparison,
            {"candidate": candidate, "base": base},
        )
        self.assertAlmostEqual(
            float(merged["squared_loss_delta"].iloc[0]),
            (0.5 - 0.4) ** 2 - (0.5 - 0.2) ** 2,
        )
        missing_key = base.copy()
        missing_key["stock"] = "another-stock"
        with self.assertRaises(ValueError):
            run_final_comparison._merge_pair(
                comparison,
                {"candidate": candidate, "base": missing_key},
            )

    def test_final_bootstrap_is_fold_contained_and_row_weighted(self) -> None:
        rows = []
        for fold, start, delta, rows_per_date in (
            ("fold_1", "2025-01-02", -2.0, 2),
            ("fold_2", "2025-02-03", 4.0, 1),
        ):
            for date in pd.date_range(start, periods=4, freq="B"):
                for row_number in range(rows_per_date):
                    rows.append(
                        {
                            "fold": fold,
                            "forecast_date": date,
                            "stock": f"{fold}-{row_number}",
                            "squared_loss_delta": delta,
                        }
                    )
        samples = run_final_comparison._moving_block_samples(
            pd.DataFrame(rows),
            block_sessions=2,
            resamples=50,
            rng=np.random.default_rng(1729),
        )
        # Every resample keeps four dates from each fold. Fold 1 contributes
        # twice as many rows per date, so (-2 * 8 + 4 * 4) / 12 = 0.
        np.testing.assert_allclose(samples, np.zeros(50), atol=0, rtol=0)

    def test_final_ladder_uses_dynamic_a7_targets(self) -> None:
        without_a7 = run_final_comparison.comparison_ladder()
        self.assertFalse(
            any(
                item.candidate == "A7" or item.base == "A7"
                for item in without_a7
            )
        )
        with_a7 = run_final_comparison.comparison_ladder(["t1_etf"])
        a7_rows = [
            item
            for item in with_a7
            if item.candidate == "A7" or item.base == "A7"
        ]
        self.assertTrue(a7_rows)
        self.assertTrue(
            all(item.targets == ("t1_etf",) for item in a7_rows)
        )
        keys = {item.key for item in a7_rows}
        self.assertIn("A7_vs_A5", keys)
        self.assertIn("A7_vs_A6", keys)
        winner_comparison = next(
            item
            for item in a7_rows
            if item.key == "A7_vs_S-B0_common_folds"
        )
        self.assertFalse(winner_comparison.controlled)

    def test_final_decision_gate_is_derived_from_required_comparisons(
        self,
    ) -> None:
        targets = ("t1_etf", "t1_loo", "t2_etf", "t2_loo")
        a5_keys = (
            "A5_vs_A0-L",
            "A5_vs_C-A5-L20",
            "A5_vs_C-A5-WS",
        )
        b4_keys = (
            "B4_vs_B0",
            "B4_vs_B1",
            "B4_vs_B2",
            "B4_vs_C-B4-L20",
            "B4_vs_C-B4-WS",
        )
        rows = []
        for target in targets:
            for comparison in (*a5_keys, *b4_keys):
                rows.append(
                    {
                        "comparison": comparison,
                        "target": target,
                        "mean_squared_loss_delta": -0.1,
                        "bootstrap_ci_upper": -0.01,
                        "folds_candidate_better": (
                            3 if comparison == "A5_vs_A0-L" else 2
                        ),
                    }
                )
        comparisons = pd.DataFrame(rows)
        passed = run_final_comparison._decision_gate(comparisons)
        self.assertEqual(len(passed), 8)
        self.assertTrue(passed["all_required_gates_passed"].all())

        failed = comparisons.copy()
        mask = (
            failed["comparison"].eq("A5_vs_C-A5-WS")
            & failed["target"].eq("t1_etf")
        )
        failed.loc[mask, "bootstrap_ci_upper"] = 0.01
        gate = run_final_comparison._decision_gate(failed)
        t1_a5 = gate[
            gate["candidate"].eq("A5")
            & gate["target"].eq("t1_etf")
        ].iloc[0]
        self.assertFalse(t1_a5["beats_wrong_stock_placebo_ci"])
        self.assertFalse(t1_a5["all_required_gates_passed"])

    def test_overwrite_removes_stale_files_and_manifest_excludes_itself(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol_path = root / "config" / "protocol.json"
            panel_path = root / "data" / "panel.parquet"
            protocol_path.parent.mkdir(parents=True)
            panel_path.parent.mkdir(parents=True)
            protocol_path.write_text("{}\n", encoding="utf-8")
            panel_path.write_bytes(b"panel fixture")
            with (
                mock.patch.object(common, "PROTOCOL_PATH", protocol_path),
                mock.patch.object(common, "PANEL_PATH", panel_path),
                mock.patch.object(
                    common, "OUTPUT_ROOT", root / "outputs"
                ),
                mock.patch.object(
                    common, "EXPERIMENT_ROOT", root / "experiments"
                ),
            ):
                write_arguments = {
                    "predictions": None,
                    "validation_predictions": None,
                    "fits": [],
                    "fold_metrics": [],
                    "summary": {"status": "complete", "metrics": []},
                    "review": {"status": "passed"},
                    "model_config": {"fixture": True},
                    "dependencies": {},
                }
                common.write_bundle(
                    "A7",
                    **write_arguments,
                    extra_outputs={"stale.json": {"stale": True}},
                )
                bundle = common.bundle_for("A7")
                stale_tracked = bundle.experiment_path / "stale.txt"
                stale_tracked.write_text("stale", encoding="utf-8")

                common.ensure_no_existing_bundle("A7", overwrite=True)
                self.assertFalse(bundle.output_path.exists())
                self.assertFalse(bundle.experiment_path.exists())

                refs = common.write_bundle(
                    "A7",
                    **write_arguments,
                    extra_outputs={},
                )
                manifest_path = bundle.output_path / "manifest.json"
                manifest = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
                self.assertNotIn("manifest.json", manifest["artifacts"])
                self.assertNotIn("stale.json", manifest["artifacts"])
                self.assertFalse(
                    (bundle.output_path / "stale.json").exists()
                )
                self.assertFalse(stale_tracked.exists())
                self.assertEqual(
                    refs["manifest.json"]["sha256"],
                    common.sha256_file(manifest_path),
                )

    def test_verified_bundle_artifact_rejects_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol_path = root / "config" / "protocol.json"
            panel_path = root / "data" / "panel.parquet"
            status_path = root / "experiments" / "status.json"
            protocol_path.parent.mkdir(parents=True)
            panel_path.parent.mkdir(parents=True)
            status_path.parent.mkdir(parents=True)
            protocol_path.write_text("{}\n", encoding="utf-8")
            panel_path.write_bytes(b"panel fixture")
            with (
                mock.patch.object(common, "PROTOCOL_PATH", protocol_path),
                mock.patch.object(common, "PANEL_PATH", panel_path),
                mock.patch.object(
                    common, "OUTPUT_ROOT", root / "outputs"
                ),
                mock.patch.object(
                    common, "EXPERIMENT_ROOT", root / "experiments"
                ),
                mock.patch.object(common, "STATUS_PATH", status_path),
            ):
                bundle = common.bundle_for("A5")
                artifact_path = bundle.output_path / "predictions.parquet"
                artifact_path.parent.mkdir(parents=True)
                artifact_path.write_bytes(b"modified artifact")
                status_path.write_text(
                    json.dumps(
                        {
                            "protocol_sha256": common.sha256_file(
                                protocol_path
                            ),
                            "bundles": [
                                {"key": "A5", "status": "complete"}
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                manifest = {
                    "bundle": "A5",
                    "protocol_sha256": common.sha256_file(protocol_path),
                    "panel_sha256": common.sha256_file(panel_path),
                    "artifacts": {
                        "predictions.parquet": {
                            "path": artifact_path.as_posix(),
                            "sha256": "0" * 64,
                        }
                    },
                }
                (bundle.output_path / "manifest.json").write_text(
                    json.dumps(manifest),
                    encoding="utf-8",
                )

                with self.assertRaisesRegex(
                    RuntimeError, "hash mismatch"
                ):
                    common.verified_bundle_artifact(
                        "A5", "predictions.parquet"
                    )

    def test_final_results_markdown_is_hash_bound_and_tracked_identically(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol_path = root / "config" / "protocol.json"
            panel_path = root / "data" / "panel.parquet"
            protocol_path.parent.mkdir(parents=True)
            panel_path.parent.mkdir(parents=True)
            protocol_path.write_text("{}\n", encoding="utf-8")
            panel_path.write_bytes(b"panel fixture")
            report = "# Final fixture\n\nHash-bound report.\n"
            with (
                mock.patch.object(common, "PROTOCOL_PATH", protocol_path),
                mock.patch.object(common, "PANEL_PATH", panel_path),
                mock.patch.object(
                    common, "OUTPUT_ROOT", root / "outputs"
                ),
                mock.patch.object(
                    common, "EXPERIMENT_ROOT", root / "experiments"
                ),
            ):
                refs = common.write_bundle(
                    "FINAL",
                    predictions=None,
                    validation_predictions=None,
                    fits=[],
                    fold_metrics=[],
                    summary={"status": "complete", "metrics": []},
                    review={"status": "passed"},
                    model_config={"fixture": True},
                    dependencies={},
                    extra_outputs={},
                    results_markdown=report,
                )
                bundle = common.bundle_for("FINAL")
                output_report = bundle.output_path / "RESULTS.md"
                tracked_report = bundle.experiment_path / "RESULTS.md"
                manifest = json.loads(
                    (bundle.output_path / "manifest.json").read_text(
                        encoding="utf-8"
                    )
                )

                self.assertEqual(
                    output_report.read_text(encoding="utf-8"), report
                )
                self.assertEqual(
                    tracked_report.read_text(encoding="utf-8"), report
                )
                self.assertEqual(
                    common.sha256_file(output_report),
                    common.sha256_file(tracked_report),
                )
                self.assertEqual(
                    manifest["artifacts"]["RESULTS.md"]["sha256"],
                    common.sha256_file(output_report),
                )
                self.assertEqual(
                    refs["RESULTS.md"]["sha256"],
                    common.sha256_file(output_report),
                )


if __name__ == "__main__":
    unittest.main()
