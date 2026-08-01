from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

from scripts.quant_deterministic_news_training_v3 import common, contract
from scripts.quant_deterministic_news_training_v3 import lock_protocol
from scripts.quant_deterministic_news_training_v3 import (
    run_final_comparison,
)
from scripts.quant_deterministic_news_training_v3 import run_linear
from scripts.quant_deterministic_news_training_v3 import run_xgboost


class V3TrainingContractTests(unittest.TestCase):
    def test_wlite_feature_partition_and_rung_counts(self) -> None:
        self.assertEqual(len(contract.WLITE_17), 17)
        self.assertEqual(len(contract.EVENT_CONTENT_4), 4)
        self.assertEqual(len(contract.SEMANTIC_CONTENT_11), 11)
        self.assertEqual(len(contract.COVERAGE_TEXT_6), 6)
        self.assertEqual(
            set(contract.SEMANTIC_CONTENT_11)
            | set(contract.COVERAGE_TEXT_6),
            set(contract.WLITE_17),
        )
        self.assertTrue(
            set(contract.EVENT_CONTENT_4).issubset(
                contract.SEMANTIC_CONTENT_11
            )
        )
        self.assertEqual(
            len(
                set(contract.WLITE_17)
                - set(contract.EVENT_CONTENT_4)
            ),
            13,
        )
        protocol = {
            "feature_blocks": {
                "q56_by_target": {
                    target: [f"q_{target}_{index}" for index in range(56)]
                    for target in contract.TARGETS
                },
                "d2_normalized_30": [
                    f"d2_feature_{index}" for index in range(30)
                ],
                "wlite_17": list(contract.WLITE_17),
            }
        }
        self.assertEqual(
            len(common.model_features("S0", "t1_etf", protocol)), 56
        )
        self.assertEqual(
            len(common.model_features("S1", "t1_etf", protocol)), 86
        )
        self.assertEqual(
            len(common.model_features("S2", "t1_etf", protocol)), 73
        )
        self.assertEqual(
            len(common.model_features("S3", "t1_etf", protocol)), 103
        )
        self.assertEqual(
            len(
                common.model_features(
                    "C-S2-COV", "t1_etf", protocol
                )
            ),
            62,
        )
        self.assertEqual(
            len(
                common.model_features(
                    "C-S3-COV", "t1_etf", protocol
                )
            ),
            92,
        )

    @staticmethod
    def _control_panel() -> pd.DataFrame:
        rows = []
        dates = pd.bdate_range("2024-01-02", periods=22)
        for sector, stocks in (
            ("A", ["A1", "A2", "A3", "A4", "A5", "A6"]),
            ("B", ["B1", "B2", "B3", "B4", "B5", "B6"]),
        ):
            for date_index, forecast_date in enumerate(dates):
                for stock_index, stock in enumerate(stocks):
                    row = {
                        "forecast_date": forecast_date,
                        "sector": sector,
                        "stock": stock,
                        "benchmark": f"{sector}ETF",
                    }
                    for feature_index, feature in enumerate(
                        contract.WLITE_17
                    ):
                        row[feature] = (
                            1000 * date_index
                            + 100 * stock_index
                            + feature_index
                        )
                    rows.append(row)
        return pd.DataFrame(rows)

    def test_stale_control_shifts_only_event_content_without_wrap(self) -> None:
        panel = self._control_panel()
        output, audit = common.prepare_stale_event_control(panel)
        original = panel.sort_values(
            list(contract.PANEL_KEYS), kind="mergesort"
        ).reset_index(drop=True)
        self.assertEqual(audit["sessions"], 20)
        self.assertEqual(audit["rows_without_stale_donor"], 20 * 12)
        early = output["forecast_date"].eq(
            output["forecast_date"].min()
        )
        self.assertTrue(
            output.loc[early, list(contract.EVENT_CONTENT_4)]
            .isna()
            .all()
            .all()
        )
        latest_date = output["forecast_date"].max()
        recipient = output[
            output["forecast_date"].eq(latest_date)
            & output["stock"].eq("A1")
        ].iloc[0]
        donor_date = sorted(output["forecast_date"].unique())[-21]
        donor = original[
            original["forecast_date"].eq(donor_date)
            & original["stock"].eq("A1")
        ].iloc[0]
        for feature in contract.EVENT_CONTENT_4:
            self.assertEqual(recipient[feature], donor[feature])
        for feature in contract.COVERAGE_TEXT_6:
            current = original[
                original["forecast_date"].eq(latest_date)
                & original["stock"].eq("A1")
            ].iloc[0]
            self.assertEqual(recipient[feature], current[feature])

    def test_wrong_stock_rotates_content_and_preserves_coverage(self) -> None:
        panel = self._control_panel()
        output, audit = common.prepare_wrong_stock_control(panel)
        date = output["forecast_date"].min()
        recipient = output[
            output["forecast_date"].eq(date)
            & output["sector"].eq("A")
            & output["stock"].eq("A1")
        ].iloc[0]
        donor = panel[
            panel["forecast_date"].eq(date)
            & panel["sector"].eq("A")
            & panel["stock"].eq("A2")
        ].iloc[0]
        original = panel[
            panel["forecast_date"].eq(date)
            & panel["sector"].eq("A")
            & panel["stock"].eq("A1")
        ].iloc[0]
        for feature in contract.SEMANTIC_CONTENT_11:
            self.assertEqual(recipient[feature], donor[feature])
        for feature in contract.COVERAGE_TEXT_6:
            self.assertEqual(recipient[feature], original[feature])
        self.assertEqual(audit["mapping"]["A/A1"], "A2")
        self.assertEqual(audit["mapping"]["A/A6"], "A1")

    def test_failed_gate_exact_counts_are_below_threshold(self) -> None:
        rate = (
            lock_protocol.EXPECTED_AGREEMENT_COUNT
            / lock_protocol.EXPECTED_SELECTED_ARTICLES
        )
        self.assertAlmostEqual(rate, 0.6512438599271114)
        self.assertLess(rate, lock_protocol.REQUIRED_AGREEMENT)
        self.assertEqual(
            lock_protocol.EXPECTED_AGREEMENT_COUNT
            + lock_protocol.EXPECTED_DISAGREEMENT_COUNT,
            lock_protocol.EXPECTED_SELECTED_ARTICLES,
        )

    def test_permutation_provenance_binds_string_seed_and_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "aggregation.json"
            panel_path = root / "daily_features_permuted.parquet"
            manifest_path = root / "daily_features_permuted.manifest.json"
            seed = "wlite-date-sector-event-permutation-v1"
            mapping_hash = "a" * 64
            config_path.write_text(
                json.dumps(
                    {
                        "variants": {
                            "permuted": {
                                "seed": seed,
                                "accepted_labels_only": True,
                                "other_features_contemporaneous": True,
                                "strata": [
                                    "forecast_date",
                                    "sector",
                                ],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            panel_path.write_bytes(b"locked-permuted-panel")
            config_hash = common.sha256_file(config_path)
            manifest_path.write_text(
                json.dumps(
                    {
                        "variant_id": "permuted",
                        "permutation_mapping_sha256": mapping_hash,
                        "inputs": {
                            "status_cue_rules": {
                                "path": config_path.as_posix(),
                                "sha256": config_hash,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            manifest_hash = common.sha256_file(manifest_path)
            protocol = {
                "controls": {
                    "date_sector_permutation": {
                        "seed": seed,
                        "permutation_mapping_sha256": mapping_hash,
                        "aggregation_config_path": config_path.as_posix(),
                        "aggregation_config_sha256": config_hash,
                        "permuted_daily_manifest_sha256": manifest_hash,
                    }
                },
                "source_artifacts": {
                    "permuted_daily_wlite": {
                        "path": panel_path.as_posix(),
                        "sha256": common.sha256_file(panel_path),
                        "manifest_path": manifest_path.as_posix(),
                        "manifest_sha256": manifest_hash,
                    }
                },
            }
            with (
                mock.patch.object(
                    contract, "AGGREGATION_CONFIG", config_path
                ),
                mock.patch.object(
                    contract,
                    "PERMUTED_DAILY_MANIFEST",
                    manifest_path,
                ),
            ):
                audit = common.verify_permutation_provenance(protocol)
                self.assertEqual(audit["seed"], seed)
                self.assertEqual(
                    audit["permutation_mapping_sha256"], mapping_hash
                )
                wrong_seed = copy.deepcopy(protocol)
                wrong_seed["controls"]["date_sector_permutation"][
                    "seed"
                ] = 1729
                with self.assertRaises(ValueError):
                    common.verify_permutation_provenance(wrong_seed)
                wrong_mapping = copy.deepcopy(protocol)
                wrong_mapping["controls"]["date_sector_permutation"][
                    "permutation_mapping_sha256"
                ] = "b" * 64
                with self.assertRaises(ValueError):
                    common.verify_permutation_provenance(wrong_mapping)

    def test_bound_implementation_includes_external_training_dependencies(
        self,
    ) -> None:
        required = {
            (
                "scripts/quant_deterministic_news_training_v3/"
                "run_final_comparison.py"
            ),
            "scripts/quant_deterministic_news_training_v2/common.py",
            "scripts/correlation_training/training_common.py",
            "scripts/correlation_training/build_modeling_panel.py",
        }
        self.assertTrue(
            required.issubset(
                {
                    path.as_posix()
                    for path in contract.BOUND_TRAINING_IMPLEMENTATION
                }
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = (root / "a.py", root / "b.py")
            for index, path in enumerate(files):
                path.write_text(f"value = {index}\n", encoding="utf-8")
            protocol = {
                "implementation": {
                    path.as_posix(): common.sha256_file(path)
                    for path in files
                }
            }
            with mock.patch.object(
                contract, "BOUND_TRAINING_IMPLEMENTATION", files
            ):
                verified = common.verify_locked_implementation(protocol)
                self.assertEqual(set(verified), set(protocol["implementation"]))
                protocol_path = root / "protocol.json"
                sidecar_path = root / "protocol.sha256"
                loadable = {
                    **protocol,
                    "experiment_id": (
                        "quant-deterministic-news-v3-w17-lite"
                    ),
                    "protocol_version": "3.0",
                    "status": "locked_before_training",
                    "targets": list(contract.TARGETS),
                    "join_keys": list(contract.PANEL_KEYS),
                    "claim_flags": {
                        "exploratory_fit_authorized": True,
                        "semantic_gate_passed": False,
                        "primary_training_eligible": False,
                        "confirmatory_eligible": False,
                    },
                    "semantic_gate": {
                        "passed": False,
                        "observed_choice_order_agreement": 0.65,
                        "required_choice_order_agreement": 0.85,
                        "override": {"authorized": True},
                    },
                    "feature_blocks": {
                        "wlite_17": list(contract.WLITE_17),
                        "d2_normalized_30": [
                            f"d2_{index}" for index in range(30)
                        ],
                        "q56_by_target": {
                            target: [
                                f"q_{target}_{index}"
                                for index in range(56)
                            ]
                            for target in contract.TARGETS
                        },
                    },
                    "folds": [
                        {"name": name}
                        for name in ("fold_1", "fold_2", "fold_3")
                    ],
                }
                protocol_path.write_text(
                    json.dumps(loadable), encoding="utf-8"
                )
                sidecar_path.write_text(
                    common.sha256_file(protocol_path) + "\n",
                    encoding="ascii",
                )
                with mock.patch.object(
                    contract, "PROTOCOL_SIDECAR", sidecar_path
                ):
                    common.load_protocol(protocol_path)
                files[1].write_text("value = 99\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    common.verify_locked_implementation(protocol)
                with mock.patch.object(
                    contract, "PROTOCOL_SIDECAR", sidecar_path
                ):
                    with self.assertRaises(ValueError):
                        common.load_protocol(protocol_path)

    def test_xgboost_budget_is_copied_and_hash_locked(self) -> None:
        v2 = common.load_json(
            Path("config/quant_deterministic_news_protocol_v2.json")
        )
        locked = lock_protocol._locked_xgboost_tuning(v2)
        protocol = {"model_tuning": {"shallow_xgboost": locked}}
        budget = run_xgboost.load_budget(protocol)
        self.assertEqual(len(budget["candidate_configs"]), 4)
        self.assertEqual(budget["max_estimators"], 2000)
        self.assertEqual(budget["early_stopping_rounds"], 75)
        self.assertEqual(budget["random_seed"], 1729)
        self.assertEqual(
            locked["source_protocol"]["sha256"],
            common.sha256_file(contract.QUANT_V1_PROTOCOL),
        )
        tampered = copy.deepcopy(protocol)
        tampered["model_tuning"]["shallow_xgboost"][
            "candidate_configs"
        ][0]["max_depth"] = 99
        with self.assertRaises(ValueError):
            run_xgboost.load_budget(tampered)

    def test_synthetic_preflight_selects_d2_tuple_as_columns(self) -> None:
        q_features = [f"q_feature_{index}" for index in range(56)]
        d_features = [f"d2_feature_{index}" for index in range(30)]
        rows = []
        for date, stock in (
            ("2025-01-02", "AAA"),
            ("2025-01-03", "BBB"),
        ):
            row = {
                "forecast_date": pd.Timestamp(date),
                "sector": "Sector",
                "stock": stock,
                "benchmark": "ETF",
                "wlite_observed_no_selected_article": 0,
            }
            row.update({name: 0.1 for name in q_features})
            row.update({name: 0.2 for name in d_features})
            rows.append(row)
        frame = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "panel.parquet"
            manifest = Path(directory) / "manifest.json"
            artifact.write_bytes(b"synthetic-panel")
            manifest.write_text("{}\n", encoding="utf-8")
            protocol = {
                "feature_blocks": {
                    "q56_by_target": {
                        target: q_features for target in contract.TARGETS
                    },
                    "d2_normalized_30": d_features,
                    "wlite_17": list(contract.WLITE_17),
                },
                "coverage": {
                    "row_count": 2,
                    "date_count": 2,
                    "stock_count": 2,
                    "sector_count": 1,
                },
                "semantic_gate": {
                    "observed_choice_order_agreement": (
                        lock_protocol.EXPECTED_AGREEMENT_COUNT
                        / lock_protocol.EXPECTED_SELECTED_ARTICLES
                    )
                },
                "source_artifacts": {
                    "canonical_joined_panel": {
                        "path": artifact.as_posix(),
                        "sha256": common.sha256_file(artifact),
                        "manifest_path": manifest.as_posix(),
                        "manifest_sha256": common.sha256_file(manifest),
                    }
                },
                "folds": [],
                "target_row_eligibility": {"expected_rows": {}},
            }
            with (
                mock.patch.object(
                    common, "_normalize_panel", return_value=frame
                ),
                mock.patch.object(
                    common.quant_common,
                    "validate_panel_information_set",
                    return_value=None,
                ),
                mock.patch.object(
                    common.quant_common,
                    "target_specs",
                    return_value=(),
                ),
            ):
                output, audit = common.load_panel_and_preflight(
                    protocol,
                    bundle_key="S0",
                    allow_failed_gate_exploratory=True,
                )
            self.assertEqual(len(output), 2)
            self.assertEqual(audit["status"], "passed")

    def test_s4_skip_is_hash_bound_and_verifiable(self) -> None:
        gate = [
            {"target": target, "eligible": False}
            for target in contract.TARGETS
        ]
        budget = {
            "candidate_configs": [{"max_depth": 2}] * 4,
            "candidate_configs_ordered_sha256": "a" * 64,
            "max_estimators": 2000,
            "early_stopping_rounds": 75,
            "random_seed": 1729,
        }
        s1_provenance = {"bundle": "S1", "artifact_sha256": "1" * 64}
        s3_provenance = {"bundle": "S3", "artifact_sha256": "3" * 64}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol_path = root / "protocol.json"
            protocol_path.write_text("{}\n", encoding="utf-8")
            bundle = SimpleNamespace(
                output_path=root / "output",
                experiment_path=root / "tracked",
            )
            protocol = {"implementation": {"runner.py": "a" * 64}}
            with (
                mock.patch.object(common, "bundle_for", return_value=bundle),
                mock.patch.object(
                    contract, "PROTOCOL_PATH", protocol_path
                ),
                mock.patch.object(common, "runtime_versions", return_value={}),
            ):
                written = run_xgboost.write_skipped_bundle(
                    protocol,
                    gate=gate,
                    s1_validation_provenance=s1_provenance,
                    s3_validation_provenance=s3_provenance,
                    budget=budget,
                )
                with (
                    mock.patch.object(
                        common,
                        "load_completed_predictions",
                        side_effect=[
                            (pd.DataFrame(), s1_provenance),
                            (pd.DataFrame(), s3_provenance),
                        ],
                    ),
                    mock.patch.object(
                        run_xgboost,
                        "validation_gate",
                        return_value=gate,
                    ),
                    mock.patch.object(
                        run_xgboost,
                        "load_budget",
                        return_value=budget,
                    ),
                ):
                    verified = run_xgboost.verify_s4_outcome(protocol)
                self.assertEqual(
                    written["manifest_sha256"],
                    verified["manifest_sha256"],
                )
                self.assertEqual(
                    verified["status"], "skipped_validation_gate"
                )
                self.assertFalse(
                    (bundle.output_path / "predictions.parquet").exists()
                )
                (bundle.output_path / "validation_gate.json").write_text(
                    "[]\n", encoding="utf-8"
                )
                with self.assertRaises(RuntimeError):
                    run_xgboost.verify_s4_outcome(protocol)

    def test_final_comparison_verifies_executed_s4_skip(self) -> None:
        gate = [
            {
                "target": target,
                "required_improved_validation_folds": 2,
                "improved_validation_folds": 0,
                "eligible": False,
                "folds": [
                    {
                        "fold": fold,
                        "rows": 2,
                        "s1_validation_mse": 0.1,
                        "s3_validation_mse": 0.2,
                        "s3_minus_s1_validation_mse": 0.1,
                        "s3_improved": False,
                    }
                    for fold in ("fold_1", "fold_2", "fold_3")
                ],
            }
            for target in contract.TARGETS
        ]
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text("{}\n", encoding="utf-8")
            outcome = {
                "status": "skipped_validation_gate",
                "eligible_targets": [],
                "reason": "no eligible targets",
                "manifest_path": manifest.as_posix(),
                "manifest_sha256": common.sha256_file(manifest),
                "validation_gate": gate,
            }
            with mock.patch.object(
                run_final_comparison.run_xgboost,
                "verify_s4_outcome",
                return_value=outcome,
            ):
                verified = run_final_comparison._verify_skipped_s4(
                    {}, gate
                )
            self.assertEqual(
                verified["status"], "skipped_validation_gate"
            )
            self.assertEqual(verified["recomputed_gate"], gate)
            self.assertEqual(
                verified["manifest_sha256"],
                common.sha256_file(manifest),
            )

    def test_predecessor_and_base_contract(self) -> None:
        self.assertEqual(run_linear.required_predecessors("S0"), ())
        self.assertEqual(
            run_linear.required_predecessors("S3"),
            ("S0", "S1", "S2"),
        )
        self.assertEqual(
            run_linear.required_predecessors("C-S2-L20"),
            ("S0", "S2"),
        )
        self.assertEqual(run_linear.primary_base("S2"), "S0")
        self.assertEqual(run_linear.primary_base("S3"), "S1")
        self.assertEqual(
            run_linear.primary_base("C-S3-PERM"), "S3"
        )

    def test_xgboost_gate_is_target_specific_two_of_three(self) -> None:
        rows = []
        for target in contract.TARGETS:
            for fold_index, fold in enumerate(
                ("fold_1", "fold_2", "fold_3")
            ):
                for stock in ("A", "B"):
                    actual = 1.0
                    s1_prediction = 0.0
                    improved = (
                        fold_index < 2 and target == "t1_etf"
                    )
                    s3_prediction = 0.5 if improved else -0.5
                    base = {
                        "fold": fold,
                        "target": target,
                        "forecast_date": pd.Timestamp(
                            "2025-01-02"
                        )
                        + pd.Timedelta(days=fold_index),
                        "sector": "Sector",
                        "stock": stock,
                        "benchmark": "ETF",
                        "actual_fisher_z": actual,
                    }
                    rows.append(
                        (
                            {**base, "predicted_fisher_z": s1_prediction},
                            {**base, "predicted_fisher_z": s3_prediction},
                        )
                    )
        s1 = pd.DataFrame([left for left, _ in rows])
        s3 = pd.DataFrame([right for _, right in rows])
        protocol = {
            "targets": list(contract.TARGETS),
            "folds": [
                {"name": name}
                for name in ("fold_1", "fold_2", "fold_3")
            ],
        }
        with mock.patch.object(
            common, "validate_prediction_panel", return_value={}
        ):
            gate = run_xgboost.validation_gate(s3, s1, protocol)
        by_target = {record["target"]: record for record in gate}
        self.assertTrue(by_target["t1_etf"]["eligible"])
        self.assertFalse(by_target["t1_loo"]["eligible"])


if __name__ == "__main__":
    unittest.main()
