from __future__ import annotations

import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from scripts import join_quant_d2_w17_lite_v3 as joiner


class JoinQuantD2W17LiteV3Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.design_path = self.root / "design.json"
        self.base_path = self.root / "base.parquet"
        self.base_manifest_path = self.root / "base.manifest.json"
        self.daily_path = self.root / "daily_features.parquet"
        self.daily_manifest_path = self.root / "daily_features.manifest.json"

        self.features = list(joiner.BOUNDED_FEATURES)
        self.audits = list(joiner.AUDIT_ONLY_COLUMNS)
        self.design_path.write_text(
            json.dumps(
                {
                    "flan_w17_lite": {
                        "arm_id": joiner.EXPECTED_ARM,
                        "contract_id": joiner.EXPECTED_CONTRACT,
                        "target_feature_count": 17,
                        "ordered_features": self.features,
                        "audit_only_nonpredictors": self.audits,
                    }
                }
            ),
            encoding="utf-8",
        )
        self.base = pd.DataFrame(
            {
                "forecast_date": pd.to_datetime(
                    ["2022-11-01", "2022-11-02"]
                ),
                "sector": ["semiconductors", "semiconductors"],
                "stock": ["AMD", "AMD"],
                "benchmark": ["SOXX", "SOXX"],
                "q": [1.25, 2.5],
                "d2": [0.1, 0.2],
            }
        )
        self.base.to_parquet(self.base_path, index=False)
        q56 = {
            target: [f"q_{target}_{index}" for index in range(56)]
            for target in ("t1_etf", "t1_loo", "t2_etf", "t2_loo")
        }
        self.base_manifest = {
            "manifest_version": joiner.EXPECTED_BASE_MANIFEST_VERSION,
            "status": joiner.EXPECTED_BASE_STATUS,
            "output": {
                "path": str(self.base_path),
                "sha256": joiner.sha256_file(self.base_path),
                "row_count": 2,
            },
            "counts": {"matched_rows": 2, "match_fraction": 1.0},
            "claim_scope": {
                "point_in_time_version_safe": False,
                "primary_training_eligible": False,
                "confirmatory_eligible": False,
                "exploratory_only": True,
            },
            "ordered_feature_lists": {
                "d2_normalized_30": [f"d2_{index}" for index in range(30)],
                "q56_by_target": q56,
            },
        }
        self.base_manifest_path.write_text(
            json.dumps(self.base_manifest), encoding="utf-8"
        )
        self.base_manifest_path.with_suffix(".sha256").write_text(
            joiner.sha256_file(self.base_manifest_path) + "\n",
            encoding="ascii",
        )
        self.daily = self._daily_frame()
        self.daily.to_parquet(self.daily_path, index=False)
        self.daily_inputs: dict[str, dict[str, str]] = {}
        for name in sorted(joiner.REQUIRED_DAILY_INPUTS):
            if name == "design":
                path = self.design_path
            else:
                path = self.root / f"{name}.txt"
                path.write_text(name, encoding="utf-8")
            self.daily_inputs[name] = {
                "path": str(path),
                "sha256": joiner.sha256_file(path),
            }
        self.daily_manifest = self._daily_manifest()
        self._write_daily_manifest()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _daily_frame(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        selected: dict[str, object] = {
            "forecast_date": pd.Timestamp("2022-11-01"),
            "sector": "semiconductors",
            "stock": "AMD",
            "benchmark": "SOXX",
            "wlite_event_share_firm_operating_financial": 0.3,
            "wlite_event_share_policy_corporate": 0.1,
            "wlite_event_share_macro_market": 0.2,
            "wlite_route_share_target_idiosyncratic": 0.2,
            "wlite_route_share_peer_idiosyncratic": 0.3,
            "wlite_rule_status_cue_share_confirmed_action": 0.1,
            "wlite_rule_status_cue_share_scheduled_expected": 0.2,
            "wlite_rule_status_cue_share_rumor_unconfirmed": 0.1,
            "wlite_rule_status_cue_share_analysis_opinion": 0.0,
            "wlite_rule_status_cue_conflict_weight_share": 0.1,
            "wlite_selection_weight_coverage": 0.9,
            "wlite_selected_headline_only_weight_share": 0.2,
            "wlite_selected_sub150_weight_share": 0.3,
            "wlite_event_order_disagreement_weight_share": 0.3,
            "wlite_observed_no_selected_article": 0,
            "wlite_event_entropy_accepted": 0.8,
            "wlite_selected_weight_hhi": 0.2,
            "wlite_route_share_common": 0.5,
            "wlite_event_accepted_coverage": 0.6,
            "wlite_event_other_or_unclear_weight_share": 0.1,
            "wlite_rule_status_cue_all_zero_weight_share": 0.6,
            "wlite_invalid_or_truncated_selected_article_count": 0,
        }
        selected.update(self._audit_flags())
        rows.append(selected)

        no_selected: dict[str, object] = {
            "forecast_date": pd.Timestamp("2022-11-02"),
            "sector": "semiconductors",
            "stock": "AMD",
            "benchmark": "SOXX",
            **{name: np.nan for name in self.features},
            **{name: np.nan for name in self.audits},
        }
        no_selected["wlite_observed_no_selected_article"] = 1
        no_selected[
            "wlite_invalid_or_truncated_selected_article_count"
        ] = 0
        no_selected.update(self._audit_flags())
        rows.append(no_selected)
        return pd.DataFrame(rows)

    @staticmethod
    def _audit_flags() -> dict[str, object]:
        return {
            "source_profile": joiner.EXPECTED_SOURCE_PROFILE,
            "point_in_time_version_safe": False,
            "primary_training_eligible": False,
            "confirmatory_eligible": False,
            "exploratory_construction_eligible": True,
            "semantic_quality_gate_passed": False,
            "exploratory_quality_gate_override": True,
        }

    def _daily_manifest(self) -> dict[str, object]:
        return {
            "manifest_version": joiner.EXPECTED_DAILY_MANIFEST_VERSION,
            "status": joiner.EXPECTED_DAILY_STATUS,
            "arm_id": joiner.EXPECTED_ARM,
            "contract_id": joiner.EXPECTED_CONTRACT,
            "source_profile": joiner.EXPECTED_SOURCE_PROFILE,
            "variant_id": "canonical",
            "ordered_feature_lists": {"wlite_17": self.features},
            "generated_files": {
                self.daily_path.name: {
                    "path": str(self.daily_path),
                    "rows": 2,
                    "sha256": joiner.sha256_file(self.daily_path),
                }
            },
            "claim_flags": {
                "point_in_time_version_safe": False,
                "primary_training_eligible": False,
                "confirmatory_eligible": False,
                "exploratory_construction_eligible": True,
                "semantic_quality_gate_passed": False,
                "exploratory_quality_gate_override": True,
            },
            "inputs": self.daily_inputs,
        }

    def _write_daily_manifest(self) -> None:
        self.daily_manifest_path.write_text(
            json.dumps(self.daily_manifest), encoding="utf-8"
        )
        self.daily_manifest_path.with_suffix(".sha256").write_text(
            joiner.sha256_file(self.daily_manifest_path) + "\n",
            encoding="ascii",
        )

    def _universe_patch(self):
        return mock.patch.multiple(
            joiner,
            EXPECTED_DATES=2,
            EXPECTED_STOCKS=1,
            EXPECTED_SECTORS=1,
            EXPECTED_FIRST_DATE="2022-11-01",
            EXPECTED_LAST_DATE="2022-11-02",
            EXPECTED_CANONICAL_NO_SELECTED=1,
        )

    def test_loads_exact_design_and_source_contracts(self) -> None:
        _, features = joiner.load_wlite_contract(self.design_path)
        self.assertEqual(features, self.features)
        base = joiner.validate_base_contract(
            base_path=self.base_path,
            base_manifest_path=self.base_manifest_path,
            expected_rows=2,
        )
        self.assertEqual(base["status"], joiner.EXPECTED_BASE_STATUS)
        daily = joiner.validate_daily_contract(
            daily_path=self.daily_path,
            manifest_path=self.daily_manifest_path,
            design_path=self.design_path,
            features=features,
            expected_rows=2,
            variant_id="canonical",
        )
        self.assertEqual(daily["arm_id"], joiner.EXPECTED_ARM)

    def test_daily_hash_mismatch_fails_closed(self) -> None:
        self.daily_manifest["generated_files"][self.daily_path.name][
            "sha256"
        ] = "0" * 64
        self._write_daily_manifest()
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            joiner.validate_daily_contract(
                daily_path=self.daily_path,
                manifest_path=self.daily_manifest_path,
                design_path=self.design_path,
                features=self.features,
                expected_rows=2,
                variant_id="canonical",
            )

    def test_validates_missingness_and_preserves_base_exactly(self) -> None:
        with self._universe_patch():
            daily, stats, _ = joiner.validate_daily_frame(
                self.daily,
                features=self.features,
                expected_rows=2,
                variant_id="canonical",
            )
            joined, joined_stats = joiner.build_joined_panel(
                self.base,
                daily,
                features=self.features,
                expected_rows=2,
                variant_id="canonical",
            )
        self.assertEqual(stats["no_selected_rows"], 1)
        self.assertEqual(joined_stats["match_fraction"], 1.0)
        pd.testing.assert_frame_equal(
            joined[list(self.base.columns)],
            self.base,
            check_dtype=True,
        )
        self.assertTrue(joined["wlite_row_matched"].eq(1).all())

    def test_selected_row_with_missing_feature_fails_closed(self) -> None:
        bad = self.daily.copy()
        bad.loc[
            0, "wlite_event_share_firm_operating_financial"
        ] = np.nan
        with self._universe_patch(), self.assertRaisesRegex(
            ValueError, "selected rows have missing"
        ):
            joiner.validate_daily_frame(
                bad,
                features=self.features,
                expected_rows=2,
                variant_id="canonical",
            )

    def test_no_selected_row_with_nonmissing_share_fails_closed(self) -> None:
        bad = self.daily.copy()
        bad.loc[
            1, "wlite_event_share_firm_operating_financial"
        ] = 0.0
        with self._universe_patch(), self.assertRaisesRegex(
            ValueError, "no-selected rows must have missing"
        ):
            joiner.validate_daily_frame(
                bad,
                features=self.features,
                expected_rows=2,
                variant_id="canonical",
            )

    def test_event_decomposition_must_sum_to_one(self) -> None:
        bad = self.daily.copy()
        bad.loc[
            0, "wlite_event_other_or_unclear_weight_share"
        ] = 0.2
        with self._universe_patch(), self.assertRaisesRegex(
            ValueError, "event decomposition"
        ):
            joiner.validate_daily_frame(
                bad,
                features=self.features,
                expected_rows=2,
                variant_id="canonical",
            )

    def test_key_set_mismatch_fails_closed(self) -> None:
        bad = self.daily.copy()
        bad.loc[1, "stock"] = "NVDA"
        with self._universe_patch(), self.assertRaisesRegex(
            ValueError, "universe differs|keys differ"
        ):
            joiner.build_joined_panel(
                self.base,
                bad,
                features=self.features,
                expected_rows=2,
                variant_id="canonical",
            )

    def test_variant_arguments_require_paired_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "names must match"):
            joiner.discover_variants(
                variant_values=["permuted=x.parquet"],
                manifest_values=[],
                include_defaults=False,
            )
        result = joiner.discover_variants(
            variant_values=["permuted=x.parquet"],
            manifest_values=["permuted=x.manifest.json"],
            include_defaults=False,
        )
        self.assertEqual(
            result["permuted"],
            (Path("x.parquet"), Path("x.manifest.json")),
        )

    def test_main_writes_hash_bound_join_without_changing_base(self) -> None:
        output_root = self.root / "output"
        with self._universe_patch(), redirect_stdout(io.StringIO()):
            result = joiner.main(
                [
                    "--base",
                    str(self.base_path),
                    "--base-manifest",
                    str(self.base_manifest_path),
                    "--design",
                    str(self.design_path),
                    "--daily",
                    str(self.daily_path),
                    "--daily-manifest",
                    str(self.daily_manifest_path),
                    "--output-root",
                    str(output_root),
                    "--expected-row-count",
                    "2",
                    "--no-default-variants",
                ]
            )
        self.assertEqual(result, 0)
        output = output_root / joiner.DEFAULT_OUTPUT_NAME
        manifest_path = output_root / joiner.MANIFEST_NAME
        self.assertTrue(output.is_file())
        self.assertTrue(manifest_path.is_file())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        record = manifest["generated_files"][output.name]
        self.assertEqual(record["sha256"], joiner.sha256_file(output))
        joined = pd.read_parquet(output)
        pd.testing.assert_frame_equal(
            joined[list(self.base.columns)],
            self.base,
            check_dtype=True,
        )
        self.assertEqual(manifest["ordered_feature_lists"]["wlite_17"], self.features)


if __name__ == "__main__":
    unittest.main()
