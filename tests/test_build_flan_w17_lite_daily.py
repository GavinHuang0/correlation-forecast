from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.semantic_news_v3 import build_w17_lite_daily as daily


def config() -> dict:
    return json.loads(daily.DEFAULT_AGGREGATION.read_text(encoding="utf-8"))


def scope(*, date: str = "2025-01-02") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "forecast_date": date,
                "sector": "Test",
                "stock": "AAA",
                "benchmark": "TEST",
            }
        ]
    )


def assignments(*, date: str = "2025-01-02") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "assignment_id": "a",
                "article_id": "article-a",
                "forecast_date": date,
                "sector": "Test",
                "stock": "AAA",
                "benchmark": "TEST",
                "role": "I",
                "aggregation_weight": 1.0,
                "event_label": "firm_operating_financial",
                "event_order_disagreement": False,
                "event_other_or_unclear": False,
                "cue_confirmed_action": True,
                "cue_scheduled_expected": False,
                "cue_rumor_unconfirmed": False,
                "cue_analysis_opinion": False,
                "cue_conflict": False,
                "cue_all_zero": False,
                "headline_only": True,
                "description_sub150": False,
                "description_ge150": False,
            },
            {
                "assignment_id": "b",
                "article_id": "article-b",
                "forecast_date": date,
                "sector": "Test",
                "stock": "AAA",
                "benchmark": "TEST",
                "role": "C",
                "aggregation_weight": 3.0,
                "event_label": None,
                "event_order_disagreement": True,
                "event_other_or_unclear": False,
                "cue_confirmed_action": False,
                "cue_scheduled_expected": True,
                "cue_rumor_unconfirmed": True,
                "cue_analysis_opinion": False,
                "cue_conflict": True,
                "cue_all_zero": False,
                "headline_only": False,
                "description_sub150": True,
                "description_ge150": False,
            },
        ]
    )


class BuildW17LiteDailyTests(unittest.TestCase):
    def test_checked_in_contract_has_exact_17_features(self) -> None:
        value = config()
        daily.validate_aggregation_config(value)
        self.assertEqual(tuple(value["ordered_features"]), daily.FEATURE_COLUMNS)
        self.assertEqual(len(daily.FEATURE_COLUMNS), 17)
        self.assertEqual(len(daily.AUDIT_COLUMNS), 10)

    def test_status_cues_are_multihot_and_modal_alone_does_not_fire(self) -> None:
        value = config()
        flags = daily.status_cues(
            "The company announced that it will report earnings next week.",
            value,
        )
        self.assertTrue(flags["confirmed_action"])
        self.assertTrue(flags["scheduled_expected"])
        self.assertFalse(flags["rumor_unconfirmed"])
        self.assertFalse(flags["analysis_opinion"])
        self.assertFalse(any(daily.status_cues("The deal may occur.", value).values()))
        self.assertTrue(
            daily.status_cues(
                "People familiar with the matter said the company is "
                "considering a sale.",
                value,
            )["rumor_unconfirmed"]
        )
        self.assertTrue(
            daily.status_cues(
                "The analyst upgraded the stock and raised its price target.",
                value,
            )["analysis_opinion"]
        )

    def test_weighted_aggregation_and_event_identity(self) -> None:
        selected = assignments()
        source = selected.drop(
            columns=[
                column
                for column in selected
                if column.startswith("event_")
                or column.startswith("cue_")
                or column in {
                    "headline_only",
                    "description_sub150",
                    "description_ge150",
                }
            ]
        )
        frame = daily.aggregate_variant(
            source=source,
            selected=selected,
            scopes=scope(),
            variant_id="canonical",
            enforce_expected_profile=False,
        )
        row = frame.iloc[0]
        self.assertAlmostEqual(
            row["wlite_event_share_firm_operating_financial"], 0.25
        )
        self.assertAlmostEqual(
            row["wlite_event_order_disagreement_weight_share"], 0.75
        )
        self.assertAlmostEqual(
            row["wlite_route_share_target_idiosyncratic"], 0.25
        )
        self.assertAlmostEqual(row["wlite_route_share_common"], 0.75)
        self.assertAlmostEqual(
            row["wlite_rule_status_cue_conflict_weight_share"], 0.75
        )
        self.assertAlmostEqual(row["wlite_selected_weight_hhi"], 0.625)
        self.assertAlmostEqual(
            row["wlite_event_decomposition_residual"], 0.0
        )
        self.assertEqual(tuple(frame.columns[-17:]), daily.FEATURE_COLUMNS)

    def test_no_candidate_row_is_missing_not_semantic_zero(self) -> None:
        empty = pd.DataFrame(
            columns=[
                "assignment_id",
                "article_id",
                "forecast_date",
                "sector",
                "stock",
                "benchmark",
                "role",
                "aggregation_weight",
            ]
        )
        selected = assignments().iloc[0:0]
        frame = daily.aggregate_variant(
            source=empty,
            selected=selected,
            scopes=scope(),
            variant_id="canonical",
            enforce_expected_profile=False,
        )
        row = frame.iloc[0]
        self.assertEqual(row["wlite_observed_no_selected_article"], 1.0)
        self.assertTrue(
            np.isnan(row["wlite_event_share_firm_operating_financial"])
        )
        self.assertTrue(np.isnan(row["wlite_selection_weight_coverage"]))

    def test_long_description_uses_full_candidate_denominator(self) -> None:
        selected = assignments()
        selected.loc[0, "description_ge150"] = True
        source = selected[
            [
                "assignment_id",
                "article_id",
                "forecast_date",
                "sector",
                "stock",
                "benchmark",
                "role",
                "aggregation_weight",
            ]
        ]
        frame = daily.aggregate_variant(
            source=source,
            selected=selected.loc[selected["description_ge150"]],
            scopes=scope(),
            variant_id="long_description",
            enforce_expected_profile=False,
        )
        self.assertAlmostEqual(
            frame.iloc[0]["wlite_selection_weight_coverage"], 0.25
        )

    def test_date_sector_permutation_is_deterministic_and_label_preserving(
        self,
    ) -> None:
        selected = assignments()
        third = selected.iloc[[0]].copy()
        third["assignment_id"] = "c"
        third["article_id"] = "article-c"
        third["event_label"] = "macro_market"
        selected = pd.concat([selected, third], ignore_index=True)
        first, first_hash = daily.apply_event_permutation(
            selected, seed="fixed"
        )
        second, second_hash = daily.apply_event_permutation(
            selected.sample(frac=1, random_state=8), seed="fixed"
        )
        self.assertEqual(first_hash, second_hash)
        observed = first.set_index("article_id")
        self.assertEqual(observed.loc["article-a", "event_label"], "macro_market")
        self.assertTrue(pd.isna(observed.loc["article-b", "event_label"]))
        self.assertEqual(
            observed.loc["article-c", "event_label"],
            "firm_operating_financial",
        )
        self.assertTrue(
            observed.loc["article-b", "event_order_disagreement"]
        )
        self.assertFalse(
            observed.loc["article-a", "event_order_disagreement"]
        )
        self.assertCountEqual(
            first["event_label"].fillna("none"),
            selected["event_label"].fillna("none"),
        )

    def test_selection_coverage_is_missing_when_variant_selects_nothing(
        self,
    ) -> None:
        selected = assignments().iloc[0:0]
        source = assignments()[
            [
                "assignment_id",
                "article_id",
                "forecast_date",
                "sector",
                "stock",
                "benchmark",
                "role",
                "aggregation_weight",
            ]
        ]
        frame = daily.aggregate_variant(
            source=source,
            selected=selected,
            scopes=scope(),
            variant_id="long_description",
            enforce_expected_profile=False,
        )
        row = frame.iloc[0]
        self.assertEqual(row["wlite_observed_no_selected_article"], 1.0)
        self.assertTrue(np.isnan(row["wlite_selection_weight_coverage"]))

    def test_config_mutation_is_rejected(self) -> None:
        value = copy.deepcopy(config())
        value["ordered_features"] = list(reversed(value["ordered_features"]))
        with self.assertRaisesRegex(ValueError, "ordered feature"):
            daily.validate_aggregation_config(value)

    def test_atomic_parquet_and_json_leave_complete_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame = pd.DataFrame({"a": [1, 2]})
            parquet = root / "value.parquet"
            manifest = root / "manifest.json"
            daily._atomic_parquet(parquet, frame)
            daily._atomic_json(manifest, {"status": "complete"})
            pd.testing.assert_frame_equal(pd.read_parquet(parquet), frame)
            self.assertEqual(
                json.loads(manifest.read_text(encoding="utf-8"))["status"],
                "complete",
            )


if __name__ == "__main__":
    unittest.main()
