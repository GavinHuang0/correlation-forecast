from __future__ import annotations

import json
import math
import unittest

import numpy as np
import pandas as pd

from scripts.semantic_news_v4 import build_soft_route_features as soft


def _scores(**overrides: float) -> dict[str, float]:
    values = {
        "firm_operating_financial": -1.0,
        "policy_corporate": -2.0,
        "macro_market": -3.0,
        "other_or_unclear": -4.0,
    }
    values.update(overrides)
    return values


def _toy_inputs(days: int = 84) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2025-01-02", periods=days).strftime("%Y-%m-%d")
    scope_rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    for stock, role, firm, macro in (
        ("AAA", "I", 0.6, 0.1),
        ("BBB", "C", 0.2, 0.5),
    ):
        for index, date in enumerate(dates):
            scope_rows.append(
                {
                    "forecast_date": date,
                    "sector": "Test",
                    "stock": stock,
                    "benchmark": "TEST",
                }
            )
            # One deliberately complete zero-news row.
            if stock == "AAA" and index == 10:
                continue
            assignment = {
                "assignment_id": f"{stock}-{index}",
                "article_id": f"article-{stock}-{index}",
                "forecast_date": date,
                "sector": "Test",
                "stock": stock,
                "benchmark": "TEST",
                "role": role,
                "aggregation_weight": 2.0,
            }
            source_rows.append(dict(assignment))
            selected_rows.append(
                {
                    **assignment,
                    "consensus_probability_firm_operating_financial": firm,
                    "consensus_probability_policy_corporate": 0.1,
                    "consensus_probability_macro_market": macro,
                    "consensus_probability_other_or_unclear": 1.0
                    - firm
                    - macro
                    - 0.1,
                    "score_order_js_divergence_normalized": 0.2,
                    "score_consensus_entropy_normalized": 0.7,
                    "probability_sum_residual": 0.0,
                }
            )
    return (
        pd.DataFrame(scope_rows),
        pd.DataFrame(source_rows),
        pd.DataFrame(selected_rows),
    )


class SoftRouteV4Tests(unittest.TestCase):
    def test_checked_in_contract_matches_exact_feature_lists(self) -> None:
        value = soft.validate_contract(soft.DEFAULT_CONTRACT)
        self.assertEqual(value["soft_route_19"], list(soft.SOFT_ROUTE19))
        self.assertEqual(len(soft.CURRENT9), 9)
        self.assertEqual(len(soft.INNOVATION9), 9)
        self.assertEqual(len(soft.SOFT_ROUTE19), 19)
        self.assertEqual(len(soft.COUPLING6), 6)
        self.assertEqual(len(soft.QUALITY4), 4)
        self.assertIn("permuted", soft.VARIANT_FILES)
        self.assertIn("permuted", soft.JOINED_FILES)

    def test_consensus_is_arithmetic_mean_of_within_order_softmaxes(self) -> None:
        canonical = _scores(firm_operating_financial=0.0)
        reversed_order = _scores(macro_market=0.5)
        result = soft.consensus_probabilities(canonical, reversed_order)
        expected = (
            soft.softmax_scores(canonical) + soft.softmax_scores(reversed_order)
        ) / 2.0
        observed = np.array(
            [result[f"consensus_probability_{name}"] for name in soft.ALL_LABELS]
        )
        np.testing.assert_allclose(observed, expected, atol=1e-15, rtol=0)
        swapped = soft.consensus_probabilities(reversed_order, canonical)
        np.testing.assert_allclose(
            observed,
            [swapped[f"consensus_probability_{name}"] for name in soft.ALL_LABELS],
            atol=1e-15,
            rtol=0,
        )
        self.assertAlmostEqual(observed.sum(), 1.0)
        self.assertGreaterEqual(result["score_order_js_divergence_normalized"], 0)
        self.assertLessEqual(result["score_order_js_divergence_normalized"], 1)

    def test_prior_ewma_excludes_current_and_requires_exactly_63_priors(self) -> None:
        values = np.arange(70, dtype=float)
        observed = soft.finite_prior_ewma(values, window=63, half_life=21)
        self.assertTrue(np.isnan(observed[:63]).all())
        ages = np.arange(62, -1, -1, dtype=float)
        weights = np.exp(-math.log(2.0) * ages / 21.0)
        expected = np.dot(values[:63], weights / weights.sum())
        self.assertAlmostEqual(observed[63], expected)
        self.assertLess(observed[63], values[63])

    def test_daily_joint_mass_zero_news_innovation_and_decomposition(self) -> None:
        scopes, source, selected = _toy_inputs()
        frame = soft.aggregate_daily(
            scopes=scopes,
            source=source,
            selected=selected,
            enforce_expected_profile=False,
        )
        zero_date = scopes.loc[scopes["stock"] == "AAA"].iloc[10][
            "forecast_date"
        ]
        zero = frame.loc[
            (frame["stock"] == "AAA")
            & (frame["forecast_date"] == zero_date)
        ].iloc[0]
        self.assertEqual(zero[soft.NO_SELECTED], 1.0)
        self.assertTrue(zero[list(soft.CURRENT9)].eq(0.0).all())
        self.assertEqual(zero[soft.QUALITY4[2]], 0.0)
        self.assertTrue(np.isnan(zero[soft.QUALITY4[0]]))
        self.assertLessEqual(
            frame[list(soft.CURRENT9)].sum(axis=1).max(), 1.0 + 1e-14
        )
        self.assertTrue(
            np.allclose(
                frame["lsoft_daily_mass_decomposition_residual"],
                0.0,
                atol=2e-14,
            )
        )
        aaa = frame.loc[frame["stock"] == "AAA"].sort_values("forecast_date")
        self.assertTrue(aaa[list(soft.INNOVATION9)].iloc[:63].isna().all().all())
        self.assertTrue(aaa[list(soft.INNOVATION9)].iloc[63:].notna().all().all())

    def test_stale_and_wrong_stock_controls_change_only_declared_fields(self) -> None:
        scopes, source, selected = _toy_inputs()
        canonical = soft.aggregate_daily(
            scopes=scopes,
            source=source,
            selected=selected,
            enforce_expected_profile=False,
        )
        stale = soft.make_stale20(canonical, enforce_expected_profile=False)
        c_aaa = canonical.loc[canonical["stock"] == "AAA"].sort_values("forecast_date")
        s_aaa = stale.loc[stale["stock"] == "AAA"].sort_values("forecast_date")
        pd.testing.assert_frame_equal(
            s_aaa[list(soft.CURRENT9)].reset_index(drop=True),
            c_aaa[list(soft.CURRENT9)].shift(20).reset_index(drop=True),
        )
        self.assertTrue(
            s_aaa[soft.NO_SELECTED].reset_index(drop=True).equals(
                c_aaa[soft.NO_SELECTED].reset_index(drop=True)
            )
        )
        self.assertTrue(
            stale["lsoft_daily_mass_decomposition_residual"].isna().all()
        )
        wrong, mapping_hash = soft.make_wrong_stock(
            canonical, enforce_expected_profile=False
        )
        self.assertRegex(mapping_hash, r"^[0-9a-f]{64}$")
        date = canonical["forecast_date"].iloc[-1]
        recipient = wrong.loc[
            (wrong["stock"] == "AAA") & (wrong["forecast_date"] == date)
        ].iloc[0]
        donor = canonical.loc[
            (canonical["stock"] == "BBB") & (canonical["forecast_date"] == date)
        ].iloc[0]
        original = canonical.loc[
            (canonical["stock"] == "AAA") & (canonical["forecast_date"] == date)
        ].iloc[0]
        self.assertEqual(
            recipient[soft._mass_name("I", "firm_operating_financial")],
            donor[soft._mass_name("I", "firm_operating_financial")],
        )
        self.assertEqual(
            recipient[soft._mass_name("C", "macro_market")],
            original[soft._mass_name("C", "macro_market")],
        )
        self.assertEqual(recipient[soft.QUALITY4[0]], original[soft.QUALITY4[0]])
        self.assertTrue(
            wrong["lsoft_daily_mass_decomposition_residual"].isna().all()
        )

    def test_probability_permutation_moves_whole_vector_and_preserves_js(self) -> None:
        _, _, selected = _toy_inputs(days=3)
        date = selected["forecast_date"].min()
        sample = selected.loc[selected["forecast_date"] == date].copy()
        probability_columns = [
            f"consensus_probability_{label}" for label in soft.ALL_LABELS
        ]
        first, first_hash = soft.apply_probability_permutation(sample)
        second, second_hash = soft.apply_probability_permutation(
            sample.sample(frac=1.0, random_state=7)
        )
        self.assertEqual(first_hash, second_hash)
        pd.testing.assert_frame_equal(first, second)
        original_vectors = sorted(
            map(tuple, sample[probability_columns].to_numpy(dtype=float))
        )
        permuted_vectors = sorted(
            map(tuple, first[probability_columns].to_numpy(dtype=float))
        )
        self.assertEqual(original_vectors, permuted_vectors)
        original = sample.set_index("assignment_id")
        observed = first.set_index("assignment_id")
        pd.testing.assert_series_equal(
            observed["score_order_js_divergence_normalized"].sort_index(),
            original["score_order_js_divergence_normalized"].sort_index(),
        )
        # The two-article pool is a derangement, not a component-wise shuffle.
        for assignment_id in original.index:
            self.assertFalse(
                np.array_equal(
                    original.loc[assignment_id, probability_columns].to_numpy(
                        dtype=float
                    ),
                    observed.loc[assignment_id, probability_columns].to_numpy(
                        dtype=float
                    ),
                )
            )

    def test_join_preserves_base_exactly(self) -> None:
        scopes, source, selected = _toy_inputs(days=3)
        daily = soft.aggregate_daily(
            scopes=scopes,
            source=source,
            selected=selected,
            enforce_expected_profile=False,
        )
        base = daily[list(soft.KEY_COLUMNS)].copy()
        base["forecast_date"] = pd.to_datetime(base["forecast_date"])
        base["q"] = np.arange(len(base), dtype=float)
        joined = soft.build_joined_panel(
            base, daily, enforce_expected_profile=False
        )
        pd.testing.assert_frame_equal(joined[list(base.columns)], base)
        self.assertTrue(joined["lsoft_row_matched"].eq(1).all())


if __name__ == "__main__":
    unittest.main()
