from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIRECTORY = REPOSITORY_ROOT / "scripts"
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

import build_v2_deterministic_news_features as d2  # noqa: E402


class D2FeatureBuilderTests(unittest.TestCase):
    def test_feature_contract_counts_and_order_are_stable(self) -> None:
        self.assertEqual(len(d2.D2_NORMALIZED_30), 30)
        self.assertEqual(len(set(d2.D2_NORMALIZED_30)), 30)
        self.assertEqual(len(d2.D2_LEVELS_5), 5)
        self.assertEqual(len(set(d2.D2_LEVELS_5)), 5)
        self.assertEqual(
            d2.D2_NORMALIZED_30[0],
            "d2_target_direct_intensity_midrank_126",
        )
        self.assertEqual(
            d2.D2_NORMALIZED_30[-1],
            "d2_common_negative_surprise_article_share",
        )

    def test_strict_cutoff_maps_exact_timestamp_to_next_session(self) -> None:
        cutoffs = [
            datetime(2024, 1, 2, 14, 0, tzinfo=timezone.utc),
            datetime(2024, 1, 3, 14, 0, tzinfo=timezone.utc),
        ]
        just_before = datetime(2024, 1, 2, 13, 59, 59, tzinfo=timezone.utc)
        exact = datetime(2024, 1, 2, 14, 0, 0, tzinfo=timezone.utc)
        self.assertEqual(
            d2.assign_forecast_session_index(just_before, cutoffs), 0
        )
        self.assertEqual(d2.assign_forecast_session_index(exact, cutoffs), 1)

    def test_roles_are_disjoint_and_exactly_one_peer_is_peer_idio(self) -> None:
        direct = d2.role_flags(
            target_ticker="AAA",
            sector_entities={"AAA"},
            sector_common=False,
            macro_common=False,
        )
        self.assertTrue(direct["target_idio"])
        self.assertFalse(direct["peer_idio"])
        self.assertFalse(direct["any_common"])

        peer = d2.role_flags(
            target_ticker="AAA",
            sector_entities={"BBB"},
            sector_common=False,
            macro_common=False,
        )
        self.assertTrue(peer["peer_idio"])
        self.assertFalse(peer["direct"])

        two_entities = d2.role_flags(
            target_ticker="AAA",
            sector_entities={"BBB", "CCC"},
            sector_common=True,
            macro_common=False,
        )
        self.assertTrue(two_entities["any_common"])
        self.assertFalse(two_entities["peer_idio"])

        target_common = d2.role_flags(
            target_ticker="AAA",
            sector_entities={"AAA", "BBB"},
            sector_common=True,
            macro_common=False,
        )
        self.assertTrue(target_common["target_common"])
        self.assertFalse(target_common["target_idio"])

    def test_spy_tag_alone_does_not_create_macro_evidence(self) -> None:
        rules = {"macro_provider_keywords": ["federal reserve", "inflation"]}
        self.assertFalse(
            d2.macro_evidence(
                "SPY rose before the opening bell.",
                set(),
                rules,
            )
        )
        self.assertTrue(
            d2.macro_evidence(
                "The Federal Reserve changed monetary policy.",
                set(),
                rules,
            )
        )

    def test_midrank_uses_only_prior_observations_and_midrank_ties(self) -> None:
        values = np.array([0.0, 1.0, 1.0, 2.0])
        result = d2.prior_midrank(values, window=3)
        self.assertTrue(np.isnan(result[:3]).all())
        self.assertAlmostEqual(result[3], 1.0)

        ties = d2.prior_midrank(np.array([1.0, 1.0, 1.0, 1.0]), window=3)
        self.assertAlmostEqual(ties[3], 0.5)

    def test_attention_delta_uses_prior_median_not_current(self) -> None:
        result = d2.prior_median_delta(
            np.array([0.0, 0.5, 1.0, 0.75]), window=3
        )
        self.assertTrue(np.isnan(result[:3]).all())
        self.assertAlmostEqual(result[3], 0.25)

    def test_short_ticker_requires_cashtag_or_provider_metadata(self) -> None:
        self.assertFalse(d2.ticker_text_present("C reported a change.", "C"))
        self.assertTrue(d2.ticker_text_present("$C reported a change.", "C"))
        self.assertTrue(d2.ticker_text_present("AMD reported a change.", "AMD"))

    def test_lexical_family_cues_do_not_need_provider_topics(self) -> None:
        cues = d2.lexical_cues(
            "The company raised guidance after revenue beat estimates."
        )
        self.assertTrue(cues["earnings_guidance"])
        self.assertTrue(cues["positive_surprise"])
        self.assertFalse(cues["negative_surprise"])


if __name__ == "__main__":
    unittest.main()
