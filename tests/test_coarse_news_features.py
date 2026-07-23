from __future__ import annotations

import importlib.util
import json
import unittest
from collections import Counter
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "coarse_news_features.py"
SPEC = importlib.util.spec_from_file_location("coarse_news_features", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fine_labels(**overrides: str) -> dict[str, str]:
    labels = {
        "relevance": "direct_target",
        "event_scope": "firm_specific",
        "event_type": "guidance",
        "target_direction": "positive",
        "sector_direction": "not_applicable",
        "peer_effect": "not_applicable",
        "information_status": "confirmed",
    }
    labels.update(overrides)
    return labels


def input_record(
    *,
    headline: str = "A routine company update",
    article_text: str = "The company published an update.",
    vendor_tickers: list[str] | None = None,
) -> dict:
    return {
        "row_number": 1,
        "article_id": "example",
        "headline": headline,
        "article_text": article_text,
        "vendor_tickers": [] if vendor_tickers is None else vendor_tickers,
        "target": {
            "company": "Advanced Micro Devices",
            "ticker": "AMD",
            "sector": "Semiconductors",
            "sector_benchmark": "SOXX",
            "known_sector_peers": ["NVDA", "INTC", "MU"],
        },
    }


class MappingTests(unittest.TestCase):
    def test_semantic_applicability_controls_scope(self) -> None:
        labels = fine_labels(relevance="irrelevant", event_scope="firm_specific")
        self.assertFalse(MODULE.semantic_applicable(labels))
        self.assertEqual(MODULE.map_fine_labels(labels)["shock_scope"], "unclear")

    def test_core_field_mappings(self) -> None:
        mapped = MODULE.map_fine_labels(
            fine_labels(
                event_scope="sector_wide",
                event_type="regulation_trade_policy",
                target_direction="negative",
                sector_direction="negative",
                peer_effect="same_direction",
                information_status="scheduled_or_expected",
            )
        )
        self.assertEqual(
            mapped,
            {
                "shock_scope": "common",
                "event_family": "regulation_legal",
                "information_status": "anticipated",
                "directional_alignment": "same_direction",
            },
        )

    def test_idiosyncratic_scope_precedes_incidental_sector_direction(self) -> None:
        mapped = MODULE.map_fine_labels(
            fine_labels(
                event_scope="firm_specific",
                target_direction="positive",
                sector_direction="positive",
            )
        )
        self.assertEqual(mapped["directional_alignment"], "single_firm_only")

    def test_explicit_peer_comparison_survives_idiosyncratic_scope(self) -> None:
        mapped = MODULE.map_fine_labels(
            fine_labels(event_scope="peer_specific", peer_effect="opposite_direction")
        )
        self.assertEqual(mapped["directional_alignment"], "opposite_direction")

    def test_full_reference_distribution_locks_audit_mapping(self) -> None:
        path = REPOSITORY_ROOT / "annotations" / "chatgpt_5_6_sol_reference.jsonl"
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        mapped = [MODULE.map_fine_labels(record["labels"]) for record in records]
        expected = {
            "shock_scope": {
                "idiosyncratic": 125,
                "common": 33,
                "mixed": 53,
                "unclear": 89,
            },
            "event_family": {
                "earnings_guidance": 41,
                "product_demand": 63,
                "supply_capacity": 20,
                "regulation_legal": 25,
                "corporate_analyst": 85,
                "macro_market": 31,
                "other_or_unclear": 35,
            },
            "information_status": {
                "confirmed": 164,
                "anticipated": 41,
                "rumor_or_opinion": 95,
                "unclear": 0,
            },
            "directional_alignment": {
                "single_firm_only": 124,
                "same_direction": 15,
                "opposite_direction": 7,
                "common_direction_unclear": 65,
                "unclear": 89,
            },
        }
        for field, field_expected in expected.items():
            observed = Counter(item[field] for item in mapped)
            for label, count in field_expected.items():
                self.assertEqual(observed[label], count, f"{field}.{label}")

    def test_invalid_fine_label_is_rejected(self) -> None:
        with self.assertRaises(MODULE.CoarseFeatureError):
            MODULE.map_fine_labels(fine_labels(event_type="not_a_real_type"))

    def test_coarse_validation_requires_exact_fields(self) -> None:
        valid = MODULE.map_fine_labels(fine_labels())
        MODULE.validate_coarse_labels(valid)
        with self.assertRaises(MODULE.CoarseFeatureError):
            MODULE.validate_coarse_labels({**valid, "extra": "value"})


class DeterministicGateTests(unittest.TestCase):
    def test_target_vendor_tag_has_priority(self) -> None:
        features = MODULE.deterministic_features(
            input_record(vendor_tickers=["NVDA", "AMD"])
        )
        self.assertTrue(features["semantic_applicable"])
        self.assertEqual(features["gate_route"], "direct_target")
        self.assertEqual(features["peer_vendor_tickers"], ["NVDA"])

    def test_peer_vendor_tag_routes_to_sector_or_peer(self) -> None:
        features = MODULE.deterministic_features(input_record(vendor_tickers=["NVDA"]))
        self.assertEqual(features["gate_route"], "sector_or_peer")

    def test_explicit_macro_phrase_routes_to_macro(self) -> None:
        features = MODULE.deterministic_features(
            input_record(article_text="The Federal Reserve changed monetary policy.")
        )
        self.assertEqual(features["gate_route"], "macro_relevant")
        self.assertIn("federal reserve", features["macro_terms_in_text"])

    def test_missing_evidence_abstains(self) -> None:
        features = MODULE.deterministic_features(input_record())
        self.assertFalse(features["semantic_applicable"])
        self.assertEqual(features["gate_route"], "no_semantic_evidence")

    def test_short_ticker_does_not_match_word_substring(self) -> None:
        features = MODULE.deterministic_features(
            input_record(article_text="Demand is much stronger this quarter.")
        )
        self.assertNotIn("MU", features["peer_tickers_in_text"])

    def test_company_name_in_text_is_direct_evidence(self) -> None:
        features = MODULE.deterministic_features(
            input_record(article_text="Advanced Micro Devices announced a product.")
        )
        self.assertTrue(features["target_company_in_text"])
        self.assertEqual(features["gate_route"], "direct_target")

    def test_rule_provenance_is_present(self) -> None:
        features = MODULE.deterministic_features(input_record())
        self.assertEqual(features["rule_version"], MODULE.DETERMINISTIC_RULE_VERSION)
        self.assertRegex(features["rules_sha256"], r"^[0-9a-f]{64}$")

    def test_company_alias_is_direct_target_evidence(self) -> None:
        features = MODULE.deterministic_features(
            input_record(article_text="AMD announced a new accelerator.")
        )
        self.assertTrue(features["target_ticker_in_text"])
        self.assertEqual(features["gate_route"], "direct_target")

    def test_explicit_surprise_requires_comparator_or_revision(self) -> None:
        self.assertEqual(
            MODULE.explicit_surprise_from_text(
                "AMD revenue beat analyst estimates."
            )[0],
            "positive",
        )
        self.assertEqual(
            MODULE.explicit_surprise_from_text(
                "AMD lowered its full-year guidance."
            )[0],
            "negative",
        )
        self.assertEqual(
            MODULE.explicit_surprise_from_text(
                "AMD shares rose after the announcement."
            )[0],
            "none",
        )

    def test_explicit_surprise_can_be_mixed(self) -> None:
        label, positive, negative = MODULE.explicit_surprise_from_text(
            "Revenue beat estimates, but the company cut its outlook."
        )
        self.assertEqual(label, "mixed")
        self.assertTrue(positive)
        self.assertTrue(negative)


class StableSplitTests(unittest.TestCase):
    @staticmethod
    def records() -> list[dict]:
        records = []
        for label, count in (("idiosyncratic", 10), ("common", 5), ("unclear", 3)):
            for index in range(count):
                records.append(
                    {
                        "article_id": f"{label}-{index}",
                        "target_ticker": "AMD",
                        "labels": {"shock_scope": label},
                    }
                )
        return records

    def test_split_is_stable_under_input_reordering(self) -> None:
        records = self.records()
        first = MODULE.stratified_split(records, development_fraction=0.2, seed="fixed")
        second = MODULE.stratified_split(
            list(reversed(records)), development_fraction=0.2, seed="fixed"
        )
        self.assertEqual(first["assignments"], second["assignments"])
        self.assertEqual(
            first["counts_by_stratum"],
            {
                "common": {"total": 5, "development": 1, "evaluation": 4},
                "idiosyncratic": {"total": 10, "development": 2, "evaluation": 8},
                "unclear": {"total": 3, "development": 1, "evaluation": 2},
            },
        )

    def test_duplicate_identity_is_rejected(self) -> None:
        record = self.records()[0]
        with self.assertRaises(MODULE.CoarseFeatureError):
            MODULE.stratified_split([record, dict(record)])


if __name__ == "__main__":
    unittest.main()
