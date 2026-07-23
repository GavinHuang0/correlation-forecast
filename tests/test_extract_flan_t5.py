from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "extract_flan_t5.py"
SPEC = importlib.util.spec_from_file_location("extract_flan_t5", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
SCHEMA = json.loads((REPOSITORY_ROOT / "config" / "news_feature_schema.json").read_text(encoding="utf-8"))


def sample_record() -> dict:
    return {
        "row_number": 1,
        "article_id": "example",
        "time_published_utc": "2024-01-01T12:00:00Z",
        "source": "Example Wire",
        "headline": "AMD raises guidance as semiconductor demand improves",
        "article_text": "Advanced Micro Devices raised its forecast above prior guidance as demand improved.",
        "vendor_tickers": ["AMD"],
        "target": {
            "company": "Advanced Micro Devices",
            "ticker": "AMD",
            "sector": "Semiconductors",
            "sector_benchmark": "SOXX",
            "known_sector_peers": ["NVDA", "INTC"],
        },
    }


class ClosedPassTests(unittest.TestCase):
    def test_strict_valid_pass(self) -> None:
        parsed = MODULE.parse_closed_pass(
            "direct_target | firm_specific | single_firm",
            MODULE.PASS_FIELDS["scope"],
            SCHEMA,
        )
        self.assertTrue(parsed["valid"])
        self.assertTrue(parsed["strict_format_valid"])
        self.assertEqual(parsed["values"]["event_scope"], "firm_specific")

    def test_labeled_output_is_parseable_but_non_strict(self) -> None:
        parsed = MODULE.parse_closed_pass(
            "RELEVANCE: direct_target | EVENT_SCOPE: firm_specific | AFFECTED_BREADTH: single_firm",
            MODULE.PASS_FIELDS["scope"],
            SCHEMA,
        )
        self.assertTrue(parsed["valid"])
        self.assertFalse(parsed["strict_format_valid"])

    def test_invalid_enum_is_rejected(self) -> None:
        parsed = MODULE.parse_closed_pass(
            "direct | firm_specific | single_firm",
            MODULE.PASS_FIELDS["scope"],
            SCHEMA,
        )
        self.assertFalse(parsed["valid"])
        self.assertIsNone(parsed["values"]["relevance"])

    def test_single_enum_strict_and_non_strict(self) -> None:
        strict = MODULE.parse_single_enum("firm_specific", "event_scope", SCHEMA)
        prefixed = MODULE.parse_single_enum("EVENT_SCOPE: firm_specific", "event_scope", SCHEMA)
        self.assertTrue(strict["valid"])
        self.assertTrue(strict["strict_format_valid"])
        self.assertTrue(prefixed["valid"])
        self.assertFalse(prefixed["strict_format_valid"])

    def test_single_enum_rejects_option_list(self) -> None:
        parsed = MODULE.parse_single_enum(
            "firm_specific, peer_specific, sector_wide",
            "event_scope",
            SCHEMA,
        )
        self.assertFalse(parsed["valid"])
        self.assertIsNone(parsed["values"]["event_scope"])


class ExtendedPassTests(unittest.TestCase):
    def test_channels(self) -> None:
        parsed = MODULE.parse_channels("technology_product, competition", SCHEMA)
        self.assertTrue(parsed["valid"])
        self.assertEqual(parsed["values"], ["technology_product", "competition"])

    def test_too_many_channels(self) -> None:
        parsed = MODULE.parse_channels("demand, competition, macro_growth", SCHEMA)
        self.assertFalse(parsed["valid"])

    def test_no_channel_is_valid(self) -> None:
        parsed = MODULE.parse_channels("NONE", SCHEMA)
        self.assertTrue(parsed["valid"])
        self.assertEqual(parsed["values"], [])

    def test_blank_channel_is_invalid(self) -> None:
        parsed = MODULE.parse_channels("", SCHEMA)
        self.assertFalse(parsed["valid"])

    def test_entities(self) -> None:
        source = "AMD and NVIDIA are semiconductor companies."
        parsed = MODULE.parse_entities("COMPANIES: AMD; NVIDIA | SECTORS: semiconductor", source)
        self.assertTrue(parsed["valid"])
        self.assertEqual(parsed["values"]["affected_companies"], ["AMD", "NVIDIA"])

    def test_evidence_must_be_exact(self) -> None:
        source = "AMD raised guidance above expectations."
        valid = MODULE.parse_evidence(
            "SCOPE: AMD | DIRECTION: raised guidance | SURPRISE: above expectations",
            source,
        )
        invalid = MODULE.parse_evidence(
            "SCOPE: AMD | DIRECTION: raised its guidance | SURPRISE: above expectations",
            source,
        )
        self.assertTrue(valid["valid"])
        self.assertFalse(invalid["valid"])

    def test_required_evidence_cannot_be_empty(self) -> None:
        labels = {
            "event_scope": "firm_specific",
            "target_direction": "positive",
            "sector_direction": "not_applicable",
            "explicit_surprise": "positive",
        }
        parsed = MODULE.parse_evidence(
            "SCOPE: NONE | DIRECTION: NONE | SURPRISE: NONE",
            "AMD raised guidance above expectations.",
            labels,
        )
        self.assertFalse(parsed["valid"])

    def test_single_entity_field(self) -> None:
        parsed = MODULE.parse_entity_field(
            "AMD; NVIDIA",
            "AMD and NVIDIA announced new products.",
            "company",
        )
        self.assertTrue(parsed["valid"])
        self.assertEqual(parsed["values"], ["AMD", "NVIDIA"])

    def test_blank_entity_field_is_invalid(self) -> None:
        parsed = MODULE.parse_entity_field("", "AMD announced a product.", "company")
        self.assertFalse(parsed["valid"])

    def test_single_evidence_field(self) -> None:
        parsed = MODULE.parse_evidence_field(
            "evidence_surprise",
            "above expectations",
            "AMD reported revenue above expectations.",
            {"explicit_surprise": "positive"},
        )
        self.assertTrue(parsed["valid"])
        self.assertEqual(parsed["values"], {"surprise": "above expectations"})

    def test_blank_optional_evidence_is_invalid(self) -> None:
        parsed = MODULE.parse_evidence_field(
            "evidence_surprise",
            "",
            "AMD announced a product.",
            {"explicit_surprise": "none"},
        )
        self.assertFalse(parsed["valid"])


class PromptTests(unittest.TestCase):
    def test_versioned_pass_counts(self) -> None:
        self.assertEqual(len(MODULE.core_passes_for(MODULE.PROMPT_VERSION)), 9)
        self.assertEqual(len(MODULE.passes_for(MODULE.PROMPT_VERSION, "full")), 15)
        self.assertEqual(len(MODULE.core_passes_for(MODULE.LEGACY_PROMPT_VERSION)), 3)
        self.assertEqual(len(MODULE.passes_for(MODULE.LEGACY_PROMPT_VERSION, "full")), 6)

    def test_all_prompts_include_article_and_target(self) -> None:
        record = sample_record()
        MODULE.validate_input_record(record)
        for pass_name in MODULE.FULL_PASSES:
            with self.subTest(pass_name=pass_name):
                prompt = MODULE.build_prompt(pass_name, record, SCHEMA)
                self.assertIn(record["headline"], prompt)
                self.assertIn(record["target"]["company"], prompt)
                self.assertIn("only", prompt.lower())

    def test_v0_2_core_prompts_ask_for_one_field(self) -> None:
        record = sample_record()
        for pass_name in MODULE.CORE_PASSES:
            prompt = MODULE.build_prompt(pass_name, record, SCHEMA)
            self.assertIn(f"What is the {pass_name} label?", prompt)
            self.assertNotIn(" | ", prompt)
            self.assertTrue(prompt.endswith("Answer:"))
            self.assertIn("Allowed labels:", prompt)

    def test_legacy_prompt_contract_is_still_available(self) -> None:
        prompt = MODULE.build_prompt(
            "scope",
            sample_record(),
            SCHEMA,
            prompt_version=MODULE.LEGACY_PROMPT_VERSION,
        )
        self.assertTrue(prompt.endswith("RELEVANCE | EVENT_SCOPE | AFFECTED_BREADTH"))


if __name__ == "__main__":
    unittest.main()
