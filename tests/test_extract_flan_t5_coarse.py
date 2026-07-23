from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def load_module(filename: str, name: str):
    path = SCRIPTS / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


coarse_features = load_module("coarse_news_features.py", "coarse_features_for_test")
extractor = load_module("extract_flan_t5_coarse.py", "coarse_extractor_for_test")


def load_schema() -> dict:
    return json.loads(
        (ROOT / "config" / "news_feature_schema_coarse.json").read_text(encoding="utf-8")
    )


def sanity_records() -> list[dict]:
    path = ROOT / "tests" / "fixtures" / "flan_coarse_sanity.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class CoarseExtractorTests(unittest.TestCase):
    def test_schema_matches_versioned_core_fields(self) -> None:
        schema = load_schema()
        extractor.validate_schema(schema)
        self.assertEqual(tuple(schema["flan_core_fields"]), tuple(coarse_features.COARSE_FIELDS))

    def test_prompts_put_article_before_task_and_use_one_field(self) -> None:
        schema = load_schema()
        record = sanity_records()[5]
        for field in coarse_features.COARSE_FIELDS:
            selected_scope = "common" if field == "directional_alignment" else None
            prompt = extractor.build_natural_prompt(record, schema, field, selected_scope)
            self.assertLess(prompt.index("ARTICLE"), prompt.index("TASK"))
            self.assertLess(prompt.index("TASK"), prompt.index("OPTIONS"))
            self.assertIn(record["headline"], prompt)
            self.assertIn(record["article_text"], prompt)
            if field == "directional_alignment":
                self.assertIn("Known sector peers", prompt)
            else:
                self.assertNotIn("Known sector peers", prompt)

    def test_letter_order_reversal_changes_position_not_semantics(self) -> None:
        schema = load_schema()
        record = sanity_records()[5]
        labels = schema["closed_label_fields"]["shock_scope"]
        _, canonical = extractor.build_letter_prompt(record, schema, "shock_scope", labels)
        _, reversed_mapping = extractor.build_letter_prompt(
            record, schema, "shock_scope", list(reversed(labels))
        )
        self.assertEqual(set(canonical), set(reversed_mapping))
        self.assertEqual(set(canonical), set(labels))
        self.assertEqual(canonical[labels[0]], "A")
        self.assertEqual(reversed_mapping[labels[0]], chr(ord("A") + len(labels) - 1))

    def test_order_averaging_and_margin_are_deterministic(self) -> None:
        labels = ["first", "second", "third"]
        first = {"first": -0.2, "second": -0.4, "third": -1.0}
        second = {"first": -0.8, "second": -0.1, "third": -0.9}
        averaged = extractor.mean_score_maps(first, second, labels)
        self.assertEqual(averaged, {"first": -0.5, "second": -0.25, "third": -0.95})
        self.assertEqual(extractor.best_label(averaged, labels), "second")
        self.assertEqual(extractor.score_margin(averaged), 0.25)

    def test_hierarchy_skips_irrelevant_and_derives_obvious_alignment(self) -> None:
        irrelevant = {
            "semantic_applicable": False,
            "text_target_mentioned": False,
            "text_peer_count": 0,
        }
        self.assertEqual(
            extractor.hierarchy_decision("shock_scope", {}, irrelevant),
            (False, None, "deterministic_relevance_gate"),
        )

        target_only = {
            "semantic_applicable": True,
            "text_target_mentioned": True,
            "text_peer_count": 0,
        }
        self.assertEqual(
            extractor.hierarchy_decision(
                "directional_alignment", {"shock_scope": "idiosyncratic"}, target_only
            ),
            (False, "single_firm_only", "single_firm_idiosyncratic"),
        )

        peer_only = {
            "semantic_applicable": True,
            "text_target_mentioned": False,
            "text_peer_count": 1,
        }
        self.assertEqual(
            extractor.hierarchy_decision(
                "directional_alignment", {"shock_scope": "idiosyncratic"}, peer_only
            ),
            (False, "single_firm_only", "single_firm_idiosyncratic"),
        )
        self.assertEqual(
            extractor.hierarchy_decision(
                "directional_alignment", {"shock_scope": "common"}, target_only
            ),
            (True, None, None),
        )

    def test_deterministic_gate_routes_synthetic_edge_cases(self) -> None:
        records = {record["article_id"]: record for record in sanity_records()}
        irrelevant = coarse_features.deterministic_features(records["sanity_gate_irrelevant"])
        self.assertFalse(irrelevant["semantic_applicable"])
        self.assertEqual(irrelevant["gate_route"], "no_semantic_evidence")
        self.assertEqual(irrelevant["deterministic_relevance_route"], "no_relevance_evidence")

        vendor_only = coarse_features.deterministic_features(records["sanity_vendor_only"])
        self.assertEqual(vendor_only["gate_route"], "direct_target")
        self.assertEqual(vendor_only["deterministic_relevance_route"], "vendor_only_review")
        self.assertFalse(vendor_only["target_ticker_in_text"])
        self.assertFalse(vendor_only["target_company_in_text"])

    def test_short_ticker_mu_does_not_match_ordinary_word_prefix(self) -> None:
        record = sanity_records()[0]
        record["target"] = {
            "company": "Micron Technology",
            "ticker": "MU",
            "sector": "Semiconductors",
            "sector_benchmark": "SOXX",
            "known_sector_peers": ["AMD", "NVDA", "INTC", "AVGO", "QCOM"],
        }
        record["headline"] = "Multiple retailers report sales"
        record["article_text"] = "Multiple retailers reported sales."
        record["vendor_tickers"] = []
        features = coarse_features.deterministic_features(record)
        self.assertFalse(features["target_ticker_in_text"])
        self.assertFalse(features["target_company_in_text"])


if __name__ == "__main__":
    unittest.main()
