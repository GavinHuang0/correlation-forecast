from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
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


extractor = load_module(
    "extract_flan_t5_final_candidate.py", "final_candidate_extractor_for_test"
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_protocol() -> dict:
    return load_json(ROOT / "config" / "flan_t5_final_candidate_v0_5.json")


def load_schema() -> dict:
    return load_json(ROOT / "config" / "news_feature_schema_coarse.json")


def sanity_record() -> dict:
    line = (
        ROOT / "tests" / "fixtures" / "flan_coarse_sanity.jsonl"
    ).read_text(encoding="utf-8").splitlines()[0]
    return json.loads(line)


class FakeTokenizer:
    def __init__(self) -> None:
        self.truncation_arguments: list[bool] = []

    def __call__(self, prompt: str, **kwargs):
        self.truncation_arguments.append(kwargs["truncation"])
        return {"input_ids": prompt.split()}


class FinalCandidateExtractorTests(unittest.TestCase):
    def test_protocol_is_hash_lockable_and_component_set_is_exact(self) -> None:
        protocol = load_protocol()
        schema = load_schema()
        extractor.validate_protocol_config(protocol, schema)
        self.assertEqual(protocol["protocol_version"], "0.5.0")
        self.assertEqual(protocol["prompt_version"], extractor.PROMPT_VERSION)
        self.assertEqual(
            tuple(protocol["candidate_component_order"]), extractor.COMPONENT_ORDER
        )
        self.assertEqual(
            extractor.COMPONENT_ORDER,
            (
                "scope_firm",
                "scope_common",
                "event_earnings_guidance",
                "event_product_demand",
                "event_supply_capacity",
                "event_regulation_legal",
                "event_corporate_analyst",
                "event_macro_market",
                "alignment_same",
                "alignment_opposite",
            ),
        )

    def test_prompts_are_article_first_binary_and_order_reversed(self) -> None:
        record = sanity_record()
        protocol = load_protocol()
        for component_name in extractor.COMPONENT_ORDER:
            canonical, canonical_mapping, reversed_prompt, reversed_mapping = (
                extractor.prompt_pair(record, protocol, component_name)
            )
            self.assertLess(canonical.index("ARTICLE"), canonical.index("TASK"))
            self.assertLess(reversed_prompt.index("ARTICLE"), reversed_prompt.index("TASK"))
            self.assertIn(record["headline"], canonical)
            self.assertIn(record["article_text"], canonical)
            self.assertIn(
                protocol["components"][component_name]["definition"], canonical
            )
            self.assertEqual(canonical_mapping, {"yes": "A", "no": "B"})
            self.assertEqual(reversed_mapping, {"no": "A", "yes": "B"})
            self.assertIn("A. yes\nB. no", canonical)
            self.assertIn("A. no\nB. yes", reversed_prompt)
            if component_name.startswith("alignment_"):
                self.assertIn("Known sector peers", canonical)
            else:
                self.assertNotIn("Known sector peers", canonical)

    def test_score_math_and_conservative_tie_break(self) -> None:
        averaged = extractor.average_binary_scores(
            {"yes": -0.2, "no": -0.8},
            {"yes": -0.6, "no": -0.4},
        )
        self.assertAlmostEqual(averaged["yes"], -0.4)
        self.assertAlmostEqual(averaged["no"], -0.6)
        self.assertEqual(extractor.binary_prediction(averaged), "yes")
        self.assertEqual(
            extractor.binary_prediction({"yes": -0.5, "no": -0.5}), "no"
        )

    def test_component_result_preserves_both_orders_and_consensus(self) -> None:
        result = extractor.make_component_result(
            canonical_prompt="canonical",
            reversed_prompt="reversed",
            canonical_mapping={"yes": "A", "no": "B"},
            reversed_mapping={"no": "A", "yes": "B"},
            canonical_scores={"yes": -0.1, "no": -0.9},
            reversed_scores={"yes": -0.3, "no": -0.7},
            canonical_truncated=False,
            reversed_truncated=False,
        )
        self.assertTrue(result["valid"])
        self.assertFalse(result["input_truncated"])
        self.assertEqual(result["canonical_prediction"], "yes")
        self.assertEqual(result["reversed_prediction"], "yes")
        self.assertTrue(result["strict_consensus_yes"])
        self.assertAlmostEqual(result["order_averaged_yes_no_log_odds"], 0.6)
        self.assertEqual(
            set(result["canonical_candidate_mean_log_probabilities"]),
            {"yes", "no"},
        )
        self.assertEqual(len(result["canonical_prompt_sha256"]), 64)
        self.assertEqual(len(result["reversed_prompt_sha256"]), 64)

        disagreed = extractor.make_component_result(
            canonical_prompt="canonical",
            reversed_prompt="reversed",
            canonical_mapping={"yes": "A", "no": "B"},
            reversed_mapping={"no": "A", "yes": "B"},
            canonical_scores={"yes": -0.1, "no": -0.9},
            reversed_scores={"yes": -0.8, "no": -0.2},
            canonical_truncated=False,
            reversed_truncated=False,
        )
        self.assertFalse(disagreed["strict_consensus_yes"])

    def test_component_result_rejects_truncation(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "refuses to record any truncated"):
            extractor.make_component_result(
                canonical_prompt="canonical",
                reversed_prompt="reversed",
                canonical_mapping={"yes": "A", "no": "B"},
                reversed_mapping={"no": "A", "yes": "B"},
                canonical_scores={"yes": -0.1, "no": -0.9},
                reversed_scores={"yes": -0.3, "no": -0.7},
                canonical_truncated=True,
                reversed_truncated=False,
            )

    def test_preflight_never_requests_tokenizer_truncation(self) -> None:
        tokenizer = FakeTokenizer()
        maxima = extractor.preflight_prompt_lengths(
            [sanity_record()], load_protocol(), tokenizer, 512
        )
        self.assertEqual(set(maxima), set(extractor.COMPONENT_ORDER))
        self.assertEqual(len(tokenizer.truncation_arguments), 20)
        self.assertEqual(set(tokenizer.truncation_arguments), {False})

    def test_source_hashes_cover_new_and_all_reused_modules(self) -> None:
        hashes = extractor.source_hashes()
        self.assertEqual(
            set(hashes),
            {
                "extract_flan_t5_final_candidate.py",
                "extract_flan_t5.py",
                "extract_flan_t5_coarse.py",
                "coarse_news_features.py",
            },
        )
        self.assertTrue(all(len(value) == 64 for value in hashes.values()))
        self.assertEqual(len(extractor.prompt_builder_hash()), 64)

    def test_validate_only_does_not_create_output_or_load_model(self) -> None:
        fixture = ROOT / "tests" / "fixtures" / "flan_coarse_sanity.jsonl"
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "candidate.jsonl"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "extract_flan_t5_final_candidate.py"),
                    "--input",
                    str(fixture),
                    "--output",
                    str(output),
                    "--protocol-config",
                    str(ROOT / "config" / "flan_t5_final_candidate_v0_5.json"),
                    "--schema",
                    str(ROOT / "config" / "news_feature_schema_coarse.json"),
                    "--limit",
                    "2",
                    "--validate-only",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            summary = json.loads(completed.stdout)
            self.assertEqual(summary["validated_input_records"], 2)
            self.assertEqual(summary["candidate_component_order"], list(extractor.COMPONENT_ORDER))
            self.assertFalse(summary["model_was_loaded"])
            self.assertFalse(output.exists())
            self.assertFalse(
                output.with_suffix(output.suffix + ".manifest.json").exists()
            )

    def test_candidate_protocol_is_isolated_from_final_label_promotion(self) -> None:
        protocol = load_protocol()
        self.assertNotIn("promotion", protocol)
        self.assertNotIn("labels", protocol)
        self.assertEqual(
            protocol["description"].endswith(
                "does not itself promote components into final coarse labels."
            ),
            True,
        )
        source = (
            SCRIPTS / "extract_flan_t5_final_candidate.py"
        ).read_text(encoding="utf-8")
        self.assertIn("raw_binary_component_scores_only_no_promoted_labels", source)
        self.assertNotIn('"labels": state', source)


if __name__ == "__main__":
    unittest.main()
