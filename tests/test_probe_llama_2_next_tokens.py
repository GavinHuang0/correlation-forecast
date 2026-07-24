from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

SPEC = importlib.util.spec_from_file_location(
    "probe_llama_2_next_tokens_for_test",
    SCRIPTS / "probe_llama_2_next_tokens.py",
)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


class ProbeLlama2NextTokensTests(unittest.TestCase):
    def test_assistant_suffix_requires_exact_frozen_prompt_prefix(self) -> None:
        self.assertEqual(
            probe.assistant_suffix_from_full_chat([1, 2, 3], [1, 2, 3, 4, 5]),
            [4, 5],
        )
        with self.assertRaisesRegex(ValueError, "token offset 1"):
            probe.assistant_suffix_from_full_chat([1, 2, 3], [1, 9, 3, 4])
        with self.assertRaisesRegex(ValueError, "did not extend"):
            probe.assistant_suffix_from_full_chat([1, 2], [1, 2])

    def test_partition_candidate_suffixes_finds_chat_scaffolding(self) -> None:
        result = probe.partition_candidate_suffixes(
            {
                "A": [10, 21, 90, 2],
                "B": [10, 22, 90, 2],
                "C": [10, 23, 90, 2],
            }
        )
        self.assertEqual(result["shared_prefix_ids"], [10])
        self.assertEqual(result["shared_suffix_ids"], [90, 2])
        self.assertEqual(
            result["candidate_specific_ids"],
            {"A": [21], "B": [22], "C": [23]},
        )

    def test_partition_candidate_suffixes_rejects_nonunique_letters(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "letter-specific|unique"
        ):
            probe.partition_candidate_suffixes(
                {"A": [10, 21, 2], "B": [10, 21, 2]}
            )

    def test_select_record_uses_one_based_jsonl_position(self) -> None:
        records = [{"row_number": 5}, {"row_number": 12}]
        self.assertEqual(probe.select_record(records, 1)["row_number"], 5)
        self.assertEqual(probe.select_record(records, 2)["row_number"], 12)
        with self.assertRaisesRegex(IndexError, "1..2"):
            probe.select_record(records, 3)

    def test_directional_alignment_requires_scope(self) -> None:
        common = [
            "--input",
            "input.jsonl",
            "--row",
            "1",
            "--field",
            "directional_alignment",
            "--output",
            "probe.json",
        ]
        with self.assertRaises(SystemExit):
            probe.parse_args(common)
        args = probe.parse_args(
            common + ["--selected-scope", "idiosyncratic"]
        )
        self.assertEqual(args.selected_scope, "idiosyncratic")


if __name__ == "__main__":
    unittest.main()
