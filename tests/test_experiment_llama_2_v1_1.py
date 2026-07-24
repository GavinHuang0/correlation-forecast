from __future__ import annotations

import importlib.util
import json
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


experiment = load_module(
    "experiment_llama_2_v1_1.py", "llama_2_v1_1_experiment_for_test"
)


class FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 2

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        chat_template=None,
    ):
        assert tokenize is True
        prompt = messages[0]["content"]
        prompt_ids = [1, len(prompt), 20]
        if len(messages) == 1:
            return prompt_ids
        reply = messages[1]["content"]
        assert reply.startswith("Answer: ")
        letter = reply[-1]
        return prompt_ids + [30, 31, 32, 100 + ord(letter) - ord("A"), 2]

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens,
        clean_up_tokenization_spaces,
    ):
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        if list(token_ids) == [30, 31, 32]:
            return " Answer:"
        return "?"


class LlamaV11ExperimentTests(unittest.TestCase):
    def test_cyclic_rotations_put_every_label_in_every_position(self) -> None:
        labels = ["one", "two", "three", "four"]
        rotations = experiment.cyclic_rotations(labels)
        self.assertEqual(len(rotations), len(labels))
        self.assertEqual(rotations[0], labels)
        for position in range(len(labels)):
            self.assertEqual(
                {rotation[position] for rotation in rotations}, set(labels)
            )

    def test_exact_answer_cue_contract_excludes_eos(self) -> None:
        tokenizer = FakeTokenizer()
        choice = experiment.build_answer_cue_choice(
            tokenizer=tokenizer,
            prompt='Reply exactly "Answer: X".',
            semantic_to_letter={
                "idiosyncratic": "A",
                "common": "B",
                "mixed": "C",
                "unclear": "D",
            },
        )
        self.assertEqual(choice["shared_assistant_prefix_ids"], [30, 31, 32])
        self.assertEqual(choice["shared_assistant_suffix_ids"], [2])
        self.assertEqual(choice["input_ids"][-3:], [30, 31, 32])
        self.assertEqual(
            choice["letter_to_token_id"],
            {"A": 100, "B": 101, "C": 102, "D": 103},
        )
        self.assertIn("Answer:", choice["shared_assistant_prefix_text"])

    def test_contract_rejects_nonunique_candidate_tokens(self) -> None:
        class BrokenTokenizer(FakeTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                ids = super().apply_chat_template(messages, **kwargs)
                if len(messages) == 2:
                    ids[-2] = 100
                return ids

        with self.assertRaisesRegex(
            ValueError, "candidate-specific token|unique"
        ):
            experiment.build_answer_cue_choice(
                tokenizer=BrokenTokenizer(),
                prompt="test",
                semantic_to_letter={"one": "A", "two": "B"},
            )

    def test_rotation_mean_and_schema_tie_break(self) -> None:
        labels = ["first", "second", "third"]
        score_maps = [
            {"first": -1.0, "second": -2.0, "third": -3.0},
            {"first": -2.0, "second": -1.0, "third": -3.0},
            {"first": -3.0, "second": -3.0, "third": -1.0},
        ]
        means = experiment.mean_rotation_scores(score_maps, labels)
        self.assertEqual(means, {"first": -2.0, "second": -2.0, "third": -7 / 3})
        self.assertEqual(experiment.v1.best_label(means, labels), "first")

    def test_validate_only_does_not_create_output(self) -> None:
        fixture = (
            ROOT / "tests" / "fixtures" / "flan_coarse_sanity.jsonl"
        ).read_text(encoding="utf-8").splitlines()[0]
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.jsonl"
            output_path = Path(directory) / "output.jsonl"
            input_path.write_text(fixture + "\n", encoding="utf-8")
            result = experiment.main(
                [
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--schema",
                    str(ROOT / "config" / "news_feature_schema_coarse.json"),
                    "--validate-only",
                ]
            )
            self.assertEqual(result, 0)
            self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()
