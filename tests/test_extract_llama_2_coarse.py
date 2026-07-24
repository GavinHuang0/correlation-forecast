from __future__ import annotations

import builtins
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


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


extractor = load_module("extract_llama_2_coarse.py", "llama_2_coarse_extractor_for_test")


def load_schema() -> dict:
    return json.loads(
        (ROOT / "config" / "news_feature_schema_coarse.json").read_text(
            encoding="utf-8"
        )
    )


def sanity_records() -> list[dict]:
    path = ROOT / "tests" / "fixtures" / "flan_coarse_sanity.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class FakeChatTokenizer:
    chat_template = "fake official Llama 2 template"
    pad_token_id = 0
    eos_token_id = 99

    def __init__(self) -> None:
        self.templates: list[str] = []

    def apply_chat_template(
        self,
        messages,
        *,
        chat_template,
        tokenize,
        add_generation_prompt,
    ):
        assert tokenize is True
        self.templates.append(chat_template)
        user = messages[0]["content"]
        return [10, len(user), 20]

    def __call__(self, text, *, add_special_tokens, truncation):
        assert add_special_tokens is False
        assert truncation is False
        return {"input_ids": [100 + ord(text) - ord("A")]}


class Llama2CoarseExtractorTests(unittest.TestCase):
    def test_default_checkpoint_is_pinned_and_custom_revisions_are_immutable(self) -> None:
        args = extractor.parse_args(
            ["--input", "in.jsonl", "--output", "out.jsonl", "--validate-only"]
        )
        self.assertEqual(args.model_id, "meta-llama/Llama-2-7b-chat-hf")
        self.assertEqual(
            args.revision, "f5db02db724555f92da89c216ac04704f23d4590"
        )
        self.assertEqual(args.device, "cuda")
        self.assertEqual(args.precision, "float16")
        self.assertEqual(args.quantization, "nf4")
        self.assertEqual(args.batch_size, 1)
        self.assertEqual(args.max_input_tokens, 4096)
        with self.assertRaises(SystemExit):
            extractor.parse_args(
                [
                    "--input",
                    "in.jsonl",
                    "--output",
                    "out.jsonl",
                    "--revision",
                    "main",
                ]
            )
        with self.assertRaises(SystemExit):
            extractor.parse_args(
                [
                    "--input",
                    "in.jsonl",
                    "--output",
                    "out.jsonl",
                    "--revision",
                    "0" * 40,
                ]
            )
        with self.assertRaises(SystemExit):
            extractor.parse_args(
                [
                    "--input",
                    "in.jsonl",
                    "--output",
                    "out.jsonl",
                    "--device",
                    "cpu",
                    "--quantization",
                    "nf4",
                ]
            )

    def test_prompts_are_article_first_and_field_specific(self) -> None:
        schema = load_schema()
        record = sanity_records()[5]
        for field in extractor.coarse.COARSE_FIELDS:
            scope = "common" if field == "directional_alignment" else None
            labels = extractor.allowed_labels_for(schema, field, scope)
            prompt, mapping = extractor.build_letter_prompt(
                record, schema, field, labels, scope
            )
            self.assertLess(prompt.index("ARTICLE"), prompt.index("TASK"))

    def test_post_cutoff_validation_is_strict_and_timezone_aware(self) -> None:
        record = dict(sanity_records()[0])
        record["time_published_utc"] = "2024-05-01T12:00:00Z"
        parsed = extractor.validate_post_cutoff_record(record)
        self.assertEqual(parsed.isoformat(), "2024-05-01T12:00:00+00:00")

        record["time_published_utc"] = "2023-07-31T23:59:59Z"
        with self.assertRaisesRegex(ValueError, "not after"):
            extractor.validate_post_cutoff_record(record)

        record["time_published_utc"] = "2024-01-01T00:00:00"
        with self.assertRaisesRegex(ValueError, "explicit UTC offset"):
            extractor.validate_post_cutoff_record(record)
            self.assertLess(prompt.index("TASK"), prompt.index("OPTIONS"))
            self.assertIn(record["headline"], prompt)
            self.assertIn(record["article_text"], prompt)
            self.assertIn("Do not use external knowledge", prompt)
            self.assertIn("later outcomes", prompt)
            self.assertEqual(list(mapping.values())[0], "A")
            if field == "directional_alignment":
                self.assertIn("Known sector peers", prompt)
            else:
                self.assertNotIn("Known sector peers", prompt)

    def test_reversed_order_changes_letters_not_semantic_choices(self) -> None:
        schema = load_schema()
        record = sanity_records()[5]
        labels = schema["closed_label_fields"]["shock_scope"]
        _, canonical = extractor.build_letter_prompt(
            record, schema, "shock_scope", labels
        )
        _, reversed_mapping = extractor.build_letter_prompt(
            record, schema, "shock_scope", list(reversed(labels))
        )
        self.assertEqual(set(canonical), set(labels))
        self.assertEqual(set(reversed_mapping), set(labels))
        self.assertEqual(canonical[labels[0]], "A")
        self.assertEqual(
            reversed_mapping[labels[0]], chr(ord("A") + len(labels) - 1)
        )

    def test_official_chat_template_and_single_token_choices(self) -> None:
        tokenizer = FakeChatTokenizer()
        prompt = "ARTICLE\nText: test\n\nTASK\nChoose.\n\nOPTIONS\nA. one\nB. two"
        prompt_ids, token_ids = extractor.candidate_letter_token_ids(
            tokenizer, prompt, ["A", "B"]
        )
        self.assertEqual(prompt_ids, [10, len(prompt), 20])
        self.assertEqual(token_ids, {"A": 100, "B": 101})
        self.assertEqual(
            tokenizer.templates,
            [extractor.LLAMA_2_SINGLE_TURN_CHAT_TEMPLATE],
        )
        self.assertIn("[INST]", tokenizer.templates[0])

    def test_candidate_score_math_and_order_average(self) -> None:
        canonical_mapping = {"first": "A", "second": "B", "third": "C"}
        reversed_mapping = {"third": "A", "second": "B", "first": "C"}
        token_ids = {"A": 1, "B": 2, "C": 3}
        canonical = extractor.score_map_from_next_token_log_probs(
            [-99.0, -0.2, -0.4, -1.0],
            canonical_mapping,
            token_ids,
        )
        reversed_scores = extractor.score_map_from_next_token_log_probs(
            [-99.0, -0.9, -0.1, -0.8],
            reversed_mapping,
            token_ids,
        )
        averaged = extractor.mean_score_maps(
            canonical, reversed_scores, ["first", "second", "third"]
        )
        self.assertEqual(
            averaged, {"first": -0.5, "second": -0.25, "third": -0.95}
        )
        self.assertEqual(
            extractor.best_label(averaged, ["first", "second", "third"]),
            "second",
        )
        self.assertEqual(extractor.score_margin(averaged), 0.25)

    def test_hierarchy_exactly_reuses_v04_baseline_logic(self) -> None:
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
                "directional_alignment",
                {"shock_scope": "idiosyncratic"},
                target_only,
            ),
            (False, "single_firm_only", "single_firm_idiosyncratic"),
        )
        self.assertEqual(
            extractor.hierarchy_decision(
                "directional_alignment",
                {"shock_scope": "common"},
                target_only,
            ),
            (True, None, None),
        )

    def test_validate_only_never_imports_or_loads_model_packages(self) -> None:
        record = sanity_records()[0]
        with tempfile.TemporaryDirectory() as temp_directory:
            input_path = Path(temp_directory) / "input.jsonl"
            output_path = Path(temp_directory) / "unused.jsonl"
            input_path.write_text(
                json.dumps(record, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            real_import = builtins.__import__

            def guarded_import(name, *args, **kwargs):
                if name.split(".", 1)[0] in {
                    "torch",
                    "transformers",
                    "huggingface_hub",
                    "bitsandbytes",
                }:
                    raise AssertionError(f"validate-only imported {name}")
                return real_import(name, *args, **kwargs)

            stdout = io.StringIO()
            with mock.patch("builtins.__import__", side_effect=guarded_import):
                with contextlib.redirect_stdout(stdout):
                    result = extractor.main(
                        [
                            "--input",
                            str(input_path),
                            "--output",
                            str(output_path),
                            "--schema",
                            str(
                                ROOT
                                / "config"
                                / "news_feature_schema_coarse.json"
                            ),
                            "--validate-only",
                        ]
                    )
            payload = json.loads(stdout.getvalue())
            self.assertEqual(result, 0)
            self.assertFalse(payload["model_packages_imported"])
            self.assertFalse(payload["model_was_loaded"])
            self.assertEqual(payload["validated_input_records"], 1)
            self.assertFalse(output_path.exists())

    def test_nf4_device_map_rejects_cpu_or_disk_offload(self) -> None:
        self.assertEqual(
            extractor.validate_nf4_device_map(SimpleNamespace(hf_device_map={"": 0})),
            {"": 0},
        )
        with self.assertRaisesRegex(RuntimeError, "forbids CPU or disk"):
            extractor.validate_nf4_device_map(
                SimpleNamespace(
                    hf_device_map={"model.layers.0": "cuda:0", "lm_head": "cpu"}
                )
            )
        with self.assertRaisesRegex(RuntimeError, "nonempty hf_device_map"):
            extractor.validate_nf4_device_map(SimpleNamespace())


if __name__ == "__main__":
    unittest.main()
