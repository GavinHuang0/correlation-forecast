from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import evaluate_flan_coarse

SCRIPT = SCRIPTS / "extract_llama_3_1_coarse.py"
SPEC = importlib.util.spec_from_file_location("extract_llama31_for_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
extractor = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = extractor
SPEC.loader.exec_module(extractor)


class FakeTokenizer:
    chat_template = "official"

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt
    ):
        self.assert_contract(tokenize)
        if len(messages) == 1:
            if not add_generation_prompt:
                raise AssertionError("Expected generation prompt")
            return [1, 2, 3]
        if add_generation_prompt:
            raise AssertionError("Complete assistant reply must not add a prompt")
        content = messages[-1]["content"]
        if content.endswith("A"):
            return [1, 2, 3, 10, 11, 90]
        if content.endswith("B"):
            return [1, 2, 3, 10, 12, 13, 90]
        raise AssertionError(content)

    @staticmethod
    def assert_contract(tokenize):
        if tokenize is not True:
            raise AssertionError("Expected tokenized official template")

    @staticmethod
    def decode(token_ids, **_kwargs):
        return "Answer:" if token_ids == [10] else "?"


class ExtractLlama31Tests(unittest.TestCase):
    def test_official_chat_choice_supports_multitoken_candidate(self) -> None:
        choice = extractor.build_sequence_choice(
            tokenizer=FakeTokenizer(),
            prompt="article and task",
            semantic_to_letter={"first": "A", "second": "B"},
        )
        self.assertEqual(choice["input_ids"], [1, 2, 3, 10])
        self.assertEqual(choice["candidate_token_ids"]["A"], [11])
        self.assertEqual(choice["candidate_token_ids"]["B"], [12, 13])
        self.assertEqual(choice["shared_assistant_suffix_ids"], [90])

    def test_candidate_prediction_positions_and_rotation_mean(self) -> None:
        self.assertEqual(
            extractor.candidate_prediction_positions(
                prefix_length=10, candidate_length=3
            ),
            [9, 10, 11],
        )
        self.assertEqual(
            extractor.shared_experiment.mean_rotation_scores(
                [
                    {"first": -2.0, "second": -4.0},
                    {"first": -4.0, "second": -2.0},
                ],
                ["first", "second"],
            ),
            {"first": -3.0, "second": -3.0},
        )

    def test_runtime_configuration_is_scoped_and_restored(self) -> None:
        original_model = extractor.shared_runtime.MODEL_DEFAULT
        original_prompt = extractor.shared_experiment.PROMPT_VERSION
        with extractor.configured_shared_runtime():
            self.assertEqual(
                extractor.shared_runtime.MODEL_DEFAULT, extractor.MODEL_ID
            )
            self.assertEqual(
                extractor.shared_runtime.CONSERVATIVE_DATA_CUTOFF,
                "2023-12-31",
            )
            self.assertIs(
                extractor.shared_experiment.build_answer_cue_choice,
                extractor.build_sequence_choice,
            )
        self.assertEqual(extractor.shared_runtime.MODEL_DEFAULT, original_model)
        self.assertEqual(
            extractor.shared_experiment.PROMPT_VERSION, original_prompt
        )

    def test_snapshot_contract_rejects_wrong_revision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text(
                (
                    '{"manifest_version":"llama-3.1-local-snapshot-v1",'
                    f'"model_id":"{extractor.MODEL_ID}",'
                    '"model_revision":"0000000000000000000000000000000000000000"}'
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "wrong revision"):
                extractor.validate_snapshot_contract(path)

    def test_snapshot_contract_rejects_changed_cached_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            (snapshot / "config.json").write_text("{}\n", encoding="utf-8")
            (snapshot / "tokenizer_config.json").write_text(
                "{}\n", encoding="utf-8"
            )
            (snapshot / "tokenizer.json").write_text("{}\n", encoding="utf-8")
            (snapshot / "model.safetensors").write_bytes(b"weights")
            files = extractor.cache_common.snapshot_file_manifest(snapshot)
            manifest = root / "manifest.json"
            manifest.write_text(
                __import__("json").dumps(
                    {
                        "manifest_version": "llama-3.1-local-snapshot-v1",
                        "model_id": extractor.MODEL_ID,
                        "model_revision": extractor.MODEL_REVISION,
                        "snapshot_path": str(snapshot),
                        "file_count": len(files),
                        "total_size_bytes": sum(
                            item["size_bytes"] for item in files.values()
                        ),
                        "files": files,
                    }
                ),
                encoding="utf-8",
            )
            (snapshot / "model.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "bytes differ"):
                extractor.validate_snapshot_contract(manifest)

    def test_bound_snapshot_download_forces_verified_path_and_restores(
        self,
    ) -> None:
        fake_hub = types.ModuleType("huggingface_hub")
        original = lambda **_kwargs: "unverified"  # noqa: E731
        fake_hub.snapshot_download = original
        verified = Path("verified-snapshot").resolve()
        with mock.patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
            with extractor.bound_snapshot_download(verified):
                self.assertEqual(
                    fake_hub.snapshot_download(
                        repo_id=extractor.MODEL_ID,
                        revision=extractor.MODEL_REVISION,
                        local_files_only=True,
                    ),
                    str(verified),
                )
                with self.assertRaisesRegex(ValueError, "other than"):
                    fake_hub.snapshot_download(
                        repo_id="wrong/model",
                        revision=extractor.MODEL_REVISION,
                        local_files_only=True,
                    )
            self.assertIs(fake_hub.snapshot_download, original)

    def test_shared_evaluator_accepts_the_frozen_prompt_contract(self) -> None:
        self.assertIn(
            extractor.PROMPT_VERSION,
            evaluate_flan_coarse.ACCEPTED_PROMPT_VERSIONS,
        )

    def test_synthetic_llama31_manifest_passes_real_evaluator_contract(
        self,
    ) -> None:
        input_record = {
            "row_number": 1,
            "article_id": "synthetic_irrelevant",
            "time_published_utc": "2024-06-01T12:00:00Z",
            "source": "Synthetic",
            "headline": "Local weather report",
            "article_text": "Rain is expected this afternoon.",
            "vendor_tickers": [],
            "target": {
                "company": "Advanced Micro Devices",
                "ticker": "AMD",
                "sector": "Semiconductors",
                "sector_benchmark": "SOXX",
                "known_sector_peers": ["NVDA", "INTC"],
            },
        }
        fine_reference = {
            "row_number": 1,
            "article_id": "synthetic_irrelevant",
            "target_ticker": "AMD",
            "labels": {
                "relevance": "irrelevant",
                "event_scope": "unclear",
                "event_type": "other",
                "affected_breadth": "unclear",
                "target_direction": "not_applicable",
                "sector_direction": "not_applicable",
                "peer_effect": "not_applicable",
                "explicit_surprise": "none",
                "information_status": "unclear",
                "transmission_channels": [],
                "affected_companies": [],
                "affected_sectors": [],
                "evidence": {"scope": "", "direction": "", "surprise": ""},
                "abstain_reason": None,
            },
        }
        deterministic = extractor.coarse.deterministic_features(input_record)
        labels = {field: None for field in extractor.coarse.COARSE_FIELDS}
        origins = {
            field: "deterministic_gate"
            for field in extractor.coarse.COARSE_FIELDS
        }
        passes = {
            field: {
                "origin": "deterministic_gate",
                "value": None,
                "reason": "deterministic_relevance_gate",
                "valid": True,
                "input_truncated": False,
            }
            for field in extractor.coarse.COARSE_FIELDS
        }
        prediction = {
            "row_number": 1,
            "article_id": "synthetic_irrelevant",
            "target_ticker": "AMD",
            "model_revision": extractor.MODEL_REVISION,
            "prompt_version": extractor.PROMPT_VERSION,
            "semantic_applicable": False,
            "deterministic_features": deterministic,
            "labels": labels,
            "label_origins": origins,
            "passes": passes,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root / "inputs.jsonl"
            reference = root / "reference.jsonl"
            predictions = root / "predictions.jsonl"
            report = root / "report.json"
            inputs.write_text(
                __import__("json").dumps(input_record) + "\n",
                encoding="utf-8",
            )
            reference.write_text(
                __import__("json").dumps(fine_reference) + "\n",
                encoding="utf-8",
            )
            predictions.write_text(
                __import__("json").dumps(prediction) + "\n",
                encoding="utf-8",
            )
            schema = ROOT / "config" / "news_feature_schema_coarse.json"
            manifest = {
                "status": "complete",
                "output_sha256": extractor.base.sha256_file(predictions),
                "schema_sha256": extractor.base.sha256_file(schema),
                "prompt_version": extractor.PROMPT_VERSION,
                "model_revision": extractor.MODEL_REVISION,
                "total_prediction_count": 1,
                "selected_article_ids_sha256": extractor.base.sha256_text(
                    "synthetic_irrelevant"
                ),
                "deterministic_module_sha256": extractor.base.sha256_file(
                    Path(extractor.coarse.__file__).resolve()
                ),
                "decoding": extractor.DECODING_METHOD,
            }
            predictions.with_suffix(".jsonl.manifest.json").write_text(
                __import__("json").dumps(manifest),
                encoding="utf-8",
            )
            argv = [
                "evaluate_flan_coarse.py",
                "--reference",
                str(reference),
                "--predictions",
                str(predictions),
                "--inputs",
                str(inputs),
                "--schema",
                str(schema),
                "--split",
                "all",
                "--output",
                str(report),
            ]
            with mock.patch.object(sys, "argv", argv):
                self.assertEqual(evaluate_flan_coarse.main(), 0)
            self.assertTrue(report.is_file())


if __name__ == "__main__":
    unittest.main()
