from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from scripts.semantic_news_v2 import flan_w17


def assignment(
    assignment_id: str,
    *,
    forecast_date: str = "2025-01-02",
    stock: str = "AAA",
    weight: float = 1.0,
) -> dict:
    headline = f"Headline {assignment_id}"
    description = f"Description {assignment_id}"
    return {
        "assignment_id": assignment_id,
        "article_id": f"article-{assignment_id}",
        "forecast_date": forecast_date,
        "target": {
            "ticker": stock,
            "company": f"{stock} Company",
            "sector": "Test Sector",
            "benchmark": "TEST",
            "known_sector_peers": ["BBB", "CCC"],
        },
        "role": "I",
        "headline": headline,
        "description": description,
        "headline_sha256": flan_w17.sha256_text(headline),
        "description_sha256": flan_w17.sha256_text(description),
        "text_sha256": flan_w17.sha256_text(
            flan_w17.shared_model_text(headline, description)
        ),
        "source_profile": "ordinary_massive_retrospective",
        "source_query_scope_complete": True,
        "candidate_assignment_complete": True,
        "aggregation_weight": weight,
    }


def stock_day(
    expected: int,
    *,
    forecast_date: str = "2025-01-02",
    stock: str = "AAA",
    source_complete: bool = True,
    assignment_complete: bool = True,
    expected_ids: list[str] | None = None,
) -> dict:
    value = {
        "forecast_date": forecast_date,
        "target": {
            "ticker": stock,
            "company": f"{stock} Company",
            "sector": "Test Sector",
            "benchmark": "TEST",
            "known_sector_peers": ["BBB", "CCC"],
        },
        "source_profile": "ordinary_massive_retrospective",
        "source_query_scope_complete": source_complete,
        "candidate_assignment_complete": assignment_complete,
        "expected_assignment_count": expected,
    }
    if expected_ids is not None:
        value["expected_assignment_ids_sha256"] = flan_w17.sha256_text(
            "\n".join(sorted(expected_ids))
        )
    return value


def successful_prediction(
    raw_assignment: dict,
    labels: dict[str, str | None],
    *,
    failed: bool = False,
) -> dict:
    normalized = flan_w17.normalize_assignment(raw_assignment)
    config = flan_w17.default_acceptance_config()
    fields = {}
    for field in flan_w17.W17_FIELDS:
        label = labels[field]
        accepted = label is not None
        if label is None:
            label = flan_w17.ABSTENTION_LABELS[field][0]
        score_labels = (
            flan_w17.PREDICTIVE_LABELS[field]
            + flan_w17.ABSTENTION_LABELS[field]
        )
        scores = {
            value: (0.0 if value == label else -1.0)
            for value in score_labels
        }
        fields[field] = {
            "structurally_applicable": True,
            "raw_label": label,
            "calibrated_label": label,
            "accepted": accepted,
            "acceptance_reason": "accepted" if accepted else "unclear",
            "quality_weight": 1.0 if accepted else None,
            "adjusted_margin": 1.0,
            "enabled_by_human_calibration": False,
            "canonical_candidate_mean_log_probabilities": dict(scores),
            "reversed_candidate_mean_log_probabilities": dict(scores),
            "order_averaged_mean_log_probabilities": dict(scores),
            "raw_margin": 1.0,
            "adjusted_scores": dict(scores),
            "prompt_sha256": "1" * 64,
            "reversed_prompt_sha256": "2" * 64,
            "input_tokens_max": 100,
            "input_truncated": False,
        }
    if failed:
        fields = {
            field: {
                "structurally_applicable": True,
                "accepted": False,
                "acceptance_reason": "invalid",
                "input_truncated": False,
            }
            for field in flan_w17.W17_FIELDS
        }
    return {
        "record_version": "flan-w17-terminal-v2",
        "assignment_id": normalized["assignment_id"],
        "assignment_input_sha256": normalized["assignment_input_sha256"],
        "article_id": normalized["article_id"],
        "forecast_date": normalized["forecast_date"],
        "sector": normalized["target"]["sector"],
        "stock": normalized["target"]["ticker"],
        "benchmark": normalized["target"]["benchmark"],
        "role": normalized["role"],
        "source_profile": normalized["source_profile"],
        "headline_sha256": normalized["headline_sha256"],
        "description_sha256": normalized["description_sha256"],
        "description_available": normalized["description_available"],
        "text_sha256": normalized["text_sha256"],
        "extractor_id": flan_w17.EXTRACTOR_ID,
        "model_id": flan_w17.MODEL_ID,
        "model_revision": flan_w17.MODEL_REVISION,
        "contract_id": flan_w17.CONTRACT_ID,
        "acceptance_config_sha256": flan_w17.sha256_json(config),
        "runtime_contract_sha256": "a" * 64,
        "preflight_manifest_sha256": "b" * 64,
        "shard_index": 0,
        "shard_count": 1,
        "terminal_state": "inference_failed" if failed else "complete",
        "fields": fields,
        "no_input_truncation": True,
    }


class FakeEngine:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def predict(self, value: dict) -> dict:
        self.calls.append(value["assignment_id"])
        fields = {}
        labels = {
            "shock_scope": "idiosyncratic",
            "event_family": "product_demand",
            "information_status": "confirmed",
        }
        for field in flan_w17.W17_FIELDS:
            score_labels = (
                flan_w17.PREDICTIVE_LABELS[field]
                + flan_w17.ABSTENTION_LABELS[field]
            )
            scores = {
                value: (0.0 if value == labels[field] else -1.0)
                for value in score_labels
            }
            fields[field] = {
                "structurally_applicable": True,
                "raw_label": labels[field],
                "calibrated_label": labels[field],
                "canonical_candidate_mean_log_probabilities": dict(scores),
                "reversed_candidate_mean_log_probabilities": dict(scores),
                "order_averaged_mean_log_probabilities": dict(scores),
                "raw_margin": 1.0,
                "adjusted_scores": dict(scores),
                "prompt_sha256": "3" * 64,
                "reversed_prompt_sha256": "4" * 64,
                "input_tokens_max": 100,
                "accepted": True,
                "acceptance_reason": "accepted",
                "quality_weight": 1.0,
                "adjusted_margin": 1.0,
                "enabled_by_human_calibration": False,
                "input_truncated": False,
            }
        return {
            "terminal_state": "complete",
            "fields": fields,
            "no_input_truncation": True,
        }


class FakeTokenizer:
    def __init__(self, count: int) -> None:
        self.count = count

    def __call__(self, _prompt: str, **_kwargs: object) -> dict:
        return {"input_ids": list(range(self.count))}


class FakeBatchTokenizer:
    def __call__(self, prompts: list[str], **_kwargs: object) -> dict:
        return {"input_ids": [[0] * min(500, len(value)) for value in prompts]}


class FlanW17Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.acceptance = flan_w17.default_acceptance_config()

    def _write_preflight(
        self,
        input_path: Path,
        output_path: Path,
    ) -> dict:
        with mock.patch.object(
            flan_w17,
            "_tokenizer_only",
            return_value=(FakeBatchTokenizer(), {"fake": True}),
        ):
            return flan_w17.preflight_corpus(
                input_path=input_path,
                output_path=output_path,
                prompt_batch_size=18,
            )

    def _write_corpus_manifest(
        self,
        *,
        assignments_path: Path,
        assignments: list[dict],
        stock_days_path: Path,
        stock_days: list[dict],
        manifest_path: Path,
    ) -> dict:
        value = {
            "manifest_version": "semantic-corpus-manifest-v1",
            "status": "complete_exploratory_retrospective",
            "audit": {
                "assignment_count": len(assignments),
                "stock_day_scope_rows": len(stock_days),
            },
            "generated_files": {
                assignments_path.name: {
                    "path": str(assignments_path.resolve()),
                    "sha256": flan_w17.sha256_file(assignments_path),
                    "rows": len(assignments),
                },
                stock_days_path.name: {
                    "path": str(stock_days_path.resolve()),
                    "sha256": flan_w17.sha256_file(stock_days_path),
                    "rows": len(stock_days),
                },
            },
            "claim_flags": {
                "point_in_time_version_safe": False,
                "effective_dated_entity_map": False,
                "complete_untickered_macro_coverage": False,
                "exploratory_construction_eligible": True,
            },
        }
        manifest_path.write_text(
            json.dumps(value, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest_path.with_suffix(".sha256").write_text(
            flan_w17.sha256_file(manifest_path) + "\n",
            encoding="ascii",
        )
        return value

    def test_checked_in_active_contract_and_exact_feature_contract(self) -> None:
        contract, _, _ = flan_w17.active_runner.load_active_contract()
        self.assertEqual(contract["extractor_id"], flan_w17.EXTRACTOR_ID)
        self.assertEqual(len(flan_w17.WLLM17_COLUMNS), 17)
        self.assertEqual(
            flan_w17.W17_FIELDS,
            ("shock_scope", "event_family", "information_status"),
        )
        flan_w17.validate_acceptance_config(self.acceptance)
        self.assertFalse(self.acceptance["human_calibrated"])
        self.assertFalse(self.acceptance["primary_training_eligible"])

    def test_assignment_hashes_are_verified_and_routing_is_not_regated(self) -> None:
        raw = assignment("a")
        normalized = flan_w17.normalize_assignment(raw)
        self.assertEqual(normalized["role"], "target_idiosyncratic")
        adapted = flan_w17.extractor_record(normalized)
        self.assertNotIn("vendor_tickers", adapted)
        self.assertNotIn("semantic_applicable", adapted)
        changed = copy.deepcopy(raw)
        changed["description"] = "changed"
        with self.assertRaisesRegex(ValueError, "description_sha256"):
            flan_w17.normalize_assignment(changed)

    def test_article_consistent_sharding_only_caches_identical_prompts(self) -> None:
        first_raw = assignment("shared-article", stock="AAA")
        second_raw = copy.deepcopy(first_raw)
        second_raw["assignment_id"] = "shared-article-second-target"
        second_raw["target"]["ticker"] = "BBB"
        second_raw["target"]["company"] = "BBB Company"
        second_raw["target"]["known_sector_peers"] = ["AAA", "CCC"]
        first = flan_w17.normalize_assignment(first_raw)
        second = flan_w17.normalize_assignment(second_raw)
        self.assertEqual(
            flan_w17.assignment_shard(first, 32),
            flan_w17.assignment_shard(second, 32),
        )
        schema = flan_w17._active_schema()
        prompts = {}
        for field in flan_w17.W17_FIELDS:
            prompts[field] = (
                flan_w17.coarse_runner.prompt_variants_for_preflight(
                    flan_w17.extractor_record(first),
                    schema,
                    field,
                    "zero_shot",
                ),
                flan_w17.coarse_runner.prompt_variants_for_preflight(
                    flan_w17.extractor_record(second),
                    schema,
                    field,
                    "zero_shot",
                ),
            )
        self.assertNotEqual(*prompts["shock_scope"])
        self.assertEqual(*prompts["event_family"])
        self.assertEqual(*prompts["information_status"])

    def test_prompt_overflow_is_terminal_and_never_truncated(self) -> None:
        normalized = flan_w17.normalize_assignment(assignment("too-long"))

        class OverflowEngine:
            def predict(self, value: dict) -> dict:
                flan_w17.prompt_token_counts(value, FakeTokenizer(513))
                raise AssertionError("unreachable")

        terminal = flan_w17.terminal_prediction(
            normalized,
            OverflowEngine(),
            self.acceptance,
            shard_index=0,
            shard_count=1,
            preflight_manifest_sha256="b" * 64,
        )
        self.assertEqual(terminal["terminal_state"], "prompt_too_long")
        self.assertEqual(terminal["failure_reason"], "truncated")
        self.assertTrue(terminal["no_input_truncation"])
        self.assertTrue(
            all(
                item["input_truncated"] is False
                for item in terminal["fields"].values()
            )
        )

    def test_acceptance_is_exact_silver_lock_and_excludes_abstentions(self) -> None:
        scope_scores = {
            "idiosyncratic": -1.0,
            "common": 1.0,
            "mixed": -0.5,
            "unclear": 0.0,
        }
        abstained = flan_w17.apply_acceptance(
            field="shock_scope",
            label="unclear",
            adjusted_scores=scope_scores,
            config=self.acceptance,
        )
        self.assertFalse(abstained["accepted"])
        self.assertEqual(abstained["acceptance_reason"], "unclear")

        accepted = flan_w17.apply_acceptance(
            field="shock_scope",
            label="common",
            adjusted_scores=scope_scores,
            config=self.acceptance,
        )
        self.assertTrue(accepted["accepted"])
        self.assertEqual(accepted["quality_weight"], 1.0)
        malicious = copy.deepcopy(self.acceptance)
        malicious["confirmatory_eligible"] = True
        with self.assertRaisesRegex(ValueError, "confirmatory"):
            flan_w17.validate_acceptance_config(malicious)

    def test_sharded_inference_is_resumable(self) -> None:
        records = [assignment(f"a-{index}") for index in range(8)]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "assignments.jsonl"
            output_path = root / "predictions.jsonl"
            preflight_path = root / "preflight.json"
            flan_w17.write_records(input_path, records)
            self._write_preflight(input_path, preflight_path)
            engine = FakeEngine()
            first = flan_w17.run_inference(
                input_path=input_path,
                output_path=output_path,
                acceptance=self.acceptance,
                engine=engine,
                shard_index=1,
                shard_count=3,
                preflight_manifest_path=preflight_path,
            )
            first_calls = list(engine.calls)
            second = flan_w17.run_inference(
                input_path=input_path,
                output_path=output_path,
                acceptance=self.acceptance,
                engine=engine,
                shard_index=1,
                shard_count=3,
                preflight_manifest_path=preflight_path,
            )
            self.assertEqual(engine.calls, first_calls)
            self.assertTrue(first["all_selected_assignments_terminal"])
            self.assertEqual(first["terminal_record_count"], len(first_calls))
            self.assertEqual(first, second)
            selected = {
                value["assignment_id"]
                for value in map(flan_w17.normalize_assignment, records)
                if flan_w17.assignment_shard(value, 3) == 1
            }
            self.assertEqual(set(first_calls), selected)

    def test_streaming_preflight_binds_input_and_runtime(self) -> None:
        records = [assignment("preflight-a"), assignment("preflight-b")]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "assignments.jsonl"
            output_path = root / "preflight.json"
            flan_w17.write_records(input_path, records)
            manifest = self._write_preflight(input_path, output_path)
            self.assertEqual(manifest["status"], "passed")
            self.assertEqual(manifest["assignment_count"], 2)
            self.assertEqual(manifest["prompt_count"], 18)
            self.assertEqual(manifest["violation_count"], 0)
            flan_w17.validate_preflight_manifest(output_path, input_path)

    def test_acceptance_mapping_key_order_is_irrelevant(self) -> None:
        reordered = copy.deepcopy(self.acceptance)
        reordered["fields"] = dict(
            reversed(list(reordered["fields"].items()))
        )
        for field in flan_w17.W17_FIELDS:
            reordered["fields"][field]["classes"] = dict(
                reversed(
                    list(reordered["fields"][field]["classes"].items())
                )
            )
        flan_w17.validate_acceptance_config(reordered)

    def test_daily_aggregation_uses_weights_and_exact_17_columns(self) -> None:
        first = assignment("a", weight=1.0)
        second = assignment("b", weight=3.0)
        predictions = [
            successful_prediction(
                first,
                {
                    "shock_scope": "idiosyncratic",
                    "event_family": "earnings_guidance",
                    "information_status": "confirmed",
                },
            ),
            successful_prediction(
                second,
                {
                    "shock_scope": "common",
                    "event_family": "macro_market",
                    "information_status": "anticipated",
                },
            ),
        ]
        frame = flan_w17.aggregate_daily_w17(
            assignments=[first, second],
            predictions=predictions,
            stock_days=[stock_day(2, expected_ids=["a", "b"])],
            acceptance=self.acceptance,
        )
        row = frame.iloc[0]
        self.assertTrue(row["semantic_row_complete"])
        self.assertEqual(tuple(frame.columns[-17:]), flan_w17.WLLM17_COLUMNS)
        self.assertAlmostEqual(row["wllm_scope_share_idiosyncratic"], 0.25)
        self.assertAlmostEqual(row["wllm_scope_share_common"], 0.75)
        self.assertAlmostEqual(
            row["wllm_scope_common_minus_idiosyncratic"], 0.5
        )
        self.assertAlmostEqual(row["wllm_event_share_macro_market"], 0.75)
        self.assertAlmostEqual(row["wllm_coverage_shock_scope"], 1.0)

    def test_empty_and_all_abstain_states_are_not_conflated(self) -> None:
        empty = flan_w17.aggregate_daily_w17(
            assignments=[],
            predictions=[],
            stock_days=[
                stock_day(
                    0,
                    forecast_date="2025-01-02",
                    expected_ids=[],
                )
            ],
            acceptance=self.acceptance,
        ).iloc[0]
        self.assertEqual(
            empty["wllm_observed_no_eligible_semantic_article"], 1.0
        )
        self.assertTrue(math.isnan(empty["wllm_coverage_shock_scope"]))

        raw = assignment("abstain", forecast_date="2025-01-03")
        prediction = successful_prediction(
            raw,
            {
                "shock_scope": None,
                "event_family": None,
                "information_status": None,
            },
        )
        abstained = flan_w17.aggregate_daily_w17(
            assignments=[raw],
            predictions=[prediction],
            stock_days=[
                stock_day(
                    1,
                    forecast_date="2025-01-03",
                    expected_ids=["abstain"],
                )
            ],
            acceptance=self.acceptance,
        ).iloc[0]
        self.assertEqual(
            abstained["wllm_observed_no_eligible_semantic_article"], 0.0
        )
        self.assertEqual(abstained["wllm_coverage_shock_scope"], 0.0)
        self.assertTrue(
            math.isnan(abstained["wllm_scope_share_idiosyncratic"])
        )

    def test_missing_or_failed_prediction_invalidates_whole_stock_day(self) -> None:
        raw = assignment("missing")
        missing = flan_w17.aggregate_daily_w17(
            assignments=[raw],
            predictions=[],
            stock_days=[stock_day(1, expected_ids=["missing"])],
            acceptance=self.acceptance,
        ).iloc[0]
        self.assertFalse(missing["semantic_row_complete"])
        self.assertEqual(
            missing["semantic_ineligibility_reason"], "missing_prediction"
        )
        self.assertTrue(
            np.isnan(missing[list(flan_w17.WLLM17_COLUMNS)].to_numpy(dtype=float)).all()
        )

        failed = successful_prediction(
            raw,
            {
                "shock_scope": "common",
                "event_family": "macro_market",
                "information_status": "confirmed",
            },
            failed=True,
        )
        failed_row = flan_w17.aggregate_daily_w17(
            assignments=[raw],
            predictions=[failed],
            stock_days=[stock_day(1, expected_ids=["missing"])],
            acceptance=self.acceptance,
        ).iloc[0]
        self.assertFalse(failed_row["semantic_row_complete"])
        self.assertEqual(
            failed_row["semantic_ineligibility_reason"],
            "terminal_inference_failed",
        )

    def test_source_incompleteness_invalidates_even_an_empty_day(self) -> None:
        frame = flan_w17.aggregate_daily_w17(
            assignments=[],
            predictions=[],
            stock_days=[
                stock_day(
                    0,
                    source_complete=False,
                    expected_ids=[],
                )
            ],
            acceptance=self.acceptance,
        )
        self.assertFalse(frame.iloc[0]["semantic_row_complete"])
        self.assertTrue(
            np.isnan(
                frame.iloc[0][list(flan_w17.WLLM17_COLUMNS)].to_numpy(dtype=float)
            ).all()
        )
        with self.assertRaisesRegex(ValueError, "cannot enter training"):
            flan_w17.validate_complete_daily_panel(frame)

    def test_nonfinite_or_incomplete_score_maps_fail_closed(self) -> None:
        raw = assignment("numeric")
        prediction = successful_prediction(
            raw,
            {
                "shock_scope": "common",
                "event_family": "macro_market",
                "information_status": "confirmed",
            },
        )
        prediction["fields"]["shock_scope"]["adjusted_scores"]["common"] = (
            float("nan")
        )
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            flan_w17.validate_terminal_record(prediction)
        prediction = successful_prediction(
            raw,
            {
                "shock_scope": "common",
                "event_family": "macro_market",
                "information_status": "confirmed",
            },
        )
        prediction["fields"]["shock_scope"]["adjusted_scores"].pop("unclear")
        with self.assertRaisesRegex(ValueError, "ontology"):
            flan_w17.validate_terminal_record(prediction)

    def test_terminal_labels_and_margins_are_bound_to_score_maps(self) -> None:
        raw = assignment("score-binding")
        labels = {
            "shock_scope": "idiosyncratic",
            "event_family": "macro_market",
            "information_status": "confirmed",
        }
        prediction = successful_prediction(raw, labels)
        prediction["fields"]["shock_scope"]["calibrated_label"] = "common"
        with self.assertRaisesRegex(ValueError, "adjusted-score argmax"):
            flan_w17.validate_terminal_record(prediction)

        prediction = successful_prediction(raw, labels)
        prediction["fields"]["shock_scope"][
            "order_averaged_mean_log_probabilities"
        ]["common"] = 2.0
        with self.assertRaisesRegex(ValueError, "order-averaged"):
            flan_w17.validate_terminal_record(prediction)

        prediction = successful_prediction(raw, labels)
        prediction["fields"]["shock_scope"]["raw_margin"] = 0.5
        with self.assertRaisesRegex(ValueError, "raw margin"):
            flan_w17.validate_terminal_record(prediction)

    def test_limit_is_smoke_only_and_cannot_enter_daily_aggregation(self) -> None:
        records = [assignment("smoke-a"), assignment("smoke-b")]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "assignments.jsonl"
            output_path = root / "predictions.jsonl"
            preflight_path = root / "preflight.json"
            flan_w17.write_records(input_path, records)
            self._write_preflight(input_path, preflight_path)
            manifest = flan_w17.run_inference(
                input_path=input_path,
                output_path=output_path,
                acceptance=self.acceptance,
                engine=FakeEngine(),
                shard_index=0,
                shard_count=1,
                limit=1,
                preflight_manifest_path=preflight_path,
            )
            self.assertEqual(manifest["status"], "smoke_complete")
            with self.assertRaisesRegex(ValueError, "non-smoke"):
                flan_w17.validate_complete_inference_manifests(
                    prediction_paths=[output_path],
                    assignments_path=input_path,
                    preflight_manifest_path=preflight_path,
                    acceptance=self.acceptance,
                )

    def test_append_tail_recovery_is_narrow_and_durable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.jsonl"
            path.write_bytes(b'{"complete":true}\n{"partial":')
            audit = flan_w17.recover_append_tail(path)
            self.assertEqual(audit["action"], "truncated_incomplete_tail")
            self.assertEqual(path.read_bytes(), b'{"complete":true}\n')
            path.write_bytes(b'{"complete":true}')
            audit = flan_w17.recover_append_tail(path)
            self.assertEqual(audit["action"], "terminated_valid_tail")
            self.assertEqual(path.read_bytes(), b'{"complete":true}\n')

    def test_immutable_aggregation_snapshot_rejects_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.jsonl"
            path.write_text('{"value":1}\n', encoding="utf-8")
            snapshot = flan_w17._file_snapshot([path])
            flan_w17._verify_file_snapshot(snapshot)
            path.write_text('{"value":2}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "input changed"):
                flan_w17._verify_file_snapshot(snapshot)

    def test_systemic_inference_failures_abort_the_shard(self) -> None:
        class BrokenEngine:
            def predict(self, _value: dict) -> dict:
                raise RuntimeError("GPU unavailable")

        records = [assignment(f"broken-{index}") for index in range(4)]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "assignments.jsonl"
            output_path = root / "predictions.jsonl"
            preflight_path = root / "preflight.json"
            flan_w17.write_records(input_path, records)
            self._write_preflight(input_path, preflight_path)
            with self.assertRaisesRegex(RuntimeError, "consecutive"):
                flan_w17.run_inference(
                    input_path=input_path,
                    output_path=output_path,
                    acceptance=self.acceptance,
                    engine=BrokenEngine(),
                    shard_index=0,
                    shard_count=1,
                    preflight_manifest_path=preflight_path,
                    max_consecutive_failures=2,
                )
            manifest = json.loads(
                output_path.with_suffix(".jsonl.manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["status"], "aborted_systemic_failures")
            self.assertEqual(manifest["terminal_record_count"], 2)

    def test_authoritative_disk_backed_daily_construction(self) -> None:
        records = [
            assignment("daily-a", weight=1.0),
            assignment("daily-b", weight=2.0),
        ]
        scopes = [stock_day(2, expected_ids=["daily-a", "daily-b"])]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            assignments_path = root / "assignments.jsonl"
            stock_days_path = root / "stock_days.jsonl"
            corpus_manifest_path = root / "manifest.json"
            preflight_path = root / "preflight.json"
            predictions_path = root / "predictions.jsonl"
            output_path = root / "daily.parquet"
            flan_w17.write_records(assignments_path, records)
            flan_w17.write_records(stock_days_path, scopes)
            self._write_corpus_manifest(
                assignments_path=assignments_path,
                assignments=records,
                stock_days_path=stock_days_path,
                stock_days=scopes,
                manifest_path=corpus_manifest_path,
            )
            self._write_preflight(assignments_path, preflight_path)
            inference = flan_w17.run_inference(
                input_path=assignments_path,
                output_path=predictions_path,
                acceptance=self.acceptance,
                engine=FakeEngine(),
                shard_index=0,
                shard_count=1,
                preflight_manifest_path=preflight_path,
            )
            self.assertEqual(inference["status"], "complete")
            manifest = flan_w17.write_complete_daily_panel(
                assignments_path=assignments_path,
                prediction_paths=[predictions_path],
                stock_days_path=stock_days_path,
                corpus_manifest_path=corpus_manifest_path,
                preflight_manifest_path=preflight_path,
                output_path=output_path,
                acceptance=self.acceptance,
                work_root=root / "work",
            )
            self.assertEqual(
                manifest["status"],
                "complete_exploratory_silver_fit",
            )
            self.assertEqual(manifest["daily_rows"], 1)
            self.assertFalse(manifest["primary_training_eligible"])
            self.assertIn("disk-backed SQLite", manifest["join_memory_contract"])
            self.assertTrue(output_path.is_file())


if __name__ == "__main__":
    unittest.main()
