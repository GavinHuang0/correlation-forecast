from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.semantic_news_v2 import gpt_r70


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def assignment(
    assignment_id: str = "assignment-1",
    forecast_date: str = "2025-01-02",
) -> dict:
    text = (
        "AAA and peer BBB reported stronger demand in the Semiconductors "
        "sector. The confirmed surprise lifted both companies."
    )
    return {
        "assignment_id": assignment_id,
        "provider_article_id": "article-1",
        "forecast_date": forecast_date,
        "cutoff_utc": f"{forecast_date}T14:00:00Z",
        "published_at_utc": f"{forecast_date}T12:00:00Z",
        "target_ticker": "AAA",
        "target_company": "AAA",
        "sector": "Semiconductors",
        "benchmark": "SYN",
        "known_sector_peers": ["BBB", "CCC"],
        "detected_entities": ["AAA", "BBB"],
        "role_direct": True,
        "role_target_idio": False,
        "role_target_common": True,
        "role_peer_idio": False,
        "role_sector_common": True,
        "role_macro_common": False,
        "role_any_common": True,
        "model_text": text,
        "model_text_sha256": gpt_r70.sha256_text(text),
        "source_query_complete": True,
        "candidate_assignment_complete": True,
        "recency_weight": math.exp(-math.log(2) * 2 / 12),
        "duplicate_group_size": 1,
    }


def fine_prediction() -> dict:
    return {
        "relevance": "direct_target",
        "event_scope": "sector_wide",
        "event_type": "demand_customer_contract",
        "affected_breadth": "several_same_sector",
        "target_direction": "positive",
        "sector_direction": "positive",
        "peer_effect": "same_direction",
        "explicit_surprise": "positive",
        "information_status": "confirmed",
        "transmission_channels": ["demand", "pricing_margin"],
        "affected_companies": ["AAA", "BBB"],
        "affected_sectors": ["Semiconductors"],
        "evidence": {
            "scope": "AAA and peer BBB",
            "direction": "lifted both companies",
            "surprise": "confirmed surprise",
        },
        "abstain_reason": None,
    }


def valid_view_record(view: str) -> dict:
    normalized = gpt_r70.normalize_assignment(assignment())
    return {
        "assignment_id": "assignment-1",
        "view": view,
        "valid": True,
        "terminal_state": "valid_prediction",
        "model_text_sha256": normalized["model_text_sha256"],
        "assignment_input_sha256": normalized["assignment_input_sha256"],
        "labels": {
            key: value
            for key, value in fine_prediction().items()
            if key in gpt_r70.SINGLE_FIELD_CLASSES
        },
        "transmission_channels": ["demand", "pricing_margin"],
        "affected_companies": ["AAA", "BBB"],
        "affected_sectors": ["Semiconductors"],
        "evidence_present": {
            "scope": True,
            "direction": True,
            "surprise": True,
        },
        "evidence_sha256": {
            "scope": "scope-hash",
            "direction": "direction-hash",
            "surprise": "surprise-hash",
        },
        "response_id": f"response-{view}",
    }


def batch_response(custom_id: str) -> dict:
    return {
        "id": f"batch-request-{custom_id}",
        "custom_id": custom_id,
        "response": {
            "status_code": 200,
            "request_id": f"request-{custom_id}",
            "body": {
                "id": f"response-{custom_id}",
                "object": "response",
                "status": "completed",
                "model": "gpt-5.6-sol-2026-07-01",
                "system_fingerprint": "fp_test",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps(fine_prediction()),
                            }
                        ],
                    }
                ],
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "total_tokens": 150,
                },
            },
        },
        "error": None,
    }


class GPTR70Tests(unittest.TestCase):
    def test_cli_defaults_use_the_canonical_jsonl_assignment_ledger(self) -> None:
        self.assertEqual(
            gpt_r70.DEFAULT_ASSIGNMENTS.name,
            "article_target_assignments.jsonl.gz",
        )

    def test_contract_has_exact_seventy_columns(self) -> None:
        contract = gpt_r70.contract_manifest()
        self.assertEqual(len(gpt_r70.RLLM70_FEATURES), 70)
        self.assertEqual(len(set(gpt_r70.RLLM70_FEATURES)), 70)
        self.assertEqual(contract["raw_feature_count"], 70)
        self.assertEqual(contract["effective_uncalibrated_feature_count"], 69)
        response_schema = contract["structured_output_schema"]
        self.assertFalse(response_schema["additionalProperties"])
        self.assertEqual(
            set(response_schema["required"]),
            set(response_schema["properties"]),
        )

    def test_preparation_creates_two_tool_free_responses_requests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            assignments = root / "assignments.jsonl"
            write_jsonl(assignments, [assignment()])
            preflight = root / "preflight.json"
            gpt_r70.preflight_assignments(
                assignments_path=assignments,
                output_path=preflight,
            )
            output = root / "batches"
            manifest = gpt_r70.prepare_batch_files(
                assignments_path=assignments,
                output_root=output,
                max_requests_per_file=10,
                preflight_path=preflight,
            )
            self.assertEqual(manifest["assignment_count"], 1)
            self.assertEqual(manifest["request_count"], 2)
            self.assertEqual(manifest["status"], "prepared_not_submitted")
            requests = list(gpt_r70.read_jsonl(output / "batch_0001.jsonl"))
            self.assertEqual(len(requests), 2)
            self.assertEqual(
                {row["body"]["model"] for row in requests},
                {"gpt-5.6-sol"},
            )
            self.assertEqual({row["url"] for row in requests}, {"/v1/responses"})
            self.assertEqual(
                {row["body"]["reasoning"]["effort"] for row in requests},
                {"high"},
            )
            self.assertTrue(all("tools" not in row["body"] for row in requests))
            self.assertTrue(
                all(
                    row["body"]["text"]["format"]["type"] == "json_schema"
                    and row["body"]["text"]["format"]["strict"]
                    for row in requests
                )
            )
            self.assertNotEqual(
                requests[0]["body"]["input"][0]["content"],
                requests[1]["body"]["input"][0]["content"],
            )
            self.assertNotIn(
                "deterministic_assignment_roles",
                requests[0]["body"]["input"][1]["content"],
            )

    def test_candidate_routing_is_not_exposed_as_visible_gpt_metadata(
        self,
    ) -> None:
        raw = assignment()
        raw["candidate_roles"] = dict(raw)
        raw["candidate_roles"] = {
            "direct": True,
            "target_idio": True,
            "target_common": False,
            "peer_idio": False,
            "sector_common": False,
            "macro_common": False,
            "any_common": False,
        }
        raw.update(
            {
                "role_direct": False,
                "role_target_idio": False,
                "role_target_common": False,
                "role_peer_idio": False,
                "role_sector_common": False,
                "role_macro_common": False,
                "role_any_common": False,
                "detected_entities": [],
            }
        )
        normalized = gpt_r70.normalize_assignment(raw)
        self.assertTrue(normalized["candidate_roles"]["direct"])
        self.assertFalse(normalized["roles"]["direct"])
        rendered = gpt_r70.render_user_input(normalized)
        self.assertNotIn("candidate_roles", rendered)
        self.assertNotIn("deterministic_assignment_roles", rendered)

    def test_merge_validates_both_views_and_strips_quoted_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            assignments = root / "assignments.jsonl"
            write_jsonl(assignments, [assignment()])
            preflight = root / "preflight.json"
            gpt_r70.preflight_assignments(
                assignments_path=assignments,
                output_path=preflight,
            )
            batch_root = root / "batches"
            gpt_r70.prepare_batch_files(
                assignments_path=assignments,
                output_root=batch_root,
                preflight_path=preflight,
            )
            index = list(gpt_r70.read_jsonl(batch_root / "request_index.jsonl"))
            downloaded = root / "downloaded.jsonl"
            write_jsonl(
                downloaded,
                [batch_response(row["custom_id"]) for row in index],
            )
            submission = root / "submission.json"
            gpt_r70.write_json(
                submission,
                {
                    "status": "complete_downloaded",
                    "batch_manifest_path": str(
                        (batch_root / "manifest.json").resolve()
                    ),
                    "batch_manifest_sha256": gpt_r70.sha256_file(
                        batch_root / "manifest.json"
                    ),
                    "model": gpt_r70.MODEL_ID,
                    "submitted_shards": [
                        {
                            "batch_file": "batch_0001.jsonl",
                            "input_file_id": "file-input",
                            "batch_id": "batch-one",
                            "submission_state": "submitted",
                            "status": "completed",
                            "output_file_id": "file-output",
                            "output_file": {
                                "path": str(downloaded.resolve()),
                                "sha256": gpt_r70.sha256_file(downloaded),
                                "bytes": downloaded.stat().st_size,
                            },
                        }
                    ],
                },
            )
            output = root / "views.jsonl"
            result = gpt_r70.merge_batch_outputs(
                manifest_path=batch_root / "manifest.json",
                submission_path=submission,
                assignments_path=assignments,
                output_paths=[downloaded],
                output_path=output,
            )
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["valid_prediction_count"], 2)
            records = list(gpt_r70.read_jsonl(output))
            self.assertTrue(all(row["valid"] for row in records))
            serialized = output.read_text(encoding="utf-8")
            self.assertNotIn("lifted both companies", serialized)
            self.assertNotIn('"evidence":', serialized)
            self.assertIn('"evidence_sha256"', serialized)

    def test_retry_preparation_and_merge_resolve_one_failed_view(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            assignments = root / "assignments.jsonl"
            write_jsonl(assignments, [assignment()])
            preflight = root / "preflight.json"
            gpt_r70.preflight_assignments(
                assignments_path=assignments,
                output_path=preflight,
            )
            batch_root = root / "batches"
            gpt_r70.prepare_batch_files(
                assignments_path=assignments,
                output_root=batch_root,
                preflight_path=preflight,
            )
            index = list(gpt_r70.read_jsonl(batch_root / "request_index.jsonl"))
            initial_output = root / "initial.output.jsonl"
            write_jsonl(
                initial_output,
                [
                    batch_response(index[0]["custom_id"]),
                    {
                        "custom_id": index[1]["custom_id"],
                        "response": None,
                        "error": {"code": "server_error"},
                    },
                ],
            )
            initial_submission = root / "initial.submission.json"
            gpt_r70.write_json(
                initial_submission,
                {
                    "status": "complete_downloaded",
                    "batch_manifest_path": str(
                        (batch_root / "manifest.json").resolve()
                    ),
                    "batch_manifest_sha256": gpt_r70.sha256_file(
                        batch_root / "manifest.json"
                    ),
                    "model": gpt_r70.MODEL_ID,
                    "submitted_shards": [
                        {
                            "batch_file": "batch_0001.jsonl",
                            "input_file_id": "file-initial-input",
                            "batch_id": "batch-initial",
                            "submission_state": "submitted",
                            "status": "completed",
                            "output_file_id": "file-initial-output",
                            "error_file_id": None,
                            "output_file": {
                                "path": str(initial_output.resolve()),
                                "sha256": gpt_r70.sha256_file(initial_output),
                                "bytes": initial_output.stat().st_size,
                            },
                        }
                    ],
                },
            )
            initial_views = root / "initial.views.jsonl"
            initial_merge = gpt_r70.merge_batch_outputs(
                manifest_path=batch_root / "manifest.json",
                submission_path=initial_submission,
                assignments_path=assignments,
                output_paths=[initial_output],
                output_path=initial_views,
                work_root=root / "work",
            )
            self.assertEqual(
                initial_merge["status"], "incomplete_fail_closed"
            )

            retry_root = root / "retry_01"
            retry_manifest = gpt_r70.prepare_retry_batch_files(
                base_manifest_path=batch_root / "manifest.json",
                parent_views_path=initial_views,
                parent_merge_manifest_path=initial_views.with_suffix(
                    ".manifest.json"
                ),
                output_root=retry_root,
                work_root=root / "work",
            )
            self.assertEqual(retry_manifest["request_count"], 1)
            retry_index = list(
                gpt_r70.read_jsonl(retry_root / "request_index.jsonl")
            )
            self.assertEqual(
                retry_index[0]["logical_custom_id"],
                index[1]["custom_id"],
            )
            self.assertTrue(retry_index[0]["custom_id"].endswith("-r01"))
            retry_output = root / "retry.output.jsonl"
            write_jsonl(
                retry_output,
                [batch_response(retry_index[0]["custom_id"])],
            )
            retry_submission = root / "retry.submission.json"
            gpt_r70.write_json(
                retry_submission,
                {
                    "status": "complete_downloaded",
                    "batch_manifest_path": str(
                        (retry_root / "manifest.json").resolve()
                    ),
                    "batch_manifest_sha256": gpt_r70.sha256_file(
                        retry_root / "manifest.json"
                    ),
                    "model": gpt_r70.MODEL_ID,
                    "submitted_shards": [
                        {
                            "batch_file": "batch_0001.jsonl",
                            "input_file_id": "file-retry-input",
                            "batch_id": "batch-retry",
                            "submission_state": "submitted",
                            "status": "completed",
                            "output_file_id": "file-retry-output",
                            "error_file_id": None,
                            "output_file": {
                                "path": str(retry_output.resolve()),
                                "sha256": gpt_r70.sha256_file(retry_output),
                                "bytes": retry_output.stat().st_size,
                            },
                        }
                    ],
                },
            )
            final_views = root / "final.views.jsonl"
            final_merge = gpt_r70.merge_batch_outputs(
                manifest_path=batch_root / "manifest.json",
                submission_path=initial_submission,
                assignments_path=assignments,
                output_paths=[initial_output],
                retry_submission_paths=[retry_submission],
                output_path=final_views,
                work_root=root / "work",
            )
            self.assertEqual(final_merge["status"], "complete")
            self.assertEqual(final_merge["valid_prediction_count"], 2)
            rows = list(gpt_r70.read_jsonl(final_views))
            retried = next(
                row for row in rows if row["attempt_count"] == 2
            )
            self.assertEqual(retried["valid_attempt_count"], 1)
            self.assertTrue(
                Path(final_merge["attempt_ledger"]["path"]).is_file()
            )

    def test_prediction_rejects_ungrounded_evidence(self) -> None:
        raw = fine_prediction()
        raw["evidence"]["scope"] = "not in the supplied article"
        with self.assertRaisesRegex(
            gpt_r70.SemanticConstructionError,
            "not an exact source substring",
        ):
            gpt_r70.validate_fine_prediction(
                raw,
                assignment=gpt_r70.normalize_assignment(assignment()),
                fine_schema=gpt_r70.load_fine_schema(),
            )

    def test_timing_contract_requires_exact_nine_am_new_york_cutoff(self) -> None:
        winter = assignment()
        gpt_r70.normalize_assignment(winter)

        summer = assignment(forecast_date="2025-07-01")
        summer["cutoff_utc"] = "2025-07-01T13:00:00Z"
        summer["published_at_utc"] = "2025-07-01T11:00:00Z"
        summer["recency_weight"] = math.exp(-math.log(2) * 2 / 12)
        gpt_r70.normalize_assignment(summer)

        late = dict(summer)
        late["cutoff_utc"] = "2025-07-01T14:00:00Z"
        with self.assertRaisesRegex(
            gpt_r70.SemanticConstructionError, "exactly 09:00:00"
        ):
            gpt_r70.normalize_assignment(late)

        at_cutoff = dict(summer)
        at_cutoff["published_at_utc"] = at_cutoff["cutoff_utc"]
        with self.assertRaisesRegex(
            gpt_r70.SemanticConstructionError, "strictly before"
        ):
            gpt_r70.normalize_assignment(at_cutoff)

    def test_consensus_adjudication_and_r70_aggregation(self) -> None:
        assignments = [assignment()]
        views = [
            valid_view_record("taxonomy_first"),
            valid_view_record("evidence_first"),
        ]
        adjudicated = gpt_r70.adjudicate_views(
            assignments=assignments,
            view_records=views,
        )
        self.assertEqual(len(adjudicated), 1)
        self.assertTrue(adjudicated[0]["valid"])
        self.assertTrue(
            adjudicated[0]["fields"]["event_scope"]["accepted_predictive"]
        )
        universe = [
            {
                "forecast_date": "2025-01-02",
                "sector": "Semiconductors",
                "stock": "AAA",
                "benchmark": "SYN",
                "source_query_complete": True,
                "candidate_assignment_complete": True,
                "expected_assignment_count": 1,
                "expected_assignment_ids_sha256": (
                    gpt_r70._assignment_ids_sha256(["assignment-1"])
                ),
            },
            {
                "forecast_date": "2025-01-03",
                "sector": "Semiconductors",
                "stock": "AAA",
                "benchmark": "SYN",
                "source_query_complete": True,
                "candidate_assignment_complete": True,
                "expected_assignment_count": 0,
                "expected_assignment_ids_sha256": (
                    gpt_r70._assignment_ids_sha256([])
                ),
            },
            {
                "forecast_date": "2025-01-04",
                "sector": "Semiconductors",
                "stock": "AAA",
                "benchmark": "SYN",
                "source_query_complete": False,
                "candidate_assignment_complete": True,
                "expected_assignment_count": 0,
                "expected_assignment_ids_sha256": (
                    gpt_r70._assignment_ids_sha256([])
                ),
            },
        ]
        daily = gpt_r70.aggregate_daily_r70(
            row_universe=universe,
            assignments=assignments,
            adjudicated=adjudicated,
        )
        first = daily[0]
        self.assertEqual(first["rllm_relevance_share_direct_target"], 1.0)
        self.assertEqual(first["rllm_scope_share_sector_wide"], 1.0)
        self.assertEqual(
            first["rllm_scope_common_minus_idiosyncratic"], 1.0
        )
        self.assertEqual(
            first["rllm_target_sector_same_direction_share"], 1.0
        )
        self.assertEqual(
            first["rllm_target_sector_opposite_direction_share"], 0.0
        )
        self.assertEqual(
            first["rllm_channel_accepted_claim_share_demand"], 0.5
        )
        self.assertEqual(first["rllm_mean_accepted_quality_weight"], 1.0)
        self.assertEqual(
            first["rllm_observed_no_eligible_semantic_article"], 0
        )
        second = daily[1]
        self.assertEqual(
            second["rllm_observed_no_eligible_semantic_article"], 1
        )
        self.assertIsNone(second["rllm_relevance_share_direct_target"])
        third = daily[2]
        self.assertEqual(third["rllm_row_eligible"], 0)
        self.assertIsNone(
            third["rllm_observed_no_eligible_semantic_article"]
        )
        for row in daily:
            self.assertTrue(set(gpt_r70.RLLM70_FEATURES).issubset(row))

    def test_disagreement_abstains_only_the_disputed_field(self) -> None:
        left = valid_view_record("taxonomy_first")
        right = valid_view_record("evidence_first")
        right["labels"]["event_type"] = "guidance"
        result = gpt_r70.adjudicate_views(
            assignments=[assignment()],
            view_records=[left, right],
        )[0]
        self.assertFalse(
            result["fields"]["event_type"]["accepted_predictive"]
        )
        self.assertEqual(
            result["fields"]["event_type"]["reason"], "view_disagreement"
        )
        self.assertTrue(
            result["fields"]["information_status"]["accepted_predictive"]
        )

    def test_unclear_transmission_channel_is_exclusive(self) -> None:
        prediction = fine_prediction()
        prediction["transmission_channels"] = ["demand", "unclear"]
        with self.assertRaisesRegex(
            gpt_r70.SemanticConstructionError, "must be exclusive"
        ):
            gpt_r70.validate_fine_prediction(
                prediction,
                assignment=assignment(),
                fine_schema=gpt_r70.load_fine_schema(
                    ROOT / "config" / "news_feature_schema.json"
                ),
            )

        left = valid_view_record("taxonomy_first")
        right = valid_view_record("evidence_first")
        left["transmission_channels"] = ["demand", "unclear"]
        right["transmission_channels"] = ["demand"]
        with self.assertRaisesRegex(
            gpt_r70.SemanticConstructionError,
            "mixes 'unclear' with predictive channels",
        ):
            gpt_r70.adjudicate_views(
                assignments=[assignment()],
                view_records=[left, right],
            )

    def test_disk_backed_adjudication_and_daily_join(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            assignments_path = root / "assignments.jsonl"
            views_path = root / "views.jsonl"
            adjudicated_path = root / "adjudicated.jsonl"
            universe_path = root / "universe.jsonl"
            universe_manifest_path = root / "universe.manifest.json"
            daily_path = root / "daily.jsonl"
            write_jsonl(assignments_path, [assignment()])
            write_jsonl(
                views_path,
                [
                    valid_view_record("taxonomy_first"),
                    valid_view_record("evidence_first"),
                ],
            )
            views_manifest = root / "views.manifest.json"
            gpt_r70.write_json(
                views_manifest,
                {
                    "status": "complete",
                    "inputs": {
                        "assignments": {
                            "sha256": gpt_r70.sha256_file(assignments_path)
                        }
                    },
                    "output": {
                        "sha256": gpt_r70.sha256_file(views_path)
                    },
                },
            )
            adjudication = gpt_r70.adjudicate_views_file(
                assignments_path=assignments_path,
                views_path=views_path,
                views_manifest_path=views_manifest,
                output_path=adjudicated_path,
            )
            self.assertEqual(adjudication["status"], "complete")
            write_jsonl(
                universe_path,
                [
                    {
                        "forecast_date": "2025-01-02",
                        "sector": "Semiconductors",
                        "stock": "AAA",
                        "benchmark": "SYN",
                        "source_query_complete": True,
                        "candidate_assignment_complete": True,
                        "expected_assignment_count": 1,
                        "expected_assignment_ids_sha256": (
                            gpt_r70._assignment_ids_sha256(["assignment-1"])
                        ),
                    }
                ],
            )
            gpt_r70.write_json(
                universe_manifest_path,
                {
                    "manifest_version": "semantic-corpus-manifest-v1",
                    "status": "complete_exploratory_retrospective",
                    "audit": {"stock_day_scope_rows": 1},
                    "generated_files": {
                        universe_path.name: {
                            "sha256": gpt_r70.sha256_file(universe_path),
                            "rows": 1,
                        }
                    },
                },
            )
            aggregation = gpt_r70.aggregate_daily_r70_files(
                row_universe_path=universe_path,
                row_universe_manifest_path=universe_manifest_path,
                assignments_path=assignments_path,
                adjudicated_path=adjudicated_path,
                adjudication_manifest_path=(
                    adjudicated_path.with_suffix(".manifest.json")
                ),
                output_path=daily_path,
            )
            self.assertEqual(aggregation["status"], "complete")
            self.assertEqual(aggregation["row_count"], 1)
            row = next(gpt_r70.read_jsonl(daily_path))
            self.assertEqual(row["rllm_scope_share_sector_wide"], 1.0)

    def test_paid_submission_is_explicitly_gated(self) -> None:
        fake_client = SimpleNamespace(files=SimpleNamespace(), batches=SimpleNamespace())
        with self.assertRaises(PermissionError):
            gpt_r70.submit_prepared_batches(
                manifest_path=Path("unused.json"),
                client=fake_client,
                confirm_paid_submission=False,
                confirm_licensed_text_processing=False,
            )

    def test_submission_rejects_stale_runtime_before_upload(self) -> None:
        class FakeFiles:
            called = False

            def create(self, **_kwargs):
                self.called = True
                raise AssertionError("stale preparation reached upload")

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            assignments = root / "assignments.jsonl"
            write_jsonl(assignments, [assignment()])
            preflight = root / "preflight.json"
            gpt_r70.preflight_assignments(
                assignments_path=assignments,
                output_path=preflight,
            )
            batches = root / "batches"
            gpt_r70.prepare_batch_files(
                assignments_path=assignments,
                output_root=batches,
                preflight_path=preflight,
            )
            manifest_path = batches / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["runtime_contract_sha256"] = "0" * 64
            gpt_r70.write_json(manifest_path, manifest)
            files = FakeFiles()
            with self.assertRaisesRegex(
                gpt_r70.SemanticConstructionError, "runtime contract"
            ):
                gpt_r70.submit_prepared_batches(
                    manifest_path=manifest_path,
                    client=SimpleNamespace(
                        files=files, batches=SimpleNamespace()
                    ),
                    confirm_paid_submission=True,
                    confirm_licensed_text_processing=True,
                    max_paid_requests=2,
                )
            self.assertFalse(files.called)

    def test_sqlite_work_files_cannot_use_trackable_repo_paths(self) -> None:
        forbidden = ROOT / "experiments" / "unsafe_semantic_work"
        with self.assertRaisesRegex(
            gpt_r70.SemanticConstructionError,
            "must remain under data",
        ):
            gpt_r70._private_work_database(
                "merge", ROOT / "experiments" / "views.jsonl", forbidden
            )
        self.assertFalse(forbidden.exists())

    def test_daily_aggregation_rejects_truncated_authoritative_universe(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            universe = root / "stock_day_scope.jsonl"
            write_jsonl(
                universe,
                [
                    {
                        "forecast_date": "2025-01-02",
                        "sector": "Semiconductors",
                        "stock": "AAA",
                        "benchmark": "SYN",
                    }
                ],
            )
            manifest = root / "manifest.json"
            gpt_r70.write_json(
                manifest,
                {
                    "manifest_version": "semantic-corpus-manifest-v1",
                    "status": "complete_exploratory_retrospective",
                    "audit": {"stock_day_scope_rows": 1},
                    "generated_files": {
                        universe.name: {
                            "sha256": gpt_r70.sha256_file(universe),
                            "rows": 1,
                        }
                    },
                },
            )
            universe.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(
                gpt_r70.SemanticConstructionError, "authoritative"
            ):
                gpt_r70.aggregate_daily_r70_files(
                    row_universe_path=universe,
                    row_universe_manifest_path=manifest,
                    assignments_path=root / "unused-assignments.jsonl",
                    adjudicated_path=root / "unused-adjudicated.jsonl",
                    adjudication_manifest_path=root / "unused-manifest.json",
                    output_path=root / "daily.jsonl",
                )

    def test_completed_all_error_batch_can_merge_and_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            assignments = root / "assignments.jsonl"
            write_jsonl(assignments, [assignment()])
            preflight = root / "preflight.json"
            gpt_r70.preflight_assignments(
                assignments_path=assignments,
                output_path=preflight,
            )
            batches = root / "batches"
            gpt_r70.prepare_batch_files(
                assignments_path=assignments,
                output_root=batches,
                preflight_path=preflight,
            )
            error_file = root / "batch.error.jsonl"
            error_file.write_text('{"error":"all requests failed"}\n')
            submission = root / "submission.json"
            gpt_r70.write_json(
                submission,
                {
                    "status": "complete_downloaded",
                    "batch_manifest_path": str(
                        (batches / "manifest.json").resolve()
                    ),
                    "batch_manifest_sha256": gpt_r70.sha256_file(
                        batches / "manifest.json"
                    ),
                    "model": gpt_r70.MODEL_ID,
                    "submitted_shards": [
                        {
                            "batch_file": "batch_0001.jsonl",
                            "input_file_id": "input",
                            "batch_id": "batch",
                            "submission_state": "submitted",
                            "status": "completed",
                            "output_file_id": None,
                            "error_file_id": "errors",
                            "error_file": {
                                "path": str(error_file.resolve()),
                                "sha256": gpt_r70.sha256_file(error_file),
                                "bytes": error_file.stat().st_size,
                            },
                        }
                    ],
                },
            )
            views = root / "views.jsonl"
            result = gpt_r70.merge_batch_outputs(
                manifest_path=batches / "manifest.json",
                submission_path=submission,
                assignments_path=assignments,
                output_paths=[],
                output_path=views,
                work_root=root / "work",
            )
            self.assertEqual(result["status"], "incomplete_fail_closed")
            self.assertEqual(result["invalid_or_missing_count"], 2)
            retry = gpt_r70.prepare_retry_batch_files(
                base_manifest_path=batches / "manifest.json",
                parent_views_path=views,
                parent_merge_manifest_path=views.with_suffix(
                    ".manifest.json"
                ),
                output_root=root / "retry",
                work_root=root / "work",
            )
            self.assertEqual(retry["request_count"], 2)

    def test_remote_cleanup_resumes_and_accepts_verified_404(self) -> None:
        class NotFoundError(RuntimeError):
            status_code = 404

        class FakeFiles:
            def __init__(self):
                self.calls = []

            def delete(self, file_id):
                self.calls.append(file_id)
                if file_id == "input" and self.calls.count(file_id) == 1:
                    raise NotFoundError("already absent")
                return SimpleNamespace(deleted=True)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output = root / "output.jsonl"
            output.write_text("{}\n", encoding="utf-8")
            submission = root / "submission.json"
            gpt_r70.write_json(
                submission,
                {
                    "status": "partial_complete_downloaded",
                    "submitted_shards": [
                        {
                            "status": "completed",
                            "input_file_id": "input",
                            "output_file_id": "output",
                            "error_file_id": None,
                            "output_file": {
                                "path": str(output.resolve()),
                                "remote_file_id": "output",
                                "sha256": gpt_r70.sha256_file(output),
                                "bytes": output.stat().st_size,
                            },
                        }
                    ],
                },
            )
            cleanup_path = root / "cleanup.json"
            result = gpt_r70.cleanup_remote_files(
                submission_path=submission,
                client=SimpleNamespace(files=FakeFiles()),
                confirm_delete_remote_files=True,
                cleanup_path=cleanup_path,
            )
            self.assertEqual(result["status"], "complete")
            self.assertEqual(
                result["files"]["input"]["confirmation"],
                "already_absent_404",
            )
            self.assertEqual(
                result["files"]["output"]["confirmation"], "api_confirmed"
            )
            self.assertNotIn(
                "remote_file_cleanup",
                json.loads(submission.read_text(encoding="utf-8")),
            )

    def test_submission_is_resumable_and_collection_downloads_output(self) -> None:
        class FakeFiles:
            def create(self, *, file, purpose, expires_after):
                self.uploaded = file.read()
                self.purpose = purpose
                self.expires_after = expires_after
                return SimpleNamespace(id="file-input")

            def content(self, file_id):
                self.requested_file_id = file_id
                return SimpleNamespace(content=b'{"custom_id":"one"}\n')

        class FakeBatches:
            def create(self, **_kwargs):
                return SimpleNamespace(id="batch-one", status="validating")

            def retrieve(self, batch_id):
                self.retrieved = batch_id
                return SimpleNamespace(
                    status="completed",
                    output_file_id="file-output",
                    error_file_id=None,
                )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            assignments = root / "assignments.jsonl"
            write_jsonl(assignments, [assignment()])
            preflight = root / "preflight.json"
            gpt_r70.preflight_assignments(
                assignments_path=assignments,
                output_path=preflight,
            )
            batches = root / "batches"
            gpt_r70.prepare_batch_files(
                assignments_path=assignments,
                output_root=batches,
                max_requests_per_file=10,
                preflight_path=preflight,
            )
            client = SimpleNamespace(files=FakeFiles(), batches=FakeBatches())
            submission = batches / "submission.json"
            submitted = gpt_r70.submit_prepared_batches(
                manifest_path=batches / "manifest.json",
                client=client,
                confirm_paid_submission=True,
                confirm_licensed_text_processing=True,
                submission_path=submission,
                max_paid_requests=2,
            )
            self.assertEqual(submitted["status"], "submitted")
            collected = gpt_r70.collect_submitted_batches(
                submission_path=submission,
                client=client,
                download_root=root / "downloaded",
            )
            self.assertEqual(collected["status"], "complete_downloaded")
            downloaded = root / "downloaded" / "batch_0001.output.jsonl"
            self.assertTrue(downloaded.is_file())
            self.assertEqual(client.batches.retrieved, "batch-one")
            self.assertEqual(client.files.requested_file_id, "file-output")
            self.assertEqual(
                collected["submitted_shards"][0]["output_file"][
                    "remote_file_id"
                ],
                "file-output",
            )

    def test_collection_replaces_unbound_preexisting_download(self) -> None:
        class FakeFiles:
            def __init__(self) -> None:
                self.calls = []

            def content(self, file_id):
                self.calls.append(file_id)
                return SimpleNamespace(content=b'{"custom_id":"fresh"}\n')

        class FakeBatches:
            def retrieve(self, _batch_id):
                return SimpleNamespace(
                    status="completed",
                    input_file_id="file-input",
                    output_file_id="file-output",
                    error_file_id=None,
                )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            batch_manifest = root / "manifest.json"
            gpt_r70.write_json(
                batch_manifest,
                {"batch_files": [{"file": "batch_0001.jsonl"}]},
            )
            submission = root / "submission.json"
            gpt_r70.write_json(
                submission,
                {
                    "status": "submitted",
                    "batch_manifest_path": str(batch_manifest.resolve()),
                    "batch_manifest_sha256": gpt_r70.sha256_file(
                        batch_manifest
                    ),
                    "model": gpt_r70.MODEL_ID,
                    "submitted_shards": [
                        {
                            "batch_file": "batch_0001.jsonl",
                            "batch_id": "batch-one",
                            "input_file_id": "file-input",
                            "submission_state": "submitted",
                        }
                    ],
                },
            )
            download_root = root / "downloaded"
            download_root.mkdir()
            preexisting = download_root / "batch_0001.output.jsonl"
            preexisting.write_text('{"custom_id":"stale"}\n', encoding="utf-8")
            files = FakeFiles()
            collected = gpt_r70.collect_submitted_batches(
                submission_path=submission,
                client=SimpleNamespace(files=files, batches=FakeBatches()),
                download_root=download_root,
            )
            self.assertEqual(files.calls, ["file-output"])
            self.assertEqual(
                preexisting.read_bytes(), b'{"custom_id":"fresh"}\n'
            )
            record = collected["submitted_shards"][0]["output_file"]
            self.assertEqual(record["remote_file_id"], "file-output")
            self.assertEqual(record["sha256"], gpt_r70.sha256_file(preexisting))

    def test_cleanup_rejects_corrupted_local_download(self) -> None:
        class FakeFiles:
            def __init__(self) -> None:
                self.calls = []

            def delete(self, file_id):
                self.calls.append(file_id)
                return SimpleNamespace(deleted=True)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output = root / "output.jsonl"
            output.write_text("{}\n", encoding="utf-8")
            submission = root / "submission.json"
            gpt_r70.write_json(
                submission,
                {
                    "status": "partial_complete_downloaded",
                    "submitted_shards": [
                        {
                            "status": "completed",
                            "input_file_id": "input",
                            "output_file_id": "output",
                            "error_file_id": None,
                            "output_file": {
                                "path": str(output.resolve()),
                                "remote_file_id": "output",
                                "sha256": gpt_r70.sha256_file(output),
                                "bytes": output.stat().st_size,
                            },
                        }
                    ],
                },
            )
            output.write_text('{"corrupt":true}\n', encoding="utf-8")
            files = FakeFiles()
            with self.assertRaisesRegex(
                gpt_r70.SemanticConstructionError,
                "no longer matches remote file",
            ):
                gpt_r70.cleanup_remote_files(
                    submission_path=submission,
                    client=SimpleNamespace(files=files),
                    confirm_delete_remote_files=True,
                    cleanup_path=root / "cleanup.json",
                )
            self.assertEqual(files.calls, [])


if __name__ == "__main__":
    unittest.main()
