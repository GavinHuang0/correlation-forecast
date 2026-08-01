from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from scripts.semantic_news_v3 import flan_w17_lite as lite


def _candidate_roles(
    *,
    direct: bool = False,
    sector: bool = False,
    macro: bool = False,
) -> dict[str, bool]:
    return {
        "any_common": sector or macro,
        "direct": direct,
        "macro_common": macro,
        "peer_idio": False,
        "sector_common": sector,
        "target_common": False,
        "target_idio": direct,
    }


def _assignment(
    article_id: str,
    *,
    assignment_id: str | None = None,
    weight: float = 1.0,
    published: str = "2025-01-01T12:00:00Z",
    direct: bool = True,
    sector: bool = False,
    macro: bool = False,
) -> dict[str, object]:
    headline = f"Headline {article_id}"
    description = f"Description {article_id}"
    model_text = f"{headline}\n\n{description}"
    return {
        "assignment_id": assignment_id or f"assignment-{article_id}",
        "article_id": article_id,
        "forecast_date": "2025-01-02",
        "published_at_utc": published,
        "cutoff_utc": "2025-01-02T14:00:00Z",
        "target_ticker": "AAA",
        "sector": "Technology",
        "role": "I",
        "candidate_roles": _candidate_roles(
            direct=direct, sector=sector, macro=macro
        ),
        "aggregation_weight": weight,
        "source_profile": lite.SOURCE_PROFILE,
        "source_query_scope_complete": True,
        "candidate_assignment_complete": True,
        "headline": headline,
        "description": description,
        "model_text": model_text,
        "headline_sha256": lite.sha256_text(headline),
        "description_sha256": lite.sha256_text(description),
        "text_sha256": lite.sha256_text(model_text),
        "description_available": 1,
    }


def _article(article_id: str) -> dict[str, object]:
    assignment = _assignment(article_id)
    return {
        "provider_article_id": article_id,
        "published_at_utc": assignment["published_at_utc"],
        "forecast_date": assignment["forecast_date"],
        "cutoff_utc": assignment["cutoff_utc"],
        "headline": assignment["headline"],
        "description": assignment["description"],
        "description_available": 1,
        "retained_description_char_count": len(
            str(assignment["description"])
        ),
        "description_was_bounded": False,
        "model_text": assignment["model_text"],
        "headline_sha256": assignment["headline_sha256"],
        "description_sha256": assignment["description_sha256"],
        "model_text_sha256": assignment["text_sha256"],
        "source_profile": lite.SOURCE_PROFILE,
    }


def _queue_article(article_id: str) -> dict[str, object]:
    source = _article(article_id)
    payload = {
        "record_version": "flan-w17-lite-selected-article-v1",
        "article_id": article_id,
        "forecast_date": source["forecast_date"],
        "published_at_utc": source["published_at_utc"],
        "cutoff_utc": source["cutoff_utc"],
        "headline": source["headline"],
        "description": source["description"],
        "description_available": True,
        "retained_description_char_count": source[
            "retained_description_char_count"
        ],
        "description_was_bounded": False,
        "model_text": source["model_text"],
        "headline_sha256": source["headline_sha256"],
        "description_sha256": source["description_sha256"],
        "model_text_sha256": source["model_text_sha256"],
        "source_profile": lite.SOURCE_PROFILE,
    }
    payload["article_input_sha256"] = lite.sha256_json(payload)
    return payload


class FakeEngine:
    def __init__(
        self,
        *,
        canonical: str = "firm_operating_financial",
        reversed_prediction: str | None = None,
        interrupt_on_call: int | None = None,
    ) -> None:
        self.canonical = canonical
        self.reversed_prediction = reversed_prediction or canonical
        self.interrupt_on_call = interrupt_on_call
        self.calls: list[str] = []

    @staticmethod
    def _scores(best: str) -> dict[str, float]:
        return {
            label: (0.0 if label == best else -float(index + 1))
            for index, label in enumerate(lite.EVENT_GROUP_LABELS)
        }

    def predict(self, article: dict[str, object]) -> dict[str, object]:
        self.calls.append(str(article["article_id"]))
        if (
            self.interrupt_on_call is not None
            and len(self.calls) == self.interrupt_on_call
        ):
            raise KeyboardInterrupt
        canonical = self._scores(self.canonical)
        reversed_scores = self._scores(self.reversed_prediction)
        averaged = {
            label: (canonical[label] + reversed_scores[label]) / 2
            for label in lite.EVENT_GROUP_LABELS
        }
        averaged_prediction = max(
            lite.EVENT_GROUP_LABELS, key=lambda label: averaged[label]
        )
        agreement = self.canonical == self.reversed_prediction
        accepted = agreement and self.canonical != lite.ABSTENTION_LABEL
        canonical_prompt, reversed_prompt = lite.event_prompts(
            article, lite.load_event_schema()
        )
        return {
            "terminal_state": "complete",
            "no_input_truncation": True,
            "event_group": {
                "canonical_prediction": self.canonical,
                "reversed_prediction": self.reversed_prediction,
                "order_averaged_prediction": averaged_prediction,
                "canonical_candidate_mean_log_probabilities": canonical,
                "reversed_candidate_mean_log_probabilities": reversed_scores,
                "order_averaged_mean_log_probabilities": averaged,
                "order_averaged_top1_top2_margin": lite._score_margin(
                    averaged
                ),
                "order_agreement": agreement,
                "accepted": accepted,
                "accepted_label": self.canonical if accepted else None,
                "acceptance_reason": (
                    "accepted_order_agreement"
                    if accepted
                    else (
                        "agreed_other_or_unclear"
                        if agreement
                        else "order_disagreement"
                    )
                ),
                "prompt_sha256": lite.sha256_text(canonical_prompt),
                "reversed_prompt_sha256": lite.sha256_text(reversed_prompt),
                "input_tokens_canonical": 100,
                "input_tokens_reversed": 101,
                "input_tokens_max": 101,
                "input_truncated": False,
                "calibration_used": False,
            },
        }

    def diagnostics(self) -> dict[str, object]:
        return {"fake": True, "calls": len(self.calls)}


class InvalidEngine:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def predict(self, article: dict[str, object]) -> dict[str, object]:
        self.calls.append(str(article["article_id"]))
        return {
            "terminal_state": "complete",
            "no_input_truncation": True,
            "event_group": {"malformed": True},
        }


class FlanW17LiteTests(unittest.TestCase):
    def test_schema_is_exact_four_way_order_agreement_contract(self) -> None:
        schema = lite.load_event_schema()
        self.assertEqual(
            tuple(schema["closed_label_fields"][lite.EVENT_FIELD]),
            lite.EVENT_GROUP_LABELS,
        )
        changed = copy.deepcopy(schema)
        changed["acceptance"]["requires_order_agreement"] = False
        with self.assertRaisesRegex(ValueError, "acceptance"):
            lite.validate_event_schema(changed)

    def test_k16_selector_deduplicates_before_rank_and_is_row_order_invariant(
        self,
    ) -> None:
        assignments = []
        articles = []
        for index in range(20):
            article_id = f"article-{index:02d}"
            assignments.append(
                _assignment(
                    article_id,
                    weight=float(20 - index),
                    sector=index == 18,
                    macro=index == 19,
                )
            )
            articles.append(_article(article_id))
        assignments.append(
            _assignment(
                "article-00",
                assignment_id="assignment-article-00-duplicate",
                weight=20.0,
            )
        )
        first = lite.build_k16_selection(
            pd.DataFrame(assignments),
            pd.DataFrame(articles),
            enforce_expected_profile=False,
        )
        shuffled = lite.build_k16_selection(
            pd.DataFrame(assignments).sample(frac=1, random_state=7),
            pd.DataFrame(articles).sample(frac=1, random_state=11),
            enforce_expected_profile=False,
        )
        first_membership, first_selected, first_queue, first_audit = first
        second_membership, second_selected, second_queue, second_audit = (
            shuffled
        )
        self.assertEqual(
            first_audit["eligible_group_article_rows"]["direct"], 20
        )
        self.assertEqual(
            first_audit["selected_group_article_rows"]["direct"], 16
        )
        self.assertEqual(first_audit["selected_unique_articles"], 18)
        self.assertIn("article-18", {row["article_id"] for row in first_queue})
        self.assertIn("article-19", {row["article_id"] for row in first_queue})
        self.assertEqual(
            first_audit["ordered_selected_article_ids_sha256"],
            second_audit["ordered_selected_article_ids_sha256"],
        )
        pd.testing.assert_frame_equal(first_membership, second_membership)
        pd.testing.assert_frame_equal(first_selected, second_selected)
        self.assertEqual(first_queue, second_queue)
        self.assertEqual(len(first_selected), 19)

    def test_acceptance_requires_order_agreement_and_predictive_label(
        self,
    ) -> None:
        article = _queue_article("article")
        common = {
            "run_id": "run",
            "queue_sha256": "queue",
            "selection_manifest_sha256": "selection",
            "preflight_manifest_sha256": "preflight",
            "runtime_contract_sha256_value": "runtime",
            "attempt": 1,
        }
        agreed = lite.terminal_prediction(
            article, FakeEngine(), **common
        )
        self.assertTrue(agreed["event_group"]["accepted"])
        lite.validate_terminal_record(agreed)

        disagreed = lite.terminal_prediction(
            article,
            FakeEngine(reversed_prediction="policy_corporate"),
            **common,
        )
        self.assertFalse(disagreed["event_group"]["accepted"])
        self.assertEqual(
            disagreed["event_group"]["acceptance_reason"],
            "order_disagreement",
        )
        lite.validate_terminal_record(disagreed)

        abstained = lite.terminal_prediction(
            article,
            FakeEngine(canonical=lite.ABSTENTION_LABEL),
            **common,
        )
        self.assertFalse(abstained["event_group"]["accepted"])
        self.assertEqual(
            abstained["event_group"]["acceptance_reason"],
            "agreed_other_or_unclear",
        )
        lite.validate_terminal_record(abstained)

    def _run_fixture(self, root: Path) -> tuple[Path, Path, Path, Path, dict]:
        queue = root / "queue.jsonl.gz"
        selection = root / "selection.json"
        preflight = root / "preflight.json"
        output = root / "predictions.jsonl"
        articles = [_queue_article(f"article-{index}") for index in range(3)]
        lite._atomic_jsonl(queue, articles, gzip_output=True)
        lite._atomic_json(selection, {"placeholder": True})
        lite._atomic_json(preflight, {"placeholder": True})
        selection_value = {
            "audit": {
                "selected_unique_articles": len(articles),
                "ordered_selected_article_ids_sha256": lite._ordered_lf_hash(
                    [str(record["article_id"]) for record in articles]
                ),
            }
        }
        return queue, selection, preflight, output, selection_value

    def test_interrupted_inference_resumes_without_recomputing_successes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue, selection, preflight, output, selection_value = (
                self._run_fixture(root)
            )
            runtime = {"runtime": "fixed"}
            preflight_value = {
                "runtime_contract_sha256": lite.sha256_json(runtime)
            }
            interrupted_engine = FakeEngine(interrupt_on_call=2)
            with (
                mock.patch.object(
                    lite,
                    "validate_selection_manifest",
                    return_value=selection_value,
                ),
                mock.patch.object(
                    lite,
                    "validate_preflight_manifest",
                    return_value=preflight_value,
                ),
                mock.patch.object(
                    lite, "runtime_contract", return_value=runtime
                ),
            ):
                first = lite.run_inference(
                    queue_path=queue,
                    selection_manifest_path=selection,
                    preflight_manifest_path=preflight,
                    output_path=output,
                    engine=interrupted_engine,
                )
                self.assertEqual(first["status"], "interrupted")
                self.assertEqual(first["successful_article_count"], 1)
                self.assertEqual(
                    len(list(lite.iter_jsonl(output))), 1
                )

                resumed_engine = FakeEngine()
                second = lite.run_inference(
                    queue_path=queue,
                    selection_manifest_path=selection,
                    preflight_manifest_path=preflight,
                    output_path=output,
                    engine=resumed_engine,
                )
                self.assertEqual(second["status"], "complete")
                self.assertEqual(
                    resumed_engine.calls, ["article-1", "article-2"]
                )
                self.assertEqual(
                    len(list(lite.iter_jsonl(output))), 3
                )

                unused_engine = FakeEngine()
                third = lite.run_inference(
                    queue_path=queue,
                    selection_manifest_path=selection,
                    preflight_manifest_path=preflight,
                    output_path=output,
                    engine=unused_engine,
                )
                self.assertEqual(third, second)
                self.assertEqual(unused_engine.calls, [])

    def test_max_new_articles_then_full_run_processes_only_remaining(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue, selection, preflight, output, selection_value = (
                self._run_fixture(root)
            )
            runtime = {"runtime": "fixed"}
            preflight_value = {
                "runtime_contract_sha256": lite.sha256_json(runtime)
            }
            first_engine = FakeEngine()
            with (
                mock.patch.object(
                    lite,
                    "validate_selection_manifest",
                    return_value=selection_value,
                ),
                mock.patch.object(
                    lite,
                    "validate_preflight_manifest",
                    return_value=preflight_value,
                ),
                mock.patch.object(
                    lite, "runtime_contract", return_value=runtime
                ),
            ):
                first = lite.run_inference(
                    queue_path=queue,
                    selection_manifest_path=selection,
                    preflight_manifest_path=preflight,
                    output_path=output,
                    engine=first_engine,
                    max_new_articles=1,
                )
                self.assertEqual(first["status"], "in_progress")
                self.assertEqual(first_engine.calls, ["article-0"])
                second_engine = FakeEngine()
                second = lite.run_inference(
                    queue_path=queue,
                    selection_manifest_path=selection,
                    preflight_manifest_path=preflight,
                    output_path=output,
                    engine=second_engine,
                )
                self.assertEqual(second["status"], "complete")
                self.assertEqual(
                    second_engine.calls, ["article-1", "article-2"]
                )

    def test_append_tail_recovery_is_narrow(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "records.jsonl"
            first = {"article_id": "first"}
            path.write_bytes(
                (lite.canonical_json(first) + "\n{\"article_id\":").encode()
            )
            recovery = lite.recover_append_tail(path)
            self.assertEqual(
                recovery["action"], "truncated_incomplete_tail"
            )
            self.assertEqual(list(lite.iter_jsonl(path)), [first])

            path.write_text(
                lite.canonical_json(first), encoding="utf-8"
            )
            recovery = lite.recover_append_tail(path)
            self.assertEqual(recovery["action"], "terminated_valid_tail")
            self.assertTrue(path.read_bytes().endswith(b"\n"))

    def test_live_writer_lock_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.lock"
            owner = {**lite._process_identity(), "run_id": "prior"}
            lite._atomic_json(path, owner)
            with self.assertRaisesRegex(RuntimeError, "holds the lock"):
                with lite.inference_lock(
                    path, run_id="new", recover_stale=True
                ):
                    self.fail("A live lock must not be entered")

    def test_complete_with_failures_is_a_nonzero_cli_result(self) -> None:
        with mock.patch.object(
            lite,
            "run_inference",
            return_value={
                "status": "complete_with_failures",
                "failure_article_count": 1,
            },
        ):
            self.assertEqual(lite.main(["infer"]), 2)

    def test_three_invalid_outputs_abort_instead_of_running_whole_corpus(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue, selection, preflight, output, selection_value = (
                self._run_fixture(root)
            )
            runtime = {"runtime": "fixed"}
            preflight_value = {
                "runtime_contract_sha256": lite.sha256_json(runtime)
            }
            engine = InvalidEngine()
            with (
                mock.patch.object(
                    lite,
                    "validate_selection_manifest",
                    return_value=selection_value,
                ),
                mock.patch.object(
                    lite,
                    "validate_preflight_manifest",
                    return_value=preflight_value,
                ),
                mock.patch.object(
                    lite, "runtime_contract", return_value=runtime
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "terminal failures"
                ):
                    lite.run_inference(
                        queue_path=queue,
                        selection_manifest_path=selection,
                        preflight_manifest_path=preflight,
                        output_path=output,
                        engine=engine,
                    )
            self.assertEqual(len(engine.calls), 3)
            manifest = json.loads(
                output.with_suffix(".jsonl.manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                manifest["status"], "aborted_systemic_failures"
            )


if __name__ == "__main__":
    unittest.main()
