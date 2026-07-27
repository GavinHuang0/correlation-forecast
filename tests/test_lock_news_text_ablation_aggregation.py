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

SPEC = importlib.util.spec_from_file_location(
    "lock_news_text_ablation_aggregation_for_test",
    SCRIPTS / "lock_news_text_ablation_aggregation.py",
)
assert SPEC is not None and SPEC.loader is not None
locker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(locker)


SCHEMA = json.loads(
    (ROOT / "config" / "news_feature_schema_coarse.json").read_text(
        encoding="utf-8"
    )
)


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


class AggregationLockFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.schema = root / "schema.json"
        write_json(self.schema, SCHEMA)
        self.reference = root / "reference.jsonl"
        write_jsonl(
            self.reference,
            [
                {
                    "article_id": article_id,
                    "labels": {
                        "shock_scope": "idiosyncratic",
                        "event_family": "product_demand",
                        "information_status": "confirmed",
                        "directional_alignment": "single_firm_only",
                    },
                }
                for article_id in ("article-a", "article-b", "article-c")
            ],
        )
        self.description_input = root / "massive_description.jsonl"
        self.fulltext_input = root / "fulltext_evidence_chunks.jsonl"
        write_jsonl(
            self.description_input,
            [{"article_id": value} for value in ("article-a", "article-b", "article-c")],
        )
        write_jsonl(
            self.fulltext_input,
            [
                {
                    "chunk_id": f"chunk-{value}",
                    "article_id": f"article-{value}",
                }
                for value in ("a", "b", "c")
            ],
        )
        self.manifest = root / "manifest.json"
        write_json(
            self.manifest,
            {
                "status": "complete",
                "evaluation_schema": {
                    "sha256": locker.evaluator.sha256_file(self.schema)
                },
                "split": {
                    "development_article_ids": ["article-a", "article-b"],
                    "evaluation_article_ids": ["article-c"],
                },
                "variants": {
                    "massive_description": {
                        "file": self.description_input.name,
                        "sha256": locker.evaluator.sha256_file(
                            self.description_input
                        ),
                        "article_ids": [
                            "article-a",
                            "article-b",
                            "article-c",
                        ],
                    },
                    "fulltext_evidence_chunks": {
                        "file": self.fulltext_input.name,
                        "sha256": locker.evaluator.sha256_file(
                            self.fulltext_input
                        ),
                        "article_ids": [
                            "article-a",
                            "article-b",
                            "article-c",
                        ],
                    },
                },
                "fulltext_chunking": {
                    "chunk_id_to_article_id": [
                        {
                            "chunk_id": f"chunk-{value}",
                            "article_id": f"article-{value}",
                            "chunk_index": 0,
                            "chunk_count": 1,
                        }
                        for value in ("a", "b", "c")
                    ]
                },
            },
        )
        self.predictions = {
            "massive_description": root / "description_predictions.jsonl",
            "fulltext_evidence_chunks": root / "fulltext_predictions.jsonl",
        }
        for variant, path in self.predictions.items():
            input_path = (
                self.description_input
                if variant == "massive_description"
                else self.fulltext_input
            )
            rows = []
            for value in ("a", "b", "c"):
                row = {
                    "article_id": f"article-{value}",
                    "labels": {
                        "shock_scope": "idiosyncratic",
                        "event_family": "product_demand",
                        "information_status": "confirmed",
                        "directional_alignment": "single_firm_only",
                    },
                }
                if variant == "fulltext_evidence_chunks":
                    row["chunk_id"] = f"chunk-{value}"
                    row["parent_article_id"] = f"article-{value}"
                rows.append(row)
            write_jsonl(path, rows)
            write_json(
                locker.prediction_sidecar_path(path),
                {
                    "status": "complete",
                    "input_path": str(input_path.resolve()),
                    "input_sha256": locker.evaluator.sha256_file(input_path),
                    "output_sha256": locker.evaluator.sha256_file(path),
                    "schema_sha256": locker.evaluator.sha256_file(self.schema),
                    "total_prediction_count": 3,
                    "model_id": "example/model",
                    "model_revision": "frozen-revision",
                    "prompt_version": "prompt-v1",
                    "extractor_source_sha256": "a" * 64,
                    "deterministic_rule_version": "rules-v1",
                },
            )
        self.reports: dict[str, Path] = {}
        metrics = {
            "mean_score": (0.70, 0.60, 0.70),
            "plurality": (0.70, 0.60, 0.70),
            "best_margin": (0.69, 0.90, 0.90),
            "max_score": (0.71, 0.10, 0.10),
        }
        for reducer, values in metrics.items():
            path = root / f"development_{reducer}.json"
            write_json(path, self._report(reducer, *values))
            self.reports[reducer] = path

    def _report(
        self,
        reducer: str,
        primary: float,
        mean_macro_f1: float,
        mean_accuracy: float,
    ) -> dict:
        input_hashes = {
            "benchmark_manifest": locker.evaluator.sha256_file(self.manifest),
            "reference": locker.evaluator.sha256_file(self.reference),
            "schema": locker.evaluator.sha256_file(self.schema),
            **{
                f"predictions:{variant}": locker.evaluator.sha256_file(path)
                for variant, path in self.predictions.items()
            },
        }
        fulltext = {
            "article_count": 2,
            "article_coverage_of_selected_split": 1.0,
            "aggregation": {
                "requested_chunk_aggregation": reducer,
            },
            "field_metrics": {
                "shock_scope": {"macro_f1": primary},
                "directional_alignment": {"macro_f1": primary},
            },
            "mean_field_macro_f1": mean_macro_f1,
            "mean_field_accuracy": mean_accuracy,
        }
        return {
            "report_version": locker.evaluator.REPORT_VERSION,
            "split": "development",
            "reference_filter": "all",
            "source_filter": None,
            "strict_manifest_completeness": True,
            "selected_reference_article_count": 2,
            "aggregation_contract": {
                "requested_fulltext_chunk_aggregation": reducer
            },
            "variants": {
                "massive_description": {
                    "article_count": 2,
                    "article_coverage_of_selected_split": 1.0,
                },
                "fulltext_evidence_chunks": fulltext,
            },
            "input_sha256": input_hashes,
        }

    def build(self) -> dict:
        return locker.build_aggregation_lock(
            model_name="test model",
            benchmark_manifest_path=self.manifest,
            reference_path=self.reference,
            schema_path=self.schema,
            prediction_paths=self.predictions,
            sidecar_paths={},
            development_report_paths=self.reports,
        )


class AggregationSelectionTests(unittest.TestCase):
    def test_primary_metric_precedes_secondary_metrics(self) -> None:
        candidates = {
            reducer: {
                "primary_two_field_macro_f1": 0.5,
                "fulltext_mean_field_macro_f1": 0.5,
                "fulltext_mean_field_accuracy": 0.5,
            }
            for reducer in locker.evaluator.CHUNK_AGGREGATIONS
        }
        candidates["max_score"] = {
            "primary_two_field_macro_f1": 0.51,
            "fulltext_mean_field_macro_f1": 0.0,
            "fulltext_mean_field_accuracy": 0.0,
        }
        self.assertEqual(locker.select_reducer(candidates), "max_score")

    def test_fixed_tie_order_is_used_last(self) -> None:
        candidates = {
            reducer: {
                "primary_two_field_macro_f1": 0.5,
                "fulltext_mean_field_macro_f1": 0.5,
                "fulltext_mean_field_accuracy": 0.5,
            }
            for reducer in locker.evaluator.CHUNK_AGGREGATIONS
        }
        self.assertEqual(locker.select_reducer(candidates), "mean_score")
        candidates["mean_score"]["fulltext_mean_field_accuracy"] = 0.4
        self.assertEqual(locker.select_reducer(candidates), "plurality")


class AggregationProvenanceTests(unittest.TestCase):
    def test_builds_lock_and_revalidates_it_for_holdout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AggregationLockFixture(Path(temporary))
            lock = fixture.build()
            self.assertEqual(lock["selected_chunk_aggregation"], "max_score")
            self.assertEqual(
                lock["model_identity"]["extractor_source_sha256"], "a" * 64
            )
            lock_path = Path(temporary) / "aggregation_lock.json"
            write_json(lock_path, lock)
            validated = locker.validate_lock_for_evaluation(
                lock_path=lock_path,
                benchmark_manifest_path=fixture.manifest,
                reference_path=fixture.reference,
                schema_path=fixture.schema,
                prediction_paths=fixture.predictions,
                chunk_aggregation="max_score",
            )
            self.assertEqual(
                validated["selected_chunk_aggregation"], "max_score"
            )
            self.assertEqual(
                validated["sha256"],
                locker.evaluator.sha256_file(lock_path),
            )

    def test_sidecar_output_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AggregationLockFixture(Path(temporary))
            sidecar_path = locker.prediction_sidecar_path(
                fixture.predictions["massive_description"]
            )
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            sidecar["output_sha256"] = "0" * 64
            write_json(sidecar_path, sidecar)
            with self.assertRaisesRegex(
                locker.AggregationLockError, "output differs"
            ):
                fixture.build()

    def test_report_input_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AggregationLockFixture(Path(temporary))
            path = fixture.reports["plurality"]
            report = json.loads(path.read_text(encoding="utf-8"))
            report["input_sha256"]["reference"] = "0" * 64
            write_json(path, report)
            with self.assertRaisesRegex(
                locker.AggregationLockError, "input hash differs"
            ):
                fixture.build()

    def test_extractor_protocol_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AggregationLockFixture(Path(temporary))
            sidecar_path = locker.prediction_sidecar_path(
                fixture.predictions["fulltext_evidence_chunks"]
            )
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            sidecar["prompt_version"] = "different-prompt"
            write_json(sidecar_path, sidecar)
            with self.assertRaisesRegex(
                locker.AggregationLockError, "extractor protocol"
            ):
                fixture.build()

    def test_prediction_drift_after_lock_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AggregationLockFixture(Path(temporary))
            lock_path = Path(temporary) / "aggregation_lock.json"
            write_json(lock_path, fixture.build())
            with fixture.predictions["massive_description"].open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write('{"article_id":"unexpected"}\n')
            with self.assertRaisesRegex(
                locker.AggregationLockError, "predictions differ"
            ):
                locker.validate_lock_for_evaluation(
                    lock_path=lock_path,
                    benchmark_manifest_path=fixture.manifest,
                    reference_path=fixture.reference,
                    schema_path=fixture.schema,
                    prediction_paths=fixture.predictions,
                    chunk_aggregation="max_score",
                )

    def test_evaluator_cli_enforces_lock_on_holdout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AggregationLockFixture(Path(temporary))
            lock_path = Path(temporary) / "aggregation_lock.json"
            output_path = Path(temporary) / "holdout_report.json"
            write_json(lock_path, fixture.build())
            command = [
                sys.executable,
                str(SCRIPTS / "evaluate_news_text_ablation.py"),
                "--reference",
                str(fixture.reference),
                "--benchmark-manifest",
                str(fixture.manifest),
                "--schema",
                str(fixture.schema),
                "--predictions",
                "massive_description="
                + str(fixture.predictions["massive_description"]),
                "--predictions",
                "fulltext_evidence_chunks="
                + str(fixture.predictions["fulltext_evidence_chunks"]),
                "--split",
                "evaluation",
                "--chunk-aggregation",
                "max_score",
                "--aggregation-lock",
                str(lock_path),
                "--output",
                str(output_path),
            ]
            completed = subprocess.run(
                command,
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=completed.stdout + completed.stderr,
            )
            report = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(
                report["aggregation_lock"]["selected_chunk_aggregation"],
                "max_score",
            )
            self.assertEqual(report["split"], "evaluation")


if __name__ == "__main__":
    unittest.main()
