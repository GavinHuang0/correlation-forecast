from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
FIXTURES = ROOT / "tests" / "fixtures" / "news_text_ablation"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

SPEC = importlib.util.spec_from_file_location(
    "evaluate_news_text_ablation_for_test",
    SCRIPTS / "evaluate_news_text_ablation.py",
)
assert SPEC is not None and SPEC.loader is not None
evaluator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluator)


SCHEMA = {
    "closed_label_fields": {
        "shock_scope": ["idiosyncratic", "common"],
        "event_family": ["product_demand", "macro_market"],
        "information_status": ["confirmed", "anticipated"],
        "directional_alignment": [
            "single_firm_only",
            "common_direction_unclear",
        ],
    }
}


def read_jsonl(name: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (FIXTURES / name).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def benchmark_manifest() -> dict:
    mapping = [
        {
            "chunk_id": "chunk-a-0",
            "article_id": "article-a",
            "chunk_index": 0,
            "chunk_count": 2,
        },
        {
            "chunk_id": "chunk-a-1",
            "article_id": "article-a",
            "chunk_index": 1,
            "chunk_count": 2,
        },
        {
            "chunk_id": "chunk-b-0",
            "article_id": "article-b",
            "chunk_index": 0,
            "chunk_count": 3,
        },
        {
            "chunk_id": "chunk-b-1",
            "article_id": "article-b",
            "chunk_index": 1,
            "chunk_count": 3,
        },
        {
            "chunk_id": "chunk-b-2",
            "article_id": "article-b",
            "chunk_index": 2,
            "chunk_count": 3,
        },
    ]
    return {
        "status": "complete",
        "split": {
            "development_article_ids": [],
            "evaluation_article_ids": ["article-a", "article-b"],
        },
        "variants": {
            "massive_description": {
                "article_ids": ["article-a", "article-b"]
            },
            "alpha_summary": {"article_ids": ["article-a"]},
            "fulltext_evidence_chunks": {
                "article_ids": ["article-a", "article-b"]
            },
        },
        "ticker_provenance_contract": {
            "article_assignment_basis": {
                "article-a": "direct_target_tag",
                "article-b": "peer_only_tag",
            }
        },
        "fulltext_chunking": {"chunk_id_to_article_id": mapping},
    }


class NewsTextAblationAggregationTests(unittest.TestCase):
    def test_complete_scores_take_priority_and_plurality_is_fallback(self) -> None:
        aggregated, metadata = evaluator.aggregate_variant_predictions(
            variant="fulltext_evidence_chunks",
            predictions=read_jsonl("fulltext_predictions.jsonl"),
            schema=SCHEMA,
            manifest=benchmark_manifest(),
            selected_article_ids={"article-a", "article-b"},
        )
        self.assertEqual(
            aggregated["article-a"]["shock_scope"], "idiosyncratic"
        )
        self.assertEqual(aggregated["article-b"]["shock_scope"], "common")
        self.assertEqual(
            metadata["field_aggregation_method_counts"]["shock_scope"],
            {"mean_candidate_score": 1, "plurality": 1},
        )
        self.assertTrue(metadata["complete_for_selected_split"])

    def test_schema_order_breaks_plurality_and_score_ties(self) -> None:
        rows = [
            {"labels": {"shock_scope": "common"}},
            {"labels": {"shock_scope": "idiosyncratic"}},
        ]
        label, method = evaluator.aggregate_field(
            rows, "shock_scope", SCHEMA["closed_label_fields"]["shock_scope"]
        )
        self.assertEqual((label, method), ("idiosyncratic", "plurality"))
        score_rows = [
            {
                "scores": {
                    "shock_scope": {"idiosyncratic": -0.5, "common": -0.5}
                }
            }
        ]
        label, method = evaluator.aggregate_field(
            score_rows,
            "shock_scope",
            SCHEMA["closed_label_fields"]["shock_scope"],
        )
        self.assertEqual(
            (label, method), ("idiosyncratic", "mean_candidate_score")
        )

    def test_available_complete_scores_are_used_when_other_chunks_have_labels_only(
        self,
    ) -> None:
        rows = [
            {
                "labels": {"shock_scope": "common"},
                "passes": {
                    "shock_scope": {
                        "order_averaged_mean_log_probabilities": {
                            "idiosyncratic": -0.1,
                            "common": -0.4,
                        }
                    }
                },
            },
            {"labels": {"shock_scope": "common"}},
        ]
        label, method = evaluator.aggregate_field(
            rows, "shock_scope", SCHEMA["closed_label_fields"]["shock_scope"]
        )
        self.assertEqual(
            (label, method), ("idiosyncratic", "mean_candidate_score")
        )

    def test_partial_chunk_score_coverage_is_reported(self) -> None:
        predictions = read_jsonl("fulltext_predictions.jsonl")
        predictions[1]["scores"].pop("event_family")
        _, metadata = evaluator.aggregate_variant_predictions(
            variant="fulltext_evidence_chunks",
            predictions=predictions,
            schema=SCHEMA,
            manifest=benchmark_manifest(),
            selected_article_ids={"article-a", "article-b"},
        )
        coverage = metadata["field_score_coverage"]["event_family"]
        self.assertEqual(coverage["partial_chunks_scored_article_count"], 1)
        self.assertEqual(coverage["all_chunks_scored_article_count"], 0)
        self.assertEqual(coverage["no_chunks_scored_article_count"], 1)
        self.assertEqual(coverage["chunks_with_complete_scores"], 1)
        self.assertEqual(coverage["total_chunks"], 5)
        self.assertTrue(coverage["partial_scores_are_used_when_available"])

    def test_chunk_score_reducers_are_explicit_and_deterministic(self) -> None:
        rows = [
            {
                "labels": {"shock_scope": "idiosyncratic"},
                "scores": {
                    "shock_scope": {
                        "idiosyncratic": 0.6,
                        "common": 0.4,
                    }
                },
            },
            {
                "labels": {"shock_scope": "common"},
                "scores": {
                    "shock_scope": {
                        "idiosyncratic": 0.1,
                        "common": 0.95,
                    }
                },
            },
        ]
        allowed = SCHEMA["closed_label_fields"]["shock_scope"]
        self.assertEqual(
            evaluator.aggregate_field(
                rows,
                "shock_scope",
                allowed,
                score_aggregation="mean_score",
            ),
            ("common", "mean_candidate_score"),
        )
        self.assertEqual(
            evaluator.aggregate_field(
                rows,
                "shock_scope",
                allowed,
                score_aggregation="max_score",
            ),
            ("common", "max_candidate_score"),
        )
        self.assertEqual(
            evaluator.aggregate_field(
                rows,
                "shock_scope",
                allowed,
                score_aggregation="best_margin",
            ),
            ("common", "best_margin_candidate_score"),
        )
        self.assertEqual(
            evaluator.aggregate_field(
                rows,
                "shock_scope",
                allowed,
                score_aggregation="plurality",
            ),
            ("idiosyncratic", "plurality"),
        )

    def test_calibrated_flan_scores_take_priority_over_raw_scores(self) -> None:
        rows = [
            {
                "labels": {"shock_scope": "common"},
                "passes": {
                    "shock_scope": {
                        "order_averaged_mean_log_probabilities": {
                            "idiosyncratic": -0.1,
                            "common": -0.4,
                        },
                        "calibration": {
                            "adjusted_scores": {
                                "idiosyncratic": -0.8,
                                "common": -0.2,
                            }
                        },
                    }
                },
            }
        ]
        label, method = evaluator.aggregate_field(
            rows, "shock_scope", SCHEMA["closed_label_fields"]["shock_scope"]
        )
        self.assertEqual((label, method), ("common", "mean_candidate_score"))

    def test_llama_rotation_averaged_scores_are_recognized(self) -> None:
        rows = [
            {
                "labels": {"shock_scope": "common"},
                "passes": {
                    "shock_scope": {
                        "all_rotation_mean_log_probabilities": {
                            "idiosyncratic": -0.2,
                            "common": -0.6,
                        }
                    }
                },
            }
        ]
        label, method = evaluator.aggregate_field(
            rows, "shock_scope", SCHEMA["closed_label_fields"]["shock_scope"]
        )
        self.assertEqual(
            (label, method), ("idiosyncratic", "mean_candidate_score")
        )

    def test_missing_chunk_prediction_is_rejected_in_strict_mode(self) -> None:
        predictions = read_jsonl("fulltext_predictions.jsonl")[:-1]
        with self.assertRaisesRegex(ValueError, "missing"):
            evaluator.aggregate_variant_predictions(
                variant="fulltext_evidence_chunks",
                predictions=predictions,
                schema=SCHEMA,
                manifest=benchmark_manifest(),
                selected_article_ids={"article-a", "article-b"},
            )


class NewsTextAblationEvaluationTests(unittest.TestCase):
    def test_closed_metrics_report_majority_baseline_kappa_and_support(self) -> None:
        metrics = evaluator.closed_label_metrics(
            ["idiosyncratic", "idiosyncratic", "common"],
            ["idiosyncratic", "idiosyncratic", "idiosyncratic"],
            ["idiosyncratic", "common"],
        )
        self.assertEqual(metrics["majority_class_baseline_accuracy"], 2 / 3)
        self.assertAlmostEqual(metrics["cohens_kappa"], 0.0)
        self.assertEqual(metrics["macro_f1_supported_class_count"], 2)
        self.assertEqual(metrics["schema_class_count"], 2)

    def test_report_computes_per_field_metrics_and_paired_deltas(self) -> None:
        report = evaluator.build_evaluation_report(
            reference_records=read_jsonl("coarse_reference.jsonl"),
            predictions_by_variant={
                "massive_description": read_jsonl(
                    "massive_predictions.jsonl"
                ),
                "alpha_summary": read_jsonl("alpha_predictions.jsonl"),
                "fulltext_evidence_chunks": read_jsonl(
                    "fulltext_predictions.jsonl"
                ),
            },
            manifest=benchmark_manifest(),
            schema=SCHEMA,
            split="evaluation",
        )
        self.assertEqual(
            report["variants"]["massive_description"]["field_metrics"][
                "shock_scope"
            ]["accuracy"],
            0.5,
        )
        self.assertEqual(
            report["variants"]["fulltext_evidence_chunks"][
                "mean_field_accuracy"
            ],
            1.0,
        )
        full_minus_massive = report["paired_deltas"][
            "fulltext_evidence_chunks_minus_massive_description"
        ]
        self.assertEqual(full_minus_massive["paired_article_count"], 2)
        self.assertEqual(full_minus_massive["mean_field_accuracy_delta"], 0.5)
        self.assertIn(
            "primary_two_field_macro_f1_delta_bootstrap_95pct",
            full_minus_massive["aggregate_uncertainty"],
        )
        alpha_minus_massive = report["paired_deltas"][
            "alpha_summary_minus_massive_description"
        ]
        self.assertEqual(alpha_minus_massive["paired_article_count"], 1)
        self.assertEqual(alpha_minus_massive["mean_field_accuracy_delta"], -1.0)

    def test_fine_gpt_reference_is_coarsened_deterministically(self) -> None:
        fine = {
            "article_id": "fine-one",
            "labels": {
                "relevance": "direct_target",
                "event_scope": "firm_specific",
                "event_type": "product_technology",
                "information_status": "confirmed",
                "peer_effect": "none_stated",
                "target_direction": "positive",
                "sector_direction": "not_applicable",
            },
        }
        mapped = evaluator.reference_by_id([fine])
        self.assertEqual(
            mapped["fine-one"]["labels"],
            {
                "shock_scope": "idiosyncratic",
                "event_family": "product_demand",
                "information_status": "confirmed",
                "directional_alignment": "single_firm_only",
            },
        )
        self.assertFalse(mapped["fine-one"]["abstained"])

    def test_assignment_filter_selects_declared_provenance_stratum(self) -> None:
        report = evaluator.build_evaluation_report(
            reference_records=read_jsonl("coarse_reference.jsonl"),
            predictions_by_variant={
                "massive_description": read_jsonl(
                    "massive_predictions.jsonl"
                ),
                "alpha_summary": read_jsonl("alpha_predictions.jsonl"),
                "fulltext_evidence_chunks": read_jsonl(
                    "fulltext_predictions.jsonl"
                ),
            },
            manifest=benchmark_manifest(),
            schema=SCHEMA,
            split="evaluation",
            assignment_filter="direct_target_tag",
        )
        self.assertEqual(report["assignment_filter"], "direct_target_tag")
        self.assertEqual(report["selected_reference_article_count"], 1)
        self.assertEqual(
            report["variants"]["massive_description"]["article_count"], 1
        )

    def test_prediction_spec_requires_known_unique_variant(self) -> None:
        paths = evaluator.parse_prediction_specs(
            ["massive_description=massive.jsonl"]
        )
        self.assertEqual(paths["massive_description"], Path("massive.jsonl"))
        with self.assertRaisesRegex(ValueError, "Unknown variant"):
            evaluator.parse_prediction_specs(["other=predictions.jsonl"])


if __name__ == "__main__":
    unittest.main()
