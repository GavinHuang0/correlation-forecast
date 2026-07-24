from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

SPEC = importlib.util.spec_from_file_location(
    "select_flan_t5_final_candidate_for_test",
    SCRIPTS / "select_flan_t5_final_candidate.py",
)
assert SPEC is not None and SPEC.loader is not None
selector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selector)


def score_row(
    article_id: str,
    *,
    firm: float,
    common: float,
    event_scores: dict[str, float] | None = None,
    alignment_same: bool = False,
    alignment_opposite: bool = False,
) -> dict:
    event_scores = event_scores or {}

    def component(score: float, consensus: bool = False) -> dict:
        return {
            "order_averaged_yes_no_log_odds": score,
            "strict_consensus_yes": consensus,
            "valid": True,
            "input_truncated": False,
        }

    components = {
        "scope_firm": component(firm),
        "scope_common": component(common),
        **{
            f"event_{family}": component(event_scores.get(family, -2.0))
            for family in selector.EVENT_FAMILIES
        },
        "alignment_same": component(1.0 if alignment_same else -1.0, alignment_same),
        "alignment_opposite": component(
            1.0 if alignment_opposite else -1.0, alignment_opposite
        ),
    }
    return {
        "article_id": article_id,
        "target_ticker": "AMD",
        "semantic_applicable": True,
        "deterministic_features": {"semantic_applicable": True},
        "components": components,
        "validity": {
            "all_components_valid": True,
            "no_input_truncation": True,
            "expected_component_set": True,
        },
    }


def load_schema() -> dict:
    return json.loads(
        (ROOT / "config" / "news_feature_schema_coarse.json").read_text(
            encoding="utf-8"
        )
    )


class ThresholdCalibrationTests(unittest.TestCase):
    def test_threshold_candidates_are_midpoints_with_finite_outer_sentinels(self) -> None:
        candidates = selector.threshold_candidates([-1.0, 1.0])
        self.assertEqual(len(candidates), 3)
        self.assertLess(candidates[0], -1.0)
        self.assertEqual(candidates[1], 0.0)
        self.assertGreater(candidates[2], 1.0)

    def test_joint_scope_calibration_recovers_compositional_labels(self) -> None:
        rows = [
            score_row("idio", firm=2.0, common=-2.0),
            score_row("common", firm=-2.0, common=2.0),
            score_row("mixed", firm=2.0, common=2.0),
            score_row("unclear", firm=-2.0, common=-2.0),
        ]
        truths = {
            "idio": {"shock_scope": "idiosyncratic"},
            "common": {"shock_scope": "common"},
            "mixed": {"shock_scope": "mixed"},
            "unclear": {"shock_scope": "unclear"},
        }
        fit = selector.fit_scope_thresholds(
            rows,
            truths,
            ["idiosyncratic", "common", "mixed", "unclear"],
        )
        guesses = [
            selector.predict_scope(
                row, fit["firm_threshold"], fit["common_threshold"]
            )
            for row in rows
        ]
        self.assertEqual(
            guesses, ["idiosyncratic", "common", "mixed", "unclear"]
        )
        self.assertEqual(fit["development_metrics"]["macro_f1"], 1.0)
        self.assertGreater(fit["development_metrics"]["per_class"]["mixed"]["f1"], 0.0)

    def test_event_one_vs_rest_thresholds_and_margin_resolution(self) -> None:
        rows = []
        truths: dict[str, dict[str, str]] = {}
        for family in selector.EVENT_FAMILIES:
            article_id = f"event-{family}"
            rows.append(
                score_row(
                    article_id,
                    firm=1.0,
                    common=-1.0,
                    event_scores={family: 2.0},
                )
            )
            truths[article_id] = {"event_family": family}
        rows.append(score_row("other", firm=1.0, common=-1.0))
        truths["other"] = {"event_family": "other_or_unclear"}

        fit = selector.fit_event_thresholds(rows, truths)
        guesses = [selector.predict_event(row, fit) for row in rows]
        self.assertEqual(
            guesses, [*selector.EVENT_FAMILIES, "other_or_unclear"]
        )
        for family in selector.EVENT_FAMILIES:
            self.assertEqual(
                fit[family]["development_one_vs_rest_metrics"]["f1"], 1.0
            )


class ConservativeAlignmentTests(unittest.TestCase):
    def test_alignment_requires_exactly_one_strict_consensus_for_common_scope(self) -> None:
        same = score_row(
            "same", firm=-1.0, common=1.0, alignment_same=True
        )
        opposite = score_row(
            "opposite", firm=-1.0, common=1.0, alignment_opposite=True
        )
        both = score_row(
            "both",
            firm=1.0,
            common=1.0,
            alignment_same=True,
            alignment_opposite=True,
        )
        neither = score_row("neither", firm=-1.0, common=1.0)
        self.assertEqual(selector.predict_alignment(same, "common"), "same_direction")
        self.assertEqual(
            selector.predict_alignment(opposite, "common"), "opposite_direction"
        )
        self.assertEqual(
            selector.predict_alignment(both, "mixed"), "common_direction_unclear"
        )
        self.assertEqual(
            selector.predict_alignment(neither, "common"),
            "common_direction_unclear",
        )
        self.assertEqual(
            selector.predict_alignment(same, "idiosyncratic"), "single_firm_only"
        )
        self.assertEqual(selector.predict_alignment(same, "unclear"), "unclear")


def synthetic_metric(
    accuracy: float, macro_f1: float, *, mixed_f1: float = 0.0
) -> dict:
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": {"mixed": {"f1": mixed_f1}},
    }


class PromotionRuleTests(unittest.TestCase):
    def test_full_promotion_rule_passes_only_when_every_guard_passes(self) -> None:
        baseline = {
            field: synthetic_metric(0.50, 0.40)
            for field in selector.coarse.COARSE_FIELDS
        }
        candidate = {
            "shock_scope": synthetic_metric(0.56, 0.46, mixed_f1=0.25),
            "event_family": synthetic_metric(0.56, 0.46),
            "information_status": synthetic_metric(0.50, 0.40),
            "directional_alignment": synthetic_metric(0.56, 0.46),
        }
        decision = selector.promotion_decision(
            candidate,
            baseline,
            valid_nontruncated_rate=1.0,
            bootstrap_positive_fraction=0.96,
        )
        self.assertTrue(decision["promoted"])
        self.assertTrue(all(decision["checks"].values()))

        candidate["shock_scope"]["per_class"]["mixed"]["f1"] = 0.0
        rejected = selector.promotion_decision(
            candidate,
            baseline,
            valid_nontruncated_rate=1.0,
            bootstrap_positive_fraction=0.96,
        )
        self.assertFalse(rejected["promoted"])
        self.assertFalse(rejected["checks"]["mixed_scope_f1_is_nonzero"])

    def test_paired_bootstrap_is_seeded_and_detects_clear_improvement(self) -> None:
        schema = load_schema()
        ids = ["a", "b", "c", "d"]
        truth_labels = [
            {
                "shock_scope": "idiosyncratic",
                "event_family": "earnings_guidance",
                "information_status": "confirmed",
                "directional_alignment": "single_firm_only",
            },
            {
                "shock_scope": "common",
                "event_family": "macro_market",
                "information_status": "anticipated",
                "directional_alignment": "same_direction",
            },
            {
                "shock_scope": "mixed",
                "event_family": "regulation_legal",
                "information_status": "rumor_or_opinion",
                "directional_alignment": "opposite_direction",
            },
            {
                "shock_scope": "unclear",
                "event_family": "other_or_unclear",
                "information_status": "unclear",
                "directional_alignment": "unclear",
            },
        ]
        truth_by_id = dict(zip(ids, truth_labels))
        candidate = {
            article_id: {"labels": labels}
            for article_id, labels in zip(ids, truth_labels)
        }
        baseline = {
            article_id: {
                "labels": {
                    field: selector.FALLBACK_BY_FIELD[field]
                    for field in selector.coarse.COARSE_FIELDS
                }
            }
            for article_id in ids
        }
        first = selector.paired_bootstrap_positive_fraction(
            ids,
            truth_by_id,
            candidate,
            baseline,
            schema,
            samples=200,
            seed=123,
        )
        second = selector.paired_bootstrap_positive_fraction(
            ids,
            truth_by_id,
            candidate,
            baseline,
            schema,
            samples=200,
            seed=123,
        )
        self.assertEqual(first, second)
        self.assertGreater(first["positive_delta_fraction"], 0.95)


class CalibrationLockTests(unittest.TestCase):
    def test_calibration_requires_matching_file_and_self_hashes(self) -> None:
        schema_text = (
            ROOT / "config" / "news_feature_schema_coarse.json"
        ).read_text(encoding="utf-8")
        source_description = {
            "path": "development.jsonl",
            "sha256": "a" * 64,
            "manifest_path": "development.jsonl.manifest.json",
            "manifest_sha256": "b" * 64,
            "prompt_version": "flan-stock-sector-news-binary-v0.5.0",
            "model_id": "google/flan-t5-large",
            "model_revision": "revision",
        }
        calibration = {
            "selector_version": selector.SELECTOR_VERSION,
            "selector_source_sha256": selector.base.sha256_file(
                Path(selector.__file__).resolve()
            ),
            "schema": {
                "sha256": hashlib.sha256(schema_text.encode("utf-8")).hexdigest()
            },
            "development_article_ids": ["development-a"],
            "development_record_count": 1,
            "development_article_ids_sha256": selector.base.sha256_text(
                "development-a"
            ),
            "sources": {"development_scores": source_description},
            "source_bundle_sha256": selector.sha256_object(
                {"development_scores": source_description}
            ),
        }
        calibration["calibration_payload_sha256"] = selector.sha256_object(
            calibration
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "calibration.json"
            selector.write_json(path, calibration, overwrite=False)
            file_hash = selector.base.sha256_file(path)
            loaded = selector.validate_locked_calibration(
                path, file_hash, schema_text
            )
            self.assertEqual(loaded, calibration)
            with self.assertRaisesRegex(ValueError, "file hash mismatch"):
                selector.validate_locked_calibration(path, "0" * 64, schema_text)

            changed = dict(calibration)
            changed["selector_version"] = "tampered"
            selector.write_json(path, changed, overwrite=True)
            with self.assertRaisesRegex(ValueError, "file hash mismatch"):
                selector.validate_locked_calibration(path, file_hash, schema_text)

    def test_locked_development_scores_and_full_extractor_identity_are_replayed(
        self,
    ) -> None:
        identity = {
            "model_id": "google/flan-t5-large",
            "model_revision": "revision",
            "prompt_version": "flan-stock-sector-news-binary-v0.5.0",
            "protocol_version": "0.5.0",
            "protocol_config_sha256": "1" * 64,
            "prompt_builder_sha256": "2" * 64,
            "source_files_sha256": {"extractor.py": "3" * 64},
            "candidate_component_order": list(selector.EXPECTED_COMPONENTS),
            "binary_semantics": {"yes": "supported", "no": "not supported"},
            "max_input_tokens": 512,
            "precision": "float16",
            "model_files_sha256": {"model.safetensors": "4" * 64},
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            scores = root / "development.jsonl"
            scores.write_text('{"article_id":"development-a"}\n', encoding="utf-8")
            manifest_path = scores.with_suffix(".jsonl.manifest.json")
            manifest = {
                "status": "complete",
                "output_sha256": selector.base.sha256_file(scores),
                **identity,
            }
            selector.write_json(manifest_path, manifest, overwrite=False)
            calibration = {
                "sources": {
                    "development_scores": {
                        "path": str(scores),
                        "sha256": selector.base.sha256_file(scores),
                        "manifest_path": str(manifest_path),
                        "manifest_sha256": selector.base.sha256_file(manifest_path),
                    }
                }
            }
            loaded = selector.load_locked_development_score_manifest(calibration)
            self.assertEqual(loaded, manifest)
            selector.verify_extractor_identity(loaded, dict(manifest))

            changed = dict(manifest)
            changed["precision"] = "float32"
            with self.assertRaisesRegex(ValueError, "precision"):
                selector.verify_extractor_identity(loaded, changed)

            scores.write_text('{"article_id":"tampered"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "score file hash"):
                selector.load_locked_development_score_manifest(calibration)


if __name__ == "__main__":
    unittest.main()
