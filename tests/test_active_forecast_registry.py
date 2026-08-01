from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_PATH = ROOT / "models" / "active" / "registry.json"
ARCHIVE_PATH = ROOT / "models" / "archive" / "registry.json"
QUANT_SUMMARY_PATH = (
    ROOT
    / "experiments"
    / "quant_training"
    / "v2"
    / "comparisons"
    / "summary.json"
)
V4_COMPARISON_PATH = (
    ROOT
    / "experiments"
    / "quant_deterministic_news"
    / "v4"
    / "training"
    / "comparisons"
    / "final"
    / "summary.json"
)
RRES_SUMMARY_PATH = (
    ROOT
    / "experiments"
    / "quant_deterministic_news"
    / "v4"
    / "training"
    / "models"
    / "rres_coupling6_en"
    / "summary.json"
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class ActiveForecastRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.active = load_json(ACTIVE_PATH)
        self.archive = load_json(ARCHIVE_PATH)

    def test_registry_has_four_quant_routes_and_one_semantic_overlay(self):
        quant = self.active["quantitative_models"]
        semantic = self.active["semantic_overlays"]

        self.assertEqual(len(quant), 4)
        self.assertEqual(len(semantic), 1)
        self.assertEqual(
            {row["target"] for row in quant},
            {"t1_etf", "t1_loo", "t2_etf", "t2_loo"},
        )
        self.assertEqual(semantic[0]["id"], "rres_c6_t2_etf")
        self.assertEqual(semantic[0]["target"], "t2_etf")

    def test_quant_routes_equal_stored_target_level_winners(self):
        summary = load_json(QUANT_SUMMARY_PATH)
        winners = {row["target"]: row for row in summary["best_overall"]}

        for selected in self.active["quantitative_models"]:
            source = winners[selected["target"]]
            self.assertEqual(selected["rung"], source["rung"])
            self.assertEqual(selected["model"], source["model"])
            self.assertEqual(selected["evaluation"], {
                "rows": source["rows"],
                "fisher_z_mae": source["fisher_z_mae"],
                "fisher_z_rmse": source["fisher_z_rmse"],
                "raw_correlation_mae": source["raw_correlation_mae"],
                "raw_correlation_rmse": source["raw_correlation_rmse"],
                "oos_r2_vs_persistence": source[
                    "oos_r2_vs_persistence"
                ],
            })

    def test_rres_c6_route_equals_stored_model_and_paired_result(self):
        selected = self.active["semantic_overlays"][0]
        model_summary = load_json(RRES_SUMMARY_PATH)
        model_metric = next(
            row for row in model_summary["metrics"]
            if row["target"] == "t2_etf"
        )
        comparison_summary = load_json(V4_COMPARISON_PATH)
        comparison = next(
            row for row in comparison_summary["comparisons"]
            if row["comparison"] == "RRES_C6_vs_R0"
            and row["target"] == "t2_etf"
        )
        evaluation = selected["evaluation"]

        for key in (
            "rows",
            "fisher_z_mae",
            "fisher_z_rmse",
            "raw_correlation_mae",
            "raw_correlation_rmse",
            "oos_r2_vs_persistence",
        ):
            self.assertEqual(evaluation[key], model_metric[key])

        self.assertTrue(comparison["point_loss_delta_gate_pass"])
        self.assertTrue(comparison["loss_delta_ci_upper_gate_pass"])
        self.assertTrue(comparison["fold_improvement_gate_pass"])
        self.assertEqual(
            evaluation["incremental_mse_r2_vs_matched_base"],
            comparison["incremental_mse_r2"],
        )
        self.assertEqual(
            evaluation["incremental_mse_r2_bootstrap_95pct"],
            comparison["incremental_mse_r2_bootstrap_95pct"],
        )
        self.assertEqual(
            evaluation["bootstrap_probability_gain_nonpositive"],
            comparison[
                "bootstrap_probability_incremental_mse_r2_le_zero"
            ],
        )
        self.assertEqual(evaluation["folds_improved"], 3)
        self.assertEqual(evaluation["folds_total"], 5)

    def test_public_evidence_paths_and_archive_cross_links_resolve(self):
        evidence_paths = [
            row["evidence"]
            for row in self.active["quantitative_models"]
        ]
        overlay = self.active["semantic_overlays"][0]
        evidence_paths.extend(
            [overlay["model_card"], overlay["comparison_evidence"]]
        )
        for relative in evidence_paths:
            self.assertTrue((ROOT / relative).is_file(), relative)

        self.assertEqual(
            self.active["archive_registry"],
            "models/archive/registry.json",
        )
        self.assertEqual(
            self.archive["active_registry"],
            "models/active/registry.json",
        )


if __name__ == "__main__":
    unittest.main()
