from __future__ import annotations

import hashlib
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT / "experiments" / "flan_t5"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FlanExperimentRegistryTests(unittest.TestCase):
    def test_v0_4_is_active_research_baseline_and_v0_5_is_archived(self) -> None:
        registry = load_json(EXPERIMENT_ROOT / "registry.json")
        active = load_json(EXPERIMENT_ROOT / "active_baseline.json")
        self.assertIsNone(registry["active_frozen_version"])
        self.assertEqual(registry["active_baseline_version"], "v0_4")
        self.assertEqual(active["version"], registry["active_baseline_version"])
        self.assertEqual(active["status"], "active_research_baseline")
        self.assertIs(active["production_ready"], False)
        self.assertEqual(registry["best_historical_version"], "v0_4")
        versions = {row["version"]: row for row in registry["versions"]}
        self.assertEqual(versions["v0_4"]["status"], "active_research_baseline")
        self.assertEqual(versions["v0_5"]["status"], "archived_rejected")
        self.assertIs(versions["v0_5"]["promotion_decision"], False)
        for version in versions.values():
            for key in ("protocol_path", "evaluation_summary_path"):
                path = version[key]
                if path is not None:
                    self.assertTrue((ROOT / path).is_file(), path)

    def test_preregistered_files_still_match_the_pre_score_lock(self) -> None:
        archive = EXPERIMENT_ROOT / "archive" / "v0_5"
        lock = load_json(archive / "protocol_lock.json")
        self.assertEqual(
            sha256_file(archive / "PROTOCOL.md"),
            lock["tracked_sources"]["protocol"]["sha256"],
        )
        self.assertEqual(
            sha256_file(archive / "promotion_rule.json"),
            lock["tracked_sources"]["promotion_rule"]["sha256"],
        )

    def test_compact_result_records_a_failed_promotion(self) -> None:
        summary = load_json(
            EXPERIMENT_ROOT / "archive" / "v0_5" / "evaluation_summary.json"
        )
        self.assertEqual(summary["status"], "rejected")
        self.assertIs(summary["promoted"], False)
        aggregate = summary["aggregate_metrics"]
        self.assertLess(aggregate["mean_macro_f1_delta"], 0.0)
        self.assertLess(aggregate["mean_accuracy_delta"], 0.0)
        self.assertEqual(summary["bootstrap"]["positive_delta_fraction"], 0.0)
        self.assertIsNone(summary["active_frozen_version"])
        checks = summary["promotion_checks"]
        self.assertTrue(checks["mixed_scope_f1_is_nonzero"])
        self.assertTrue(checks["valid_and_nontruncated_rate_is_100_percent"])
        self.assertFalse(
            checks["mean_macro_f1_delta_at_least_0_02"]
        )
        self.assertFalse(
            checks[
                "paired_bootstrap_positive_delta_fraction_at_least_0_95"
            ]
        )

    def test_experiment_document_links_resolve(self) -> None:
        documents = [
            ROOT / "README.md",
            EXPERIMENT_ROOT / "README.md",
            EXPERIMENT_ROOT / "archive" / "README.md",
            EXPERIMENT_ROOT
            / "archive"
            / "repository_reorganization"
            / "README.md",
            EXPERIMENT_ROOT
            / "archive"
            / "v0_1_v0_2"
            / "protocol_and_null_fix.md",
            EXPERIMENT_ROOT / "archive" / "v0_3" / "README.md",
            EXPERIMENT_ROOT / "v0_4" / "README.md",
            EXPERIMENT_ROOT / "archive" / "v0_5" / "README.md",
            ROOT / "experiments" / "llama_2" / "README.md",
        ]
        for document in documents:
            text = document.read_text(encoding="utf-8")
            for target in re.findall(r"\]\(([^)]+)\)", text):
                if target.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                path_text = target.split("#", 1)[0]
                resolved = (document.parent / path_text).resolve()
                self.assertTrue(
                    resolved.exists(),
                    f"{document.relative_to(ROOT)} has broken link {target!r}",
                )

    def test_only_active_and_archive_namespaces_remain_at_flan_top_level(self) -> None:
        self.assertTrue((EXPERIMENT_ROOT / "v0_4").is_dir())
        self.assertTrue((EXPERIMENT_ROOT / "archive").is_dir())
        for obsolete in ("legacy", "candidates", "rejected"):
            self.assertFalse((EXPERIMENT_ROOT / obsolete).exists(), obsolete)


if __name__ == "__main__":
    unittest.main()
