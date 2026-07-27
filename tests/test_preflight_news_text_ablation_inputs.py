from __future__ import annotations

import importlib.util
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
    "preflight_news_text_ablation_for_test",
    SCRIPTS / "preflight_news_text_ablation_inputs.py",
)
assert SPEC is not None and SPEC.loader is not None
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


class WordTokenizer:
    def __call__(self, text, **_kwargs):
        return {"input_ids": list(range(len(text.split())))}


def record(article_id: str, words: int) -> dict:
    return {
        "row_number": 1,
        "article_id": article_id,
        "time_published_utc": "2026-03-02T09:00:00Z",
        "source": "Synthetic",
        "headline": "Synthetic headline",
        "article_text": "word " * words,
        "vendor_tickers": ["AMD"],
        "target": {
            "company": "Advanced Micro Devices",
            "ticker": "AMD",
            "sector": "Semiconductors",
            "sector_benchmark": "SOXX",
            "known_sector_peers": ["NVDA", "INTC"],
        },
    }


class NewsTextAblationPreflightTests(unittest.TestCase):
    def test_explicit_zero_context_limit_is_not_replaced_by_default(self) -> None:
        self.assertEqual(preflight.resolve_max_input_tokens(None, 512), 512)
        self.assertEqual(preflight.resolve_max_input_tokens(0, 512), 0)

    def test_flan_prompt_counts_use_all_fields_and_find_violations(self) -> None:
        schema = json.loads(
            (ROOT / "config" / "news_feature_schema_coarse.json").read_text(
                encoding="utf-8"
            )
        )
        counts = preflight.flan_prompt_counts(
            [record("short", 10), record("long", 400)],
            schema,
            WordTokenizer(),
        )
        self.assertEqual(
            {item["field"] for item in counts},
            {
                "shock_scope",
                "event_family",
                "information_status",
                "directional_alignment",
            },
        )
        summary = preflight.summarize_counts(counts, max_input_tokens=300)
        self.assertFalse(summary["passes_context_limit"])
        self.assertGreater(summary["violation_count"], 0)
        self.assertIn(
            "long",
            {item["article_id"] for item in summary["violations"]},
        )

    def test_snapshot_contract_hashes_only_frozen_tokenizer_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            records = {}
            for filename in ("spiece.model", "tokenizer_config.json"):
                path = snapshot / filename
                path.write_text(f"synthetic-{filename}", encoding="utf-8")
                records[filename] = {
                    "size_bytes": path.stat().st_size,
                    "sha256": preflight.sha256_file(path),
                }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "manifest_version": "flan-t5-xl-local-snapshot-v1",
                        "model_id": "google/flan-t5-xl",
                        "model_revision": (
                            "7d6315df2c2fb742f0f5b556879d730926ca9001"
                        ),
                        "snapshot_path": str(snapshot),
                        "files": records,
                    }
                ),
                encoding="utf-8",
            )
            manifest, resolved = preflight.load_snapshot_contract(
                manifest_path, "flan-t5-xl"
            )
            self.assertEqual(resolved, snapshot.resolve())
            self.assertEqual(manifest["model_id"], "google/flan-t5-xl")


if __name__ == "__main__":
    unittest.main()
