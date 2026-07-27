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


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load_module(
    "prepare_gpt56_silver_batches_for_test",
    "prepare_gpt56_silver_batches.py",
)
merge = load_module(
    "merge_gpt56_silver_labels_for_test",
    "merge_gpt56_silver_labels.py",
)


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def target(ticker: str) -> dict:
    return {
        "company": f"{ticker} Synthetic Corporation",
        "ticker": ticker,
        "sector": "Synthetic Components",
        "sector_benchmark": "SYN",
        "known_sector_peers": ["AAA", "BBB", "CCC", "DDD"],
    }


def parent_record(number: int, ticker: str) -> dict:
    article_id = f"parent-{number}"
    return {
        "row_number": number,
        "article_id": article_id,
        "time_published_utc": f"2026-03-{number + 1:02d}T09:00:00Z",
        "source": "Synthetic Wire",
        "headline": f"{ticker} synthetic headline {number}",
        "article_text": "Provider description that must not enter GPT batches.",
        "vendor_tickers": [ticker],
        "target": target(ticker),
        "text_variant": "massive_description",
        "benchmark_split": "development" if number == 1 else "evaluation",
        "source_match_method": "native_massive_description",
    }


def chunk_records(parent: dict) -> list[dict]:
    article_id = parent["article_id"]
    first = f"Complete parent body {article_id} first paragraph."
    second = f"Confirmed synthetic event for {article_id}."
    result = []
    for index, text in enumerate((first, second)):
        result.append(
            {
                "row_number": (parent["row_number"] - 1) * 2 + index + 1,
                "article_id": f"{article_id}__chunk_{index:04d}",
                "chunk_id": f"{article_id}__chunk_{index:04d}",
                "parent_article_id": article_id,
                "chunk_index": index,
                "chunk_count": 2,
                "time_published_utc": parent["time_published_utc"],
                "source": parent["source"],
                "headline": parent["headline"],
                "article_text": text,
                "vendor_tickers": parent["vendor_tickers"],
                "target": parent["target"],
                "text_variant": "fulltext_evidence_chunks",
                "benchmark_split": parent["benchmark_split"],
                "source_match_method": "exact_id",
            }
        )
    return result


def silver_row(article_id: str) -> dict:
    return {
        "article_id": article_id,
        "annotator": "gpt-5.6-sol",
        "protocol_version": "news-fulltext-gpt-silver-v1.0.0",
        "labels": {
            "shock_scope": "idiosyncratic",
            "event_family": "other_or_unclear",
            "information_status": "confirmed",
            "directional_alignment": "single_firm_only",
        },
        "evidence": {
            "shock_scope": f"Complete parent body {article_id}",
            "event_family": "",
            "information_status": f"Confirmed synthetic event for {article_id}.",
            "directional_alignment": f"Complete parent body {article_id}",
        },
        "abstain_reason": None,
    }


class GPTSilverWorkflowTests(unittest.TestCase):
    def build_private_batches(self, root: Path) -> tuple[Path, list[dict]]:
        parents = [
            parent_record(1, "AAA"),
            parent_record(2, "BBB"),
            parent_record(3, "CCC"),
            parent_record(4, "DDD"),
        ]
        chunks = [
            row
            for parent in parents
            for row in reversed(chunk_records(parent))
        ]
        parent_path = root / "massive_description.jsonl"
        chunk_path = root / "fulltext_evidence_chunks.jsonl"
        output_root = root / "gpt_batches"
        write_jsonl(parent_path, list(reversed(parents)))
        write_jsonl(chunk_path, chunks)
        manifest = prepare.prepare_batches(
            parent_path=parent_path,
            chunk_path=chunk_path,
            schema_path=ROOT / "config" / "news_feature_schema_coarse.json",
            protocol_path=(
                ROOT
                / "experiments"
                / "news_provider_fulltext"
                / "v1_0"
                / "GPT_SILVER_PROTOCOL.md"
            ),
            output_root=output_root,
            cutoff="2026-03-01T00:00:00Z",
            batch_size=2,
            expected_documents=4,
        )
        return output_root, manifest

    def test_batches_reconstruct_complete_parents_in_canonical_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_root, manifest = self.build_private_batches(root)
            self.assertEqual(manifest["document_count"], 4)
            self.assertEqual(manifest["batch_count"], 2)
            self.assertEqual(
                manifest["article_ids"],
                ["parent-1", "parent-2", "parent-3", "parent-4"],
            )
            rows = read_jsonl(output_root / "batch_001.jsonl")
            self.assertEqual([row["article_id"] for row in rows], ["parent-1", "parent-2"])
            self.assertEqual(
                rows[0]["full_text"],
                "Complete parent body parent-1 first paragraph.\n\n"
                "Confirmed synthetic event for parent-1.",
            )
            self.assertNotIn("Provider description", rows[0]["full_text"])
            self.assertEqual(
                set(rows[0]),
                {"document_number", "article_id", "headline", "full_text", "target"},
            )
            self.assertEqual(manifest["documents"][0]["chunk_count"], 2)
            self.assertTrue(
                all(entry["contains_licensed_full_text"] for entry in manifest["batches"])
            )

    def test_reconstruction_rejects_missing_chunk_index(self) -> None:
        parent = parent_record(1, "AAA")
        chunks = chunk_records(parent)
        chunks.pop(0)
        with self.assertRaisesRegex(
            prepare.BatchPreparationError, "expected chunk indices"
        ):
            prepare.reconstruct_parent_documents(
                [parent],
                chunks,
                expected_documents=1,
            )

    def test_merge_validates_grounding_and_strips_licensed_text(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            batch_root, _ = self.build_private_batches(root)
            parts = root / "annotations" / "parts.jsonl"
            write_jsonl(
                parts,
                [silver_row(f"parent-{number}") for number in range(1, 5)],
            )
            output = root / "annotations" / "merged.jsonl"
            manifest_output = root / "annotations" / "manifest.json"
            manifest = merge.validate_and_merge(
                batch_root=batch_root,
                part_paths=[parts],
                schema_path=ROOT / "config" / "news_feature_schema_coarse.json",
                output_path=output,
                manifest_output_path=manifest_output,
            )
            rows = read_jsonl(output)
            self.assertEqual(len(rows), 4)
            self.assertEqual(manifest["document_count"], 4)
            self.assertEqual(manifest["reference_kind"], "silver_not_ground_truth")
            self.assertTrue(
                manifest["grounding"]["all_nonempty_evidence_grounded"]
            )
            self.assertFalse(manifest["privacy"]["contains_licensed_text"])
            serialized = output.read_text(encoding="utf-8")
            self.assertNotIn("Complete parent body", serialized)
            self.assertNotIn("Confirmed synthetic event", serialized)
            self.assertNotIn("headline", serialized)
            self.assertNotIn('"evidence":', serialized)
            self.assertTrue(
                all(row["evidence_grounded"]["shock_scope"] for row in rows)
            )
            self.assertTrue(
                all(row["evidence_sha256"]["shock_scope"] for row in rows)
            )

    def test_merge_rejects_ungrounded_evidence_and_missing_parents(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            batch_root, _ = self.build_private_batches(root)
            invalid_rows = [
                silver_row(f"parent-{number}") for number in range(1, 5)
            ]
            invalid_rows[0]["evidence"]["shock_scope"] = "not in the source"
            invalid_part = root / "invalid.jsonl"
            write_jsonl(invalid_part, invalid_rows)
            with self.assertRaisesRegex(
                merge.SilverLabelValidationError, "not an exact source substring"
            ):
                merge.validate_and_merge(
                    batch_root=batch_root,
                    part_paths=[invalid_part],
                    schema_path=ROOT / "config" / "news_feature_schema_coarse.json",
                    output_path=root / "invalid-merged.jsonl",
                    manifest_output_path=root / "invalid-manifest.json",
                )

            missing_part = root / "missing.jsonl"
            write_jsonl(missing_part, invalid_rows[:3])
            with self.assertRaisesRegex(
                merge.SilverLabelValidationError, "Missing labels"
            ):
                merge.validate_and_merge(
                    batch_root=batch_root,
                    part_paths=[missing_part],
                    schema_path=ROOT / "config" / "news_feature_schema_coarse.json",
                    output_path=root / "missing-merged.jsonl",
                    manifest_output_path=root / "missing-manifest.json",
                )

    def test_merge_rejects_labels_outside_the_frozen_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            batch_root, _ = self.build_private_batches(root)
            rows = [silver_row(f"parent-{number}") for number in range(1, 5)]
            rows[2]["labels"]["shock_scope"] = "sector_wide"
            part = root / "invalid-label.jsonl"
            write_jsonl(part, rows)
            with self.assertRaisesRegex(
                merge.SilverLabelValidationError, "invalid shock_scope label"
            ):
                merge.validate_and_merge(
                    batch_root=batch_root,
                    part_paths=[part],
                    schema_path=ROOT / "config" / "news_feature_schema_coarse.json",
                    output_path=root / "invalid-label-merged.jsonl",
                    manifest_output_path=root / "invalid-label-manifest.json",
                )

    def test_merge_rejects_scope_alignment_inconsistent_with_local_runners(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            batch_root, _ = self.build_private_batches(root)
            rows = [silver_row(f"parent-{number}") for number in range(1, 5)]
            rows[1]["labels"]["directional_alignment"] = (
                "common_direction_unclear"
            )
            part = root / "invalid-hierarchy.jsonl"
            write_jsonl(part, rows)
            with self.assertRaisesRegex(
                merge.SilverLabelValidationError,
                "inconsistent with shock_scope",
            ):
                merge.validate_and_merge(
                    batch_root=batch_root,
                    part_paths=[part],
                    schema_path=ROOT / "config" / "news_feature_schema_coarse.json",
                    output_path=root / "invalid-hierarchy-merged.jsonl",
                    manifest_output_path=root / "invalid-hierarchy-manifest.json",
                )


if __name__ == "__main__":
    unittest.main()
