from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

SPEC = importlib.util.spec_from_file_location(
    "prepare_llama_2_benchmark_for_test",
    SCRIPTS / "prepare_llama_2_benchmark.py",
)
assert SPEC is not None and SPEC.loader is not None
prepare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prepare
SPEC.loader.exec_module(prepare)

import coarse_news_features as coarse  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, records: list[dict], *, pretty_spacing: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    separators = None if pretty_spacing else (",", ":")
    payload = "".join(
        json.dumps(record, ensure_ascii=False, separators=separators) + "\n"
        for record in records
    )
    path.write_text(payload, encoding="utf-8", newline="\n")


def input_record(index: int, timestamp: str = "2024-02-01T12:00:00Z") -> dict:
    return {
        "row_number": index,
        "article_id": f"article-{index}",
        "time_published_utc": timestamp,
        "source": "Example Wire",
        "headline": f"AMD update {index}",
        "article_text": "Advanced Micro Devices announced an update.",
        "vendor_tickers": ["AMD"],
        "target": {
            "company": "Advanced Micro Devices",
            "ticker": "AMD",
            "sector": "Semiconductors",
            "sector_benchmark": "SOXX",
            "known_sector_peers": ["NVDA", "INTC"],
        },
    }


def fine_labels() -> dict:
    return {
        "relevance": "direct_target",
        "event_scope": "firm_specific",
        "event_type": "product_technology",
        "affected_breadth": "single_firm",
        "target_direction": "neutral",
        "sector_direction": "not_applicable",
        "peer_effect": "not_applicable",
        "explicit_surprise": "none",
        "information_status": "confirmed",
        "transmission_channels": ["technology_product"],
        "affected_companies": ["Advanced Micro Devices"],
        "affected_sectors": [],
        "evidence": {"scope": "", "direction": "", "surprise": ""},
        "abstain_reason": None,
    }


def make_sources(root: Path) -> tuple[prepare.BenchmarkPaths, prepare.BenchmarkContract]:
    schema = ROOT / "config" / "news_feature_schema_coarse.json"
    records = [input_record(index) for index in range(1, 5)]
    development = records[:1]
    evaluation = records[1:]
    fine = [
        {
            "row_number": record["row_number"],
            "article_id": record["article_id"],
            "target_ticker": "AMD",
            "labels": fine_labels(),
        }
        for record in records
    ]
    coarse_records = [
        {
            "row_number": record["row_number"],
            "article_id": record["article_id"],
            "target_ticker": "AMD",
            "semantic_applicable": True,
            "labels": coarse.map_fine_labels(fine_labels()),
        }
        for record in records
    ]
    paths = prepare.BenchmarkPaths(
        all_inputs=root / "all.jsonl",
        development_inputs=root / "development.jsonl",
        evaluation_inputs=root / "evaluation.jsonl",
        fine_reference=root / "fine.jsonl",
        coarse_reference=root / "coarse.jsonl",
        schema=schema,
    )
    write_jsonl(paths.all_inputs, records, pretty_spacing=True)
    write_jsonl(paths.development_inputs, development, pretty_spacing=True)
    write_jsonl(paths.evaluation_inputs, evaluation, pretty_spacing=True)
    write_jsonl(paths.fine_reference, fine)
    write_jsonl(paths.coarse_reference, coarse_records)
    expected_hashes = {
        name: sha256(getattr(paths, name))
        for name in (
            "all_inputs",
            "development_inputs",
            "evaluation_inputs",
            "fine_reference",
            "coarse_reference",
            "schema",
        )
    }
    contract = prepare.BenchmarkContract(
        expected_hashes=expected_hashes,
        all_count=4,
        development_count=1,
        evaluation_count=3,
            knowledge_cutoff=date(2023, 7, 31),
    )
    return paths, contract


class BenchmarkPreparationTests(unittest.TestCase):
    def test_preparation_byte_copies_inputs_and_keeps_labels_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths, contract = make_sources(root / "source")
            output = root / "prepared"
            manifest = prepare.prepare_benchmark(
                paths, output, contract=contract
            )

            self.assertEqual(
                (output / "development_inputs.jsonl").read_bytes(),
                paths.development_inputs.read_bytes(),
            )
            self.assertEqual(
                (output / "evaluation_inputs.jsonl").read_bytes(),
                paths.evaluation_inputs.read_bytes(),
            )
            self.assertFalse(
                any(
                    "labels" in json.loads(line)
                    for path in (
                        output / "development_inputs.jsonl",
                        output / "evaluation_inputs.jsonl",
                    )
                    for line in path.read_text(encoding="utf-8").splitlines()
                )
            )
            self.assertEqual(manifest["counts"], {"all": 4, "development": 1, "evaluation": 3})
            self.assertEqual(manifest["manifest_version"], "llama-2-benchmark-v1")
            self.assertEqual(manifest["knowledge_cutoff"], "2023-07-31")
            self.assertEqual(
                manifest["model_eligibility_basis"]["latest_reported_training_data"],
                "July 2023",
            )
            self.assertTrue(manifest["all_articles_after_cutoff"])
            self.assertFalse(manifest["labels_present_in_extractor_inputs"])
            stored = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(stored, manifest)

    def test_validate_only_performs_checks_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths, contract = make_sources(root / "source")
            output = root / "prepared"
            manifest = prepare.prepare_benchmark(
                paths, output, contract=contract, validate_only=True
            )
            self.assertEqual(manifest["mode"], "validate_only")
            self.assertFalse(output.exists())

    def test_existing_outputs_require_explicit_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths, contract = make_sources(root / "source")
            output = root / "prepared"
            prepare.prepare_benchmark(paths, output, contract=contract)
            with self.assertRaisesRegex(FileExistsError, "--overwrite"):
                prepare.prepare_benchmark(paths, output, contract=contract)
            prepare.prepare_benchmark(
                paths, output, contract=contract, overwrite=True
            )

    def test_pre_cutoff_article_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths, _ = make_sources(root / "source")
            all_records = [
                json.loads(line)
                for line in paths.all_inputs.read_text(encoding="utf-8").splitlines()
            ]
            all_records[0]["time_published_utc"] = "2023-07-31T23:59:59Z"
            write_jsonl(paths.all_inputs, all_records, pretty_spacing=True)
            write_jsonl(paths.development_inputs, all_records[:1], pretty_spacing=True)
            expected = {
                name: sha256(getattr(paths, name))
                for name in (
                    "all_inputs",
                    "development_inputs",
                    "evaluation_inputs",
                    "fine_reference",
                    "coarse_reference",
                    "schema",
                )
            }
            contract = prepare.BenchmarkContract(
                expected_hashes=expected,
                all_count=4,
                development_count=1,
                evaluation_count=3,
            )
            with self.assertRaisesRegex(ValueError, "not after"):
                prepare.validate_benchmark_sources(paths, contract=contract)

    def test_overlapping_splits_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths, _ = make_sources(root / "source")
            all_records = [
                json.loads(line)
                for line in paths.all_inputs.read_text(encoding="utf-8").splitlines()
            ]
            write_jsonl(
                paths.evaluation_inputs,
                [all_records[0], all_records[2], all_records[3]],
                pretty_spacing=True,
            )
            expected = {
                name: sha256(getattr(paths, name))
                for name in (
                    "all_inputs",
                    "development_inputs",
                    "evaluation_inputs",
                    "fine_reference",
                    "coarse_reference",
                    "schema",
                )
            }
            contract = prepare.BenchmarkContract(
                expected_hashes=expected,
                all_count=4,
                development_count=1,
                evaluation_count=3,
            )
            with self.assertRaisesRegex(ValueError, "overlap"):
                prepare.validate_benchmark_sources(paths, contract=contract)


if __name__ == "__main__":
    unittest.main()
