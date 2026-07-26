"""Prepare the fixed 300-document benchmark for Llama 3.1.

The script reuses the already-audited FLAN/Llama 2 inputs byte for byte. It
does not call or import an LLM and never places silver labels in extractor
inputs.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import extract_flan_t5 as base
import prepare_llama_2_benchmark as benchmark_common


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = (
    ROOT / "outputs" / "flan_t5" / "shared" / "benchmark_300" / "annotation_batches"
)
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "llama_3_1" / "shared" / "benchmark_300"
KNOWLEDGE_CUTOFF = date(2023, 12, 31)
MANIFEST_VERSION = "llama-3.1-benchmark-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and byte-copy the fixed 72/228 benchmark splits for "
            "Llama 3.1. No model is downloaded or loaded."
        )
    )
    parser.add_argument(
        "--all-inputs", type=Path, default=SOURCE_ROOT / "all_inputs.jsonl"
    )
    parser.add_argument(
        "--development-inputs",
        type=Path,
        default=SOURCE_ROOT / "coarse_v0_3_development_inputs.jsonl",
    )
    parser.add_argument(
        "--evaluation-inputs",
        type=Path,
        default=SOURCE_ROOT / "coarse_v0_3_evaluation_inputs.jsonl",
    )
    parser.add_argument(
        "--fine-reference",
        type=Path,
        default=ROOT / "annotations" / "chatgpt_5_6_sol_reference.jsonl",
    )
    parser.add_argument(
        "--coarse-reference",
        type=Path,
        default=ROOT
        / "annotations"
        / "chatgpt_5_6_sol_reference_coarse_v0_2.jsonl",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=ROOT / "config" / "news_feature_schema_coarse.json",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def model_specific_manifest(manifest: dict) -> dict:
    result = dict(manifest)
    result.update(
        {
            "manifest_version": MANIFEST_VERSION,
            "dataset_name": "Llama 3.1 reuse of fixed FLAN 300-article benchmark",
            "model_eligibility_basis": {
                "model_family": "Llama 3.1",
                "official_knowledge_cutoff": "December 2023",
                "conservative_cutoff": KNOWLEDGE_CUTOFF.isoformat(),
            },
            "knowledge_cutoff": KNOWLEDGE_CUTOFF.isoformat(),
            "warning": (
                "These are GPT-5.6-derived silver labels, not human ground truth. "
                "The 72-record development split is for model/protocol selection; "
                "freeze the configuration before evaluating the other 228 records."
            ),
        }
    )
    return result


def main() -> int:
    args = parse_args()
    paths = benchmark_common.BenchmarkPaths(
        all_inputs=args.all_inputs,
        development_inputs=args.development_inputs,
        evaluation_inputs=args.evaluation_inputs,
        fine_reference=args.fine_reference,
        coarse_reference=args.coarse_reference,
        schema=args.schema,
    )
    contract = benchmark_common.BenchmarkContract(
        benchmark_common.EXPECTED_SOURCE_HASHES,
        knowledge_cutoff=KNOWLEDGE_CUTOFF,
    )
    manifest = model_specific_manifest(
        benchmark_common.validate_benchmark_sources(paths, contract=contract)
    )
    if args.validate_only:
        manifest["mode"] = "validate_only"
        print(json.dumps(manifest, indent=2))
        return 0

    development_output = args.output_root / "development_inputs.jsonl"
    evaluation_output = args.output_root / "evaluation_inputs.jsonl"
    manifest_output = args.output_root / "manifest.json"
    outputs = (development_output, evaluation_output, manifest_output)
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Prepared Llama 3.1 benchmark already exists; pass --overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    benchmark_common._atomic_copy(args.development_inputs, development_output)
    benchmark_common._atomic_copy(args.evaluation_inputs, evaluation_output)
    if development_output.read_bytes() != args.development_inputs.read_bytes():
        raise RuntimeError("Development input byte-copy verification failed")
    if evaluation_output.read_bytes() != args.evaluation_inputs.read_bytes():
        raise RuntimeError("Evaluation input byte-copy verification failed")
    manifest.update(
        {
            "mode": "prepared",
            "prepared_outputs": {
                "development_inputs": {
                    "path": benchmark_common._path_for_manifest(development_output),
                    "sha256": base.sha256_file(development_output),
                    "byte_identical_to_source": True,
                },
                "evaluation_inputs": {
                    "path": benchmark_common._path_for_manifest(evaluation_output),
                    "sha256": base.sha256_file(evaluation_output),
                    "byte_identical_to_source": True,
                },
            },
        }
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"Prepared Llama 3.1 benchmark at {args.output_root} "
        f"({manifest['counts']['development']} development, "
        f"{manifest['counts']['evaluation']} evaluation)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
