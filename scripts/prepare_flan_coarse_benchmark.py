from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import coarse_news_features as coarse
import evaluate_flan_coarse as evaluator
import extract_flan_t5 as base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map the fine GPT silver labels onto v0.3 and create a fixed stratified split."
    )
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--coarse-reference-output", required=True, type=Path)
    parser.add_argument("--development-input-output", required=True, type=Path)
    parser.add_argument("--evaluation-input-output", required=True, type=Path)
    return parser.parse_args()


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> int:
    args = parse_args()
    reference = base.read_jsonl(args.reference)
    inputs = base.read_jsonl(args.inputs)
    input_by_id = {record["article_id"]: record for record in inputs}
    if len(input_by_id) != len(inputs):
        raise ValueError("Input file contains duplicate article_id values")
    reference_ids = [record["article_id"] for record in reference]
    if len(reference_ids) != len(set(reference_ids)):
        raise ValueError("Reference file contains duplicate article_id values")
    coarse_by_id = evaluator.map_reference_records(reference, input_by_id)
    if set(coarse_by_id) != set(input_by_id):
        raise ValueError("Reference and input article_id sets must match")
    development_ids = evaluator.stable_development_ids(coarse_by_id)

    ordered_inputs = sorted(inputs, key=lambda record: record["row_number"])
    coarse_records = [coarse_by_id[record["article_id"]] for record in ordered_inputs]
    development_inputs = [
        record for record in ordered_inputs if record["article_id"] in development_ids
    ]
    evaluation_inputs = [
        record for record in ordered_inputs if record["article_id"] not in development_ids
    ]
    write_jsonl(args.coarse_reference_output, coarse_records)
    write_jsonl(args.development_input_output, development_inputs)
    write_jsonl(args.evaluation_input_output, evaluation_inputs)

    manifest = {
        "reference_type": "deterministically coarsened GPT-5.6 Sol silver annotations",
        "source_reference": str(args.reference),
        "source_reference_sha256": base.sha256_file(args.reference),
        "source_inputs": str(args.inputs),
        "source_inputs_sha256": base.sha256_file(args.inputs),
        "coarse_schema_path": str(coarse.DEFAULT_SCHEMA_PATH),
        "coarse_schema_version": coarse.load_schema()["schema_version"],
        "coarse_schema_sha256": base.sha256_file(coarse.DEFAULT_SCHEMA_PATH),
        "coarse_mapping_module_sha256": base.sha256_file(Path(coarse.__file__).resolve()),
        "coarse_reference_count": len(coarse_records),
        "development_count": len(development_inputs),
        "evaluation_count": len(evaluation_inputs),
        "split_seed": "flan-t5-coarse-v0.3-development",
        "split_method": "within-shock_scope seeded sha256 order, rounded 24 percent",
        "warning": "This is a development benchmark derived from GPT silver labels, not new human ground truth.",
        "outputs": {
            "coarse_reference_sha256": base.sha256_file(args.coarse_reference_output),
            "development_inputs_sha256": base.sha256_file(args.development_input_output),
            "evaluation_inputs_sha256": base.sha256_file(args.evaluation_input_output),
        },
    }
    manifest_path = args.coarse_reference_output.with_suffix(
        args.coarse_reference_output.suffix + ".manifest.json"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(coarse_records)} coarse references, {len(development_inputs)} development inputs, "
        f"and {len(evaluation_inputs)} evaluation inputs"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
