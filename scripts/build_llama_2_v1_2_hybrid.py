from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import coarse_news_features as coarse
import experiment_llama_2_v1_1 as v1_1
import extract_flan_t5 as base
import extract_llama_2_coarse as v1_0


HYBRID_VERSION = "llama-2-stock-sector-news-v1.2.0-development-selected-hybrid"
MANIFEST_VERSION = "llama-2-fieldwise-hybrid-v1"
FIELD_SOURCE = {
    "shock_scope": "v1_0",
    "event_family": "v1_1",
    "information_status": "v1_1",
    "directional_alignment": "v1_0",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the development-selected Llama 2 v1.2 fieldwise hybrid. "
            "No model is loaded: scope/alignment come from v1.0 and event/status "
            "come from the corrected v1.1 answer-cue decoder."
        )
    )
    parser.add_argument("--v1-0-predictions", required=True, type=Path)
    parser.add_argument("--v1-1-predictions", required=True, type=Path)
    parser.add_argument(
        "--v1-0-manifest",
        type=Path,
        help="Defaults to <v1-0-predictions>.manifest.json.",
    )
    parser.add_argument(
        "--v1-1-manifest",
        type=Path,
        help="Defaults to <v1-1-predictions>.manifest.json.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _by_id(
    records: Sequence[Mapping[str, Any]], *, source: str
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for record in records:
        article_id = record.get("article_id")
        if not isinstance(article_id, str) or not article_id:
            raise ValueError(f"{source} has a record without article_id")
        if article_id in result:
            raise ValueError(f"{source} has duplicate article_id {article_id}")
        result[article_id] = record
    return result


def validate_source_pair(
    old: Mapping[str, Any], corrected: Mapping[str, Any]
) -> None:
    for key in (
        "row_number",
        "article_id",
        "target_ticker",
        "model_revision",
        "protocol_version",
        "semantic_applicable",
        "deterministic_features",
    ):
        if old.get(key) != corrected.get(key):
            raise ValueError(
                f"{old.get('article_id')}: source mismatch for {key}"
            )
    if old.get("model_revision") != v1_0.REVISION_DEFAULT:
        raise ValueError("v1.0 source uses the wrong model revision")
    if old.get("prompt_version") != v1_0.PROMPT_VERSION:
        raise ValueError("v1.0 source uses the wrong prompt version")
    if corrected.get("prompt_version") != v1_1.PROMPT_VERSION:
        raise ValueError("v1.1 source uses the wrong corrected prompt version")


def build_hybrid_rows(
    old_rows: Sequence[Mapping[str, Any]],
    corrected_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    old_by_id = _by_id(old_rows, source="v1.0 predictions")
    corrected_by_id = _by_id(corrected_rows, source="v1.1 predictions")
    ordered_ids = [str(row["article_id"]) for row in old_rows]
    if set(old_by_id) != set(corrected_by_id):
        raise ValueError("v1.0 and v1.1 article IDs differ")
    if [row["article_id"] for row in corrected_rows] != ordered_ids:
        raise ValueError("v1.0 and v1.1 row order differs")

    results: list[dict[str, Any]] = []
    for article_id in ordered_ids:
        old = old_by_id[article_id]
        corrected = corrected_by_id[article_id]
        validate_source_pair(old, corrected)
        labels: dict[str, Any] = {}
        origins: dict[str, Any] = {}
        passes: dict[str, Any] = {}
        for field in coarse.COARSE_FIELDS:
            source = old if FIELD_SOURCE[field] == "v1_0" else corrected
            labels[field] = source["labels"][field]
            origins[field] = source["label_origins"][field]
            passes[field] = {
                **source["passes"][field],
                "hybrid_source": FIELD_SOURCE[field],
            }
        results.append(
            {
                "row_number": old["row_number"],
                "article_id": article_id,
                "target_ticker": old["target_ticker"],
                "extractor": v1_0.MODEL_DEFAULT,
                "model_revision": v1_0.REVISION_DEFAULT,
                "prompt_version": HYBRID_VERSION,
                "protocol_version": old["protocol_version"],
                "semantic_applicable": old["semantic_applicable"],
                "deterministic_features": old["deterministic_features"],
                "labels": labels,
                "label_origins": origins,
                "passes": passes,
                "hybrid_field_source": dict(FIELD_SOURCE),
                "validity": {
                    "all_fields_resolved_when_applicable": (
                        all(labels[field] is not None for field in coarse.COARSE_FIELDS)
                        if old["semantic_applicable"]
                        else True
                    ),
                    "no_input_truncation": (
                        old["validity"]["no_input_truncation"]
                        and corrected["validity"]["no_input_truncation"]
                    ),
                },
            }
        )
    return results


def validate_source_manifest(
    manifest: Mapping[str, Any],
    *,
    source_name: str,
    predictions_path: Path,
    prompt_version: str,
    expected_count: int,
) -> None:
    expected = {
        "status": "complete",
        "model_id": v1_0.MODEL_DEFAULT,
        "model_revision": v1_0.REVISION_DEFAULT,
        "prompt_version": prompt_version,
        "total_prediction_count": expected_count,
        "output_sha256": base.sha256_file(predictions_path),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"{source_name} manifest mismatch for {key}: "
                f"{manifest.get(key)!r} != {value!r}"
            )
    if manifest.get("device") != "cuda" or manifest.get("precision") != "float16":
        raise ValueError(f"{source_name} did not use frozen CUDA/float16")
    quantization = manifest.get("quantization")
    if not isinstance(quantization, dict):
        raise ValueError(f"{source_name} has no quantization manifest")
    expected_quantization = {
        "method": "nf4",
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "compute_dtype": "float16",
        "hf_device_map": {"": 0},
    }
    if quantization != expected_quantization:
        raise ValueError(
            f"{source_name} quantization mismatch: {quantization!r}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    if (args.output.exists() or manifest_path.exists()) and not args.overwrite:
        raise FileExistsError("Hybrid output exists; pass --overwrite")
    old_manifest_path = args.v1_0_manifest or args.v1_0_predictions.with_suffix(
        args.v1_0_predictions.suffix + ".manifest.json"
    )
    corrected_manifest_path = (
        args.v1_1_manifest
        or args.v1_1_predictions.with_suffix(
            args.v1_1_predictions.suffix + ".manifest.json"
        )
    )
    for path in (
        args.v1_0_predictions,
        args.v1_1_predictions,
        old_manifest_path,
        corrected_manifest_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    old_rows = base.read_jsonl(args.v1_0_predictions)
    corrected_rows = base.read_jsonl(args.v1_1_predictions)
    if not old_rows or not corrected_rows:
        raise ValueError("Both source prediction files must be nonempty")
    old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8"))
    corrected_manifest = json.loads(
        corrected_manifest_path.read_text(encoding="utf-8")
    )
    validate_source_manifest(
        old_manifest,
        source_name="v1.0",
        predictions_path=args.v1_0_predictions,
        prompt_version=v1_0.PROMPT_VERSION,
        expected_count=len(old_rows),
    )
    validate_source_manifest(
        corrected_manifest,
        source_name="v1.1",
        predictions_path=args.v1_1_predictions,
        prompt_version=v1_1.PROMPT_VERSION,
        expected_count=len(corrected_rows),
    )
    for key in ("input_sha256", "selected_article_ids_sha256", "schema_sha256"):
        if old_manifest.get(key) != corrected_manifest.get(key):
            raise ValueError(f"Source manifests disagree on {key}")
    results = build_hybrid_rows(old_rows, corrected_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for result in results:
            handle.write(
                json.dumps(result, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "status": "complete",
        "hybrid_version": HYBRID_VERSION,
        "development_selected": True,
        "model_id": v1_0.MODEL_DEFAULT,
        "model_revision": v1_0.REVISION_DEFAULT,
        "field_source": FIELD_SOURCE,
        "source_v1_0_path": str(args.v1_0_predictions),
        "source_v1_0_sha256": base.sha256_file(args.v1_0_predictions),
        "source_v1_0_manifest_path": str(old_manifest_path),
        "source_v1_0_manifest_sha256": base.sha256_file(old_manifest_path),
        "source_v1_1_path": str(args.v1_1_predictions),
        "source_v1_1_sha256": base.sha256_file(args.v1_1_predictions),
        "source_v1_1_manifest_path": str(corrected_manifest_path),
        "source_v1_1_manifest_sha256": base.sha256_file(
            corrected_manifest_path
        ),
        "input_sha256": old_manifest["input_sha256"],
        "schema_sha256": old_manifest["schema_sha256"],
        "builder_source_sha256": base.sha256_file(Path(__file__).resolve()),
        "record_count": len(results),
        "selected_article_ids_sha256": base.sha256_text(
            "\n".join(result["article_id"] for result in results)
        ),
        "output_sha256": base.sha256_file(args.output),
        "claim_limit": (
            "The field assignment was selected on the 72-document development "
            "split. A fresh holdout is required for confirmation."
        ),
    }
    base.write_manifest(manifest_path, manifest)
    print(f"Wrote {len(results)} hybrid predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
