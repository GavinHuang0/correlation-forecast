from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import coarse_news_features as coarse
import extract_flan_t5 as base


HYBRID_PROMPT_VERSION = "flan-stock-sector-news-hybrid-v0.4.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine development-selected FLAN-only fields: v0.2 scope/alignment and v0.4 event/status."
        )
    )
    parser.add_argument("--coarse-predictions", required=True, type=Path)
    parser.add_argument("--fine-v0-2-predictions", required=True, type=Path)
    parser.add_argument("--schema", default=Path("config/news_feature_schema_coarse.json"), type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_complete_manifest(prediction_path: Path) -> dict[str, Any]:
    manifest_path = prediction_path.with_suffix(prediction_path.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"Incomplete source manifest: {manifest_path}")
    if manifest.get("output_sha256") != base.sha256_file(prediction_path):
        raise ValueError(f"Source prediction hash mismatch: {prediction_path}")
    return manifest


def legacy_pass_truncated(record: dict[str, Any], pass_names: tuple[str, ...]) -> bool:
    passes = record.get("passes", {})
    return any(bool(passes.get(name, {}).get("input_truncated")) for name in pass_names)


def main() -> int:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite")
    schema_text = args.schema.read_text(encoding="utf-8")
    schema = json.loads(schema_text)
    coarse.validate_schema(schema)
    coarse_predictions = base.read_jsonl(args.coarse_predictions)
    fine_predictions = base.read_jsonl(args.fine_v0_2_predictions)
    coarse_by_id = {record["article_id"]: record for record in coarse_predictions}
    fine_by_id = {record["article_id"]: record for record in fine_predictions}
    if len(coarse_by_id) != len(coarse_predictions) or len(fine_by_id) != len(fine_predictions):
        raise ValueError("Source predictions contain duplicate article_id values")
    if not set(coarse_by_id) <= set(fine_by_id):
        raise ValueError("Fine v0.2 predictions do not cover every coarse prediction")

    coarse_manifest = load_complete_manifest(args.coarse_predictions)
    fine_manifest = load_complete_manifest(args.fine_v0_2_predictions)
    if coarse_manifest.get("prompt_version") != "flan-stock-sector-news-v0.4.0":
        raise ValueError("Coarse source is not a v0.4 extraction run")
    if coarse_manifest.get("prompt_profile") != "zero_shot":
        raise ValueError("Hybrid selection was frozen for the zero-shot v0.4 profile")
    if coarse_manifest.get("decoding") != "order_averaged_letter_score":
        raise ValueError("Hybrid selection requires v0.4 order-averaged letter scoring")
    if fine_manifest.get("prompt_version") != "flan-stock-sector-news-v0.2.0":
        raise ValueError("Fine source is not the frozen v0.2 extraction run")
    for key in ("model_id", "model_revision"):
        if coarse_manifest.get(key) != fine_manifest.get(key):
            raise ValueError(f"FLAN source manifests disagree on {key}")

    combined: list[dict[str, Any]] = []
    for coarse_record in coarse_predictions:
        article_id = coarse_record["article_id"]
        fine_record = fine_by_id[article_id]
        if coarse_record.get("target_ticker") != fine_record.get("target_ticker"):
            raise ValueError(f"Target mismatch between FLAN sources for {article_id}")
        result = json.loads(json.dumps(coarse_record))
        result["prompt_version"] = HYBRID_PROMPT_VERSION
        if result.get("semantic_applicable"):
            mapped = coarse.map_fine_labels(fine_record["labels"])
            replacements = {
                "shock_scope": {
                    "fine_fields": {"event_scope": fine_record["labels"]["event_scope"]},
                    "fine_passes": ("event_scope",),
                },
                "directional_alignment": {
                    "fine_fields": {
                        "event_scope": fine_record["labels"]["event_scope"],
                        "target_direction": fine_record["labels"]["target_direction"],
                        "sector_direction": fine_record["labels"]["sector_direction"],
                        "peer_effect": fine_record["labels"]["peer_effect"],
                    },
                    "fine_passes": (
                        "event_scope",
                        "target_direction",
                        "sector_direction",
                        "peer_effect",
                    ),
                },
            }
            for field, detail in replacements.items():
                value = mapped[field]
                if value not in schema["closed_label_fields"][field]:
                    raise ValueError(f"Mapped v0.2 value is invalid: {field}={value!r}")
                result["labels"][field] = value
                result["label_origins"][field] = "flan_v0_2_mapped"
                result["passes"][field] = {
                    "origin": "flan_v0_2_mapped",
                    "value": value,
                    "source_fine_labels": detail["fine_fields"],
                    "source_prediction_sha256": base.sha256_text(
                        json.dumps(fine_record, sort_keys=True, separators=(",", ":"))
                    ),
                    "valid": True,
                    "input_truncated": legacy_pass_truncated(
                        fine_record, detail["fine_passes"]
                    ),
                }
        result["validity"]["no_input_truncation"] = not any(
            pass_record.get("input_truncated")
            for pass_record in result["passes"].values()
        )
        combined.append(result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in combined:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    manifest = {
        "status": "complete",
        "prompt_version": HYBRID_PROMPT_VERSION,
        "selection_basis": "fixed 72-article development split",
        "field_sources": {
            "shock_scope": "FLAN v0.2 event_scope mapped to coarse labels",
            "event_family": "FLAN v0.4 order-averaged coarse prompt",
            "information_status": "FLAN v0.4 order-averaged coarse prompt",
            "directional_alignment": "FLAN v0.2 scope and direction fields mapped to coarse labels",
            "semantic_applicable": "deterministic gate",
            "explicit_surprise_rule": "deterministic text rule",
        },
        "model_id": coarse_manifest["model_id"],
        "model_revision": coarse_manifest["model_revision"],
        "device": coarse_manifest.get("device"),
        "precision": coarse_manifest.get("precision"),
        "decoding": "development_selected_fieldwise_flan_ensemble",
        "prompt_profile": coarse_manifest.get("prompt_profile"),
        "schema_name": schema["schema_name"],
        "schema_version": schema["schema_version"],
        "schema_sha256": hashlib.sha256(schema_text.encode("utf-8")).hexdigest(),
        "deterministic_rule_version": coarse_manifest.get("deterministic_rule_version"),
        "deterministic_rules_payload_sha256": coarse_manifest.get(
            "deterministic_rules_payload_sha256"
        ),
        "deterministic_module_sha256": coarse_manifest.get("deterministic_module_sha256"),
        "source_files": {
            "coarse_predictions": str(args.coarse_predictions),
            "coarse_predictions_sha256": base.sha256_file(args.coarse_predictions),
            "coarse_manifest_sha256": base.sha256_file(
                args.coarse_predictions.with_suffix(args.coarse_predictions.suffix + ".manifest.json")
            ),
            "fine_v0_2_predictions": str(args.fine_v0_2_predictions),
            "fine_v0_2_predictions_sha256": base.sha256_file(args.fine_v0_2_predictions),
            "fine_v0_2_manifest_sha256": base.sha256_file(
                args.fine_v0_2_predictions.with_suffix(
                    args.fine_v0_2_predictions.suffix + ".manifest.json"
                )
            ),
        },
        "total_prediction_count": len(combined),
        "selected_article_ids_sha256": base.sha256_text(
            "\n".join(record["article_id"] for record in combined)
        ),
        "semantic_applicable_count": sum(
            int(record["semantic_applicable"]) for record in combined
        ),
        "output_sha256": base.sha256_file(args.output),
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    base.write_manifest(manifest_path, manifest)
    print(f"Wrote {len(combined)} FLAN-only hybrid predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
