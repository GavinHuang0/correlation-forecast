"""Validate and merge private GPT-5.6 Sol coarse silver-label parts.

Raw part files may contain short quoted evidence for grounding checks.  The
merged artifact intentionally removes the full evidence strings, the article
headline, and full text.  It retains only coarse labels, provenance, grounding
booleans, and evidence hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from prepare_gpt56_silver_batches import (
    DEFAULT_OUTPUT_ROOT as DEFAULT_BATCH_ROOT,
    REPOSITORY_ROOT,
    canonical_json,
    read_jsonl,
    sha256_bytes,
    sha256_file,
    stable_id_hash,
    write_json,
    write_jsonl,
)


DEFAULT_SCHEMA = REPOSITORY_ROOT / "config" / "news_feature_schema_coarse.json"
DEFAULT_PARTS_ROOT = (
    REPOSITORY_ROOT
    / "annotations"
    / "news_provider_fulltext"
    / "v1_0"
    / "parts"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "annotations"
    / "news_provider_fulltext"
    / "v1_0"
    / "gpt_5_6_sol_silver.jsonl"
)
DEFAULT_MANIFEST_OUTPUT = DEFAULT_OUTPUT.with_name(
    "gpt_5_6_sol_silver.manifest.json"
)
ANNOTATOR = "gpt-5.6-sol"
PROTOCOL_VERSION = "news-fulltext-gpt-silver-v1.0.0"
MANIFEST_VERSION = "gpt-5.6-sol-silver-merge-v1"
MAX_EVIDENCE_CHARACTERS = 280
COARSE_FIELDS = (
    "shock_scope",
    "event_family",
    "information_status",
    "directional_alignment",
)
FORBIDDEN_MERGED_TEXT_KEYS = frozenset(
    {
        "article_text",
        "full_text",
        "body",
        "content",
        "headline",
        "summary",
        "description",
        "evidence",
        "abstain_reason",
    }
)


class SilverLabelValidationError(ValueError):
    """Raised when a silver-label part or merge output is invalid."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate one GPT-5.6 Sol label per parent document, ground quoted "
            "evidence, and merge labels without licensed text."
        )
    )
    parser.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH_ROOT)
    parser.add_argument(
        "--parts",
        action="append",
        type=Path,
        default=[],
        help="JSONL part file or directory; repeat as needed.",
    )
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--manifest-output", type=Path, default=DEFAULT_MANIFEST_OUTPUT
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only the configured merged output and manifest.",
    )
    return parser.parse_args()


def discover_part_files(paths: Sequence[Path]) -> list[Path]:
    supplied = list(paths) or [DEFAULT_PARTS_ROOT]
    discovered: set[Path] = set()
    for path in supplied:
        if path.is_file():
            if path.suffix.casefold() != ".jsonl":
                raise SilverLabelValidationError(
                    f"Part file must use .jsonl: {path}"
                )
            discovered.add(path.resolve())
        elif path.is_dir():
            discovered.update(value.resolve() for value in path.rglob("*.jsonl"))
        else:
            raise FileNotFoundError(path)
    if not discovered:
        raise SilverLabelValidationError("No GPT label part files were found")
    return sorted(discovered, key=lambda value: str(value).casefold())


def load_schema(path: Path) -> dict[str, Any]:
    try:
        schema = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SilverLabelValidationError(f"Invalid schema JSON: {exc}") from exc
    closed = schema.get("closed_label_fields")
    if not isinstance(closed, Mapping) or tuple(closed) != COARSE_FIELDS:
        raise SilverLabelValidationError(
            f"Schema fields must be ordered exactly as {COARSE_FIELDS!r}"
        )
    for field in COARSE_FIELDS:
        values = closed[field]
        if not isinstance(values, list) or not values or not all(
            isinstance(value, str) and value for value in values
        ):
            raise SilverLabelValidationError(
                f"Schema field {field!r} has invalid labels"
            )
    return schema


def load_batch_documents(
    batch_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], Path]:
    manifest_path = batch_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SilverLabelValidationError(
            f"Invalid batch manifest JSON: {exc}"
        ) from exc
    if manifest.get("manifest_version") != "gpt-5.6-sol-silver-batches-v1":
        raise SilverLabelValidationError("Unexpected batch manifest_version")
    batches = manifest.get("batches")
    if not isinstance(batches, list) or not batches:
        raise SilverLabelValidationError("Batch manifest has no batches")

    documents: dict[str, dict[str, Any]] = {}
    ordered_ids: list[str] = []
    for entry in batches:
        if not isinstance(entry, Mapping):
            raise SilverLabelValidationError("Batch manifest entry must be an object")
        filename = entry.get("file")
        if not isinstance(filename, str) or not filename:
            raise SilverLabelValidationError("Batch entry is missing file")
        batch_path = (batch_root / filename).resolve()
        try:
            batch_path.relative_to(batch_root.resolve())
        except ValueError as exc:
            raise SilverLabelValidationError(
                f"Batch path escapes batch root: {filename!r}"
            ) from exc
        if sha256_file(batch_path) != entry.get("sha256"):
            raise SilverLabelValidationError(
                f"Batch file hash differs from manifest: {batch_path}"
            )
        records = read_jsonl(batch_path)
        if len(records) != entry.get("document_count"):
            raise SilverLabelValidationError(
                f"Batch record count differs from manifest: {batch_path}"
            )
        for record in records:
            article_id = record.get("article_id")
            if not isinstance(article_id, str) or not article_id:
                raise SilverLabelValidationError(
                    f"{batch_path}: missing article_id"
                )
            if article_id in documents:
                raise SilverLabelValidationError(
                    f"Duplicate batch article_id {article_id!r}"
                )
            full_text = record.get("full_text")
            headline = record.get("headline")
            target = record.get("target")
            if not isinstance(full_text, str) or not full_text.strip():
                raise SilverLabelValidationError(
                    f"{article_id!r}: full_text must be nonempty"
                )
            if not isinstance(headline, str):
                raise SilverLabelValidationError(
                    f"{article_id!r}: headline must be a string"
                )
            if not isinstance(target, Mapping) or not isinstance(
                target.get("ticker"), str
            ):
                raise SilverLabelValidationError(
                    f"{article_id!r}: target.ticker is required"
                )
            documents[article_id] = record
            ordered_ids.append(article_id)

    if ordered_ids != manifest.get("article_ids"):
        raise SilverLabelValidationError(
            "Batch article order differs from manifest article_ids"
        )
    if stable_id_hash(ordered_ids) != manifest.get("article_ids_sha256"):
        raise SilverLabelValidationError(
            "Batch article_ids hash differs from manifest"
        )
    document_manifest = manifest.get("documents")
    if not isinstance(document_manifest, list):
        raise SilverLabelValidationError("Batch manifest has no document metadata")
    metadata_by_id: dict[str, Mapping[str, Any]] = {}
    for metadata in document_manifest:
        if not isinstance(metadata, Mapping):
            raise SilverLabelValidationError(
                "Batch document metadata entry must be an object"
            )
        article_id = metadata.get("article_id")
        if not isinstance(article_id, str) or article_id in metadata_by_id:
            raise SilverLabelValidationError(
                "Batch document metadata has an invalid or duplicate article_id"
            )
        metadata_by_id[article_id] = metadata
    if set(metadata_by_id) != set(documents):
        raise SilverLabelValidationError(
            "Batch document metadata IDs differ from batch records"
        )
    for article_id, document in documents.items():
        metadata = metadata_by_id[article_id]
        if metadata.get("target_ticker") != document["target"]["ticker"]:
            raise SilverLabelValidationError(
                f"Batch target ticker differs from manifest for {article_id!r}"
            )
        if sha256_bytes(document["full_text"].encode("utf-8")) != metadata.get(
            "full_text_sha256"
        ):
            raise SilverLabelValidationError(
                f"Batch full-text hash differs from manifest for {article_id!r}"
            )
        document["_benchmark_split"] = metadata.get("benchmark_split", "")
    return documents, manifest, manifest_path


def _validate_labels(
    value: Any, schema: Mapping[str, Any], *, context: str
) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(COARSE_FIELDS):
        raise SilverLabelValidationError(
            f"{context}: labels must contain exactly the fields "
            f"{COARSE_FIELDS!r}"
        )
    result: dict[str, str] = {}
    for field in COARSE_FIELDS:
        label = value[field]
        if label not in schema["closed_label_fields"][field]:
            raise SilverLabelValidationError(
                f"{context}: invalid {field} label {label!r}"
            )
        result[field] = str(label)
    scope = result["shock_scope"]
    alignment = result["directional_alignment"]
    alignment_by_scope = {
        "unclear": {"unclear"},
        "idiosyncratic": {
            "single_firm_only",
            "same_direction",
            "opposite_direction",
        },
        "common": {
            "same_direction",
            "opposite_direction",
            "common_direction_unclear",
        },
        "mixed": {
            "same_direction",
            "opposite_direction",
            "common_direction_unclear",
        },
    }
    if alignment not in alignment_by_scope[scope]:
        raise SilverLabelValidationError(
            f"{context}: directional_alignment={alignment!r} is inconsistent "
            f"with shock_scope={scope!r}"
        )
    return result


def _validate_evidence(
    value: Any, source_text: str, *, context: str
) -> tuple[dict[str, bool], dict[str, str | None]]:
    if not isinstance(value, Mapping) or set(value) != set(COARSE_FIELDS):
        raise SilverLabelValidationError(
            f"{context}: evidence must contain exactly the fields "
            f"{COARSE_FIELDS!r}"
        )
    present: dict[str, bool] = {}
    hashes: dict[str, str | None] = {}
    for field in COARSE_FIELDS:
        evidence = value[field]
        if not isinstance(evidence, str):
            raise SilverLabelValidationError(
                f"{context}: evidence.{field} must be a string"
            )
        if len(evidence) > MAX_EVIDENCE_CHARACTERS:
            raise SilverLabelValidationError(
                f"{context}: evidence.{field} exceeds "
                f"{MAX_EVIDENCE_CHARACTERS} characters"
            )
        has_evidence = bool(evidence)
        if has_evidence and evidence not in source_text:
            raise SilverLabelValidationError(
                f"{context}: evidence.{field} is not an exact source substring"
            )
        present[field] = has_evidence
        hashes[field] = (
            sha256_bytes(evidence.encode("utf-8")) if has_evidence else None
        )
    return present, hashes


def _validate_abstain_reason(value: Any, *, context: str) -> bool:
    if value is None:
        return False
    if not isinstance(value, str) or not value.strip():
        raise SilverLabelValidationError(
            f"{context}: abstain_reason must be null or a nonempty string"
        )
    if len(value) > 500:
        raise SilverLabelValidationError(
            f"{context}: abstain_reason exceeds 500 characters"
        )
    return True


def _forbidden_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if key in FORBIDDEN_MERGED_TEXT_KEYS:
                found.add(str(key))
            found.update(_forbidden_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            found.update(_forbidden_keys(nested))
    return found


def validate_and_merge(
    *,
    batch_root: Path = DEFAULT_BATCH_ROOT,
    part_paths: Sequence[Path] = (),
    schema_path: Path = DEFAULT_SCHEMA,
    output_path: Path = DEFAULT_OUTPUT,
    manifest_output_path: Path = DEFAULT_MANIFEST_OUTPUT,
    overwrite: bool = False,
) -> dict[str, Any]:
    if output_path.resolve() == manifest_output_path.resolve():
        raise SilverLabelValidationError(
            "Merged output and manifest output must be different files"
        )
    existing = [
        path for path in (output_path, manifest_output_path) if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            "Merge output exists; pass --overwrite to replace configured files: "
            + ", ".join(str(path) for path in existing)
        )

    schema = load_schema(schema_path)
    documents, batch_manifest, batch_manifest_path = load_batch_documents(
        batch_root
    )
    files = discover_part_files(part_paths)
    raw_by_id: dict[str, dict[str, Any]] = {}
    part_manifest: list[dict[str, Any]] = []
    for path in files:
        rows = read_jsonl(path)
        part_manifest.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "record_count": len(rows),
            }
        )
        for row_index, row in enumerate(rows, start=1):
            context = f"{path}:{row_index}"
            allowed_keys = {
                "article_id",
                "annotator",
                "protocol_version",
                "labels",
                "evidence",
                "abstain_reason",
            }
            if set(row) != allowed_keys:
                raise SilverLabelValidationError(
                    f"{context}: top-level keys must be exactly "
                    f"{sorted(allowed_keys)!r}"
                )
            article_id = row.get("article_id")
            if not isinstance(article_id, str) or not article_id:
                raise SilverLabelValidationError(
                    f"{context}: article_id must be a nonempty string"
                )
            if article_id not in documents:
                raise SilverLabelValidationError(
                    f"{context}: unknown article_id {article_id!r}"
                )
            if article_id in raw_by_id:
                raise SilverLabelValidationError(
                    f"Duplicate label for article_id {article_id!r}"
                )
            if row.get("annotator") != ANNOTATOR:
                raise SilverLabelValidationError(
                    f"{context}: annotator must be {ANNOTATOR!r}"
                )
            if row.get("protocol_version") != PROTOCOL_VERSION:
                raise SilverLabelValidationError(
                    f"{context}: protocol_version must be {PROTOCOL_VERSION!r}"
                )
            raw_by_id[article_id] = row

    expected_ids = list(batch_manifest["article_ids"])
    missing = [article_id for article_id in expected_ids if article_id not in raw_by_id]
    if missing:
        raise SilverLabelValidationError(
            f"Missing labels for {len(missing)} parent documents; first={missing[:5]!r}"
        )
    if len(raw_by_id) != len(expected_ids):
        raise SilverLabelValidationError(
            "Label count differs from parent-document count"
        )

    distributions = {field: Counter() for field in COARSE_FIELDS}
    evidence_counts = Counter()
    abstention_count = 0
    merged: list[dict[str, Any]] = []
    for row_number, article_id in enumerate(expected_ids, start=1):
        raw = raw_by_id[article_id]
        source = documents[article_id]
        context = f"article_id={article_id!r}"
        labels = _validate_labels(raw.get("labels"), schema, context=context)
        source_text = f"{source['headline']}\n\n{source['full_text']}"
        evidence_present, evidence_hashes = _validate_evidence(
            raw.get("evidence"), source_text, context=context
        )
        abstained = _validate_abstain_reason(
            raw.get("abstain_reason"), context=context
        )
        abstention_count += int(abstained)
        for field in COARSE_FIELDS:
            distributions[field][labels[field]] += 1
            evidence_counts[field] += int(evidence_present[field])
        result = {
            "row_number": row_number,
            "article_id": article_id,
            "target_ticker": source["target"]["ticker"],
            "benchmark_split": source.get("_benchmark_split", ""),
            "annotator": ANNOTATOR,
            "protocol_version": PROTOCOL_VERSION,
            "labels": labels,
            "evidence_present": evidence_present,
            "evidence_sha256": evidence_hashes,
            "evidence_grounded": {
                field: True if evidence_present[field] else None
                for field in COARSE_FIELDS
            },
            "abstained": abstained,
        }
        forbidden = _forbidden_keys(result)
        if forbidden:
            raise AssertionError(
                f"Merged output contains forbidden text keys: {sorted(forbidden)!r}"
            )
        merged.append(result)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, merged)
    manifest: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "status": "complete",
        "annotator": ANNOTATOR,
        "protocol_version": PROTOCOL_VERSION,
        "reference_kind": "silver_not_ground_truth",
        "annotation_execution": {
            "surface": "Codex collaboration agent",
            "requested_model": ANNOTATOR,
            "dated_api_snapshot_pinned": False,
            "platform_system_instructions_replaced": False,
            "protocol_delivery": (
                "tracked protocol file plus the collaboration task message"
            ),
        },
        "document_count": len(merged),
        "article_ids": expected_ids,
        "article_ids_sha256": stable_id_hash(expected_ids),
        "labels_sha256": sha256_bytes(
            canonical_json(
                [
                    {"article_id": row["article_id"], "labels": row["labels"]}
                    for row in merged
                ]
            ).encode("utf-8")
        ),
        "distribution": {
            field: dict(sorted(counts.items()))
            for field, counts in distributions.items()
        },
        "grounding": {
            "method": (
                "case-sensitive exact substring of supplied headline or complete "
                "reconstructed parent full text"
            ),
            "maximum_evidence_characters": MAX_EVIDENCE_CHARACTERS,
            "evidence_present_by_field": {
                field: evidence_counts[field] for field in COARSE_FIELDS
            },
            "all_nonempty_evidence_grounded": True,
            "quoted_evidence_removed_from_merged_output": True,
        },
        "abstention_count": abstention_count,
        "privacy": {
            "contains_article_full_text": False,
            "contains_headlines": False,
            "contains_quoted_evidence": False,
            "contains_abstain_reason_text": False,
            "contains_licensed_text": False,
        },
        "inputs": {
            "batch_manifest": {
                "path": str(batch_manifest_path.resolve()),
                "sha256": sha256_file(batch_manifest_path),
            },
            "schema": {
                "path": str(schema_path.resolve()),
                "sha256": sha256_file(schema_path),
            },
            "parts": part_manifest,
        },
        "output": {
            "path": str(output_path.resolve()),
            "sha256": sha256_file(output_path),
            "bytes": output_path.stat().st_size,
        },
    }
    write_json(manifest_output_path, manifest)
    return manifest


def main() -> int:
    args = parse_args()
    manifest = validate_and_merge(
        batch_root=args.batch_root,
        part_paths=args.parts,
        schema_path=args.schema,
        output_path=args.output,
        manifest_output_path=args.manifest_output,
        overwrite=args.overwrite,
    )
    print(
        f"Validated and merged {manifest['document_count']} GPT-5.6 Sol "
        f"silver labels to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
