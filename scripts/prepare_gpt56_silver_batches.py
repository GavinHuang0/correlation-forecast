"""Prepare deterministic parent-document batches for GPT-5.6 silver labeling.

This script is deliberately data-only: it does not call a model or a provider.
It reconstructs each complete parent document from the benchmark's ordered
``fulltext_evidence_chunks.jsonl`` rows and writes private, licensed-text
batches below the Git-ignored ``data/`` tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_ROOT = (
    REPOSITORY_ROOT / "data" / "benchmarks" / "news_text_ablation_300" / "v1"
)
DEFAULT_PARENT_INPUT = DEFAULT_BENCHMARK_ROOT / "massive_description.jsonl"
DEFAULT_CHUNK_INPUT = DEFAULT_BENCHMARK_ROOT / "fulltext_evidence_chunks.jsonl"
DEFAULT_OUTPUT_ROOT = DEFAULT_BENCHMARK_ROOT / "gpt_batches"
DEFAULT_SCHEMA = REPOSITORY_ROOT / "config" / "news_feature_schema_coarse.json"
DEFAULT_PROTOCOL = (
    REPOSITORY_ROOT
    / "experiments"
    / "news_provider_fulltext"
    / "v1_0"
    / "GPT_SILVER_PROTOCOL.md"
)
DEFAULT_CUTOFF = "2026-03-01T00:00:00Z"
DEFAULT_BATCH_SIZE = 20
DEFAULT_EXPECTED_DOCUMENTS = 300
MANIFEST_VERSION = "gpt-5.6-sol-silver-batches-v1"


class BatchPreparationError(ValueError):
    """Raised when the private benchmark violates the batching contract."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct complete parent full text and split it into deterministic "
            "private batches for GPT-5.6 Sol silver labeling."
        )
    )
    parser.add_argument("--parents", type=Path, default=DEFAULT_PARENT_INPUT)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNK_INPUT)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cutoff", default=DEFAULT_CUTOFF)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--expected-documents",
        type=int,
        default=DEFAULT_EXPECTED_DOCUMENTS,
        help="Set to 0 to accept any positive parent-document count.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only known batch_*.jsonl files and manifest.json.",
    )
    return parser.parse_args()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id_hash(values: Iterable[str]) -> str:
    return sha256_bytes("\x1f".join(values).encode("utf-8"))


def parse_timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise BatchPreparationError(f"Invalid timestamp {value!r}")
    text = value.strip()
    normalized = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise BatchPreparationError(f"Invalid timestamp {value!r}") from exc
    if parsed.tzinfo is None:
        raise BatchPreparationError(
            f"Timestamp must include a timezone: {value!r}"
        )
    return parsed.astimezone(timezone.utc)


def format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BatchPreparationError(
                    f"{path}:{line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise BatchPreparationError(
                    f"{path}:{line_number}: record must be an object"
                )
            records.append(record)
    if not records:
        raise BatchPreparationError(f"{path} contains no records")
    return records


def write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(canonical_json(record) + "\n")
    os.replace(temporary, path)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _nonempty_string(record: Mapping[str, Any], field: str, context: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise BatchPreparationError(f"{context}: {field} must be a nonempty string")
    return value.strip()


def _validate_target(target: Any, context: str) -> dict[str, Any]:
    if not isinstance(target, Mapping):
        raise BatchPreparationError(f"{context}: target must be an object")
    expected = {
        "company",
        "ticker",
        "sector",
        "sector_benchmark",
        "known_sector_peers",
    }
    if set(target) != expected:
        raise BatchPreparationError(
            f"{context}: target keys must be exactly {sorted(expected)!r}"
        )
    for field in ("company", "ticker", "sector", "sector_benchmark"):
        _nonempty_string(target, field, f"{context}.target")
    if not isinstance(target["known_sector_peers"], list) or not all(
        isinstance(value, str) and value.strip()
        for value in target["known_sector_peers"]
    ):
        raise BatchPreparationError(
            f"{context}.target: known_sector_peers must be a string list"
        )
    return dict(target)


def _parent_sort_key(record: Mapping[str, Any]) -> tuple[int, str]:
    row_number = record.get("row_number")
    if not isinstance(row_number, int) or row_number < 1:
        raise BatchPreparationError("Every parent needs a positive integer row_number")
    return row_number, str(record.get("article_id", ""))


def reconstruct_parent_documents(
    parent_records: Sequence[Mapping[str, Any]],
    chunk_records: Sequence[Mapping[str, Any]],
    *,
    cutoff: str | datetime = DEFAULT_CUTOFF,
    expected_documents: int = DEFAULT_EXPECTED_DOCUMENTS,
) -> list[dict[str, Any]]:
    """Return complete parent documents in the canonical parent-file order."""

    cutoff_time = parse_timestamp(cutoff) if isinstance(cutoff, str) else cutoff
    cutoff_time = cutoff_time.astimezone(timezone.utc)
    parents_by_id: dict[str, Mapping[str, Any]] = {}
    row_numbers: set[int] = set()
    for index, parent in enumerate(parent_records, start=1):
        context = f"parent record {index}"
        article_id = _nonempty_string(parent, "article_id", context)
        if article_id in parents_by_id:
            raise BatchPreparationError(f"Duplicate parent article_id {article_id!r}")
        row_number = parent.get("row_number")
        if not isinstance(row_number, int) or row_number < 1:
            raise BatchPreparationError(
                f"{context}: row_number must be a positive integer"
            )
        if row_number in row_numbers:
            raise BatchPreparationError(f"Duplicate parent row_number {row_number}")
        row_numbers.add(row_number)
        published = parse_timestamp(
            _nonempty_string(parent, "time_published_utc", context)
        )
        if published <= cutoff_time:
            raise BatchPreparationError(
                f"{context}: publication must be strictly after "
                f"{format_utc(cutoff_time)}"
            )
        _nonempty_string(parent, "headline", context)
        _validate_target(parent.get("target"), context)
        parents_by_id[article_id] = parent

    if expected_documents > 0 and len(parents_by_id) != expected_documents:
        raise BatchPreparationError(
            f"Expected {expected_documents} parent documents, found "
            f"{len(parents_by_id)}"
        )

    chunks_by_parent: dict[str, dict[int, Mapping[str, Any]]] = {}
    declared_counts: dict[str, set[int]] = {}
    reconstructed_hashes: dict[str, set[str]] = {}
    source_hashes: dict[str, set[str]] = {}
    for index, chunk in enumerate(chunk_records, start=1):
        context = f"chunk record {index}"
        parent_id = _nonempty_string(chunk, "parent_article_id", context)
        if parent_id not in parents_by_id:
            raise BatchPreparationError(
                f"{context}: unknown parent_article_id {parent_id!r}"
            )
        chunk_index = chunk.get("chunk_index")
        chunk_count = chunk.get("chunk_count")
        if not isinstance(chunk_index, int) or chunk_index < 0:
            raise BatchPreparationError(
                f"{context}: chunk_index must be a nonnegative integer"
            )
        if not isinstance(chunk_count, int) or chunk_count < 1:
            raise BatchPreparationError(
                f"{context}: chunk_count must be a positive integer"
            )
        text = _nonempty_string(chunk, "article_text", context)
        if chunk.get("text_variant") not in (None, "fulltext_evidence_chunks"):
            raise BatchPreparationError(
                f"{context}: expected text_variant='fulltext_evidence_chunks'"
            )
        parent = parents_by_id[parent_id]
        for field in ("headline", "time_published_utc", "benchmark_split"):
            if chunk.get(field) != parent.get(field):
                raise BatchPreparationError(
                    f"{context}: {field} differs from parent {parent_id!r}"
                )
        if canonical_json(chunk.get("target")) != canonical_json(parent.get("target")):
            raise BatchPreparationError(
                f"{context}: target differs from parent {parent_id!r}"
            )
        group = chunks_by_parent.setdefault(parent_id, {})
        if chunk_index in group:
            raise BatchPreparationError(
                f"Duplicate chunk_index {chunk_index} for parent {parent_id!r}"
            )
        group[chunk_index] = {**chunk, "article_text": text}
        declared_counts.setdefault(parent_id, set()).add(chunk_count)
        reconstructed_hash = chunk.get("reconstructed_fulltext_sha256")
        source_hash = chunk.get("source_fulltext_sha256")
        if (reconstructed_hash is None) != (source_hash is None):
            raise BatchPreparationError(
                f"{context}: source and reconstruction hashes must appear together"
            )
        if reconstructed_hash is not None:
            reconstructed_hashes.setdefault(parent_id, set()).add(
                _nonempty_string(
                    chunk, "reconstructed_fulltext_sha256", context
                )
            )
            source_hashes.setdefault(parent_id, set()).add(
                _nonempty_string(chunk, "source_fulltext_sha256", context)
            )

    documents: list[dict[str, Any]] = []
    for parent in sorted(parent_records, key=_parent_sort_key):
        article_id = str(parent["article_id"])
        indexed = chunks_by_parent.get(article_id)
        if not indexed:
            raise BatchPreparationError(
                f"Parent {article_id!r} has no full-text chunks"
            )
        counts = declared_counts[article_id]
        if len(counts) != 1:
            raise BatchPreparationError(
                f"Parent {article_id!r} has inconsistent chunk_count values"
            )
        declared = next(iter(counts))
        expected_indices = list(range(declared))
        if sorted(indexed) != expected_indices:
            raise BatchPreparationError(
                f"Parent {article_id!r} expected chunk indices "
                f"{expected_indices!r}, found {sorted(indexed)!r}"
            )
        full_text = "\n\n".join(
            str(indexed[chunk_index]["article_text"])
            for chunk_index in expected_indices
        )
        if not full_text.strip():
            raise BatchPreparationError(
                f"Parent {article_id!r} reconstructed to empty full text"
            )
        parent_reconstruction_hashes = reconstructed_hashes.get(article_id)
        parent_source_hashes = source_hashes.get(article_id)
        hashes_present = parent_reconstruction_hashes is not None
        if hashes_present:
            if (
                len(parent_reconstruction_hashes) != 1
                or parent_source_hashes is None
                or len(parent_source_hashes) != 1
            ):
                raise BatchPreparationError(
                    f"Parent {article_id!r} has inconsistent text hashes"
                )
            reconstructed_sha256 = next(iter(parent_reconstruction_hashes))
            if sha256_bytes(full_text.encode("utf-8")) != reconstructed_sha256:
                raise BatchPreparationError(
                    f"Parent {article_id!r} reconstructed-text hash mismatch"
                )
            source_sha256 = next(iter(parent_source_hashes))
        else:
            reconstructed_sha256 = None
            source_sha256 = None
        documents.append(
            {
                "document_number": int(parent["row_number"]),
                "article_id": article_id,
                "headline": str(parent["headline"]),
                "full_text": full_text,
                "target": dict(parent["target"]),
                "_manifest_metadata": {
                    "time_published_utc": str(parent["time_published_utc"]),
                    "benchmark_split": str(parent.get("benchmark_split", "")),
                    "chunk_count": declared,
                    "source_fulltext_sha256": source_sha256,
                    "reconstructed_fulltext_sha256": reconstructed_sha256,
                    "reconstruction_hash_verified": hashes_present,
                },
            }
        )
    return documents


def _private_output_check(path: Path) -> None:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError:
        return
    if not relative.parts or relative.parts[0].casefold() != "data":
        raise BatchPreparationError(
            "Licensed GPT batch outputs inside the repository must remain under data/"
        )


def prepare_batches(
    *,
    parent_path: Path = DEFAULT_PARENT_INPUT,
    chunk_path: Path = DEFAULT_CHUNK_INPUT,
    schema_path: Path = DEFAULT_SCHEMA,
    protocol_path: Path = DEFAULT_PROTOCOL,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    cutoff: str = DEFAULT_CUTOFF,
    batch_size: int = DEFAULT_BATCH_SIZE,
    expected_documents: int = DEFAULT_EXPECTED_DOCUMENTS,
    overwrite: bool = False,
) -> dict[str, Any]:
    if batch_size < 1:
        raise BatchPreparationError("batch_size must be positive")
    _private_output_check(output_root)
    for required in (schema_path, protocol_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    parents = read_jsonl(parent_path)
    chunks = read_jsonl(chunk_path)
    documents = reconstruct_parent_documents(
        parents,
        chunks,
        cutoff=cutoff,
        expected_documents=expected_documents,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    existing_batches = sorted(output_root.glob("batch_*.jsonl"))
    manifest_path = output_root / "manifest.json"
    existing = [*existing_batches, *([manifest_path] if manifest_path.exists() else [])]
    if existing and not overwrite:
        raise FileExistsError(
            "Batch output already exists; pass --overwrite to replace known files: "
            + ", ".join(str(path) for path in existing)
        )
    if overwrite:
        for path in existing:
            path.unlink()

    batch_entries: list[dict[str, Any]] = []
    document_entries: list[dict[str, Any]] = []
    for batch_offset in range(0, len(documents), batch_size):
        batch_number = batch_offset // batch_size + 1
        batch_documents = documents[batch_offset : batch_offset + batch_size]
        batch_records = [
            {
                key: value
                for key, value in document.items()
                if key != "_manifest_metadata"
            }
            for document in batch_documents
        ]
        batch_path = output_root / f"batch_{batch_number:03d}.jsonl"
        write_jsonl(batch_path, batch_records)
        batch_entries.append(
            {
                "batch_number": batch_number,
                "file": batch_path.name,
                "document_count": len(batch_records),
                "article_ids": [row["article_id"] for row in batch_records],
                "article_ids_sha256": stable_id_hash(
                    row["article_id"] for row in batch_records
                ),
                "sha256": sha256_file(batch_path),
                "bytes": batch_path.stat().st_size,
                "contains_licensed_full_text": True,
            }
        )
        for position, (row, private_document) in enumerate(
            zip(batch_records, batch_documents), start=1
        ):
            metadata = private_document["_manifest_metadata"]
            document_entries.append(
                {
                    "article_id": row["article_id"],
                    "target_ticker": row["target"]["ticker"],
                    "time_published_utc": metadata["time_published_utc"],
                    "benchmark_split": metadata["benchmark_split"],
                    "batch_file": batch_path.name,
                    "batch_position": position,
                    "chunk_count": metadata["chunk_count"],
                    "full_text_sha256": sha256_bytes(
                        row["full_text"].encode("utf-8")
                    ),
                }
            )

    article_ids = [row["article_id"] for row in documents]
    manifest: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "status": "complete",
        "model_target": "gpt-5.6-sol",
        "protocol_version": "news-fulltext-gpt-silver-v1.0.0",
        "input_text": "complete reconstructed parent full text",
        "cutoff_utc": format_utc(parse_timestamp(cutoff)),
        "cutoff_relation": "strictly_after",
        "batch_size": batch_size,
        "document_count": len(documents),
        "batch_count": len(batch_entries),
        "article_ids_sha256": stable_id_hash(article_ids),
        "article_ids": article_ids,
        "source_inputs": {
            "parents": {
                "path": str(parent_path.resolve()),
                "sha256": sha256_file(parent_path),
                "bytes": parent_path.stat().st_size,
            },
            "fulltext_chunks": {
                "path": str(chunk_path.resolve()),
                "sha256": sha256_file(chunk_path),
                "bytes": chunk_path.stat().st_size,
            },
            "schema": {
                "path": str(schema_path.resolve()),
                "sha256": sha256_file(schema_path),
            },
            "protocol": {
                "path": str(protocol_path.resolve()),
                "sha256": sha256_file(protocol_path),
            },
        },
        "batches": batch_entries,
        "documents": document_entries,
        "privacy": {
            "batch_files_contain_licensed_full_text": True,
            "batch_root_must_remain_git_ignored": True,
            "merged_annotation_must_strip_full_text_and_quoted_evidence": True,
        },
    }
    write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    args = parse_args()
    manifest = prepare_batches(
        parent_path=args.parents,
        chunk_path=args.chunks,
        schema_path=args.schema,
        protocol_path=args.protocol,
        output_root=args.output_root,
        cutoff=args.cutoff,
        batch_size=args.batch_size,
        expected_documents=args.expected_documents,
        overwrite=args.overwrite,
    )
    print(
        f"Wrote {manifest['document_count']} complete parent documents in "
        f"{manifest['batch_count']} batches to {args.output_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
