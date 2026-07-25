from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODEL_ID = "google/flan-t5-xl"
MODEL_REVISION = "7d6315df2c2fb742f0f5b556879d730926ca9001"
CONSERVATIVE_DATA_START = "2022-11-01"
ALLOW_PATTERNS = (
    "README.md",
    "config.json",
    "generation_config.json",
    "model*.safetensors",
    "model.safetensors.index.json",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer_config.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cache the exact public FLAN-T5-XL safetensors snapshot and write "
            "a content-hash manifest. This script does not run inference."
        )
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/flan_t5_xl/model_snapshot_manifest.json"),
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Verify a cached snapshot without contacting Hugging Face.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the pinned configuration without downloading or importing Hub libraries.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.model_id != MODEL_ID:
        parser.error(f"--model-id must remain pinned to {MODEL_ID}")
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("--revision must be a 40-character lowercase commit SHA")
    if args.revision != MODEL_REVISION:
        parser.error(f"--revision must remain pinned to {MODEL_REVISION}")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


SAFETENSORS_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "F64": 8,
    "I64": 8,
    "U64": 8,
}


def tensor_nbytes(metadata: dict[str, Any]) -> int:
    dtype = metadata.get("dtype")
    shape = metadata.get("shape")
    if dtype not in SAFETENSORS_DTYPE_BYTES:
        raise RuntimeError(f"Unsupported safetensors dtype: {dtype!r}")
    if not isinstance(shape, list) or not all(
        isinstance(dimension, int) and dimension >= 0 for dimension in shape
    ):
        raise RuntimeError(f"Invalid safetensors shape: {shape!r}")
    elements = 1
    for dimension in shape:
        elements *= dimension
    return elements * SAFETENSORS_DTYPE_BYTES[dtype]


def safetensors_layout(path: Path) -> tuple[dict[str, dict[str, Any]], int]:
    with path.open("rb") as handle:
        header_size_bytes = handle.read(8)
        if len(header_size_bytes) != 8:
            raise RuntimeError(f"Safetensors shard has no complete header: {path.name}")
        header_size = int.from_bytes(
            header_size_bytes, byteorder="little", signed=False
        )
        header_bytes = handle.read(header_size)
    if header_size <= 0 or len(header_bytes) != header_size:
        raise RuntimeError(f"Safetensors shard has a truncated header: {path.name}")
    try:
        raw_header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Safetensors shard has an invalid JSON header: {path.name}") from exc
    if not isinstance(raw_header, dict):
        raise RuntimeError(f"Safetensors shard header is not an object: {path.name}")
    tensors = {
        name: metadata
        for name, metadata in raw_header.items()
        if name != "__metadata__"
    }
    if not tensors:
        raise RuntimeError(f"Safetensors shard contains no tensors: {path.name}")

    payload_size = path.stat().st_size - 8 - header_size
    if payload_size < 0:
        raise RuntimeError(f"Safetensors shard has an invalid header: {path.name}")
    ranges: list[tuple[int, int, str]] = []
    tensor_bytes = 0
    for name, metadata in tensors.items():
        if not isinstance(metadata, dict):
            raise RuntimeError(f"Invalid metadata for tensor {name!r} in {path.name}")
        offsets = metadata.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(offset, int) for offset in offsets)
        ):
            raise RuntimeError(f"Invalid offsets for tensor {name!r} in {path.name}")
        start, end = offsets
        if start < 0 or end < start or end > payload_size:
            raise RuntimeError(
                f"Out-of-range offsets for tensor {name!r} in {path.name}"
            )
        expected_bytes = tensor_nbytes(metadata)
        if end - start != expected_bytes:
            raise RuntimeError(
                f"Tensor byte count differs from shape/dtype for {name!r} in {path.name}"
            )
        ranges.append((start, end, name))
        tensor_bytes += expected_bytes

    ranges.sort()
    cursor = 0
    for start, end, name in ranges:
        if start != cursor:
            raise RuntimeError(
                f"Safetensors payload has a gap or overlap before {name!r} in {path.name}"
            )
        cursor = end
    if cursor != payload_size:
        raise RuntimeError(
            f"Safetensors payload is truncated or has trailing bytes: {path.name}"
        )
    return tensors, tensor_bytes


def snapshot_file_manifest(snapshot_path: Path) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(snapshot_path.rglob("*")):
        if path.is_file():
            relative = path.relative_to(snapshot_path).as_posix()
            files[relative] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }

    index_path = snapshot_path / "model.safetensors.index.json"
    if not index_path.is_file():
        raise RuntimeError("Cached snapshot is missing model.safetensors.index.json")
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Cached safetensors index is unreadable") from exc
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError("Cached safetensors index has no weight_map")
    expected_shards = set(weight_map.values())
    if not all(
        isinstance(name, str) and name.endswith(".safetensors")
        for name in expected_shards
    ):
        raise RuntimeError("Cached safetensors index contains invalid shard names")
    missing_shards = expected_shards - set(files)
    if missing_shards:
        raise RuntimeError(
            "Cached snapshot is missing indexed safetensors shards: "
            f"{sorted(missing_shards)}"
        )

    declared_total = index.get("metadata", {}).get("total_size")
    if not isinstance(declared_total, int) or declared_total <= 0:
        raise RuntimeError("Cached safetensors index has no valid metadata.total_size")

    stored_tensors: dict[str, tuple[str, dict[str, Any]]] = {}
    actual_tensor_total = 0
    for shard_name in expected_shards:
        tensors, shard_tensor_bytes = safetensors_layout(snapshot_path / shard_name)
        actual_tensor_total += shard_tensor_bytes
        for tensor_name, metadata in tensors.items():
            if tensor_name in stored_tensors:
                raise RuntimeError(
                    f"Tensor {tensor_name!r} is duplicated across safetensors shards"
                )
            stored_tensors[tensor_name] = (shard_name, metadata)

    indexed_names = set(weight_map)
    unexpected_stored = set(stored_tensors) - indexed_names
    if unexpected_stored:
        raise RuntimeError(
            "Safetensors shards contain tensors absent from the index: "
            f"{sorted(unexpected_stored)}"
        )
    for tensor_name, (shard_name, _metadata) in stored_tensors.items():
        if weight_map[tensor_name] != shard_name:
            raise RuntimeError(
                f"Index assigns {tensor_name!r} to a different shard"
            )

    # This pinned FLAN-T5-XL checkpoint represents the encoder and decoder
    # embedding parameters as aliases of shared.weight. Safetensors deliberately
    # stores the underlying bytes only once, while metadata.total_size counts the
    # two logical aliases. Validate that exact, documented discrepancy rather
    # than mistaking it for a truncated download.
    expected_omitted_aliases = {
        "encoder.embed_tokens.weight",
        "decoder.embed_tokens.weight",
    }
    omitted_indexed_names = indexed_names - set(stored_tensors)
    if omitted_indexed_names != expected_omitted_aliases:
        raise RuntimeError(
            "Unexpected indexed tensors are absent from safetensors storage: "
            f"{sorted(omitted_indexed_names)}"
        )
    if "shared.weight" not in stored_tensors:
        raise RuntimeError("Safetensors shards do not contain shared.weight")
    shared_shard, shared_metadata = stored_tensors["shared.weight"]
    for alias in expected_omitted_aliases:
        if weight_map[alias] != shared_shard:
            raise RuntimeError(f"Embedding alias {alias!r} is not mapped with shared.weight")
    omitted_alias_bytes = len(expected_omitted_aliases) * tensor_nbytes(
        shared_metadata
    )
    logical_total = actual_tensor_total + omitted_alias_bytes
    if logical_total != declared_total:
        raise RuntimeError(
            "Cached safetensors logical bytes do not match the index after "
            "accounting for the two shared embedding aliases: "
            f"expected {declared_total}, found {logical_total}"
        )

    required = {
        "config.json",
        "tokenizer_config.json",
        "spiece.model",
    }
    missing = required - set(files)
    if missing:
        raise RuntimeError(f"Cached snapshot is missing required files: {sorted(missing)}")
    return files


def main() -> int:
    args = parse_args()
    if args.validate_only:
        print(
            json.dumps(
                {
                    "model_id": MODEL_ID,
                    "model_revision": MODEL_REVISION,
                    "conservative_backtest_start": CONSERVATIVE_DATA_START,
                    "allow_patterns": list(ALLOW_PATTERNS),
                    "model_was_downloaded": False,
                    "model_was_loaded": False,
                },
                indent=2,
            )
        )
        return 0

    if args.manifest.exists() and not args.overwrite:
        raise FileExistsError(f"{args.manifest} exists; pass --overwrite")
    try:
        import huggingface_hub
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Install requirements-flan-t5-xl.txt before caching the model."
        ) from exc

    snapshot_path = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            allow_patterns=list(ALLOW_PATTERNS),
            local_files_only=args.local_files_only,
        )
    ).resolve()
    files = snapshot_file_manifest(snapshot_path)
    manifest = {
        "manifest_version": "flan-t5-xl-local-snapshot-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "conservative_backtest_start": CONSERVATIVE_DATA_START,
        "snapshot_path": str(snapshot_path),
        "local_files_only": args.local_files_only,
        "file_count": len(files),
        "total_size_bytes": sum(item["size_bytes"] for item in files.values()),
        "files": files,
        "runtime": {
            "python_version": sys.version,
            "huggingface_hub_version": huggingface_hub.__version__,
        },
        "model_was_loaded": False,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Cached and verified {len(files)} files at {snapshot_path}")
    print(f"Wrote snapshot manifest to {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
