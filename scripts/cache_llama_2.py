from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
MODEL_REVISION = "f5db02db724555f92da89c216ac04704f23d4590"
CONSERVATIVE_DATA_CUTOFF = "2023-07-31"
MODEL_CONTEXT_WINDOW = 4096
ALLOW_PATTERNS = (
    "LICENSE.txt",
    "README.md",
    "USE_POLICY.md",
    "config.json",
    "generation_config.json",
    "model*.safetensors",
    "model.safetensors.index.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cache the exact gated Llama 2 7B Chat snapshot and write a "
            "content-hash manifest. This script does not run inference."
        )
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/llama_2/model_snapshot_manifest.json"),
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Verify an already cached snapshot without contacting Hugging Face.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the pinned configuration without importing Hub libraries.",
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


def safetensors_payload_size(path: Path) -> int:
    with path.open("rb") as handle:
        header_size_bytes = handle.read(8)
    if len(header_size_bytes) != 8:
        raise RuntimeError(f"Safetensors shard has no complete header: {path.name}")
    header_size = int.from_bytes(header_size_bytes, byteorder="little", signed=False)
    payload_size = path.stat().st_size - 8 - header_size
    if header_size <= 0 or payload_size < 0:
        raise RuntimeError(f"Safetensors shard has an invalid header: {path.name}")
    return payload_size


def snapshot_file_manifest(snapshot_path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(snapshot_path.rglob("*")):
        if path.is_file():
            relative = path.relative_to(snapshot_path).as_posix()
            result[relative] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    if not any(name.endswith(".safetensors") for name in result):
        raise RuntimeError("Cached snapshot contains no safetensors model weights")

    index_path = snapshot_path / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("Cached safetensors index is unreadable") from exc
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise RuntimeError("Cached safetensors index has no weight_map")
        expected_shards = set(weight_map.values())
        if not expected_shards or not all(
            isinstance(name, str) and name.endswith(".safetensors")
            for name in expected_shards
        ):
            raise RuntimeError("Cached safetensors index contains invalid shard names")
        missing_shards = expected_shards - set(result)
        if missing_shards:
            raise RuntimeError(
                "Cached snapshot is missing indexed safetensors shards: "
                f"{sorted(missing_shards)}"
            )

        declared_total = index.get("metadata", {}).get("total_size")
        if not isinstance(declared_total, int) or declared_total <= 0:
            raise RuntimeError(
                "Cached safetensors index has no valid metadata.total_size"
            )
        actual_payload_total = sum(
            safetensors_payload_size(snapshot_path / name)
            for name in expected_shards
        )
        if actual_payload_total != declared_total:
            raise RuntimeError(
                "Cached safetensors payload bytes do not match the index: "
                f"expected {declared_total}, found {actual_payload_total}"
            )

    required = {"config.json", "tokenizer_config.json"}
    missing = required - set(result)
    if missing:
        raise RuntimeError(f"Cached snapshot is missing required files: {sorted(missing)}")
    if not {"tokenizer.json", "tokenizer.model"} & set(result):
        raise RuntimeError("Cached snapshot contains no tokenizer vocabulary")
    return result


def main() -> int:
    args = parse_args()
    if args.validate_only:
        print(
            json.dumps(
                {
                    "model_id": args.model_id,
                    "revision": args.revision,
                    "conservative_model_data_cutoff": CONSERVATIVE_DATA_CUTOFF,
                    "model_context_window": MODEL_CONTEXT_WINDOW,
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
            "Install requirements-llama.txt before caching the model."
        ) from exc

    snapshot_path = Path(
        snapshot_download(
            repo_id=args.model_id,
            revision=args.revision,
            allow_patterns=list(ALLOW_PATTERNS),
            local_files_only=args.local_files_only,
        )
    ).resolve()
    files = snapshot_file_manifest(snapshot_path)
    manifest = {
        "manifest_version": "llama-2-local-snapshot-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model_id,
        "model_revision": args.revision,
        "conservative_model_data_cutoff": CONSERVATIVE_DATA_CUTOFF,
        "model_context_window": MODEL_CONTEXT_WINDOW,
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
    print(f"Cached {len(files)} files at {snapshot_path}")
    print(f"Wrote snapshot manifest to {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
