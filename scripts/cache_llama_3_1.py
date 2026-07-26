from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import cache_llama_2 as cache_common


MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
CONSERVATIVE_DATA_CUTOFF = "2023-12-31"
MODEL_CONTEXT_WINDOW = 131072
OPERATIONAL_CONTEXT_LIMIT = 1024
MANIFEST_VERSION = "llama-3.1-local-snapshot-v1"
ALLOW_PATTERNS = (
    "LICENSE",
    "README.md",
    "USE_POLICY.md",
    "config.json",
    "generation_config.json",
    "model*.safetensors",
    "model.safetensors.index.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cache and hash the exact gated Llama 3.1 8B Instruct snapshot. "
            "This script downloads weights but never loads the model."
        )
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/llama_3_1/model_snapshot_manifest.json"),
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.model_id != MODEL_ID:
        parser.error(f"--model-id must remain pinned to {MODEL_ID}")
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("--revision must be a 40-character lowercase commit SHA")
    if args.revision != MODEL_REVISION:
        parser.error(f"--revision must remain pinned to {MODEL_REVISION}")
    return args


def main() -> int:
    args = parse_args()
    if args.validate_only:
        print(
            json.dumps(
                {
                    "model_id": MODEL_ID,
                    "revision": MODEL_REVISION,
                    "conservative_model_data_cutoff": CONSERVATIVE_DATA_CUTOFF,
                    "reported_model_context_window": MODEL_CONTEXT_WINDOW,
                    "operational_context_limit": OPERATIONAL_CONTEXT_LIMIT,
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
            "Install requirements-llama-3-1.txt before caching Llama 3.1."
        ) from exc

    snapshot_path = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            allow_patterns=list(ALLOW_PATTERNS),
            local_files_only=args.local_files_only,
        )
    ).resolve()
    files = cache_common.snapshot_file_manifest(snapshot_path)
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "conservative_model_data_cutoff": CONSERVATIVE_DATA_CUTOFF,
        "reported_model_context_window": MODEL_CONTEXT_WINDOW,
        "operational_context_limit": OPERATIONAL_CONTEXT_LIMIT,
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
    temporary_manifest = args.manifest.with_name(
        f".{args.manifest.name}.{os.getpid()}.tmp"
    )
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary_manifest, args.manifest)
    print(f"Cached {len(files)} files at {snapshot_path}")
    print(f"Wrote snapshot manifest to {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
