"""Tokenizer-only context preflight for the isolated text-ablation benchmark.

The frozen local runners refuse truncation, but their lightweight
``--validate-only`` modes intentionally do not import tokenizers.  This helper
loads only the pinned tokenizer bytes, builds the exact frozen prompts, and
reports any record that would exceed FLAN-T5-XL's 512-token or Llama 3.1's
1,024-token operational limit.  It never loads model weights or runs inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA = REPOSITORY_ROOT / "config" / "news_feature_schema_coarse.json"
DEFAULT_FLAN_MANIFEST = (
    REPOSITORY_ROOT / "outputs" / "flan_t5_xl" / "model_snapshot_manifest.json"
)
DEFAULT_LLAMA_MANIFEST = (
    REPOSITORY_ROOT / "outputs" / "llama_3_1" / "model_snapshot_manifest.json"
)
RUNNERS = {
    "flan-t5-xl": {
        "model_id": "google/flan-t5-xl",
        "model_revision": "7d6315df2c2fb742f0f5b556879d730926ca9001",
        "manifest_version": "flan-t5-xl-local-snapshot-v1",
        "default_manifest": DEFAULT_FLAN_MANIFEST,
        "max_input_tokens": 512,
        "required_tokenizer_files": (
            "spiece.model",
            "tokenizer_config.json",
        ),
    },
    "llama-3.1": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "model_revision": "0e9e39f249a16976918f6564b8830bc894c89659",
        "manifest_version": "llama-3.1-local-snapshot-v1",
        "default_manifest": DEFAULT_LLAMA_MANIFEST,
        "max_input_tokens": 1024,
        "required_tokenizer_files": (
            "tokenizer.json",
            "tokenizer_config.json",
        ),
    },
}
REPORT_VERSION = "news-text-ablation-tokenizer-preflight-v1"


class ContextPreflightError(ValueError):
    """Raised when a tokenizer or benchmark contract is invalid."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build exact frozen local-model prompts and check their token counts "
            "without loading model weights."
        )
    )
    parser.add_argument(
        "--runner", required=True, choices=tuple(RUNNERS)
    )
    parser.add_argument(
        "--input", required=True, action="append", type=Path
    )
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument(
        "--snapshot-manifest",
        type=Path,
        help="Defaults to the selected runner's frozen local snapshot manifest.",
    )
    parser.add_argument("--max-input-tokens", type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--allow-violations",
        action="store_true",
        help="Write the report and return success even when prompts are too long.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContextPreflightError(
                    f"{path}:{line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ContextPreflightError(
                    f"{path}:{line_number}: record must be an object"
                )
            records.append(value)
    if not records:
        raise ContextPreflightError(f"{path} contains no records")
    return records


def load_snapshot_contract(
    path: Path, runner_name: str
) -> tuple[dict[str, Any], Path]:
    specification = RUNNERS[runner_name]
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContextPreflightError(
            f"Invalid snapshot manifest JSON: {exc}"
        ) from exc
    for field, expected in (
        ("manifest_version", specification["manifest_version"]),
        ("model_id", specification["model_id"]),
        ("model_revision", specification["model_revision"]),
    ):
        if manifest.get(field) != expected:
            raise ContextPreflightError(
                f"Snapshot {field} differs from the frozen {runner_name} contract"
            )
    snapshot_raw = manifest.get("snapshot_path")
    if not isinstance(snapshot_raw, str) or not snapshot_raw:
        raise ContextPreflightError("Snapshot manifest has no snapshot_path")
    snapshot = Path(snapshot_raw).resolve()
    if not snapshot.is_dir():
        raise FileNotFoundError(snapshot)
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ContextPreflightError("Snapshot manifest has no file inventory")
    for filename in specification["required_tokenizer_files"]:
        record = files.get(filename)
        file_path = snapshot / filename
        if not isinstance(record, Mapping) or not file_path.is_file():
            raise ContextPreflightError(
                f"Frozen tokenizer file is missing: {filename}"
            )
        if file_path.stat().st_size != record.get("size_bytes"):
            raise ContextPreflightError(
                f"Frozen tokenizer file size changed: {filename}"
            )
        if sha256_file(file_path) != record.get("sha256"):
            raise ContextPreflightError(
                f"Frozen tokenizer file hash changed: {filename}"
            )
    return manifest, snapshot


def _record_identity(record: Mapping[str, Any], context: str) -> str:
    article_id = record.get("article_id")
    if not isinstance(article_id, str) or not article_id:
        raise ContextPreflightError(f"{context}: article_id is required")
    return article_id


def flan_prompt_counts(
    records: Sequence[Mapping[str, Any]],
    schema: Mapping[str, Any],
    tokenizer: Any,
) -> list[dict[str, Any]]:
    import coarse_news_features as coarse
    import extract_flan_t5 as base
    import extract_flan_t5_coarse as runner

    runner.validate_schema(dict(schema))
    results: list[dict[str, Any]] = []
    for row_number, record in enumerate(records, start=1):
        context = f"record {row_number}"
        base.validate_input_record(dict(record))
        coarse.validate_input_record(record)
        article_id = _record_identity(record, context)
        for field in coarse.COARSE_FIELDS:
            prompts = runner.prompt_variants_for_preflight(
                dict(record), dict(schema), field, "zero_shot"
            )
            for prompt_index, prompt in enumerate(prompts):
                tokens = tokenizer(
                    prompt, add_special_tokens=True, truncation=False
                )["input_ids"]
                results.append(
                    {
                        "article_id": article_id,
                        "field": field,
                        "prompt_index": prompt_index,
                        "tokens": len(tokens),
                    }
                )
    return results


def llama_prompt_counts(
    records: Sequence[Mapping[str, Any]],
    schema: Mapping[str, Any],
    tokenizer: Any,
) -> list[dict[str, Any]]:
    import coarse_news_features as coarse
    import extract_flan_t5 as base
    import extract_llama_3_1_coarse as runner

    results: list[dict[str, Any]] = []
    with runner.configured_shared_runtime():
        runner.shared_runtime.validate_schema(schema)
        for row_number, record in enumerate(records, start=1):
            context = f"record {row_number}"
            base.validate_input_record(dict(record))
            coarse.validate_input_record(record)
            runner.shared_runtime.validate_post_cutoff_record(record)
            article_id = _record_identity(record, context)
            for field in coarse.COARSE_FIELDS:
                scopes: tuple[str | None, ...] = (
                    ("common", "idiosyncratic")
                    if field == "directional_alignment"
                    else (None,)
                )
                prompt_index = 0
                for scope in scopes:
                    labels = runner.shared_runtime.allowed_labels_for(
                        schema, field, scope
                    )
                    for rotation in runner.shared_experiment.cyclic_rotations(labels):
                        prompt, mapping = (
                            runner.shared_experiment.build_cued_letter_prompt(
                                record, schema, field, rotation, scope
                            )
                        )
                        choice = runner.build_sequence_choice(
                            tokenizer=tokenizer,
                            prompt=prompt,
                            semantic_to_letter=mapping,
                        )
                        results.append(
                            {
                                "article_id": article_id,
                                "field": field,
                                "prompt_index": prompt_index,
                                "tokens": len(choice["input_ids"]),
                            }
                        )
                        prompt_index += 1
    return results


def summarize_counts(
    counts: Sequence[Mapping[str, Any]], max_input_tokens: int
) -> dict[str, Any]:
    maxima: dict[str, int] = defaultdict(int)
    violations: list[dict[str, Any]] = []
    for row in counts:
        field = str(row["field"])
        tokens = int(row["tokens"])
        maxima[field] = max(maxima[field], tokens)
        if tokens > max_input_tokens:
            violations.append(dict(row))
    violations.sort(
        key=lambda row: (
            -int(row["tokens"]),
            str(row["article_id"]),
            str(row["field"]),
            int(row["prompt_index"]),
        )
    )
    return {
        "prompt_count": len(counts),
        "maximum_prompt_tokens_by_field": dict(sorted(maxima.items())),
        "violation_count": len(violations),
        "violating_article_count": len(
            {str(row["article_id"]) for row in violations}
        ),
        "violations": violations,
        "passes_context_limit": not violations,
    }


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def resolve_max_input_tokens(
    requested: int | None, default: int
) -> int:
    return default if requested is None else requested


def run_preflight(
    *,
    runner_name: str,
    input_paths: Sequence[Path],
    schema_path: Path,
    snapshot_manifest_path: Path,
    max_input_tokens: int,
    tokenizer: Any,
) -> dict[str, Any]:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    manifest, snapshot = load_snapshot_contract(
        snapshot_manifest_path, runner_name
    )
    files: list[dict[str, Any]] = []
    all_violations = 0
    for input_path in input_paths:
        records = read_jsonl(input_path)
        ids = [_record_identity(record, str(input_path)) for record in records]
        if len(ids) != len(set(ids)):
            raise ContextPreflightError(
                f"{input_path} contains duplicate article_id values"
            )
        if runner_name == "flan-t5-xl":
            counts = flan_prompt_counts(records, schema, tokenizer)
        else:
            counts = llama_prompt_counts(records, schema, tokenizer)
        summary = summarize_counts(counts, max_input_tokens)
        all_violations += int(summary["violation_count"])
        files.append(
            {
                "path": str(input_path.resolve()),
                "sha256": sha256_file(input_path),
                "record_count": len(records),
                **summary,
            }
        )
    return {
        "report_version": REPORT_VERSION,
        "status": "complete",
        "runner": runner_name,
        "model_id": manifest["model_id"],
        "model_revision": manifest["model_revision"],
        "snapshot_manifest_path": str(snapshot_manifest_path.resolve()),
        "snapshot_manifest_sha256": sha256_file(snapshot_manifest_path),
        "tokenizer_snapshot_path": str(snapshot),
        "max_input_tokens": max_input_tokens,
        "input_files": files,
        "total_violation_count": all_violations,
        "passes_context_limit": all_violations == 0,
        "transformers_tokenizer_runtime_imported": True,
        "model_weights_loaded": False,
        "inference_performed": False,
    }


def main() -> int:
    args = parse_args()
    specification = RUNNERS[args.runner]
    snapshot_manifest = (
        args.snapshot_manifest or specification["default_manifest"]
    )
    max_input_tokens = resolve_max_input_tokens(
        args.max_input_tokens, specification["max_input_tokens"]
    )
    if max_input_tokens < 1:
        raise ContextPreflightError("max_input_tokens must be positive")
    _, snapshot = load_snapshot_contract(snapshot_manifest, args.runner)
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Run this preflight inside the selected model's documented environment"
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, use_fast=True
    )
    report = run_preflight(
        runner_name=args.runner,
        input_paths=args.input,
        schema_path=args.schema,
        snapshot_manifest_path=snapshot_manifest,
        max_input_tokens=max_input_tokens,
        tokenizer=tokenizer,
    )
    write_json(args.output, report)
    print(
        f"Tokenizer-only {args.runner} preflight checked "
        f"{sum(item['record_count'] for item in report['input_files'])} records; "
        f"violations={report['total_violation_count']}"
    )
    return 0 if report["passes_context_limit"] or args.allow_violations else 2


if __name__ == "__main__":
    raise SystemExit(main())
