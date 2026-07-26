from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_POINTER = ROOT / "experiments" / "active_extractor.json"
EXPECTED_EXTRACTOR_ID = "flan-t5-xl-v1.1"
EXPECTED_MODEL_ID = "google/flan-t5-xl"
EXPECTED_MODEL_REVISION = "7d6315df2c2fb742f0f5b556879d730926ca9001"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_path(value: str, root: Path = ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def verify_artifact(record: dict[str, Any], root: Path = ROOT) -> Path:
    path = repository_path(record["path"], root)
    if not path.is_file():
        raise FileNotFoundError(f"Frozen active artifact is missing: {path}")
    actual = sha256_file(path)
    if actual != record["sha256"]:
        raise ValueError(f"Frozen active artifact hash changed: {path}")
    return path


def load_active_contract(
    pointer_path: Path = ACTIVE_POINTER, root: Path = ROOT
) -> tuple[dict[str, Any], Path, Path]:
    contract = json.loads(pointer_path.read_text(encoding="utf-8"))
    if contract.get("extractor_id") != EXPECTED_EXTRACTOR_ID:
        raise ValueError("The repository active pointer is not FLAN-T5-XL v1.1")
    if contract.get("status") != "active_research_extractor":
        raise ValueError("The active extractor contract is not enabled")
    if contract.get("production_ready") is not False:
        raise ValueError("The active extractor must retain its research-only warning")
    model = contract.get("model", {})
    if model.get("id") != EXPECTED_MODEL_ID:
        raise ValueError("The active extractor points to a different model")
    if model.get("revision") != EXPECTED_MODEL_REVISION:
        raise ValueError("The active extractor points to a different model revision")
    pipeline = contract.get("pipeline", {})
    if pipeline.get("mode") != "development_selected_score_calibration":
        raise ValueError("Unsupported active XL pipeline mode")
    verify_artifact(pipeline["raw_runner"], root)
    verify_artifact(pipeline["postprocessor"], root)
    schema_path = verify_artifact(pipeline["schema"], root)
    calibration_path = verify_artifact(pipeline["calibration"], root)
    return contract, schema_path, calibration_path


def default_raw_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}.raw{output.suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen FLAN-T5-XL v1.1 research extractor, then apply "
            "its development-selected score calibration."
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--raw-output",
        type=Path,
        help="Optional raw XL output path; defaults beside --output with .raw.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate inputs and prompts without loading the model or writing output.",
    )
    mode.add_argument(
        "--apply-only",
        action="store_true",
        help="Skip inference and apply the frozen calibration to an existing raw output.",
    )
    return parser.parse_args()


def raw_command(
    *,
    input_path: Path,
    raw_output: Path,
    schema_path: Path,
    limit: int | None,
    overwrite: bool,
    validate_only: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_flan_t5_xl_coarse.py"),
        "--input",
        str(input_path),
        "--output",
        str(raw_output),
        "--schema",
        str(schema_path),
        "--decoding",
        "order_averaged_letter_score",
        "--prompt-profile",
        "zero_shot",
        "--device",
        "cuda",
        "--precision",
        "float16",
        "--batch-size",
        "1",
    ]
    if limit is not None:
        command.extend(["--limit", str(limit)])
    if overwrite:
        command.append("--overwrite")
    if validate_only:
        command.append("--validate-only")
    return command


def apply_command(
    *,
    raw_output: Path,
    output: Path,
    schema_path: Path,
    calibration_path: Path,
    overwrite: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "experiment_flan_t5_xl_v1_1.py"),
        "apply",
        "--predictions",
        str(raw_output),
        "--schema",
        str(schema_path),
        "--config",
        str(calibration_path),
        "--output",
        str(output),
    ]
    if overwrite:
        command.append("--overwrite")
    return command


def main() -> int:
    args = parse_args()
    _, schema_path, calibration_path = load_active_contract()
    raw_output = args.raw_output or default_raw_path(args.output)
    if raw_output.resolve() == args.output.resolve():
        raise ValueError("--raw-output and --output must be different paths")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")

    environment = os.environ.copy()
    environment["HF_HUB_OFFLINE"] = "1"
    if not args.apply_only:
        subprocess.run(
            raw_command(
                input_path=args.input,
                raw_output=raw_output,
                schema_path=schema_path,
                limit=args.limit,
                overwrite=args.overwrite,
                validate_only=args.validate_only,
            ),
            cwd=ROOT,
            env=environment,
            check=True,
        )
    if args.validate_only:
        return 0
    subprocess.run(
        apply_command(
            raw_output=raw_output,
            output=args.output,
            schema_path=schema_path,
            calibration_path=calibration_path,
            overwrite=args.overwrite,
        ),
        cwd=ROOT,
        env=environment,
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
