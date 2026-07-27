"""Run one repository script with packages from a preserved Python 3.11 venv.

This launcher is for recovery only.  It lets an official embeddable Python
3.11 runtime execute a script while importing the already-installed packages
from a model-specific virtual environment whose original base interpreter is
no longer present.  It does not modify that environment, its model cache, or
any active experiment pointer.

Example:

    data/runtime/python311-embed/base/python.exe \
      scripts/run_with_isolated_python311.py \
      --site-packages .venv-llama31/Lib/site-packages \
      --script scripts/extract_llama_3_1_coarse.py \
      -- --input INPUT.jsonl --output OUTPUT.jsonl
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]


def repository_path(value: Path) -> Path:
    path = value if value.is_absolute() else ROOT / value
    return path.resolve()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-packages", required=True, type=Path)
    parser.add_argument("--script", required=True, type=Path)
    parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Arguments for the target script, optionally preceded by --.",
    )
    args = parser.parse_args(argv)
    if args.script_args[:1] == ["--"]:
        args.script_args = args.script_args[1:]
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    site_packages = repository_path(args.site_packages)
    script = repository_path(args.script)
    if not site_packages.is_dir():
        raise FileNotFoundError(f"site-packages directory is missing: {site_packages}")
    if not script.is_file():
        raise FileNotFoundError(f"target script is missing: {script}")
    try:
        script.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError("target script must remain inside the repository") from exc

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    sys.path.insert(0, str(site_packages))
    sys.path.insert(0, str(script.parent))
    sys.path.insert(0, str(ROOT))
    sys.argv = [str(script), *args.script_args]
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
