"""Fetch resumable historical earnings records from Alpha Vantage.

This is a *partial* earnings-event source. Alpha Vantage's EARNINGS endpoint
provides historical ``reportedDate`` values and surprise data, but not a
point-in-time scheduled calendar or a reliable before/after-market timestamp.
The output therefore records ``announcement_time_known=false`` and is not
automatically added to the primary pre-open feature panel.

The free API is rate-limited. By default at most 20 new symbols are requested
per invocation; completed symbol responses are cached and skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from fetch_alpaca_bars import (
    load_dotenv_values,
    load_universe,
    parse_date,
    parse_symbols,
    sha256_file,
    symbols_from_universe,
    write_json_atomic,
)


API_URL = "https://www.alphavantage.co/query"
DEFAULT_CONFIG = Path("config/price_universe.json")
DEFAULT_OUTPUT_DIR = Path("data/external/alpha-vantage-earnings")
DEFAULT_ENV_FILE = Path(".env")
DEFAULT_START = date(2016, 1, 1)
DEFAULT_END = date(2026, 6, 30)
DEFAULT_MAX_NEW_SYMBOLS = 20
CSV_COLUMNS = (
    "symbol",
    "reported_date",
    "fiscal_date_ending",
    "reported_eps",
    "estimated_eps",
    "surprise",
    "surprise_percentage",
    "announcement_time_known",
    "point_in_time_schedule_available",
)


def load_api_key(env_file: Path) -> str:
    values = load_dotenv_values(env_file)
    key = os.environ.get("ALPHA_VANTAGE_KEY") or values.get("ALPHA_VANTAGE_KEY")
    if not key:
        raise RuntimeError(
            "Missing ALPHA_VANTAGE_KEY in the process environment or .env"
        )
    return key


def fetch_symbol(
    symbol: str,
    api_key: str,
    *,
    timeout: float = 60,
    retries: int = 3,
) -> bytes:
    query = urllib.parse.urlencode(
        {"function": "EARNINGS", "symbol": symbol, "apikey": api_key}
    )
    url = f"{API_URL}?{query}"
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "stock-sector-correlation-research/1.0",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504}:
                raise RuntimeError(
                    f"Alpha Vantage returned HTTP {exc.code} for {symbol}"
                ) from exc
            last_error = exc
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
        if attempt < retries:
            time.sleep(min(30, 2**attempt))
    raise RuntimeError(f"Alpha Vantage request failed for {symbol}: {last_error}")


def parse_earnings(
    raw: bytes, *, symbol: str, start: date, end: date
) -> list[dict[str, str]]:
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected Alpha Vantage response for {symbol}")
    provider_message = (
        payload.get("Note")
        or payload.get("Information")
        or payload.get("Error Message")
    )
    if provider_message:
        raise RuntimeError(f"Alpha Vantage response for {symbol}: {provider_message}")
    records = payload.get("quarterlyEarnings")
    if not isinstance(records, list):
        raise RuntimeError(
            f"Alpha Vantage returned no quarterlyEarnings array for {symbol}"
        )
    rows: list[dict[str, str]] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        reported_date = str(record.get("reportedDate", ""))
        try:
            parsed_date = date.fromisoformat(reported_date)
        except ValueError:
            continue
        if not start <= parsed_date <= end:
            continue
        rows.append(
            {
                "symbol": symbol,
                "reported_date": reported_date,
                "fiscal_date_ending": str(record.get("fiscalDateEnding", "")),
                "reported_eps": str(record.get("reportedEPS", "")),
                "estimated_eps": str(record.get("estimatedEPS", "")),
                "surprise": str(record.get("surprise", "")),
                "surprise_percentage": str(record.get("surprisePercentage", "")),
                "announcement_time_known": "false",
                "point_in_time_schedule_available": "false",
            }
        )
    return sorted(rows, key=lambda row: row["reported_date"])


def write_bytes_atomic(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=path.name + ".", delete=False
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(value)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def write_csv_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=path.name + ".",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def symbol_paths(output_dir: Path, symbol: str) -> tuple[Path, Path]:
    return (
        output_dir / "raw" / f"{symbol}.json",
        output_dir / "raw" / f"{symbol}.manifest.json",
    )


def is_complete(raw_path: Path, manifest_path: Path) -> bool:
    if not raw_path.exists() or not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        manifest.get("status") == "complete"
        and manifest.get("raw_sha256") == sha256_file(raw_path)
    )


def cached_rows(
    output_dir: Path, symbols: Sequence[str], *, start: date, end: date
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for symbol in symbols:
        raw_path, manifest_path = symbol_paths(output_dir, symbol)
        if not is_complete(raw_path, manifest_path):
            continue
        rows.extend(
            parse_earnings(
                raw_path.read_bytes(), symbol=symbol, start=start, end=end
            )
        )
    return sorted(rows, key=lambda row: (row["reported_date"], row["symbol"]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--symbols", type=parse_symbols)
    parser.add_argument("--start", type=parse_date, default=DEFAULT_START)
    parser.add_argument("--end", type=parse_date, default=DEFAULT_END)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument(
        "--max-new-symbols", type=int, default=DEFAULT_MAX_NEW_SYMBOLS
    )
    parser.add_argument("--request-pause", type=float, default=12.5)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.start > args.end:
        raise ValueError("start must be on or before end")
    if args.max_new_symbols < 0:
        raise ValueError("max-new-symbols must be nonnegative")
    universe = load_universe(args.config)
    all_symbols = args.symbols or [
        symbol
        for symbol in symbols_from_universe(universe)
        if symbol not in {
            sector["benchmark"] for sector in universe["sectors"]
        }
        and symbol not in set(universe.get("controls", []))
    ]
    completed = []
    pending = []
    for symbol in all_symbols:
        raw_path, manifest_path = symbol_paths(args.output_dir, symbol)
        if not args.overwrite and is_complete(raw_path, manifest_path):
            completed.append(symbol)
        else:
            pending.append(symbol)
    selected = pending[: args.max_new_symbols]
    print(
        json.dumps(
            {
                "mode": "dry-run" if args.dry_run else "download",
                "symbols": all_symbols,
                "completed_symbols": completed,
                "pending_symbols": pending,
                "new_symbols_this_run": selected,
                "max_new_symbols": args.max_new_symbols,
                "output_dir": str(args.output_dir),
                "limitation": (
                    "reported dates only; no point-in-time scheduled calendar "
                    "or announcement timestamp"
                ),
            },
            indent=2,
        )
    )
    if args.dry_run:
        return 0
    api_key = load_api_key(args.env_file)
    for index, symbol in enumerate(selected):
        raw = fetch_symbol(
            symbol, api_key, timeout=args.timeout, retries=args.retries
        )
        # Parse before committing the raw response so rate-limit/error payloads
        # never become completed cache entries.
        rows = parse_earnings(
            raw, symbol=symbol, start=args.start, end=args.end
        )
        raw_path, manifest_path = symbol_paths(args.output_dir, symbol)
        write_bytes_atomic(raw_path, raw)
        manifest = {
            "status": "complete",
            "schema_version": 1,
            "source": "Alpha Vantage EARNINGS endpoint",
            "endpoint": API_URL,
            "symbol": symbol,
            "row_count_in_requested_range": len(rows),
            "raw_path": str(raw_path),
            "raw_sha256": sha256_file(raw_path),
            "fetched_at_utc": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "announcement_time_known": False,
            "point_in_time_schedule_available": False,
            "credential_values_recorded": False,
        }
        write_json_atomic(manifest_path, manifest)
        print(f"[{index + 1}/{len(selected)}] downloaded {symbol}: {len(rows)} rows")
        if index + 1 < len(selected) and args.request_pause:
            time.sleep(args.request_pause)

    combined = cached_rows(
        args.output_dir, all_symbols, start=args.start, end=args.end
    )
    combined_path = args.output_dir / "reported_earnings.csv"
    write_csv_atomic(combined_path, combined)
    summary = {
        "status": "complete",
        "schema_version": 1,
        "requested_symbol_count": len(all_symbols),
        "cached_symbol_count": sum(
            is_complete(*symbol_paths(args.output_dir, symbol))
            for symbol in all_symbols
        ),
        "row_count": len(combined),
        "output": str(combined_path),
        "output_sha256": sha256_file(combined_path),
        "announcement_time_known": False,
        "point_in_time_schedule_available": False,
        "recommended_use": "exploratory/robustness indicator only",
    }
    write_json_atomic(args.output_dir / "manifest.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
