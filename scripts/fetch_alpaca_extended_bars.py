"""Fetch only premarket/aftermarket Alpaca bars without touching RTH outputs.

Unlike ``fetch_alpaca_bars.py``, this downloader makes one multi-symbol request
per trading date and requested window. The request bounds are narrow:

* premarket: 04:00 ET through the configurable forecast cutoff (09:00 by default)
* aftermarket: the official Alpaca calendar close through 20:00 ET

That design avoids downloading the regular-session bars a second time. Output
is written under ``data/prices/alpaca-extended`` and never modifies the regular
bar directory.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
import tempfile
from datetime import date, datetime, time as wall_time, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from fetch_alpaca_bars import (
    API_URL,
    DOWNLOAD_SCHEMA_VERSION,
    NEW_YORK,
    PAGE_LIMIT,
    AlpacaClient,
    AlpacaDownloadError,
    MarketSession,
    load_credentials,
    load_universe,
    normalize_bar,
    normalize_market_calendar,
    parse_date,
    parse_market_time,
    parse_symbols,
    sha256_file,
    symbols_from_universe,
    write_json_atomic,
)


DEFAULT_CONFIG = Path("config/price_universe.json")
DEFAULT_CALENDAR_CACHE = Path(
    "data/prices/alpaca/calendar/2016-01-01_2026-06-30.json"
)
DEFAULT_OUTPUT_DIR = Path("data/prices/alpaca-extended")
DEFAULT_START = date(2022, 11, 1)
DEFAULT_END = date(2026, 6, 30)
DEFAULT_ENV_FILE = Path(".env")
EXTENDED_SCHEMA_VERSION = 1
CSV_COLUMNS = (
    "symbol",
    "session",
    "timestamp_utc",
    "timestamp_et",
    "trade_date",
    "bar_start_et",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_count",
    "vwap",
)


def parse_clock(value: str) -> wall_time:
    try:
        return parse_market_time(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_sessions(value: str) -> tuple[str, ...]:
    result = tuple(
        dict.fromkeys(item.strip().lower() for item in value.split(",") if item.strip())
    )
    allowed = {"premarket", "aftermarket"}
    invalid = sorted(set(result) - allowed)
    if not result or invalid:
        raise argparse.ArgumentTypeError(
            "sessions must contain premarket and/or aftermarket"
        )
    return result


def load_calendar_cache(
    path: Path, *, start: date, end: date
) -> dict[date, MarketSession]:
    """Load and validate a previously downloaded Alpaca calendar, read-only."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete" or not isinstance(
        payload.get("sessions"), list
    ):
        raise AlpacaDownloadError(f"Incomplete or invalid calendar cache: {path}")
    sessions = normalize_market_calendar(payload["sessions"])
    if not sessions:
        raise AlpacaDownloadError(f"Calendar cache contains no sessions: {path}")
    cached_start = date.fromisoformat(str(payload.get("start", min(sessions))))
    cached_end = date.fromisoformat(str(payload.get("end", max(sessions))))
    if start < cached_start or end > cached_end:
        raise AlpacaDownloadError(
            f"Calendar cache covers {cached_start} through {cached_end}, "
            f"not requested range {start} through {end}"
        )
    return {day: session for day, session in sessions.items() if start <= day <= end}


def utc_bound(trade_date: date, local_time: wall_time) -> str:
    local = datetime.combine(trade_date, local_time, tzinfo=NEW_YORK)
    return local.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def request_window(
    trade_date: date,
    market_session: MarketSession,
    session_name: str,
    *,
    premarket_start: wall_time,
    premarket_cutoff: wall_time,
    aftermarket_end: wall_time,
) -> tuple[wall_time, wall_time]:
    if session_name == "premarket":
        start_time, end_time = premarket_start, premarket_cutoff
        if not start_time < end_time <= market_session.open_time:
            raise ValueError(
                "Premarket bounds must satisfy start < cutoff <= market open"
            )
        return start_time, end_time
    if session_name == "aftermarket":
        if not market_session.close_time < aftermarket_end:
            raise ValueError("Aftermarket end must be after the official market close")
        return market_session.close_time, aftermarket_end
    raise ValueError(f"Unknown session: {session_name}")


def is_in_window(
    row: Mapping[str, Any],
    *,
    session_name: str,
    start_time: wall_time,
    end_time: wall_time,
) -> bool:
    if row.get("session") != session_name:
        return False
    bar_time = parse_market_time(str(row["bar_start_et"]))
    return start_time <= bar_time < end_time


def fetch_window(
    client: AlpacaClient,
    *,
    symbols: Sequence[str],
    trade_date: date,
    session_name: str,
    start_time: wall_time,
    end_time: wall_time,
    timeframe: str,
    feed: str,
    adjustment: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Fetch one exact daily window for every symbol in a single paged query."""

    end_exclusive = datetime.combine(
        trade_date, end_time, tzinfo=NEW_YORK
    ) - timedelta(microseconds=1)
    params: dict[str, str | int] = {
        "symbols": ",".join(symbols),
        "timeframe": timeframe,
        "start": utc_bound(trade_date, start_time),
        "end": end_exclusive.astimezone(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "limit": PAGE_LIMIT,
        "adjustment": adjustment,
        "feed": feed,
        "sort": "asc",
    }
    rows: list[dict[str, Any]] = []
    request_ids: list[str] = []
    while True:
        payload, request_id = client.get_json(params)
        if request_id:
            request_ids.append(request_id)
        payload_bars = payload.get("bars")
        if not isinstance(payload_bars, dict):
            raise AlpacaDownloadError("Alpaca response field 'bars' is not an object")
        for symbol in symbols:
            symbol_bars = payload_bars.get(symbol, [])
            if not isinstance(symbol_bars, list):
                raise AlpacaDownloadError(f"Bars for {symbol} are not a list")
            for bar in symbol_bars:
                if not isinstance(bar, dict):
                    continue
                row = normalize_bar(symbol, bar)
                row["session"] = session_name
                if (
                    row["trade_date"] == trade_date.isoformat()
                    and is_in_window(
                        row,
                        session_name=session_name,
                        start_time=start_time,
                        end_time=end_time,
                    )
                ):
                    rows.append(row)
        page_token = payload.get("next_page_token")
        if not page_token:
            break
        params["page_token"] = str(page_token)
    rows.sort(key=lambda item: (str(item["timestamp_utc"]), str(item["symbol"])))
    identities = [(row["symbol"], row["timestamp_utc"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise AlpacaDownloadError("Duplicate symbol/timestamp rows in extended window")
    return rows, request_ids


def output_paths(
    output_dir: Path,
    *,
    timeframe: str,
    feed: str,
    adjustment: str,
    session_name: str,
    trade_date: date,
) -> tuple[Path, Path]:
    base = (
        output_dir
        / timeframe.lower()
        / feed
        / adjustment
        / session_name
        / str(trade_date.year)
    )
    stem = trade_date.isoformat()
    return base / f"{stem}.csv.gz", base / f"{stem}.manifest.json"


def write_csv_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=path.name + ".", delete=False
        ) as raw:
            temp_path = Path(raw.name)
            with gzip.open(raw, mode="wt", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def manifest_is_complete(csv_path: Path, manifest_path: Path) -> bool:
    if not csv_path.exists() or not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        manifest.get("status") == "complete"
        and manifest.get("extended_schema_version") == EXTENDED_SCHEMA_VERSION
        and manifest.get("csv_sha256") == sha256_file(csv_path)
    )


def download_window(
    client: AlpacaClient,
    *,
    symbols: Sequence[str],
    trade_date: date,
    market_session: MarketSession,
    session_name: str,
    premarket_start: wall_time,
    premarket_cutoff: wall_time,
    aftermarket_end: wall_time,
    timeframe: str,
    feed: str,
    adjustment: str,
    output_dir: Path,
    calendar_cache: Path,
    overwrite: bool,
) -> dict[str, Any]:
    start_time, end_time = request_window(
        trade_date,
        market_session,
        session_name,
        premarket_start=premarket_start,
        premarket_cutoff=premarket_cutoff,
        aftermarket_end=aftermarket_end,
    )
    csv_path, manifest_path = output_paths(
        output_dir,
        timeframe=timeframe,
        feed=feed,
        adjustment=adjustment,
        session_name=session_name,
        trade_date=trade_date,
    )
    if not overwrite and manifest_is_complete(csv_path, manifest_path):
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {
            "action": "skipped",
            "date": trade_date.isoformat(),
            "session": session_name,
            "row_count": existing.get("row_count", 0),
        }

    rows, request_ids = fetch_window(
        client,
        symbols=symbols,
        trade_date=trade_date,
        session_name=session_name,
        start_time=start_time,
        end_time=end_time,
        timeframe=timeframe,
        feed=feed,
        adjustment=adjustment,
    )
    write_csv_atomic(csv_path, rows)
    symbol_counts: dict[str, int] = {}
    for row in rows:
        symbol = str(row["symbol"])
        symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
    manifest = {
        "status": "complete",
        "extended_schema_version": EXTENDED_SCHEMA_VERSION,
        "regular_download_schema_version_reference": DOWNLOAD_SCHEMA_VERSION,
        "source": "Alpaca Market Data API v2 historical stock bars",
        "endpoint": API_URL,
        "trade_date": trade_date.isoformat(),
        "session": session_name,
        "request": {
            "symbols": list(symbols),
            "timeframe": timeframe,
            "feed": feed,
            "adjustment": adjustment,
            "window_et": {
                "start_inclusive": start_time.isoformat(timespec="minutes"),
                "end_exclusive": end_time.isoformat(timespec="minutes"),
            },
        },
        "official_market_session_et": {
            "open": market_session.open_time.isoformat(timespec="minutes"),
            "close": market_session.close_time.isoformat(timespec="minutes"),
        },
        "calendar_cache": {
            "path": str(calendar_cache),
            "sha256": sha256_file(calendar_cache),
            "access": "read-only",
        },
        "request_ids": request_ids,
        "row_count": len(rows),
        "symbol_counts": symbol_counts,
        "first_timestamp_utc": rows[0]["timestamp_utc"] if rows else None,
        "last_timestamp_utc": rows[-1]["timestamp_utc"] if rows else None,
        "csv_path": str(csv_path),
        "csv_sha256": sha256_file(csv_path),
        "fetched_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "credential_values_recorded": False,
    }
    write_json_atomic(manifest_path, manifest)
    return {
        "action": "downloaded",
        "date": trade_date.isoformat(),
        "session": session_name,
        "row_count": len(rows),
        "csv_path": str(csv_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--symbols", type=parse_symbols)
    parser.add_argument("--start", type=parse_date, default=DEFAULT_START)
    parser.add_argument("--end", type=parse_date, default=DEFAULT_END)
    parser.add_argument(
        "--sessions", type=parse_sessions, default=("premarket", "aftermarket")
    )
    parser.add_argument("--premarket-start", type=parse_clock, default=wall_time(4, 0))
    parser.add_argument(
        "--premarket-cutoff", type=parse_clock, default=wall_time(9, 0)
    )
    parser.add_argument("--aftermarket-end", type=parse_clock, default=wall_time(20, 0))
    parser.add_argument("--calendar-cache", type=Path, default=DEFAULT_CALENDAR_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--timeframe")
    parser.add_argument("--feed", choices=("sip", "iex"))
    parser.add_argument(
        "--adjustment",
        choices=("raw", "split", "dividend", "spin-off", "all"),
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--request-pause", type=float, default=0.35)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.start > args.end:
        raise ValueError("start must be on or before end")
    universe = load_universe(args.config)
    symbols = args.symbols or symbols_from_universe(universe)
    timeframe = args.timeframe or str(universe.get("timeframe", "15Min"))
    feed = args.feed or str(universe.get("feed", "sip"))
    adjustment = args.adjustment or str(universe.get("adjustment", "all"))
    sessions = load_calendar_cache(
        args.calendar_cache, start=args.start, end=args.end
    )
    plan = [
        (trade_date, session_name)
        for trade_date in sorted(sessions)
        for session_name in args.sessions
    ]
    print(
        json.dumps(
            {
                "mode": "dry-run" if args.dry_run else "download",
                "symbols": symbols,
                "symbol_count": len(symbols),
                "start": args.start.isoformat(),
                "end": args.end.isoformat(),
                "market_dates": len(sessions),
                "sessions": list(args.sessions),
                "request_windows": len(plan),
                "premarket_window_et": [
                    args.premarket_start.isoformat(timespec="minutes"),
                    args.premarket_cutoff.isoformat(timespec="minutes"),
                ],
                "aftermarket_end_et": args.aftermarket_end.isoformat(
                    timespec="minutes"
                ),
                "regular_session_bars_requested": False,
                "calendar_cache_access": "read-only",
                "calendar_cache": str(args.calendar_cache),
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )
    if args.dry_run:
        return 0

    client = AlpacaClient(
        load_credentials(args.env_file),
        timeout=args.timeout,
        retries=args.retries,
        request_pause=args.request_pause,
    )
    downloaded = skipped = rows = 0
    for index, (trade_date, session_name) in enumerate(plan, start=1):
        result = download_window(
            client,
            symbols=symbols,
            trade_date=trade_date,
            market_session=sessions[trade_date],
            session_name=session_name,
            premarket_start=args.premarket_start,
            premarket_cutoff=args.premarket_cutoff,
            aftermarket_end=args.aftermarket_end,
            timeframe=timeframe,
            feed=feed,
            adjustment=adjustment,
            output_dir=args.output_dir,
            calendar_cache=args.calendar_cache,
            overwrite=args.overwrite,
        )
        rows += int(result["row_count"])
        if result["action"] == "downloaded":
            downloaded += 1
        else:
            skipped += 1
        print(
            f"[{index}/{len(plan)}] {result['action']} "
            f"{trade_date} {session_name}: {result['row_count']} rows"
        )
    print(
        json.dumps(
            {
                "status": "complete",
                "downloaded_windows": downloaded,
                "skipped_windows": skipped,
                "rows": rows,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AlpacaDownloadError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
