"""Download resumable, research-ready historical stock bars from Alpaca.

The downloader intentionally uses the REST API through Python's standard
library. It therefore does not require alpaca-py, requests, pandas, or a
working project virtual environment.

Outputs are one gzip-compressed CSV plus one manifest per symbol-year chunk:

    data/prices/alpaca/15min/sip/all/AMD/2024.csv.gz
    data/prices/alpaca/15min/sip/all/AMD/2024.manifest.json

Credentials are read from process environment variables or a local .env file:

    ALPACA_PUBLIC_KEY
    ALPACA_SECRET_KEY

Credential values are sent only in HTTPS headers and are never printed or
written into output manifests.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import random
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, time as wall_time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence
from zoneinfo import ZoneInfo


API_URL = "https://data.alpaca.markets/v2/stocks/bars"
PAPER_CALENDAR_URL = "https://paper-api.alpaca.markets/v2/calendar"
LIVE_CALENDAR_URL = "https://api.alpaca.markets/v2/calendar"
DEFAULT_CONFIG = Path("config/price_universe.json")
DEFAULT_OUTPUT_DIR = Path("data/prices/alpaca")
DEFAULT_START = date(2016, 1, 1)
DEFAULT_END = date(2026, 6, 30)
DEFAULT_ENV_FILE = Path(".env")
PAGE_LIMIT = 10_000
NEW_YORK = ZoneInfo("America/New_York")
UTC = timezone.utc
CSV_COLUMNS = (
    "symbol",
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
DOWNLOAD_SCHEMA_VERSION = 2


class AlpacaDownloadError(RuntimeError):
    """Raised when a historical-data request cannot be completed."""


@dataclass(frozen=True)
class DateChunk:
    start: date
    end: date

    @property
    def label(self) -> str:
        if (
            self.start == date(self.start.year, 1, 1)
            and self.end == date(self.start.year, 12, 31)
        ):
            return str(self.start.year)
        return f"{self.start.isoformat()}_{self.end.isoformat()}"


@dataclass(frozen=True)
class Credentials:
    public_key: str
    secret_key: str


@dataclass(frozen=True)
class MarketSession:
    open_time: wall_time
    close_time: wall_time


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected an ISO date such as 2024-01-02, received {value!r}"
        ) from exc


def parse_symbols(value: str) -> list[str]:
    symbols = [item.strip().upper() for item in value.split(",") if item.strip()]
    if not symbols:
        raise argparse.ArgumentTypeError("At least one symbol is required")
    invalid = [symbol for symbol in symbols if not symbol.replace(".", "").isalpha()]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Unsupported symbol format: {', '.join(invalid)}"
        )
    return list(dict.fromkeys(symbols))


def load_dotenv_values(path: Path) -> dict[str, str]:
    """Read a small .env subset without changing process environment."""

    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"{path}:{line_number}: expected KEY=VALUE")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"{path}:{line_number}: empty environment key")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def load_credentials(env_file: Path) -> Credentials:
    file_values = load_dotenv_values(env_file)
    public_key = os.environ.get("ALPACA_PUBLIC_KEY") or file_values.get(
        "ALPACA_PUBLIC_KEY"
    )
    secret_key = os.environ.get("ALPACA_SECRET_KEY") or file_values.get(
        "ALPACA_SECRET_KEY"
    )
    missing = [
        name
        for name, value in (
            ("ALPACA_PUBLIC_KEY", public_key),
            ("ALPACA_SECRET_KEY", secret_key),
        )
        if not value
    ]
    if missing:
        raise AlpacaDownloadError(
            "Missing Alpaca credential variable(s): "
            + ", ".join(missing)
            + f". Add them to the process environment or {env_file}."
        )
    return Credentials(public_key=public_key, secret_key=secret_key)


def load_universe(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    sectors = data.get("sectors")
    if not isinstance(sectors, list) or not sectors:
        raise ValueError(f"{path}: sectors must be a non-empty list")

    seen_stocks: set[str] = set()
    seen_benchmarks: set[str] = set()
    for sector in sectors:
        name = sector.get("name")
        benchmark = sector.get("benchmark")
        stocks = sector.get("stocks")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{path}: every sector requires a name")
        if not isinstance(benchmark, str) or not benchmark.strip():
            raise ValueError(f"{path}: sector {name!r} requires a benchmark")
        if not isinstance(stocks, list) or len(stocks) < 2:
            raise ValueError(f"{path}: sector {name!r} requires at least two stocks")
        overlap = seen_stocks.intersection(stocks)
        if overlap:
            raise ValueError(
                f"{path}: stocks assigned to multiple sectors: {sorted(overlap)}"
            )
        seen_stocks.update(stocks)
        seen_benchmarks.add(benchmark)
    return data


def symbols_from_universe(universe: Mapping[str, Any]) -> list[str]:
    symbols: list[str] = []
    for sector in universe["sectors"]:
        symbols.extend(str(symbol).upper() for symbol in sector["stocks"])
        symbols.append(str(sector["benchmark"]).upper())
    symbols.extend(str(symbol).upper() for symbol in universe.get("controls", []))
    return list(dict.fromkeys(symbols))


def yearly_chunks(start: date, end: date) -> list[DateChunk]:
    if start > end:
        raise ValueError("start must be on or before end")
    chunks: list[DateChunk] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(end, date(cursor.year, 12, 31))
        chunks.append(DateChunk(cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return chunks


def request_bounds(chunk: DateChunk) -> tuple[str, str]:
    """Return UTC RFC-3339 bounds covering inclusive New York dates."""

    local_start = datetime.combine(chunk.start, wall_time.min, tzinfo=NEW_YORK)
    local_end_exclusive = datetime.combine(
        chunk.end + timedelta(days=1), wall_time.min, tzinfo=NEW_YORK
    )
    utc_start = local_start.astimezone(UTC)
    utc_end = (local_end_exclusive - timedelta(microseconds=1)).astimezone(UTC)
    return (
        utc_start.isoformat().replace("+00:00", "Z"),
        utc_end.isoformat().replace("+00:00", "Z"),
    )


def parse_timestamp(value: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError(f"Timestamp has no timezone: {value!r}")
    return parsed.astimezone(UTC)


def is_regular_session_bar(
    timestamp_utc: datetime,
    sessions: Mapping[date, MarketSession] | None = None,
) -> bool:
    local = timestamp_utc.astimezone(NEW_YORK)
    local_time = local.timetz().replace(tzinfo=None)
    if sessions is None:
        if local.weekday() >= 5:
            return False
        session = MarketSession(wall_time(9, 30), wall_time(16, 0))
    else:
        session = sessions.get(local.date())
        if session is None:
            return False
    return session.open_time <= local_time < session.close_time


def parse_market_time(value: str) -> wall_time:
    for format_string in ("%H:%M", "%H:%M:%S"):
        try:
            return datetime.strptime(value, format_string).time()
        except ValueError:
            continue
    raise ValueError(f"Unsupported market-calendar time: {value!r}")


def normalize_market_calendar(
    payload: Sequence[Mapping[str, Any]],
) -> dict[date, MarketSession]:
    sessions: dict[date, MarketSession] = {}
    for record in payload:
        trade_date = date.fromisoformat(str(record["date"]))
        session = MarketSession(
            open_time=parse_market_time(str(record["open"])),
            close_time=parse_market_time(str(record["close"])),
        )
        if session.open_time >= session.close_time:
            raise AlpacaDownloadError(
                f"Invalid market session on {trade_date}: "
                f"{session.open_time} to {session.close_time}"
            )
        sessions[trade_date] = session
    return sessions


def normalize_bar(symbol: str, bar: Mapping[str, Any]) -> dict[str, Any]:
    timestamp_utc = parse_timestamp(str(bar["t"]))
    timestamp_et = timestamp_utc.astimezone(NEW_YORK)
    return {
        "symbol": symbol,
        "timestamp_utc": timestamp_utc.isoformat().replace("+00:00", "Z"),
        "timestamp_et": timestamp_et.isoformat(),
        "trade_date": timestamp_et.date().isoformat(),
        "bar_start_et": timestamp_et.strftime("%H:%M:%S"),
        "open": bar.get("o"),
        "high": bar.get("h"),
        "low": bar.get("l"),
        "close": bar.get("c"),
        "volume": bar.get("v"),
        "trade_count": bar.get("n"),
        "vwap": bar.get("vw"),
    }


def validate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    regular_only: bool,
    sessions: Mapping[date, MarketSession] | None = None,
) -> None:
    timestamps = [str(row["timestamp_utc"]) for row in rows]
    if timestamps != sorted(timestamps):
        raise AlpacaDownloadError("Bars are not sorted by timestamp")
    if len(timestamps) != len(set(timestamps)):
        raise AlpacaDownloadError("Duplicate timestamps detected")
    if regular_only:
        invalid = [
            value
            for value in timestamps
            if not is_regular_session_bar(parse_timestamp(value), sessions)
        ]
        if invalid:
            raise AlpacaDownloadError(
                f"Found {len(invalid)} bars outside the regular session"
            )


def safe_error_body(error: urllib.error.HTTPError) -> str:
    try:
        raw = error.read(4096).decode("utf-8", errors="replace")
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            message = parsed.get("message") or parsed.get("error")
            if message:
                return str(message)
        return raw.strip()
    except Exception:
        return ""


class AlpacaClient:
    def __init__(
        self,
        credentials: Credentials,
        *,
        timeout: float = 60.0,
        retries: int = 5,
        request_pause: float = 0.35,
    ) -> None:
        self.credentials = credentials
        self.timeout = timeout
        self.retries = retries
        self.request_pause = request_pause

    @property
    def headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.credentials.public_key,
            "APCA-API-SECRET-KEY": self.credentials.secret_key,
            "Accept": "application/json",
            "User-Agent": "stock-sector-correlation-research/1.0",
        }

    def _get_url_json(
        self, url: str, *, historical_market_data: bool
    ) -> tuple[Any, str | None]:
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(url, headers=self.headers, method="GET")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                    request_id = response.headers.get("X-Request-ID")
                if self.request_pause:
                    time.sleep(self.request_pause)
                return payload, request_id
            except urllib.error.HTTPError as exc:
                message = safe_error_body(exc)
                if exc.code in {401, 403}:
                    hint = " Check that the .env keys are Alpaca Trading API credentials."
                    if historical_market_data:
                        hint += (
                            " Historical SIP requests older than 15 minutes should "
                            "be eligible; recent SIP data requires a subscription."
                        )
                    raise AlpacaDownloadError(
                        f"Alpaca returned HTTP {exc.code}: {message or exc.reason}.{hint}"
                    ) from exc
                if exc.code not in {429, 500, 502, 503, 504}:
                    raise AlpacaDownloadError(
                        f"Alpaca returned HTTP {exc.code}: {message or exc.reason}"
                    ) from exc
                last_error = exc
                retry_after = exc.headers.get("Retry-After")
                wait = (
                    float(retry_after)
                    if retry_after and retry_after.replace(".", "", 1).isdigit()
                    else min(30.0, (2**attempt) + random.random())
                )
                time.sleep(wait)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
                time.sleep(min(30.0, (2**attempt) + random.random()))
        raise AlpacaDownloadError(
            f"Alpaca request failed after {self.retries + 1} attempts: {last_error}"
        )

    def get_json(
        self, params: Mapping[str, str | int]
    ) -> tuple[dict[str, Any], str | None]:
        query = urllib.parse.urlencode(params)
        payload, request_id = self._get_url_json(
            f"{API_URL}?{query}", historical_market_data=True
        )
        if not isinstance(payload, dict):
            raise AlpacaDownloadError("Alpaca returned a non-object bars response")
        return payload, request_id

    def market_calendar(
        self,
        start: date,
        end: date,
        *,
        api_base: str = "auto",
    ) -> tuple[list[dict[str, Any]], str, str | None]:
        bases = {
            "paper": [PAPER_CALENDAR_URL],
            "live": [LIVE_CALENDAR_URL],
            "auto": [PAPER_CALENDAR_URL, LIVE_CALENDAR_URL],
        }[api_base]
        errors: list[str] = []
        params = urllib.parse.urlencode(
            {
                "start": start.isoformat(),
                "end": end.isoformat(),
                "date_type": "TRADING",
            }
        )
        for base in bases:
            try:
                payload, request_id = self._get_url_json(
                    f"{base}?{params}", historical_market_data=False
                )
            except AlpacaDownloadError as exc:
                errors.append(f"{base}: {exc}")
                continue
            if not isinstance(payload, list) or not all(
                isinstance(item, dict) for item in payload
            ):
                raise AlpacaDownloadError(
                    f"Alpaca calendar at {base} returned an unexpected response"
                )
            return payload, base, request_id
        raise AlpacaDownloadError(
            "Could not authenticate to an Alpaca market-calendar endpoint. "
            + " | ".join(errors)
        )

    def bars(
        self,
        symbol: str,
        chunk: DateChunk,
        *,
        timeframe: str,
        feed: str,
        adjustment: str,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        start, end = request_bounds(chunk)
        params: dict[str, str | int] = {
            "symbols": symbol,
            "timeframe": timeframe,
            "start": start,
            "end": end,
            "limit": PAGE_LIMIT,
            "adjustment": adjustment,
            "feed": feed,
            "sort": "asc",
        }
        bars: list[dict[str, Any]] = []
        request_ids: list[str] = []
        while True:
            payload, request_id = self.get_json(params)
            if request_id:
                request_ids.append(request_id)
            payload_bars = payload.get("bars", {})
            if not isinstance(payload_bars, dict):
                raise AlpacaDownloadError("Alpaca response field 'bars' is not an object")
            symbol_bars = payload_bars.get(symbol, [])
            if not isinstance(symbol_bars, list):
                raise AlpacaDownloadError(
                    f"Alpaca response bars for {symbol} are not a list"
                )
            bars.extend(item for item in symbol_bars if isinstance(item, dict))
            page_token = payload.get("next_page_token")
            if not page_token:
                break
            params["page_token"] = str(page_token)
        return bars, request_ids


def output_paths(
    output_dir: Path,
    *,
    timeframe: str,
    feed: str,
    adjustment: str,
    symbol: str,
    chunk: DateChunk,
) -> tuple[Path, Path]:
    normalized_timeframe = timeframe.lower().replace("min", "min")
    base = output_dir / normalized_timeframe / feed / adjustment / symbol
    return base / f"{chunk.label}.csv.gz", base / f"{chunk.label}.manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=path.name + ".", delete=False
        ) as raw_handle:
            temp_path = Path(raw_handle.name)
            with gzip.open(raw_handle, mode="wt", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=path.name + ".",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
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
        and manifest.get("download_schema_version") == DOWNLOAD_SCHEMA_VERSION
        and manifest.get("csv_sha256") == sha256_file(csv_path)
    )


def calendar_cache_path(output_dir: Path, start: date, end: date) -> Path:
    return output_dir / "calendar" / f"{start.isoformat()}_{end.isoformat()}.json"


def load_or_fetch_market_calendar(
    client: AlpacaClient,
    *,
    start: date,
    end: date,
    output_dir: Path,
    api_base: str,
    overwrite: bool,
) -> tuple[dict[date, MarketSession], Path, str]:
    path = calendar_cache_path(output_dir, start, end)
    if path.exists() and not overwrite:
        cached = json.loads(path.read_text(encoding="utf-8"))
        if (
            cached.get("status") == "complete"
            and cached.get("start") == start.isoformat()
            and cached.get("end") == end.isoformat()
            and isinstance(cached.get("sessions"), list)
        ):
            sessions = normalize_market_calendar(cached["sessions"])
            return sessions, path, sha256_file(path)

    payload, source_url, request_id = client.market_calendar(
        start, end, api_base=api_base
    )
    sessions = normalize_market_calendar(payload)
    normalized_sessions = [
        {
            "date": session_date.isoformat(),
            "open": session.open_time.isoformat(timespec="minutes"),
            "close": session.close_time.isoformat(timespec="minutes"),
        }
        for session_date, session in sorted(sessions.items())
    ]
    cache = {
        "status": "complete",
        "source": "Alpaca Trading API market calendar",
        "source_url": source_url,
        "request_id": request_id,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "session_count": len(normalized_sessions),
        "sessions": normalized_sessions,
        "fetched_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "credential_values_recorded": False,
    }
    write_json_atomic(path, cache)
    return sessions, path, sha256_file(path)


def chunk_plan(
    symbols: Iterable[str], start: date, end: date
) -> Iterator[tuple[str, DateChunk]]:
    chunks = yearly_chunks(start, end)
    for symbol in symbols:
        for chunk in chunks:
            yield symbol, chunk


def download_chunk(
    client: AlpacaClient,
    *,
    symbol: str,
    chunk: DateChunk,
    timeframe: str,
    feed: str,
    adjustment: str,
    regular_only: bool,
    output_dir: Path,
    overwrite: bool,
    sessions: Mapping[date, MarketSession],
    calendar_path: Path,
    calendar_sha256: str,
) -> dict[str, Any]:
    csv_path, manifest_path = output_paths(
        output_dir,
        timeframe=timeframe,
        feed=feed,
        adjustment=adjustment,
        symbol=symbol,
        chunk=chunk,
    )
    if not overwrite and manifest_is_complete(csv_path, manifest_path):
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {
            "action": "skipped",
            "symbol": symbol,
            "chunk": chunk.label,
            "row_count": existing.get("row_count", 0),
            "csv_path": str(csv_path),
        }

    raw_bars, request_ids = client.bars(
        symbol,
        chunk,
        timeframe=timeframe,
        feed=feed,
        adjustment=adjustment,
    )
    rows: list[dict[str, Any]] = []
    for bar in raw_bars:
        row = normalize_bar(symbol, bar)
        trade_date = date.fromisoformat(str(row["trade_date"]))
        if not (chunk.start <= trade_date <= chunk.end):
            continue
        if regular_only and not is_regular_session_bar(
            parse_timestamp(str(row["timestamp_utc"])), sessions
        ):
            continue
        rows.append(row)
    rows.sort(key=lambda item: str(item["timestamp_utc"]))
    validate_rows(rows, regular_only=regular_only, sessions=sessions)

    write_csv_atomic(csv_path, rows)
    dates = sorted({str(row["trade_date"]) for row in rows})
    daily_counts: dict[str, int] = {}
    for row in rows:
        day = str(row["trade_date"])
        daily_counts[day] = daily_counts.get(day, 0) + 1
    manifest = {
        "status": "complete",
        "download_schema_version": DOWNLOAD_SCHEMA_VERSION,
        "source": "Alpaca Market Data API v2 historical stock bars",
        "endpoint": API_URL,
        "symbol": symbol,
        "chunk": {"start": chunk.start.isoformat(), "end": chunk.end.isoformat()},
        "request": {
            "timeframe": timeframe,
            "feed": feed,
            "adjustment": adjustment,
            "regular_session_only": regular_only,
            "limit": PAGE_LIMIT,
        },
        "request_ids": request_ids,
        "market_calendar": {
            "path": str(calendar_path),
            "sha256": calendar_sha256,
            "filter": "official daily open inclusive, close exclusive",
        },
        "fetched_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "raw_bar_count": len(raw_bars),
        "row_count": len(rows),
        "trading_date_count": len(dates),
        "first_timestamp_utc": rows[0]["timestamp_utc"] if rows else None,
        "last_timestamp_utc": rows[-1]["timestamp_utc"] if rows else None,
        "minimum_bars_per_date": min(daily_counts.values()) if daily_counts else None,
        "maximum_bars_per_date": max(daily_counts.values()) if daily_counts else None,
        "csv_path": str(csv_path),
        "csv_sha256": sha256_file(csv_path),
        "credential_values_recorded": False,
    }
    write_json_atomic(manifest_path, manifest)
    return {
        "action": "downloaded",
        "symbol": symbol,
        "chunk": chunk.label,
        "row_count": len(rows),
        "trading_date_count": len(dates),
        "raw_bar_count": len(raw_bars),
        "csv_path": str(csv_path),
        "manifest_path": str(manifest_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--symbols",
        type=parse_symbols,
        help="Comma-separated override. Defaults to every configured stock, ETF, and control.",
    )
    parser.add_argument("--start", type=parse_date, default=DEFAULT_START)
    parser.add_argument("--end", type=parse_date, default=DEFAULT_END)
    parser.add_argument("--timeframe", default=None)
    parser.add_argument("--feed", choices=("sip", "iex"), default=None)
    parser.add_argument(
        "--adjustment",
        choices=("raw", "split", "dividend", "spin-off", "all"),
        default=None,
    )
    parser.add_argument(
        "--include-extended-hours",
        action="store_true",
        help="Keep all returned bars. The default keeps 09:30-16:00 America/New_York.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument(
        "--calendar-api-base",
        choices=("auto", "paper", "live"),
        default="auto",
        help="Trading API calendar credential target. Auto tries paper, then live.",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--request-pause", type=float, default=0.35)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and print the plan without loading credentials.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    universe = load_universe(args.config)
    symbols = args.symbols or symbols_from_universe(universe)
    timeframe = args.timeframe or str(universe.get("timeframe", "15Min"))
    feed = args.feed or str(universe.get("feed", "sip"))
    adjustment = args.adjustment or str(universe.get("adjustment", "all"))
    regular_only = not args.include_extended_hours
    plan = list(chunk_plan(symbols, args.start, args.end))

    print(
        json.dumps(
            {
                "mode": "dry-run" if args.dry_run else "download",
                "symbols": symbols,
                "symbol_count": len(symbols),
                "start": args.start.isoformat(),
                "end": args.end.isoformat(),
                "chunk_count": len(plan),
                "timeframe": timeframe,
                "feed": feed,
                "adjustment": adjustment,
                "regular_session_only": regular_only,
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )
    if args.dry_run:
        return 0

    credentials = load_credentials(args.env_file)
    client = AlpacaClient(
        credentials,
        timeout=args.timeout,
        retries=args.retries,
        request_pause=args.request_pause,
    )
    sessions, calendar_path, calendar_sha256 = load_or_fetch_market_calendar(
        client,
        start=args.start,
        end=args.end,
        output_dir=args.output_dir,
        api_base=args.calendar_api_base,
        overwrite=args.overwrite,
    )
    print(
        f"Loaded {len(sessions)} official market sessions from {calendar_path}"
    )
    downloaded = skipped = rows = 0
    for index, (symbol, chunk) in enumerate(plan, start=1):
        result = download_chunk(
            client,
            symbol=symbol,
            chunk=chunk,
            timeframe=timeframe,
            feed=feed,
            adjustment=adjustment,
            regular_only=regular_only,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            sessions=sessions,
            calendar_path=calendar_path,
            calendar_sha256=calendar_sha256,
        )
        rows += int(result["row_count"])
        if result["action"] == "downloaded":
            downloaded += 1
        else:
            skipped += 1
        print(
            f"[{index}/{len(plan)}] {result['action']} "
            f"{symbol} {chunk.label}: {result['row_count']} rows"
        )
    print(
        json.dumps(
            {
                "status": "complete",
                "downloaded_chunks": downloaded,
                "skipped_chunks": skipped,
                "total_rows_seen": rows,
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
