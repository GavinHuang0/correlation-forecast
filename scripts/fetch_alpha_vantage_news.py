"""Collect Alpha Vantage NEWS_SENTIMENT history one ticker at a time.

Alpha Vantage treats a comma-separated ``tickers`` query as a logical AND, so
this collector intentionally issues a separate request for every ticker and
time slice.  Responses at the 1,000-item API ceiling are not accepted as
complete: their time slice is bisected and queued for later requests.  The
queue and response hashes live in ``state.json``, making the collection
resumable across the default 25-call invocations.

Only Python's standard library is required.  ``ALPHA_VANTAGE_KEY`` is read from
the process environment or a local ``.env`` file.  It is placed only in the
HTTPS query request and is never printed or stored in response files, state,
or manifests.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, time as wall_time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


API_URL = "https://www.alphavantage.co/query"
DEFAULT_CONFIG = Path("config/price_universe.json")
DEFAULT_ENV_FILE = Path(".env")
DEFAULT_OUTPUT_DIR = Path("data/raw/alpha_vantage_news/full_backfill_v1")
DEFAULT_START = date(2016, 1, 1)
DEFAULT_END = date(2026, 6, 30)
DEFAULT_MAX_CALLS = 25
DEFAULT_SLICE_DAYS = 30
DEFAULT_CALLS_PER_MINUTE = 5.0
RESULT_LIMIT = 1_000
STATE_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
UTC = timezone.utc
ONE_MINUTE = timedelta(minutes=1)
USER_AGENT = "correlation-forecast-alpha-vantage-news/1.0"
_ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TICKER = re.compile(r"^[A-Z][A-Z0-9.-]*$")


class AlphaVantageNewsError(RuntimeError):
    """Raised when a NEWS_SENTIMENT response cannot safely be collected."""


class AlphaVantageRateLimitError(AlphaVantageNewsError):
    """Raised for Alpha Vantage ``Information`` or ``Note`` responses."""


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected an ISO date such as 2024-01-02, received {value!r}"
        ) from exc


def parse_tickers(value: str) -> list[str]:
    tickers = [part.strip().upper() for part in value.split(",") if part.strip()]
    if not tickers:
        raise argparse.ArgumentTypeError("At least one ticker is required")
    invalid = [ticker for ticker in tickers if not _TICKER.fullmatch(ticker)]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Unsupported ticker format: {', '.join(invalid)}"
        )
    return list(dict.fromkeys(tickers))


def _parse_dotenv_value(raw_value: str, *, path: Path, line_number: int) -> str:
    value = raw_value.strip()
    if not value:
        return ""
    if value[0] not in {"'", '"'}:
        comment = re.search(r"\s+#", value)
        return value[: comment.start()].rstrip() if comment else value

    quote = value[0]
    output: list[str] = []
    escaped = False
    closing_index: int | None = None
    for index, character in enumerate(value[1:], start=1):
        if quote == '"' and escaped:
            output.append(
                {"n": "\n", "r": "\r", "t": "\t"}.get(character, character)
            )
            escaped = False
        elif quote == '"' and character == "\\":
            escaped = True
        elif character == quote:
            closing_index = index
            break
        else:
            output.append(character)
    if escaped:
        output.append("\\")
    if closing_index is None:
        raise ValueError(f"{path}:{line_number}: unterminated quoted value")
    trailing = value[closing_index + 1 :].strip()
    if trailing and not trailing.startswith("#"):
        raise ValueError(
            f"{path}:{line_number}: unexpected text after quoted value"
        )
    return "".join(output)


def load_dotenv_values(path: Path) -> dict[str, str]:
    """Read a conservative dotenv subset without changing the environment."""

    if not path.exists():
        return {}
    values: dict[str, str] = {}
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
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not _ENV_KEY.fullmatch(key):
            raise ValueError(f"{path}:{line_number}: invalid environment key")
        values[key] = _parse_dotenv_value(
            raw_value, path=path, line_number=line_number
        )
    return values


def load_api_key(env_file: Path) -> str:
    values = load_dotenv_values(env_file)
    api_key = os.environ.get("ALPHA_VANTAGE_KEY") or values.get(
        "ALPHA_VANTAGE_KEY"
    )
    if not api_key:
        raise AlphaVantageNewsError(
            "Missing ALPHA_VANTAGE_KEY in the process environment or dotenv file"
        )
    return api_key


def load_stock_tickers(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sectors = payload.get("sectors") if isinstance(payload, dict) else None
    if not isinstance(sectors, list) or not sectors:
        raise ValueError(f"{path}: sectors must be a non-empty array")
    tickers: list[str] = []
    for sector in sectors:
        if not isinstance(sector, Mapping):
            raise ValueError(f"{path}: every sector must be an object")
        stocks = sector.get("stocks")
        if not isinstance(stocks, list) or not stocks:
            raise ValueError(f"{path}: every sector requires a stocks array")
        for raw_ticker in stocks:
            ticker = str(raw_ticker).strip().upper()
            if not _TICKER.fullmatch(ticker):
                raise ValueError(f"{path}: invalid stock ticker {raw_ticker!r}")
            tickers.append(ticker)
    if len(tickers) != len(set(tickers)):
        raise ValueError(f"{path}: stock tickers must be unique")
    return tickers


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_bytes_atomic(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=path.name + ".", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(value)
        for attempt in range(10):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    write_bytes_atomic(path, encoded)


def deterministic_gzip(value: bytes) -> bytes:
    return gzip.compress(value, compresslevel=9, mtime=0)


def redact_secret(value: str, secret: str) -> str:
    return value.replace(secret, "[REDACTED]") if secret else value


def assert_secret_absent(value: Any, secret: str, *, destination: str) -> None:
    if not secret:
        return
    if isinstance(value, bytes):
        present = secret.encode("utf-8") in value
    else:
        present = secret in json.dumps(value, sort_keys=True, ensure_ascii=False)
    if present:
        raise AlphaVantageNewsError(
            f"Refusing to write a credential value to {destination}"
        )


def format_alpha_time(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y%m%dT%H%M")


def parse_alpha_time(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y%m%dT%H%M").replace(tzinfo=UTC)
    except ValueError as exc:
        raise AlphaVantageNewsError(
            f"Invalid Alpha Vantage time in resume state: {value!r}"
        ) from exc


def build_initial_slices(
    tickers: Sequence[str],
    *,
    start: date,
    end: date,
    slice_days: int,
) -> list[dict[str, Any]]:
    if start > end:
        raise ValueError("start must be on or before end")
    if slice_days <= 0:
        raise ValueError("slice_days must be positive")
    overall_start = datetime.combine(start, wall_time.min, tzinfo=UTC)
    overall_end = datetime.combine(end, wall_time(23, 59), tzinfo=UTC)
    slices: list[dict[str, Any]] = []
    for ticker in tickers:
        cursor = overall_start
        while cursor <= overall_end:
            slice_end = min(
                overall_end, cursor + timedelta(days=slice_days) - ONE_MINUTE
            )
            slices.append(
                {
                    "ticker": ticker,
                    "time_from": format_alpha_time(cursor),
                    "time_to": format_alpha_time(slice_end),
                    "split_depth": 0,
                }
            )
            cursor = slice_end + ONE_MINUTE
    return slices


def split_slice(task: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    start = parse_alpha_time(str(task["time_from"]))
    end = parse_alpha_time(str(task["time_to"]))
    total_minutes = int((end - start).total_seconds() // 60)
    if total_minutes < 1:
        raise AlphaVantageNewsError(
            "A one-minute NEWS_SENTIMENT slice still reached 1,000 items"
        )
    midpoint = start + timedelta(minutes=total_minutes // 2)
    depth = int(task.get("split_depth", 0)) + 1
    left = {
        "ticker": str(task["ticker"]),
        "time_from": format_alpha_time(start),
        "time_to": format_alpha_time(midpoint),
        "split_depth": depth,
    }
    right = {
        "ticker": str(task["ticker"]),
        "time_from": format_alpha_time(midpoint + ONE_MINUTE),
        "time_to": format_alpha_time(end),
        "split_depth": depth,
    }
    return left, right


def provider_message(payload: Mapping[str, Any]) -> tuple[str, str] | None:
    for field in ("Information", "Note", "Error Message"):
        value = payload.get(field)
        if value:
            return field, str(value)
    return None


def response_at_result_ceiling(payload: Mapping[str, Any]) -> bool:
    feed = payload.get("feed")
    if not isinstance(feed, list):
        return False
    try:
        reported_items = int(str(payload.get("items", len(feed))))
    except ValueError:
        reported_items = len(feed)
    return len(feed) >= RESULT_LIMIT or reported_items >= RESULT_LIMIT


class AlphaVantageClient:
    """One-ticker NEWS_SENTIMENT client with request pacing."""

    def __init__(
        self,
        api_key: str,
        *,
        timeout: float = 60.0,
        calls_per_minute: float = DEFAULT_CALLS_PER_MINUTE,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if calls_per_minute <= 0:
            raise ValueError("calls_per_minute must be positive")
        self.api_key = api_key
        self.timeout = timeout
        self.minimum_interval = 60.0 / calls_per_minute
        self._sleeper = sleeper
        self._clock = clock
        self._last_request_at: float | None = None

    def _pace(self) -> None:
        now = self._clock()
        if self._last_request_at is not None:
            wait = self.minimum_interval - (now - self._last_request_at)
            if wait > 0:
                self._sleeper(wait)
                now = self._clock()
        self._last_request_at = now

    def fetch_slice(
        self, *, ticker: str, time_from: str, time_to: str
    ) -> tuple[bytes, dict[str, Any]]:
        if "," in ticker:
            raise AlphaVantageNewsError(
                "NEWS_SENTIMENT requests must contain exactly one ticker"
            )
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "time_from": time_from,
            "time_to": time_to,
            "sort": "EARLIEST",
            "limit": str(RESULT_LIMIT),
            "apikey": self.api_key,
        }
        request = urllib.request.Request(
            f"{API_URL}?{urllib.parse.urlencode(params)}",
            headers={"Accept": "application/json", "User-Agent": USER_AGENT},
            method="GET",
        )
        self._pace()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                raise AlphaVantageRateLimitError(
                    f"Alpha Vantage rate limit reached for {ticker}"
                ) from exc
            raise AlphaVantageNewsError(
                f"Alpha Vantage returned HTTP {exc.code} for {ticker}"
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            reason = redact_secret(str(getattr(exc, "reason", "")), self.api_key)
            raise AlphaVantageNewsError(
                f"Alpha Vantage request failed for {ticker}: {reason[:300]}"
            ) from exc

        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AlphaVantageNewsError(
                f"Alpha Vantage returned invalid JSON for {ticker}"
            ) from exc
        if not isinstance(payload, dict):
            raise AlphaVantageNewsError(
                f"Alpha Vantage returned a non-object for {ticker}"
            )
        message = provider_message(payload)
        if message is not None:
            field, detail = message
            detail = redact_secret(detail, self.api_key)
            if field in {"Information", "Note"}:
                raise AlphaVantageRateLimitError(
                    f"Alpha Vantage {field} for {ticker}: {detail[:300]}"
                )
            raise AlphaVantageNewsError(
                f"Alpha Vantage error for {ticker}: {detail[:300]}"
            )
        # Alpha may echo an API key inside a provider error/limit message.
        # Parse and redact such messages before applying the strict raw-page
        # persistence guard. Successful feed payloads must still be entirely
        # credential-free before they can be cached.
        assert_secret_absent(raw, self.api_key, destination="a response page")
        feed = payload.get("feed")
        if not isinstance(feed, list):
            raise AlphaVantageNewsError(
                f"Alpha Vantage response for {ticker} has no feed array"
            )
        return raw, payload


def collection_scope(
    *,
    tickers: Sequence[str],
    start: date,
    end: date,
    slice_days: int,
) -> dict[str, Any]:
    return {
        "tickers": list(tickers),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "initial_slice_days": slice_days,
        "limit": RESULT_LIMIT,
        "sort": "EARLIEST",
        "ticker_request_semantics": "one ticker per request",
    }


def _new_state(
    scope: Mapping[str, Any], pending_slices: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "provider": "Alpha Vantage",
        "dataset": "NEWS_SENTIMENT",
        "endpoint": API_URL,
        "scope": dict(scope),
        "status": "in_progress",
        "pending_slices": [dict(task) for task in pending_slices],
        "completed_slices": [],
        "responses": [],
    }


def _safe_cached_path(output_dir: Path, relative_path: str) -> Path:
    root = output_dir.resolve()
    path = (output_dir / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise AlphaVantageNewsError(
            "Cached response path escapes the output directory"
        ) from exc
    return path


def _validate_cached_responses(
    output_dir: Path, state: Mapping[str, Any]
) -> None:
    responses = state.get("responses")
    if not isinstance(responses, list):
        raise AlphaVantageNewsError("Resume state responses field is invalid")
    for record in responses:
        if not isinstance(record, Mapping):
            raise AlphaVantageNewsError(
                "Resume state contains an invalid response record"
            )
        path = _safe_cached_path(output_dir, str(record.get("path", "")))
        if not path.is_file():
            raise AlphaVantageNewsError(f"Cached response is missing: {path}")
        if sha256_file(path) != record.get("gzip_sha256"):
            raise AlphaVantageNewsError(f"Cached response hash mismatch: {path}")


def load_or_initialize_state(
    output_dir: Path,
    scope: Mapping[str, Any],
    pending_slices: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    state_path = output_dir / "state.json"
    if not state_path.exists():
        return _new_state(scope, pending_slices)
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AlphaVantageNewsError(
            f"Could not read resume state at {state_path}"
        ) from exc
    if not isinstance(state, dict):
        raise AlphaVantageNewsError("Resume state must be a JSON object")
    if state.get("schema_version") != STATE_SCHEMA_VERSION:
        raise AlphaVantageNewsError("Resume state schema version is incompatible")
    if state.get("scope") != dict(scope):
        raise AlphaVantageNewsError(
            "Existing resume state has a different collection scope; "
            "choose another output directory"
        )
    _validate_cached_responses(output_dir, state)
    return state


def build_manifest(
    state: Mapping[str, Any], *, calls_per_minute: float
) -> dict[str, Any]:
    scope = dict(state["scope"])
    responses = list(state.get("responses", []))
    completed = list(state.get("completed_slices", []))
    pending = list(state.get("pending_slices", []))
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": state["status"],
        "provider": "Alpha Vantage",
        "dataset": "NEWS_SENTIMENT",
        "endpoint": API_URL,
        "scope": scope,
        "time_slicing": {
            "initial_days": scope["initial_slice_days"],
            "ceiling_behavior": (
                "bisect every response containing 1000 or more feed items"
            ),
            "minimum_granularity": "one minute",
        },
        "rate_policy": {
            "maximum_calls_per_default_invocation": DEFAULT_MAX_CALLS,
            "calls_per_minute": calls_per_minute,
            "minimum_interval_seconds": 60.0 / calls_per_minute,
        },
        "completed_slices": completed,
        "pending_slice_count": len(pending),
        "responses": responses,
        "totals": {
            "configured_tickers": len(scope["tickers"]),
            "completed_slices": len(completed),
            "pending_slices": len(pending),
            "responses": len(responses),
            "accepted_feed_items": sum(
                int(response["feed_count"])
                for response in responses
                if not response["at_result_ceiling"]
            ),
            "ceiling_responses": sum(
                bool(response["at_result_ceiling"]) for response in responses
            ),
        },
        "multi_ticker_batching": False,
        "credential_transport": "HTTPS query parameter",
        "credential_values_recorded": False,
    }


def _persist_state_and_manifest(
    output_dir: Path,
    state: Mapping[str, Any],
    *,
    calls_per_minute: float,
    secret: str,
) -> None:
    manifest = build_manifest(state, calls_per_minute=calls_per_minute)
    assert_secret_absent(state, secret, destination="resume state")
    assert_secret_absent(manifest, secret, destination="audit manifest")
    write_json_atomic(output_dir / "state.json", state)
    write_json_atomic(output_dir / "manifest.json", manifest)


def _response_path(output_dir: Path, task: Mapping[str, Any]) -> Path:
    return (
        output_dir
        / "responses"
        / str(task["ticker"])
        / f"{task['time_from']}_{task['time_to']}.json.gz"
    )


def collect_news(
    *,
    client: AlphaVantageClient,
    output_dir: Path,
    tickers: Sequence[str],
    start: date,
    end: date,
    max_calls: int = DEFAULT_MAX_CALLS,
    slice_days: int = DEFAULT_SLICE_DAYS,
    calls_per_minute: float = DEFAULT_CALLS_PER_MINUTE,
) -> dict[str, Any]:
    if start > end:
        raise ValueError("start must be on or before end")
    if max_calls < 0:
        raise ValueError("max_calls must be nonnegative")
    if slice_days <= 0:
        raise ValueError("slice_days must be positive")
    normalized_tickers = parse_tickers(",".join(tickers))
    initial = build_initial_slices(
        normalized_tickers, start=start, end=end, slice_days=slice_days
    )
    scope = collection_scope(
        tickers=normalized_tickers,
        start=start,
        end=end,
        slice_days=slice_days,
    )
    state = load_or_initialize_state(output_dir, scope, initial)
    _persist_state_and_manifest(
        output_dir,
        state,
        calls_per_minute=calls_per_minute,
        secret=client.api_key,
    )

    calls_made = 0
    while (
        state["status"] == "in_progress"
        and state["pending_slices"]
        and calls_made < max_calls
    ):
        task = dict(state["pending_slices"][0])
        raw, payload = client.fetch_slice(
            ticker=str(task["ticker"]),
            time_from=str(task["time_from"]),
            time_to=str(task["time_to"]),
        )
        calls_made += 1
        feed = payload["feed"]
        at_ceiling = response_at_result_ceiling(payload)
        path = _response_path(output_dir, task)
        compressed = deterministic_gzip(raw)
        write_bytes_atomic(path, compressed)
        relative_path = path.relative_to(output_dir).as_posix()
        response_record = {
            "ticker": task["ticker"],
            "time_from": task["time_from"],
            "time_to": task["time_to"],
            "split_depth": task["split_depth"],
            "path": relative_path,
            "gzip_sha256": sha256_bytes(compressed),
            "json_sha256": sha256_bytes(raw),
            "feed_count": len(feed),
            "at_result_ceiling": at_ceiling,
        }
        assert_secret_absent(
            response_record, client.api_key, destination="response audit metadata"
        )
        state["responses"].append(response_record)

        if at_ceiling:
            try:
                left, right = split_slice(task)
            except AlphaVantageNewsError:
                state["status"] = "blocked_at_one_minute_ceiling"
                _persist_state_and_manifest(
                    output_dir,
                    state,
                    calls_per_minute=calls_per_minute,
                    secret=client.api_key,
                )
                raise
            state["pending_slices"] = [
                left,
                right,
                *state["pending_slices"][1:],
            ]
        else:
            state["completed_slices"].append(task)
            state["pending_slices"] = state["pending_slices"][1:]
        if not state["pending_slices"]:
            state["status"] = "complete"
        _persist_state_and_manifest(
            output_dir,
            state,
            calls_per_minute=calls_per_minute,
            secret=client.api_key,
        )

    return {
        "status": state["status"],
        "calls_made": calls_made,
        "responses_cached": len(state["responses"]),
        "completed_slices": len(state["completed_slices"]),
        "pending_slices": len(state["pending_slices"]),
        "stop_reason": (
            "complete"
            if state["status"] == "complete"
            else state["status"]
            if str(state["status"]).startswith("blocked_")
            else "max_calls"
        ),
        "output_dir": str(output_dir),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--tickers",
        type=parse_tickers,
        help="Comma-separated override; defaults to the 30 configured stocks.",
    )
    parser.add_argument("--start", type=parse_date, default=DEFAULT_START)
    parser.add_argument("--end", type=parse_date, default=DEFAULT_END)
    parser.add_argument("--slice-days", type=int, default=DEFAULT_SLICE_DAYS)
    parser.add_argument("--max-calls", type=int, default=DEFAULT_MAX_CALLS)
    parser.add_argument(
        "--calls-per-minute", type=float, default=DEFAULT_CALLS_PER_MINUTE
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.start > args.end:
        raise ValueError("start must be on or before end")
    if args.slice_days <= 0:
        raise ValueError("slice-days must be positive")
    if args.max_calls < 0:
        raise ValueError("max-calls must be nonnegative")
    if args.calls_per_minute <= 0:
        raise ValueError("calls-per-minute must be positive")
    tickers = args.tickers or load_stock_tickers(args.config)
    initial_slices = build_initial_slices(
        tickers, start=args.start, end=args.end, slice_days=args.slice_days
    )
    plan = {
        "mode": "dry-run" if args.dry_run else "download",
        "provider": "Alpha Vantage",
        "function": "NEWS_SENTIMENT",
        "tickers": tickers,
        "ticker_count": len(tickers),
        "one_ticker_per_request": True,
        "start": args.start.isoformat(),
        "end": args.end.isoformat(),
        "initial_slice_days": args.slice_days,
        "initial_slice_count": len(initial_slices),
        "result_limit": RESULT_LIMIT,
        "max_calls": args.max_calls,
        "calls_per_minute": args.calls_per_minute,
        "output_dir": str(args.output_dir),
        "resume_state": str(args.output_dir / "state.json"),
    }
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return 0

    api_key = load_api_key(args.env_file)
    client = AlphaVantageClient(
        api_key,
        timeout=args.timeout,
        calls_per_minute=args.calls_per_minute,
    )
    result = collect_news(
        client=client,
        output_dir=args.output_dir,
        tickers=tickers,
        start=args.start,
        end=args.end,
        max_calls=args.max_calls,
        slice_days=args.slice_days,
        calls_per_minute=args.calls_per_minute,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        AlphaVantageNewsError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
