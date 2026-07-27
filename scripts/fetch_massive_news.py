"""Collect ordinary Massive news for the configured 30-stock universe.

The collector uses only Python's standard library and is deliberately separate
from the model experiment paths.  Each API response is stored as one
deterministic gzip-compressed JSON page.  ``state.json`` records the next
cursor, so stopping at ``--max-requests`` (or stopping the process) is safe to
resume with the same collection scope.

Credentials are read from ``MASSIVE_API_KEY`` in the process environment or a
local ``.env`` file.  The key is sent only in the HTTPS Authorization header;
it is never printed or written to state, response files, or manifests.
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
from datetime import date
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


API_URL = "https://api.massive.com/v2/reference/news"
DEFAULT_CONFIG = Path("config/price_universe.json")
DEFAULT_ENV_FILE = Path(".env")
DEFAULT_OUTPUT_DIR = Path("data/raw/massive/ordinary_news/free_v1")
DEFAULT_START = date(2016, 6, 22)
DEFAULT_END = date(2026, 6, 30)
DEFAULT_MAX_REQUESTS = 5
DEFAULT_CALLS_PER_MINUTE = 5.0
PAGE_LIMIT = 1_000
STATE_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
USER_AGENT = "correlation-forecast-massive-news/1.0"
_ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TICKER = re.compile(r"^[A-Z][A-Z0-9.-]*$")


class MassiveNewsError(RuntimeError):
    """Raised when a Massive response cannot safely be collected."""


class MassiveRateLimitError(MassiveNewsError):
    """Raised for an explicit Massive rate-limit response."""


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
    """Parse a dotenv value, including quoted values and trailing comments."""

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
    """Read dotenv values without mutating ``os.environ``."""

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
    api_key = os.environ.get("MASSIVE_API_KEY") or values.get("MASSIVE_API_KEY")
    if not api_key:
        raise MassiveNewsError(
            "Missing MASSIVE_API_KEY in the process environment or dotenv file"
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
                # A short-lived Windows reader or indexer can hold the
                # destination between close and replace. The temporary file
                # remains on the same volume, so retrying preserves atomicity.
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
        raise MassiveNewsError(
            f"Refusing to write a credential value to {destination}"
        )


def extract_cursor(next_url: Any) -> str | None:
    if next_url in {None, ""}:
        return None
    if not isinstance(next_url, str):
        raise MassiveNewsError("Massive next_url is not a string")
    query = urllib.parse.parse_qs(
        urllib.parse.urlsplit(next_url).query, keep_blank_values=True
    )
    cursors = query.get("cursor", [])
    if len(cursors) != 1 or not cursors[0]:
        raise MassiveNewsError("Massive next_url does not contain one cursor")
    return cursors[0]


class MassiveClient:
    """Small HTTPS client with a sliding request-start interval."""

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

    def fetch_page(
        self,
        *,
        ticker: str,
        start: date,
        end: date,
        cursor: str | None,
        limit: int = PAGE_LIMIT,
    ) -> tuple[bytes, dict[str, Any]]:
        params: dict[str, str | int] = {
            "ticker": ticker,
            "published_utc.gte": f"{start.isoformat()}T00:00:00Z",
            "published_utc.lte": f"{end.isoformat()}T23:59:59.999999Z",
            "sort": "published_utc",
            "order": "asc",
            "limit": limit,
        }
        if cursor:
            params["cursor"] = cursor
        url = f"{API_URL}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": USER_AGENT,
            },
            method="GET",
        )
        self._pace()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read(4096).decode("utf-8", errors="replace")
            message = redact_secret(body.strip(), self.api_key)
            if exc.code == 429:
                raise MassiveRateLimitError(
                    f"Massive rate limit reached for {ticker}"
                ) from exc
            detail = f": {message[:300]}" if message else ""
            raise MassiveNewsError(
                f"Massive returned HTTP {exc.code} for {ticker}{detail}"
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            reason = redact_secret(str(getattr(exc, "reason", "")), self.api_key)
            raise MassiveNewsError(
                f"Massive request failed for {ticker}: {reason[:300]}"
            ) from exc

        assert_secret_absent(raw, self.api_key, destination="a response page")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MassiveNewsError(
                f"Massive returned invalid JSON for {ticker}"
            ) from exc
        if not isinstance(payload, dict):
            raise MassiveNewsError(f"Massive returned a non-object for {ticker}")
        if str(payload.get("status", "")).upper() == "ERROR":
            message = redact_secret(str(payload.get("error", "")), self.api_key)
            raise MassiveNewsError(
                f"Massive returned an error for {ticker}: {message[:300]}"
            )
        results = payload.get("results")
        if not isinstance(results, list):
            raise MassiveNewsError(
                f"Massive response for {ticker} has no results array"
            )
        return raw, payload


def collection_scope(
    *, tickers: Sequence[str], start: date, end: date
) -> dict[str, Any]:
    return {
        "tickers": list(tickers),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "limit": PAGE_LIMIT,
        "sort": "published_utc",
        "order": "asc",
    }


def _new_state(scope: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "provider": "Massive",
        "dataset": "ordinary_news",
        "endpoint": API_URL,
        "scope": dict(scope),
        "status": "in_progress",
        "ticker_index": 0,
        "next_cursor": None,
        "next_page_number": 1,
        "completed_tickers": [],
        "pages": [],
    }


def _safe_cached_path(output_dir: Path, relative_path: str) -> Path:
    root = output_dir.resolve()
    path = (output_dir / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise MassiveNewsError("Cached page path escapes the output directory") from exc
    return path


def _validate_cached_pages(output_dir: Path, state: Mapping[str, Any]) -> None:
    pages = state.get("pages")
    if not isinstance(pages, list):
        raise MassiveNewsError("Resume state pages field is invalid")
    for record in pages:
        if not isinstance(record, Mapping):
            raise MassiveNewsError("Resume state contains an invalid page record")
        path = _safe_cached_path(output_dir, str(record.get("path", "")))
        if not path.is_file():
            raise MassiveNewsError(f"Cached response page is missing: {path}")
        if sha256_file(path) != record.get("gzip_sha256"):
            raise MassiveNewsError(f"Cached response page hash mismatch: {path}")


def load_or_initialize_state(
    output_dir: Path, scope: Mapping[str, Any]
) -> dict[str, Any]:
    state_path = output_dir / "state.json"
    if not state_path.exists():
        return _new_state(scope)
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MassiveNewsError(f"Could not read resume state at {state_path}") from exc
    if not isinstance(state, dict):
        raise MassiveNewsError("Resume state must be a JSON object")
    if state.get("schema_version") != STATE_SCHEMA_VERSION:
        raise MassiveNewsError("Resume state schema version is incompatible")
    if state.get("scope") != dict(scope):
        raise MassiveNewsError(
            "Existing resume state has a different collection scope; "
            "choose another output directory"
        )
    _validate_cached_pages(output_dir, state)
    return state


def build_manifest(
    state: Mapping[str, Any], *, calls_per_minute: float
) -> dict[str, Any]:
    pages = list(state.get("pages", []))
    completed = list(state.get("completed_tickers", []))
    scope = dict(state["scope"])
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": state["status"],
        "provider": "Massive",
        "dataset": "ordinary_news",
        "endpoint": API_URL,
        "scope": scope,
        "pagination": "cursor",
        "rate_policy": {
            "calls_per_minute": calls_per_minute,
            "minimum_interval_seconds": 60.0 / calls_per_minute,
        },
        "completed_tickers": completed,
        "pages": pages,
        "totals": {
            "configured_tickers": len(scope["tickers"]),
            "completed_tickers": len(completed),
            "pages": len(pages),
            "results": sum(int(page["result_count"]) for page in pages),
        },
        "credential_transport": "Authorization header",
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


def collect_news(
    *,
    client: MassiveClient,
    output_dir: Path,
    tickers: Sequence[str],
    start: date,
    end: date,
    max_requests: int,
    calls_per_minute: float = DEFAULT_CALLS_PER_MINUTE,
) -> dict[str, Any]:
    if start > end:
        raise ValueError("start must be on or before end")
    if max_requests < 0:
        raise ValueError("max_requests must be nonnegative")
    normalized_tickers = parse_tickers(",".join(tickers))
    scope = collection_scope(tickers=normalized_tickers, start=start, end=end)
    state = load_or_initialize_state(output_dir, scope)
    _persist_state_and_manifest(
        output_dir,
        state,
        calls_per_minute=calls_per_minute,
        secret=client.api_key,
    )

    requests_made = 0
    while (
        state["status"] != "complete"
        and requests_made < max_requests
        and int(state["ticker_index"]) < len(normalized_tickers)
    ):
        ticker_index = int(state["ticker_index"])
        ticker = normalized_tickers[ticker_index]
        page_number = int(state["next_page_number"])
        request_cursor = state.get("next_cursor")
        raw, payload = client.fetch_page(
            ticker=ticker,
            start=start,
            end=end,
            cursor=str(request_cursor) if request_cursor else None,
        )
        requests_made += 1
        next_cursor = extract_cursor(payload.get("next_url"))
        if next_cursor == client.api_key:
            raise MassiveNewsError("Massive returned the credential as a cursor")

        page_path = (
            output_dir
            / "pages"
            / ticker
            / f"page_{page_number:06d}.json.gz"
        )
        compressed = deterministic_gzip(raw)
        write_bytes_atomic(page_path, compressed)
        relative_path = page_path.relative_to(output_dir).as_posix()
        results = payload["results"]
        page_record = {
            "ticker": ticker,
            "page_number": page_number,
            "path": relative_path,
            "gzip_sha256": sha256_bytes(compressed),
            "json_sha256": sha256_bytes(raw),
            "result_count": len(results),
            "request_cursor_sha256": (
                sha256_bytes(str(request_cursor).encode("utf-8"))
                if request_cursor
                else None
            ),
            "request_id": payload.get("request_id"),
            "has_next_page": next_cursor is not None,
        }
        assert_secret_absent(
            page_record, client.api_key, destination="page audit metadata"
        )
        state["pages"].append(page_record)

        if next_cursor is None:
            state["completed_tickers"].append(ticker)
            state["ticker_index"] = ticker_index + 1
            state["next_cursor"] = None
            state["next_page_number"] = 1
        else:
            state["next_cursor"] = next_cursor
            state["next_page_number"] = page_number + 1
        if int(state["ticker_index"]) >= len(normalized_tickers):
            state["status"] = "complete"
        _persist_state_and_manifest(
            output_dir,
            state,
            calls_per_minute=calls_per_minute,
            secret=client.api_key,
        )

    return {
        "status": state["status"],
        "requests_made": requests_made,
        "pages_cached": len(state["pages"]),
        "results_cached": sum(
            int(page["result_count"]) for page in state["pages"]
        ),
        "completed_tickers": len(state["completed_tickers"]),
        "configured_tickers": len(normalized_tickers),
        "stop_reason": (
            "complete" if state["status"] == "complete" else "max_requests"
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
    parser.add_argument("--max-requests", type=int, default=DEFAULT_MAX_REQUESTS)
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
    if args.max_requests < 0:
        raise ValueError("max-requests must be nonnegative")
    if args.calls_per_minute <= 0:
        raise ValueError("calls-per-minute must be positive")
    tickers = args.tickers or load_stock_tickers(args.config)
    plan = {
        "mode": "dry-run" if args.dry_run else "download",
        "provider": "Massive",
        "endpoint": API_URL,
        "tickers": tickers,
        "ticker_count": len(tickers),
        "start": args.start.isoformat(),
        "end": args.end.isoformat(),
        "max_requests": args.max_requests,
        "calls_per_minute": args.calls_per_minute,
        "output_dir": str(args.output_dir),
        "resume_state": str(args.output_dir / "state.json"),
    }
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return 0

    api_key = load_api_key(args.env_file)
    client = MassiveClient(
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
        max_requests=args.max_requests,
        calls_per_minute=args.calls_per_minute,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (MassiveNewsError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
