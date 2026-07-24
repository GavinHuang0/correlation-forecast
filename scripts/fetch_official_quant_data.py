"""Fetch official non-price inputs for the correlation feature pipeline.

The script is deliberately independent of Alpaca, so it can run while the
historical Alpaca bars backfill is active. It retrieves:

* VIX and 2/5/10-year Treasury series from FRED;
* daily U.S. Fama-French 3 factors and momentum from Ken French's library;
* BLS release-calendar events;
* BEA machine-readable release dates (currently 2025 onward);
* scheduled FOMC decision dates from Federal Reserve calendar/archive pages.

All outputs are written atomically under ``data/external/official-quant`` by
default. No API keys are required.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import date, datetime, time as wall_time, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo


DEFAULT_OUTPUT_DIR = Path("data/external/official-quant")
DEFAULT_START = date(2016, 1, 1)
DEFAULT_END = date(2026, 6, 30)
USER_AGENT = "stock-sector-correlation-research/1.0 (academic data fetcher)"
NEW_YORK = ZoneInfo("America/New_York")
UTC = timezone.utc
FRED_SERIES = {
    "VIXCLS": "CBOE Volatility Index: VIX",
    "DGS2": "Market Yield on U.S. Treasury Securities at 2-Year Constant Maturity",
    "DGS5": "Market Yield on U.S. Treasury Securities at 5-Year Constant Maturity",
    "DGS10": "Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity",
}
FRENCH_URLS = {
    "ff3": (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
        "F-F_Research_Data_Factors_daily_CSV.zip"
    ),
    "momentum": (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
        "F-F_Momentum_Factor_daily_CSV.zip"
    ),
}
BEA_JSON_URL = "https://apps.bea.gov/API/signup/release_dates.json"
BLS_YEAR_URL = "https://www.bls.gov/schedule/{year}/home.htm"
FOMC_CURRENT_URL = (
    "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
)
FOMC_HISTORICAL_URL = (
    "https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm"
)
SELECTED_BLS_RELEASES = (
    "Consumer Price Index",
    "Employment Cost Index",
    "Employment Situation",
    "Job Openings and Labor Turnover",
    "Producer Price Index",
    "Productivity and Costs",
)
SELECTED_BEA_RELEASES = {
    "Gross Domestic Product",
    "Personal Income and Outlays",
    "U.S. International Trade in Goods and Services",
}
MONTHS = {
    name: number
    for number, name in enumerate(
        (
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ),
        start=1,
    )
}
MONTH_ALIASES = {
    name[:3].lower(): number for name, number in MONTHS.items()
}


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected ISO date, received {value!r}") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_bytes(
    url: str, *, timeout: float = 60.0, retries: int = 4, pause: float = 0.2
) -> bytes:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "*/*"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read()
            if pause:
                time.sleep(pause)
            return body
        except urllib.error.HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504}:
                raise RuntimeError(f"HTTP {exc.code} from {url}") from exc
            last_error = exc
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
        if attempt < retries:
            time.sleep(min(10.0, 2**attempt))
    raise RuntimeError(f"Request failed after {retries + 1} attempts: {url}: {last_error}")


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


def write_csv_atomic(
    path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]
) -> None:
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
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
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


def fetch_fred_series(
    series_id: str,
    *,
    start: date,
    end: date,
    fetcher=fetch_bytes,
) -> tuple[list[dict[str, str]], str, bytes]:
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?" + urllib.parse.urlencode(
        {"id": series_id, "cosd": start.isoformat(), "coed": end.isoformat()}
    )
    raw = fetcher(url)
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    rows: list[dict[str, str]] = []
    date_column = next(
        (
            name
            for name in (reader.fieldnames or [])
            if name.lower() in {"date", "observation_date"}
        ),
        None,
    )
    if date_column is None:
        raise RuntimeError(
            f"FRED response has no date column for {series_id}: {reader.fieldnames}"
        )
    value_column = next(
        (name for name in (reader.fieldnames or []) if name != date_column), None
    )
    if value_column is None:
        raise RuntimeError(f"FRED response has no value column for {series_id}")
    for record in reader:
        observation_date = record.get(date_column, "")
        if not observation_date:
            continue
        rows.append(
            {
                "date": observation_date,
                "series_id": series_id,
                "value": "" if record.get(value_column) in {"", "."} else str(record[value_column]),
            }
        )
    if not rows:
        raise RuntimeError(f"FRED returned no observations for {series_id}")
    return rows, url, raw


def parse_french_zip(raw: bytes, expected_columns: set[str]) -> list[dict[str, str]]:
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise RuntimeError("Expected exactly one CSV inside Ken French archive")
        text = archive.read(csv_names[0]).decode("utf-8-sig", errors="replace")
    lines = text.splitlines()
    parsed_lines = list(csv.reader(lines))
    header_index = next(
        (
            index
            for index, cells in enumerate(parsed_lines)
            if expected_columns.issubset({cell.strip() for cell in cells})
        ),
        None,
    )
    if header_index is None:
        raise RuntimeError(
            f"Could not find expected Ken French columns: {sorted(expected_columns)}"
        )
    header = [cell.strip() for cell in parsed_lines[header_index]]
    column_indices = {
        column: header.index(column)
        for column in expected_columns
    }
    rows: list[dict[str, str]] = []
    for cells in parsed_lines[header_index + 1 :]:
        date_token = cells[0].strip() if cells else ""
        if not re.fullmatch(r"\d{8}", date_token):
            continue
        normalized = {"date": datetime.strptime(date_token, "%Y%m%d").date().isoformat()}
        for column, index in column_indices.items():
            clean_value = cells[index].strip() if index < len(cells) else ""
            if clean_value:
                normalized[column] = str(float(clean_value) / 100.0)
        rows.append(normalized)
    if not rows:
        raise RuntimeError("Ken French archive contained no daily factor rows")
    return rows


class TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._in_row = False
        self._in_cell = False
        self._cell_parts: list[str] = []
        self._row: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "tr":
            self._in_row = True
            self._row = []
        elif self._in_row and tag in {"td", "th"}:
            self._in_cell = True
            self._cell_parts = []

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._in_cell and tag in {"td", "th"}:
            value = " ".join(" ".join(self._cell_parts).split())
            self._row.append(value)
            self._in_cell = False
        elif self._in_row and tag == "tr":
            if self._row:
                self.rows.append(self._row)
            self._in_row = False


class HeadingParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.headings: list[tuple[str, str]] = []
        self._tag: str | None = None
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"h2", "h3", "h4", "h5", "h6"}:
            self._tag = tag
            self._parts = []

    def handle_data(self, data: str) -> None:
        if self._tag:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._tag == tag:
            text = " ".join(" ".join(self._parts).split())
            if text:
                self.headings.append((tag, text))
            self._tag = None


class CurrentFomcParser(HTMLParser):
    """Extract year/month/day cells from the current Fed calendar layout."""

    def __init__(self) -> None:
        super().__init__()
        self.current_year: int | None = None
        self._capture: str | None = None
        self._parts: list[str] = []
        self._pending_month: str | None = None
        self.meetings: list[tuple[int, str, str]] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        attributes = dict(attrs)
        classes = set(attributes.get("class", "").split())
        if tag == "h4":
            self._capture = "year"
            self._parts = []
        elif tag == "div" and "fomc-meeting__month" in classes:
            self._capture = "month"
            self._parts = []
        elif tag == "div" and "fomc-meeting__date" in classes:
            self._capture = "days"
            self._parts = []

    def handle_data(self, data: str) -> None:
        if self._capture:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._capture == "year" and tag == "h4":
            text = " ".join(" ".join(self._parts).split())
            match = re.search(r"\b(20\d{2})\b", text)
            if match:
                self.current_year = int(match.group(1))
            self._capture = None
        elif self._capture == "month" and tag == "div":
            month = " ".join(" ".join(self._parts).split())
            month_parts = [part.strip().lower()[:3] for part in month.split("/")]
            self._pending_month = (
                month if month_parts and all(part in MONTH_ALIASES for part in month_parts)
                else None
            )
            self._capture = None
        elif self._capture == "days" and tag == "div":
            days = " ".join(" ".join(self._parts).split())
            if self.current_year is not None and self._pending_month is not None:
                self.meetings.append(
                    (self.current_year, self._pending_month, days)
                )
            self._pending_month = None
            self._capture = None


def parse_bls_calendar(html: bytes, year: int) -> list[dict[str, str]]:
    parser = TableParser()
    parser.feed(html.decode("utf-8", errors="replace"))
    rows: list[dict[str, str]] = []
    for cells in parser.rows:
        if len(cells) < 3:
            continue
        date_text, time_text, release = cells[0], cells[1], cells[2]
        if not any(release.startswith(prefix) for prefix in SELECTED_BLS_RELEASES):
            continue
        try:
            release_date = datetime.strptime(date_text, "%A, %B %d, %Y").date()
            release_time = datetime.strptime(time_text, "%I:%M %p").time()
        except ValueError:
            continue
        if release_date.year != year:
            continue
        rows.append(
            {
                "source": "BLS",
                "event_name": release,
                "release_date": release_date.isoformat(),
                "release_time_et": release_time.isoformat(timespec="minutes"),
                "before_09_cutoff": str(release_time < wall_time(9)).lower(),
                "schedule_status": "archived_schedule",
                "source_url": BLS_YEAR_URL.format(year=year),
            }
        )
    return rows


def parse_bea_json(raw: bytes) -> list[dict[str, str]]:
    payload = json.loads(raw)
    rows: list[dict[str, str]] = []
    for event_name in sorted(SELECTED_BEA_RELEASES):
        record = payload.get(event_name, {})
        release_dates = record.get("release_dates", []) if isinstance(record, dict) else []
        for value in release_dates:
            instant = datetime.fromisoformat(str(value)).astimezone(NEW_YORK)
            rows.append(
                {
                    "source": "BEA",
                    "event_name": event_name,
                    "release_date": instant.date().isoformat(),
                    "release_time_et": instant.time().isoformat(timespec="minutes"),
                    "before_09_cutoff": str(instant.time() < wall_time(9)).lower(),
                    "schedule_status": "machine_readable_schedule",
                    "source_url": BEA_JSON_URL,
                }
            )
    unique = {
        (row["event_name"], row["release_date"], row["release_time_et"]): row
        for row in rows
    }
    return list(unique.values())


def parse_fomc_headings(html: bytes, fallback_year: int | None) -> list[dict[str, str]]:
    if fallback_year is None:
        current_parser = CurrentFomcParser()
        current_parser.feed(html.decode("utf-8", errors="replace"))
        current_rows: list[dict[str, str]] = []
        for year, month, days in current_parser.meetings:
            day_values = [int(value) for value in re.findall(r"\d{1,2}", days)]
            if not day_values:
                continue
            decision_month = MONTH_ALIASES[month.split("/")[-1].strip().lower()[:3]]
            try:
                decision_date = date(year, decision_month, day_values[-1])
            except ValueError:
                continue
            current_rows.append(
                {
                    "source": "Federal Reserve",
                    "event_name": "Scheduled FOMC policy decision",
                    "release_date": decision_date.isoformat(),
                    "release_time_et": "14:00",
                    "before_09_cutoff": "false",
                    "schedule_status": "scheduled_meeting",
                    "source_url": FOMC_CURRENT_URL,
                }
            )
        return list(
            {row["release_date"]: row for row in current_rows}.values()
        )

    parser = HeadingParser()
    parser.feed(html.decode("utf-8", errors="replace"))
    rows: list[dict[str, str]] = []
    current_year = fallback_year
    pattern = re.compile(
        r"(?P<month>"
        + "|".join(MONTHS)
        + r")\s+(?P<start>\d{1,2})(?:\s*[-–]\s*(?P<end>\d{1,2}))?"
    )
    for _, heading in parser.headings:
        year_match = re.search(r"\b(20\d{2})\b", heading)
        if year_match and (
            heading.strip().isdigit()
            or "FOMC Meetings" in heading
            or "Meeting" in heading
        ):
            current_year = int(year_match.group(1))
        lower = heading.lower()
        if (
            "meeting" not in lower
            or "unscheduled" in lower
            or "cancelled" in lower
            or "notation vote" in lower
        ):
            continue
        match = pattern.search(heading)
        if match is None:
            continue
        explicit_year = int(year_match.group(1)) if year_match else current_year
        if explicit_year is None:
            continue
        decision_day = int(match.group("end") or match.group("start"))
        try:
            decision_date = date(
                explicit_year, MONTHS[match.group("month")], decision_day
            )
        except ValueError:
            continue
        rows.append(
            {
                "source": "Federal Reserve",
                "event_name": "Scheduled FOMC policy decision",
                "release_date": decision_date.isoformat(),
                "release_time_et": "14:00",
                "before_09_cutoff": "false",
                "schedule_status": "scheduled_meeting",
                "source_url": (
                    FOMC_HISTORICAL_URL.format(year=explicit_year)
                    if explicit_year <= 2020
                    else FOMC_CURRENT_URL
                ),
            }
        )
    unique = {row["release_date"]: row for row in rows}
    return list(unique.values())


def merge_factor_rows(
    ff3_rows: Iterable[Mapping[str, str]],
    momentum_rows: Iterable[Mapping[str, str]],
    *,
    start: date,
    end: date,
) -> list[dict[str, str]]:
    by_date: dict[str, dict[str, str]] = {}
    for record in ff3_rows:
        by_date.setdefault(record["date"], {}).update(record)
    for record in momentum_rows:
        by_date.setdefault(record["date"], {}).update(record)
    output = []
    for day, record in sorted(by_date.items()):
        parsed_day = date.fromisoformat(day)
        if start <= parsed_day <= end:
            output.append(
                {
                    "date": day,
                    "mkt_rf": record.get("Mkt-RF", ""),
                    "smb": record.get("SMB", ""),
                    "hml": record.get("HML", ""),
                    "mom": record.get("Mom", ""),
                    "rf": record.get("RF", ""),
                }
            )
    return output


def fetch_all(
    *,
    output_dir: Path,
    start: date,
    end: date,
    timeout: float,
    retries: int,
    pause: float,
    fetcher=None,
) -> dict[str, Any]:
    if fetcher is None:
        fetcher = lambda url: fetch_bytes(  # noqa: E731
            url, timeout=timeout, retries=retries, pause=pause
        )
    source_records: list[dict[str, Any]] = []

    fred_rows: list[dict[str, str]] = []
    for series_id in FRED_SERIES:
        rows, url, raw = fetch_fred_series(
            series_id, start=start, end=end, fetcher=fetcher
        )
        raw_path = output_dir / "raw" / "fred" / f"{series_id}.csv"
        write_bytes_atomic(raw_path, raw)
        fred_rows.extend(rows)
        source_records.append(
            {
                "name": f"FRED {series_id}",
                "url": url,
                "raw_path": str(raw_path),
                "sha256": sha256_file(raw_path),
                "rows": len(rows),
            }
        )
    fred_path = output_dir / "fred_market_series.csv"
    write_csv_atomic(
        fred_path, fred_rows, ("date", "series_id", "value")
    )

    factor_parts: dict[str, list[dict[str, str]]] = {}
    for name, url in FRENCH_URLS.items():
        raw = fetcher(url)
        raw_path = output_dir / "raw" / "ken_french" / f"{name}.zip"
        write_bytes_atomic(raw_path, raw)
        expected = {"Mkt-RF", "SMB", "HML", "RF"} if name == "ff3" else {"Mom"}
        factor_parts[name] = parse_french_zip(raw, expected)
        source_records.append(
            {
                "name": f"Ken French {name}",
                "url": url,
                "raw_path": str(raw_path),
                "sha256": sha256_file(raw_path),
                "rows": len(factor_parts[name]),
            }
        )
    factors = merge_factor_rows(
        factor_parts["ff3"], factor_parts["momentum"], start=start, end=end
    )
    factors_path = output_dir / "fama_french_daily.csv"
    write_csv_atomic(
        factors_path,
        factors,
        ("date", "mkt_rf", "smb", "hml", "mom", "rf"),
    )

    macro_rows: list[dict[str, str]] = []
    for year in range(start.year, end.year + 1):
        url = BLS_YEAR_URL.format(year=year)
        raw = fetcher(url)
        raw_path = output_dir / "raw" / "bls" / f"{year}.html"
        write_bytes_atomic(raw_path, raw)
        parsed = parse_bls_calendar(raw, year)
        macro_rows.extend(parsed)
        source_records.append(
            {
                "name": f"BLS release schedule {year}",
                "url": url,
                "raw_path": str(raw_path),
                "sha256": sha256_file(raw_path),
                "rows": len(parsed),
            }
        )

    bea_raw = fetcher(BEA_JSON_URL)
    bea_path = output_dir / "raw" / "bea" / "release_dates.json"
    write_bytes_atomic(bea_path, bea_raw)
    bea_rows = parse_bea_json(bea_raw)
    macro_rows.extend(bea_rows)
    source_records.append(
        {
            "name": "BEA machine-readable release dates",
            "url": BEA_JSON_URL,
            "raw_path": str(bea_path),
            "sha256": sha256_file(bea_path),
            "rows": len(bea_rows),
            "coverage_note": "The live JSON currently begins in 2025.",
        }
    )

    for year in range(start.year, min(end.year, 2020) + 1):
        url = FOMC_HISTORICAL_URL.format(year=year)
        raw = fetcher(url)
        raw_path = output_dir / "raw" / "federal_reserve" / f"fomc_{year}.html"
        write_bytes_atomic(raw_path, raw)
        parsed = parse_fomc_headings(raw, year)
        macro_rows.extend(parsed)
        source_records.append(
            {
                "name": f"Federal Reserve FOMC archive {year}",
                "url": url,
                "raw_path": str(raw_path),
                "sha256": sha256_file(raw_path),
                "rows": len(parsed),
            }
        )
    if end.year >= 2021:
        raw = fetcher(FOMC_CURRENT_URL)
        raw_path = output_dir / "raw" / "federal_reserve" / "fomc_current.html"
        write_bytes_atomic(raw_path, raw)
        parsed = parse_fomc_headings(raw, None)
        parsed = [
            row
            for row in parsed
            if start <= date.fromisoformat(row["release_date"]) <= end
        ]
        macro_rows.extend(parsed)
        source_records.append(
            {
                "name": "Federal Reserve current FOMC calendars",
                "url": FOMC_CURRENT_URL,
                "raw_path": str(raw_path),
                "sha256": sha256_file(raw_path),
                "rows": len(parsed),
            }
        )

    macro_rows = [
        row
        for row in macro_rows
        if start <= date.fromisoformat(row["release_date"]) <= end
    ]
    macro_rows.sort(
        key=lambda row: (
            row["release_date"],
            row["release_time_et"],
            row["source"],
            row["event_name"],
        )
    )
    macro_path = output_dir / "macro_release_calendar.csv"
    macro_columns = (
        "source",
        "event_name",
        "release_date",
        "release_time_et",
        "before_09_cutoff",
        "schedule_status",
        "source_url",
    )
    write_csv_atomic(macro_path, macro_rows, macro_columns)

    manifest = {
        "status": "complete",
        "schema_version": 1,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "fetched_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "outputs": {
            "fred_market_series": {
                "path": str(fred_path),
                "sha256": sha256_file(fred_path),
                "rows": len(fred_rows),
            },
            "fama_french_daily": {
                "path": str(factors_path),
                "sha256": sha256_file(factors_path),
                "rows": len(factors),
                "units": "decimal returns",
                "revision_note": (
                    "Ken French's current research returns may contain historical "
                    "revisions and are not a point-in-time vintage archive."
                ),
            },
            "macro_release_calendar": {
                "path": str(macro_path),
                "sha256": sha256_file(macro_path),
                "rows": len(macro_rows),
                "bea_coverage_note": (
                    "BEA's machine-readable JSON currently covers 2025 onward; "
                    "BLS and scheduled FOMC events cover the requested history."
                ),
            },
        },
        "sources": source_records,
        "credential_values_recorded": False,
    }
    manifest_path = output_dir / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=parse_date, default=DEFAULT_START)
    parser.add_argument("--end", type=parse_date, default=DEFAULT_END)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--request-pause", type=float, default=0.2)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.start > args.end:
        raise ValueError("start must be on or before end")
    plan = {
        "mode": "dry-run" if args.dry_run else "download",
        "start": args.start.isoformat(),
        "end": args.end.isoformat(),
        "fred_series": list(FRED_SERIES),
        "ken_french_archives": list(FRENCH_URLS),
        "bls_years": list(range(args.start.year, args.end.year + 1)),
        "bea_machine_readable_schedule": True,
        "fomc_historical_years": list(
            range(args.start.year, min(args.end.year, 2020) + 1)
        ),
        "fomc_current_calendar": args.end.year >= 2021,
        "alpaca_requests": 0,
        "output_dir": str(args.output_dir),
    }
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return 0
    manifest = fetch_all(
        output_dir=args.output_dir,
        start=args.start,
        end=args.end,
        timeout=args.timeout,
        retries=args.retries,
        pause=args.request_pause,
    )
    print(json.dumps(manifest["outputs"], indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
