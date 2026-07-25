"""Build the 22 feasible Bollerslev-Li-Tang correlation features.

This is a daily stock-sector adaptation of the feature construction in
"Forecasting and Managing Correlation Risks" (Management Science, 2026).
It builds:

* 3 HAR realized correlations and 3 negative semicorrelations;
* 4 finite-window exponentially weighted correlations and 4 negative
  semicorrelations; and
* 4 sector-state exponential correlations and 4 negative sector-state
  semicorrelations.

The paper's three characteristic-projection factor features are deliberately
omitted. They require a full point-in-time characteristic panel and are not
equivalent to ordinary rolling factor regressions.

The script is offline. It snapshots completed regular-session Alpaca chunks,
reads them without modification, and writes a separate Parquet, CSV, or
compressed CSV feature file. For forecast session t, every feature is computed
through session t-1.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, UTC
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_CONFIG = Path("config/price_universe.json")
DEFAULT_REGULAR_DIR = Path("data/prices/alpaca/15min/sip/all")
DEFAULT_CALENDAR = Path(
    "data/prices/alpaca/calendar/2016-01-01_2026-06-30.json"
)
DEFAULT_OUTPUT = Path("data/features/quant/bollerslev_core_features.parquet")

HAR_HORIZONS: Mapping[str, int] = {"d": 1, "w": 5, "m": 21}
EXPONENTIAL_CENTERS: Mapping[str, int] = {
    "d": 1,
    "w": 5,
    "m": 21,
    "q": 63,
}
DEFAULT_EXPONENTIAL_WINDOW = 500
SCHEMA_VERSION = 1

PAIR_COMPONENT_COLUMNS = [
    "realized_covariance",
    "left_realized_variance",
    "right_realized_variance",
    "negative_realized_covariance",
    "left_negative_realized_variance",
    "right_negative_realized_variance",
]

PAIR_EXPONENTIAL_COLUMNS = [
    *(f"exp_rc_{label}" for label in EXPONENTIAL_CENTERS),
    *(f"exp_rc_negative_{label}" for label in EXPONENTIAL_CENTERS),
]

FEATURE_COLUMNS = [
    *(f"rc_{label}" for label in HAR_HORIZONS),
    *(f"rc_negative_{label}" for label in HAR_HORIZONS),
    *PAIR_EXPONENTIAL_COLUMNS,
    *(f"sector_exp_rc_{label}" for label in EXPONENTIAL_CENTERS),
    *(f"sector_exp_rc_negative_{label}" for label in EXPONENTIAL_CENTERS),
]


@dataclass(frozen=True)
class Sector:
    name: str
    benchmark: str
    stocks: tuple[str, ...]


@dataclass(frozen=True)
class InputChunk:
    csv_path: Path
    manifest_path: Path
    csv_sha256: str
    row_count: int


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid ISO date: {value}") from exc


def parse_symbols(value: str) -> list[str]:
    symbols = [part.strip().upper() for part in value.split(",") if part.strip()]
    if not symbols:
        raise argparse.ArgumentTypeError("At least one symbol is required")
    return list(dict.fromkeys(symbols))


def load_sectors(path: Path) -> list[Sector]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sectors = [
        Sector(
            name=str(item["name"]),
            benchmark=str(item["benchmark"]).upper(),
            stocks=tuple(str(symbol).upper() for symbol in item["stocks"]),
        )
        for item in payload["sectors"]
    ]
    stocks = [stock for sector in sectors for stock in sector.stocks]
    if len(stocks) != len(set(stocks)):
        raise ValueError("Each configured stock must belong to exactly one sector")
    return sectors


def manifest_path_for_csv(csv_path: Path) -> Path:
    return csv_path.with_suffix("").with_suffix(".manifest.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_calendar_records(
    path: Path,
) -> tuple[tuple[str, str, str], ...]:
    """Load the date/open/close identity of an official market calendar."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise ValueError(f"Calendar is not complete: {path}")
    records_list: list[tuple[str, str, str]] = []
    for item in payload["sessions"]:
        session_date = str(item["date"])
        open_time = str(item["open"])
        close_time = str(item["close"])
        try:
            parsed_date = date.fromisoformat(session_date)
            parsed_open = datetime.strptime(open_time, "%H:%M")
            parsed_close = datetime.strptime(close_time, "%H:%M")
        except ValueError as exc:
            raise ValueError(
                f"Calendar contains a malformed session: {path}"
            ) from exc
        if parsed_date.isoformat() != session_date:
            raise ValueError(f"Calendar date is not canonical ISO: {path}")
        if parsed_open >= parsed_close:
            raise ValueError(f"Calendar open must precede close: {path}")
        records_list.append((session_date, open_time, close_time))
    records = tuple(records_list)
    dates = [item[0] for item in records]
    if len(dates) != len(set(dates)) or dates != sorted(dates):
        raise ValueError(f"Calendar sessions must be unique and sorted: {path}")
    return records


def expected_interval_keys(
    records: Sequence[tuple[str, str, str]],
    *,
    include_overnight: bool,
    expected_minutes: int = 15,
) -> dict[pd.Timestamp, frozenset[str]]:
    """Return the official return-interval keys for every market session.

    Bar labels denote interval starts.  The first regular-session return is
    open-to-first-close and later returns are close-to-close.  When requested,
    ``overnight`` is an additional close-to-open interval.
    """

    if expected_minutes <= 0:
        raise ValueError("expected_minutes must be positive")
    output: dict[pd.Timestamp, frozenset[str]] = {}
    for session_date, open_time, close_time in records:
        start = pd.Timestamp(f"{session_date} {open_time}")
        close = pd.Timestamp(f"{session_date} {close_time}")
        labels = {
            timestamp.strftime("%H:%M:%S")
            for timestamp in pd.date_range(
                start, close, freq=f"{expected_minutes}min", inclusive="left"
            )
        }
        if include_overnight:
            labels.add("overnight")
        output[pd.Timestamp(session_date).normalize()] = frozenset(labels)
    return output


def calendar_records_in_range(
    records: Sequence[tuple[str, str, str]],
    start: str,
    end: str,
) -> tuple[tuple[str, str, str], ...]:
    try:
        start_date = date.fromisoformat(start)
        end_date = date.fromisoformat(end)
    except ValueError as exc:
        raise ValueError("Chunk bounds must be canonical ISO dates") from exc
    if (
        start_date.isoformat() != start
        or end_date.isoformat() != end
        or start_date > end_date
    ):
        raise ValueError("Chunk bounds must be ordered canonical ISO dates")
    selected = tuple(item for item in records if start <= item[0] <= end)
    if not selected:
        raise ValueError(
            f"Calendar has no sessions in chunk range {start} through {end}"
        )
    return selected


def snapshot_completed_chunks(
    base: Path,
    symbols: Iterable[str],
    *,
    verify_hashes: bool = False,
    build_calendar_path: Path | None = None,
    project_root: Path | None = None,
) -> dict[str, tuple[InputChunk, ...]]:
    """Snapshot completed, contract-compatible RTH chunks once at startup."""

    root = (project_root or Path.cwd()).resolve()
    calendar_cache: dict[
        Path, tuple[str, tuple[tuple[str, str, str], ...]]
    ] = {}

    def cached_calendar(
        path: Path,
    ) -> tuple[str, tuple[tuple[str, str, str], ...]]:
        resolved = path if path.is_absolute() else root / path
        resolved = resolved.resolve()
        if resolved not in calendar_cache:
            if not resolved.exists():
                raise FileNotFoundError(
                    f"Calendar referenced by an input manifest is missing: "
                    f"{resolved}"
                )
            calendar_cache[resolved] = (
                sha256_file(resolved),
                load_calendar_records(resolved),
            )
        return calendar_cache[resolved]

    build_calendar_records: tuple[tuple[str, str, str], ...] | None = None
    if build_calendar_path is not None:
        _, build_calendar_records = cached_calendar(build_calendar_path)

    snapshot: dict[str, tuple[InputChunk, ...]] = {}
    for symbol in sorted(set(symbols)):
        chunks: list[InputChunk] = []
        for csv_path in sorted((base / symbol).glob("*.csv.gz")):
            manifest_path = manifest_path_for_csv(csv_path)
            if not manifest_path.exists():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            request = manifest.get("request", {})
            compatible = (
                manifest.get("status") == "complete"
                and manifest.get("download_schema_version") == 2
                and manifest.get("symbol") == symbol
                and request.get("timeframe") == "15Min"
                and request.get("feed") == "sip"
                and request.get("adjustment") == "all"
                and request.get("regular_session_only") is True
            )
            if not compatible:
                continue
            if build_calendar_records is not None:
                chunk = manifest.get("chunk", {})
                chunk_start = str(chunk.get("start", ""))
                chunk_end = str(chunk.get("end", ""))
                calendar_metadata = manifest.get("market_calendar", {})
                recorded_path = str(calendar_metadata.get("path", ""))
                recorded_hash = str(calendar_metadata.get("sha256", ""))
                if not (
                    chunk_start
                    and chunk_end
                    and recorded_path
                    and recorded_hash
                ):
                    raise ValueError(
                        f"Input manifest lacks auditable calendar metadata: "
                        f"{manifest_path}"
                    )
                actual_hash, source_calendar_records = cached_calendar(
                    Path(recorded_path)
                )
                if actual_hash != recorded_hash:
                    raise ValueError(
                        f"Input calendar hash mismatch: {manifest_path}"
                    )
                source_slice = calendar_records_in_range(
                    source_calendar_records, chunk_start, chunk_end
                )
                build_slice = calendar_records_in_range(
                    build_calendar_records, chunk_start, chunk_end
                )
                if source_slice != build_slice:
                    raise ValueError(
                        "Input and build calendars disagree over chunk range: "
                        f"{manifest_path}"
                    )
            expected_hash = str(manifest.get("csv_sha256", ""))
            if not expected_hash:
                continue
            if verify_hashes and sha256_file(csv_path) != expected_hash:
                raise ValueError(f"Input hash mismatch: {csv_path}")
            chunks.append(
                InputChunk(
                    csv_path=csv_path,
                    manifest_path=manifest_path,
                    csv_sha256=expected_hash,
                    row_count=int(manifest.get("row_count", 0)),
                )
            )
        if not chunks:
            raise FileNotFoundError(
                f"No complete compatible regular-session chunks for {symbol}"
            )
        snapshot[symbol] = tuple(chunks)
    return snapshot


def read_symbol_bars(
    chunks: Sequence[InputChunk],
    symbol: str,
    *,
    end: date | None = None,
) -> pd.DataFrame:
    frames = [
        pd.read_csv(
            chunk.csv_path,
            usecols=[
                "symbol",
                "timestamp_utc",
                "trade_date",
                "bar_start_et",
                "open",
                "close",
            ],
        )
        for chunk in chunks
    ]
    bars = pd.concat(frames, ignore_index=True)
    bars = bars[bars["symbol"].eq(symbol)].copy()
    bars["trade_date"] = pd.to_datetime(bars["trade_date"]).dt.normalize()
    bars["timestamp_utc"] = pd.to_datetime(bars["timestamp_utc"], utc=True)
    if end is not None:
        bars = bars[bars["trade_date"] <= pd.Timestamp(end)]
    bars = bars.sort_values(["trade_date", "timestamp_utc"]).reset_index(drop=True)
    if bars.empty:
        raise ValueError(f"No bars remain for {symbol}")
    if bars.duplicated(["symbol", "timestamp_utc"]).any():
        raise ValueError(f"Duplicate symbol/timestamp bars found for {symbol}")
    prices = bars[["open", "close"]].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError(f"Nonpositive or nonfinite prices found for {symbol}")
    return bars


def build_interval_returns(
    bars: pd.DataFrame,
    *,
    expected_minutes: int = 15,
    include_overnight: bool = True,
    official_sessions: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Reconstruct synchronized full-day return intervals for one symbol.

    Each session has one close-to-open overnight return, a first-bar
    open-to-close return, and subsequent 15-minute close-to-close returns.
    A missing bar never creates a mismatched 30-minute return.
    """

    if bars.empty:
        return pd.DataFrame(
            columns=["symbol", "trade_date", "interval_key", "log_return"]
        )
    symbols = bars["symbol"].unique()
    if len(symbols) != 1:
        raise ValueError("build_interval_returns expects exactly one symbol")
    symbol = str(symbols[0])
    rows: list[dict[str, object]] = []
    previous_session_close: float | None = None
    previous_observed_date: pd.Timestamp | None = None
    predecessor: dict[pd.Timestamp, pd.Timestamp] | None = None
    if official_sessions is not None:
        normalized_sessions = pd.DatetimeIndex(official_sessions).normalize()
        predecessor = {
            normalized_sessions[position]: normalized_sessions[position - 1]
            for position in range(1, len(normalized_sessions))
        }

    for trade_date, group in bars.groupby("trade_date", sort=True):
        trade_date = pd.Timestamp(trade_date).normalize()
        group = group.sort_values("timestamp_utc")
        opens = group["open"].to_numpy(dtype=float)
        closes = group["close"].to_numpy(dtype=float)
        timestamps = pd.DatetimeIndex(group["timestamp_utc"])
        interval_labels = group["bar_start_et"].astype(str).to_numpy()

        immediately_preceded = (
            previous_session_close is not None
            and (
                predecessor is None
                or predecessor.get(trade_date) == previous_observed_date
            )
        )
        if include_overnight and immediately_preceded:
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": trade_date,
                    "interval_key": "overnight",
                    "log_return": math.log(opens[0] / previous_session_close),
                }
            )

        # A faithful full-day component cannot be formed for the first
        # observed session. Keep its close only as the next overnight anchor.
        if include_overnight and previous_session_close is None:
            previous_session_close = float(closes[-1])
            previous_observed_date = trade_date
            continue

        rows.append(
            {
                "symbol": symbol,
                "trade_date": trade_date,
                "interval_key": str(interval_labels[0]),
                "log_return": math.log(closes[0] / opens[0]),
            }
        )
        if len(group) > 1:
            gaps = np.asarray(
                (timestamps[1:] - timestamps[:-1])
                / pd.Timedelta(minutes=1),
                dtype=float,
            )
            log_closes = np.log(closes)
            close_returns = np.diff(log_closes)
            for position, (gap, value) in enumerate(
                zip(gaps, close_returns, strict=True), start=1
            ):
                if not np.isclose(gap, expected_minutes):
                    continue
                rows.append(
                    {
                        "symbol": symbol,
                        "trade_date": trade_date,
                        "interval_key": str(interval_labels[position]),
                        "log_return": float(value),
                    }
                )
        previous_session_close = float(closes[-1])
        previous_observed_date = trade_date

    output = pd.DataFrame(rows)
    output["trade_date"] = pd.to_datetime(output["trade_date"]).dt.normalize()
    if output.duplicated(["trade_date", "interval_key"]).any():
        raise ValueError(f"Duplicate return intervals found for {symbol}")
    # Rows were appended in economic interval order: overnight first, then
    # consecutive regular-session intervals. Preserve that order for audits.
    return output.reset_index(drop=True)


def pair_daily_components(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    min_aligned_returns: int = 15,
    min_alignment_ratio: float = 0.8,
    require_overnight: bool = True,
    expected_keys: Mapping[pd.Timestamp, frozenset[str]] | None = None,
    require_complete_schedule: bool = False,
) -> pd.DataFrame:
    """Calculate pairwise daily covariance and (negative) variance components."""

    if not 0 < min_alignment_ratio <= 1:
        raise ValueError("min_alignment_ratio must be in (0, 1]")
    left_frame = left[["trade_date", "interval_key", "log_return"]].rename(
        columns={"log_return": "left_return"}
    )
    right_frame = right[["trade_date", "interval_key", "log_return"]].rename(
        columns={"log_return": "right_return"}
    )
    aligned = left_frame.merge(
        right_frame,
        on=["trade_date", "interval_key"],
        how="inner",
        validate="one_to_one",
    )
    if aligned.empty:
        return pd.DataFrame(columns=["trade_date", *PAIR_COMPONENT_COLUMNS])

    x = aligned["left_return"].to_numpy(dtype=float)
    y = aligned["right_return"].to_numpy(dtype=float)
    both_negative = (x < 0) & (y < 0)
    aligned["xy"] = x * y
    aligned["x2"] = x * x
    aligned["y2"] = y * y
    aligned["negative_xy"] = np.where(both_negative, x * y, 0.0)
    aligned["negative_x2"] = np.where(x < 0, x * x, 0.0)
    aligned["negative_y2"] = np.where(y < 0, y * y, 0.0)
    aligned["is_overnight"] = aligned["interval_key"].eq("overnight")

    components = (
        aligned.groupby("trade_date", as_index=False)
        .agg(
            realized_covariance=("xy", "sum"),
            left_realized_variance=("x2", "sum"),
            right_realized_variance=("y2", "sum"),
            negative_realized_covariance=("negative_xy", "sum"),
            left_negative_realized_variance=("negative_x2", "sum"),
            right_negative_realized_variance=("negative_y2", "sum"),
            aligned_return_count=("xy", "size"),
            aligned_overnight_count=("is_overnight", "sum"),
        )
        .sort_values("trade_date")
    )
    if expected_keys is None:
        left_count = left.groupby("trade_date")["log_return"].size()
        right_count = right.groupby("trade_date")["log_return"].size()
        available = pd.concat(
            [left_count.rename("left_count"), right_count.rename("right_count")],
            axis=1,
            sort=False,
        )
        available["expected_return_count"] = available[
            ["left_count", "right_count"]
        ].max(axis=1)
        components = components.merge(
            available[["expected_return_count"]],
            left_on="trade_date",
            right_index=True,
            how="left",
        )
    else:
        components["expected_return_count"] = components["trade_date"].map(
            lambda value: len(expected_keys.get(pd.Timestamp(value), ()))
        )
        aligned_key_sets = aligned.groupby("trade_date")["interval_key"].agg(
            lambda values: frozenset(values)
        )
        components["complete_official_schedule"] = components["trade_date"].map(
            lambda value: aligned_key_sets.get(
                pd.Timestamp(value), frozenset()
            )
            == expected_keys.get(pd.Timestamp(value), frozenset())
        )
    components["alignment_ratio"] = (
        components["aligned_return_count"]
        / components["expected_return_count"].replace(0, np.nan)
    )
    invalid = (
        components["aligned_return_count"].lt(min_aligned_returns)
        | components["alignment_ratio"].lt(min_alignment_ratio)
    )
    if require_complete_schedule:
        if expected_keys is None:
            raise ValueError(
                "require_complete_schedule requires official expected_keys"
            )
        invalid |= ~components["complete_official_schedule"]
    if require_overnight:
        invalid |= components["aligned_overnight_count"].ne(1)
    components.loc[invalid, PAIR_COMPONENT_COLUMNS] = np.nan
    return components


def component_correlation(
    covariance: pd.Series | np.ndarray,
    left_variance: pd.Series | np.ndarray,
    right_variance: pd.Series | np.ndarray,
) -> np.ndarray:
    covariance_array = np.asarray(covariance, dtype=float)
    left_array = np.asarray(left_variance, dtype=float)
    right_array = np.asarray(right_variance, dtype=float)
    denominator = np.sqrt(left_array * right_array)
    result = np.full(covariance_array.shape, np.nan, dtype=float)
    valid = (
        np.isfinite(covariance_array)
        & np.isfinite(denominator)
        & (denominator > 0)
    )
    result[valid] = covariance_array[valid] / denominator[valid]
    return np.clip(result, -1.0, 1.0)


def reindex_components(
    components: pd.DataFrame, sessions: pd.DatetimeIndex
) -> pd.DataFrame:
    frame = components.set_index("trade_date").reindex(sessions)
    frame.index.name = "asof_session"
    return frame


def build_har_features(
    components: pd.DataFrame,
    *,
    horizons: Mapping[str, int] = HAR_HORIZONS,
) -> pd.DataFrame:
    output = pd.DataFrame(index=components.index)
    for label, window in horizons.items():
        aggregate = components[PAIR_COMPONENT_COLUMNS].rolling(
            window, min_periods=window
        ).sum()
        output[f"rc_{label}"] = component_correlation(
            aggregate["realized_covariance"],
            aggregate["left_realized_variance"],
            aggregate["right_realized_variance"],
        )
        output[f"rc_negative_{label}"] = component_correlation(
            aggregate["negative_realized_covariance"],
            aggregate["left_negative_realized_variance"],
            aggregate["right_negative_realized_variance"],
        )
    return output


def center_of_mass_decay(center_of_mass: int | float) -> float:
    """Return exp(-lambda) implied by the paper's CoM equation."""

    if center_of_mass <= 0:
        raise ValueError("center_of_mass must be positive")
    return float(center_of_mass / (center_of_mass + 1.0))


def finite_exponential_average(
    values: pd.Series | np.ndarray,
    *,
    center_of_mass: int | float,
    window: int = DEFAULT_EXPONENTIAL_WINDOW,
    min_valid: int | None = None,
    minimum_weight_fraction: float = 1.0,
    require_current: bool = True,
) -> np.ndarray:
    """Finite-window EW average with weights proportional to exp(-k lambda).

    The most recent observation receives weight 1, the preceding observation
    q, and the oldest q**(window-1), where q=h/(h+1). This is algebraically
    identical to the paper's k=1..window normalization.
    """

    if window < 1:
        raise ValueError("window must be positive")
    required = window if min_valid is None else min_valid
    if not 1 <= required <= window:
        raise ValueError("min_valid must be between 1 and window")
    if not 0 < minimum_weight_fraction <= 1:
        raise ValueError("minimum_weight_fraction must be in (0, 1]")
    array = np.asarray(values, dtype=float)
    valid = np.isfinite(array)
    filled = np.where(valid, array, 0.0)
    q = center_of_mass_decay(center_of_mass)
    q_window = q**window
    full_denominator = (1.0 - q_window) / (1.0 - q)
    numerator = 0.0
    denominator = 0.0
    valid_count = 0
    output = np.full(len(array), np.nan, dtype=float)

    for index in range(len(array)):
        numerator = filled[index] + q * numerator
        denominator = float(valid[index]) + q * denominator
        valid_count += int(valid[index])
        if index >= window:
            numerator -= q_window * filled[index - window]
            denominator -= q_window * float(valid[index - window])
            valid_count -= int(valid[index - window])
        weight_fraction = denominator / full_denominator
        if (
            index >= window - 1
            and valid_count >= required
            and denominator > 0
            and weight_fraction >= minimum_weight_fraction - 1e-12
            and (valid[index] or not require_current)
        ):
            output[index] = numerator / denominator
    return output


def build_exponential_features(
    components: pd.DataFrame,
    *,
    centers: Mapping[str, int] = EXPONENTIAL_CENTERS,
    window: int = DEFAULT_EXPONENTIAL_WINDOW,
    min_valid: int | None = None,
    minimum_weight_fraction: float = 1.0,
) -> pd.DataFrame:
    output = pd.DataFrame(index=components.index)
    for label, center in centers.items():
        weighted = {
            column: finite_exponential_average(
                components[column],
                center_of_mass=center,
                window=window,
                min_valid=min_valid,
                minimum_weight_fraction=minimum_weight_fraction,
            )
            for column in PAIR_COMPONENT_COLUMNS
        }
        output[f"exp_rc_{label}"] = component_correlation(
            weighted["realized_covariance"],
            weighted["left_realized_variance"],
            weighted["right_realized_variance"],
        )
        output[f"exp_rc_negative_{label}"] = component_correlation(
            weighted["negative_realized_covariance"],
            weighted["left_negative_realized_variance"],
            weighted["right_negative_realized_variance"],
        )
    return output


def build_pair_features(
    left: pd.DataFrame,
    right: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    *,
    min_aligned_returns: int = 15,
    min_alignment_ratio: float = 0.8,
    require_overnight: bool = True,
    exponential_window: int = DEFAULT_EXPONENTIAL_WINDOW,
    min_exponential_valid: int | None = None,
    min_exponential_weight_fraction: float = 1.0,
    expected_keys: Mapping[pd.Timestamp, frozenset[str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = pair_daily_components(
        left,
        right,
        min_aligned_returns=min_aligned_returns,
        min_alignment_ratio=min_alignment_ratio,
        require_overnight=require_overnight,
        expected_keys=expected_keys,
    )
    daily = reindex_components(daily, sessions)
    features = pd.concat(
        [
            build_har_features(daily),
            build_exponential_features(
                daily,
                window=exponential_window,
                min_valid=min_exponential_valid,
                minimum_weight_fraction=min_exponential_weight_fraction,
            ),
        ],
        axis=1,
    )
    return features, daily


def build_sector_state(
    pair_feature_frames: Sequence[pd.DataFrame],
    *,
    minimum_pair_fraction: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average already-normalized exponential features across peer pairs."""

    if not pair_feature_frames:
        raise ValueError("At least one within-sector pair is required")
    if not 0 < minimum_pair_fraction <= 1:
        raise ValueError("minimum_pair_fraction must be in (0, 1]")
    values = pd.concat(
        [frame[PAIR_EXPONENTIAL_COLUMNS] for frame in pair_feature_frames],
        keys=range(len(pair_feature_frames)),
        names=["pair_id", "asof_session"],
    )
    means = values.groupby(level="asof_session").mean()
    counts = values.groupby(level="asof_session").count()
    required = math.ceil(len(pair_feature_frames) * minimum_pair_fraction)
    means = means.where(counts >= required)
    means = means.rename(
        columns={
            column: f"sector_{column}" for column in PAIR_EXPONENTIAL_COLUMNS
        }
    )
    coverage = counts / len(pair_feature_frames)
    coverage = coverage.rename(
        columns={
            column: f"diagnostic_sector_pair_coverage_{column}"
            for column in PAIR_EXPONENTIAL_COLUMNS
        }
    )
    return means, coverage


def load_official_sessions(path: Path) -> pd.DatetimeIndex:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise ValueError(f"Calendar is not complete: {path}")
    sessions = pd.DatetimeIndex(
        pd.to_datetime([item["date"] for item in payload["sessions"]])
    ).normalize()
    if sessions.has_duplicates or not sessions.is_monotonic_increasing:
        raise ValueError("Official calendar sessions must be unique and sorted")
    return sessions


def build_core_panel(
    returns: Mapping[str, pd.DataFrame],
    sectors: Sequence[Sector],
    sessions: pd.DatetimeIndex,
    *,
    output_stocks: Sequence[str] | None = None,
    min_aligned_returns: int = 15,
    min_alignment_ratio: float = 0.8,
    require_overnight: bool = True,
    exponential_window: int = DEFAULT_EXPONENTIAL_WINDOW,
    min_exponential_valid: int | None = None,
    min_exponential_weight_fraction: float = 1.0,
    minimum_sector_pair_fraction: float = 1.0,
    require_complete: bool = True,
    expected_keys: Mapping[pd.Timestamp, frozenset[str]] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    all_configured_stocks = {
        stock for sector in sectors for stock in sector.stocks
    }
    selected_stocks = (
        set(output_stocks) if output_stocks is not None else all_configured_stocks
    )
    unknown = selected_stocks - all_configured_stocks
    if unknown:
        raise ValueError(f"Stocks not present in configured sectors: {sorted(unknown)}")

    minimum_date = max(frame["trade_date"].min() for frame in returns.values())
    maximum_date = min(frame["trade_date"].max() for frame in returns.values())
    usable_sessions = sessions[
        (sessions >= minimum_date) & (sessions <= maximum_date)
    ]
    if len(usable_sessions) < exponential_window + 1:
        raise ValueError(
            f"Only {len(usable_sessions)} common sessions; need at least "
            f"{exponential_window + 1}"
        )

    pair_cache: dict[tuple[str, str], tuple[pd.DataFrame, pd.DataFrame]] = {}

    def pair_result(left_symbol: str, right_symbol: str):
        key = tuple(sorted((left_symbol, right_symbol)))
        if key not in pair_cache:
            pair_cache[key] = build_pair_features(
                returns[left_symbol],
                returns[right_symbol],
                usable_sessions,
                min_aligned_returns=min_aligned_returns,
                min_alignment_ratio=min_alignment_ratio,
                require_overnight=require_overnight,
                exponential_window=exponential_window,
                min_exponential_valid=min_exponential_valid,
                min_exponential_weight_fraction=min_exponential_weight_fraction,
                expected_keys=expected_keys,
            )
        return pair_cache[key]

    sector_states: dict[str, pd.DataFrame] = {}
    sector_coverages: dict[str, pd.DataFrame] = {}
    for sector in sectors:
        if not selected_stocks.intersection(sector.stocks):
            continue
        peer_frames = [
            pair_result(left, right)[0]
            for left, right in itertools.combinations(sector.stocks, 2)
        ]
        state, coverage = build_sector_state(
            peer_frames,
            minimum_pair_fraction=minimum_sector_pair_fraction,
        )
        sector_states[sector.name] = state
        sector_coverages[sector.name] = coverage

    next_session = pd.Series(
        usable_sessions[1:],
        index=usable_sessions[:-1],
        name="forecast_date",
    )
    panels: list[pd.DataFrame] = []
    daily_valid_fractions: list[float] = []

    for sector in sectors:
        for stock in sector.stocks:
            if stock not in selected_stocks:
                continue
            pair_features, daily = pair_result(stock, sector.benchmark)
            frame = pair_features.join(sector_states[sector.name], how="left")
            frame = frame.join(next_session, how="left")
            frame = frame.reset_index()
            frame.insert(0, "benchmark", sector.benchmark)
            frame.insert(0, "stock", stock)
            frame.insert(0, "sector", sector.name)
            panels.append(frame)
            daily_valid_fractions.append(
                float(daily["realized_covariance"].notna().mean())
            )

    panel = pd.concat(panels, ignore_index=True)
    panel = panel.dropna(subset=["forecast_date"])
    for column in FEATURE_COLUMNS:
        panel[column] = panel[column].clip(-1.0, 1.0)
    complete_mask = panel[FEATURE_COLUMNS].notna().all(axis=1)
    post_warmup = panel["asof_session"].ge(
        usable_sessions[exponential_window - 1]
    )
    prefilter_rows = len(panel)
    if require_complete:
        panel = panel[complete_mask].copy()
    panel = panel[
        ["sector", "stock", "benchmark", "asof_session", "forecast_date", *FEATURE_COLUMNS]
    ].sort_values(["forecast_date", "sector", "stock"])
    panel = panel.reset_index(drop=True)

    coverage_summary: dict[str, object] = {
        "rows_before_complete_case_filter": prefilter_rows,
        "complete_feature_row_fraction": float(complete_mask.mean()),
        "post_warmup_complete_feature_row_fraction": float(
            complete_mask[post_warmup].mean()
        ),
        "mean_primary_pair_valid_day_fraction": float(
            np.mean(daily_valid_fractions)
        ),
        "sector_pair_coverage_minimum": {
            sector_name: float(
                coverage.loc[
                    coverage.index >= usable_sessions[exponential_window - 1]
                ]
                .min(axis=1, skipna=True)
                .min(skipna=True)
            )
            for sector_name, coverage in sector_coverages.items()
        },
    }
    return panel, coverage_summary


def write_frame_atomic(path: Path, frame: pd.DataFrame) -> None:
    """Write Parquet, CSV, or gzip-compressed CSV through a sibling temp file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        lower_name = path.name.lower()
        if lower_name.endswith(".parquet"):
            try:
                frame.to_parquet(temporary, index=False)
            except ImportError as exc:
                raise ImportError(
                    "Parquet output requires pyarrow (declared in "
                    "requirements-quant.txt). Alternatively choose an output "
                    "ending in .csv or .csv.gz."
                ) from exc
        elif lower_name.endswith(".csv.gz"):
            frame.to_csv(temporary, index=False, compression="gzip")
        elif lower_name.endswith(".csv"):
            frame.to_csv(temporary, index=False)
        else:
            raise ValueError("Output must end in .parquet, .csv, or .csv.gz")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def output_manifest_path(output_path: Path) -> Path:
    if output_path.name.lower().endswith(".csv.gz"):
        return output_path.with_suffix("").with_suffix(".manifest.json")
    return output_path.with_suffix(".manifest.json")


def snapshot_digest(
    snapshot: Mapping[str, Sequence[InputChunk]], base: Path
) -> str:
    digest = hashlib.sha256()
    for symbol in sorted(snapshot):
        for chunk in snapshot[symbol]:
            try:
                relative = chunk.csv_path.relative_to(base)
            except ValueError:
                relative = chunk.csv_path
            digest.update(
                f"{relative.as_posix()}|{chunk.csv_sha256}\n".encode("utf-8")
            )
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--regular-dir", type=Path, default=DEFAULT_REGULAR_DIR)
    parser.add_argument("--calendar", type=Path, default=DEFAULT_CALENDAR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--stocks",
        type=parse_symbols,
        help="Optional comma-separated output stocks. Their full sectors are still loaded.",
    )
    parser.add_argument("--start", type=parse_date)
    parser.add_argument("--end", type=parse_date)
    parser.add_argument("--min-aligned-returns", type=int, default=15)
    parser.add_argument("--min-alignment-ratio", type=float, default=0.8)
    parser.add_argument(
        "--exponential-window",
        type=int,
        default=DEFAULT_EXPONENTIAL_WINDOW,
    )
    parser.add_argument(
        "--min-exponential-valid",
        type=int,
        help="Defaults to the full exponential window (faithful complete-window mode).",
    )
    parser.add_argument(
        "--min-exponential-weight-fraction",
        type=float,
        default=1.0,
        help="Minimum retained finite-window weight for every component.",
    )
    parser.add_argument("--minimum-sector-pair-fraction", type=float, default=1.0)
    parser.add_argument(
        "--exclude-overnight",
        action="store_true",
        help="RTH-only robustness mode. The paper-style default includes close-to-open.",
    )
    parser.add_argument("--keep-incomplete", action="store_true")
    parser.add_argument("--verify-input-hashes", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sectors = load_sectors(args.config)
    configured_stocks = {stock for sector in sectors for stock in sector.stocks}
    selected_stocks = set(args.stocks or configured_stocks)
    unknown = selected_stocks - configured_stocks
    if unknown:
        raise ValueError(f"Unknown configured stocks: {sorted(unknown)}")
    selected_sectors = [
        sector for sector in sectors if selected_stocks.intersection(sector.stocks)
    ]
    required_symbols = {
        symbol
        for sector in selected_sectors
        for symbol in (*sector.stocks, sector.benchmark)
    }

    snapshot = snapshot_completed_chunks(
        args.regular_dir,
        required_symbols,
        verify_hashes=args.verify_input_hashes,
        build_calendar_path=args.calendar,
    )
    plan = {
        "mode": "dry-run" if args.dry_run else "build",
        "selected_stocks": sorted(selected_stocks),
        "required_symbols": sorted(required_symbols),
        "input_chunk_count": sum(len(chunks) for chunks in snapshot.values()),
        "regular_dir": str(args.regular_dir),
        "calendar": str(args.calendar),
        "output": str(args.output),
        "feature_count": len(FEATURE_COLUMNS),
        "exponential_window": args.exponential_window,
        "min_exponential_valid": (
            args.min_exponential_valid or args.exponential_window
        ),
        "include_overnight": not args.exclude_overnight,
        "min_exponential_weight_fraction": (
            args.min_exponential_weight_fraction
        ),
    }
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return 0

    calendar_records = load_calendar_records(args.calendar)
    sessions = load_official_sessions(args.calendar)
    official_expected_keys = expected_interval_keys(
        calendar_records,
        include_overnight=not args.exclude_overnight,
    )
    returns: dict[str, pd.DataFrame] = {}
    for index, symbol in enumerate(sorted(required_symbols), start=1):
        bars = read_symbol_bars(snapshot[symbol], symbol, end=args.end)
        returns[symbol] = build_interval_returns(
            bars,
            include_overnight=not args.exclude_overnight,
            official_sessions=sessions,
        )
        print(
            f"[{index}/{len(required_symbols)}] {symbol}: "
            f"{len(bars):,} bars -> {len(returns[symbol]):,} intervals"
        )

    panel, coverage = build_core_panel(
        returns,
        selected_sectors,
        sessions,
        output_stocks=sorted(selected_stocks),
        min_aligned_returns=args.min_aligned_returns,
        min_alignment_ratio=args.min_alignment_ratio,
        require_overnight=not args.exclude_overnight,
        exponential_window=args.exponential_window,
        min_exponential_valid=args.min_exponential_valid,
        min_exponential_weight_fraction=(
            args.min_exponential_weight_fraction
        ),
        minimum_sector_pair_fraction=args.minimum_sector_pair_fraction,
        require_complete=not args.keep_incomplete,
        expected_keys=official_expected_keys,
    )
    if args.start is not None:
        panel = panel[panel["forecast_date"] >= pd.Timestamp(args.start)]
    if args.end is not None:
        panel = panel[panel["forecast_date"] <= pd.Timestamp(args.end)]
    if panel.empty:
        raise ValueError("No feature rows remain after filters")
    panel = panel.reset_index(drop=True)
    write_frame_atomic(args.output, panel)

    chunk_count = sum(len(chunks) for chunks in snapshot.values())
    input_rows = sum(
        chunk.row_count for chunks in snapshot.values() for chunk in chunks
    )
    manifest = {
        "status": "complete",
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source_method": {
            "paper": "Bollerslev, Li, and Tang (2026), Forecasting and Managing Correlation Risks",
            "doi": "10.1287/mnsc.2024.08294",
            "adaptation": "daily stock-to-sector-ETF pairs",
            "omitted_exact_features": ["FRCd", "FRCw", "FRCm"],
        },
        "feature_columns": FEATURE_COLUMNS,
        "feature_count": len(FEATURE_COLUMNS),
        "row_count": len(panel),
        "stock_count": int(panel["stock"].nunique()),
        "first_forecast_date": panel["forecast_date"].min().date().isoformat(),
        "last_forecast_date": panel["forecast_date"].max().date().isoformat(),
        "output_file": {
            "path": str(args.output),
            "sha256": sha256_file(args.output),
            "format": (
                "parquet"
                if args.output.name.lower().endswith(".parquet")
                else "csv.gz"
                if args.output.name.lower().endswith(".csv.gz")
                else "csv"
            ),
        },
        "parameters": {
            "har_horizons": dict(HAR_HORIZONS),
            "exponential_centers_of_mass": dict(EXPONENTIAL_CENTERS),
            "exponential_window": args.exponential_window,
            "min_exponential_valid": (
                args.min_exponential_valid or args.exponential_window
            ),
            "min_exponential_weight_fraction": (
                args.min_exponential_weight_fraction
            ),
            "min_aligned_returns": args.min_aligned_returns,
            "min_alignment_ratio": args.min_alignment_ratio,
            "alignment_denominator": "official expected interval count",
            "minimum_sector_pair_fraction": args.minimum_sector_pair_fraction,
            "include_overnight": not args.exclude_overnight,
            "require_complete_rows": not args.keep_incomplete,
            "feature_timing": "forecast t uses data through previous official session t-1",
        },
        "coverage": coverage,
        "input_snapshot": {
            "regular_dir": str(args.regular_dir),
            "chunk_count": chunk_count,
            "manifest_row_count": input_rows,
            "combined_path_and_manifest_hash": snapshot_digest(
                snapshot, args.regular_dir
            ),
            "actual_hashes_verified": args.verify_input_hashes,
            "calendar_path": str(args.calendar),
            "calendar_sha256": sha256_file(args.calendar),
            "input_calendar_sessions_validated": True,
        },
        "limitations": [
            "Alpaca trade-bar OHLC prices are used instead of TAQ midquotes.",
            "The configured fixed liquid-stock universe is not point-in-time constituent data.",
            "Sector-state features average the configured peer pairs and are attached to stock-ETF pairs.",
            "The three characteristic-projection factor features are intentionally omitted.",
        ],
    }
    manifest_path = output_manifest_path(args.output)
    write_json_atomic(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": "complete",
                "rows": len(panel),
                "stocks": int(panel["stock"].nunique()),
                "first_forecast_date": manifest["first_forecast_date"],
                "last_forecast_date": manifest["last_forecast_date"],
                "output": str(args.output),
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        ImportError,
        OSError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
