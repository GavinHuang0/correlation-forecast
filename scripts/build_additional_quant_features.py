"""Build leakage-aware additional quant features from completed raw downloads.

This script makes no network requests. It reads:

* regular Alpaca 15-minute chunks (read-only);
* the separate extended-hours supplement;
* official context files from ``fetch_official_quant_data.py``.

The output has one row per configured stock and forecast date. All features
that depend on a completed regular session are lagged. Current-date premarket
features use bars strictly before the configured cutoff.

Do not run this against the main regular-bars directory until the active
backfill is complete. Tests use isolated synthetic inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_CONFIG = Path("config/price_universe.json")
DEFAULT_REGULAR_DIR = Path("data/prices/alpaca/15min/sip/all")
DEFAULT_EXTENDED_DIR = Path("data/prices/alpaca-extended/15min/sip/all")
DEFAULT_OFFICIAL_DIR = Path("data/external/official-quant")
DEFAULT_CALENDAR = Path(
    "data/prices/alpaca/calendar/2016-01-01_2026-06-30.json"
)
DEFAULT_OUTPUT = Path("data/features/quant/additional_quant_features.parquet")


@dataclass(frozen=True)
class Pair:
    sector: str
    stock: str
    benchmark: str


def load_pairs(path: Path) -> list[Pair]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pairs = []
    for sector in payload["sectors"]:
        for stock in sector["stocks"]:
            pairs.append(
                Pair(
                    sector=str(sector["name"]),
                    stock=str(stock),
                    benchmark=str(sector["benchmark"]),
                )
            )
    return pairs


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def completed_csv_paths(
    base: Path, *, verify_hashes: bool = False
) -> list[Path]:
    """Return only files with a complete sibling manifest.

    The hash is not rechecked here because the source downloader already
    verifies it on resume; the builder snapshots this list at startup.
    """

    paths = []
    for path in base.rglob("*.csv.gz"):
        manifest_path = path.with_suffix("").with_suffix(".manifest.json")
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("status") == "complete":
            expected_hash = str(manifest.get("csv_sha256", ""))
            if verify_hashes:
                if not expected_hash:
                    raise ValueError(
                        f"Complete input manifest lacks csv_sha256: "
                        f"{manifest_path}"
                    )
                if sha256_file(path) != expected_hash:
                    raise ValueError(f"Input hash mismatch: {path}")
            paths.append(path)
    return sorted(paths)


def snapshot_digest(paths: Sequence[Path]) -> str:
    """Hash the ordered input paths and their manifest-declared hashes."""

    digest = hashlib.sha256()
    for path in sorted(paths):
        manifest_path = path.with_suffix("").with_suffix(".manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        digest.update(str(path).replace("\\", "/").encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(manifest.get("csv_sha256", "")).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def read_bar_files(paths: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(
            path,
            usecols=[
                "symbol",
                "timestamp_utc",
                "trade_date",
                "bar_start_et",
                "open",
                "close",
                "volume",
            ],
        )
        frames.append(frame)
    if not frames:
        return pd.DataFrame(
            columns=[
                "symbol",
                "timestamp_utc",
                "trade_date",
                "bar_start_et",
                "open",
                "close",
                "volume",
            ]
        )
    output = pd.concat(frames, ignore_index=True)
    output["trade_date"] = pd.to_datetime(output["trade_date"])
    output["timestamp_utc"] = pd.to_datetime(output["timestamp_utc"], utc=True)
    output = output.sort_values(["symbol", "timestamp_utc"])
    if output.duplicated(["symbol", "timestamp_utc"]).any():
        raise ValueError("Duplicate symbol/timestamp bars found")
    return output


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


def load_expected_regular_interval_keys(
    path: Path,
) -> dict[pd.Timestamp, frozenset[str]]:
    """Return the exact official 15-minute bar starts for every session."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise ValueError(f"Calendar is not complete: {path}")
    output: dict[pd.Timestamp, frozenset[str]] = {}
    for item in payload["sessions"]:
        session = pd.Timestamp(item["date"]).normalize()
        start = pd.Timestamp(f"{item['date']} {item['open']}")
        close = pd.Timestamp(f"{item['date']} {item['close']}")
        if start >= close:
            raise ValueError(f"Calendar open must precede close: {path}")
        keys = frozenset(
            timestamp.strftime("%H:%M:%S")
            for timestamp in pd.date_range(
                start, close, freq="15min", inclusive="left"
            )
        )
        if not keys:
            raise ValueError(f"Calendar session has no 15-minute bars: {session}")
        output[session] = keys
    if len(output) != len(payload["sessions"]):
        raise ValueError("Official calendar sessions must be unique")
    return output


def aggregate_regular_bars(
    bars: pd.DataFrame,
    official_sessions: pd.DatetimeIndex | None = None,
    expected_interval_keys: Mapping[pd.Timestamp, frozenset[str]] | None = None,
) -> pd.DataFrame:
    """Build schedule-aware daily regular-session features.

    The opening bar contributes ``log(close/open)``. Later returns are used
    only across exact 15-minute timestamp gaps, so a missing bar is never
    compressed into a spurious longer-horizon return.  Reindexing to the
    official session calendar also prevents lags and close returns from
    jumping across an entirely missing trading session.
    """

    if bars.empty:
        return pd.DataFrame()
    ordered = bars.sort_values(
        ["symbol", "trade_date", "timestamp_utc"]
    ).copy()
    prices = ordered[["open", "close"]].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Regular bars contain nonpositive or nonfinite prices")
    rows: list[dict[str, object]] = []
    for (symbol, trade_date), group in ordered.groupby(
        ["symbol", "trade_date"], sort=True
    ):
        group = group.sort_values("timestamp_utc")
        opens = group["open"].to_numpy(dtype=float)
        closes = group["close"].to_numpy(dtype=float)
        timestamps = pd.DatetimeIndex(group["timestamp_utc"])
        observed_keys = frozenset(group["bar_start_et"].astype(str))
        normalized_date = pd.Timestamp(trade_date).normalize()
        expected_keys = (
            expected_interval_keys.get(normalized_date)
            if expected_interval_keys is not None
            else None
        )
        complete_schedule = (
            expected_keys is None or observed_keys == expected_keys
        )
        interval_returns = [math.log(closes[0] / opens[0])]
        if len(group) > 1:
            gaps = np.asarray(
                (timestamps[1:] - timestamps[:-1])
                / pd.Timedelta(minutes=1),
                dtype=float,
            )
            close_returns = np.diff(np.log(closes))
            interval_returns.extend(
                float(value)
                for gap, value in zip(gaps, close_returns, strict=True)
                if np.isclose(gap, 15)
            )
        if expected_keys is not None:
            complete_schedule &= len(interval_returns) == len(expected_keys)
        row = {
            "symbol": str(symbol),
            "trade_date": normalized_date,
            "regular_session_complete": bool(complete_schedule),
            "expected_return_count": len(expected_keys or observed_keys),
            "valid_return_count": len(interval_returns),
            "bar_count": len(group),
        }
        if complete_schedule:
            row.update(
                {
                    "rth_open": float(opens[0]),
                    "rth_close": float(closes[-1]),
                    "rth_simple_return": float(closes[-1] / opens[0] - 1),
                    "rth_log_return": float(math.log(closes[-1] / opens[0])),
                    "daily_volume": float(group["volume"].sum()),
                    "realized_variance": float(
                        np.square(interval_returns).sum()
                    ),
                }
            )
        else:
            row.update(
                {
                    "rth_open": np.nan,
                    "rth_close": np.nan,
                    "rth_simple_return": np.nan,
                    "rth_log_return": np.nan,
                    "daily_volume": np.nan,
                    "realized_variance": np.nan,
                }
            )
        rows.append(row)
    observed = pd.DataFrame(rows)
    sessions = (
        pd.DatetimeIndex(official_sessions).normalize()
        if official_sessions is not None
        else pd.DatetimeIndex(
            sorted(observed["trade_date"].unique())
        ).normalize()
    )
    frames: list[pd.DataFrame] = []
    for symbol, frame in observed.groupby("symbol", sort=True):
        frame = frame.set_index("trade_date").reindex(sessions)
        frame.index.name = "trade_date"
        frame["symbol"] = str(symbol)
        frame["realized_volatility"] = np.sqrt(frame["realized_variance"])
        frame["daily_close_return"] = (
            frame["rth_close"] / frame["rth_close"].shift(1) - 1
        )
        frame["lagged_realized_volatility"] = frame[
            "realized_volatility"
        ].shift(1)
        frame["lagged_daily_volume"] = frame["daily_volume"].shift(1)
        trailing_volume = frame["daily_volume"].shift(2).rolling(
            20, min_periods=10
        ).mean()
        frame["lagged_relative_daily_volume_20d"] = (
            frame["lagged_daily_volume"] / trailing_volume
        )
        frames.append(frame.reset_index())
    return pd.concat(frames, ignore_index=True).sort_values(
        ["symbol", "trade_date"]
    )


def aggregate_extended_bars(
    bars: pd.DataFrame,
    session_name: str,
    official_sessions: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    if bars.empty:
        return pd.DataFrame()
    ordered = bars.sort_values(["symbol", "timestamp_utc"])
    daily = (
        ordered.groupby(["symbol", "trade_date"], as_index=False)
        .agg(
            first_open=("open", "first"),
            last_close=("close", "last"),
            session_volume=("volume", "sum"),
            session_bar_count=("close", "size"),
        )
        .sort_values(["symbol", "trade_date"])
    )
    if official_sessions is not None:
        sessions = pd.DatetimeIndex(official_sessions).normalize()
        frames: list[pd.DataFrame] = []
        for symbol, frame in daily.groupby("symbol", sort=True):
            frame = frame.set_index("trade_date").reindex(sessions)
            frame.index.name = "trade_date"
            frame["symbol"] = str(symbol)
            frames.append(frame.reset_index())
        daily = pd.concat(frames, ignore_index=True).sort_values(
            ["symbol", "trade_date"]
        )
    daily[f"{session_name}_return"] = (
        daily["last_close"] / daily["first_open"] - 1
    )
    trailing = daily.groupby("symbol", sort=False)["session_volume"].transform(
        lambda values: values.shift(1).rolling(20, min_periods=10).mean()
    )
    daily[f"relative_{session_name}_volume_20d"] = (
        daily["session_volume"] / trailing
    )
    return daily.rename(
        columns={
            "first_open": f"{session_name}_first_open",
            "last_close": f"{session_name}_last_close",
            "session_volume": f"{session_name}_volume",
            "session_bar_count": f"{session_name}_bar_count",
        }
    )


def sector_dispersion(daily: pd.DataFrame, pairs: Sequence[Pair]) -> pd.DataFrame:
    stock_sector = pd.DataFrame(
        [{"symbol": pair.stock, "sector": pair.sector} for pair in pairs]
    )
    frame = daily.merge(stock_sector, on="symbol", how="inner")
    required = stock_sector.groupby("sector")["symbol"].nunique()
    dispersion = frame.groupby(["sector", "trade_date"], as_index=False).agg(
        sector_return_dispersion=("daily_close_return", "std"),
        observed_stock_count=("daily_close_return", "count"),
    )
    dispersion["required_stock_count"] = dispersion["sector"].map(required)
    dispersion.loc[
        dispersion["observed_stock_count"].ne(
            dispersion["required_stock_count"]
        ),
        "sector_return_dispersion",
    ] = np.nan
    dispersion = dispersion.sort_values(["sector", "trade_date"])
    dispersion["lagged_sector_return_dispersion"] = dispersion.groupby(
        "sector", sort=False
    )["sector_return_dispersion"].shift(1)
    return dispersion[
        ["sector", "trade_date", "lagged_sector_return_dispersion"]
    ]


def attach_market_context(panel: pd.DataFrame, official_dir: Path) -> pd.DataFrame:
    fred_path = official_dir / "fred_market_series.csv"
    macro_path = official_dir / "macro_release_calendar.csv"
    output = panel.sort_values("forecast_date").copy()

    if fred_path.exists():
        fred = pd.read_csv(fred_path, parse_dates=["date"])
        fred["value"] = pd.to_numeric(fred["value"], errors="coerce")
        wide = fred.pivot(index="date", columns="series_id", values="value").sort_index()
        calendar_dates = pd.DatetimeIndex(sorted(output["forecast_date"].unique()))
        context = pd.DataFrame(index=calendar_dates)
        # VIX close from the immediately preceding completed trading session.
        vix = wide.get("VIXCLS", pd.Series(dtype=float)).reindex(
            calendar_dates
        ).ffill()
        context["vix_lag1"] = vix.shift(1)
        context["vix_change_lag1"] = vix.diff().shift(1)
        # H.15 constant-maturity yields are conservatively delayed two target
        # sessions because the prior observation is published after the next
        # morning forecast cutoff.
        for series_id, suffix in (("DGS2", "2y"), ("DGS5", "5y"), ("DGS10", "10y")):
            values = wide.get(series_id, pd.Series(dtype=float)).reindex(
                calendar_dates
            ).ffill()
            context[f"treasury_{suffix}_lag2"] = values.shift(2)
            context[f"treasury_{suffix}_change_lag2"] = values.diff().shift(2)
        context.index.name = "forecast_date"
        output = output.merge(context.reset_index(), on="forecast_date", how="left")

    if macro_path.exists():
        macro = pd.read_csv(macro_path, parse_dates=["release_date"])
        macro["preopen"] = macro["before_09_cutoff"].astype(str).str.lower().eq("true")
        macro["is_bls"] = macro["source"].eq("BLS")
        macro["is_bea"] = macro["source"].eq("BEA")
        macro["is_fomc"] = macro["source"].eq("Federal Reserve")
        by_date = (
            macro.groupby("release_date", as_index=False)
            .agg(
                scheduled_macro_event_count=("event_name", "size"),
                scheduled_preopen_macro_count=("preopen", "sum"),
                bls_release_day=("is_bls", "max"),
                bea_release_day=("is_bea", "max"),
                fomc_decision_day=("is_fomc", "max"),
            )
            .rename(columns={"release_date": "forecast_date"})
        )
        output = output.merge(by_date, on="forecast_date", how="left")
        for column in (
            "scheduled_macro_event_count",
            "scheduled_preopen_macro_count",
            "bls_release_day",
            "bea_release_day",
            "fomc_decision_day",
        ):
            output[column] = output[column].fillna(0).astype(int)
    return output


def rolling_factor_implied_correlations(
    daily: pd.DataFrame,
    factors_path: Path,
    pairs: Sequence[Pair],
    *,
    window: int = 252,
    minimum: int = 126,
    factor_lag_sessions: int = 2,
) -> pd.DataFrame:
    """Estimate common-factor-implied stock/ETF correlation.

    The model uses Mkt-RF, SMB, HML, and Mom. For forecast date t, the rolling
    sample ends ``factor_lag_sessions`` rows before t. Residual covariance is
    assumed zero; residual variances remain in each asset's total variance.
    """

    if not factors_path.exists():
        return pd.DataFrame(
            columns=["stock", "forecast_date", "factor_implied_correlation"]
        )
    factors = pd.read_csv(factors_path, parse_dates=["date"]).rename(
        columns={"date": "trade_date"}
    )
    factor_columns = ["mkt_rf", "smb", "hml", "mom"]
    for column in factor_columns + ["rf"]:
        factors[column] = pd.to_numeric(factors[column], errors="coerce")
    returns = daily.pivot(
        index="trade_date", columns="symbol", values="daily_close_return"
    ).sort_index()
    combined = factors.set_index("trade_date").join(returns, how="inner")
    results: list[dict[str, object]] = []
    for pair in pairs:
        if pair.stock not in combined or pair.benchmark not in combined:
            continue
        subset = combined[
            factor_columns + ["rf", pair.stock, pair.benchmark]
        ].dropna()
        if len(subset) < minimum + factor_lag_sessions:
            continue
        x_all = subset[factor_columns].to_numpy(dtype=float)
        stock_all = subset[pair.stock].to_numpy(dtype=float) - subset["rf"].to_numpy(
            dtype=float
        )
        benchmark_all = subset[pair.benchmark].to_numpy(dtype=float) - subset[
            "rf"
        ].to_numpy(dtype=float)
        dates = subset.index.to_list()
        for forecast_position in range(minimum + factor_lag_sessions, len(subset)):
            end = forecast_position - factor_lag_sessions
            begin = max(0, end - window)
            if end - begin < minimum:
                continue
            factors_window = x_all[begin:end]
            design = np.column_stack(
                [np.ones(len(factors_window), dtype=float), factors_window]
            )
            stock_window = stock_all[begin:end]
            benchmark_window = benchmark_all[begin:end]
            stock_coefficients, _, _, _ = np.linalg.lstsq(
                design, stock_window, rcond=None
            )
            benchmark_coefficients, _, _, _ = np.linalg.lstsq(
                design, benchmark_window, rcond=None
            )
            stock_residual = stock_window - design @ stock_coefficients
            benchmark_residual = benchmark_window - design @ benchmark_coefficients
            factor_covariance = np.cov(factors_window, rowvar=False, ddof=1)
            stock_beta = stock_coefficients[1:]
            benchmark_beta = benchmark_coefficients[1:]
            common_covariance = float(
                stock_beta @ factor_covariance @ benchmark_beta
            )
            stock_variance = float(
                stock_beta @ factor_covariance @ stock_beta
                + np.var(stock_residual, ddof=1)
            )
            benchmark_variance = float(
                benchmark_beta @ factor_covariance @ benchmark_beta
                + np.var(benchmark_residual, ddof=1)
            )
            denominator = math.sqrt(max(stock_variance * benchmark_variance, 0.0))
            value = common_covariance / denominator if denominator > 0 else np.nan
            results.append(
                {
                    "stock": pair.stock,
                    "forecast_date": dates[forecast_position],
                    "factor_implied_correlation": float(np.clip(value, -1, 1)),
                }
            )
    return pd.DataFrame(results)


def build_panel(
    regular_bars: pd.DataFrame,
    premarket_bars: pd.DataFrame,
    aftermarket_bars: pd.DataFrame,
    *,
    pairs: Sequence[Pair],
    official_dir: Path,
    include_factor_feature: bool,
    official_sessions: pd.DatetimeIndex | None = None,
    expected_regular_interval_keys: (
        Mapping[pd.Timestamp, frozenset[str]] | None
    ) = None,
) -> pd.DataFrame:
    daily = aggregate_regular_bars(
        regular_bars,
        official_sessions,
        expected_regular_interval_keys,
    )
    if daily.empty:
        raise ValueError("No completed regular-session bars were found")
    premarket = aggregate_extended_bars(
        premarket_bars, "premarket", official_sessions
    )
    aftermarket = aggregate_extended_bars(
        aftermarket_bars, "aftermarket", official_sessions
    )
    dispersion = sector_dispersion(daily, pairs)

    stock_rows = pd.DataFrame(
        [
            {
                "sector": pair.sector,
                "stock": pair.stock,
                "benchmark": pair.benchmark,
            }
            for pair in pairs
        ]
    )
    panel_dates = (
        pd.DatetimeIndex(official_sessions).normalize()
        if official_sessions is not None
        else pd.DatetimeIndex(sorted(daily["trade_date"].unique())).normalize()
    )
    dates = pd.DataFrame({"forecast_date": panel_dates})
    stock_rows["_key"] = 1
    dates["_key"] = 1
    panel = stock_rows.merge(dates, on="_key").drop(columns="_key")

    regular_lookup = daily.rename(
        columns={"symbol": "stock", "trade_date": "forecast_date"}
    )
    regular_columns = [
        "stock",
        "forecast_date",
        "lagged_realized_volatility",
        "lagged_relative_daily_volume_20d",
    ]
    panel = panel.merge(regular_lookup[regular_columns], on=["stock", "forecast_date"])

    benchmark_lookup = daily.rename(
        columns={
            "symbol": "benchmark",
            "trade_date": "forecast_date",
            "lagged_realized_volatility": "sector_lagged_realized_volatility",
            "lagged_relative_daily_volume_20d": "sector_lagged_relative_daily_volume_20d",
        }
    )
    panel = panel.merge(
        benchmark_lookup[
            [
                "benchmark",
                "forecast_date",
                "sector_lagged_realized_volatility",
                "sector_lagged_relative_daily_volume_20d",
            ]
        ],
        on=["benchmark", "forecast_date"],
        how="left",
    )
    panel = panel.merge(
        dispersion.rename(columns={"trade_date": "forecast_date"}),
        on=["sector", "forecast_date"],
        how="left",
    )

    if not premarket.empty:
        previous_close = daily[
            ["symbol", "trade_date", "rth_close"]
        ].sort_values(["symbol", "trade_date"])
        previous_close["prior_rth_close"] = previous_close.groupby(
            "symbol", sort=False
        )["rth_close"].shift(1)
        premarket = premarket.merge(
            previous_close[["symbol", "trade_date", "prior_rth_close"]],
            on=["symbol", "trade_date"],
            how="left",
            validate="one_to_one",
        )
        premarket["overnight_return"] = (
            premarket["premarket_last_close"] / premarket["prior_rth_close"] - 1
        )

        stock_pm = premarket.rename(
            columns={"symbol": "stock", "trade_date": "forecast_date"}
        )
        panel = panel.merge(
            stock_pm[
                [
                    "stock",
                    "forecast_date",
                    "overnight_return",
                    "premarket_return",
                    "premarket_volume",
                    "relative_premarket_volume_20d",
                    "premarket_bar_count",
                ]
            ].rename(columns={"overnight_return": "stock_overnight_return"}),
            on=["stock", "forecast_date"],
            how="left",
        )
        benchmark_pm = premarket.rename(
            columns={"symbol": "benchmark", "trade_date": "forecast_date"}
        )
        panel = panel.merge(
            benchmark_pm[
                [
                    "benchmark",
                    "forecast_date",
                    "overnight_return",
                    "premarket_return",
                    "premarket_volume",
                    "relative_premarket_volume_20d",
                    "premarket_bar_count",
                ]
            ].rename(
                columns={
                    "overnight_return": "sector_overnight_return",
                    "premarket_return": "sector_premarket_return",
                    "premarket_volume": "sector_premarket_volume",
                    "relative_premarket_volume_20d": (
                        "sector_relative_premarket_volume_20d"
                    ),
                    "premarket_bar_count": "sector_premarket_bar_count",
                }
            ),
            on=["benchmark", "forecast_date"],
            how="left",
        )
        panel["stock_minus_sector_overnight_return"] = (
            panel["stock_overnight_return"] - panel["sector_overnight_return"]
        )

    if not aftermarket.empty:
        after = aftermarket.sort_values(["symbol", "trade_date"])
        trading_dates = list(panel_dates)
        next_session = {
            trading_dates[index]: trading_dates[index + 1]
            for index in range(len(trading_dates) - 1)
        }
        after["forecast_date"] = after["trade_date"].map(next_session)
        stock_after = after.rename(columns={"symbol": "stock"})
        panel = panel.merge(
            stock_after[
                [
                    "stock",
                    "forecast_date",
                    "aftermarket_return",
                    "aftermarket_volume",
                    "relative_aftermarket_volume_20d",
                ]
            ].rename(
                columns={
                    "aftermarket_return": "prior_aftermarket_return",
                    "aftermarket_volume": "prior_aftermarket_volume",
                    "relative_aftermarket_volume_20d": (
                        "prior_relative_aftermarket_volume_20d"
                    ),
                }
            ),
            on=["stock", "forecast_date"],
            how="left",
        )

    availability_sources = {
        "stock_premarket_available": "premarket_bar_count",
        "sector_premarket_available": "sector_premarket_bar_count",
        "stock_prior_aftermarket_available": "prior_aftermarket_return",
    }
    for indicator, source in availability_sources.items():
        panel[indicator] = (
            panel[source].notna().astype("int8")
            if source in panel
            else np.int8(0)
        )

    panel = attach_market_context(panel, official_dir)
    if include_factor_feature:
        factor_feature = rolling_factor_implied_correlations(
            daily, official_dir / "fama_french_daily.csv", pairs
        )
        panel = panel.merge(
            factor_feature, on=["stock", "forecast_date"], how="left"
        )
    return panel.sort_values(["forecast_date", "sector", "stock"]).reset_index(drop=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--regular-dir", type=Path, default=DEFAULT_REGULAR_DIR)
    parser.add_argument("--extended-dir", type=Path, default=DEFAULT_EXTENDED_DIR)
    parser.add_argument("--official-dir", type=Path, default=DEFAULT_OFFICIAL_DIR)
    parser.add_argument("--calendar", type=Path, default=DEFAULT_CALENDAR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-factor-feature", action="store_true")
    parser.add_argument("--verify-input-hashes", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pairs = load_pairs(args.config)
    regular_paths = completed_csv_paths(
        args.regular_dir, verify_hashes=args.verify_input_hashes
    )
    premarket_paths = completed_csv_paths(
        args.extended_dir / "premarket",
        verify_hashes=args.verify_input_hashes,
    )
    aftermarket_paths = completed_csv_paths(
        args.extended_dir / "aftermarket",
        verify_hashes=args.verify_input_hashes,
    )
    plan = {
        "mode": "dry-run" if args.dry_run else "build",
        "pair_count": len(pairs),
        "regular_files_snapshotted": len(regular_paths),
        "premarket_files_snapshotted": len(premarket_paths),
        "aftermarket_files_snapshotted": len(aftermarket_paths),
        "network_requests": 0,
        "regular_input_access": "read-only",
        "output": str(args.output),
        "factor_implied_correlation": not args.skip_factor_feature,
        "input_hashes_verified": bool(args.verify_input_hashes),
    }
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return 0

    official_sessions = load_official_sessions(args.calendar)
    expected_regular_keys = load_expected_regular_interval_keys(args.calendar)
    panel = build_panel(
        read_bar_files(regular_paths),
        read_bar_files(premarket_paths),
        read_bar_files(aftermarket_paths),
        pairs=pairs,
        official_dir=args.official_dir,
        include_factor_feature=not args.skip_factor_feature,
        official_sessions=official_sessions,
        expected_regular_interval_keys=expected_regular_keys,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = args.output.with_suffix(args.output.suffix + ".tmp")
    panel.to_parquet(temp_path, index=False)
    temp_path.replace(args.output)
    manifest = {
        "status": "complete",
        "schema_version": 1,
        "rows": len(panel),
        "columns": list(panel.columns),
        "first_forecast_date": (
            panel["forecast_date"].min().date().isoformat() if len(panel) else None
        ),
        "last_forecast_date": (
            panel["forecast_date"].max().date().isoformat() if len(panel) else None
        ),
        "regular_files_snapshotted": [str(path) for path in regular_paths],
        "premarket_files_snapshotted": [str(path) for path in premarket_paths],
        "aftermarket_files_snapshotted": [str(path) for path in aftermarket_paths],
        "input_snapshot": {
            "actual_hashes_verified": bool(args.verify_input_hashes),
            "regular_digest": snapshot_digest(regular_paths),
            "premarket_digest": snapshot_digest(premarket_paths),
            "aftermarket_digest": snapshot_digest(aftermarket_paths),
            "calendar_path": str(args.calendar),
            "calendar_sha256": sha256_file(args.calendar),
        },
        "parameters": {
            "official_session_reindex": True,
            "complete_regular_schedule_required": True,
            "extended_volume_window": "20 official sessions",
            "include_factor_feature": not args.skip_factor_feature,
        },
        "coverage": {
            column: int(panel[column].notna().sum())
            for column in panel.columns
            if column not in {"sector", "stock", "benchmark", "forecast_date"}
        },
        "output": {
            "path": str(args.output),
            "sha256": sha256_file(args.output),
        },
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "rows": len(panel)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
