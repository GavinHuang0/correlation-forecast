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
import json
import math
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


DEFAULT_CONFIG = Path("config/price_universe.json")
DEFAULT_REGULAR_DIR = Path("data/prices/alpaca/15min/sip/all")
DEFAULT_EXTENDED_DIR = Path("data/prices/alpaca-extended/15min/sip/all")
DEFAULT_OFFICIAL_DIR = Path("data/external/official-quant")
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


def completed_csv_paths(base: Path) -> list[Path]:
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
            paths.append(path)
    return sorted(paths)


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


def aggregate_regular_bars(bars: pd.DataFrame) -> pd.DataFrame:
    if bars.empty:
        return pd.DataFrame()
    ordered = bars.sort_values(["symbol", "timestamp_utc"]).copy()
    ordered["log_close"] = np.log(ordered["close"].astype(float))
    ordered["intraday_log_return"] = ordered.groupby(
        ["symbol", "trade_date"], sort=False
    )["log_close"].diff()
    ordered["squared_intraday_return"] = ordered["intraday_log_return"].pow(2)
    daily = (
        ordered.groupby(["symbol", "trade_date"], as_index=False)
        .agg(
            rth_open=("open", "first"),
            rth_close=("close", "last"),
            daily_volume=("volume", "sum"),
            realized_variance=("squared_intraday_return", "sum"),
            bar_count=("close", "size"),
        )
        .sort_values(["symbol", "trade_date"])
    )
    daily["realized_volatility"] = np.sqrt(daily["realized_variance"])
    grouped = daily.groupby("symbol", sort=False)
    daily["daily_close_return"] = grouped["rth_close"].pct_change(fill_method=None)
    daily["lagged_realized_volatility"] = grouped["realized_volatility"].shift(1)
    daily["lagged_daily_volume"] = grouped["daily_volume"].shift(1)
    trailing_volume = grouped["daily_volume"].transform(
        lambda values: values.shift(2).rolling(20, min_periods=10).mean()
    )
    daily["lagged_relative_daily_volume_20d"] = (
        daily["lagged_daily_volume"] / trailing_volume
    )
    return daily


def aggregate_extended_bars(
    bars: pd.DataFrame, session_name: str
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
    daily[f"{session_name}_return"] = daily["last_close"] / daily["first_open"] - 1
    trailing = daily.groupby("symbol", sort=False)["session_volume"].transform(
        lambda values: values.shift(1).rolling(20, min_periods=10).mean()
    )
    daily[f"relative_{session_name}_volume_20d"] = daily["session_volume"] / trailing
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
    dispersion = (
        frame.groupby(["sector", "trade_date"], as_index=False)[
            "daily_close_return"
        ]
        .std(ddof=1)
        .rename(columns={"daily_close_return": "sector_return_dispersion"})
        .sort_values(["sector", "trade_date"])
    )
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
) -> pd.DataFrame:
    daily = aggregate_regular_bars(regular_bars)
    if daily.empty:
        raise ValueError("No completed regular-session bars were found")
    premarket = aggregate_extended_bars(premarket_bars, "premarket")
    aftermarket = aggregate_extended_bars(aftermarket_bars, "aftermarket")
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
    dates = pd.DataFrame(
        {"forecast_date": sorted(daily["trade_date"].unique())}
    )
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
        prior_close = daily[
            ["symbol", "trade_date", "rth_close"]
        ].sort_values(["symbol", "trade_date"])
        prior_close = prior_close.rename(
            columns={
                "trade_date": "prior_trade_date",
                "rth_close": "prior_rth_close",
            }
        )
        matched_premarket = []
        for symbol, symbol_premarket in premarket.groupby("symbol", sort=False):
            symbol_prior = prior_close[prior_close["symbol"] == symbol]
            if symbol_prior.empty:
                matched_premarket.append(symbol_premarket)
                continue
            matched_premarket.append(
                pd.merge_asof(
                    symbol_premarket.sort_values("trade_date"),
                    symbol_prior.drop(columns="symbol").sort_values(
                        "prior_trade_date"
                    ),
                    left_on="trade_date",
                    right_on="prior_trade_date",
                    direction="backward",
                    allow_exact_matches=False,
                )
            )
        premarket = pd.concat(matched_premarket, ignore_index=True)
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
                ]
            ].rename(
                columns={
                    "overnight_return": "sector_overnight_return",
                    "premarket_return": "sector_premarket_return",
                    "premarket_volume": "sector_premarket_volume",
                    "relative_premarket_volume_20d": (
                        "sector_relative_premarket_volume_20d"
                    ),
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
        trading_dates = sorted(daily["trade_date"].unique())
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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-factor-feature", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pairs = load_pairs(args.config)
    regular_paths = completed_csv_paths(args.regular_dir)
    premarket_paths = completed_csv_paths(args.extended_dir / "premarket")
    aftermarket_paths = completed_csv_paths(args.extended_dir / "aftermarket")
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
    }
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return 0

    panel = build_panel(
        read_bar_files(regular_paths),
        read_bar_files(premarket_paths),
        read_bar_files(aftermarket_paths),
        pairs=pairs,
        official_dir=args.official_dir,
        include_factor_feature=not args.skip_factor_feature,
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
        "output": str(args.output),
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
