"""Build audited T1/T2 ETF and five-peer leave-one-out correlation targets.

The target contract is intentionally stricter than the historical feature
contract.  For every stock-date, stock, ETF, and all five peer returns must be
present on the exact official regular-session interval schedule.  ETF and LOO
therefore use the same observations.  Realized correlation is non-demeaned:

    sum(x*y) / sqrt(sum(x**2) * sum(y**2))

T1 is the current regular session.  T2 sums covariance and variance
components over the current plus next four official sessions before
normalizing.  LOO returns are exact equal-weight simple-return averages
converted back to log returns at every interval.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_bollerslev_core_features as core  # noqa: E402


DEFAULT_CONFIG = Path("config/price_universe.json")
DEFAULT_REGULAR_DIR = Path("data/prices/alpaca/15min/sip/all")
DEFAULT_CALENDAR = Path(
    "data/prices/alpaca/calendar/2016-01-01_2026-06-30.json"
)
DEFAULT_OUTPUT = Path(
    "data/features/quant/correlation_targets_and_loo_features.parquet"
)
DEFAULT_AUDIT_OUTPUT = Path(
    "experiments/quant_training/v1/construction/targets/audit.json"
)
TARGET_SCHEMA_VERSION = 1
COMPONENTS = (
    "realized_covariance",
    "stock_realized_variance",
    "benchmark_realized_variance",
)
NEGATIVE_COMPONENTS = (
    "negative_realized_covariance",
    "stock_negative_realized_variance",
    "benchmark_negative_realized_variance",
)
DAILY_RETURN_COMPONENTS = (
    "stock_rth_log_return",
    "benchmark_rth_log_return",
)
LOO_PAIR_COLUMNS = [
    *(f"loo_rc_{label}" for label in core.HAR_HORIZONS),
    *(f"loo_rc_negative_{label}" for label in core.HAR_HORIZONS),
    *(f"loo_exp_rc_{label}" for label in core.EXPONENTIAL_CENTERS),
    *(f"loo_exp_rc_negative_{label}" for label in core.EXPONENTIAL_CENTERS),
]


def exact_equal_weight_log_return(values: np.ndarray) -> np.ndarray:
    """Convert peer log returns to an equal-weight simple-return basket."""

    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("Peer return array must be two dimensional")
    simple = np.expm1(values)
    basket_simple = simple.mean(axis=1)
    if np.any(basket_simple <= -1):
        raise ValueError("Equal-weight peer basket produced a return <= -100%")
    return np.log1p(basket_simple)


def _wide_returns(
    returns: Mapping[str, pd.DataFrame],
    symbols: Sequence[str],
) -> pd.DataFrame:
    frames = []
    for symbol in symbols:
        frame = returns[symbol][
            ["trade_date", "interval_key", "log_return"]
        ].rename(columns={"log_return": symbol})
        frames.append(frame.set_index(["trade_date", "interval_key"]))
    wide = pd.concat(frames, axis=1, join="inner").reset_index()
    if wide.duplicated(["trade_date", "interval_key"]).any():
        raise ValueError("Duplicate return intervals found while aligning legs")
    return wide


def _components_from_aligned(
    aligned: pd.DataFrame,
    *,
    stock_column: str,
    benchmark_column: str,
    expected_keys: Mapping[pd.Timestamp, frozenset[str]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for trade_date, group in aligned.groupby("trade_date", sort=True):
        trade_date = pd.Timestamp(trade_date).normalize()
        expected = expected_keys.get(trade_date)
        keys = frozenset(group["interval_key"].astype(str))
        complete = expected is not None and keys == expected
        row: dict[str, object] = {
            "trade_date": trade_date,
            "aligned_return_count": len(group),
            "expected_return_count": len(expected or ()),
            "complete_official_schedule": bool(complete),
        }
        if not complete:
            row.update(
                {
                    column: np.nan
                    for column in (
                        *COMPONENTS,
                        *NEGATIVE_COMPONENTS,
                        *DAILY_RETURN_COMPONENTS,
                    )
                }
            )
            rows.append(row)
            continue
        stock = group[stock_column].to_numpy(dtype=float)
        benchmark = group[benchmark_column].to_numpy(dtype=float)
        if not np.isfinite(stock).all() or not np.isfinite(benchmark).all():
            row.update(
                {
                    column: np.nan
                    for column in (
                        *COMPONENTS,
                        *NEGATIVE_COMPONENTS,
                        *DAILY_RETURN_COMPONENTS,
                    )
                }
            )
            rows.append(row)
            continue
        both_negative = (stock < 0) & (benchmark < 0)
        row.update(
            {
                "realized_covariance": float(np.sum(stock * benchmark)),
                "stock_realized_variance": float(np.sum(stock**2)),
                "benchmark_realized_variance": float(np.sum(benchmark**2)),
                "negative_realized_covariance": float(
                    np.sum(np.where(both_negative, stock * benchmark, 0.0))
                ),
                "stock_negative_realized_variance": float(
                    np.sum(np.where(stock < 0, stock**2, 0.0))
                ),
                "benchmark_negative_realized_variance": float(
                    np.sum(np.where(benchmark < 0, benchmark**2, 0.0))
                ),
                "stock_rth_log_return": float(np.sum(stock)),
                "benchmark_rth_log_return": float(np.sum(benchmark)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("trade_date").reset_index(drop=True)


def build_common_rth_components(
    returns: Mapping[str, pd.DataFrame],
    sector: core.Sector,
    stock: str,
    expected_keys: Mapping[pd.Timestamp, frozenset[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build ETF and LOO components from an identical strict interval set."""

    peers = [symbol for symbol in sector.stocks if symbol != stock]
    if len(peers) != 5 or len(sector.stocks) != 6:
        raise ValueError(
            f"{stock} requires exactly five peers in a six-stock sector"
        )
    symbols = [stock, sector.benchmark, *peers]
    aligned = _wide_returns(returns, symbols)
    aligned["loo"] = exact_equal_weight_log_return(
        aligned[peers].to_numpy(dtype=float)
    )
    etf = _components_from_aligned(
        aligned,
        stock_column=stock,
        benchmark_column=sector.benchmark,
        expected_keys=expected_keys,
    )
    loo = _components_from_aligned(
        aligned,
        stock_column=stock,
        benchmark_column="loo",
        expected_keys=expected_keys,
    )
    if not etf[
        ["trade_date", "aligned_return_count", "complete_official_schedule"]
    ].equals(
        loo[
            ["trade_date", "aligned_return_count", "complete_official_schedule"]
        ]
    ):
        raise AssertionError("ETF and LOO target observation sets diverged")
    return etf, loo


def _correlation(
    covariance: pd.Series,
    left_variance: pd.Series,
    right_variance: pd.Series,
) -> np.ndarray:
    return core.component_correlation(covariance, left_variance, right_variance)


def _future_component_sum(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    return (
        frame.iloc[::-1]
        .rolling(horizon, min_periods=horizon)
        .sum()
        .iloc[::-1]
    )


def target_and_lag_frame(
    components: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    *,
    prefix: str,
    forward_horizon: int = 5,
) -> pd.DataFrame:
    indexed = components.set_index("trade_date").reindex(sessions)
    indexed.index.name = "forecast_date"
    output = pd.DataFrame(index=sessions)
    output.index.name = "forecast_date"

    t1_rho = _correlation(
        indexed["realized_covariance"],
        indexed["stock_realized_variance"],
        indexed["benchmark_realized_variance"],
    )
    output[f"target_{prefix}_t1_correlation"] = t1_rho
    output[f"target_{prefix}_t1_fisher_z"] = np.arctanh(
        np.clip(t1_rho, -0.995, 0.995)
    )
    for column in COMPONENTS:
        output[f"target_{prefix}_t1_{column}"] = indexed[column]
    output[f"target_{prefix}_t1_aligned_return_count"] = indexed[
        "aligned_return_count"
    ]
    output[f"target_{prefix}_t1_expected_return_count"] = indexed[
        "expected_return_count"
    ]

    forward = _future_component_sum(
        indexed[list(COMPONENTS)], forward_horizon
    )
    t2_rho = _correlation(
        forward["realized_covariance"],
        forward["stock_realized_variance"],
        forward["benchmark_realized_variance"],
    )
    output[f"target_{prefix}_t2_correlation"] = t2_rho
    output[f"target_{prefix}_t2_fisher_z"] = np.arctanh(
        np.clip(t2_rho, -0.995, 0.995)
    )
    for column in COMPONENTS:
        output[f"target_{prefix}_t2_{column}"] = forward[column]
    end_dates = pd.Series(pd.NaT, index=sessions, dtype="datetime64[ns]")
    if len(sessions) >= forward_horizon:
        end_dates.iloc[: -(forward_horizon - 1)] = sessions[
            forward_horizon - 1 :
        ].to_numpy()
    output[f"target_{prefix}_t2_end_date"] = end_dates

    past_components = indexed[list(COMPONENTS)].shift(1)
    for label, window in {"d": 1, "w": 5, "m": 21}.items():
        aggregate = past_components.rolling(window, min_periods=window).sum()
        output[f"{prefix}_rth_rc_{label}"] = _correlation(
            aggregate["realized_covariance"],
            aggregate["stock_realized_variance"],
            aggregate["benchmark_realized_variance"],
        )
    for column in DAILY_RETURN_COMPONENTS:
        output[f"history_{prefix}_{column}_lag1"] = indexed[column].shift(1)
    return output.reset_index()


def build_strict_bucket_returns(
    returns: Mapping[str, pd.DataFrame],
    peers: Sequence[str],
    expected_keys: Mapping[pd.Timestamp, frozenset[str]],
    *,
    symbol: str,
) -> pd.DataFrame:
    aligned = _wide_returns(returns, list(peers))
    aligned["log_return"] = exact_equal_weight_log_return(
        aligned[list(peers)].to_numpy(dtype=float)
    )
    keep: list[pd.DataFrame] = []
    for trade_date, group in aligned.groupby("trade_date", sort=True):
        trade_date = pd.Timestamp(trade_date).normalize()
        if frozenset(group["interval_key"].astype(str)) != expected_keys.get(
            trade_date, frozenset()
        ):
            continue
        frame = group[["trade_date", "interval_key", "log_return"]].copy()
        frame.insert(0, "symbol", symbol)
        keep.append(frame)
    if not keep:
        return pd.DataFrame(
            columns=["symbol", "trade_date", "interval_key", "log_return"]
        )
    return pd.concat(keep, ignore_index=True)


def build_loo_pair_features(
    full_day_returns: Mapping[str, pd.DataFrame],
    sector: core.Sector,
    stock: str,
    sessions: pd.DatetimeIndex,
    expected_full_day_keys: Mapping[pd.Timestamp, frozenset[str]],
    *,
    exponential_window: int,
    min_exponential_valid: int,
    minimum_weight_fraction: float,
) -> pd.DataFrame:
    peers = [symbol for symbol in sector.stocks if symbol != stock]
    if len(peers) != 5:
        raise ValueError(f"{stock} requires exactly five LOO peers")
    bucket = build_strict_bucket_returns(
        full_day_returns,
        peers,
        expected_full_day_keys,
        symbol=f"{stock}_LOO5",
    )
    features, _ = core.build_pair_features(
        full_day_returns[stock],
        bucket,
        sessions,
        min_aligned_returns=15,
        min_alignment_ratio=0.8,
        require_overnight=True,
        exponential_window=exponential_window,
        min_exponential_valid=min_exponential_valid,
        min_exponential_weight_fraction=minimum_weight_fraction,
        expected_keys=expected_full_day_keys,
    )
    features = features.shift(1)
    rename = {
        column: f"loo_{column}"
        for column in features.columns
        if column.startswith(("rc_", "exp_rc_"))
    }
    return features.rename(columns=rename).reset_index().rename(
        columns={"asof_session": "forecast_date", "index": "forecast_date"}
    )


def build_targets(
    rth_returns: Mapping[str, pd.DataFrame],
    full_day_returns: Mapping[str, pd.DataFrame],
    sectors: Sequence[core.Sector],
    sessions: pd.DatetimeIndex,
    expected_rth_keys: Mapping[pd.Timestamp, frozenset[str]],
    expected_full_day_keys: Mapping[pd.Timestamp, frozenset[str]],
    *,
    exponential_window: int = core.DEFAULT_EXPONENTIAL_WINDOW,
    min_exponential_valid: int = 490,
    minimum_weight_fraction: float = 0.99,
) -> pd.DataFrame:
    panels: list[pd.DataFrame] = []
    for sector in sectors:
        for stock in sector.stocks:
            etf_components, loo_components = build_common_rth_components(
                rth_returns, sector, stock, expected_rth_keys
            )
            etf = target_and_lag_frame(
                etf_components, sessions, prefix="etf"
            )
            loo = target_and_lag_frame(
                loo_components, sessions, prefix="loo"
            )
            loo_features = build_loo_pair_features(
                full_day_returns,
                sector,
                stock,
                sessions,
                expected_full_day_keys,
                exponential_window=exponential_window,
                min_exponential_valid=min_exponential_valid,
                minimum_weight_fraction=minimum_weight_fraction,
            )
            frame = etf.merge(
                loo, on="forecast_date", validate="one_to_one"
            ).merge(
                loo_features, on="forecast_date", how="left", validate="one_to_one"
            )
            frame.insert(0, "benchmark", sector.benchmark)
            frame.insert(0, "stock", stock)
            frame.insert(0, "sector", sector.name)
            panels.append(frame)
    panel = pd.concat(panels, ignore_index=True)
    panel = panel.sort_values(["forecast_date", "sector", "stock"]).reset_index(
        drop=True
    )
    if panel.duplicated(["stock", "forecast_date"]).any():
        raise AssertionError("Duplicate stock-date targets")
    for horizon in ("t1", "t2"):
        etf_valid = panel[f"target_etf_{horizon}_correlation"].notna()
        loo_valid = panel[f"target_loo_{horizon}_correlation"].notna()
        if not etf_valid.equals(loo_valid):
            raise AssertionError(
                f"ETF and LOO {horizon.upper()} eligibility diverged"
            )
    correlation_columns = [
        column
        for column in panel
        if column.endswith("_correlation") or "_rc_" in column
    ]
    for column in correlation_columns:
        values = panel[column].dropna()
        if not values.between(-1.0, 1.0).all():
            raise AssertionError(f"Out-of-range correlation in {column}")
    return panel


def build_audit(panel: pd.DataFrame) -> dict[str, object]:
    by_target: dict[str, object] = {}
    for horizon in ("t1", "t2"):
        for target in ("etf", "loo"):
            column = f"target_{target}_{horizon}_correlation"
            values = panel[column].dropna()
            by_target[f"{horizon}_{target}"] = {
                "valid_rows": int(len(values)),
                "first_date": (
                    panel.loc[values.index, "forecast_date"]
                    .min()
                    .date()
                    .isoformat()
                    if len(values)
                    else None
                ),
                "last_date": (
                    panel.loc[values.index, "forecast_date"]
                    .max()
                    .date()
                    .isoformat()
                    if len(values)
                    else None
                ),
                "mean": float(values.mean()) if len(values) else None,
                "std": float(values.std()) if len(values) else None,
                "minimum": float(values.min()) if len(values) else None,
                "maximum": float(values.max()) if len(values) else None,
            }
    normal_day_count = 26
    counts = panel.loc[
        panel["target_etf_t1_correlation"].notna(),
        "target_etf_t1_aligned_return_count",
    ]
    return {
        "status": "complete",
        "schema_version": TARGET_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "row_count": int(len(panel)),
        "stock_count": int(panel["stock"].nunique()),
        "date_count": int(panel["forecast_date"].nunique()),
        "target_summaries": by_target,
        "aligned_return_count_distribution": {
            str(int(key)): int(value)
            for key, value in counts.value_counts().sort_index().items()
        },
        "expected_normal_rth_return_count": normal_day_count,
        "loo_pair_feature_coverage": {
            column: int(panel[column].notna().sum())
            for column in LOO_PAIR_COLUMNS
        },
        "contracts": {
            "correlation": "non-demeaned realized correlation",
            "t1": "current official RTH session",
            "t2": "component sum over current plus next four sessions",
            "loo": "equal-weight five-peer simple-return factor per interval",
            "common_interval_set": True,
            "complete_official_schedule_required": True,
        },
    }


def parser() -> argparse.ArgumentParser:
    output = argparse.ArgumentParser(description=__doc__)
    output.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    output.add_argument("--regular-dir", type=Path, default=DEFAULT_REGULAR_DIR)
    output.add_argument("--calendar", type=Path, default=DEFAULT_CALENDAR)
    output.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    output.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT_OUTPUT)
    output.add_argument("--start", type=core.parse_date)
    output.add_argument("--end", type=core.parse_date)
    output.add_argument("--exponential-window", type=int, default=500)
    output.add_argument("--min-exponential-valid", type=int, default=490)
    output.add_argument(
        "--min-exponential-weight-fraction", type=float, default=0.99
    )
    output.add_argument("--verify-input-hashes", action="store_true")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    sectors = core.load_sectors(args.config)
    symbols = {
        symbol
        for sector in sectors
        for symbol in (*sector.stocks, sector.benchmark)
    }
    snapshot = core.snapshot_completed_chunks(
        args.regular_dir,
        symbols,
        verify_hashes=args.verify_input_hashes,
        build_calendar_path=args.calendar,
    )
    calendar_records = core.load_calendar_records(args.calendar)
    sessions = core.load_official_sessions(args.calendar)
    rth_keys = core.expected_interval_keys(
        calendar_records, include_overnight=False
    )
    full_keys = core.expected_interval_keys(
        calendar_records, include_overnight=True
    )
    rth_returns: dict[str, pd.DataFrame] = {}
    full_returns: dict[str, pd.DataFrame] = {}
    for position, symbol in enumerate(sorted(symbols), start=1):
        bars = core.read_symbol_bars(snapshot[symbol], symbol, end=args.end)
        rth_returns[symbol] = core.build_interval_returns(
            bars, include_overnight=False, official_sessions=sessions
        )
        full_returns[symbol] = core.build_interval_returns(
            bars, include_overnight=True, official_sessions=sessions
        )
        print(f"[{position}/{len(symbols)}] prepared {symbol}", flush=True)
    panel = build_targets(
        rth_returns,
        full_returns,
        sectors,
        sessions,
        rth_keys,
        full_keys,
        exponential_window=args.exponential_window,
        min_exponential_valid=args.min_exponential_valid,
        minimum_weight_fraction=args.min_exponential_weight_fraction,
    )
    if args.start is not None:
        panel = panel[panel["forecast_date"] >= pd.Timestamp(args.start)]
    if args.end is not None:
        panel = panel[panel["forecast_date"] <= pd.Timestamp(args.end)]
    if panel.empty:
        raise ValueError("No rows remain after target date filters")
    panel = panel.reset_index(drop=True)
    core.write_frame_atomic(args.output, panel)
    audit = build_audit(panel)
    audit["output"] = {
        "path": str(args.output),
        "sha256": core.sha256_file(args.output),
    }
    audit["input_snapshot"] = {
        "actual_hashes_verified": bool(args.verify_input_hashes),
        "combined_path_and_manifest_hash": core.snapshot_digest(
            snapshot, args.regular_dir
        ),
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    core.write_json_atomic(args.audit_output, audit)
    core.write_json_atomic(core.output_manifest_path(args.output), audit)
    print(json.dumps(audit, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
