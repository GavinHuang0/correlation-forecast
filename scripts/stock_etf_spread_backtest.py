"""Run the frozen stock-sector-ETF convergence strategy backtest.

The strategy converts the existing five-session stock-ETF correlation
forecasts into a filter for a separate beta-adjusted return-divergence signal.
All signal inputs end before the 09:30 ET opening trade. Positions are held
only during regular trading hours, and five overlapping sleeves align the
economic holding window with the T2 target.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_bollerslev_core_features as core  # noqa: E402


DEFAULT_PROTOCOL = Path("config/stock_etf_spread_backtest_v1.json")
VARIANTS = (
    "unconditional",
    "historical_level",
    "ml_level",
    "ml_strengthening",
    "ml_weakening",
)


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != "1.0":
        raise ValueError("Unsupported stock-ETF spread protocol")
    variants = tuple(protocol["signal"]["variants"])
    if variants != VARIANTS:
        raise ValueError("Protocol variants do not match the implemented ladder")
    holding = int(protocol["portfolio"]["holding_sessions"])
    fraction = float(protocol["portfolio"]["daily_sleeve_fraction"])
    if holding < 1 or not np.isclose(fraction, 1.0 / holding):
        raise ValueError("Daily sleeve fraction must equal one over holding sessions")
    return protocol


def complete_daily_rth_returns(
    interval_returns: pd.DataFrame,
    expected_keys: Mapping[pd.Timestamp, frozenset[str]],
) -> pd.DataFrame:
    """Aggregate only exact official-session interval sets."""

    rows: list[dict[str, object]] = []
    symbol_values = interval_returns["symbol"].dropna().unique()
    if len(symbol_values) != 1:
        raise ValueError("Daily aggregation requires exactly one symbol")
    symbol = str(symbol_values[0])
    for trade_date, group in interval_returns.groupby("trade_date", sort=True):
        trade_date = pd.Timestamp(trade_date).normalize()
        expected = expected_keys.get(trade_date)
        observed = frozenset(group["interval_key"].astype(str))
        if expected is None or observed != expected:
            continue
        values = group["log_return"].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            continue
        log_return = float(values.sum())
        rows.append(
            {
                "trade_date": trade_date,
                "symbol": symbol,
                "rth_log_return": log_return,
                "rth_simple_return": float(np.expm1(log_return)),
                "interval_count": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def load_daily_returns(
    *,
    universe_path: Path,
    regular_dir: Path,
    calendar_path: Path,
    end: pd.Timestamp,
    verify_hashes: bool,
) -> tuple[pd.DataFrame, pd.DatetimeIndex, list[core.Sector]]:
    sectors = core.load_sectors(universe_path)
    symbols = sorted(
        {
            symbol
            for sector in sectors
            for symbol in (*sector.stocks, sector.benchmark)
        }
    )
    snapshot = core.snapshot_completed_chunks(
        regular_dir,
        symbols,
        verify_hashes=verify_hashes,
        build_calendar_path=calendar_path,
    )
    calendar_records = core.load_calendar_records(calendar_path)
    sessions = core.load_official_sessions(calendar_path)
    expected_keys = core.expected_interval_keys(
        calendar_records, include_overnight=False
    )
    frames: list[pd.DataFrame] = []
    for position, symbol in enumerate(symbols, start=1):
        bars = core.read_symbol_bars(
            snapshot[symbol], symbol, end=end.date()
        )
        interval_returns = core.build_interval_returns(
            bars,
            include_overnight=False,
            official_sessions=sessions,
        )
        daily = complete_daily_rth_returns(interval_returns, expected_keys)
        if daily.empty:
            raise ValueError(f"No complete daily returns for {symbol}")
        frames.append(daily)
        print(
            f"[prices {position}/{len(symbols)}] prepared {symbol}",
            flush=True,
        )
    output = pd.concat(frames, ignore_index=True)
    if output.duplicated(["trade_date", "symbol"]).any():
        raise ValueError("Daily return panel contains duplicate symbol-dates")
    return (
        output.sort_values(["trade_date", "symbol"]).reset_index(drop=True),
        sessions[sessions <= end],
        sectors,
    )


def load_oos_predictions(
    path: Path, *, target: str, model: str
) -> pd.DataFrame:
    predictions = pd.read_parquet(path)
    output = predictions[
        predictions["target"].eq(target) & predictions["model"].eq(model)
    ].copy()
    if output.empty:
        raise ValueError(f"No predictions found for {target}/{model}")
    output["forecast_date"] = pd.to_datetime(
        output["forecast_date"]
    ).dt.normalize()
    keys = ["fold", "forecast_date", "stock"]
    if output.duplicated(keys).any():
        raise ValueError("Selected predictions contain duplicate keys")
    required = {
        "fold",
        "forecast_date",
        "sector",
        "stock",
        "benchmark",
        "predicted_correlation",
        "persistence_correlation",
    }
    missing = required.difference(output.columns)
    if missing:
        raise ValueError(f"Predictions lack required columns: {sorted(missing)}")
    keep = sorted(required)
    output = output[keep].copy()
    numeric = output[
        ["predicted_correlation", "persistence_correlation"]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Selected predictions contain nonfinite correlations")
    return output.sort_values(keys).reset_index(drop=True)


def rolling_beta_and_divergence(
    stock_returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    beta_window: int,
    beta_minimum: int,
    divergence_window: int,
    beta_lower: float,
    beta_upper: float,
) -> pd.DataFrame:
    """Create pre-open beta and divergence using returns through t-1 only."""

    aligned = pd.concat(
        [
            stock_returns.rename("stock"),
            benchmark_returns.rename("benchmark"),
        ],
        axis=1,
        join="outer",
    ).sort_index()
    paired_benchmark = aligned["benchmark"].where(aligned["stock"].notna())
    covariance = aligned["stock"].rolling(
        beta_window, min_periods=beta_minimum
    ).cov(paired_benchmark)
    variance = paired_benchmark.rolling(
        beta_window, min_periods=beta_minimum
    ).var()
    beta = (covariance / variance).replace([np.inf, -np.inf], np.nan)
    beta = beta.clip(beta_lower, beta_upper).shift(1)
    stock_trailing = (
        aligned["stock"]
        .rolling(divergence_window, min_periods=divergence_window)
        .sum()
        .shift(1)
    )
    benchmark_trailing = (
        aligned["benchmark"]
        .rolling(divergence_window, min_periods=divergence_window)
        .sum()
        .shift(1)
    )
    output = pd.DataFrame(
        {
            "rolling_beta": beta,
            "trailing_stock_log_return": stock_trailing,
            "trailing_benchmark_log_return": benchmark_trailing,
        }
    )
    output["return_divergence"] = (
        output["rolling_beta"] * output["trailing_benchmark_log_return"]
        - output["trailing_stock_log_return"]
    )
    output.index.name = "forecast_date"
    return output


def attach_signal_features(
    predictions: pd.DataFrame,
    daily_returns: pd.DataFrame,
    protocol: Mapping[str, Any],
) -> pd.DataFrame:
    signal = protocol["signal"]
    log_returns = daily_returns.pivot(
        index="trade_date", columns="symbol", values="rth_log_return"
    ).sort_index()
    frames: list[pd.DataFrame] = []
    for (stock, benchmark), group in predictions.groupby(
        ["stock", "benchmark"], sort=True
    ):
        if stock not in log_returns or benchmark not in log_returns:
            raise ValueError(f"Missing return series for {stock}/{benchmark}")
        features = rolling_beta_and_divergence(
            log_returns[stock],
            log_returns[benchmark],
            beta_window=int(signal["beta_window_sessions"]),
            beta_minimum=int(signal["beta_minimum_sessions"]),
            divergence_window=int(signal["divergence_lookback_sessions"]),
            beta_lower=float(signal["beta_clip_lower"]),
            beta_upper=float(signal["beta_clip_upper"]),
        ).reset_index()
        merged = group.merge(
            features,
            on="forecast_date",
            how="left",
            validate="one_to_one",
        )
        frames.append(merged)
    output = pd.concat(frames, ignore_index=True)
    feature_columns = [
        "rolling_beta",
        "trailing_stock_log_return",
        "trailing_benchmark_log_return",
        "return_divergence",
    ]
    if output[feature_columns].isna().any().any():
        counts = output[feature_columns].isna().sum()
        raise ValueError(f"Signal features are incomplete: {counts.to_dict()}")
    output["predicted_correlation_change"] = (
        output["predicted_correlation"]
        - output["persistence_correlation"]
    )
    return output.sort_values(
        ["forecast_date", "sector", "stock"]
    ).reset_index(drop=True)


def _variant_gate(
    frame: pd.DataFrame,
    variant: str,
    *,
    correlation_floor: float,
) -> tuple[pd.Series, pd.Series]:
    if variant == "unconditional":
        eligible = pd.Series(True, index=frame.index)
        strength_source = pd.Series(1.0, index=frame.index)
    elif variant == "historical_level":
        eligible = frame["persistence_correlation"].ge(correlation_floor)
        strength_source = frame["persistence_correlation"]
    elif variant == "ml_level":
        eligible = frame["predicted_correlation"].ge(correlation_floor)
        strength_source = frame["predicted_correlation"]
    elif variant == "ml_strengthening":
        eligible = frame["predicted_correlation"].ge(correlation_floor) & frame[
            "predicted_correlation_change"
        ].gt(0)
        strength_source = frame["predicted_correlation_change"]
    elif variant == "ml_weakening":
        eligible = frame["predicted_correlation"].ge(correlation_floor) & frame[
            "predicted_correlation_change"
        ].lt(0)
        strength_source = -frame["predicted_correlation_change"]
    else:
        raise ValueError(f"Unknown strategy variant {variant}")
    strength = pd.Series(0.0, index=frame.index)
    if eligible.any():
        strength.loc[eligible] = strength_source.loc[eligible].rank(
            method="average", pct=True
        )
    return eligible, strength


def construct_sleeve_weights(
    signals: pd.DataFrame,
    *,
    variant: str,
    correlation_floor: float,
    minimum_names: int,
    sleeve_gross: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build sector-neutral stock weights and their netted ETF hedges."""

    signal_records: list[pd.DataFrame] = []
    position_records: list[dict[str, object]] = []
    for (fold, forecast_date), date_frame in signals.groupby(
        ["fold", "forecast_date"], sort=True
    ):
        eligible, gate_strength = _variant_gate(
            date_frame, variant, correlation_floor=correlation_floor
        )
        annotated = date_frame.copy()
        annotated["variant"] = variant
        annotated["gate_eligible"] = eligible.astype(bool)
        annotated["gate_strength"] = gate_strength.astype(float)
        annotated["raw_stock_score"] = 0.0
        sector_weights: list[pd.DataFrame] = []
        for sector, sector_frame in annotated.groupby("sector", sort=True):
            selected = sector_frame[sector_frame["gate_eligible"]].copy()
            if len(selected) < minimum_names:
                continue
            divergence_rank = selected["return_divergence"].rank(
                method="average"
            )
            divergence_rank -= float(divergence_rank.mean())
            raw = divergence_rank * selected["gate_strength"]
            raw -= float(raw.mean())
            gross = float(raw.abs().sum())
            if not np.isfinite(gross) or gross <= 0:
                continue
            selected["raw_stock_score"] = raw
            selected["sector_stock_weight"] = raw / gross
            annotated.loc[
                selected.index, "raw_stock_score"
            ] = selected["raw_stock_score"]
            sector_weights.append(selected)
        signal_records.append(annotated)
        if not sector_weights:
            continue
        active = pd.concat(sector_weights, ignore_index=False)
        active_sector_count = int(active["sector"].nunique())
        active["stock_weight"] = (
            active["sector_stock_weight"] / active_sector_count
        )
        etf_weights = (
            active.assign(
                etf_contribution=-active["rolling_beta"]
                * active["stock_weight"]
            )
            .groupby(["sector", "benchmark"], sort=True)[
                "etf_contribution"
            ]
            .sum()
        )
        combined_gross = float(active["stock_weight"].abs().sum()) + float(
            etf_weights.abs().sum()
        )
        if not np.isfinite(combined_gross) or combined_gross <= 0:
            continue
        scale = sleeve_gross / combined_gross
        active["stock_weight"] *= scale
        etf_weights *= scale
        for row in active.itertuples():
            position_records.append(
                {
                    "fold": fold,
                    "forecast_date": forecast_date,
                    "variant": variant,
                    "sector": row.sector,
                    "leg": "stock",
                    "symbol": row.stock,
                    "weight": float(row.stock_weight),
                }
            )
        for (sector, benchmark), weight in etf_weights.items():
            if np.isclose(weight, 0.0):
                continue
            position_records.append(
                {
                    "fold": fold,
                    "forecast_date": forecast_date,
                    "variant": variant,
                    "sector": sector,
                    "leg": "etf",
                    "symbol": benchmark,
                    "weight": float(weight),
                }
            )
    annotated_signals = pd.concat(signal_records, ignore_index=True)
    positions = pd.DataFrame(position_records)
    if not positions.empty:
        gross = positions.groupby(
            ["fold", "forecast_date", "variant"], sort=True
        )["weight"].apply(lambda values: float(values.abs().sum()))
        if not np.allclose(gross.to_numpy(dtype=float), sleeve_gross):
            raise AssertionError("Sleeve weights do not meet the gross target")
        stock_net = (
            positions[positions["leg"].eq("stock")]
            .groupby(
                ["fold", "forecast_date", "variant", "sector"], sort=True
            )["weight"]
            .sum()
        )
        if not np.allclose(stock_net.to_numpy(dtype=float), 0.0, atol=1e-12):
            raise AssertionError("Stock weights are not sector neutral")
    return annotated_signals, positions


def evaluation_calendar(
    signals: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    *,
    holding_sessions: int,
) -> pd.DataFrame:
    normalized = pd.DatetimeIndex(sessions).normalize()
    positions = {date: index for index, date in enumerate(normalized)}
    rows: list[dict[str, object]] = []
    for fold, frame in signals.groupby("fold", sort=True):
        first = pd.Timestamp(frame["forecast_date"].min()).normalize()
        last_start = pd.Timestamp(frame["forecast_date"].max()).normalize()
        if first not in positions or last_start not in positions:
            raise ValueError("Prediction dates are absent from the calendar")
        last_position = positions[last_start] + holding_sessions - 1
        if last_position >= len(normalized):
            raise ValueError("Holding window extends beyond the calendar")
        for date in normalized[positions[first] : last_position + 1]:
            rows.append({"fold": fold, "trade_date": date})
    output = pd.DataFrame(rows)
    if output.duplicated(["fold", "trade_date"]).any():
        raise ValueError("Fold evaluation calendars overlap")
    return output


def expand_overlapping_sleeves(
    sleeve_positions: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    *,
    holding_sessions: int,
) -> pd.DataFrame:
    if sleeve_positions.empty:
        return pd.DataFrame(
            columns=[
                "fold",
                "trade_date",
                "variant",
                "sector",
                "leg",
                "symbol",
                "weight",
            ]
        )
    normalized = pd.DatetimeIndex(sessions).normalize()
    session_position = {date: index for index, date in enumerate(normalized)}
    records: list[dict[str, object]] = []
    sleeve_fraction = 1.0 / holding_sessions
    for row in sleeve_positions.itertuples(index=False):
        start = pd.Timestamp(row.forecast_date).normalize()
        if start not in session_position:
            raise ValueError(f"Sleeve start {start} is absent from calendar")
        first = session_position[start]
        dates = normalized[first : first + holding_sessions]
        if len(dates) != holding_sessions:
            raise ValueError("A sleeve extends beyond available sessions")
        for trade_date in dates:
            records.append(
                {
                    "fold": row.fold,
                    "trade_date": trade_date,
                    "variant": row.variant,
                    "sector": row.sector,
                    "leg": row.leg,
                    "symbol": row.symbol,
                    "weight": float(row.weight) * sleeve_fraction,
                }
            )
    output = (
        pd.DataFrame(records)
        .groupby(
            ["fold", "trade_date", "variant", "sector", "leg", "symbol"],
            sort=True,
            as_index=False,
        )["weight"]
        .sum()
    )
    return output


def daily_strategy_returns(
    daily_positions: pd.DataFrame,
    daily_asset_returns: pd.DataFrame,
    calendar: pd.DataFrame,
    variants: Sequence[str],
    cost_bps: Sequence[float],
) -> pd.DataFrame:
    merged = daily_positions.merge(
        daily_asset_returns[
            ["trade_date", "symbol", "rth_simple_return"]
        ],
        on=["trade_date", "symbol"],
        how="left",
        validate="many_to_one",
    )
    if merged["rth_simple_return"].isna().any():
        missing = merged.loc[
            merged["rth_simple_return"].isna(),
            ["trade_date", "symbol"],
        ].drop_duplicates()
        raise ValueError(f"Missing traded returns: {missing.to_dict('records')}")
    merged["pnl_contribution"] = (
        merged["weight"] * merged["rth_simple_return"]
    )
    aggregated = (
        merged.groupby(["fold", "trade_date", "variant"], sort=True)
        .agg(
            gross_return=("pnl_contribution", "sum"),
            gross_exposure=("weight", lambda values: float(values.abs().sum())),
            position_count=("symbol", "nunique"),
        )
        .reset_index()
    )
    grid = pd.MultiIndex.from_product(
        [variants, sorted(float(value) for value in cost_bps)],
        names=["variant", "cost_bps_per_side"],
    ).to_frame(index=False)
    grid = calendar.merge(grid, how="cross")
    output = grid.merge(
        aggregated,
        on=["fold", "trade_date", "variant"],
        how="left",
        validate="many_to_one",
    )
    output[["gross_return", "gross_exposure", "position_count"]] = output[
        ["gross_return", "gross_exposure", "position_count"]
    ].fillna(0.0)
    output["turnover"] = 2.0 * output["gross_exposure"]
    output["transaction_cost"] = (
        output["turnover"] * output["cost_bps_per_side"] / 10_000.0
    )
    output["net_return"] = (
        output["gross_return"] - output["transaction_cost"]
    )
    return output.sort_values(
        ["variant", "cost_bps_per_side", "trade_date"]
    ).reset_index(drop=True)


def _maximum_drawdown(returns: np.ndarray) -> float:
    wealth = np.concatenate(([1.0], np.cumprod(1.0 + returns)))
    peak = np.maximum.accumulate(wealth)
    return float(np.min(wealth / peak - 1.0))


def _newey_west_t_stat(values: np.ndarray, lag: int) -> float:
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        return math.nan
    centered = values - values.mean()
    long_run_variance = float(centered @ centered / n)
    for offset in range(1, min(lag, n - 1) + 1):
        weight = 1.0 - offset / (lag + 1.0)
        covariance = float(centered[offset:] @ centered[:-offset] / n)
        long_run_variance += 2.0 * weight * covariance
    if long_run_variance <= 0:
        return math.nan
    standard_error = math.sqrt(long_run_variance / n)
    return float(values.mean() / standard_error)


def performance_metrics(
    daily: pd.DataFrame, *, annualization: int, hac_lag: int
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groupings = [
        ("pooled", ["variant", "cost_bps_per_side"]),
        ("fold", ["variant", "cost_bps_per_side", "fold"]),
    ]
    for scope, columns in groupings:
        for keys, frame in daily.groupby(columns, sort=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            returns = frame["net_return"].to_numpy(dtype=float)
            mean = float(returns.mean())
            volatility = float(returns.std(ddof=1))
            rows.append(
                {
                    "scope": scope,
                    **dict(zip(columns, keys, strict=True)),
                    "dates": int(len(frame)),
                    "active_day_fraction": float(
                        frame["gross_exposure"].gt(0).mean()
                    ),
                    "mean_daily_return": mean,
                    "annualized_arithmetic_return": mean * annualization,
                    "annualized_volatility": volatility
                    * math.sqrt(annualization),
                    "annualized_sharpe": (
                        mean / volatility * math.sqrt(annualization)
                        if volatility > 0
                        else math.nan
                    ),
                    "compounded_return": float(np.prod(1.0 + returns) - 1.0),
                    "maximum_drawdown": _maximum_drawdown(returns),
                    "daily_hit_rate": float((returns > 0).mean()),
                    "mean_gross_exposure": float(
                        frame["gross_exposure"].mean()
                    ),
                    "annualized_turnover": float(frame["turnover"].mean())
                    * annualization,
                    "annualized_transaction_cost": float(
                        frame["transaction_cost"].mean()
                    )
                    * annualization,
                    "newey_west_t_stat": _newey_west_t_stat(
                        returns, hac_lag
                    ),
                }
            )
    return pd.DataFrame(rows)


def _moving_block_mean_samples(
    frame: pd.DataFrame,
    *,
    value_column: str,
    block_sessions: int,
    resamples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    sampled_sums = np.zeros(resamples, dtype=float)
    sampled_counts = np.zeros(resamples, dtype=float)
    for _, fold_frame in frame.groupby("fold", sort=True):
        values = fold_frame.sort_values("trade_date")[
            value_column
        ].to_numpy(dtype=float)
        if len(values) < block_sessions:
            raise ValueError("A fold is shorter than one bootstrap block")
        blocks_needed = math.ceil(len(values) / block_sessions)
        starts = rng.integers(
            0,
            len(values) - block_sessions + 1,
            size=(resamples, blocks_needed),
        )
        offsets = np.arange(block_sessions)
        indices = (starts[:, :, None] + offsets).reshape(resamples, -1)
        indices = indices[:, : len(values)]
        sampled_sums += values[indices].sum(axis=1)
        sampled_counts += len(values)
    return sampled_sums / sampled_counts


def paired_bootstrap_comparisons(
    daily: pd.DataFrame,
    *,
    primary_variant: str,
    comparators: Sequence[str],
    primary_cost: float,
    block_sessions: int,
    resamples: int,
    seed: int,
    annualization: int,
) -> pd.DataFrame:
    selected = daily[
        daily["cost_bps_per_side"].eq(float(primary_cost))
    ][["fold", "trade_date", "variant", "net_return"]]
    primary = selected[selected["variant"].eq(primary_variant)].rename(
        columns={"net_return": "primary_return"}
    )
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    for comparator in comparators:
        base = selected[selected["variant"].eq(comparator)].rename(
            columns={"net_return": "comparator_return"}
        )
        merged = primary.merge(
            base,
            on=["fold", "trade_date"],
            validate="one_to_one",
        )
        merged["return_difference"] = (
            merged["primary_return"] - merged["comparator_return"]
        )
        samples = _moving_block_mean_samples(
            merged,
            value_column="return_difference",
            block_sessions=block_sessions,
            resamples=resamples,
            rng=rng,
        )
        lower, upper = np.quantile(samples * annualization, [0.025, 0.975])
        observed = float(merged["return_difference"].mean() * annualization)
        rows.append(
            {
                "primary": primary_variant,
                "comparator": comparator,
                "cost_bps_per_side": float(primary_cost),
                "dates": int(len(merged)),
                "annualized_mean_return_difference": observed,
                "bootstrap_ci_lower": float(lower),
                "bootstrap_ci_upper": float(upper),
                "bootstrap_probability_primary_better": float(
                    np.mean(samples > 0)
                ),
                "block_sessions": int(block_sessions),
                "resamples": int(resamples),
            }
        )
    return pd.DataFrame(rows)


def reporting_subsample_results(
    daily: pd.DataFrame,
    protocol: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute prespecified pooled metrics on named fold subsets."""

    specifications = protocol.get("reporting_subsamples", {})
    if not specifications:
        return pd.DataFrame(), pd.DataFrame()
    inference = protocol["inference"]
    execution = protocol["execution"]
    signal = protocol["signal"]
    available_folds = set(daily["fold"].unique())
    metric_frames: list[pd.DataFrame] = []
    comparison_frames: list[pd.DataFrame] = []
    for sample, folds in specifications.items():
        requested = list(folds)
        missing = set(requested).difference(available_folds)
        if missing:
            raise ValueError(
                f"Reporting subsample {sample} lacks folds: {sorted(missing)}"
            )
        selected = daily[daily["fold"].isin(requested)].copy()
        metrics = performance_metrics(
            selected,
            annualization=int(inference["annualization_sessions"]),
            hac_lag=int(inference["moving_block_sessions"]),
        )
        metrics = metrics[metrics["scope"].eq("pooled")].copy()
        metrics.insert(0, "sample", sample)
        metric_frames.append(metrics)
        comparisons = paired_bootstrap_comparisons(
            selected,
            primary_variant=str(signal["primary_variant"]),
            comparators=inference["comparators"],
            primary_cost=float(execution["primary_cost_bps_per_side"]),
            block_sessions=int(inference["moving_block_sessions"]),
            resamples=int(inference["bootstrap_resamples"]),
            seed=int(inference["bootstrap_seed"]),
            annualization=int(inference["annualization_sessions"]),
        )
        comparisons.insert(0, "sample", sample)
        comparison_frames.append(comparisons)
    return (
        pd.concat(metric_frames, ignore_index=True),
        pd.concat(comparison_frames, ignore_index=True),
    )


def sha256_or_none(path: Path) -> str | None:
    return core.sha256_file(path) if path.exists() else None


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Return strict-JSON records, replacing pandas missing values with null."""

    clean = frame.astype(object).where(pd.notna(frame), None)
    return clean.to_dict("records")


def write_outputs(
    *,
    protocol_path: Path,
    protocol: Mapping[str, Any],
    signals: pd.DataFrame,
    sleeve_positions: pd.DataFrame,
    daily_positions: pd.DataFrame,
    daily_returns: pd.DataFrame,
    metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
    subsample_metrics: pd.DataFrame,
    subsample_comparisons: pd.DataFrame,
    input_hashes_verified: bool,
) -> dict[str, Any]:
    output_dir = Path(protocol["outputs"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "signals": output_dir / "signals.parquet",
        "sleeve_positions": output_dir / "sleeve_positions.parquet",
        "daily_positions": output_dir / "daily_positions.parquet",
        "daily_returns": output_dir / "daily_returns.parquet",
        "metrics": output_dir / "metrics.parquet",
        "comparisons": output_dir / "paired_bootstrap.parquet",
    }
    frames: list[tuple[str, pd.DataFrame]] = [
        ("signals", signals),
        ("sleeve_positions", sleeve_positions),
        ("daily_positions", daily_positions),
        ("daily_returns", daily_returns),
        ("metrics", metrics),
        ("comparisons", comparisons),
    ]
    if not subsample_metrics.empty:
        paths["subsample_metrics"] = output_dir / "subsample_metrics.parquet"
        frames.append(("subsample_metrics", subsample_metrics))
    if not subsample_comparisons.empty:
        paths["subsample_comparisons"] = (
            output_dir / "subsample_paired_bootstrap.parquet"
        )
        frames.append(("subsample_comparisons", subsample_comparisons))
    for key, frame in frames:
        core.write_frame_atomic(paths[key], frame)

    primary_variant = str(protocol["signal"]["primary_variant"])
    primary_cost = float(protocol["execution"]["primary_cost_bps_per_side"])
    pooled = metrics[
        metrics["scope"].eq("pooled")
        & metrics["cost_bps_per_side"].eq(primary_cost)
    ].drop(columns=["fold"], errors="ignore")
    summary = {
        "status": "complete_exploratory_development_backtest",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "protocol_version": protocol["protocol_version"],
        "experiment_id": protocol.get(
            "experiment_id", protocol["strategy_name"]
        ),
        "primary_variant": primary_variant,
        "primary_cost_bps_per_side": primary_cost,
        "claim_boundary": protocol["claim_boundary"],
        "protocol_inheritance": protocol.get("protocol_inheritance"),
        "forecast_selection": {
            "target": protocol["inputs"]["prediction_target"],
            "model": protocol["inputs"]["prediction_model"],
            "selection_record": protocol.get("model_selection"),
        },
        "sample": {
            "first_forecast_date": signals["forecast_date"]
            .min()
            .date()
            .isoformat(),
            "last_forecast_date": signals["forecast_date"]
            .max()
            .date()
            .isoformat(),
            "forecast_dates": int(signals["forecast_date"].nunique()),
            "stocks": int(signals["stock"].nunique()),
            "folds": sorted(signals["fold"].unique().tolist()),
            "unique_stock_date_forecasts": int(
                signals[
                    ["fold", "forecast_date", "stock"]
                ].drop_duplicates().shape[0]
            ),
            "trading_dates": int(daily_returns["trade_date"].nunique()),
        },
        "primary_cost_metrics": _json_records(pooled),
        "paired_bootstrap": _json_records(comparisons),
        "reporting_subsamples": _json_records(
            subsample_metrics[
                subsample_metrics["variant"].eq(primary_variant)
                & subsample_metrics["cost_bps_per_side"].eq(primary_cost)
            ]
            if not subsample_metrics.empty
            else subsample_metrics
        ),
        "artifacts": {
            key: {
                "path": str(path),
                "sha256": core.sha256_file(path),
            }
            for key, path in paths.items()
        },
        "inputs": {
            "protocol": {
                "path": str(protocol_path),
                "sha256": core.sha256_file(protocol_path),
            },
            "predictions": {
                "path": protocol["inputs"]["predictions"],
                "sha256": sha256_or_none(
                    Path(protocol["inputs"]["predictions"])
                ),
            },
            "regular_bar_hashes_verified": bool(input_hashes_verified),
        },
    }
    summary_path = output_dir / "summary.json"
    core.write_json_atomic(summary_path, summary)
    experiment_summary = Path(protocol["outputs"]["experiment_summary"])
    core.write_json_atomic(experiment_summary, summary)
    return summary


def parser() -> argparse.ArgumentParser:
    output = argparse.ArgumentParser(description=__doc__)
    output.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    output.add_argument("--verify-input-hashes", action="store_true")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    protocol = load_protocol(args.protocol)
    inputs = protocol["inputs"]
    signal_config = protocol["signal"]
    portfolio = protocol["portfolio"]
    execution = protocol["execution"]
    inference = protocol["inference"]

    predictions = load_oos_predictions(
        Path(inputs["predictions"]),
        target=str(inputs["prediction_target"]),
        model=str(inputs["prediction_model"]),
    )
    holding_sessions = int(portfolio["holding_sessions"])
    calendar_path = Path(inputs["calendar"])
    all_sessions = core.load_official_sessions(calendar_path)
    last_forecast = pd.Timestamp(predictions["forecast_date"].max())
    forecast_position = {
        date: index
        for index, date in enumerate(pd.DatetimeIndex(all_sessions).normalize())
    }[last_forecast]
    end = pd.Timestamp(all_sessions[forecast_position + holding_sessions - 1])
    asset_returns, sessions, _ = load_daily_returns(
        universe_path=Path(inputs["price_universe"]),
        regular_dir=Path(inputs["regular_bar_dir"]),
        calendar_path=calendar_path,
        end=end,
        verify_hashes=args.verify_input_hashes,
    )
    signals = attach_signal_features(predictions, asset_returns, protocol)

    signal_frames: list[pd.DataFrame] = []
    sleeve_frames: list[pd.DataFrame] = []
    for variant in VARIANTS:
        annotated, positions = construct_sleeve_weights(
            signals,
            variant=variant,
            correlation_floor=float(
                signal_config["predicted_correlation_floor"]
            ),
            minimum_names=int(
                portfolio["minimum_eligible_names_per_sector"]
            ),
            sleeve_gross=float(portfolio["sleeve_combined_gross"]),
        )
        signal_frames.append(annotated)
        sleeve_frames.append(positions)
    strategy_signals = pd.concat(signal_frames, ignore_index=True)
    sleeve_positions = pd.concat(sleeve_frames, ignore_index=True)
    daily_positions = expand_overlapping_sleeves(
        sleeve_positions,
        sessions,
        holding_sessions=holding_sessions,
    )
    calendar = evaluation_calendar(
        signals, sessions, holding_sessions=holding_sessions
    )
    strategy_returns = daily_strategy_returns(
        daily_positions,
        asset_returns,
        calendar,
        VARIANTS,
        execution["cost_bps_per_side"],
    )
    metrics = performance_metrics(
        strategy_returns,
        annualization=int(inference["annualization_sessions"]),
        hac_lag=int(inference["moving_block_sessions"]),
    )
    comparisons = paired_bootstrap_comparisons(
        strategy_returns,
        primary_variant=str(signal_config["primary_variant"]),
        comparators=inference["comparators"],
        primary_cost=float(execution["primary_cost_bps_per_side"]),
        block_sessions=int(inference["moving_block_sessions"]),
        resamples=int(inference["bootstrap_resamples"]),
        seed=int(inference["bootstrap_seed"]),
        annualization=int(inference["annualization_sessions"]),
    )
    subsample_metrics, subsample_comparisons = reporting_subsample_results(
        strategy_returns, protocol
    )
    summary = write_outputs(
        protocol_path=args.protocol,
        protocol=protocol,
        signals=strategy_signals,
        sleeve_positions=sleeve_positions,
        daily_positions=daily_positions,
        daily_returns=strategy_returns,
        metrics=metrics,
        comparisons=comparisons,
        subsample_metrics=subsample_metrics,
        subsample_comparisons=subsample_comparisons,
        input_hashes_verified=args.verify_input_hashes,
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
