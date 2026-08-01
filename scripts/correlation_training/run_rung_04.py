"""Run causally refit bivariate Gaussian DCC-GARCH benchmarks."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_bollerslev_core_features as core  # noqa: E402
from scripts.correlation_training import dcc  # noqa: E402
from scripts.correlation_training import training_common as common  # noqa: E402


def validate_dcc_history_panel(
    evaluation_panel: pd.DataFrame,
    history_panel: pd.DataFrame,
    official_sessions: pd.DatetimeIndex,
) -> bool:
    """Require dense daily history while allowing filtered evaluation rows."""

    keys = ["stock", "forecast_date"]
    if history_panel.duplicated(keys).any():
        raise ValueError("DCC history panel contains duplicate stock-date rows")
    evaluation_stocks = set(evaluation_panel["stock"])
    history_stocks = set(history_panel["stock"])
    if evaluation_stocks != history_stocks:
        raise ValueError("DCC history and evaluation stock universes differ")
    stock_count = len(history_stocks)
    stocks_per_date = history_panel.groupby("forecast_date")["stock"].nunique()
    if stocks_per_date.empty or not stocks_per_date.eq(stock_count).all():
        raise ValueError("DCC history panel is not cross-sectionally dense")
    history_dates = pd.DatetimeIndex(
        sorted(pd.to_datetime(history_panel["forecast_date"]).unique())
    ).normalize()
    official_sessions = pd.DatetimeIndex(official_sessions).normalize()
    expected_dates = official_sessions[
        official_sessions.to_series(index=official_sessions).between(
            history_dates.min(), history_dates.max(), inclusive="both"
        ).to_numpy()
    ]
    if not history_dates.equals(expected_dates):
        raise ValueError(
            "DCC history panel is not complete on the official-session calendar"
        )
    evaluation_keys = pd.MultiIndex.from_frame(evaluation_panel[keys])
    history_keys = pd.MultiIndex.from_frame(history_panel[keys])
    if not evaluation_keys.isin(history_keys).all():
        raise ValueError("DCC history panel does not cover every evaluation row")
    if (
        history_panel["forecast_date"].min()
        > evaluation_panel["forecast_date"].min()
        or history_panel["forecast_date"].max()
        < evaluation_panel["forecast_date"].max()
    ):
        raise ValueError("DCC history panel does not span the evaluation period")
    return True


def fit_pair_forecasts(
    stock_frame: pd.DataFrame,
    spec: common.TargetSpec,
    *,
    first_test_forecast_date: pd.Timestamp,
    left_parameters: dcc.GarchParameters | None = None,
) -> tuple[pd.DataFrame, dict[str, object], dcc.GarchParameters]:
    frame = stock_frame.sort_values("forecast_date").reset_index(drop=True)
    left_values = frame[spec.history_stock_return].to_numpy(dtype=float)
    right_values = frame[spec.history_benchmark_return].to_numpy(dtype=float)
    # The first actual test-session row contains only the preceding official
    # session's lagged return. It is known at that forecast cutoff and may be
    # used in the parameter fit. Calendar boundary dates can be holidays, so
    # the actual first forecast date is passed explicitly.
    estimation_mask = frame["forecast_date"].le(first_test_forecast_date)
    paired_estimation = (
        estimation_mask.to_numpy()
        & np.isfinite(left_values)
        & np.isfinite(right_values)
    )
    if paired_estimation.sum() < 250:
        raise ValueError("Insufficient paired pre-test returns for DCC")
    if left_parameters is None:
        left_parameters = dcc.fit_garch(left_values[paired_estimation])
    right_parameters = dcc.fit_garch(right_values[paired_estimation])
    left_variance, left_residual, left_standardized = dcc.filter_garch(
        left_values, left_parameters
    )
    right_variance, right_residual, right_standardized = dcc.filter_garch(
        right_values, right_parameters
    )
    standardized = np.column_stack(
        [left_standardized, right_standardized]
    )
    dcc_parameters = dcc.fit_dcc(standardized[estimation_mask.to_numpy()])
    _, q_after = dcc.filter_dcc(standardized, dcc_parameters)
    one_step = np.full(len(frame), np.nan)
    five_step = np.full(len(frame), np.nan)
    for position in range(len(frame)):
        one_step[position], five_step[position] = dcc.correlation_forecasts(
            left_variance[position],
            right_variance[position],
            left_residual[position],
            right_residual[position],
            q_after[position],
            left_parameters,
            right_parameters,
            dcc_parameters,
            horizon=5,
        )
    forecasts = frame[["forecast_date"]].copy()
    forecasts["dcc_t1_correlation"] = one_step
    forecasts["dcc_t2_correlation"] = five_step
    parameters = dcc.parameter_record(
        left_parameters, right_parameters, dcc_parameters
    )
    parameters["estimation_rows"] = int(paired_estimation.sum())
    parameters["parameter_rows_through_forecast_date"] = (
        first_test_forecast_date.date().isoformat()
    )
    if "asof_session" in frame:
        information_sessions = pd.to_datetime(
            frame.loc[estimation_mask, "asof_session"]
        ).dropna()
        parameters["return_information_through"] = (
            information_sessions.max().date().isoformat()
            if len(information_sessions)
            else None
        )
    return forecasts, parameters, left_parameters


def run_rung_04(
    panel: pd.DataFrame,
    protocol: dict[str, object],
    *,
    history_panel: pd.DataFrame | None = None,
    official_sessions: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    history_panel = panel if history_panel is None else history_panel
    dense_history_passed = validate_dcc_history_panel(
        panel, history_panel, official_sessions
    )
    predictions: list[pd.DataFrame] = []
    fit_records: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    specs = {spec.name: spec for spec in common.target_specs()}
    left_parameter_cache: dict[
        tuple[str, str], dcc.GarchParameters
    ] = {}
    forecast_cache: dict[tuple[str, str, str], pd.DataFrame] = {}
    common_stock_history = bool(
        np.allclose(
            history_panel["history_etf_stock_rth_log_return_lag1"],
            history_panel["history_loo_stock_rth_log_return_lag1"],
            equal_nan=True,
        )
    )
    if not common_stock_history:
        raise AssertionError("ETF and LOO stock-return histories differ")
    for fold in protocol["folds"]:
        test_start = pd.Timestamp(fold["test_start"])
        test_end = pd.Timestamp(fold["test_end"])
        actual_test_dates = panel.loc[
            panel["forecast_date"].between(
                test_start, test_end, inclusive="both"
            ),
            "forecast_date",
        ]
        if actual_test_dates.empty:
            raise ValueError(f"No test sessions found for {fold['name']}")
        first_test_forecast_date = pd.Timestamp(actual_test_dates.min())
        for benchmark in ("etf", "loo"):
            representative = specs[f"t1_{benchmark}"]
            for stock, stock_frame in history_panel.groupby(
                "stock", sort=True
            ):
                cache_key = (fold["name"], benchmark, stock)
                left_key = (fold["name"], stock)
                try:
                    forecast, parameters, fitted_left = fit_pair_forecasts(
                        stock_frame,
                        representative,
                        first_test_forecast_date=first_test_forecast_date,
                        left_parameters=left_parameter_cache.get(left_key),
                    )
                    left_parameter_cache[left_key] = fitted_left
                    forecast_cache[cache_key] = forecast
                    fit_records.append(
                        {
                            "fold": fold["name"],
                            "benchmark_type": benchmark,
                            "stock": stock,
                            "parameters": parameters,
                        }
                    )
                    print(
                        f"[{len(fit_records)}/"
                        f"{len(protocol['folds']) * 2 * panel['stock'].nunique()}] "
                        f"fit {fold['name']} {benchmark} {stock}",
                        flush=True,
                    )
                except (ValueError, RuntimeError, FloatingPointError) as exc:
                    failures.append(
                        {
                            "fold": fold["name"],
                            "benchmark_type": benchmark,
                            "stock": stock,
                            "error": str(exc),
                        }
                    )

        for spec in common.target_specs():
            masks = common.split_masks(panel, spec, fold)
            test = panel.loc[masks["test"]].copy()
            target_rows: list[pd.DataFrame] = []
            prediction_column = (
                "dcc_t1_correlation"
                if spec.horizon == "t1"
                else "dcc_t2_correlation"
            )
            for stock, stock_test in test.groupby("stock", sort=True):
                key = (fold["name"], spec.benchmark, stock)
                if key not in forecast_cache:
                    continue
                merged = stock_test.merge(
                    forecast_cache[key],
                    on="forecast_date",
                    how="left",
                    validate="many_to_one",
                )
                if merged[prediction_column].isna().any():
                    raise AssertionError(
                        f"Missing DCC forecasts for {spec.name} {stock}"
                    )
                target_rows.append(merged)
            if target_rows:
                ordered_test = pd.concat(target_rows, ignore_index=True).sort_values(
                    ["forecast_date", "sector", "stock"]
                )
                prediction_map = ordered_test.set_index(
                    ["stock", "forecast_date"]
                )[prediction_column]
                aligned_prediction = np.array(
                    [
                        prediction_map.loc[(row.stock, row.forecast_date)]
                        for row in test.itertuples()
                        if (row.stock, row.forecast_date) in prediction_map.index
                    ],
                    dtype=float,
                )
                eligible = test.apply(
                    lambda row: (row["stock"], row["forecast_date"])
                    in prediction_map.index,
                    axis=1,
                )
                test = test.loc[eligible].copy()
                predictions.append(
                    common.prediction_frame(
                        test,
                        spec,
                        fold_name=fold["name"],
                        model_name="dcc_garch",
                        predicted_fisher=aligned_prediction,
                    )
                )
    if not predictions:
        raise RuntimeError("No DCC-GARCH forecasts were produced")
    prediction_panel = pd.concat(predictions, ignore_index=True)
    metrics = common.summarize_predictions(prediction_panel)
    fold_metrics = common.summarize_predictions(
        prediction_panel, ("fold", "target", "model")
    )
    expected_fit_count = len(protocol["folds"]) * 2 * panel["stock"].nunique()
    review = {
        "status": "passed",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "fit_count": len(fit_records),
        "expected_fit_count": int(expected_fit_count),
        "failure_count": len(failures),
        "failure_free": not failures,
        "fit_count_complete": len(fit_records) == expected_fit_count,
        "all_fitted_parameters_converged": all(
            record["parameters"][section]["converged"]
            for record in fit_records
            for section in ("left_garch", "right_garch", "dcc")
        ),
        "all_dcc_persistence_stationary": all(
            record["parameters"]["dcc"]["a"]
            + record["parameters"]["dcc"]["b"]
            < dcc.MAX_PERSISTENCE
            for record in fit_records
        ),
        "dcc_fits_above_0_995_persistence": sum(
            record["parameters"]["dcc"]["a"]
            + record["parameters"]["dcc"]["b"]
            > 0.995
            for record in fit_records
        ),
        "systems_with_marginal_above_0_995_persistence": sum(
            any(
                record["parameters"][section]["alpha"]
                + record["parameters"][section]["beta"]
                > 0.995
                for section in ("left_garch", "right_garch")
            )
            for record in fit_records
        ),
        "predictions_finite": bool(
            np.isfinite(
                prediction_panel[
                    ["predicted_fisher_z", "predicted_correlation"]
                ].to_numpy(dtype=float)
            ).all()
        ),
        "prediction_bounds_valid": bool(
            prediction_panel["predicted_correlation"].between(-1, 1).all()
        ),
        "prediction_keys_unique": bool(
            ~prediction_panel.duplicated(
                ["fold", "target", "model", "stock", "forecast_date"]
            ).any()
        ),
        "parameter_information_set": (
            "the first actual test-session row supplies only the preceding "
            "official session's lagged RTH return; states then update "
            "sequentially"
        ),
        "dense_daily_history_panel_passed": dense_history_passed,
        "official_session_calendar_complete": dense_history_passed,
        "history_panel_rows": int(len(history_panel)),
        "history_panel_session_count": int(
            history_panel["forecast_date"].nunique()
        ),
        "history_panel_first_date": (
            history_panel["forecast_date"].min().date().isoformat()
        ),
        "history_panel_last_date": (
            history_panel["forecast_date"].max().date().isoformat()
        ),
        "etf_loo_stock_history_identical": common_stock_history,
        "t2_forecast": (
            "sum five recursively forecast covariance/variance matrices, "
            "then normalize"
        ),
    }
    required = [
        "failure_free",
        "fit_count_complete",
        "all_fitted_parameters_converged",
        "all_dcc_persistence_stationary",
        "predictions_finite",
        "prediction_bounds_valid",
        "prediction_keys_unique",
        "dense_daily_history_panel_passed",
        "official_session_calendar_complete",
        "etf_loo_stock_history_identical",
    ]
    if not all(bool(review[key]) for key in required):
        review["status"] = "failed"
        raise AssertionError(f"Rung 4 review failed: {review}")
    return prediction_panel, metrics, {
        "fits": fit_records,
        "failures": failures,
        "fold_metrics": fold_metrics.to_dict(orient="records"),
        "review": review,
    }


def parser() -> argparse.ArgumentParser:
    output = argparse.ArgumentParser(description=__doc__)
    output.add_argument("--panel", type=Path)
    output.add_argument("--dcc-history-panel", type=Path)
    output.add_argument("--protocol", type=Path, default=common.PROTOCOL_PATH)
    output.add_argument("--experiment-root", type=Path)
    output.add_argument("--output-root", type=Path)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    protocol = common.load_protocol(args.protocol)
    paths = common.resolve_training_paths(
        protocol,
        panel=args.panel,
        experiment_root=args.experiment_root,
        output_root=args.output_root,
    )
    predictions, metrics, details = run_rung_04(
        common.load_panel(paths.panel),
        protocol,
        history_panel=common.load_panel(
            args.dcc_history_panel or paths.dcc_history_panel
        ),
        official_sessions=core.load_official_sessions(paths.dcc_calendar),
    )
    output_root = paths.output_root / "rung_04"
    experiment_root = paths.experiment_root / "rung_04"
    common.write_parquet(output_root / "predictions.parquet", predictions)
    common.write_json(output_root / "fits.json", details["fits"])
    common.write_json(output_root / "failures.json", details["failures"])
    common.write_json(output_root / "fold_metrics.json", details["fold_metrics"])
    common.write_json(experiment_root / "review.json", details["review"])
    common.write_json(
        experiment_root / "summary.json",
        {"status": "complete", "metrics": metrics.to_dict(orient="records")},
    )
    for target, frame in predictions.groupby("target", sort=True):
        common.write_parquet(
            output_root / target / "predictions.parquet", frame
        )
        common.write_json(
            experiment_root / target / "metrics.json",
            metrics[metrics["target"].eq(target)].to_dict(orient="records"),
        )
    print(metrics.to_string(index=False))
    print(json.dumps(details["review"], indent=2))
    if details["failures"]:
        print(json.dumps({"failures": details["failures"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
