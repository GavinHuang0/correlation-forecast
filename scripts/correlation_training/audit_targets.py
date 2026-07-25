"""Audit a materialized ETF/LOO T1/T2 target artifact."""

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


def maximum_absolute_difference(left: pd.Series, right: pd.Series) -> float:
    valid = left.notna() & right.notna()
    if not valid.any():
        return float("nan")
    return float(np.max(np.abs(left[valid].to_numpy() - right[valid].to_numpy())))


def audit_targets(
    panel: pd.DataFrame, sessions: pd.DatetimeIndex
) -> dict[str, object]:
    panel = panel.copy()
    panel["forecast_date"] = pd.to_datetime(panel["forecast_date"]).dt.normalize()
    if panel.duplicated(["stock", "forecast_date"]).any():
        raise AssertionError("Duplicate stock-date rows")
    checks: dict[str, object] = {}
    for horizon in ("t1", "t2"):
        etf_valid = panel[f"target_etf_{horizon}_correlation"].notna()
        loo_valid = panel[f"target_loo_{horizon}_correlation"].notna()
        checks[f"{horizon}_eligibility_identical"] = bool(
            etf_valid.equals(loo_valid)
        )
        for target in ("etf", "loo"):
            stem = f"target_{target}_{horizon}"
            denominator = np.sqrt(
                panel[f"{stem}_stock_realized_variance"]
                * panel[f"{stem}_benchmark_realized_variance"]
            )
            reconstructed = panel[f"{stem}_realized_covariance"] / denominator
            checks[f"{horizon}_{target}_component_max_abs_error"] = (
                maximum_absolute_difference(
                    panel[f"{stem}_correlation"], reconstructed
                )
            )
            expected_z = np.arctanh(
                np.clip(panel[f"{stem}_correlation"], -0.995, 0.995)
            )
            checks[f"{horizon}_{target}_fisher_max_abs_error"] = (
                maximum_absolute_difference(
                    panel[f"{stem}_fisher_z"], expected_z
                )
            )

    common_stock_variance_error = maximum_absolute_difference(
        panel["target_etf_t1_stock_realized_variance"],
        panel["target_loo_t1_stock_realized_variance"],
    )
    checks["t1_common_stock_variance_max_abs_error"] = (
        common_stock_variance_error
    )

    session_position = {date: position for position, date in enumerate(sessions)}
    expected_end = panel["forecast_date"].map(
        lambda value: (
            sessions[session_position[value] + 4]
            if value in session_position
            and session_position[value] + 4 < len(sessions)
            else pd.NaT
        )
    )
    for target in ("etf", "loo"):
        actual = pd.to_datetime(panel[f"target_{target}_t2_end_date"])
        valid = panel[f"target_{target}_t2_correlation"].notna()
        checks[f"t2_{target}_end_date_exact"] = bool(
            np.array_equal(
                actual[valid].to_numpy(dtype="datetime64[ns]"),
                pd.Series(expected_end[valid]).to_numpy(
                    dtype="datetime64[ns]"
                ),
            )
        )

    forward_errors: dict[str, float] = {}
    lag_errors: dict[str, float] = {}
    for stock, frame in panel.groupby("stock", sort=False):
        frame = frame.set_index("forecast_date").reindex(sessions)
        for target in ("etf", "loo"):
            columns = [
                f"target_{target}_t1_realized_covariance",
                f"target_{target}_t1_stock_realized_variance",
                f"target_{target}_t1_benchmark_realized_variance",
            ]
            expected = (
                frame[columns].iloc[::-1].rolling(5, min_periods=5).sum().iloc[::-1]
            )
            for source, destination in zip(
                columns,
                [
                    f"target_{target}_t2_realized_covariance",
                    f"target_{target}_t2_stock_realized_variance",
                    f"target_{target}_t2_benchmark_realized_variance",
                ],
                strict=True,
            ):
                key = f"{target}_{source}_{stock}"
                forward_errors[key] = maximum_absolute_difference(
                    expected[source], frame[destination]
                )
            lag_errors[f"{target}_{stock}"] = maximum_absolute_difference(
                frame[f"{target}_rth_rc_d"],
                frame[f"target_{target}_t1_correlation"].shift(1),
            )
    checks["t2_component_sum_max_abs_error"] = float(
        np.nanmax(list(forward_errors.values()))
    )
    checks["rth_lag1_max_abs_error"] = float(
        np.nanmax(list(lag_errors.values()))
    )
    checks["all_correlations_bounded"] = bool(
        all(
            panel[column].dropna().between(-1, 1).all()
            for column in panel
            if column.endswith("_correlation") or "_rc_" in column
        )
    )
    checks["valid_interval_counts"] = sorted(
        int(value)
        for value in panel.loc[
            panel["target_etf_t1_correlation"].notna(),
            "target_etf_t1_aligned_return_count",
        ].dropna().unique()
    )
    exact_boolean_checks = [
        value for value in checks.values() if isinstance(value, bool)
    ]
    numeric_errors = [
        value
        for key, value in checks.items()
        if key.endswith(("max_abs_error", "error"))
    ]
    passed = all(exact_boolean_checks) and all(
        np.isfinite(value) and value <= 1e-12 for value in numeric_errors
    )
    return {
        "status": "passed" if passed else "failed",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "checks": checks,
    }


def parser() -> argparse.ArgumentParser:
    output = argparse.ArgumentParser(description=__doc__)
    output.add_argument(
        "--targets",
        type=Path,
        default=Path(
            "data/features/quant/training_v1/"
            "correlation_targets_and_loo_features.parquet"
        ),
    )
    output.add_argument(
        "--calendar",
        type=Path,
        default=Path(
            "data/prices/alpaca/calendar/2016-01-01_2026-06-30.json"
        ),
    )
    output.add_argument(
        "--output",
        type=Path,
        default=Path(
            "experiments/quant_training/v1/construction/audits/"
            "target_integrity.json"
        ),
    )
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    report = audit_targets(
        pd.read_parquet(args.targets),
        core.load_official_sessions(args.calendar),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    core.write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
