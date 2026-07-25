"""Join audited core, context, target, and LOO feature artifacts one-to-one."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_bollerslev_core_features as core  # noqa: E402


KEYS = ["sector", "stock", "benchmark", "forecast_date"]
ETF_PAIR_FEATURES = [
    *(f"etf_rc_{label}" for label in core.HAR_HORIZONS),
    *(f"etf_rc_negative_{label}" for label in core.HAR_HORIZONS),
    *(f"etf_exp_rc_{label}" for label in core.EXPONENTIAL_CENTERS),
    *(f"etf_exp_rc_negative_{label}" for label in core.EXPONENTIAL_CENTERS),
]
LOO_PAIR_FEATURES = [
    *(f"loo_rc_{label}" for label in core.HAR_HORIZONS),
    *(f"loo_rc_negative_{label}" for label in core.HAR_HORIZONS),
    *(f"loo_exp_rc_{label}" for label in core.EXPONENTIAL_CENTERS),
    *(f"loo_exp_rc_negative_{label}" for label in core.EXPONENTIAL_CENTERS),
]
SECTOR_STATE_FEATURES = [
    *(f"sector_exp_rc_{label}" for label in core.EXPONENTIAL_CENTERS),
    *(
        f"sector_exp_rc_negative_{label}"
        for label in core.EXPONENTIAL_CENTERS
    ),
]
DENSE_CONTEXT_FEATURES = [
    "lagged_relative_daily_volume_20d",
    "sector_lagged_relative_daily_volume_20d",
    "lagged_sector_return_dispersion",
    "vix_lag1",
    "vix_change_lag1",
    "treasury_2y_lag2",
    "treasury_2y_change_lag2",
    "treasury_5y_lag2",
    "treasury_5y_change_lag2",
    "treasury_10y_lag2",
    "treasury_10y_change_lag2",
    "scheduled_macro_event_count",
    "scheduled_preopen_macro_count",
    "bls_release_day",
    "fomc_decision_day",
]
VOLATILITY_FEATURES = [
    "lagged_realized_volatility",
    "sector_lagged_realized_volatility",
]
EXTENDED_FEATURES = [
    "stock_overnight_return",
    "premarket_return",
    "premarket_volume",
    "relative_premarket_volume_20d",
    "premarket_bar_count",
    "sector_overnight_return",
    "sector_premarket_return",
    "sector_premarket_volume",
    "sector_relative_premarket_volume_20d",
    "sector_premarket_bar_count",
    "stock_minus_sector_overnight_return",
    "prior_aftermarket_return",
    "prior_aftermarket_volume",
    "prior_relative_aftermarket_volume_20d",
    "stock_premarket_available",
    "sector_premarket_available",
    "stock_prior_aftermarket_available",
]
LOG1P_FEATURES = [
    "lagged_relative_daily_volume_20d",
    "sector_lagged_relative_daily_volume_20d",
    "scheduled_macro_event_count",
    "scheduled_preopen_macro_count",
    "premarket_volume",
    "relative_premarket_volume_20d",
    "premarket_bar_count",
    "sector_premarket_volume",
    "sector_relative_premarket_volume_20d",
    "sector_premarket_bar_count",
    "prior_aftermarket_volume",
    "prior_relative_aftermarket_volume_20d",
]


def assert_unique(frame: pd.DataFrame, label: str) -> None:
    if frame.duplicated(KEYS).any():
        raise ValueError(f"{label} has duplicate modeling keys")


def build_modeling_panel(
    core_frame: pd.DataFrame,
    context_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    frames = {
        "core": core_frame.copy(),
        "context": context_frame.copy(),
        "targets": target_frame.copy(),
    }
    for label, frame in frames.items():
        frame["forecast_date"] = pd.to_datetime(
            frame["forecast_date"]
        ).dt.normalize()
        assert_unique(frame, label)

    pair_columns = [
        column
        for column in core.FEATURE_COLUMNS
        if not column.startswith("sector_")
    ]
    rename = {column: f"etf_{column}" for column in pair_columns}
    frames["core"] = frames["core"].rename(columns=rename)
    core_keep = [
        *KEYS,
        "asof_session",
        *ETF_PAIR_FEATURES,
        *SECTOR_STATE_FEATURES,
    ]
    missing_core = set(core_keep) - set(frames["core"])
    if missing_core:
        raise ValueError(f"Core artifact misses {sorted(missing_core)}")

    panel = frames["core"][core_keep].merge(
        frames["targets"],
        on=KEYS,
        how="inner",
        validate="one_to_one",
    )
    context_payload = [
        column for column in frames["context"] if column not in KEYS
    ]
    panel = panel.merge(
        frames["context"][[*KEYS, *context_payload]],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    panel = panel[
        panel["forecast_date"].between(start, end, inclusive="both")
    ].copy()
    panel = panel.sort_values(["forecast_date", "sector", "stock"]).reset_index(
        drop=True
    )
    assert_unique(panel, "modeling panel")
    if panel.empty:
        raise ValueError("No rows remain in the matched modeling period")
    for horizon in ("t1", "t2"):
        if not panel[f"target_etf_{horizon}_correlation"].notna().equals(
            panel[f"target_loo_{horizon}_correlation"].notna()
        ):
            raise AssertionError(f"{horizon} target eligibility differs")
    feature_groups = {
        "etf_core_22": [*ETF_PAIR_FEATURES, *SECTOR_STATE_FEATURES],
        "loo_core_22": [*LOO_PAIR_FEATURES, *SECTOR_STATE_FEATURES],
        "dense_context": DENSE_CONTEXT_FEATURES,
        "volatility": VOLATILITY_FEATURES,
        "extended_hours": EXTENDED_FEATURES,
    }
    for group, columns in feature_groups.items():
        missing = set(columns) - set(panel)
        if missing:
            raise ValueError(f"{group} misses {sorted(missing)}")
    return panel, feature_groups


def parser() -> argparse.ArgumentParser:
    output = argparse.ArgumentParser(description=__doc__)
    base = Path("data/features/quant/training_v1")
    output.add_argument(
        "--core", type=Path, default=base / "bollerslev_core_features.parquet"
    )
    output.add_argument(
        "--context", type=Path, default=base / "additional_quant_features.parquet"
    )
    output.add_argument(
        "--targets",
        type=Path,
        default=base / "correlation_targets_and_loo_features.parquet",
    )
    output.add_argument(
        "--output", type=Path, default=base / "modeling_panel.parquet"
    )
    output.add_argument("--start", type=core.parse_date, default="2022-11-01")
    output.add_argument("--end", type=core.parse_date, default="2026-06-30")
    output.add_argument(
        "--audit-output",
        type=Path,
        default=Path(
            "experiments/quant_training/v1/construction/audits/"
            "modeling_panel.json"
        ),
    )
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    panel, groups = build_modeling_panel(
        pd.read_parquet(args.core),
        pd.read_parquet(args.context),
        pd.read_parquet(args.targets),
        start=pd.Timestamp(args.start),
        end=pd.Timestamp(args.end),
    )
    core.write_frame_atomic(args.output, panel)
    coverage = {
        group: {
            column: int(panel[column].notna().sum()) for column in columns
        }
        for group, columns in groups.items()
    }
    audit = {
        "status": "complete",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "row_count": int(len(panel)),
        "stock_count": int(panel["stock"].nunique()),
        "date_count": int(panel["forecast_date"].nunique()),
        "first_date": panel["forecast_date"].min().date().isoformat(),
        "last_date": panel["forecast_date"].max().date().isoformat(),
        "feature_groups": groups,
        "coverage": coverage,
        "inputs": {
            "core": {
                "path": str(args.core),
                "sha256": core.sha256_file(args.core),
            },
            "context": {
                "path": str(args.context),
                "sha256": core.sha256_file(args.context),
            },
            "targets": {
                "path": str(args.targets),
                "sha256": core.sha256_file(args.targets),
            },
        },
        "output": {
            "path": str(args.output),
            "sha256": core.sha256_file(args.output),
        },
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    core.write_json_atomic(args.audit_output, audit)
    core.write_json_atomic(core.output_manifest_path(args.output), audit)
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
