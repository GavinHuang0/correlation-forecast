"""Shared, leakage-audited utilities for the locked training ladder."""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    enet_path,
    lasso_path,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_bollerslev_core_features as core  # noqa: E402
from scripts.correlation_training import build_modeling_panel as panel_schema  # noqa: E402


PROTOCOL_PATH = Path("config/quant_training_protocol_v1.json")
PANEL_PATH = Path("data/features/quant/training_v1/modeling_panel.parquet")
EXPERIMENT_ROOT = Path("experiments/quant_training/v1")
OUTPUT_ROOT = Path("outputs/quant_training/v1")
DCC_CALENDAR_PATH = Path(
    "data/prices/alpaca/calendar/2016-01-01_2026-06-30.json"
)


@dataclass(frozen=True)
class TargetSpec:
    name: str
    horizon: str
    benchmark: str
    response_column: str
    correlation_column: str
    target_end_column: str | None
    persistence_column: str
    core_features: tuple[str, ...]
    har_features: tuple[str, ...]
    shar_features: tuple[str, ...]
    history_stock_return: str
    history_benchmark_return: str


@dataclass(frozen=True)
class TrainingPaths:
    panel: Path
    dcc_history_panel: Path
    dcc_calendar: Path
    experiment_root: Path
    output_root: Path


def target_specs() -> tuple[TargetSpec, ...]:
    specs = []
    for horizon in ("t1", "t2"):
        for benchmark in ("etf", "loo"):
            pair_prefix = benchmark
            har = tuple(f"{pair_prefix}_rc_{label}" for label in ("d", "w", "m"))
            shar = (
                *har,
                *(
                    f"{pair_prefix}_rc_negative_{label}"
                    for label in ("d", "w", "m")
                ),
            )
            core_features = (
                panel_schema.ETF_PAIR_FEATURES
                if benchmark == "etf"
                else panel_schema.LOO_PAIR_FEATURES
            )
            specs.append(
                TargetSpec(
                    name=f"{horizon}_{benchmark}",
                    horizon=horizon,
                    benchmark=benchmark,
                    response_column=f"target_{benchmark}_{horizon}_fisher_z",
                    correlation_column=(
                        f"target_{benchmark}_{horizon}_correlation"
                    ),
                    target_end_column=(
                        f"target_{benchmark}_t2_end_date"
                        if horizon == "t2"
                        else None
                    ),
                    persistence_column=(
                        f"{benchmark}_rth_rc_d"
                        if horizon == "t1"
                        else f"{benchmark}_rth_rc_w"
                    ),
                    core_features=tuple(
                        [*core_features, *panel_schema.SECTOR_STATE_FEATURES]
                    ),
                    har_features=har,
                    shar_features=tuple(shar),
                    history_stock_return=(
                        f"history_{benchmark}_stock_rth_log_return_lag1"
                    ),
                    history_benchmark_return=(
                        f"history_{benchmark}_benchmark_rth_log_return_lag1"
                    ),
                )
            )
    return tuple(specs)


def load_protocol(path: Path = PROTOCOL_PATH) -> dict[str, object]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if not protocol.get(
        "initial_model_families_folds_and_feature_blocks_"
        "locked_before_outer_test_results"
    ):
        raise ValueError("Training protocol is not locked")
    return protocol


def resolve_training_paths(
    protocol: Mapping[str, object],
    *,
    panel: Path | None = None,
    experiment_root: Path | None = None,
    output_root: Path | None = None,
) -> TrainingPaths:
    """Resolve run-specific artifacts while preserving the v1 defaults."""

    configured = protocol.get("artifact_paths", {})
    if configured is None:
        configured = {}
    if not isinstance(configured, Mapping):
        raise ValueError("artifact_paths must be a mapping when provided")

    def selected(
        override: Path | None,
        key: str,
        fallback: Path,
    ) -> Path:
        if override is not None:
            return Path(override)
        value = configured.get(key)
        return Path(value) if value is not None else fallback

    selected_panel = selected(panel, "panel", PANEL_PATH)
    return TrainingPaths(
        panel=selected_panel,
        dcc_history_panel=selected(
            None, "dcc_history_panel", selected_panel
        ),
        dcc_calendar=selected(
            None, "dcc_calendar", DCC_CALENDAR_PATH
        ),
        experiment_root=selected(
            experiment_root, "experiment_root", EXPERIMENT_ROOT
        ),
        output_root=selected(output_root, "output_root", OUTPUT_ROOT),
    )


def load_panel(path: Path = PANEL_PATH) -> pd.DataFrame:
    panel = pd.read_parquet(path)
    panel["forecast_date"] = pd.to_datetime(panel["forecast_date"]).dt.normalize()
    for column in [
        name for name in panel if name.endswith("_end_date")
    ]:
        panel[column] = pd.to_datetime(panel[column]).dt.normalize()
    if panel.duplicated(["stock", "forecast_date"]).any():
        raise ValueError("Modeling panel has duplicate stock-date rows")
    return panel.sort_values(["forecast_date", "sector", "stock"]).reset_index(
        drop=True
    )


def split_masks(
    panel: pd.DataFrame,
    spec: TargetSpec,
    fold: Mapping[str, str],
) -> dict[str, pd.Series]:
    # Every model for a target is evaluated on the same rows as its declared
    # persistence benchmark. This was automatically satisfied in the v1
    # matched period, while earlier history contains a few valid targets whose
    # lagged benchmark is unavailable after an input gap.
    target_valid = (
        panel[spec.response_column].notna()
        & panel[spec.persistence_column].notna()
    )
    masks: dict[str, pd.Series] = {}
    for block, start_key, end_key in (
        ("train", "train_start", "train_end"),
        ("validation", "validation_start", "validation_end"),
        ("test", "test_start", "test_end"),
    ):
        start = pd.Timestamp(fold[start_key])
        end = pd.Timestamp(fold[end_key])
        mask = (
            target_valid
            & panel["forecast_date"].between(start, end, inclusive="both")
        )
        if spec.target_end_column is not None:
            mask &= panel[spec.target_end_column].notna()
            mask &= panel[spec.target_end_column].le(end)
        masks[block] = mask
    if any(
        (masks[left] & masks[right]).any()
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
    ):
        raise AssertionError("Chronological blocks overlap")
    return masks


def validate_feature_columns(features: Sequence[str]) -> None:
    if not features or len(features) != len(set(features)):
        raise ValueError("Feature list must be nonempty and unique")
    forbidden = [
        column
        for column in features
        if column.startswith("target_")
        or column.endswith("_end_date")
        or "realized_covariance" in column
        or "target_" in column
    ]
    if forbidden:
        raise ValueError(f"Target leakage columns requested: {forbidden}")


def validate_panel_information_set(panel: pd.DataFrame) -> bool:
    """Verify structural timing invariants shared by every trained rung."""

    required = {"stock", "forecast_date", "asof_session"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"Modeling panel lacks timing columns: {sorted(missing)}")
    if panel.duplicated(["stock", "forecast_date"]).any():
        raise ValueError("Modeling panel contains duplicate stock-date rows")
    forecast_dates = pd.to_datetime(panel["forecast_date"])
    asof_sessions = pd.to_datetime(panel["asof_session"])
    if asof_sessions.isna().any() or not (asof_sessions < forecast_dates).all():
        raise ValueError("Regular-session features do not end before forecast date")
    return True


def apply_fixed_log1p_transforms(
    frame: pd.DataFrame, features: Sequence[str]
) -> pd.DataFrame:
    """Apply the protocol's non-learned transform to nonnegative quantities."""

    output = frame.copy()
    for column in set(features).intersection(panel_schema.LOG1P_FEATURES):
        output[column] = output[column].astype(float)
        valid = output[column].notna()
        if (output.loc[valid, column] < 0).any():
            raise ValueError(f"{column} contains negative values before log1p")
        output.loc[valid, column] = np.log1p(
            output.loc[valid, column].astype(float)
        )
    return output


def linear_pipeline(kind: str, **parameters: float) -> Pipeline:
    if kind == "ols":
        estimator = LinearRegression()
    elif kind == "lasso":
        estimator = Lasso(
            max_iter=100_000,
            precompute=True,
            selection="random",
            random_state=1729,
            **parameters,
        )
    elif kind == "elastic_net":
        estimator = ElasticNet(
            max_iter=100_000,
            precompute=True,
            selection="random",
            random_state=1729,
            **parameters,
        )
    else:
        raise ValueError(f"Unknown linear estimator {kind}")
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=False,
                    keep_empty_features=True,
                ),
            ),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


def fisher_to_correlation(values: np.ndarray | pd.Series) -> np.ndarray:
    return np.tanh(np.asarray(values, dtype=float))


def correlation_to_fisher(values: np.ndarray | pd.Series) -> np.ndarray:
    return np.arctanh(np.clip(np.asarray(values, dtype=float), -0.995, 0.995))


def mse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.square(actual - predicted)))


def select_linear_hyperparameters(
    kind: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: Sequence[str],
    response: str,
    *,
    alpha_grid: Sequence[float],
    l1_ratios: Sequence[float] = (1.0,),
) -> tuple[dict[str, float], list[dict[str, float]]]:
    validate_feature_columns(features)
    imputer = SimpleImputer(
        strategy="median",
        add_indicator=False,
        keep_empty_features=True,
    )
    scaler = StandardScaler()
    train_imputed = imputer.fit_transform(train[list(features)])
    validation_imputed = imputer.transform(validation[list(features)])
    train_matrix = scaler.fit_transform(train_imputed)
    validation_matrix = scaler.transform(validation_imputed)
    train_response = train[response].to_numpy(dtype=float)
    response_mean = float(train_response.mean())
    centered_response = train_response - response_mean
    validation_response = validation[response].to_numpy(dtype=float)
    requested_alphas = np.asarray(
        sorted({float(value) for value in alpha_grid}, reverse=True)
    )
    ratios = l1_ratios if kind == "elastic_net" else (1.0,)
    candidates: list[dict[str, float]] = []

    def evaluate_path(alphas: np.ndarray, *, boundary_expansion: bool) -> None:
        for l1_ratio in ratios:
            if kind == "lasso":
                path_alphas, coefficients, _, iterations = lasso_path(
                    train_matrix,
                    centered_response,
                    alphas=alphas,
                    precompute=True,
                    max_iter=100_000,
                    return_n_iter=True,
                )
            elif kind == "elastic_net":
                path_alphas, coefficients, _, iterations = enet_path(
                    train_matrix,
                    centered_response,
                    l1_ratio=float(l1_ratio),
                    alphas=alphas,
                    precompute=True,
                    max_iter=100_000,
                    return_n_iter=True,
                )
            else:
                raise ValueError(f"Path selection is unsupported for {kind}")
            for position, alpha in enumerate(path_alphas):
                prediction = (
                    response_mean
                    + validation_matrix @ coefficients[:, position]
                )
                parameters = {"alpha": float(alpha)}
                if kind == "elastic_net":
                    parameters["l1_ratio"] = float(l1_ratio)
                candidates.append(
                    {
                        **parameters,
                        "validation_mse": mse(
                            validation_response, prediction
                        ),
                        "path_iterations": int(iterations[position]),
                        "boundary_expansion": boundary_expansion,
                    }
                )

    evaluate_path(requested_alphas, boundary_expansion=False)
    initial_best = min(candidates, key=lambda item: item["validation_mse"])
    minimum_alpha = float(requested_alphas.min())
    maximum_alpha = float(requested_alphas.max())
    expanded: list[float] = []
    if np.isclose(initial_best["alpha"], minimum_alpha):
        expanded.extend([minimum_alpha / 10, minimum_alpha / 3])
    if np.isclose(initial_best["alpha"], maximum_alpha):
        expanded.extend([maximum_alpha * 3, maximum_alpha * 10])
    if expanded:
        evaluate_path(
            np.asarray(sorted(set(expanded), reverse=True)),
            boundary_expansion=True,
        )
    candidates.sort(
        key=lambda item: (
            item["validation_mse"],
            item["alpha"],
            item.get("l1_ratio", 1.0),
        )
    )
    best = {
        key: value
        for key, value in candidates[0].items()
        if key not in {
            "validation_mse",
            "path_iterations",
            "boundary_expansion",
        }
    }
    return best, candidates


def fit_predict_linear(
    kind: str,
    train_and_validation: pd.DataFrame,
    test: pd.DataFrame,
    features: Sequence[str],
    response: str,
    parameters: Mapping[str, float] | None = None,
) -> tuple[np.ndarray, Pipeline]:
    validate_feature_columns(features)
    model = linear_pipeline(kind, **dict(parameters or {}))
    model.fit(train_and_validation[list(features)], train_and_validation[response])
    return model.predict(test[list(features)]), model


def prediction_frame(
    test: pd.DataFrame,
    spec: TargetSpec,
    *,
    fold_name: str,
    model_name: str,
    predicted_fisher: np.ndarray,
) -> pd.DataFrame:
    output = test[
        [
            "forecast_date",
            "sector",
            "stock",
            "benchmark",
            spec.response_column,
            spec.correlation_column,
            spec.persistence_column,
        ]
    ].copy()
    output = output.rename(
        columns={
            spec.response_column: "actual_fisher_z",
            spec.correlation_column: "actual_correlation",
        }
    )
    output["predicted_fisher_z"] = np.asarray(predicted_fisher, dtype=float)
    output["predicted_correlation"] = fisher_to_correlation(predicted_fisher)
    output["persistence_correlation"] = output[spec.persistence_column]
    output["persistence_fisher_z"] = correlation_to_fisher(
        output["persistence_correlation"]
    )
    output.insert(0, "model", model_name)
    output.insert(0, "target", spec.name)
    output.insert(0, "fold", fold_name)
    return output.drop(columns=[spec.persistence_column])


def summarize_predictions(
    predictions: pd.DataFrame,
    group_columns: Sequence[str] = ("target", "model"),
) -> pd.DataFrame:
    rows = []
    for keys, frame in predictions.groupby(list(group_columns), sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        actual_z = frame["actual_fisher_z"].to_numpy(dtype=float)
        predicted_z = frame["predicted_fisher_z"].to_numpy(dtype=float)
        baseline_z = frame["persistence_fisher_z"].to_numpy(dtype=float)
        actual_rho = frame["actual_correlation"].to_numpy(dtype=float)
        predicted_rho = frame["predicted_correlation"].to_numpy(dtype=float)
        model_sse = np.sum(np.square(actual_z - predicted_z))
        baseline_sse = np.sum(np.square(actual_z - baseline_z))
        rows.append(
            {
                **dict(zip(group_columns, keys, strict=True)),
                "rows": len(frame),
                "fisher_z_rmse": math.sqrt(model_sse / len(frame)),
                "fisher_z_mae": float(np.mean(np.abs(actual_z - predicted_z))),
                "raw_correlation_rmse": float(
                    np.sqrt(np.mean(np.square(actual_rho - predicted_rho)))
                ),
                "raw_correlation_mae": float(
                    np.mean(np.abs(actual_rho - predicted_rho))
                ),
                "oos_r2_vs_persistence": (
                    float(1 - model_sse / baseline_sse)
                    if baseline_sse > 0
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def extract_linear_coefficients(
    pipeline: Pipeline, features: Sequence[str]
) -> list[dict[str, float]]:
    model = pipeline.named_steps["model"]
    if not hasattr(model, "coef_"):
        return []
    return [
        {"feature": feature, "coefficient_standardized": float(coefficient)}
        for feature, coefficient in zip(features, model.coef_, strict=True)
    ]


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    core.write_frame_atomic(path, frame)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    core.write_json_atomic(path, payload)
