"""Shared contracts and artifact utilities for Q+D training v1."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import shutil
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import enet_path
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.correlation_training import build_modeling_panel as quant_schema  # noqa: E402
from scripts.correlation_training import training_common as quant_common  # noqa: E402


PROTOCOL_PATH = Path("config/quant_deterministic_news_protocol_v1.json")
PANEL_PATH = Path(
    "data/features/q_plus_d/massive_v1/modeling_panel_q_plus_d.parquet"
)
MANIFEST_PATH = Path("data/features/q_plus_d/massive_v1/manifest.json")
EXPERIMENT_ROOT = Path("experiments/quant_deterministic_news/v1")
OUTPUT_ROOT = Path("outputs/quant_deterministic_news/v1")
STATUS_PATH = EXPERIMENT_ROOT / "status.json"
STATUS_MARKDOWN_PATH = EXPERIMENT_ROOT / "STATUS.md"

PREDICTION_KEYS = [
    "fold",
    "target",
    "forecast_date",
    "sector",
    "stock",
    "benchmark",
]
PANEL_KEYS = ["forecast_date", "sector", "stock", "benchmark"]


@dataclass(frozen=True)
class Bundle:
    order: int
    key: str
    track: str
    slug: str
    label: str

    @property
    def experiment_path(self) -> Path:
        return EXPERIMENT_ROOT / self.track / self.slug

    @property
    def output_path(self) -> Path:
        return OUTPUT_ROOT / self.track / self.slug


BUNDLES = (
    Bundle(0, "PRE", "construction", "preflight", "Construction preflight"),
    Bundle(1, "A0-L", "track_a_joint", "a0_l_q56_elastic_net", "A0-L Q56 Elastic Net"),
    Bundle(2, "A0-T", "track_a_joint", "a0_t_q56_xgboost", "A0-T Q56 XGBoost"),
    Bundle(3, "A1", "track_a_joint", "a1_d43_elastic_net", "A1 D43 Elastic Net"),
    Bundle(4, "A2", "track_a_joint", "a2_q56_d1_elastic_net", "A2 Q56+D1 Elastic Net"),
    Bundle(5, "A3", "track_a_joint", "a3_q56_d1_d2_elastic_net", "A3 Q56+D1+D2 Elastic Net"),
    Bundle(6, "A4", "track_a_joint", "a4_q56_d1_d2_d3_elastic_net", "A4 Q56+D1+D2+D3 Elastic Net"),
    Bundle(7, "A5", "track_a_joint", "a5_q56_d43_elastic_net", "A5 Q56+D43 Elastic Net"),
    Bundle(8, "A6", "track_a_joint", "a6_q56_d43_xgboost", "A6 Q56+D43 XGBoost"),
    Bundle(9, "A7", "track_a_joint", "a7_q56_d43_ensemble", "A7 Elastic Net/XGBoost ensemble"),
    Bundle(10, "B0", "track_b_residual", "b0_zero_correction", "B0 zero correction"),
    Bundle(11, "B1", "track_b_residual", "b1_mean_error", "B1 mean-error correction"),
    Bundle(12, "B2", "track_b_residual", "b2_linear_calibration", "B2 linear calibration"),
    Bundle(13, "B3", "track_b_residual", "b3_d43_elastic_net", "B3 D43 residual Elastic Net"),
    Bundle(14, "B4", "track_b_residual", "b4_forecast_d43_elastic_net", "B4 forecast+D43 residual Elastic Net"),
    Bundle(15, "B5", "track_b_residual", "b5_forecast_d43_xgboost", "B5 forecast+D43 residual XGBoost"),
    Bundle(16, "S-B0", "track_b_winner_sensitivity", "b0_zero_correction", "Winner-base B0 sensitivity"),
    Bundle(17, "S-B1", "track_b_winner_sensitivity", "b1_mean_error", "Winner-base B1 sensitivity"),
    Bundle(18, "S-B2", "track_b_winner_sensitivity", "b2_linear_calibration", "Winner-base B2 sensitivity"),
    Bundle(19, "S-B3", "track_b_winner_sensitivity", "b3_d43_elastic_net", "Winner-base B3 sensitivity"),
    Bundle(20, "S-B4", "track_b_winner_sensitivity", "b4_forecast_d43_elastic_net", "Winner-base B4 sensitivity"),
    Bundle(21, "S-B5", "track_b_winner_sensitivity", "b5_forecast_d43_xgboost", "Winner-base B5 sensitivity"),
    Bundle(22, "C-A5-L20", "controls", "a5_stale_news_lag20", "A5 stale-news control"),
    Bundle(23, "C-A5-WS", "controls", "a5_wrong_stock", "A5 wrong-stock control"),
    Bundle(24, "C-B4-L20", "controls", "b4_stale_news_lag20", "B4 stale-news control"),
    Bundle(25, "C-B4-WS", "controls", "b4_wrong_stock", "B4 wrong-stock control"),
    Bundle(26, "FINAL", "comparisons", "final", "Final comparison and inference"),
)
BUNDLE_BY_KEY = {bundle.key: bundle for bundle in BUNDLES}


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def ordered_sha256(values: Sequence[str]) -> str:
    return sha256_json(list(values))


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def load_protocol(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    protocol = load_json(path)
    if protocol.get("status") != "locked_before_training":
        raise ValueError("Q+D protocol must be locked before training")
    if protocol.get("experiment_id") != "quant-deterministic-news-v1":
        raise ValueError("Unexpected experiment ID")
    return protocol


def xgboost_candidate_configs(
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    configured = protocol["model_tuning"]["xgboost_candidates"]
    if isinstance(configured, list):
        candidates = [dict(value) for value in configured]
    elif configured == (
        "reuse config/quant_training_protocol_v1.json rung_3 "
        "candidate_configs"
    ):
        quant_protocol_path = Path(protocol["isolation"]["quant_protocol"])
        winner_summary_path = Path(
            protocol["isolation"]["quant_winner_summary"]
        )
        winner_summary = load_json(winner_summary_path)
        expected_hash = winner_summary["artifact_sha256"][
            quant_protocol_path.as_posix()
        ]
        _verify_hash(
            quant_protocol_path,
            expected_hash,
            "referenced quant-v1 protocol",
        )
        quant_protocol = load_json(quant_protocol_path)
        candidates = [
            dict(value)
            for value in quant_protocol["rung_3"]["candidate_configs"]
        ]
    else:
        raise TypeError("Unsupported XGBoost candidate configuration")
    required = {
        "max_depth",
        "learning_rate",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "reg_lambda",
    }
    if len(candidates) != 4 or any(
        set(candidate) != required for candidate in candidates
    ):
        raise ValueError("Resolved XGBoost candidates fail the quant-v1 schema")
    return candidates


def feature_blocks(protocol: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    raw = protocol["feature_blocks"]
    return {key: tuple(value) for key, value in raw.items()}


def deterministic_features(protocol: Mapping[str, Any]) -> tuple[str, ...]:
    blocks = feature_blocks(protocol)
    values = (
        *blocks["d_activity_scope_18"],
        *blocks["d_cues_8"],
        *blocks["d_timing_6"],
        *blocks["d_source_title_7"],
        *blocks["d_burst_1"],
        *blocks["d_missingness_3"],
    )
    if len(values) != 43 or len(set(values)) != 43:
        raise ValueError("D43 feature contract is not unique")
    return tuple(values)


def materialized_deterministic_features(
    protocol: Mapping[str, Any],
) -> tuple[str, ...]:
    return tuple(
        feature
        for feature in deterministic_features(protocol)
        if not feature.startswith("missing_")
    )


def quant_features(
    spec: quant_common.TargetSpec, protocol: Mapping[str, Any]
) -> tuple[str, ...]:
    blocks = feature_blocks(protocol)
    pair = (
        blocks["q_pair_etf_14"]
        if spec.benchmark == "etf"
        else blocks["q_pair_loo_14"]
    )
    values = (
        *pair,
        *blocks["q_sector_state_8"],
        *blocks["q_dense_15"],
        *blocks["q_volatility_2"],
        *blocks["q_extended_17"],
    )
    if len(values) != 56 or len(set(values)) != 56:
        raise ValueError(f"Q56 contract failed for {spec.name}")
    return tuple(values)


def validate_model_features(features: Sequence[str]) -> None:
    if not features or len(features) != len(set(features)):
        raise ValueError("Feature list must be nonempty and unique")
    forbidden_exact = {
        "sector",
        "stock",
        "benchmark",
        "forecast_date",
        "asof_session",
        "news_row_matched",
        "archive_scope",
        "point_in_time_version_safe",
        "primary_training_eligible",
    }
    forbidden = [
        column
        for column in features
        if column in forbidden_exact
        or column.startswith("target_etf_")
        or column.startswith("target_loo_")
        or column.endswith("_end_date")
        or "realized_covariance" in column
        or "realized_variance" in column
        or "aligned_return_count" in column
        or "expected_return_count" in column
    ]
    if forbidden:
        raise ValueError(f"Target/audit leakage columns requested: {forbidden}")


def joint_features(
    bundle_key: str,
    spec: quant_common.TargetSpec,
    protocol: Mapping[str, Any],
) -> tuple[str, ...]:
    blocks = feature_blocks(protocol)
    q = quant_features(spec, protocol)
    d1 = blocks["d_activity_scope_18"]
    d2 = blocks["d_cues_8"]
    d3 = blocks["d_timing_6"]
    d4 = (*blocks["d_source_title_7"], *blocks["d_burst_1"])
    missing = blocks["d_missingness_3"]
    if bundle_key in {"A0-L", "A0-T"}:
        values = q
    elif bundle_key == "A1":
        values = (*d1, *d2, *d3, *d4, *missing)
    elif bundle_key == "A2":
        values = (*q, *d1)
    elif bundle_key == "A3":
        values = (*q, *d1, *d2)
    elif bundle_key == "A4":
        values = (*q, *d1, *d2, *d3, *missing[:2])
    elif bundle_key in {
        "A5",
        "A6",
        "C-A5-L20",
        "C-A5-WS",
    }:
        values = (*q, *d1, *d2, *d3, *d4, *missing)
    else:
        raise KeyError(f"No direct joint features for {bundle_key}")
    values = tuple(values)
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate features for {bundle_key}/{spec.name}")
    validate_model_features(values)
    return values


def add_missingness_indicators(
    panel: pd.DataFrame, protocol: Mapping[str, Any]
) -> pd.DataFrame:
    output = panel.copy()
    mapping = {
        "missing_hours_since_latest_precise_target_article":
            "hours_since_latest_precise_target_article",
        "missing_hours_since_latest_precise_common_article":
            "hours_since_latest_precise_common_article",
        "missing_observed_target_news_burst_60_session":
            "observed_target_news_burst_60_session",
    }
    expected = set(feature_blocks(protocol)["d_missingness_3"])
    if set(mapping) != expected:
        raise ValueError("Missingness indicator mapping differs from protocol")
    for indicator, source in mapping.items():
        output[indicator] = output[source].isna().astype(np.int8)
    return output


def apply_fixed_transforms(
    frame: pd.DataFrame,
    features: Sequence[str],
    protocol: Mapping[str, Any],
) -> pd.DataFrame:
    output = frame.copy()
    requested = set(features)
    configured = set(quant_schema.LOG1P_FEATURES)
    configured.update(protocol["preprocessing"]["fixed_log1p"]["deterministic"])
    for column in sorted(requested.intersection(configured)):
        output[column] = output[column].astype(float)
        valid = output[column].notna()
        if (output.loc[valid, column] < 0).any():
            raise ValueError(f"{column} contains negative values before log1p")
        output.loc[valid, column] = np.log1p(output.loc[valid, column])
    return output


def select_linear_hyperparameters(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: Sequence[str],
    response: str,
    *,
    alpha_grid: Sequence[float],
    l1_ratios: Sequence[float],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Quant-v1 Elastic Net selection with the Q+D leakage allowlist."""

    validate_model_features(features)
    imputer = SimpleImputer(strategy="median", add_indicator=False)
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
    candidates: list[dict[str, Any]] = []

    def evaluate_path(
        alphas: np.ndarray, *, boundary_expansion: bool
    ) -> None:
        for l1_ratio in l1_ratios:
            path_alphas, coefficients, _, iterations = enet_path(
                train_matrix,
                centered_response,
                l1_ratio=float(l1_ratio),
                alphas=alphas,
                precompute=True,
                max_iter=100_000,
                return_n_iter=True,
            )
            for position, alpha in enumerate(path_alphas):
                prediction = (
                    response_mean
                    + validation_matrix @ coefficients[:, position]
                )
                candidates.append(
                    {
                        "alpha": float(alpha),
                        "l1_ratio": float(l1_ratio),
                        "validation_mse": quant_common.mse(
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
            item["l1_ratio"],
        )
    )
    best = {
        "alpha": float(candidates[0]["alpha"]),
        "l1_ratio": float(candidates[0]["l1_ratio"]),
    }
    return best, candidates


def fit_predict_linear(
    kind: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: Sequence[str],
    response: str,
    parameters: Mapping[str, float] | None = None,
) -> tuple[np.ndarray, Any]:
    validate_model_features(features)
    model = quant_common.linear_pipeline(kind, **dict(parameters or {}))
    model.fit(train[list(features)], train[response])
    return model.predict(test[list(features)]), model


def _verify_hash(path: Path, expected: str, label: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(
            f"{label} hash mismatch: expected {expected}, observed {observed}"
        )


def load_panel_and_preflight(
    protocol: Mapping[str, Any],
    *,
    allow_exploratory: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not allow_exploratory:
        raise ValueError(
            "This source is non-version-safe; pass --allow-exploratory"
        )
    sources = protocol["source_artifacts"]
    _verify_hash(PANEL_PATH, sources["joined_panel_sha256"], "Q+D panel")
    _verify_hash(
        MANIFEST_PATH, sources["joined_manifest_sha256"], "Q+D manifest"
    )
    for path_key, hash_key in (
        ("quant_rung_2_predictions", "quant_rung_2_predictions_sha256"),
        ("quant_rung_3_predictions", "quant_rung_3_predictions_sha256"),
        ("quant_winner_summary", "quant_winner_summary_sha256"),
    ):
        _verify_hash(
            Path(protocol["isolation"][path_key]),
            sources[hash_key],
            path_key,
        )
    manifest = load_json(MANIFEST_PATH)
    panel = quant_common.load_panel(PANEL_PATH)
    panel = add_missingness_indicators(panel, protocol)
    expected_shape = (
        int(sources["row_count"]),
        int(sources["date_count"]),
        int(sources["stock_count"]),
        int(sources["sector_count"]),
    )
    observed_shape = (
        len(panel),
        panel["forecast_date"].nunique(),
        panel["stock"].nunique(),
        panel["sector"].nunique(),
    )
    if observed_shape != expected_shape:
        raise ValueError(
            f"Panel coverage mismatch: {observed_shape} != {expected_shape}"
        )
    if panel.duplicated(PANEL_KEYS).any():
        raise ValueError("Q+D panel has duplicate join keys")
    if not panel["news_row_matched"].eq(1).all():
        raise ValueError("Not every quant row has a news match")
    for column in (
        "ordinary_ticker_news_collection_complete",
        "target_ticker_query_complete",
        "sector_stock_ticker_queries_complete",
        "sector_benchmark_query_complete",
        "control_query_complete",
    ):
        if not panel[column].eq(1).all():
            raise ValueError(f"Collection completeness failed for {column}")
    if panel["point_in_time_version_safe"].ne(0).any():
        raise ValueError("Unexpected version-safe rows in exploratory panel")
    if panel["primary_training_eligible"].ne(0).any():
        raise ValueError("Unexpected primary-training-eligible rows")
    recommended = tuple(
        manifest["modeling_preflight"]["recommended_feature_columns"]
    )
    if set(recommended) != set(
        materialized_deterministic_features(protocol)
    ):
        raise ValueError("Protocol D40 differs from joined manifest")
    all_features: set[str] = set(deterministic_features(protocol))
    for spec in quant_common.target_specs():
        all_features.update(quant_features(spec, protocol))
    missing_columns = all_features.difference(panel.columns)
    if missing_columns:
        raise ValueError(f"Panel misses model features: {sorted(missing_columns)}")
    quant_common.validate_panel_information_set(panel)
    counts = {}
    for spec in quant_common.target_specs():
        counts[spec.name] = int(panel[spec.response_column].notna().sum())
    expected_counts = {
        "t1_etf": 27_510,
        "t1_loo": 27_510,
        "t2_etf": 27_390,
        "t2_loo": 27_390,
    }
    if counts != expected_counts:
        raise ValueError(f"Target coverage mismatch: {counts}")
    audit = {
        "status": "passed",
        "generated_at_utc": utc_now(),
        "protocol_path": PROTOCOL_PATH.as_posix(),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "panel_path": PANEL_PATH.as_posix(),
        "panel_sha256": sha256_file(PANEL_PATH),
        "manifest_path": MANIFEST_PATH.as_posix(),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "rows": len(panel),
        "dates": int(panel["forecast_date"].nunique()),
        "stocks": int(panel["stock"].nunique()),
        "sectors": int(panel["sector"].nunique()),
        "target_nonnull_rows": counts,
        "d40_matches_manifest": True,
        "derived_missingness_flags": list(
            feature_blocks(protocol)["d_missingness_3"]
        ),
        "claim_scope": protocol["claim_scope"],
        "exploratory_override_confirmed": True,
    }
    return panel, audit


def load_frozen_predictions(
    protocol: Mapping[str, Any],
    family: str,
    *,
    target: str | None = None,
) -> pd.DataFrame:
    if family == "elastic_net":
        path = Path(protocol["isolation"]["quant_rung_2_predictions"])
        expected_hash = protocol["source_artifacts"][
            "quant_rung_2_predictions_sha256"
        ]
        model = "core_dense_volatility_extended_elastic_net"
    elif family == "xgboost":
        path = Path(protocol["isolation"]["quant_rung_3_predictions"])
        expected_hash = protocol["source_artifacts"][
            "quant_rung_3_predictions_sha256"
        ]
        model = "xgboost"
    elif family == "winner_t1_etf":
        path = Path(protocol["isolation"]["quant_rung_3_predictions"])
        expected_hash = protocol["source_artifacts"][
            "quant_rung_3_predictions_sha256"
        ]
        model = "elastic_xgboost_ensemble"
        target = "t1_etf"
    else:
        raise ValueError(f"Unknown frozen prediction family {family}")
    _verify_hash(path, expected_hash, f"{family} frozen predictions")
    predictions = pd.read_parquet(path)
    predictions["forecast_date"] = pd.to_datetime(
        predictions["forecast_date"]
    ).dt.normalize()
    predictions = predictions[predictions["model"].eq(model)].copy()
    if target is not None:
        predictions = predictions[predictions["target"].eq(target)].copy()
    if predictions.empty:
        raise ValueError(f"No frozen predictions found for {family}")
    if predictions.duplicated(PREDICTION_KEYS).any():
        raise ValueError(f"Frozen {family} predictions have duplicate keys")
    predictions["base_model"] = model
    predictions["base_artifact_sha256"] = expected_hash
    return predictions.sort_values(PREDICTION_KEYS).reset_index(drop=True)


def validate_frozen_actuals(
    predictions: pd.DataFrame, panel: pd.DataFrame
) -> None:
    spec_by_name = {spec.name: spec for spec in quant_common.target_specs()}
    for target, frame in predictions.groupby("target", sort=True):
        spec = spec_by_name[target]
        merged = frame.merge(
            panel[[*PANEL_KEYS, spec.response_column]],
            on=PANEL_KEYS,
            how="left",
            validate="many_to_one",
        )
        delta = (
            merged["actual_fisher_z"] - merged[spec.response_column]
        ).abs()
        if delta.isna().any() or float(delta.max()) > 1e-12:
            raise ValueError(f"Frozen actual target mismatch for {target}")


def parity_statistics(
    replay: pd.DataFrame, frozen: pd.DataFrame
) -> dict[str, Any]:
    left = replay.copy()
    right = frozen.copy()
    merged = left.merge(
        right[
            [
                *PREDICTION_KEYS,
                "predicted_fisher_z",
                "predicted_correlation",
            ]
        ],
        on=PREDICTION_KEYS,
        how="outer",
        validate="one_to_one",
        suffixes=("_replay", "_frozen"),
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("Replay and frozen prediction keys differ")
    fisher_delta = (
        merged["predicted_fisher_z_replay"]
        - merged["predicted_fisher_z_frozen"]
    ).abs()
    correlation_delta = (
        merged["predicted_correlation_replay"]
        - merged["predicted_correlation_frozen"]
    ).abs()
    return {
        "rows": len(merged),
        "maximum_absolute_fisher_z_difference": float(fisher_delta.max()),
        "mean_absolute_fisher_z_difference": float(fisher_delta.mean()),
        "maximum_absolute_correlation_difference": float(
            correlation_delta.max()
        ),
    }


def prepare_control_panel(
    panel: pd.DataFrame,
    protocol: Mapping[str, Any],
    control: str | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if control is None:
        output = panel.copy()
        output["_control_available"] = True
        return output, {"control": None}
    d43 = list(deterministic_features(protocol))
    ordered = panel.sort_values(["stock", "forecast_date"]).copy()
    if control == "lag20":
        shifted = ordered.groupby("stock", sort=False)[d43].shift(20)
        for feature in d43:
            ordered[feature] = shifted[feature].to_numpy()
        ordered["_control_available"] = (
            ordered.groupby("stock", sort=False).cumcount() >= 20
        )
        return ordered.sort_values(
            ["forecast_date", "sector", "stock"]
        ).reset_index(drop=True), {
            "control": "20-session within-stock stale-news shift",
            "sessions": 20,
            "wrapped": False,
        }
    if control == "wrong_stock":
        sector_stocks = {
            sector: sorted(group["stock"].unique().tolist())
            for sector, group in panel.groupby("sector", sort=True)
        }
        mapping: dict[tuple[str, str], str] = {}
        for sector, stocks in sector_stocks.items():
            if len(stocks) != 6:
                raise ValueError(
                    f"Expected six stocks in {sector}, found {len(stocks)}"
                )
            for position, stock in enumerate(stocks):
                mapping[(sector, stock)] = stocks[(position + 1) % len(stocks)]
        base = panel.copy()
        base["_donor_stock"] = [
            mapping[(sector, stock)]
            for sector, stock in zip(
                base["sector"], base["stock"], strict=True
            )
        ]
        donor = panel[
            ["forecast_date", "sector", "stock", *d43]
        ].rename(
            columns={
                "stock": "_donor_stock",
                **{feature: f"_donor_{feature}" for feature in d43},
            }
        )
        base = base.merge(
            donor,
            on=["forecast_date", "sector", "_donor_stock"],
            how="left",
            validate="many_to_one",
        )
        for feature in d43:
            base[feature] = base.pop(f"_donor_{feature}")
        if base[d43].isna().all(axis=1).any():
            raise ValueError("Wrong-stock donor join produced empty D43 rows")
        base["_control_available"] = True
        base = base.drop(columns=["_donor_stock"])
        return base.sort_values(
            ["forecast_date", "sector", "stock"]
        ).reset_index(drop=True), {
            "control": "fixed within-sector same-date wrong-stock rotation",
            "mapping": {
                f"{sector}:{stock}": donor_stock
                for (sector, stock), donor_stock in sorted(mapping.items())
            },
        }
    raise ValueError(f"Unknown control {control}")


def split_masks(
    panel: pd.DataFrame,
    spec: quant_common.TargetSpec,
    fold: Mapping[str, str],
) -> dict[str, pd.Series]:
    masks = quant_common.split_masks(panel, spec, fold)
    if "_control_available" in panel:
        masks = {
            block: mask & panel["_control_available"].fillna(False)
            for block, mask in masks.items()
        }
    return masks


def summarize_with_base(
    predictions: pd.DataFrame, base: pd.DataFrame
) -> list[dict[str, Any]]:
    merged = predictions.merge(
        base[
            [
                *PREDICTION_KEYS,
                "actual_fisher_z",
                "predicted_fisher_z",
            ]
        ],
        on=PREDICTION_KEYS,
        how="inner",
        validate="one_to_one",
        suffixes=("_model", "_base"),
    )
    if len(merged) != len(predictions):
        raise ValueError("Model/base comparison keys are incomplete")
    actual_delta = (
        merged["actual_fisher_z_model"]
        - merged["actual_fisher_z_base"]
    ).abs()
    if float(actual_delta.max()) > 1e-12:
        raise ValueError("Model/base actual targets differ")
    rows = []
    for target, frame in merged.groupby("target", sort=True):
        actual = frame["actual_fisher_z_model"].to_numpy(dtype=float)
        model_prediction = frame["predicted_fisher_z_model"].to_numpy(
            dtype=float
        )
        base_prediction = frame["predicted_fisher_z_base"].to_numpy(dtype=float)
        model_loss = np.square(actual - model_prediction)
        base_loss = np.square(actual - base_prediction)
        rows.append(
            {
                "target": target,
                "rows": len(frame),
                "base_fisher_z_rmse": float(np.sqrt(base_loss.mean())),
                "model_fisher_z_rmse": float(np.sqrt(model_loss.mean())),
                "incremental_r2_news_given_quant": (
                    float(1 - model_loss.sum() / base_loss.sum())
                    if base_loss.sum() > 0
                    else math.nan
                ),
                "mean_squared_loss_delta_model_minus_base": float(
                    np.mean(model_loss - base_loss)
                ),
            }
        )
    return rows


def validate_prediction_panel(
    predictions: pd.DataFrame,
    *,
    expected_folds: Sequence[str],
    expected_targets: Sequence[str],
    expected_rows: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    duplicate = predictions.duplicated(PREDICTION_KEYS).any()
    finite = np.isfinite(
        predictions[
            ["predicted_fisher_z", "predicted_correlation"]
        ].to_numpy(dtype=float)
    ).all()
    bounded = predictions["predicted_correlation"].between(-1, 1).all()
    observed_counts = {
        target: {
            fold: int(
                len(
                    predictions[
                        predictions["target"].eq(target)
                        & predictions["fold"].eq(fold)
                    ]
                )
            )
            for fold in expected_folds
        }
        for target in expected_targets
    }
    expected = {
        target: {
            fold: int(expected_rows[target][fold]) for fold in expected_folds
        }
        for target in expected_targets
    }
    review = {
        "prediction_keys_unique": not bool(duplicate),
        "predictions_finite": bool(finite),
        "prediction_bounds_valid": bool(bounded),
        "expected_targets_present": set(predictions["target"])
        == set(expected_targets),
        "expected_folds_present": set(predictions["fold"])
        == set(expected_folds),
        "observed_rows": observed_counts,
        "expected_rows": expected,
        "row_counts_match": observed_counts == expected,
    }
    if not all(
        bool(review[key])
        for key in (
            "prediction_keys_unique",
            "predictions_finite",
            "prediction_bounds_valid",
            "expected_targets_present",
            "expected_folds_present",
            "row_counts_match",
        )
    ):
        raise AssertionError(f"Prediction review failed: {review}")
    return review


def track_a_expected_rows(
    protocol: Mapping[str, Any],
) -> dict[str, dict[str, int]]:
    raw = protocol["target_row_eligibility"]["expected_rows"]
    output: dict[str, dict[str, int]] = {}
    for target in protocol["targets"]:
        horizon = target.split("_", 1)[0]
        index = 2
        output[target] = {
            fold: int(values[f"{horizon}_train_validation_test"][index])
            for fold, values in raw.items()
        }
    return output


def track_b_expected_rows() -> dict[str, dict[str, int]]:
    return {
        "t1_etf": {"fold_2": 3840, "fold_3": 3690},
        "t1_loo": {"fold_2": 3840, "fold_3": 3690},
        "t2_etf": {"fold_2": 3720, "fold_3": 3570},
        "t2_loo": {"fold_2": 3720, "fold_3": 3570},
    }


def initialize_status(protocol_sha256: str) -> dict[str, Any]:
    if STATUS_PATH.exists():
        status = load_json(STATUS_PATH)
        if status["protocol_sha256"] != protocol_sha256:
            raise ValueError("Existing status uses a different protocol hash")
        return status
    status = {
        "experiment_id": "quant-deterministic-news-v1",
        "protocol_sha256": protocol_sha256,
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "bundles": [
            {
                "order": bundle.order,
                "key": bundle.key,
                "track": bundle.track,
                "slug": bundle.slug,
                "label": bundle.label,
                "status": "planned",
                "attempts": 0,
                "started_at_utc": None,
                "completed_at_utc": None,
                "summary": None,
            }
            for bundle in BUNDLES
        ],
    }
    quant_common.write_json(STATUS_PATH, status)
    render_status_markdown(status)
    return status


def require_completed_preflight() -> None:
    verify_completed_bundle("PRE")


def verify_completed_bundle(
    bundle_key: str,
    *,
    allowed_states: Sequence[str] = ("complete",),
) -> dict[str, Any]:
    """Verify status plus protocol/panel provenance for one bundle."""

    bundle = bundle_for(bundle_key)
    protocol_hash = sha256_file(PROTOCOL_PATH)
    panel_hash = sha256_file(PANEL_PATH)
    status = load_json(STATUS_PATH)
    if status["protocol_sha256"] != protocol_hash:
        raise RuntimeError("Status ledger uses a stale protocol hash")
    record = next(
        item for item in status["bundles"] if item["key"] == bundle_key
    )
    if record["status"] not in set(allowed_states):
        raise RuntimeError(
            f"{bundle_key} status {record['status']!r} is not one of "
            f"{tuple(allowed_states)!r}"
        )
    manifest_path = bundle.output_path / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"{bundle_key} manifest is missing")
    manifest = load_json(manifest_path)
    if manifest.get("bundle") != bundle_key:
        raise RuntimeError(f"{bundle_key} manifest names another bundle")
    if manifest.get("protocol_sha256") != protocol_hash:
        raise RuntimeError(f"{bundle_key} manifest uses a stale protocol hash")
    if manifest.get("panel_sha256") != panel_hash:
        raise RuntimeError(f"{bundle_key} manifest uses a stale panel hash")
    return {
        "bundle": bundle_key,
        "state": record["status"],
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "protocol_sha256": protocol_hash,
        "panel_sha256": panel_hash,
        "manifest": manifest,
    }


def verified_bundle_artifact(
    bundle_key: str,
    filename: str,
    *,
    allowed_states: Sequence[str] = ("complete",),
) -> tuple[Path, dict[str, Any]]:
    """Resolve and hash-check one artifact listed by a completed bundle."""

    provenance = verify_completed_bundle(
        bundle_key, allowed_states=allowed_states
    )
    manifest = provenance.pop("manifest")
    artifact = manifest.get("artifacts", {}).get(filename)
    if not isinstance(artifact, Mapping):
        raise RuntimeError(
            f"{bundle_key} manifest does not list artifact {filename}"
        )
    canonical_path = bundle_for(bundle_key).output_path / filename
    recorded_path = Path(str(artifact.get("path", "")))
    if recorded_path.resolve() != canonical_path.resolve():
        raise RuntimeError(
            f"{bundle_key}/{filename} manifest path is noncanonical"
        )
    if not canonical_path.exists():
        raise RuntimeError(f"{bundle_key}/{filename} is missing")
    observed_hash = sha256_file(canonical_path)
    if observed_hash != artifact.get("sha256"):
        raise RuntimeError(f"{bundle_key}/{filename} hash mismatch")
    provenance["artifact_path"] = canonical_path.as_posix()
    provenance["artifact_sha256"] = observed_hash
    return canonical_path, provenance


def update_status(
    bundle_key: str,
    state: str,
    *,
    summary: str | None = None,
) -> dict[str, Any]:
    if state not in {
        "planned",
        "running",
        "review_pending",
        "complete",
        "failed",
        "skipped",
    }:
        raise ValueError(f"Invalid bundle state {state}")
    status = load_json(STATUS_PATH)
    record = next(
        item for item in status["bundles"] if item["key"] == bundle_key
    )
    now = utc_now()
    if state == "running":
        record["attempts"] = int(record["attempts"]) + 1
        record["started_at_utc"] = now
        record["completed_at_utc"] = None
    if state in {"complete", "failed", "skipped"}:
        record["completed_at_utc"] = now
    record["status"] = state
    if summary is not None:
        record["summary"] = summary
    status["updated_at_utc"] = now
    quant_common.write_json(STATUS_PATH, status)
    render_status_markdown(status)
    return status


def render_status_markdown(status: Mapping[str, Any]) -> None:
    lines = [
        "# Quant plus deterministic-news v1 status",
        "",
        f"Protocol SHA-256: `{status['protocol_sha256']}`",
        "",
        "This file is updated after each isolated model bundle. Heavy predictions",
        "and fit records are under `outputs/quant_deterministic_news/v1`.",
        "",
        "| Order | Bundle | Track | Status | Attempts | Result |",
        "|---:|---|---|---|---:|---|",
    ]
    for item in sorted(status["bundles"], key=lambda value: value["order"]):
        bundle = BUNDLE_BY_KEY[item["key"]]
        result_link = (
            f"[results]({bundle.track}/{bundle.slug}/RESULTS.md)"
            if item["status"] in {"complete", "skipped"}
            else ""
        )
        summary = item.get("summary") or ""
        result = f"{summary} {result_link}".strip()
        lines.append(
            f"| {item['order']} | {item['key']} - {item['label']} | "
            f"`{item['track']}` | **{item['status']}** | "
            f"{item['attempts']} | {result} |"
        )
    lines.extend(
        [
            "",
            "All results are exploratory development estimates because the news",
            "archive is not historical-version-safe and the quant outer blocks were",
            "already inspected in quant v1.",
            "",
        ]
    )
    STATUS_MARKDOWN_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATUS_MARKDOWN_PATH.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(STATUS_MARKDOWN_PATH)


def bundle_for(key: str) -> Bundle:
    try:
        return BUNDLE_BY_KEY[key]
    except KeyError as error:
        raise ValueError(f"Unknown bundle {key}") from error


def _result_markdown(
    bundle: Bundle,
    summary: Mapping[str, Any],
    review: Mapping[str, Any],
    artifact_refs: Mapping[str, Any],
) -> str:
    lines = [
        f"# {bundle.label}",
        "",
        f"Status: **{summary.get('status', 'complete')}**",
        "",
        f"Completed: `{summary.get('generated_at_utc', '')}`",
        "",
    ]
    metrics = summary.get("metrics", [])
    if metrics:
        lines.extend(
            [
                "| Target | Rows | Fisher-z RMSE | Fisher-z MAE | Raw RMSE | OOS R^2 vs persistence |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for item in metrics:
            lines.append(
                f"| {item['target']} | {item['rows']} | "
                f"{item['fisher_z_rmse']:.6f} | {item['fisher_z_mae']:.6f} | "
                f"{item['raw_correlation_rmse']:.6f} | "
                f"{item['oos_r2_vs_persistence']:.6f} |"
            )
        lines.append("")
    comparisons = summary.get("base_comparison", [])
    if comparisons:
        lines.extend(
            [
                "| Target | Base RMSE | Model RMSE | Incremental R^2 news given Q | Delta squared loss |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for item in comparisons:
            lines.append(
                f"| {item['target']} | {item['base_fisher_z_rmse']:.6f} | "
                f"{item['model_fisher_z_rmse']:.6f} | "
                f"{item['incremental_r2_news_given_quant']:.6f} | "
                f"{item['mean_squared_loss_delta_model_minus_base']:.8f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Review",
            "",
            f"- Review status: **{review.get('status', 'passed')}**",
            f"- Prediction keys unique: `{review.get('prediction_keys_unique')}`",
            f"- Predictions finite: `{review.get('predictions_finite')}`",
            f"- Bounds valid: `{review.get('prediction_bounds_valid')}`",
            "",
            "## Artifacts",
            "",
        ]
    )
    for name, value in artifact_refs.items():
        if isinstance(value, Mapping) and "path" in value:
            lines.append(
                f"- {name}: `{value['path']}` - SHA-256 `{value['sha256']}`"
            )
    lines.extend(
        [
            "",
            "Interpretation is exploratory only; see the experiment-level README",
            "for the point-in-time and model-selection limitations.",
            "",
        ]
    )
    return "\n".join(lines)


def implementation_provenance(bundle_key: str) -> dict[str, Any]:
    """Hash the code and record package versions used by a bundle."""

    if bundle_key == "PRE":
        runner = ROOT / "scripts/quant_deterministic_news_training/preflight.py"
    elif bundle_key in {
        "A0-L",
        "A1",
        "A2",
        "A3",
        "A4",
        "A5",
        "C-A5-L20",
        "C-A5-WS",
    }:
        runner = (
            ROOT
            / "scripts/quant_deterministic_news_training/run_joint_linear.py"
        )
    elif bundle_key in {"A0-T", "A6"}:
        runner = (
            ROOT
            / "scripts/quant_deterministic_news_training/"
            "run_joint_xgboost.py"
        )
    elif bundle_key == "A7":
        runner = (
            ROOT
            / "scripts/quant_deterministic_news_training/"
            "run_joint_ensemble.py"
        )
    elif bundle_key == "FINAL":
        runner = (
            ROOT
            / "scripts/quant_deterministic_news_training/"
            "run_final_comparison.py"
        )
    else:
        runner = (
            ROOT
            / "scripts/quant_deterministic_news_training/run_residual.py"
        )
    source_paths = [
        ROOT / "scripts/quant_deterministic_news_training/common.py",
        runner,
        ROOT / "scripts/correlation_training/training_common.py",
        ROOT / "scripts/correlation_training/build_modeling_panel.py",
        ROOT / "scripts/correlation_training/run_rung_03.py",
    ]
    files = {}
    for path in source_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Implementation source missing for {bundle_key}: {path}"
            )
        relative = path.relative_to(ROOT).as_posix()
        files[relative] = sha256_file(path)

    packages: dict[str, str | None] = {}
    for package in (
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "pyarrow",
        "xgboost",
    ):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {
        "source_sha256": files,
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "packages": packages,
        },
    }


def write_bundle(
    bundle_key: str,
    *,
    predictions: pd.DataFrame | None,
    validation_predictions: pd.DataFrame | None,
    fits: object,
    fold_metrics: object,
    summary: dict[str, Any],
    review: dict[str, Any],
    model_config: dict[str, Any],
    dependencies: Mapping[str, Any] | None = None,
    extra_outputs: Mapping[str, pd.DataFrame | object] | None = None,
    results_markdown: str | None = None,
) -> dict[str, Any]:
    bundle = bundle_for(bundle_key)
    output = bundle.output_path
    tracked = bundle.experiment_path
    output.mkdir(parents=True, exist_ok=True)
    tracked.mkdir(parents=True, exist_ok=True)
    if predictions is not None:
        quant_common.write_parquet(output / "predictions.parquet", predictions)
    if validation_predictions is not None:
        quant_common.write_parquet(
            output / "validation_predictions.parquet",
            validation_predictions,
        )
    quant_common.write_json(output / "fits.json", fits)
    quant_common.write_json(output / "fold_metrics.json", fold_metrics)
    for name, value in (extra_outputs or {}).items():
        path = output / name
        if isinstance(value, pd.DataFrame):
            quant_common.write_parquet(path, value)
        else:
            quant_common.write_json(path, value)
    if results_markdown is not None:
        temporary_report = output / "RESULTS.md.tmp"
        temporary_report.write_text(results_markdown, encoding="utf-8")
        temporary_report.replace(output / "RESULTS.md")
    output_files = sorted(
        path
        for path in output.iterdir()
        if path.is_file() and path.name != "manifest.json"
    )
    refs = {
        path.name: {"path": path.as_posix(), "sha256": sha256_file(path)}
        for path in output_files
    }
    manifest = {
        "bundle": bundle_key,
        "generated_at_utc": utc_now(),
        "protocol_path": PROTOCOL_PATH.as_posix(),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "panel_path": PANEL_PATH.as_posix(),
        "panel_sha256": sha256_file(PANEL_PATH),
        "model_config": model_config,
        "dependencies": dict(dependencies or {}),
        "implementation_provenance": implementation_provenance(bundle_key),
        "artifacts": refs,
    }
    quant_common.write_json(output / "manifest.json", manifest)
    refs["manifest.json"] = {
        "path": (output / "manifest.json").as_posix(),
        "sha256": sha256_file(output / "manifest.json"),
    }
    summary = {
        **summary,
        "status": summary.get("status", "complete"),
        "generated_at_utc": summary.get("generated_at_utc", utc_now()),
    }
    review = {**review, "status": review.get("status", "passed")}
    quant_common.write_json(tracked / "config.json", model_config)
    quant_common.write_json(tracked / "review.json", review)
    quant_common.write_json(tracked / "summary.json", summary)
    quant_common.write_json(tracked / "artifact_refs.json", refs)
    for target in sorted(
        {item["target"] for item in summary.get("metrics", [])}
    ):
        target_metrics = [
            item
            for item in summary["metrics"]
            if item["target"] == target
        ]
        quant_common.write_json(
            tracked / target / "metrics.json", target_metrics
        )
    result = results_markdown or _result_markdown(
        bundle, summary, review, refs
    )
    result_path = tracked / "RESULTS.md"
    temporary_result = result_path.with_suffix(".md.tmp")
    temporary_result.write_text(result, encoding="utf-8")
    temporary_result.replace(result_path)
    return refs


def summary_status_text(summary: Mapping[str, Any]) -> str:
    metrics = summary.get("metrics", [])
    if not metrics:
        included = summary.get("included_targets")
        if included is not None:
            return f"included targets: {', '.join(included) or 'none'}"
        return summary.get("status", "complete")
    values = ", ".join(
        f"{item['target']} {item['fisher_z_rmse']:.4f}"
        for item in metrics
    )
    return f"Fisher-z RMSE: {values}"


def expected_validation_counts(
    protocol: Mapping[str, Any],
) -> dict[str, dict[str, int]]:
    raw = protocol["target_row_eligibility"]["expected_rows"]
    output: dict[str, dict[str, int]] = {}
    for target in protocol["targets"]:
        horizon = target.split("_", 1)[0]
        output[target] = {
            fold: int(values[f"{horizon}_train_validation_test"][1])
            for fold, values in raw.items()
        }
    return output


def validate_validation_predictions(
    predictions: pd.DataFrame,
    protocol: Mapping[str, Any],
    *,
    targets: Sequence[str] | None = None,
) -> dict[str, Any]:
    targets = list(targets or protocol["targets"])
    counts = expected_validation_counts(protocol)
    return validate_prediction_panel(
        predictions,
        expected_folds=["fold_1", "fold_2", "fold_3"],
        expected_targets=targets,
        expected_rows={target: counts[target] for target in targets},
    )


def residual_prediction_frame(
    evaluation: pd.DataFrame,
    *,
    model_name: str,
    correction: np.ndarray,
) -> pd.DataFrame:
    output = evaluation[
        [
            "fold",
            "target",
            "forecast_date",
            "sector",
            "stock",
            "benchmark",
            "actual_fisher_z",
            "actual_correlation",
            "persistence_fisher_z",
            "persistence_correlation",
            "base_model",
            "base_artifact_sha256",
            "predicted_fisher_z",
        ]
    ].copy()
    output = output.rename(
        columns={"predicted_fisher_z": "base_predicted_fisher_z"}
    )
    output["correction_fisher_z"] = np.asarray(correction, dtype=float)
    output["predicted_fisher_z"] = (
        output["base_predicted_fisher_z"] + output["correction_fisher_z"]
    )
    output["predicted_correlation"] = quant_common.fisher_to_correlation(
        output["predicted_fisher_z"]
    )
    output["model"] = model_name
    return output


def metrics_records(predictions: pd.DataFrame) -> list[dict[str, Any]]:
    return quant_common.summarize_predictions(predictions).to_dict(
        orient="records"
    )


def fold_metric_records(predictions: pd.DataFrame) -> list[dict[str, Any]]:
    return quant_common.summarize_predictions(
        predictions, ("fold", "target", "model")
    ).to_dict(orient="records")


def ensure_no_existing_bundle(bundle_key: str, *, overwrite: bool) -> None:
    bundle = bundle_for(bundle_key)
    occupied = any(
        path.exists() and any(path.iterdir())
        for path in (bundle.output_path, bundle.experiment_path)
    )
    if occupied and not overwrite:
        raise FileExistsError(
            f"{bundle_key} already has a canonical output; use --overwrite"
        )
    if occupied and overwrite:
        for path, root in (
            (bundle.output_path, OUTPUT_ROOT),
            (bundle.experiment_path, EXPERIMENT_ROOT),
        ):
            resolved = path.resolve()
            resolved_root = root.resolve()
            if resolved == resolved_root or resolved_root not in resolved.parents:
                raise ValueError(f"Unsafe bundle cleanup path: {resolved}")
            if path.exists():
                shutil.rmtree(path)


def mean_metric(metrics: Iterable[Mapping[str, Any]], name: str) -> float:
    values = [float(item[name]) for item in metrics]
    return float(np.mean(values)) if values else math.nan
