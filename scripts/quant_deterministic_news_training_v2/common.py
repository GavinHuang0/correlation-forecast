"""Shared contracts and artifact utilities for deterministic-news v2 training."""

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
from typing import Any, Mapping, Sequence

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


PROTOCOL_PATH = Path("config/quant_deterministic_news_protocol_v2.json")
PANEL_PATH = Path(
    "data/features/q_plus_d/massive_v2/modeling_panel_q_d2.parquet"
)
EXPERIMENT_ROOT = Path("experiments/quant_deterministic_news/v2/training")
OUTPUT_ROOT = Path("outputs/quant_deterministic_news/v2")
STATUS_PATH = EXPERIMENT_ROOT / "status.json"
STATUS_MARKDOWN_PATH = EXPERIMENT_ROOT / "STATUS.md"

PANEL_KEYS = ["forecast_date", "sector", "stock", "benchmark"]
PREDICTION_KEYS = ["fold", "target", *PANEL_KEYS]
TARGETS = ("t1_etf", "t1_loo", "t2_etf", "t2_loo")


@dataclass(frozen=True)
class Bundle:
    order: int
    key: str
    slug: str
    label: str
    group: str = "deterministic"

    @property
    def experiment_path(self) -> Path:
        return EXPERIMENT_ROOT / self.group / self.slug

    @property
    def output_path(self) -> Path:
        return OUTPUT_ROOT / self.group / self.slug


BUNDLES = (
    Bundle(0, "D0", "d0_q56_elastic_net", "V2-D0 matched Q56 Elastic Net"),
    Bundle(1, "D1", "d1_d2_elastic_net", "V2-D1 D2-only Elastic Net"),
    Bundle(
        2,
        "D2",
        "d2_q56_d43_recomputed_elastic_net",
        "V2-D2 Q56 plus matched D43 Elastic Net",
    ),
    Bundle(
        3,
        "D3",
        "d3_q56_d2_elastic_net",
        "V2-D3 Q56 plus D2-Normalized Elastic Net",
    ),
    Bundle(
        4,
        "D4",
        "d4_q56_d2_levels_elastic_net",
        "V2-D4 Q56 plus D2-Normalized and D2-Levels Elastic Net",
    ),
    Bundle(
        5,
        "D5",
        "d5_q56_d2_shallow_xgboost",
        "V2-D5 validation-gated Q56 plus D2-Normalized shallow XGBoost",
    ),
    Bundle(
        6,
        "C-D3-L20",
        "d3_stale_d2_lag20",
        "V2-D3 20-session stale-D2 falsification control",
        "controls",
    ),
    Bundle(
        7,
        "C-D3-WS",
        "d3_wrong_stock_d2",
        "V2-D3 fixed wrong-stock-D2 falsification control",
        "controls",
    ),
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


def bundle_for(key: str) -> Bundle:
    try:
        return BUNDLE_BY_KEY[key]
    except KeyError as error:
        raise ValueError(f"Unknown v2 deterministic bundle {key}") from error


def _string_list(
    mapping: Mapping[str, Any], key: str, *, expected_count: int
) -> tuple[str, ...]:
    raw = mapping.get(key)
    if not isinstance(raw, list) or not all(
        isinstance(value, str) and value for value in raw
    ):
        raise TypeError(f"feature_blocks.{key} must be a list of names")
    values = tuple(raw)
    if len(values) != expected_count or len(set(values)) != expected_count:
        raise ValueError(
            f"feature_blocks.{key} must contain {expected_count} unique names"
        )
    return values


def _validate_folds(protocol: Mapping[str, Any]) -> None:
    folds = protocol.get("folds")
    if not isinstance(folds, list) or len(folds) != 3:
        raise ValueError("v2 deterministic training requires exactly three folds")
    required = {
        "name",
        "train_start",
        "train_end",
        "validation_start",
        "validation_end",
        "test_start",
        "test_end",
    }
    names: list[str] = []
    previous_test_end: pd.Timestamp | None = None
    for fold in folds:
        if not isinstance(fold, Mapping) or set(fold) != required:
            raise ValueError(f"Malformed fold record: {fold!r}")
        name = str(fold["name"])
        names.append(name)
        dates = {
            key: pd.Timestamp(str(fold[key]))
            for key in required
            if key != "name"
        }
        if not (
            dates["train_start"]
            <= dates["train_end"]
            < dates["validation_start"]
            <= dates["validation_end"]
            < dates["test_start"]
            <= dates["test_end"]
        ):
            raise ValueError(f"Fold {name} is not strictly chronological")
        if previous_test_end is not None and dates["test_end"] <= previous_test_end:
            raise ValueError("Fold test endpoints must advance")
        previous_test_end = dates["test_end"]
    if names != ["fold_1", "fold_2", "fold_3"]:
        raise ValueError("Fold names/order must be fold_1, fold_2, fold_3")


def load_protocol(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    protocol = load_json(path)
    if protocol.get("experiment_id") != "quant-deterministic-news-v2":
        raise ValueError("Unexpected v2 experiment ID")
    if protocol.get("status") != "locked_before_training":
        raise ValueError("The v2 protocol must be locked before training")
    if tuple(protocol.get("targets", ())) != TARGETS:
        raise ValueError(f"targets must be exactly {TARGETS}")
    if not isinstance(protocol.get("claim_scope"), str):
        raise ValueError("claim_scope must describe the exploratory claim boundary")
    if not isinstance(protocol.get("source_profile"), str):
        raise ValueError("source_profile must be named")
    _validate_folds(protocol)

    availability = protocol.get("bundle_availability", {})
    if not isinstance(availability, Mapping):
        raise TypeError("bundle_availability must be an object when present")
    protocol_bundle_keys = {"D0", "D1", "D2", "D3", "D4", "D5"}
    for key, record in availability.items():
        if key not in protocol_bundle_keys:
            raise ValueError(f"bundle_availability names unknown bundle {key}")
        if not isinstance(record, Mapping) or not isinstance(
            record.get("available"), bool
        ):
            raise ValueError(
                f"bundle_availability.{key} must contain boolean available"
            )
        if not record["available"] and not (
            isinstance(record.get("reason"), str) and record["reason"].strip()
        ):
            raise ValueError(
                f"Unavailable bundle {key} requires a nonempty reason"
            )

    blocks = protocol.get("feature_blocks")
    if not isinstance(blocks, Mapping):
        raise TypeError("feature_blocks must be an object")
    # Resolve every block once so malformed feature contracts fail before a fit.
    for spec in quant_common.target_specs():
        quant_features(spec, protocol)
    _string_list(blocks, "d2_normalized_30", expected_count=30)
    _string_list(blocks, "d2_levels_5", expected_count=5)
    d43_available = bool(
        availability.get("D2", {"available": True})["available"]
    )
    raw_d43 = blocks.get("d43_recomputed_43", [])
    if d43_available:
        _string_list(blocks, "d43_recomputed_43", expected_count=43)
    elif raw_d43 not in (None, []):
        _string_list(blocks, "d43_recomputed_43", expected_count=43)

    tuning = protocol.get("model_tuning")
    if not isinstance(tuning, Mapping):
        raise TypeError("model_tuning must be an object")
    alphas = tuning.get("linear_alpha_grid")
    ratios = tuning.get("elastic_net_l1_ratio_grid")
    if (
        not isinstance(alphas, list)
        or not alphas
        or any(float(value) <= 0 for value in alphas)
    ):
        raise ValueError("linear_alpha_grid must contain positive values")
    if (
        not isinstance(ratios, list)
        or not ratios
        or any(not 0 < float(value) <= 1 for value in ratios)
    ):
        raise ValueError("elastic_net_l1_ratio_grid must lie in (0, 1]")
    preprocessing = protocol.get("preprocessing")
    if not isinstance(preprocessing, Mapping):
        raise TypeError("preprocessing must be an object")
    fixed_log1p = preprocessing.get("fixed_log1p_features")
    if fixed_log1p != list(quant_schema.LOG1P_FEATURES):
        raise ValueError(
            "preprocessing.fixed_log1p_features must exactly lock the "
            "ordered quant-v1 transform contract"
        )

    sources = protocol.get("source_artifacts")
    required_sources = {
        "panel_path",
        "panel_sha256",
        "panel_manifest_path",
        "panel_manifest_sha256",
        "row_count",
        "date_count",
        "stock_count",
        "sector_count",
    }
    if not isinstance(sources, Mapping) or not required_sources.issubset(sources):
        raise ValueError(
            f"source_artifacts must contain {sorted(required_sources)}"
        )
    if Path(str(sources["panel_path"])).resolve() != PANEL_PATH.resolve():
        raise ValueError("source_artifacts.panel_path is not the canonical v2 panel")

    expected = protocol.get("target_row_eligibility", {}).get("expected_rows")
    if not isinstance(expected, Mapping):
        raise ValueError("target_row_eligibility.expected_rows is required")
    for fold in protocol["folds"]:
        record = expected.get(fold["name"])
        if not isinstance(record, Mapping):
            raise ValueError(f"Expected-row record missing for {fold['name']}")
        for horizon in ("t1", "t2"):
            values = record.get(f"{horizon}_train_validation_test")
            if (
                not isinstance(values, list)
                or len(values) != 3
                or any(int(value) <= 0 for value in values)
            ):
                raise ValueError(
                    f"Expected rows malformed for {fold['name']}/{horizon}"
                )
    return protocol


def quant_features(
    spec: quant_common.TargetSpec, protocol: Mapping[str, Any]
) -> tuple[str, ...]:
    blocks = protocol["feature_blocks"]
    direct_key = f"q56_{spec.benchmark}"
    if direct_key in blocks:
        return _string_list(blocks, direct_key, expected_count=56)
    pair_key = (
        "q_pair_etf_14"
        if spec.benchmark == "etf"
        else "q_pair_loo_14"
    )
    values = (
        *_string_list(blocks, pair_key, expected_count=14),
        *_string_list(blocks, "q_sector_state_8", expected_count=8),
        *_string_list(blocks, "q_dense_15", expected_count=15),
        *_string_list(blocks, "q_volatility_2", expected_count=2),
        *_string_list(blocks, "q_extended_17", expected_count=17),
    )
    if len(values) != 56 or len(set(values)) != 56:
        raise ValueError(f"Resolved Q56 contract failed for {spec.name}")
    return tuple(values)


def deterministic_features(
    block: str, protocol: Mapping[str, Any]
) -> tuple[str, ...]:
    counts = {
        "d2_normalized_30": 30,
        "d43_recomputed_43": 43,
        "d2_levels_5": 5,
    }
    try:
        expected = counts[block]
    except KeyError as error:
        raise ValueError(f"Unknown deterministic feature block {block}") from error
    return _string_list(
        protocol["feature_blocks"], block, expected_count=expected
    )


def validate_model_features(features: Sequence[str]) -> None:
    if not features or len(features) != len(set(features)):
        raise ValueError("Model features must be nonempty and unique")
    forbidden_exact = {
        "sector",
        "stock",
        "benchmark",
        "forecast_date",
        "asof_session",
        "point_in_time_version_safe",
        "primary_training_eligible",
        "source_profile",
    }
    forbidden = [
        column
        for column in features
        if column in forbidden_exact
        or column.startswith("target_")
        or column.endswith("_end_date")
        or "realized_covariance" in column
        or "realized_variance" in column
        or "aligned_return_count" in column
        or "expected_return_count" in column
    ]
    if forbidden:
        raise ValueError(f"Target/audit leakage columns requested: {forbidden}")


def bundle_features(
    bundle_key: str,
    spec: quant_common.TargetSpec,
    protocol: Mapping[str, Any],
) -> tuple[str, ...]:
    q = quant_features(spec, protocol)
    d2 = deterministic_features("d2_normalized_30", protocol)
    levels = deterministic_features("d2_levels_5", protocol)
    if bundle_key == "D0":
        values = q
    elif bundle_key == "D1":
        values = d2
    elif bundle_key == "D2":
        available, reason = bundle_availability(protocol, "D2")
        if not available:
            raise ValueError(f"D2 is unavailable: {reason}")
        d43 = deterministic_features("d43_recomputed_43", protocol)
        values = (*q, *d43)
    elif bundle_key in {"D3", "D5", "C-D3-L20", "C-D3-WS"}:
        values = (*q, *d2)
    elif bundle_key == "D4":
        values = (*q, *d2, *levels)
    else:
        raise ValueError(f"Unknown deterministic bundle {bundle_key}")
    expected = {
        "D0": 56,
        "D1": 30,
        "D2": 99,
        "D3": 86,
        "D4": 91,
        "D5": 86,
        "C-D3-L20": 86,
        "C-D3-WS": 86,
    }
    if len(values) != expected[bundle_key] or len(set(values)) != len(values):
        raise ValueError(f"Feature count/uniqueness failed for {bundle_key}")
    validate_model_features(values)
    return tuple(values)


def _verify_hash(path: Path, expected: str, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} is missing: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(
            f"{label} hash mismatch: expected {expected}, observed {observed}"
        )


def source_artifact_provenance(
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    sources = protocol["source_artifacts"]
    panel_path = Path(str(sources["panel_path"]))
    manifest_path = Path(str(sources["panel_manifest_path"]))
    _verify_hash(panel_path, str(sources["panel_sha256"]), "v2 modeling panel")
    _verify_hash(
        manifest_path,
        str(sources["panel_manifest_sha256"]),
        "v2 panel manifest",
    )
    return {
        "panel_path": panel_path.as_posix(),
        "panel_sha256": sha256_file(panel_path),
        "panel_manifest_path": manifest_path.as_posix(),
        "panel_manifest_sha256": sha256_file(manifest_path),
    }


def bundle_availability(
    protocol: Mapping[str, Any], bundle_key: str
) -> tuple[bool, str | None]:
    bundle_for(bundle_key)
    record = protocol.get("bundle_availability", {}).get(bundle_key)
    if record is None:
        return True, None
    if bool(record["available"]):
        return True, None
    return False, str(record["reason"])


def expected_rows(
    protocol: Mapping[str, Any], *, split_index: int
) -> dict[str, dict[str, int]]:
    raw = protocol["target_row_eligibility"]["expected_rows"]
    output: dict[str, dict[str, int]] = {}
    for target in protocol["targets"]:
        horizon = target.split("_", 1)[0]
        output[target] = {
            fold["name"]: int(
                raw[fold["name"]][f"{horizon}_train_validation_test"][
                    split_index
                ]
            )
            for fold in protocol["folds"]
        }
    return output


def split_masks(
    panel: pd.DataFrame,
    spec: quant_common.TargetSpec,
    fold: Mapping[str, str],
) -> dict[str, pd.Series]:
    return quant_common.split_masks(panel, spec, fold)


def load_panel_and_preflight(
    protocol: Mapping[str, Any],
    *,
    allow_exploratory: bool,
    required_bundle_keys: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not allow_exploratory:
        raise ValueError(
            "The retrospective v2 panel is exploratory; pass --allow-exploratory"
        )
    sources = protocol["source_artifacts"]
    panel_path = Path(str(sources["panel_path"]))
    manifest_path = Path(str(sources["panel_manifest_path"]))
    source_artifact_provenance(protocol)
    panel = quant_common.load_panel(panel_path)
    observed_shape = {
        "row_count": len(panel),
        "date_count": int(panel["forecast_date"].nunique()),
        "stock_count": int(panel["stock"].nunique()),
        "sector_count": int(panel["sector"].nunique()),
    }
    expected_shape = {
        key: int(sources[key])
        for key in ("row_count", "date_count", "stock_count", "sector_count")
    }
    if observed_shape != expected_shape:
        raise ValueError(
            f"v2 panel coverage mismatch: {observed_shape} != {expected_shape}"
        )
    if panel.duplicated(PANEL_KEYS).any():
        raise ValueError("v2 panel has duplicate stock-date-benchmark keys")
    required_equalities = protocol.get("row_eligibility", {}).get(
        "required_equalities", {}
    )
    if not isinstance(required_equalities, Mapping):
        raise TypeError("row_eligibility.required_equalities must be an object")
    for column, expected_value in required_equalities.items():
        if column not in panel:
            raise ValueError(f"Required eligibility column missing: {column}")
        if not panel[column].eq(expected_value).all():
            raise ValueError(
                f"Eligibility condition failed: {column} == {expected_value!r}"
            )

    required_bundles = (
        tuple(required_bundle_keys)
        if required_bundle_keys is not None
        else tuple(
            key
            for key in BUNDLE_BY_KEY
            if bundle_availability(protocol, key)[0]
        )
    )
    unknown_bundles = set(required_bundles).difference(BUNDLE_BY_KEY)
    if unknown_bundles:
        raise ValueError(
            f"Unknown bundles requested in preflight: {sorted(unknown_bundles)}"
        )
    unavailable_bundles = {
        key: reason
        for key in required_bundles
        for available, reason in (bundle_availability(protocol, key),)
        if not available
    }
    if unavailable_bundles:
        raise ValueError(
            "Unavailable bundles requested in preflight: "
            f"{unavailable_bundles}"
        )
    all_features: set[str] = set()
    for spec in quant_common.target_specs():
        for key in required_bundles:
            all_features.update(bundle_features(key, spec, protocol))
    missing_columns = all_features.difference(panel.columns)
    if missing_columns:
        raise ValueError(
            f"v2 panel misses model features: {sorted(missing_columns)}"
        )
    numeric = panel[sorted(all_features)].select_dtypes(include=[np.number])
    if numeric.shape[1] != len(all_features):
        nonnumeric = sorted(all_features.difference(numeric.columns))
        raise TypeError(f"Model features must be numeric: {nonnumeric}")
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError("v2 model features contain infinite values")
    complete_case: list[str] = []
    if set(required_bundles).intersection(
        {"D1", "D3", "D4", "D5", "C-D3-L20", "C-D3-WS"}
    ):
        complete_case.extend(
            deterministic_features("d2_normalized_30", protocol)
        )
    if "D4" in required_bundles:
        complete_case.extend(deterministic_features("d2_levels_5", protocol))
    if complete_case and panel[complete_case].isna().any().any():
        raise ValueError(
            "Required D2-Normalized/D2-Levels features must be complete-case"
        )

    quant_common.validate_panel_information_set(panel)
    split_counts: dict[str, dict[str, dict[str, int]]] = {}
    raw_expected = protocol["target_row_eligibility"]["expected_rows"]
    target_nonnull: dict[str, int] = {}
    for spec in quant_common.target_specs():
        target_nonnull[spec.name] = int(panel[spec.response_column].notna().sum())
        split_counts[spec.name] = {}
        for fold in protocol["folds"]:
            masks = split_masks(panel, spec, fold)
            observed = {
                name: int(mask.sum()) for name, mask in masks.items()
            }
            expected_values = raw_expected[fold["name"]][
                f"{spec.horizon}_train_validation_test"
            ]
            expected_record = dict(
                zip(
                    ("train", "validation", "test"),
                    map(int, expected_values),
                    strict=True,
                )
            )
            if observed != expected_record:
                raise ValueError(
                    f"Split mismatch for {spec.name}/{fold['name']}: "
                    f"{observed} != {expected_record}"
                )
            split_counts[spec.name][fold["name"]] = observed
    audit = {
        "status": "passed",
        "generated_at_utc": utc_now(),
        "protocol_path": PROTOCOL_PATH.as_posix(),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "panel_path": panel_path.as_posix(),
        "panel_sha256": sha256_file(panel_path),
        "panel_manifest_path": manifest_path.as_posix(),
        "panel_manifest_sha256": sha256_file(manifest_path),
        **observed_shape,
        "source_profile": protocol["source_profile"],
        "claim_scope": protocol["claim_scope"],
        "required_equalities": dict(required_equalities),
        "required_bundles": list(required_bundles),
        "required_feature_count": len(all_features),
        "required_feature_set_sha256": ordered_sha256(
            sorted(all_features)
        ),
        "target_nonnull_rows": target_nonnull,
        "split_counts": split_counts,
        "exploratory_override_confirmed": True,
    }
    return panel, audit


def apply_fixed_transforms(
    frame: pd.DataFrame,
    features: Sequence[str],
    protocol: Mapping[str, Any],
) -> pd.DataFrame:
    output = frame.copy()
    configured = set(
        protocol.get("preprocessing", {}).get("fixed_log1p_features", [])
    )
    for column in sorted(set(features).intersection(configured)):
        output[column] = output[column].astype(float)
        valid = output[column].notna()
        if (output.loc[valid, column] < 0).any():
            raise ValueError(f"{column} is negative before fixed log1p")
        output.loc[valid, column] = np.log1p(output.loc[valid, column])
    return output


def validate_training_feature_availability(
    train: pd.DataFrame, features: Sequence[str], *, label: str
) -> None:
    all_missing = [
        feature for feature in features if not train[feature].notna().any()
    ]
    if all_missing:
        raise ValueError(f"{label} has all-missing training features: {all_missing}")
    values = train[list(features)].to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError(f"{label} contains infinite feature values")


def select_linear_hyperparameters(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: Sequence[str],
    response: str,
    *,
    alpha_grid: Sequence[float],
    l1_ratios: Sequence[float],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Fit preprocessing on train and select Elastic Net by validation MSE."""

    validate_model_features(features)
    validate_training_feature_availability(train, features, label="selection train")
    imputer = SimpleImputer(strategy="median", add_indicator=False)
    scaler = StandardScaler()
    train_matrix = scaler.fit_transform(
        imputer.fit_transform(train[list(features)])
    )
    validation_matrix = scaler.transform(
        imputer.transform(validation[list(features)])
    )
    train_response = train[response].to_numpy(dtype=float)
    centered_response = train_response - float(train_response.mean())
    response_mean = float(train_response.mean())
    validation_response = validation[response].to_numpy(dtype=float)
    requested = np.asarray(
        sorted({float(value) for value in alpha_grid}, reverse=True)
    )
    candidates: list[dict[str, Any]] = []

    def evaluate(alphas: np.ndarray, *, boundary_expansion: bool) -> None:
        for ratio in l1_ratios:
            path_alphas, coefficients, _, iterations = enet_path(
                train_matrix,
                centered_response,
                l1_ratio=float(ratio),
                alphas=alphas,
                precompute=True,
                max_iter=100_000,
                return_n_iter=True,
            )
            for position, alpha in enumerate(path_alphas):
                prediction = (
                    response_mean + validation_matrix @ coefficients[:, position]
                )
                candidates.append(
                    {
                        "alpha": float(alpha),
                        "l1_ratio": float(ratio),
                        "validation_mse": quant_common.mse(
                            validation_response, prediction
                        ),
                        "path_iterations": int(iterations[position]),
                        "boundary_expansion": bool(boundary_expansion),
                    }
                )

    evaluate(requested, boundary_expansion=False)
    initial = min(candidates, key=lambda item: item["validation_mse"])
    minimum = float(requested.min())
    maximum = float(requested.max())
    expanded: list[float] = []
    if np.isclose(initial["alpha"], minimum):
        expanded.extend([minimum / 10, minimum / 3])
    if np.isclose(initial["alpha"], maximum):
        expanded.extend([maximum * 3, maximum * 10])
    if expanded:
        evaluate(
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
    return {
        "alpha": float(candidates[0]["alpha"]),
        "l1_ratio": float(candidates[0]["l1_ratio"]),
    }, candidates


def validate_prediction_panel(
    predictions: pd.DataFrame,
    *,
    protocol: Mapping[str, Any],
    split_index: int,
) -> dict[str, Any]:
    expected = expected_rows(protocol, split_index=split_index)
    folds = [fold["name"] for fold in protocol["folds"]]
    observed = {
        target: {
            fold: int(
                len(
                    predictions[
                        predictions["target"].eq(target)
                        & predictions["fold"].eq(fold)
                    ]
                )
            )
            for fold in folds
        }
        for target in protocol["targets"]
    }
    review = {
        "prediction_keys_unique": not bool(
            predictions.duplicated(PREDICTION_KEYS).any()
        ),
        "predictions_finite": bool(
            np.isfinite(
                predictions[
                    ["predicted_fisher_z", "predicted_correlation"]
                ].to_numpy(dtype=float)
            ).all()
        ),
        "prediction_bounds_valid": bool(
            predictions["predicted_correlation"].between(-1, 1).all()
        ),
        "expected_targets_present": set(predictions["target"])
        == set(protocol["targets"]),
        "expected_folds_present": set(predictions["fold"]) == set(folds),
        "observed_rows": observed,
        "expected_rows": expected,
        "row_counts_match": observed == expected,
    }
    if not all(
        review[key]
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


def compare_predictions(
    candidate: pd.DataFrame, base: pd.DataFrame
) -> list[dict[str, Any]]:
    columns = [
        *PREDICTION_KEYS,
        "actual_fisher_z",
        "predicted_fisher_z",
    ]
    merged = candidate[columns].merge(
        base[columns],
        on=PREDICTION_KEYS,
        how="outer",
        validate="one_to_one",
        suffixes=("_candidate", "_base"),
        indicator=True,
    )
    if merged.empty or not merged["_merge"].eq("both").all():
        raise ValueError(
            "Candidate/base prediction keys differ: "
            f"{merged['_merge'].value_counts().to_dict()}"
        )
    actual_delta = (
        merged["actual_fisher_z_candidate"]
        - merged["actual_fisher_z_base"]
    ).abs()
    if actual_delta.isna().any() or float(actual_delta.max()) > 1e-12:
        raise ValueError("Candidate/base actual targets differ")
    records = []
    for target, frame in merged.groupby("target", sort=True):
        actual = frame["actual_fisher_z_candidate"].to_numpy(dtype=float)
        candidate_prediction = frame[
            "predicted_fisher_z_candidate"
        ].to_numpy(dtype=float)
        base_prediction = frame["predicted_fisher_z_base"].to_numpy(dtype=float)
        candidate_loss = np.square(actual - candidate_prediction)
        base_loss = np.square(actual - base_prediction)
        records.append(
            {
                "target": target,
                "rows": len(frame),
                "base_fisher_z_rmse": float(np.sqrt(base_loss.mean())),
                "model_fisher_z_rmse": float(np.sqrt(candidate_loss.mean())),
                "incremental_r2": (
                    float(1 - candidate_loss.sum() / base_loss.sum())
                    if base_loss.sum() > 0
                    else math.nan
                ),
                "mean_squared_loss_delta_model_minus_base": float(
                    np.mean(candidate_loss - base_loss)
                ),
                "folds_model_better": int(
                    sum(
                        np.mean(
                            np.square(
                                fold_frame["actual_fisher_z_candidate"]
                                - fold_frame["predicted_fisher_z_candidate"]
                            )
                            - np.square(
                                fold_frame["actual_fisher_z_base"]
                                - fold_frame["predicted_fisher_z_base"]
                            )
                        )
                        < 0
                        for _, fold_frame in frame.groupby("fold", sort=True)
                    )
                ),
            }
        )
    return records


def initialize_status(protocol_sha256: str) -> dict[str, Any]:
    if STATUS_PATH.exists():
        status = load_json(STATUS_PATH)
        if status.get("protocol_sha256") != protocol_sha256:
            raise ValueError("Existing v2 status uses a different protocol hash")
        return status
    status = {
        "experiment_id": "quant-deterministic-news-v2",
        "protocol_sha256": protocol_sha256,
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "bundles": [
            {
                "order": bundle.order,
                "key": bundle.key,
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


def update_status(
    bundle_key: str, state: str, *, summary: str | None = None
) -> dict[str, Any]:
    allowed = {"planned", "running", "complete", "failed", "skipped"}
    if state not in allowed:
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
        "# Deterministic-news v2 training status",
        "",
        f"Protocol SHA-256: `{status['protocol_sha256']}`",
        "",
        "Each row is an isolated model bundle. Heavy predictions and fit records",
        "are stored under `outputs/quant_deterministic_news/v2`.",
        "",
        "| Order | Bundle | Status | Attempts | Result |",
        "|---:|---|---|---:|---|",
    ]
    for item in sorted(status["bundles"], key=lambda value: value["order"]):
        bundle = bundle_for(item["key"])
        link = (
            f"[results]({bundle.group}/{bundle.slug}/RESULTS.md)"
            if item["status"] in {"complete", "skipped"}
            else ""
        )
        result = " ".join(
            value for value in (item.get("summary") or "", link) if value
        )
        lines.append(
            f"| {item['order']} | {item['key']} - {item['label']} | "
            f"**{item['status']}** | {item['attempts']} | {result} |"
        )
    lines.extend(
        [
            "",
            "All retrospective-news v2 results are exploratory development",
            "estimates, not confirmatory point-in-time evidence.",
            "",
        ]
    )
    STATUS_MARKDOWN_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATUS_MARKDOWN_PATH.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(STATUS_MARKDOWN_PATH)


def ensure_no_existing_bundle(bundle_key: str, *, overwrite: bool) -> None:
    bundle = bundle_for(bundle_key)
    paths = (bundle.output_path, bundle.experiment_path)
    occupied = any(path.exists() and any(path.iterdir()) for path in paths)
    if occupied and not overwrite:
        raise FileExistsError(
            f"{bundle_key} already has output; pass --overwrite to replace it"
        )
    if occupied and overwrite:
        for path, root in (
            (bundle.output_path, OUTPUT_ROOT),
            (bundle.experiment_path, EXPERIMENT_ROOT),
        ):
            resolved = path.resolve()
            resolved_root = root.resolve()
            if resolved == resolved_root or resolved_root not in resolved.parents:
                raise ValueError(f"Unsafe cleanup path: {resolved}")
            if path.exists():
                shutil.rmtree(path)


def implementation_provenance() -> dict[str, Any]:
    source_paths = [
        ROOT / "scripts/quant_deterministic_news_training_v2/common.py",
        ROOT / "scripts/quant_deterministic_news_training_v2/run_linear.py",
        ROOT / "scripts/quant_deterministic_news_training_v2/run_xgboost.py",
        ROOT / "scripts/quant_deterministic_news_training_v2/run_controls.py",
        ROOT / "scripts/correlation_training/training_common.py",
        ROOT / "scripts/correlation_training/build_modeling_panel.py",
    ]
    files = {
        path.relative_to(ROOT).as_posix(): sha256_file(path)
        for path in source_paths
        if path.exists()
    }
    packages: dict[str, str | None] = {}
    for package in ("numpy", "pandas", "scikit-learn", "pyarrow", "xgboost"):
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


def verify_completed_bundle(bundle_key: str) -> dict[str, Any]:
    bundle = bundle_for(bundle_key)
    protocol_hash = sha256_file(PROTOCOL_PATH)
    panel_hash = sha256_file(PANEL_PATH)
    status = load_json(STATUS_PATH)
    if status.get("protocol_sha256") != protocol_hash:
        raise RuntimeError("Status ledger uses a stale v2 protocol")
    record = next(
        item for item in status["bundles"] if item["key"] == bundle_key
    )
    if record["status"] != "complete":
        raise RuntimeError(f"{bundle_key} is not complete")
    manifest_path = bundle.output_path / "manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("bundle") != bundle_key:
        raise RuntimeError(f"{bundle_key} manifest names another bundle")
    if manifest.get("protocol_sha256") != protocol_hash:
        raise RuntimeError(f"{bundle_key} manifest uses a stale protocol")
    if manifest.get("panel_sha256") != panel_hash:
        raise RuntimeError(f"{bundle_key} manifest uses a stale panel")
    return {
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest": manifest,
    }


def load_completed_predictions(
    bundle_key: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    provenance = verify_completed_bundle(bundle_key)
    manifest = provenance.pop("manifest")
    artifact = manifest.get("artifacts", {}).get("predictions.parquet")
    if not isinstance(artifact, Mapping):
        raise RuntimeError(f"{bundle_key} predictions are absent from manifest")
    path = bundle_for(bundle_key).output_path / "predictions.parquet"
    if Path(str(artifact.get("path"))).resolve() != path.resolve():
        raise RuntimeError(f"{bundle_key} prediction path is noncanonical")
    _verify_hash(path, str(artifact.get("sha256")), f"{bundle_key} predictions")
    frame = pd.read_parquet(path)
    frame["forecast_date"] = pd.to_datetime(
        frame["forecast_date"]
    ).dt.normalize()
    if frame.duplicated(PREDICTION_KEYS).any():
        raise RuntimeError(f"{bundle_key} predictions have duplicate keys")
    return frame, provenance


def _result_markdown(
    bundle: Bundle,
    summary: Mapping[str, Any],
    review: Mapping[str, Any],
    refs: Mapping[str, Any],
) -> str:
    outer_evaluation_excluded = review.get(
        "outer_evaluation_excluded_from_tuning",
        review.get("outer_evaluation_excluded_from_gate_and_tuning"),
    )
    lines = [
        f"# {bundle.label}",
        "",
        f"Status: **{summary.get('status', 'complete')}**",
        "",
        f"Completed: `{summary.get('generated_at_utc', '')}`",
        "",
    ]
    if summary.get("status") == "skipped":
        lines.extend(
            [
                f"Reason: {summary.get('reason', 'not constructable')}",
                "",
            ]
        )
    metrics = summary.get("metrics", [])
    if metrics:
        lines.extend(
            [
                "| Target | Rows | Fisher-z RMSE | Fisher-z MAE | Raw RMSE | OOS R2 vs persistence |",
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
    comparisons = summary.get("base_comparison", [])
    if comparisons:
        lines.extend(
            [
                "",
                "## Exact matched Q comparison",
                "",
                "| Target | Q RMSE | Model RMSE | Incremental R2 | Loss delta | Better folds |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for item in comparisons:
            lines.append(
                f"| {item['target']} | {item['base_fisher_z_rmse']:.6f} | "
                f"{item['model_fisher_z_rmse']:.6f} | "
                f"{item['incremental_r2']:.6f} | "
                f"{item['mean_squared_loss_delta_model_minus_base']:.8f} | "
                f"{item['folds_model_better']} |"
            )
    secondary = summary.get("secondary_comparison", [])
    if secondary:
        lines.extend(
            [
                "",
                f"## {summary.get('secondary_comparison_label', 'Secondary comparison')}",
                "",
                "| Target | Base RMSE | Model RMSE | Incremental R2 | Loss delta | Better folds |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for item in secondary:
            lines.append(
                f"| {item['target']} | {item['base_fisher_z_rmse']:.6f} | "
                f"{item['model_fisher_z_rmse']:.6f} | "
                f"{item['incremental_r2']:.6f} | "
                f"{item['mean_squared_loss_delta_model_minus_base']:.8f} | "
                f"{item['folds_model_better']} |"
            )
    lines.extend(
        [
            "",
            "## Review",
            "",
            f"- Review status: **{review.get('status', 'passed')}**",
            f"- Prediction keys unique: `{review.get('prediction_keys_unique')}`",
            f"- Predictions finite: `{review.get('predictions_finite')}`",
            f"- Bounds valid: `{review.get('prediction_bounds_valid')}`",
            f"- Outer evaluation excluded from tuning/gating: `{outer_evaluation_excluded}`",
            "",
            "## Artifacts",
            "",
        ]
    )
    for name, value in refs.items():
        if isinstance(value, Mapping) and "path" in value:
            lines.append(
                f"- {name}: `{value['path']}` - SHA-256 `{value['sha256']}`"
            )
    lines.extend(
        [
            "",
            "This is an exploratory development result from retrospective news.",
            "It is not confirmatory point-in-time evidence.",
            "",
        ]
    )
    return "\n".join(lines)


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
    dependencies: Mapping[str, Any],
    extra_outputs: Mapping[str, pd.DataFrame | object] | None = None,
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
            output / "validation_predictions.parquet", validation_predictions
        )
    quant_common.write_json(output / "fits.json", fits)
    quant_common.write_json(output / "fold_metrics.json", fold_metrics)
    for name, value in (extra_outputs or {}).items():
        path = output / name
        if isinstance(value, pd.DataFrame):
            quant_common.write_parquet(path, value)
        else:
            quant_common.write_json(path, value)
    output_files = sorted(
        path
        for path in output.iterdir()
        if path.is_file() and path.name != "manifest.json"
    )
    refs = {
        path.name: {"path": path.as_posix(), "sha256": sha256_file(path)}
        for path in output_files
    }
    protocol = load_protocol()
    sources = protocol["source_artifacts"]
    manifest = {
        "bundle": bundle_key,
        "generated_at_utc": utc_now(),
        "protocol_path": PROTOCOL_PATH.as_posix(),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "panel_path": PANEL_PATH.as_posix(),
        "panel_sha256": sha256_file(PANEL_PATH),
        "panel_manifest_path": str(sources["panel_manifest_path"]),
        "panel_manifest_sha256": str(sources["panel_manifest_sha256"]),
        "source_profile": protocol["source_profile"],
        "claim_scope": protocol["claim_scope"],
        "model_config": model_config,
        "dependencies": dict(dependencies),
        "implementation_provenance": implementation_provenance(),
        "artifacts": refs,
    }
    quant_common.write_json(output / "manifest.json", manifest)
    refs["manifest.json"] = {
        "path": (output / "manifest.json").as_posix(),
        "sha256": sha256_file(output / "manifest.json"),
    }
    completed_summary = {
        **summary,
        "status": summary.get("status", "complete"),
        "generated_at_utc": summary.get("generated_at_utc", utc_now()),
    }
    completed_review = {
        **review,
        "status": review.get("status", "passed"),
    }
    quant_common.write_json(tracked / "config.json", model_config)
    quant_common.write_json(tracked / "review.json", completed_review)
    quant_common.write_json(tracked / "summary.json", completed_summary)
    quant_common.write_json(tracked / "artifact_refs.json", refs)
    for target in sorted(
        {item["target"] for item in completed_summary.get("metrics", [])}
    ):
        target_metrics = [
            item
            for item in completed_summary["metrics"]
            if item["target"] == target
        ]
        quant_common.write_json(tracked / target / "metrics.json", target_metrics)
    result = _result_markdown(
        bundle, completed_summary, completed_review, refs
    )
    temporary = (tracked / "RESULTS.md").with_suffix(".md.tmp")
    temporary.write_text(result, encoding="utf-8")
    temporary.replace(tracked / "RESULTS.md")
    return refs


def summary_status_text(summary: Mapping[str, Any]) -> str:
    metrics = summary.get("metrics", [])
    return "Fisher-z RMSE: " + ", ".join(
        f"{item['target']} {item['fisher_z_rmse']:.4f}"
        for item in metrics
    )
