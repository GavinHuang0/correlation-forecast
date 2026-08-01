"""Shared validation, fitting, and artifact helpers for v4.

All functions in this module are outcome-agnostic.  The only historical model
artifacts they read are the frozen v3 matched bases and the quant-v2 rung-03
out-of-sample prediction stream declared by the locked v4 protocol.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import time
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from scripts.correlation_training import training_common as quant_common
from scripts.quant_deterministic_news_training_v4 import contract


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def ordered_sha256(values: Sequence[str]) -> str:
    return sha256_json(list(values))


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _replace_with_retry(source: Path, target: Path) -> None:
    error: PermissionError | None = None
    for attempt in range(8):
        try:
            os.replace(source, target)
            return
        except PermissionError as caught:
            error = caught
            time.sleep(0.05 * (attempt + 1))
    if error is not None:
        raise error


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        _replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w+b",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False)
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        _replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def artifact_record(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
    }
    if rows is not None:
        record["rows"] = int(rows)
    return record


def bundle_for(key: str) -> contract.Bundle:
    try:
        return contract.BUNDLE_BY_KEY[key]
    except KeyError as error:
        raise ValueError(f"Unknown v4 bundle {key}") from error


def _string_tuple(value: Any, label: str, count: int) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or len(value) != count
        or len(set(value)) != count
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise ValueError(f"{label} must contain {count} unique names")
    return tuple(value)


def verify_artifact(record: Mapping[str, Any], *, label: str) -> Path:
    path = Path(str(record.get("path", "")))
    expected = str(record.get("sha256", ""))
    if not path.is_file() or len(expected) != 64 or sha256_file(path) != expected:
        raise ValueError(f"{label} is missing or hash-invalid")
    expected_bytes = record.get("bytes")
    if expected_bytes is not None and int(expected_bytes) != path.stat().st_size:
        raise ValueError(f"{label} byte count changed")
    return path


def load_protocol(path: Path = contract.PROTOCOL_PATH) -> dict[str, Any]:
    if not path.is_file() or not contract.PROTOCOL_SIDECAR.is_file():
        raise FileNotFoundError("Lock the v4 protocol before fitting")
    expected = contract.PROTOCOL_SIDECAR.read_text(encoding="ascii").split()[0]
    if expected != sha256_file(path):
        raise ValueError("The v4 protocol sidecar is stale")
    protocol = load_json(path)
    if (
        protocol.get("experiment_id") != "quant-deterministic-news-v4-soft-route"
        or protocol.get("protocol_version") != "4.0"
        or protocol.get("status") != "locked_before_training"
    ):
        raise ValueError("Unexpected or unlocked v4 protocol")
    if tuple(protocol.get("targets", ())) != contract.TARGETS:
        raise ValueError("The v4 target order changed")
    if tuple(protocol.get("join_keys", ())) != contract.PANEL_KEYS:
        raise ValueError("The v4 join key order changed")
    if tuple(protocol.get("short_folds", ())) != contract.SHORT_FOLDS:
        raise ValueError("The v4 short fold contract changed")
    if tuple(protocol.get("long_test_folds", ())) != contract.LONG_TEST_FOLDS:
        raise ValueError("The v4 long fold contract changed")

    blocks = protocol.get("feature_blocks")
    if not isinstance(blocks, Mapping):
        raise ValueError("The v4 feature blocks are missing")
    if _string_tuple(blocks.get("soft_route_19"), "soft_route_19", 19) != contract.SOFT_ROUTE_19:
        raise ValueError("SoftRoute19 order changed")
    if _string_tuple(blocks.get("current_9"), "current_9", 9) != contract.CURRENT_9:
        raise ValueError("Current9 order changed")
    if _string_tuple(blocks.get("coupling_6"), "coupling_6", 6) != contract.COUPLING_6:
        raise ValueError("Coupling6 order changed")
    if _string_tuple(blocks.get("quality_4"), "quality_4", 4) != contract.QUALITY_4:
        raise ValueError("Quality4 order changed")
    if (
        _string_tuple(blocks.get("quality_only_5"), "quality_only_5", 5)
        != contract.QUALITY_ONLY_5
    ):
        raise ValueError("QualityOnly5 order changed")
    q = blocks.get("q56_by_target")
    if not isinstance(q, Mapping) or set(q) != set(contract.TARGETS):
        raise ValueError("Q56 blocks are malformed")
    for target in contract.TARGETS:
        _string_tuple(q[target], f"q56_by_target.{target}", 56)
    _string_tuple(blocks.get("d2_normalized_30"), "d2_normalized_30", 30)

    implementation = protocol.get("implementation")
    if not isinstance(implementation, Mapping):
        raise ValueError("The v4 implementation binding is missing")
    expected_paths = {path.as_posix() for path in contract.BOUND_IMPLEMENTATION}
    if set(implementation) != expected_paths:
        raise ValueError("The v4 implementation file set changed")
    for implementation_path in contract.BOUND_IMPLEMENTATION:
        expected_hash = str(implementation.get(implementation_path.as_posix(), ""))
        if (
            not implementation_path.is_file()
            or sha256_file(implementation_path) != expected_hash
        ):
            raise ValueError(f"v4 implementation drift: {implementation_path}")
    return protocol


def q_features(target: str, protocol: Mapping[str, Any]) -> tuple[str, ...]:
    return _string_tuple(
        protocol["feature_blocks"]["q56_by_target"][target],
        f"q56_by_target.{target}",
        56,
    )


def d_features(protocol: Mapping[str, Any]) -> tuple[str, ...]:
    return _string_tuple(
        protocol["feature_blocks"]["d2_normalized_30"],
        "d2_normalized_30",
        30,
    )


def model_features(
    bundle_key: str, target: str, protocol: Mapping[str, Any]
) -> tuple[str, ...]:
    bundle = bundle_for(bundle_key)
    q = q_features(target, protocol)
    d = d_features(protocol)
    if bundle.features == "q_l19":
        features = (*q, *contract.SOFT_ROUTE_19)
    elif bundle.features == "q_d2_l19":
        features = (*q, *d, *contract.SOFT_ROUTE_19)
    elif bundle.features == "q_current9":
        features = (*q, *contract.CURRENT_9)
    elif bundle.features == "q_quality5":
        features = (*q, *contract.QUALITY_ONLY_5)
    elif bundle.features == "q_d2_quality5":
        features = (*q, *d, *contract.QUALITY_ONLY_5)
    elif bundle.features == "c6":
        features = contract.COUPLING_6
    elif bundle.features == "l19":
        features = contract.SOFT_ROUTE_19
    elif bundle.features == "quality5":
        features = contract.QUALITY_ONLY_5
    elif bundle.features == "base_l19":
        features = ("_base_fisher_z", *contract.SOFT_ROUTE_19)
    elif bundle.features == "base":
        features = ("_base_fisher_z",)
    else:
        raise ValueError(f"Unsupported v4 feature architecture {bundle.features}")
    if len(features) != len(set(features)):
        raise ValueError(f"{bundle_key} has duplicate features")
    return tuple(features)


def panel_source_key(bundle_key: str) -> str:
    control = bundle_for(bundle_key).control
    if control == "stale20":
        return "stale20_panel"
    if control == "wrong_stock":
        return "wrong_stock_panel"
    if control == "permuted":
        return "permuted_panel"
    return "canonical_panel"


def normalize_panel(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["forecast_date"] = pd.to_datetime(frame["forecast_date"], errors="raise").dt.normalize()
    for name in [name for name in frame.columns if name.endswith("_end_date")]:
        frame[name] = pd.to_datetime(frame[name], errors="coerce").dt.normalize()
    if frame.duplicated(list(contract.PANEL_KEYS)).any():
        raise ValueError("The v4 panel contains duplicate join keys")
    return frame.sort_values(list(contract.PANEL_KEYS), kind="mergesort").reset_index(drop=True)


def load_panel(
    protocol: Mapping[str, Any], bundle_key: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_key = panel_source_key(bundle_key)
    record = protocol.get("source_artifacts", {}).get(source_key)
    if not isinstance(record, Mapping):
        raise ValueError(f"Protocol lacks {source_key}")
    path = verify_artifact(record, label=source_key)
    frame = normalize_panel(path)
    required: set[str] = set(contract.PANEL_KEYS)
    required.update(contract.SOFT_ROUTE_19)
    required.update(contract.COUPLING_6)
    required.update(contract.QUALITY_4)
    for target in contract.TARGETS:
        required.update(q_features(target, protocol))
    required.update(d_features(protocol))
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"The v4 panel misses columns: {sorted(missing)}")
    numeric_features = sorted(required - set(contract.PANEL_KEYS))
    numeric = frame[numeric_features].apply(pd.to_numeric, errors="coerce")
    for name in numeric_features:
        if numeric[name].notna().sum() != frame[name].notna().sum():
            raise TypeError(f"The v4 feature {name} is nonnumeric")
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError("The v4 feature panel contains infinities")
    if frame[list(d_features(protocol))].isna().any().any():
        raise ValueError("D2-Normalized must remain complete-case")
    if not frame["lsoft_observed_no_selected_article"].isin([0, 1]).all():
        raise ValueError("The v4 no-selected indicator is not binary")
    quant_common.validate_panel_information_set(frame)
    expected_rows = record.get("rows")
    if expected_rows is not None and len(frame) != int(expected_rows):
        raise ValueError(f"{source_key} row count changed")
    return frame, {
        "source_key": source_key,
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "rows": len(frame),
        "dates": int(frame["forecast_date"].nunique()),
        "stocks": int(frame["stock"].nunique()),
        "sectors": int(frame["sector"].nunique()),
    }


def apply_fixed_transforms(
    frame: pd.DataFrame,
    features: Sequence[str],
    protocol: Mapping[str, Any],
) -> pd.DataFrame:
    output = frame.copy()
    configured = set(protocol.get("preprocessing", {}).get("fixed_log1p_features", ()))
    for name in sorted(set(features) & configured):
        values = pd.to_numeric(output[name], errors="coerce")
        if (values.dropna() < 0).any():
            raise ValueError(f"{name} is negative before log1p")
        output[name] = np.log1p(values)
    return output


def validate_training_features(
    frame: pd.DataFrame, features: Sequence[str], label: str
) -> None:
    missing = [name for name in features if name not in frame]
    if missing:
        raise ValueError(f"{label} misses features: {missing}")
    all_missing = [name for name in features if not frame[name].notna().any()]
    if all_missing:
        raise ValueError(f"{label} has all-missing features: {all_missing}")
    if np.isinf(frame[list(features)].to_numpy(dtype=float)).any():
        raise ValueError(f"{label} contains feature infinities")


def select_elastic_net(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: Sequence[str],
    response: str,
    protocol: Mapping[str, Any],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Select only from the exact locked alpha-by-L1 grid.

    V4 deliberately does not inherit the quant-v2 helper's adaptive boundary
    expansion.  Every Elastic-Net architecture therefore receives the same
    exhaustive, hash-bound candidate set.
    """

    tuning = protocol["model_tuning"]
    candidates: list[dict[str, Any]] = []
    actual = validation[response].to_numpy(dtype=float)
    for alpha in tuning["alpha_grid"]:
        for l1_ratio in tuning["l1_ratio_grid"]:
            model = quant_common.linear_pipeline(
                "elastic_net",
                alpha=float(alpha),
                l1_ratio=float(l1_ratio),
            )
            model.fit(train[list(features)], train[response])
            predicted = model.predict(validation[list(features)])
            candidates.append(
                {
                    "alpha": float(alpha),
                    "l1_ratio": float(l1_ratio),
                    "validation_mse": float(
                        np.mean(np.square(actual - predicted))
                    ),
                    "boundary_expansion": False,
                }
            )
    candidates.sort(
        key=lambda row: (
            row["validation_mse"],
            row["alpha"],
            row["l1_ratio"],
        )
    )
    return {
        "alpha": float(candidates[0]["alpha"]),
        "l1_ratio": float(candidates[0]["l1_ratio"]),
    }, candidates


def load_verified_v3_base(
    protocol: Mapping[str, Any], base_key: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_key = {"S0": "v3_s0_predictions", "S1": "v3_s1_predictions"}[base_key]
    record = protocol["source_artifacts"].get(source_key)
    if not isinstance(record, Mapping):
        raise ValueError(f"Protocol lacks {source_key}")
    path = verify_artifact(record, label=source_key)
    manifest_record = record.get("manifest")
    if not isinstance(manifest_record, Mapping):
        raise ValueError(f"{source_key} lacks its manifest binding")
    verify_artifact(manifest_record, label=f"{source_key} manifest")
    manifest = load_json(Path(str(manifest_record["path"])))
    artifact = manifest.get("artifacts", {}).get("predictions.parquet")
    if not isinstance(artifact, Mapping) or artifact.get("sha256") != record.get("sha256"):
        raise ValueError(f"{source_key} manifest does not bind predictions")
    frame = pd.read_parquet(path)
    frame["forecast_date"] = pd.to_datetime(frame["forecast_date"]).dt.normalize()
    if frame.duplicated(list(contract.PREDICTION_KEYS)).any():
        raise ValueError(f"{source_key} has duplicate prediction keys")
    return frame, {
        "base_key": base_key,
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "manifest_path": str(manifest_record["path"]),
        "manifest_sha256": str(manifest_record["sha256"]),
    }


def validate_matched_prediction_rows(
    candidate: pd.DataFrame, base: pd.DataFrame
) -> dict[str, Any]:
    keys = list(contract.PREDICTION_KEYS)
    left = candidate[keys].sort_values(keys, kind="mergesort").reset_index(drop=True)
    right = base[keys].sort_values(keys, kind="mergesort").reset_index(drop=True)
    if not left.equals(right):
        raise ValueError("Candidate and matched base prediction keys differ")
    merged = candidate.merge(base, on=keys, suffixes=("_candidate", "_base"), validate="one_to_one")
    actual_gap = np.max(
        np.abs(
            merged["actual_fisher_z_candidate"].to_numpy(dtype=float)
            - merged["actual_fisher_z_base"].to_numpy(dtype=float)
        )
    )
    if actual_gap > 1e-12:
        raise ValueError("Candidate and base actual targets differ")
    return {"rows": len(merged), "maximum_actual_fisher_z_gap": float(actual_gap)}


def load_long_base(
    protocol: Mapping[str, Any], panel: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    record = protocol["source_artifacts"].get("quant_v2_rung03_predictions")
    if not isinstance(record, Mapping):
        raise ValueError("Protocol lacks quant-v2 predictions")
    path = verify_artifact(record, label="quant-v2 rung03 predictions")
    base = pd.read_parquet(path)
    base["forecast_date"] = pd.to_datetime(base["forecast_date"]).dt.normalize()
    base = base[
        base["model"].eq(contract.LONG_ANCHOR_MODEL)
        & base["target"].isin(contract.TARGETS)
        & base["fold"].isin(
            tuple(f"fold_{index:02d}" for index in range(6, 14))
        )
        & base["forecast_date"].ge(pd.Timestamp("2022-11-01"))
    ].copy()
    if base.empty or base.duplicated(list(contract.PREDICTION_KEYS)).any():
        raise ValueError("The quant-v2 XGBoost OOS anchor is empty or duplicated")
    specs = {spec.name: spec for spec in quant_common.target_specs()}
    response_columns = [spec.response_column for spec in specs.values()]
    feature_columns = [
        *contract.PANEL_KEYS,
        *contract.SOFT_ROUTE_19,
        *contract.COUPLING_6,
        *contract.QUALITY_4,
        *response_columns,
    ]
    joined = base.merge(
        panel[feature_columns],
        on=list(contract.PANEL_KEYS),
        how="inner",
        validate="many_to_one",
    )
    if len(joined) != len(base):
        missing = len(base) - len(joined)
        raise ValueError(f"The v4 panel misses {missing} news-era quant-v2 OOS rows")
    target_parity: dict[str, Any] = {}
    for target in contract.TARGETS:
        response_column = specs[target].response_column
        target_rows = joined["target"].eq(target)
        saved = joined.loc[target_rows, "actual_fisher_z"].to_numpy(dtype=float)
        panel_actual = joined.loc[target_rows, response_column].to_numpy(dtype=float)
        if len(saved) == 0 or np.isnan(panel_actual).any():
            raise ValueError(f"The long anchor has no valid target parity rows for {target}")
        bitwise_equal = bool(np.array_equal(saved, panel_actual))
        maximum_gap = float(np.max(np.abs(saved - panel_actual)))
        target_parity[target] = {
            "rows": len(saved),
            "bitwise_equal": bitwise_equal,
            "maximum_absolute_fisher_z_gap": maximum_gap,
            "panel_response_column": response_column,
        }
        if not bitwise_equal or maximum_gap != 0.0:
            raise ValueError(
                f"The quant-v2 saved actual target differs from the v4 panel for {target}"
            )
    joined["_base_fisher_z"] = joined["predicted_fisher_z"].astype(float)
    joined["_actual_fisher_z"] = joined["actual_fisher_z"].astype(float)
    joined["_residual_fisher_z"] = (
        joined["_actual_fisher_z"] - joined["_base_fisher_z"]
    )
    counts = (
        joined.groupby(["target", "fold"], sort=True)
        .size()
        .rename("rows")
        .reset_index()
    )
    return joined, {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "anchor_model": contract.LONG_ANCHOR_MODEL,
        "rows": len(joined),
        "fold_target_rows": counts.to_dict(orient="records"),
        "target_parity": target_parity,
    }


def long_split(
    frame: pd.DataFrame, target: str, test_fold: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    test_number = int(test_fold.split("_")[1])
    if test_fold not in contract.LONG_TEST_FOLDS:
        raise ValueError(f"Unsupported long test fold {test_fold}")
    validation_fold = f"fold_{test_number - 1:02d}"
    training_folds = tuple(
        f"fold_{number:02d}"
        for number in range(contract.LONG_FIRST_TRAIN_FOLD, test_number - 1)
    )
    target_frame = frame[frame["target"].eq(target)]
    train = target_frame[target_frame["fold"].isin(training_folds)].copy()
    validation = target_frame[target_frame["fold"].eq(validation_fold)].copy()
    test = target_frame[target_frame["fold"].eq(test_fold)].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError(
            f"Empty long split {target}/{test_fold}: "
            f"{len(train)}/{len(validation)}/{len(test)}"
        )
    if not (
        train["forecast_date"].max() < validation["forecast_date"].min()
        and validation["forecast_date"].max() < test["forecast_date"].min()
    ):
        raise ValueError("The long-history prequential split is not chronological")
    return train, validation, test, {
        "training_folds": list(training_folds),
        "validation_fold": validation_fold,
        "test_fold": test_fold,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
    }


def long_prediction_frame(
    frame: pd.DataFrame,
    predicted_fisher_z: np.ndarray | pd.Series,
    *,
    model_name: str,
    outer_fold: str,
) -> pd.DataFrame:
    output = frame[
        [
            "target",
            *contract.PANEL_KEYS,
            "actual_fisher_z",
            "actual_correlation",
            "persistence_correlation",
            "persistence_fisher_z",
        ]
    ].copy()
    predicted = np.asarray(predicted_fisher_z, dtype=float)
    if len(predicted) != len(output) or not np.isfinite(predicted).all():
        raise ValueError("Long prediction vector is invalid")
    output["predicted_fisher_z"] = predicted
    output["predicted_correlation"] = np.tanh(predicted)
    output.insert(0, "model", model_name)
    output.insert(0, "fold", outer_fold)
    return output[
        [
            "fold",
            "target",
            "model",
            *contract.PANEL_KEYS,
            "actual_fisher_z",
            "actual_correlation",
            "predicted_fisher_z",
            "predicted_correlation",
            "persistence_correlation",
            "persistence_fisher_z",
        ]
    ]


def validate_prediction_panel(
    frame: pd.DataFrame,
    *,
    expected_folds: Sequence[str],
) -> dict[str, Any]:
    review = {
        "rows": len(frame),
        "prediction_keys_unique": not bool(frame.duplicated(list(contract.PREDICTION_KEYS)).any()),
        "predictions_finite": bool(
            np.isfinite(
                frame[["predicted_fisher_z", "predicted_correlation"]].to_numpy(dtype=float)
            ).all()
        ),
        "prediction_bounds_valid": bool(frame["predicted_correlation"].between(-1, 1).all()),
        "targets_exact": set(frame["target"]) == set(contract.TARGETS),
        "folds_exact": set(frame["fold"]) == set(expected_folds),
    }
    if not all(value for key, value in review.items() if key != "rows"):
        raise AssertionError(f"Prediction review failed: {review}")
    return review


def runtime_versions() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for package in ("numpy", "pandas", "scikit-learn", "pyarrow"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
    }


def ensure_output_absent(bundle_key: str, *, overwrite: bool) -> None:
    bundle = bundle_for(bundle_key)
    existing = [path for path in (bundle.output_path, bundle.experiment_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"v4 bundle already exists: {existing}")
    if overwrite:
        for path in existing:
            shutil.rmtree(path)


def result_markdown(bundle_key: str, summary: Mapping[str, Any]) -> str:
    bundle = bundle_for(bundle_key)
    lines = [
        f"# {bundle.label}",
        "",
        f"Status: **{summary.get('status', 'unknown')}**.",
        "",
        "This is an exploratory, retrospective v4 result. It does not alter the",
        "completed v3 experiment and is not confirmatory point-in-time evidence.",
        "",
        "| Target | Rows | Fisher-z RMSE | OOS R2 vs persistence |",
        "|---|---:|---:|---:|",
    ]
    for row in summary.get("metrics", []):
        lines.append(
            "| {target} | {rows} | {fisher_z_rmse:.6f} | {oos_r2_vs_persistence:.6f} |".format(
                **row
            )
        )
    return "\n".join(lines) + "\n"


def write_bundle(
    bundle_key: str,
    *,
    predictions: pd.DataFrame,
    validation_predictions: pd.DataFrame,
    fits: Sequence[Mapping[str, Any]],
    model_config: Mapping[str, Any],
    dependencies: Mapping[str, Any],
) -> dict[str, Any]:
    protocol = load_protocol()
    bundle = bundle_for(bundle_key)
    output = bundle.output_path
    output.mkdir(parents=True, exist_ok=False)
    atomic_parquet(output / "predictions.parquet", predictions)
    atomic_parquet(output / "validation_predictions.parquet", validation_predictions)
    atomic_json(output / "fits.json", list(fits))
    metrics = quant_common.summarize_predictions(predictions).to_dict(orient="records")
    fold_metrics = quant_common.summarize_predictions(
        predictions, ("fold", "target", "model")
    ).to_dict(orient="records")
    atomic_json(output / "fold_metrics.json", fold_metrics)
    atomic_json(output / "model_config.json", dict(model_config))
    summary = {
        "status": "complete",
        "bundle": bundle_key,
        "label": bundle.label,
        "metrics": metrics,
        "result_role": "exploratory_non_version_safe_v4",
    }
    review = {
        "status": "passed",
        "outer_test_excluded_from_tuning": True,
        "train_only_preprocessing_for_validation": True,
        "train_validation_refit_before_test": True,
        "whole_dates_kept_together": True,
        "retrospective_non_version_safe": True,
        "runtime": runtime_versions(),
    }
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "review.json", review)
    atomic_json(output / "dependencies.json", dict(dependencies))
    artifacts: dict[str, Any] = {}
    for name in (
        "predictions.parquet",
        "validation_predictions.parquet",
        "fits.json",
        "fold_metrics.json",
        "model_config.json",
        "summary.json",
        "review.json",
        "dependencies.json",
    ):
        rows = None
        if name == "predictions.parquet":
            rows = len(predictions)
        elif name == "validation_predictions.parquet":
            rows = len(validation_predictions)
        artifacts[name] = artifact_record(output / name, rows=rows)
    manifest = {
        "manifest_version": "quant-deterministic-news-v4-bundle-v1",
        "generated_at_utc": utc_now(),
        "bundle": bundle_key,
        "label": bundle.label,
        "protocol_path": contract.PROTOCOL_PATH.as_posix(),
        "protocol_sha256": sha256_file(contract.PROTOCOL_PATH),
        "claim_flags": protocol["claim_flags"],
        "artifacts": artifacts,
        "dependencies": dict(dependencies),
    }
    atomic_json(output / "manifest.json", manifest)
    atomic_text(
        output / "manifest.sha256",
        f"{sha256_file(output / 'manifest.json')}  manifest.json\n",
    )

    tracked = bundle.experiment_path
    tracked.mkdir(parents=True, exist_ok=False)
    for name in ("summary.json", "review.json", "model_config.json"):
        shutil.copy2(output / name, tracked / name)
    atomic_text(tracked / "RESULTS.md", result_markdown(bundle_key, summary))
    atomic_json(
        tracked / "artifact_pointer.json",
        {
            "manifest_path": (output / "manifest.json").as_posix(),
            "manifest_sha256": sha256_file(output / "manifest.json"),
            "protocol_sha256": sha256_file(contract.PROTOCOL_PATH),
        },
    )
    return manifest


def load_completed_bundle(bundle_key: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    bundle = bundle_for(bundle_key)
    manifest_path = bundle.output_path / "manifest.json"
    sidecar = bundle.output_path / "manifest.sha256"
    if not manifest_path.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"Incomplete v4 bundle {bundle_key}")
    expected = sidecar.read_text(encoding="ascii").split()[0]
    if expected != sha256_file(manifest_path):
        raise ValueError(f"Stale v4 manifest sidecar for {bundle_key}")
    manifest = load_json(manifest_path)
    if (
        manifest.get("bundle") != bundle_key
        or manifest.get("protocol_sha256") != sha256_file(contract.PROTOCOL_PATH)
    ):
        raise ValueError(f"Stale v4 bundle manifest {bundle_key}")
    prediction_record = manifest.get("artifacts", {}).get("predictions.parquet")
    if not isinstance(prediction_record, Mapping):
        raise ValueError(f"{bundle_key} lacks saved predictions")
    path = verify_artifact(prediction_record, label=f"{bundle_key} predictions")
    return pd.read_parquet(path), {
        "bundle": bundle_key,
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "predictions_path": path.as_posix(),
        "predictions_sha256": sha256_file(path),
    }


def incremental_mse_r2(candidate: pd.DataFrame, base: pd.DataFrame) -> float:
    keys = list(contract.PREDICTION_KEYS)
    merged = candidate.merge(base, on=keys, suffixes=("_candidate", "_base"), validate="one_to_one")
    candidate_loss = np.square(
        merged["actual_fisher_z_candidate"] - merged["predicted_fisher_z_candidate"]
    )
    base_loss = np.square(
        merged["actual_fisher_z_base"] - merged["predicted_fisher_z_base"]
    )
    denominator = float(base_loss.mean())
    return float(1 - candidate_loss.mean() / denominator) if denominator > 0 else math.nan
