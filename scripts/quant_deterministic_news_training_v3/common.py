"""Shared contracts and artifacts for exploratory v3 semantic training."""

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
from scripts.quant_deterministic_news_training_v2 import common as v2_common
from scripts.quant_deterministic_news_training_v3 import contract


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
        raise TypeError(f"{path} must contain an object")
    return value


def _replace_with_retry(source: Path, target: Path) -> None:
    last_error: PermissionError | None = None
    for attempt in range(8):
        try:
            os.replace(source, target)
            return
        except PermissionError as error:
            last_error = error
            time.sleep(0.05 * (attempt + 1))
    if last_error is not None:
        raise last_error


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


def bundle_for(key: str) -> contract.Bundle:
    try:
        return contract.BUNDLE_BY_KEY[key]
    except KeyError as error:
        raise ValueError(f"Unknown v3 bundle {key}") from error


def _string_tuple(value: Any, label: str, count: int) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or len(value) != count
        or len(set(value)) != count
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise ValueError(f"{label} must contain {count} unique names")
    return tuple(value)


def verify_locked_implementation(
    protocol: Mapping[str, Any],
) -> dict[str, str]:
    locked = protocol.get("implementation")
    if not isinstance(locked, Mapping):
        raise ValueError("v3 protocol lacks implementation hashes")
    expected_paths = tuple(contract.BOUND_TRAINING_IMPLEMENTATION)
    expected_keys = {path.as_posix() for path in expected_paths}
    if set(locked) != expected_keys:
        raise ValueError("v3 bound implementation file set changed")
    verified: dict[str, str] = {}
    for path in expected_paths:
        key = path.as_posix()
        expected_hash = locked.get(key)
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or not path.is_file()
        ):
            raise ValueError(f"v3 implementation binding is invalid: {key}")
        observed_hash = sha256_file(path)
        if observed_hash != expected_hash:
            raise ValueError(f"v3 implementation drift detected: {key}")
        verified[key] = observed_hash
    return verified


def load_protocol(
    path: Path = contract.PROTOCOL_PATH,
) -> dict[str, Any]:
    if not path.is_file() or not contract.PROTOCOL_SIDECAR.is_file():
        raise FileNotFoundError(
            "The v3 protocol and sidecar must be locked before fitting"
        )
    sidecar_hash = (
        contract.PROTOCOL_SIDECAR.read_text(encoding="ascii")
        .strip()
        .split()[0]
    )
    if sidecar_hash != sha256_file(path):
        raise ValueError("The v3 protocol sidecar hash is stale")
    protocol = load_json(path)
    if (
        protocol.get("experiment_id")
        != "quant-deterministic-news-v3-w17-lite"
        or protocol.get("protocol_version") != "3.0"
        or protocol.get("status") != "locked_before_training"
    ):
        raise ValueError("Unexpected or unlocked v3 protocol")
    verify_locked_implementation(protocol)
    if tuple(protocol.get("targets", ())) != contract.TARGETS:
        raise ValueError("v3 target order changed")
    if tuple(protocol.get("join_keys", ())) != contract.PANEL_KEYS:
        raise ValueError("v3 join-key order changed")
    claims = protocol.get("claim_flags")
    gate = protocol.get("semantic_gate")
    if (
        not isinstance(claims, Mapping)
        or claims.get("exploratory_fit_authorized") is not True
        or claims.get("semantic_gate_passed") is not False
        or claims.get("primary_training_eligible") is not False
        or claims.get("confirmatory_eligible") is not False
        or not isinstance(gate, Mapping)
        or gate.get("passed") is not False
        or gate.get("override", {}).get("authorized") is not True
    ):
        raise ValueError("The failed-gate exploratory override is absent")
    observed = float(gate["observed_choice_order_agreement"])
    required = float(gate["required_choice_order_agreement"])
    if not (0 <= observed < required == 0.85):
        raise ValueError("The semantic-gate override values changed")

    blocks = protocol.get("feature_blocks")
    if not isinstance(blocks, Mapping):
        raise ValueError("v3 feature blocks are missing")
    if _string_tuple(blocks.get("wlite_17"), "wlite_17", 17) != (
        contract.WLITE_17
    ):
        raise ValueError("W17-Lite feature order changed")
    if _string_tuple(
        blocks.get("d2_normalized_30"), "d2_normalized_30", 30
    ) != tuple(blocks["d2_normalized_30"]):
        raise AssertionError("D2 feature validation failed")
    q = blocks.get("q56_by_target")
    if not isinstance(q, Mapping) or set(q) != set(contract.TARGETS):
        raise ValueError("Q56 target feature blocks are malformed")
    for target in contract.TARGETS:
        _string_tuple(q[target], f"q56_by_target.{target}", 56)

    folds = protocol.get("folds")
    if not isinstance(folds, list) or len(folds) != 3:
        raise ValueError("v3 requires exactly three folds")
    for expected_name, fold in zip(
        ("fold_1", "fold_2", "fold_3"), folds, strict=True
    ):
        if fold.get("name") != expected_name:
            raise ValueError("v3 fold names/order changed")
    return protocol


def q_features(
    target: str, protocol: Mapping[str, Any]
) -> tuple[str, ...]:
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


def l_features(protocol: Mapping[str, Any]) -> tuple[str, ...]:
    values = _string_tuple(
        protocol["feature_blocks"]["wlite_17"], "wlite_17", 17
    )
    if values != contract.WLITE_17:
        raise ValueError("W17-Lite feature order changed")
    return values


def model_features(
    bundle_key: str,
    target: str,
    protocol: Mapping[str, Any],
) -> tuple[str, ...]:
    bundle = bundle_for(bundle_key)
    q = q_features(target, protocol)
    d = d_features(protocol)
    l = l_features(protocol)
    if bundle.architecture == "q":
        values = q
    elif bundle.architecture == "qd":
        values = (*q, *d)
    elif bundle.architecture == "ql":
        values = (*q, *l)
    elif bundle.architecture == "qdl":
        values = (*q, *d, *l)
    elif bundle.architecture == "ql_cov":
        values = (*q, *contract.COVERAGE_TEXT_6)
    elif bundle.architecture == "qdl_cov":
        values = (*q, *d, *contract.COVERAGE_TEXT_6)
    else:
        raise ValueError(f"Unknown architecture {bundle.architecture}")
    expected = {
        "q": 56,
        "qd": 86,
        "ql": 73,
        "qdl": 103,
        "ql_cov": 62,
        "qdl_cov": 92,
    }[bundle.architecture]
    if len(values) != expected or len(set(values)) != expected:
        raise ValueError(f"{bundle_key} feature contract is inconsistent")
    validate_model_features(values)
    return tuple(values)


def validate_model_features(features: Sequence[str]) -> None:
    if not features or len(features) != len(set(features)):
        raise ValueError("Features must be nonempty and unique")
    forbidden_exact = {
        "forecast_date",
        "sector",
        "stock",
        "benchmark",
        "asof_session",
        "point_in_time_version_safe",
        "primary_training_eligible",
    }
    forbidden = [
        name
        for name in features
        if name in forbidden_exact
        or name.startswith("target_")
        or name.endswith("_end_date")
        or "realized_covariance" in name
        or "realized_variance" in name
    ]
    if forbidden:
        raise ValueError(f"Target/audit leakage columns requested: {forbidden}")


def _verify_artifact(
    record: Mapping[str, Any], *, label: str
) -> Path:
    path = Path(str(record.get("path", "")))
    manifest = Path(str(record.get("manifest_path", "")))
    if (
        not path.is_file()
        or sha256_file(path) != str(record.get("sha256"))
        or not manifest.is_file()
        or sha256_file(manifest)
        != str(record.get("manifest_sha256"))
    ):
        raise ValueError(f"{label} fails its locked hashes")
    return path


def verify_permutation_provenance(
    protocol: Mapping[str, Any],
) -> dict[str, str]:
    """Verify the exact builder seed, mapping identity, and daily manifest."""

    controls = protocol.get("controls")
    control = (
        controls.get("date_sector_permutation")
        if isinstance(controls, Mapping)
        else None
    )
    if not isinstance(control, Mapping):
        raise ValueError("Protocol lacks permutation provenance")

    seed = control.get("seed")
    mapping_hash = control.get("permutation_mapping_sha256")
    config_path = Path(str(control.get("aggregation_config_path", "")))
    config_hash = str(control.get("aggregation_config_sha256", ""))
    daily_manifest_hash = str(
        control.get("permuted_daily_manifest_sha256", "")
    )
    if not isinstance(seed, str) or not seed:
        raise ValueError("Locked permutation seed must be a nonempty string")
    if (
        not isinstance(mapping_hash, str)
        or len(mapping_hash) != 64
        or any(
            character not in "0123456789abcdef"
            for character in mapping_hash
        )
    ):
        raise ValueError("Locked permutation mapping hash is malformed")
    if (
        config_path != contract.AGGREGATION_CONFIG
        or not config_path.is_file()
        or sha256_file(config_path) != config_hash
    ):
        raise ValueError("Permutation aggregation config hash changed")

    aggregation = load_json(config_path)
    variants = aggregation.get("variants")
    permuted_config = (
        variants.get("permuted")
        if isinstance(variants, Mapping)
        else None
    )
    if (
        not isinstance(permuted_config, Mapping)
        or permuted_config.get("seed") != seed
        or permuted_config.get("accepted_labels_only") is not True
        or permuted_config.get("other_features_contemporaneous") is not True
        or permuted_config.get("strata")
        != ["forecast_date", "sector"]
    ):
        raise ValueError("Permutation aggregation contract changed")

    source_artifacts = protocol.get("source_artifacts")
    daily_record = (
        source_artifacts.get("permuted_daily_wlite")
        if isinstance(source_artifacts, Mapping)
        else None
    )
    if not isinstance(daily_record, Mapping):
        raise ValueError("Protocol lacks permuted daily artifact")
    _verify_artifact(daily_record, label="permuted daily W17-Lite")
    manifest_path = Path(str(daily_record.get("manifest_path", "")))
    if (
        manifest_path != contract.PERMUTED_DAILY_MANIFEST
        or sha256_file(manifest_path) != daily_manifest_hash
        or daily_record.get("manifest_sha256") != daily_manifest_hash
    ):
        raise ValueError("Permuted daily manifest hash changed")
    manifest = load_json(manifest_path)
    if (
        manifest.get("variant_id") != "permuted"
        or manifest.get("permutation_mapping_sha256") != mapping_hash
    ):
        raise ValueError("Permutation mapping identity changed")
    config_input = manifest.get("inputs", {}).get("status_cue_rules")
    if (
        not isinstance(config_input, Mapping)
        or config_input.get("path") != config_path.as_posix()
        or config_input.get("sha256") != config_hash
    ):
        raise ValueError(
            "Permuted daily manifest no longer binds the aggregation config"
        )
    return {
        "seed": seed,
        "permutation_mapping_sha256": mapping_hash,
        "aggregation_config_sha256": config_hash,
        "permuted_daily_manifest_sha256": daily_manifest_hash,
    }


def source_panel_record(
    protocol: Mapping[str, Any], bundle_key: str
) -> tuple[str, Mapping[str, Any]]:
    control = bundle_for(bundle_key).control
    if control == "long_description":
        key = "long_description_joined_panel"
    elif control == "date_sector_permutation":
        key = "permuted_joined_panel"
    else:
        key = "canonical_joined_panel"
    record = protocol["source_artifacts"].get(key)
    if not isinstance(record, Mapping):
        raise ValueError(f"Protocol lacks source artifact {key}")
    return key, record


def _normalize_panel(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["forecast_date"] = pd.to_datetime(
        frame["forecast_date"], errors="raise"
    ).dt.normalize()
    if frame.duplicated(list(contract.PANEL_KEYS)).any():
        raise ValueError("v3 panel contains duplicate keys")
    return frame.sort_values(
        list(contract.PANEL_KEYS), kind="mergesort"
    ).reset_index(drop=True)


def prepare_stale_event_control(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    output = frame.copy()
    output = output.sort_values(
        ["stock", "forecast_date"], kind="mergesort"
    )
    donor_dates = output.groupby("stock", sort=False)[
        "forecast_date"
    ].shift(20)
    shifted = output.groupby("stock", sort=False)[
        list(contract.EVENT_CONTENT_4)
    ].shift(20)
    for name in contract.EVENT_CONTENT_4:
        output[name] = shifted[name]
    unavailable = donor_dates.isna()
    return output.sort_values(
        list(contract.PANEL_KEYS), kind="mergesort"
    ).reset_index(drop=True), {
        "control": "20-session stale FLAN event content",
        "sessions": 20,
        "shifted_features": list(contract.EVENT_CONTENT_4),
        "contemporaneous_features": [
            name
            for name in contract.WLITE_17
            if name not in contract.EVENT_CONTENT_4
        ],
        "wrapped": False,
        "rows_without_stale_donor": int(unavailable.sum()),
        "missing_stale_values_are_training_fold_imputed": True,
    }


def wrong_stock_mapping(frame: pd.DataFrame) -> dict[tuple[str, str], str]:
    mapping: dict[tuple[str, str], str] = {}
    for sector, group in frame.groupby("sector", sort=True):
        stocks = sorted(group["stock"].unique())
        if len(stocks) != 6:
            raise ValueError(f"Expected six stocks in {sector}")
        for index, stock in enumerate(stocks):
            mapping[(str(sector), str(stock))] = stocks[
                (index + 1) % len(stocks)
            ]
    return mapping


def prepare_wrong_stock_control(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    output = frame.copy()
    mapping = wrong_stock_mapping(output)
    output["_semantic_donor_stock"] = [
        mapping[(str(sector), str(stock))]
        for sector, stock in zip(
            output["sector"], output["stock"], strict=True
        )
    ]
    donor = output[
        ["forecast_date", "sector", "stock", *contract.SEMANTIC_CONTENT_11]
    ].rename(
        columns={
            "stock": "_semantic_donor_stock",
            **{
                name: f"_donor_{name}"
                for name in contract.SEMANTIC_CONTENT_11
            },
        }
    )
    output = output.merge(
        donor,
        on=["forecast_date", "sector", "_semantic_donor_stock"],
        how="left",
        validate="many_to_one",
    )
    for name in contract.SEMANTIC_CONTENT_11:
        output[name] = output.pop(f"_donor_{name}")
    output = output.drop(columns="_semantic_donor_stock")
    return output.sort_values(
        list(contract.PANEL_KEYS), kind="mergesort"
    ).reset_index(drop=True), {
        "control": "fixed same-date within-sector wrong-stock semantics",
        "rotated_features": list(contract.SEMANTIC_CONTENT_11),
        "contemporaneous_features": list(contract.COVERAGE_TEXT_6),
        "mapping": {
            f"{sector}/{stock}": donor
            for (sector, stock), donor in sorted(mapping.items())
        },
    }


def load_panel_and_preflight(
    protocol: Mapping[str, Any],
    *,
    bundle_key: str,
    allow_failed_gate_exploratory: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not allow_failed_gate_exploratory:
        raise ValueError(
            "Pass --allow-failed-gate-exploratory to acknowledge that "
            "W17-Lite missed its semantic stability gate"
        )
    source_name, source = source_panel_record(protocol, bundle_key)
    path = _verify_artifact(source, label=source_name)
    frame = _normalize_panel(path)
    expected = protocol["coverage"]
    observed = {
        "row_count": len(frame),
        "date_count": int(frame["forecast_date"].nunique()),
        "stock_count": int(frame["stock"].nunique()),
        "sector_count": int(frame["sector"].nunique()),
    }
    expected_shape = {
        name: int(expected[name])
        for name in (
            "row_count",
            "date_count",
            "stock_count",
            "sector_count",
        )
    }
    if observed != expected_shape:
        raise ValueError(f"v3 panel shape differs: {observed}")

    control_audit: dict[str, Any] | None = None
    control = bundle_for(bundle_key).control
    if control == "stale_event_lag20":
        frame, control_audit = prepare_stale_event_control(frame)
    elif control == "wrong_stock":
        frame, control_audit = prepare_wrong_stock_control(frame)
    elif control == "date_sector_permutation":
        provenance = verify_permutation_provenance(protocol)
        control_audit = {
            "control": (
                "precomputed within-date/sector accepted-event-label "
                "permutation"
            ),
            **provenance,
            "source_artifact": source_name,
            "permuted_features": list(contract.EVENT_CONTENT_4),
            "contemporaneous_features": [
                name
                for name in contract.WLITE_17
                if name not in contract.EVENT_CONTENT_4
            ],
        }
    elif control == "long_description":
        control_audit = {
            "control": "description-length-at-least-150 sensitivity",
            "source_artifact": source_name,
            "required_to_beat_for_useful_gate": False,
        }
    elif control == "coverage_text_only":
        control_audit = {
            "control": "coverage/text-quality-only feature subset",
            "features": list(contract.COVERAGE_TEXT_6),
        }

    all_features: set[str] = set()
    for target in contract.TARGETS:
        all_features.update(model_features(bundle_key, target, protocol))
    missing = all_features - set(frame.columns)
    if missing:
        raise ValueError(f"v3 panel misses model features: {sorted(missing)}")
    numeric = frame[sorted(all_features)].apply(
        pd.to_numeric, errors="coerce"
    )
    for name in all_features:
        if numeric[name].notna().sum() != frame[name].notna().sum():
            raise TypeError(f"v3 feature {name} is nonnumeric")
    values = numeric.to_numpy(dtype=float)
    if not (np.isfinite(values) | np.isnan(values)).all():
        raise ValueError("v3 model features contain infinities")
    if frame[list(d_features(protocol))].isna().any().any():
        raise ValueError("D2-Normalized must remain complete-case")
    no_selected = frame["wlite_observed_no_selected_article"]
    if not no_selected.isin([0, 1]).all():
        raise ValueError("W17-Lite no-selected indicator is not binary")

    quant_common.validate_panel_information_set(frame)
    split_counts: dict[str, Any] = {}
    for spec in quant_common.target_specs():
        split_counts[spec.name] = {}
        for fold in protocol["folds"]:
            masks = quant_common.split_masks(frame, spec, fold)
            observed_counts = {
                name: int(mask.sum()) for name, mask in masks.items()
            }
            horizon = spec.horizon
            expected_values = protocol["target_row_eligibility"][
                "expected_rows"
            ][fold["name"]][f"{horizon}_train_validation_test"]
            expected_counts = dict(
                zip(
                    ("train", "validation", "test"),
                    map(int, expected_values),
                    strict=True,
                )
            )
            if observed_counts != expected_counts:
                raise ValueError(
                    f"Split mismatch {spec.name}/{fold['name']}: "
                    f"{observed_counts} != {expected_counts}"
                )
            split_counts[spec.name][fold["name"]] = observed_counts
    return frame, {
        "status": "passed",
        "generated_at_utc": utc_now(),
        "bundle": bundle_key,
        "source_artifact": source_name,
        "panel_path": path.as_posix(),
        "panel_sha256": sha256_file(path),
        "panel_manifest_path": str(source["manifest_path"]),
        "panel_manifest_sha256": str(source["manifest_sha256"]),
        **observed,
        "semantic_gate_passed": False,
        "semantic_gate_override_confirmed": True,
        "choice_order_agreement": float(
            protocol["semantic_gate"][
                "observed_choice_order_agreement"
            ]
        ),
        "control_audit": control_audit,
        "split_counts": split_counts,
    }


def apply_fixed_transforms(
    frame: pd.DataFrame,
    features: Sequence[str],
    protocol: Mapping[str, Any],
) -> pd.DataFrame:
    output = frame.copy()
    configured = set(
        protocol.get("preprocessing", {}).get(
            "fixed_log1p_features", []
        )
    )
    for name in sorted(set(features) & configured):
        values = pd.to_numeric(output[name], errors="coerce")
        if (values.dropna() < 0).any():
            raise ValueError(f"{name} is negative before log1p")
        output[name] = np.log1p(values)
    return output


def validate_training_feature_availability(
    frame: pd.DataFrame, features: Sequence[str], label: str
) -> None:
    all_missing = [
        name for name in features if not frame[name].notna().any()
    ]
    if all_missing:
        raise ValueError(f"{label} has all-missing features: {all_missing}")
    values = frame[list(features)].to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError(f"{label} contains infinities")


def select_linear_hyperparameters(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: Sequence[str],
    response: str,
    *,
    alpha_grid: Sequence[float],
    l1_ratios: Sequence[float],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    # Use the v2 path implementation because quant_common's legacy feature
    # validator rejects legitimate target-relative semantic names containing
    # the substring ``target_``.
    return v2_common.select_linear_hyperparameters(
        train,
        validation,
        features,
        response,
        alpha_grid=alpha_grid,
        l1_ratios=l1_ratios,
    )


def validate_prediction_panel(
    frame: pd.DataFrame,
    *,
    protocol: Mapping[str, Any],
    split_index: int,
    targets: Sequence[str] | None = None,
) -> dict[str, Any]:
    target_list = list(targets or contract.TARGETS)
    folds = [record["name"] for record in protocol["folds"]]
    expected: dict[str, dict[str, int]] = {}
    observed: dict[str, dict[str, int]] = {}
    for target in target_list:
        horizon = target.split("_", 1)[0]
        expected[target] = {
            fold["name"]: int(
                protocol["target_row_eligibility"]["expected_rows"][
                    fold["name"]
                ][f"{horizon}_train_validation_test"][split_index]
            )
            for fold in protocol["folds"]
        }
        observed[target] = {
            fold: int(
                len(
                    frame[
                        frame["target"].eq(target)
                        & frame["fold"].eq(fold)
                    ]
                )
            )
            for fold in folds
        }
    review = {
        "prediction_keys_unique": not bool(
            frame.duplicated(list(contract.PREDICTION_KEYS)).any()
        ),
        "predictions_finite": bool(
            np.isfinite(
                frame[
                    ["predicted_fisher_z", "predicted_correlation"]
                ].to_numpy(dtype=float)
            ).all()
        ),
        "prediction_bounds_valid": bool(
            frame["predicted_correlation"].between(-1, 1).all()
        ),
        "targets_exact": set(frame["target"]) == set(target_list),
        "folds_exact": set(frame["fold"]) == set(folds),
        "observed_rows": observed,
        "expected_rows": expected,
        "row_counts_match": observed == expected,
    }
    if not all(
        review[name]
        for name in (
            "prediction_keys_unique",
            "predictions_finite",
            "prediction_bounds_valid",
            "targets_exact",
            "folds_exact",
            "row_counts_match",
        )
    ):
        raise AssertionError(f"Prediction review failed: {review}")
    return review


def runtime_versions() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in (
        "numpy",
        "pandas",
        "scikit-learn",
        "pyarrow",
        "xgboost",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
    }


def _implementation_hashes() -> dict[str, str]:
    return {
        path.as_posix(): sha256_file(path)
        for path in contract.BOUND_TRAINING_IMPLEMENTATION
    }


def ensure_no_existing_bundle(
    bundle_key: str, *, overwrite: bool
) -> None:
    bundle = bundle_for(bundle_key)
    paths = (bundle.output_path, bundle.experiment_path)
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"{bundle_key} already has artifacts: {existing}"
        )
    if overwrite:
        for path in existing:
            shutil.rmtree(path)


def _artifact_record(path: Path, rows: int | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if rows is not None:
        record["rows"] = rows
    return record


def result_markdown(
    bundle_key: str,
    summary: Mapping[str, Any],
    review: Mapping[str, Any],
) -> str:
    bundle = bundle_for(bundle_key)
    lines = [
        f"# {bundle.label}",
        "",
        f"Status: **{summary.get('status', 'unknown')}**.",
        "",
        (
            "Claim scope: exploratory failed-semantic-gate development "
            "diagnostic; not primary or confirmatory evidence."
        ),
        "",
        "## Metrics",
        "",
        "| Target | Fisher RMSE | Fisher MAE | OOS R2 vs persistence |",
        "|---|---:|---:|---:|",
    ]
    for row in summary.get("metrics", []):
        lines.append(
            f"| {row['target']} | {float(row['fisher_z_rmse']):.6f} | "
            f"{float(row['fisher_z_mae']):.6f} | "
            f"{float(row['oos_r2_vs_persistence']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Integrity",
            "",
            f"- Review: `{review.get('status')}`.",
            "- Choice-order semantic gate passed: `false`.",
            "- Failed-gate exploratory override: `true`.",
            "- Outer evaluation used for tuning: `false`.",
            "",
        ]
    )
    return "\n".join(lines)


def write_bundle(
    bundle_key: str,
    *,
    predictions: pd.DataFrame,
    validation_predictions: pd.DataFrame,
    fits: Sequence[Mapping[str, Any]],
    fold_metrics: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    review: Mapping[str, Any],
    model_config: Mapping[str, Any],
    dependencies: Mapping[str, Any],
    extra_outputs: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    bundle = bundle_for(bundle_key)
    output = bundle.output_path
    tracked = bundle.experiment_path
    output.mkdir(parents=True, exist_ok=False)
    tracked.mkdir(parents=True, exist_ok=False)
    paths: dict[str, Path] = {
        "predictions.parquet": output / "predictions.parquet",
        "validation_predictions.parquet": (
            output / "validation_predictions.parquet"
        ),
        "fits.json": output / "fits.json",
        "fold_metrics.json": output / "fold_metrics.json",
        "summary.json": output / "summary.json",
        "review.json": output / "review.json",
        "model_config.json": output / "model_config.json",
    }
    atomic_parquet(paths["predictions.parquet"], predictions)
    atomic_parquet(
        paths["validation_predictions.parquet"], validation_predictions
    )
    atomic_json(paths["fits.json"], list(fits))
    atomic_json(paths["fold_metrics.json"], list(fold_metrics))
    atomic_json(paths["summary.json"], dict(summary))
    atomic_json(paths["review.json"], dict(review))
    atomic_json(paths["model_config.json"], dict(model_config))
    for name, value in (extra_outputs or {}).items():
        path = output / name
        if isinstance(value, pd.DataFrame):
            atomic_parquet(path, value)
        elif isinstance(value, str):
            atomic_text(path, value)
        else:
            atomic_json(path, value)
        paths[name] = path

    manifest_path = output / "manifest.json"
    manifest = {
        "manifest_version": "quant-deterministic-news-v3-bundle-v1",
        "status": "complete",
        "bundle": bundle_key,
        "label": bundle.label,
        "generated_at_utc": utc_now(),
        "protocol_path": contract.PROTOCOL_PATH.as_posix(),
        "protocol_sha256": sha256_file(contract.PROTOCOL_PATH),
        "semantic_gate_passed": False,
        "failed_gate_override": True,
        "claim_flags": {
            "exploratory": True,
            "primary_training_eligible": False,
            "confirmatory_eligible": False,
        },
        "artifacts": {
            name: _artifact_record(
                path,
                rows=(
                    len(predictions)
                    if name == "predictions.parquet"
                    else len(validation_predictions)
                    if name == "validation_predictions.parquet"
                    else None
                ),
            )
            for name, path in paths.items()
        },
        "dependencies": dict(dependencies),
        "implementation_sha256": _implementation_hashes(),
        "runtime": runtime_versions(),
    }
    atomic_json(manifest_path, manifest)
    manifest_hash = sha256_file(manifest_path)
    atomic_text(
        manifest_path.with_suffix(".sha256"),
        f"{manifest_hash}  {manifest_path.name}\n",
    )
    atomic_json(tracked / "summary.json", dict(summary))
    atomic_json(tracked / "review.json", dict(review))
    atomic_json(
        tracked / "artifact_pointer.json",
        {
            "bundle": bundle_key,
            "manifest_path": manifest_path.as_posix(),
            "manifest_sha256": manifest_hash,
            "protocol_sha256": sha256_file(contract.PROTOCOL_PATH),
        },
    )
    atomic_text(
        tracked / "RESULTS.md",
        result_markdown(bundle_key, summary, review),
    )
    return {
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": manifest_hash,
    }


def verify_completed_bundle(bundle_key: str) -> dict[str, Any]:
    protocol = load_protocol()
    bundle = bundle_for(bundle_key)
    manifest_path = bundle.output_path / "manifest.json"
    sidecar = manifest_path.with_suffix(".sha256")
    if not manifest_path.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"{bundle_key} is incomplete")
    manifest_hash = sha256_file(manifest_path)
    if sidecar.read_text(encoding="ascii").strip().split()[0] != manifest_hash:
        raise RuntimeError(f"{bundle_key} manifest sidecar is stale")
    manifest = load_json(manifest_path)
    if (
        manifest.get("status") != "complete"
        or manifest.get("protocol_sha256")
        != sha256_file(contract.PROTOCOL_PATH)
        or manifest.get("implementation_sha256")
        != protocol.get("implementation")
    ):
        raise RuntimeError(f"{bundle_key} is stale or incomplete")
    for name, record in manifest.get("artifacts", {}).items():
        path = Path(str(record["path"]))
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"{bundle_key}/{name} fails its hash")
    return {
        "bundle": bundle_key,
        "manifest": manifest,
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": manifest_hash,
    }


def load_completed_predictions(
    bundle_key: str,
    *,
    validation: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    verified = verify_completed_bundle(bundle_key)
    name = (
        "validation_predictions.parquet"
        if validation
        else "predictions.parquet"
    )
    record = verified["manifest"]["artifacts"][name]
    path = Path(str(record["path"]))
    frame = pd.read_parquet(path)
    frame["forecast_date"] = pd.to_datetime(
        frame["forecast_date"]
    ).dt.normalize()
    provenance = {
        "bundle": bundle_key,
        "manifest_path": verified["manifest_path"],
        "manifest_sha256": verified["manifest_sha256"],
        "artifact_path": path.as_posix(),
        "artifact_sha256": record["sha256"],
    }
    return frame, provenance


def initialize_status() -> None:
    if contract.STATUS_PATH.exists():
        status = load_json(contract.STATUS_PATH)
        if status.get("protocol_sha256") != sha256_file(
            contract.PROTOCOL_PATH
        ):
            raise RuntimeError("Existing v3 status uses a stale protocol")
        return
    records = [
        {
            "order": bundle.order,
            "key": bundle.key,
            "label": bundle.label,
            "status": "pending",
            "updated_at_utc": None,
            "summary": None,
        }
        for bundle in contract.BUNDLES
    ]
    atomic_json(
        contract.STATUS_PATH,
        {
            "experiment_id": "quant-deterministic-news-v3-w17-lite",
            "protocol_sha256": sha256_file(contract.PROTOCOL_PATH),
            "semantic_gate_passed": False,
            "failed_gate_override": True,
            "bundles": records,
        },
    )
    write_status_markdown()


def update_status(
    bundle_key: str, state: str, summary: str | None = None
) -> None:
    initialize_status()
    status = load_json(contract.STATUS_PATH)
    matched = False
    for record in status["bundles"]:
        if record["key"] == bundle_key:
            record["status"] = state
            record["updated_at_utc"] = utc_now()
            record["summary"] = summary
            matched = True
            break
    if not matched:
        raise KeyError(bundle_key)
    atomic_json(contract.STATUS_PATH, status)
    write_status_markdown()


def write_status_markdown() -> None:
    status = load_json(contract.STATUS_PATH)
    lines = [
        "# V3 W17-Lite training status",
        "",
        (
            "All results are exploratory failed-semantic-gate diagnostics "
            "(choice-order agreement below 85%)."
        ),
        "",
        "| Bundle | Status | Summary |",
        "|---|---|---|",
    ]
    for record in sorted(status["bundles"], key=lambda item: item["order"]):
        summary = str(record.get("summary") or "").replace("|", "\\|")
        lines.append(
            f"| `{record['key']}` | {record['status']} | {summary} |"
        )
    lines.append("")
    atomic_text(contract.STATUS_MARKDOWN_PATH, "\n".join(lines))


def summary_status_text(summary: Mapping[str, Any]) -> str:
    metrics = summary.get("metrics", [])
    if not metrics:
        return str(summary.get("reason") or summary.get("status"))
    return "; ".join(
        f"{row['target']} RMSE={float(row['fisher_z_rmse']):.6f}"
        for row in metrics
    )
