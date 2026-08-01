"""Join the complete W17-Lite daily panel to the frozen Q+D2 panel.

The canonical output is an exploratory v3 modeling panel.  This program does
not construct semantic labels and never reads the model prediction ledger
directly.  Instead, it requires a complete hash-bound daily-feature manifest,
proves exact stock-day key equality with the frozen Q+D2 panel, preserves all
base values and dtypes, and appends the exact ordered 17-column W17-Lite
contract.

Optional sensitivity panels are joined separately.  They never overwrite or
silently replace the canonical W17-Lite columns.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
JOIN_KEYS = ["forecast_date", "sector", "stock", "benchmark"]
DEFAULT_BASE = Path(
    "data/features/q_plus_d/massive_v2/modeling_panel_q_d2.parquet"
)
DEFAULT_BASE_MANIFEST = Path("data/features/q_plus_d/massive_v2/manifest.json")
DEFAULT_DESIGN = Path("config/news_semantic_lite_design_v1.json")
DEFAULT_DAILY_ROOT = Path(
    "data/features/news_semantic/massive_v3/flan_w17_lite_k16"
)
DEFAULT_DAILY = DEFAULT_DAILY_ROOT / "daily_features.parquet"
DEFAULT_DAILY_MANIFEST = DEFAULT_DAILY_ROOT / "daily_features.manifest.json"
DEFAULT_OUTPUT_ROOT = Path("data/features/q_plus_d/massive_v3")
DEFAULT_OUTPUT_NAME = "modeling_panel_q_d2_wlite.parquet"
MANIFEST_NAME = "manifest.json"
MANIFEST_HASH_NAME = "manifest.sha256"
DEFAULT_VARIANTS = {
    "long_description": (
        DEFAULT_DAILY_ROOT / "daily_features_long_description.parquet",
        DEFAULT_DAILY_ROOT / "daily_features_long_description.manifest.json",
    ),
    "permuted": (
        DEFAULT_DAILY_ROOT / "daily_features_permuted.parquet",
        DEFAULT_DAILY_ROOT / "daily_features_permuted.manifest.json",
    ),
}

BUILDER_VERSION = "q-plus-d2-w17-lite-massive-join-v3.0.0"
EXPECTED_ROWS = 27_510
EXPECTED_DATES = 917
EXPECTED_STOCKS = 30
EXPECTED_SECTORS = 5
EXPECTED_FIRST_DATE = "2022-11-01"
EXPECTED_LAST_DATE = "2026-06-30"
EXPECTED_CANONICAL_NO_SELECTED = 204
EXPECTED_ARM = "WL17__flan_t5_xl"
EXPECTED_CONTRACT = "weak-news-semantics-lite-v1"
EXPECTED_SOURCE_PROFILE = "ordinary_massive_retrospective"
EXPECTED_DAILY_MANIFEST_VERSION = "flan-w17-lite-daily-manifest-v1"
EXPECTED_DAILY_STATUS = "complete_exploratory_non_version_safe"
EXPECTED_BASE_MANIFEST_VERSION = "q-plus-d2-massive-manifest-v2"
EXPECTED_BASE_STATUS = "complete_exploratory_non_version_safe"
REQUIRED_DAILY_INPUTS = {
    "predictions",
    "inference_manifest",
    "selection_manifest",
    "selected_assignments",
    "source_assignments",
    "stock_day_scope",
    "semantic_corpus_manifest",
    "design",
    "event_schema",
    "status_cue_rules",
    "builder",
}

AUDIT_ONLY_COLUMNS = (
    "wlite_route_share_common",
    "wlite_event_accepted_coverage",
    "wlite_event_other_or_unclear_weight_share",
    "wlite_rule_status_cue_all_zero_weight_share",
    "wlite_invalid_or_truncated_selected_article_count",
)
EVENT_FEATURES = (
    "wlite_event_share_firm_operating_financial",
    "wlite_event_share_policy_corporate",
    "wlite_event_share_macro_market",
)
ROUTE_FEATURES = (
    "wlite_route_share_target_idiosyncratic",
    "wlite_route_share_peer_idiosyncratic",
)
STATUS_FEATURES = (
    "wlite_rule_status_cue_share_confirmed_action",
    "wlite_rule_status_cue_share_scheduled_expected",
    "wlite_rule_status_cue_share_rumor_unconfirmed",
    "wlite_rule_status_cue_share_analysis_opinion",
    "wlite_rule_status_cue_conflict_weight_share",
)
BOUNDED_FEATURES = (
    *EVENT_FEATURES,
    *ROUTE_FEATURES,
    *STATUS_FEATURES,
    "wlite_selection_weight_coverage",
    "wlite_selected_headline_only_weight_share",
    "wlite_selected_sub150_weight_share",
    "wlite_event_order_disagreement_weight_share",
    "wlite_observed_no_selected_article",
    "wlite_event_entropy_accepted",
    "wlite_selected_weight_hhi",
)
VARIANT_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _resolve_manifest_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} path is missing")
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _validate_bound_file(
    path: Path,
    expected_sha256: Any,
    label: str,
    *,
    expected_path: Path | None = None,
) -> str:
    if (
        not isinstance(expected_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
    ):
        raise ValueError(f"{label} has no canonical SHA-256")
    resolved = path.resolve()
    if expected_path is not None and resolved != expected_path.resolve():
        raise ValueError(f"{label} path differs from the requested artifact")
    if not resolved.is_file():
        raise ValueError(f"{label} is missing: {resolved}")
    actual = sha256_file(resolved)
    if actual != expected_sha256:
        raise ValueError(
            f"{label} hash mismatch: expected {expected_sha256}, "
            f"observed {actual}"
        )
    return actual


def _validate_manifest_sidecar(
    manifest_path: Path, *, required: bool = True
) -> None:
    sidecar = manifest_path.with_suffix(".sha256")
    if not sidecar.is_file():
        if required:
            raise ValueError(f"Manifest hash sidecar is missing: {sidecar}")
        return
    tokens = sidecar.read_text(encoding="ascii").strip().split()
    if not tokens or tokens[0] != sha256_file(manifest_path):
        raise ValueError(f"Manifest hash sidecar differs: {sidecar}")


def _ordered_unique_strings(value: Any, label: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
        or len(value) != len(set(value))
    ):
        raise ValueError(f"{label} must be a nonempty ordered unique list")
    return list(value)


def load_wlite_contract(design_path: Path) -> tuple[dict[str, Any], list[str]]:
    design = load_json_object(design_path, "semantic Lite design")
    arm = design.get("flan_w17_lite")
    if not isinstance(arm, dict):
        raise ValueError("Semantic Lite design has no flan_w17_lite block")
    if arm.get("arm_id") != EXPECTED_ARM:
        raise ValueError("Semantic Lite design arm_id changed")
    if arm.get("contract_id") != EXPECTED_CONTRACT:
        raise ValueError("Semantic Lite design contract_id changed")
    if arm.get("target_feature_count") != 17:
        raise ValueError("Semantic Lite design must declare 17 features")
    features = _ordered_unique_strings(
        arm.get("ordered_features"), "flan_w17_lite.ordered_features"
    )
    if len(features) != 17:
        raise ValueError("W17-Lite feature contract must contain 17 names")
    if tuple(features) != tuple(BOUNDED_FEATURES):
        raise ValueError("W17-Lite ordered feature contract changed")
    audits = arm.get("audit_only_nonpredictors")
    if not isinstance(audits, list) or not set(AUDIT_ONLY_COLUMNS).issubset(
        set(audits)
    ):
        raise ValueError("W17-Lite audit-only contract is incomplete")
    return design, features


def validate_base_contract(
    *,
    base_path: Path,
    base_manifest_path: Path,
    expected_rows: int,
) -> dict[str, Any]:
    manifest = load_json_object(base_manifest_path, "Q+D2 manifest")
    _validate_manifest_sidecar(base_manifest_path)
    if manifest.get("manifest_version") != EXPECTED_BASE_MANIFEST_VERSION:
        raise ValueError("Q+D2 manifest version changed")
    if manifest.get("status") != EXPECTED_BASE_STATUS:
        raise ValueError("Q+D2 panel is not complete under its v2 contract")
    output = manifest.get("output")
    if not isinstance(output, dict):
        raise ValueError("Q+D2 manifest has no output record")
    recorded_path = _resolve_manifest_path(output.get("path"), "Q+D2 output")
    _validate_bound_file(
        base_path,
        output.get("sha256"),
        "Q+D2 panel",
        expected_path=recorded_path,
    )
    if int(output.get("row_count", -1)) != expected_rows:
        raise ValueError("Q+D2 manifest row count differs from expectation")
    counts = manifest.get("counts")
    if (
        not isinstance(counts, dict)
        or int(counts.get("matched_rows", -1)) != expected_rows
        or float(counts.get("match_fraction", -1.0)) != 1.0
    ):
        raise ValueError("Q+D2 join was not one-to-one complete")
    claims = manifest.get("claim_scope")
    if (
        not isinstance(claims, dict)
        or claims.get("point_in_time_version_safe") is not False
        or claims.get("primary_training_eligible") is not False
        or claims.get("confirmatory_eligible") is not False
        or claims.get("exploratory_only") is not True
    ):
        raise ValueError("Q+D2 claim boundary changed")
    ordered = manifest.get("ordered_feature_lists")
    if not isinstance(ordered, dict):
        raise ValueError("Q+D2 manifest has no ordered feature lists")
    d2 = _ordered_unique_strings(
        ordered.get("d2_normalized_30"), "d2_normalized_30"
    )
    q56 = ordered.get("q56_by_target")
    if len(d2) != 30 or not isinstance(q56, dict):
        raise ValueError("Q+D2 feature contract is incomplete")
    for target in ("t1_etf", "t1_loo", "t2_etf", "t2_loo"):
        features = _ordered_unique_strings(q56.get(target), f"q56.{target}")
        if len(features) != 56:
            raise ValueError(f"Q56 contract for {target} is not 56 columns")
    return manifest


def _iter_bound_input_records(
    value: Any, prefix: str = "inputs"
) -> list[tuple[str, Mapping[str, Any]]]:
    records: list[tuple[str, Mapping[str, Any]]] = []
    if isinstance(value, Mapping):
        if "path" in value and "sha256" in value:
            records.append((prefix, value))
        for key, child in value.items():
            records.extend(
                _iter_bound_input_records(child, f"{prefix}.{key}")
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            records.extend(
                _iter_bound_input_records(child, f"{prefix}[{index}]")
            )
    return records


def validate_daily_contract(
    *,
    daily_path: Path,
    manifest_path: Path,
    design_path: Path,
    features: Sequence[str],
    expected_rows: int,
    variant_id: str,
) -> dict[str, Any]:
    manifest = load_json_object(manifest_path, f"{variant_id} daily manifest")
    _validate_manifest_sidecar(manifest_path)
    if manifest.get("manifest_version") != EXPECTED_DAILY_MANIFEST_VERSION:
        raise ValueError(f"{variant_id} daily manifest version changed")
    if manifest.get("status") != EXPECTED_DAILY_STATUS:
        raise ValueError(f"{variant_id} daily panel is not complete")
    if manifest.get("arm_id") != EXPECTED_ARM:
        raise ValueError(f"{variant_id} daily arm_id changed")
    if manifest.get("contract_id") != EXPECTED_CONTRACT:
        raise ValueError(f"{variant_id} daily contract_id changed")
    if manifest.get("source_profile") != EXPECTED_SOURCE_PROFILE:
        raise ValueError(f"{variant_id} daily source profile changed")
    declared_variant = manifest.get("variant_id", "canonical")
    if declared_variant != variant_id:
        raise ValueError(
            f"Daily manifest variant {declared_variant!r} differs from "
            f"requested {variant_id!r}"
        )

    ordered = manifest.get("ordered_feature_lists")
    if not isinstance(ordered, dict):
        raise ValueError(f"{variant_id} daily manifest has no feature lists")
    declared = _ordered_unique_strings(
        ordered.get("wlite_17"), f"{variant_id}.wlite_17"
    )
    if list(features) != declared:
        raise ValueError(f"{variant_id} W17-Lite feature order changed")

    generated = manifest.get("generated_files")
    if not isinstance(generated, dict):
        raise ValueError(f"{variant_id} daily manifest has no generated_files")
    record = generated.get(daily_path.name)
    if not isinstance(record, dict):
        raise ValueError(
            f"{variant_id} manifest does not bind {daily_path.name}"
        )
    recorded_path = _resolve_manifest_path(
        record.get("path"), f"{variant_id} daily output"
    )
    _validate_bound_file(
        daily_path,
        record.get("sha256"),
        f"{variant_id} daily panel",
        expected_path=recorded_path,
    )
    if int(record.get("rows", -1)) != expected_rows:
        raise ValueError(f"{variant_id} daily row count differs")

    claims = manifest.get("claim_flags")
    if (
        not isinstance(claims, dict)
        or claims.get("point_in_time_version_safe") is not False
        or claims.get("primary_training_eligible") is not False
        or claims.get("confirmatory_eligible") is not False
        or claims.get("exploratory_construction_eligible") is not True
        or claims.get("semantic_quality_gate_passed") is not False
        or claims.get("exploratory_quality_gate_override") is not True
    ):
        raise ValueError(f"{variant_id} daily claim boundary changed")

    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError(f"{variant_id} daily manifest has no bound inputs")
    missing_inputs = REQUIRED_DAILY_INPUTS - set(inputs)
    if missing_inputs:
        raise ValueError(
            f"{variant_id} daily manifest misses required inputs: "
            f"{sorted(missing_inputs)}"
        )
    records = _iter_bound_input_records(inputs)
    if not records:
        raise ValueError(f"{variant_id} daily manifest binds no input files")
    bound_paths: list[Path] = []
    for label, input_record in records:
        path = _resolve_manifest_path(
            input_record.get("path"), f"{variant_id}.{label}"
        )
        _validate_bound_file(
            path,
            input_record.get("sha256"),
            f"{variant_id}.{label}",
        )
        bound_paths.append(path)
    if design_path.resolve() not in set(bound_paths):
        raise ValueError(f"{variant_id} daily manifest does not bind design")
    return manifest


def normalize_keys(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    missing = set(JOIN_KEYS) - set(frame.columns)
    if missing:
        raise ValueError(f"{label} misses join keys: {sorted(missing)}")
    result = frame.copy()
    dates = pd.to_datetime(result["forecast_date"], errors="raise")
    if dates.isna().any() or not dates.equals(dates.dt.normalize()):
        raise ValueError(f"{label} forecast_date must be canonical date-only")
    result["forecast_date"] = dates.dt.normalize()
    for name in ("sector", "stock", "benchmark"):
        values = result[name].astype("string")
        if values.isna().any() or values.str.len().eq(0).any():
            raise ValueError(f"{label} has an empty {name}")
        if not values.equals(values.str.strip()):
            raise ValueError(f"{label} has noncanonical {name}")
        if name in {"stock", "benchmark"} and not values.equals(
            values.str.upper()
        ):
            raise ValueError(f"{label} has lowercase {name}")
    if result.duplicated(JOIN_KEYS).any():
        raise ValueError(f"{label} has duplicate stock-day keys")
    return result


def _numeric_finite_or_missing(
    frame: pd.DataFrame, columns: Sequence[str], label: str
) -> pd.DataFrame:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{label} misses columns: {sorted(missing)}")
    numeric = frame[list(columns)].apply(pd.to_numeric, errors="coerce")
    for name in columns:
        if numeric[name].notna().sum() != frame[name].notna().sum():
            raise ValueError(f"{label} column {name} is not numeric")
        finite = numeric[name].dropna().map(math.isfinite)
        if not bool(finite.all()):
            raise ValueError(f"{label} column {name} contains infinity")
    return numeric


def validate_daily_frame(
    frame: pd.DataFrame,
    *,
    features: Sequence[str],
    expected_rows: int,
    variant_id: str,
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    daily = normalize_keys(frame, f"{variant_id} daily panel")
    if len(daily) != expected_rows:
        raise ValueError(
            f"{variant_id} expected {expected_rows} daily rows, "
            f"observed {len(daily)}"
        )
    observed = {
        "dates": int(daily["forecast_date"].nunique()),
        "stocks": int(daily["stock"].nunique()),
        "sectors": int(daily["sector"].nunique()),
        "first_date": daily["forecast_date"].min().date().isoformat(),
        "last_date": daily["forecast_date"].max().date().isoformat(),
    }
    expected = {
        "dates": EXPECTED_DATES,
        "stocks": EXPECTED_STOCKS,
        "sectors": EXPECTED_SECTORS,
        "first_date": EXPECTED_FIRST_DATE,
        "last_date": EXPECTED_LAST_DATE,
    }
    if observed != expected:
        raise ValueError(
            f"{variant_id} daily universe differs: "
            f"expected={expected}, observed={observed}"
        )

    numeric = _numeric_finite_or_missing(
        daily, [*features, *AUDIT_ONLY_COLUMNS], f"{variant_id} daily panel"
    )
    no_selected = numeric["wlite_observed_no_selected_article"]
    if no_selected.isna().any() or not no_selected.isin([0.0, 1.0]).all():
        raise ValueError(f"{variant_id} no-selected indicator is not binary")
    selected = no_selected.eq(0)
    other_features = [
        name
        for name in features
        if name != "wlite_observed_no_selected_article"
    ]
    if numeric.loc[selected, other_features].isna().any().any():
        raise ValueError(f"{variant_id} selected rows have missing features")
    if numeric.loc[~selected, other_features].notna().any().any():
        raise ValueError(
            f"{variant_id} no-selected rows must have missing shares"
        )
    if variant_id == "canonical" and int((~selected).sum()) != (
        EXPECTED_CANONICAL_NO_SELECTED
    ):
        raise ValueError("Canonical W17-Lite no-selected count changed")

    for name in BOUNDED_FEATURES:
        values = numeric[name].dropna()
        if not values.between(0.0, 1.0, inclusive="both").all():
            raise ValueError(f"{variant_id} feature {name} is outside [0,1]")
    for name in AUDIT_ONLY_COLUMNS[:-1]:
        values = numeric[name].dropna()
        if not values.between(0.0, 1.0, inclusive="both").all():
            raise ValueError(f"{variant_id} audit {name} is outside [0,1]")
    invalid = numeric[
        "wlite_invalid_or_truncated_selected_article_count"
    ]
    if invalid.isna().any() or not invalid.eq(0.0).all():
        raise ValueError(f"{variant_id} contains invalid/truncated inference")

    tolerance = 1e-9
    selected_rows = numeric.loc[selected]
    route_total = selected_rows[list(ROUTE_FEATURES)].sum(axis=1) + (
        selected_rows["wlite_route_share_common"]
    )
    if not route_total.sub(1.0).abs().le(tolerance).all():
        raise ValueError(f"{variant_id} route decomposition does not sum to 1")
    event_total = selected_rows[list(EVENT_FEATURES)].sum(axis=1)
    if not event_total.sub(
        selected_rows["wlite_event_accepted_coverage"]
    ).abs().le(tolerance).all():
        raise ValueError(f"{variant_id} accepted-event identity failed")
    terminal_total = (
        selected_rows["wlite_event_accepted_coverage"]
        + selected_rows["wlite_event_other_or_unclear_weight_share"]
        + selected_rows["wlite_event_order_disagreement_weight_share"]
    )
    if not terminal_total.sub(1.0).abs().le(tolerance).all():
        raise ValueError(f"{variant_id} event decomposition does not sum to 1")
    selection_coverage = selected_rows["wlite_selection_weight_coverage"]
    if not (
        selection_coverage.gt(0.0).all()
        and selection_coverage.le(1.0).all()
    ):
        raise ValueError(f"{variant_id} selection coverage is invalid")
    hhi = selected_rows["wlite_selected_weight_hhi"]
    if not (hhi.gt(0.0).all() and hhi.le(1.0).all()):
        raise ValueError(f"{variant_id} selected-weight HHI is invalid")
    text_share = (
        selected_rows["wlite_selected_headline_only_weight_share"]
        + selected_rows["wlite_selected_sub150_weight_share"]
    )
    if not text_share.le(1.0 + tolerance).all():
        raise ValueError(f"{variant_id} text-quality shares exceed one")

    expected_flags: dict[str, Any] = {
        "source_profile": EXPECTED_SOURCE_PROFILE,
        "point_in_time_version_safe": False,
        "primary_training_eligible": False,
        "confirmatory_eligible": False,
        "exploratory_construction_eligible": True,
        "semantic_quality_gate_passed": False,
        "exploratory_quality_gate_override": True,
    }
    for name, expected_value in expected_flags.items():
        if name not in daily:
            raise ValueError(f"{variant_id} daily panel misses audit flag {name}")
        values = daily[name].drop_duplicates()
        if len(values) != 1 or values.iloc[0] != expected_value:
            raise ValueError(f"{variant_id} audit flag {name} changed")

    audits = [
        name
        for name in daily.columns
        if name not in set(JOIN_KEYS).union(features)
    ]
    stats = {
        **observed,
        "rows": int(len(daily)),
        "matched_feature_rows": int(len(daily)),
        "no_selected_rows": int((~selected).sum()),
        "selected_rows": int(selected.sum()),
        "missing_feature_values": int(numeric[list(features)].isna().sum().sum()),
        "minimum_selection_coverage": float(selection_coverage.min()),
        "mean_selection_coverage": float(selection_coverage.mean()),
        "semantic_quality_gate_passed": False,
        "exploratory_quality_gate_override": True,
    }
    return daily, stats, audits


def build_joined_panel(
    base_frame: pd.DataFrame,
    daily_frame: pd.DataFrame,
    *,
    features: Sequence[str],
    expected_rows: int,
    variant_id: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    base = normalize_keys(base_frame, "Q+D2 panel")
    daily, daily_stats, _ = validate_daily_frame(
        daily_frame,
        features=features,
        expected_rows=expected_rows,
        variant_id=variant_id,
    )
    if len(base) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} Q+D2 rows, observed {len(base)}"
        )
    base_keys = pd.MultiIndex.from_frame(base[JOIN_KEYS])
    daily_keys = pd.MultiIndex.from_frame(daily[JOIN_KEYS])
    missing = base_keys.difference(daily_keys)
    extra = daily_keys.difference(base_keys)
    if len(missing) or len(extra):
        raise ValueError(
            f"{variant_id} and Q+D2 keys differ: "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    collisions = set(features).intersection(base.columns)
    if collisions:
        raise ValueError(
            f"{variant_id} feature names collide with Q+D2: "
            f"{sorted(collisions)}"
        )

    base_columns = list(base.columns)
    base["__base_row_order"] = range(len(base))
    payload = daily[[*JOIN_KEYS, *features]].copy()
    payload["wlite_row_matched"] = 1
    joined = base.merge(
        payload,
        on=JOIN_KEYS,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    joined = (
        joined.sort_values("__base_row_order", kind="stable")
        .drop(columns="__base_row_order")
        .reset_index(drop=True)
    )
    expected_base = base.drop(columns="__base_row_order").reset_index(drop=True)
    for name in base_columns:
        joined[name] = expected_base[name].array
    pd.testing.assert_frame_equal(
        joined[base_columns],
        expected_base[base_columns],
        check_dtype=True,
        check_like=False,
    )
    if not joined["wlite_row_matched"].eq(1).all():
        raise AssertionError("Exact key preflight passed but join lost rows")
    joined["wlite_row_matched"] = joined["wlite_row_matched"].astype("int8")
    _numeric_finite_or_missing(joined, features, f"{variant_id} joined panel")
    return joined, {
        **daily_stats,
        "base_rows": int(len(base)),
        "output_rows": int(len(joined)),
        "match_fraction": float(joined["wlite_row_matched"].mean()),
    }


def schema_records(frame: pd.DataFrame) -> list[dict[str, str]]:
    return [
        {"name": name, "dtype": str(frame[name].dtype)}
        for name in frame.columns
    ]


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )


def atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_hash_sidecar(path: Path, bound_path: Path) -> str:
    digest = sha256_file(bound_path)
    atomic_write_text(path, f"{digest}  {bound_path.name}\n")
    return digest


def _parse_named_paths(values: Sequence[str], label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not VARIANT_RE.fullmatch(name) or not path:
            raise ValueError(
                f"{label} must use canonical NAME=PATH syntax: {value!r}"
            )
        if name == "canonical" or name in result:
            raise ValueError(f"Duplicate or reserved {label} name: {name}")
        result[name] = Path(path)
    return result


def discover_variants(
    *,
    variant_values: Sequence[str],
    manifest_values: Sequence[str],
    include_defaults: bool,
) -> dict[str, tuple[Path, Path]]:
    paths = _parse_named_paths(variant_values, "variant")
    manifests = _parse_named_paths(manifest_values, "variant-manifest")
    if set(paths) != set(manifests):
        raise ValueError(
            "--variant and --variant-manifest names must match exactly"
        )
    result = {
        name: (paths[name], manifests[name])
        for name in sorted(paths)
    }
    if include_defaults:
        for name, pair in DEFAULT_VARIANTS.items():
            exists = (pair[0].exists(), pair[1].exists())
            if any(exists) and not all(exists):
                raise ValueError(
                    f"Default variant {name} is only partly materialized"
                )
            if all(exists) and name not in result:
                result[name] = pair
    return dict(sorted(result.items()))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--base", type=Path, default=DEFAULT_BASE)
    result.add_argument(
        "--base-manifest", type=Path, default=DEFAULT_BASE_MANIFEST
    )
    result.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    result.add_argument("--daily", type=Path, default=DEFAULT_DAILY)
    result.add_argument(
        "--daily-manifest", type=Path, default=DEFAULT_DAILY_MANIFEST
    )
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    result.add_argument(
        "--variant",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Additional daily sensitivity panel; repeat as needed.",
    )
    result.add_argument(
        "--variant-manifest",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Manifest paired with --variant NAME; repeat as needed.",
    )
    result.add_argument(
        "--no-default-variants",
        action="store_true",
        help="Do not auto-discover the two predeclared v3 sensitivities.",
    )
    result.add_argument(
        "--expected-row-count", type=int, default=EXPECTED_ROWS
    )
    result.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate and join in memory without writing any artifact.",
    )
    result.add_argument("--overwrite", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.expected_row_count <= 0:
        raise ValueError("expected-row-count must be positive")
    variants = discover_variants(
        variant_values=args.variant,
        manifest_values=args.variant_manifest,
        include_defaults=not args.no_default_variants,
    )
    output_paths = {
        "canonical": args.output_root / DEFAULT_OUTPUT_NAME,
        **{
            name: args.output_root
            / f"modeling_panel_q_d2_wlite_{name}.parquet"
            for name in variants
        },
    }
    manifest_path = args.output_root / MANIFEST_NAME
    manifest_hash_path = args.output_root / MANIFEST_HASH_NAME
    outputs = [*output_paths.values(), manifest_path, manifest_hash_path]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite and not args.preflight_only:
        raise FileExistsError(
            "Refusing to overwrite v3 join outputs without --overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    design, features = load_wlite_contract(args.design)
    base_manifest = validate_base_contract(
        base_path=args.base,
        base_manifest_path=args.base_manifest,
        expected_rows=args.expected_row_count,
    )
    sources = {
        "canonical": (args.daily, args.daily_manifest),
        **variants,
    }
    daily_manifests: dict[str, dict[str, Any]] = {}
    daily_frames: dict[str, pd.DataFrame] = {}
    for name, (daily_path, daily_manifest_path) in sources.items():
        daily_manifests[name] = validate_daily_contract(
            daily_path=daily_path,
            manifest_path=daily_manifest_path,
            design_path=args.design,
            features=features,
            expected_rows=args.expected_row_count,
            variant_id=name,
        )
        daily_frames[name] = pd.read_parquet(daily_path)

    base = pd.read_parquet(args.base)
    joined_frames: dict[str, pd.DataFrame] = {}
    counts: dict[str, dict[str, Any]] = {}
    for name, frame in daily_frames.items():
        joined, stats = build_joined_panel(
            base,
            frame,
            features=features,
            expected_rows=args.expected_row_count,
            variant_id=name,
        )
        joined_frames[name] = joined
        counts[name] = stats

    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "preflight_passed",
                    "would_write": {
                        name: str(path)
                        for name, path in output_paths.items()
                    },
                    "counts": counts,
                    "claim_scope": {
                        "exploratory_only": True,
                        "semantic_quality_gate_passed": False,
                    },
                },
                indent=2,
            )
        )
        return 0

    for name, frame in joined_frames.items():
        atomic_write_parquet(output_paths[name], frame)

    generated_files = {
        output_paths[name].name: {
            "path": str(output_paths[name]),
            "rows": int(len(frame)),
            "columns": int(len(frame.columns)),
            "sha256": sha256_file(output_paths[name]),
            "bytes": output_paths[name].stat().st_size,
            "variant_id": name,
            "schema_sha256": canonical_json_sha256(schema_records(frame)),
        }
        for name, frame in joined_frames.items()
    }
    base_columns = list(base.columns)
    manifest: dict[str, Any] = {
        "manifest_version": "q-plus-d2-w17-lite-massive-manifest-v3",
        "builder_version": BUILDER_VERSION,
        "builder_script_sha256": sha256_file(Path(__file__)),
        "status": EXPECTED_DAILY_STATUS,
        "generated_at_utc": utc_now(),
        "arm_id": EXPECTED_ARM,
        "contract_id": EXPECTED_CONTRACT,
        "source_profile": EXPECTED_SOURCE_PROFILE,
        "warning": (
            "EXPLORATORY DEVELOPMENT ONLY: retrospective ordinary Massive "
            "news is not historical-version safe, and W17-Lite failed its "
            "predeclared semantic quality gate. The override permits only "
            "the explicitly requested exploratory comparison."
        ),
        "claim_scope": {
            "development_only": True,
            "exploratory_only": True,
            "point_in_time_version_safe": False,
            "primary_training_eligible": False,
            "confirmatory_eligible": False,
            "semantic_quality_gate_passed": False,
            "exploratory_quality_gate_override": True,
        },
        "join_contract": {
            "keys": JOIN_KEYS,
            "key_set_equality_required": True,
            "base_row_order_preserved": True,
            "base_values_and_dtypes_preserved": True,
            "daily_variants_joined_as_separate_panels": True,
            "expected_rows": args.expected_row_count,
        },
        "ordered_feature_lists": {
            "wlite_17": features,
            "d2_normalized_30": base_manifest["ordered_feature_lists"][
                "d2_normalized_30"
            ],
            "q56_by_target": base_manifest["ordered_feature_lists"][
                "q56_by_target"
            ],
        },
        "feature_list_sha256": {
            "wlite_17": canonical_json_sha256(features),
            "d2_normalized_30": canonical_json_sha256(
                base_manifest["ordered_feature_lists"]["d2_normalized_30"]
            ),
            "q56_by_target": {
                target: canonical_json_sha256(values)
                for target, values in base_manifest["ordered_feature_lists"][
                    "q56_by_target"
                ].items()
            },
        },
        "column_roles": {
            "join_keys": JOIN_KEYS,
            "base_columns": base_columns,
            "wlite_features": features,
            "join_audit_columns": ["wlite_row_matched"],
            "daily_audit_columns_excluded_from_predictive_join": list(
                AUDIT_ONLY_COLUMNS
            ),
        },
        "counts": counts,
        "inputs": {
            "design": {
                "path": str(args.design),
                "sha256": sha256_file(args.design),
            },
            "q_plus_d2": {
                "path": str(args.base),
                "sha256": sha256_file(args.base),
                "manifest_path": str(args.base_manifest),
                "manifest_sha256": sha256_file(args.base_manifest),
            },
            "daily_panels": {
                name: {
                    "path": str(sources[name][0]),
                    "sha256": sha256_file(sources[name][0]),
                    "manifest_path": str(sources[name][1]),
                    "manifest_sha256": sha256_file(sources[name][1]),
                }
                for name in sources
            },
        },
        "generated_files": generated_files,
        "variants": sorted(variants),
        "design_contract_sha256": canonical_json_sha256(design),
    }
    atomic_write_json(manifest_path, manifest)
    manifest_sha = write_hash_sidecar(manifest_hash_path, manifest_path)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "generated_files": generated_files,
                "manifest": str(manifest_path),
                "manifest_sha256": manifest_sha,
                "counts": counts,
                "warning": manifest["warning"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
