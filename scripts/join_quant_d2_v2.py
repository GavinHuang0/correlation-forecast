"""Build and lock the exploratory quant + D2 v2 modeling panel.

This utility is intentionally separate from the completed Q+D v1 join.  It
requires exact stock-date key equality, preserves the quant panel byte-level
values after the merge, rejects missing/non-finite D2 values, and binds the
joined artifact to a machine-readable protocol before any v2 model is fit.

The default news input is expected to be produced by the v2 deterministic
builder with this manifest contract:

* ``status == "complete_exploratory_non_version_safe"``;
* ``generated_files["stock_day_features.parquet"]["sha256"]``;
* ordered ``d2_normalized_30`` and ``d2_levels_5`` feature lists;
* complete configured query roots and complete model-row eligibility; and
* explicit ``point_in_time_version_safe = false`` and
  ``primary_training_eligible = false`` limitations.

The resulting panel is development-only.  A complete retrospective join does
not repair historical article-version uncertainty.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
JOIN_KEYS = ["forecast_date", "sector", "stock", "benchmark"]
DEFAULT_QUANT = Path("data/features/quant/training_v1/modeling_panel.parquet")
DEFAULT_QUANT_MANIFEST = Path(
    "data/features/quant/training_v1/modeling_panel.manifest.json"
)
DEFAULT_NEWS = Path(
    "data/features/news_deterministic/massive_v2/"
    "stock_day_features.parquet"
)
DEFAULT_NEWS_MANIFEST = Path(
    "data/features/news_deterministic/massive_v2/manifest.json"
)
DEFAULT_OUTPUT_ROOT = Path("data/features/q_plus_d/massive_v2")
DEFAULT_OUTPUT_NAME = "modeling_panel_q_d2.parquet"
DEFAULT_PROTOCOL = Path("config/quant_deterministic_news_protocol_v2.json")
DEFAULT_PROTOCOL_HASH = Path(
    "config/quant_deterministic_news_protocol_v2.sha256"
)
MANIFEST_NAME = "manifest.json"
MANIFEST_HASH_NAME = "manifest.sha256"
BUILDER_VERSION = "q-plus-d2-massive-join-v2.0.0"
EXPECTED_MODEL_ROWS = 27_510
DEFAULT_START = "2022-11-01"
DEFAULT_END = "2026-06-30"

EXPLORATORY_WARNING = (
    "EXPLORATORY DEVELOPMENT ONLY: the retrospective Massive ordinary-news "
    "archive is complete for the configured query universe but is not "
    "historical-version safe. This exact join does not support confirmatory, "
    "causal, or production-readiness claims."
)

D2_NORMALIZED_30 = (
    "d2_target_direct_intensity_midrank_126",
    "d2_peer_idio_intensity_midrank_126",
    "d2_common_intensity_midrank_126",
    "d2_observed_no_direct_target_article",
    "d2_common_article_share",
    "d2_target_common_share",
    "d2_observed_no_peer_set_entity_mention",
    "d2_peer_set_entity_hhi",
    "d2_peer_coverage_ratio",
    "d2_target_peer_co_mention_share",
    "d2_target_attention_share_delta_63",
    "d2_macro_share_of_common",
    "d2_target_mean_recency_weight_12h",
    "d2_common_mean_recency_weight_12h",
    "d2_target_premarket_article_share",
    "d2_common_premarket_article_share",
    "d2_target_earnings_guidance_article_share",
    "d2_common_earnings_guidance_article_share",
    "d2_target_product_demand_article_share",
    "d2_common_product_demand_article_share",
    "d2_target_supply_capacity_article_share",
    "d2_common_supply_capacity_article_share",
    "d2_target_regulation_legal_article_share",
    "d2_common_regulation_legal_article_share",
    "d2_target_corporate_analyst_article_share",
    "d2_common_corporate_analyst_article_share",
    "d2_target_positive_surprise_article_share",
    "d2_target_negative_surprise_article_share",
    "d2_common_positive_surprise_article_share",
    "d2_common_negative_surprise_article_share",
)

D2_LEVELS_5 = (
    "d2_log1p_target_idio_article_count",
    "d2_log1p_target_common_article_count",
    "d2_log1p_peer_idio_article_count",
    "d2_log1p_sector_common_article_count",
    "d2_log1p_macro_common_article_count",
)

QUANT_FIXED_LOG1P_FEATURES = (
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
)

FOLDS = (
    {
        "name": "fold_1",
        "train_start": "2022-11-01",
        "train_end": "2024-06-30",
        "validation_start": "2024-07-01",
        "validation_end": "2024-12-31",
        "test_start": "2025-01-01",
        "test_end": "2025-06-30",
    },
    {
        "name": "fold_2",
        "train_start": "2022-11-01",
        "train_end": "2024-12-31",
        "validation_start": "2025-01-01",
        "validation_end": "2025-06-30",
        "test_start": "2025-07-01",
        "test_end": "2025-12-31",
    },
    {
        "name": "fold_3",
        "train_start": "2022-11-01",
        "train_end": "2025-06-30",
        "validation_start": "2025-07-01",
        "validation_end": "2025-12-31",
        "test_start": "2026-01-01",
        "test_end": "2026-06-30",
    },
)

TARGETS = ("t1_etf", "t1_loo", "t2_etf", "t2_loo")

EXPECTED_SPLIT_ROWS = {
    "fold_1": {
        "t1_train_validation_test": [12_480, 3_840, 3_660],
        "t2_train_validation_test": [12_360, 3_720, 3_540],
    },
    "fold_2": {
        "t1_train_validation_test": [16_320, 3_660, 3_840],
        "t2_train_validation_test": [16_200, 3_540, 3_720],
    },
    "fold_3": {
        "t1_train_validation_test": [19_980, 3_840, 3_690],
        "t2_train_validation_test": [19_860, 3_720, 3_570],
    },
}

SOURCE_PROFILE = "ordinary_massive_retrospective"
CLAIM_SCOPE = (
    "Exploratory development only: retrospective ordinary Massive news is "
    "not historical-version safe, primary-training eligible, or confirmatory."
)


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


def _ordered_unique_strings(value: Any, label: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
        or len(value) != len(set(value))
    ):
        raise ValueError(f"{label} must be a nonempty ordered unique list")
    return list(value)


def _validate_bound_file(
    path: Path, expected_sha256: Any, label: str
) -> str:
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
    ):
        raise ValueError(f"{label} has no valid bound SHA-256")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"{label} hash mismatch: expected {expected_sha256}, "
            f"observed {actual}"
        )
    return actual


def validate_source_contracts(
    *,
    quant_path: Path,
    quant_manifest_path: Path,
    news_path: Path,
    news_manifest_path: Path,
    expected_rows: int,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[str],
    list[str],
    list[str] | None,
]:
    """Validate source manifests and return exact ordered D2 feature lists."""

    quant_manifest = load_json_object(quant_manifest_path, "quant manifest")
    news_manifest = load_json_object(news_manifest_path, "D2 manifest")

    if quant_manifest.get("status") != "complete":
        raise ValueError("Quant manifest is not complete")
    quant_output = quant_manifest.get("output")
    if not isinstance(quant_output, dict):
        raise ValueError("Quant manifest has no output record")
    _validate_bound_file(
        quant_path, quant_output.get("sha256"), "quant artifact"
    )
    if quant_manifest.get("row_count") != expected_rows:
        raise ValueError(
            "Quant manifest row count does not equal the locked expectation"
        )

    if news_manifest.get("status") != (
        "complete_exploratory_non_version_safe"
    ):
        raise ValueError("D2 manifest is not complete under the v2 contract")
    generated = news_manifest.get("generated_files")
    if not isinstance(generated, dict):
        raise ValueError("D2 manifest has no generated_files record")
    stock_day_record = generated.get("stock_day_features.parquet")
    if not isinstance(stock_day_record, dict):
        raise ValueError(
            "D2 manifest does not bind stock_day_features.parquet"
        )
    _validate_bound_file(
        news_path, stock_day_record.get("sha256"), "D2 artifact"
    )

    ordered = news_manifest.get("ordered_feature_lists")
    if not isinstance(ordered, dict):
        raise ValueError("D2 manifest has no ordered_feature_lists record")
    d2 = _ordered_unique_strings(
        ordered.get("d2_normalized_30"), "d2_normalized_30"
    )
    levels = _ordered_unique_strings(
        ordered.get("d2_levels_5"), "d2_levels_5"
    )
    if tuple(d2) != D2_NORMALIZED_30:
        raise ValueError(
            "D2 manifest feature order does not match the frozen "
            "D2-Normalized 30-column contract"
        )
    if tuple(levels) != D2_LEVELS_5:
        raise ValueError(
            "D2 manifest feature order does not match the frozen "
            "D2-Levels five-column contract"
        )
    d43_raw = ordered.get("d43_recomputed_43")
    d43 = (
        _ordered_unique_strings(d43_raw, "d43_recomputed_43")
        if d43_raw is not None
        else None
    )
    if d43 is not None and len(d43) != 43:
        raise ValueError(
            "d43_recomputed_43 must contain exactly 43 names when declared"
        )

    completeness = news_manifest.get("source_completeness")
    if not isinstance(completeness, dict):
        raise ValueError("D2 manifest has no source_completeness record")
    required_true = (
        "all_configured_query_roots_complete",
        "all_model_rows_eligible",
    )
    incomplete = [
        name for name in required_true if completeness.get(name) is not True
    ]
    if incomplete:
        raise ValueError(
            "D2 source completeness failed for: " + ", ".join(incomplete)
        )
    if completeness.get("expected_model_row_count") != expected_rows:
        raise ValueError(
            "D2 source completeness row count does not match the lock"
        )

    limitations = news_manifest.get("source_limitations")
    if not isinstance(limitations, dict):
        raise ValueError("D2 manifest has no source_limitations record")
    if limitations.get("point_in_time_version_safe") is not False:
        raise ValueError("Retrospective D2 must be marked non-version-safe")
    if limitations.get("primary_training_eligible") is not False:
        raise ValueError(
            "Retrospective D2 must be marked primary-training-ineligible"
        )
    return quant_manifest, news_manifest, d2, levels, d43


def normalize_and_validate_keys(
    frame: pd.DataFrame, label: str
) -> pd.DataFrame:
    missing = set(JOIN_KEYS) - set(frame.columns)
    if missing:
        raise ValueError(f"{label} misses join keys: {sorted(missing)}")
    normalized = frame.copy()
    parsed = pd.to_datetime(normalized["forecast_date"], errors="raise")
    if parsed.isna().any():
        raise ValueError(f"{label} contains a missing forecast_date")
    dates = parsed.dt.normalize()
    if not parsed.equals(dates):
        raise ValueError(f"{label} forecast_date values must be date-only")
    normalized["forecast_date"] = dates
    for column in ("sector", "stock", "benchmark"):
        values = normalized[column].astype("string")
        if values.isna().any() or values.str.len().eq(0).any():
            raise ValueError(f"{label} contains an empty {column}")
        if not values.equals(values.str.strip()):
            raise ValueError(f"{label} contains noncanonical {column}")
        if column in {"stock", "benchmark"} and not values.equals(
            values.str.upper()
        ):
            raise ValueError(f"{label} contains lowercase {column}")
    if normalized.duplicated(JOIN_KEYS).any():
        examples = (
            normalized.loc[
                normalized.duplicated(JOIN_KEYS, keep=False), JOIN_KEYS
            ]
            .head(5)
            .astype(str)
            .to_dict("records")
        )
        raise ValueError(f"{label} has duplicate join keys: {examples}")
    return normalized


def _assert_complete_numeric_features(
    frame: pd.DataFrame, features: Sequence[str], label: str
) -> None:
    missing_columns = set(features) - set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"{label} misses declared features: {sorted(missing_columns)}"
        )
    null_counts = frame[list(features)].isna().sum()
    nullable = {
        column: int(count)
        for column, count in null_counts.items()
        if int(count)
    }
    if nullable:
        raise ValueError(f"{label} has missing feature values: {nullable}")
    for column in features:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.isna().any():
            raise ValueError(f"{label} feature {column} is not numeric")
        finite = numeric.map(math.isfinite)
        if not bool(finite.all()):
            raise ValueError(f"{label} feature {column} is not finite")


def _assert_nullable_numeric_features(
    frame: pd.DataFrame, features: Sequence[str], label: str
) -> None:
    missing_columns = set(features) - set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"{label} misses declared features: {sorted(missing_columns)}"
        )
    for column in features:
        source = frame[column]
        numeric = pd.to_numeric(source, errors="coerce")
        if numeric.notna().sum() != source.notna().sum():
            raise ValueError(f"{label} feature {column} is not numeric")
        finite = numeric.loc[numeric.notna()].map(math.isfinite)
        if not bool(finite.all()):
            raise ValueError(f"{label} feature {column} is not finite")


def build_joined_panel(
    quant_frame: pd.DataFrame,
    news_frame: pd.DataFrame,
    *,
    d2_features: Sequence[str],
    d2_level_features: Sequence[str],
    d43_features: Sequence[str] | None = None,
    start: pd.Timestamp,
    end: pd.Timestamp,
    expected_rows: int,
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    """Join exact key sets and prove that the quant columns are unchanged."""

    quant = normalize_and_validate_keys(quant_frame, "quant panel")
    news = normalize_and_validate_keys(news_frame, "D2 panel")
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    if start > end:
        raise ValueError("start must not be after end")
    quant = quant[
        quant["forecast_date"].between(start, end, inclusive="both")
    ].copy()
    news = news[
        news["forecast_date"].between(start, end, inclusive="both")
    ].copy()
    if len(quant) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} quant rows, observed {len(quant)}"
        )
    if len(news) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} D2 rows, observed {len(news)}"
        )

    quant_key_index = pd.MultiIndex.from_frame(quant[JOIN_KEYS])
    news_key_index = pd.MultiIndex.from_frame(news[JOIN_KEYS])
    missing_news = quant_key_index.difference(news_key_index)
    extra_news = news_key_index.difference(quant_key_index)
    if len(missing_news) or len(extra_news):
        raise ValueError(
            "Quant and D2 key sets differ: "
            f"missing_news={len(missing_news)}, extra_news={len(extra_news)}"
        )

    d43_features = list(d43_features or ())
    model_features = [*d2_features, *d2_level_features, *d43_features]
    if len(model_features) != len(set(model_features)):
        raise ValueError("D2 and D2-Levels feature lists overlap")
    _assert_complete_numeric_features(
        news, [*d2_features, *d2_level_features], "D2 panel"
    )
    _assert_nullable_numeric_features(news, d43_features, "D43 panel")
    payload = [column for column in news.columns if column not in JOIN_KEYS]
    collisions = set(payload).intersection(quant.columns)
    if collisions:
        raise ValueError(
            "D2 payload collides with quant columns: "
            f"{sorted(collisions)}"
        )

    quant_columns = list(quant.columns)
    quant["__quant_row_order"] = range(len(quant))
    news["d2_row_matched"] = 1
    joined = quant.merge(
        news[[*JOIN_KEYS, *payload, "d2_row_matched"]],
        on=JOIN_KEYS,
        how="left",
        sort=False,
        validate="one_to_one",
    )
    joined = (
        joined.sort_values("__quant_row_order", kind="stable")
        .drop(columns="__quant_row_order")
        .reset_index(drop=True)
    )
    expected_quant = quant.drop(columns="__quant_row_order").reset_index(
        drop=True
    )
    for column in quant_columns:
        joined[column] = expected_quant[column].array
    pd.testing.assert_frame_equal(
        joined[quant_columns],
        expected_quant[quant_columns],
        check_dtype=True,
        check_like=False,
    )
    if not joined["d2_row_matched"].eq(1).all():
        raise AssertionError("Exact key preflight passed but merge lost D2 rows")
    joined["d2_row_matched"] = joined["d2_row_matched"].astype("int8")
    _assert_complete_numeric_features(
        joined, [*d2_features, *d2_level_features], "joined panel"
    )
    _assert_nullable_numeric_features(joined, d43_features, "joined D43")

    audits = [
        column for column in payload if column not in set(model_features)
    ]
    stats = {
        "expected_model_rows": expected_rows,
        "quant_rows": int(len(quant)),
        "d2_rows": int(len(news)),
        "output_rows": int(len(joined)),
        "matched_rows": int(joined["d2_row_matched"].sum()),
        "match_fraction": float(joined["d2_row_matched"].mean()),
        "date_count": int(joined["forecast_date"].nunique()),
        "stock_count": int(joined["stock"].nunique()),
        "sector_count": int(joined["sector"].nunique()),
        "first_date": joined["forecast_date"].min().date().isoformat(),
        "last_date": joined["forecast_date"].max().date().isoformat(),
        "d2_missing_value_count": int(
            joined[list(d2_features)].isna().sum().sum()
        ),
        "d2_levels_missing_value_count": int(
            joined[list(d2_level_features)].isna().sum().sum()
        ),
        "d43_declared": bool(d43_features),
        "d43_missing_value_count": (
            int(joined[list(d43_features)].isna().sum().sum())
            if d43_features
            else None
        ),
    }
    return joined, stats, audits


def build_quant_feature_lists(
    quant_manifest: Mapping[str, Any],
    panel_columns: Sequence[str],
) -> dict[str, list[str]]:
    groups = quant_manifest.get("feature_groups")
    if not isinstance(groups, dict):
        raise ValueError("Quant manifest has no feature_groups record")

    def group(name: str, count: int) -> list[str]:
        values = _ordered_unique_strings(groups.get(name), name)
        if len(values) != count:
            raise ValueError(f"Quant feature group {name} must have {count}")
        return values

    etf = group("etf_core_22", 22)[:14]
    loo = group("loo_core_22", 22)[:14]
    # Guard against relying only on position: the first 14 fields must be the
    # target-specific pair block and the final eight must be sector state.
    if any(not value.startswith("etf_") for value in etf):
        raise ValueError("Quant ETF pair block is not canonical")
    if any(not value.startswith("loo_") for value in loo):
        raise ValueError("Quant LOO pair block is not canonical")
    sector_etf = group("etf_core_22", 22)[14:]
    sector_loo = group("loo_core_22", 22)[14:]
    if sector_etf != sector_loo or len(sector_etf) != 8:
        raise ValueError("Quant sector-state block differs by target")
    dense = group("dense_context", 15)
    volatility = group("volatility", 2)
    extended = group("extended_hours", 17)

    common = [*sector_etf, *dense, *volatility, *extended]
    output = {
        "t1_etf": [*etf, *common],
        "t2_etf": [*etf, *common],
        "t1_loo": [*loo, *common],
        "t2_loo": [*loo, *common],
    }
    panel_set = set(panel_columns)
    for target, features in output.items():
        if len(features) != 56 or len(features) != len(set(features)):
            raise ValueError(f"Q56 contract failed for {target}")
        missing = set(features) - panel_set
        if missing:
            raise ValueError(
                f"Q56 contract for {target} misses columns: {sorted(missing)}"
            )
    return output


def build_locked_protocol(
    *,
    quant_features: Mapping[str, Sequence[str]],
    d2_features: Sequence[str],
    d2_level_features: Sequence[str],
    d43_features: Sequence[str] | None,
    joined_path: Path,
    joined_manifest_path: Path,
    joined_sha256: str,
    joined_manifest_sha256: str,
    row_count: int,
    date_count: int,
    stock_count: int,
    sector_count: int,
    start: str,
    end: str,
) -> dict[str, Any]:
    """Return the deterministic v2 protocol bound to constructed artifacts."""

    q56_etf = list(quant_features["t1_etf"])
    q56_loo = list(quant_features["t1_loo"])
    if q56_etf != list(quant_features["t2_etf"]):
        raise ValueError("T1/T2 ETF Q56 contracts differ")
    if q56_loo != list(quant_features["t2_loo"]):
        raise ValueError("T1/T2 LOO Q56 contracts differ")
    d43 = list(d43_features or ())
    d43_available = bool(d43)
    if d43_available and len(d43) != 43:
        raise ValueError("Available D43 contract must have exactly 43 names")
    d43_reason = (
        "The exact D43 comparator was not materialized by the v2 builder; no "
        "43-name contract is fabricated from the historical v1 panel."
    )
    deterministic_rungs = [
        {
            "name": "V2-D0",
            "estimator": "elastic_net",
            "features": "q56_target_specific",
            "raw_feature_count": 56,
            "construction_available": True,
        },
        {
            "name": "V2-D1",
            "estimator": "elastic_net",
            "features": "d2_normalized_30",
            "raw_feature_count": 30,
            "construction_available": True,
        },
        {
            "name": "V2-D2",
            "estimator": "elastic_net",
            "features": "q56_target_specific + d43_recomputed_43",
            "raw_feature_count": 99,
            "construction_available": d43_available,
            **({"blocked_reason": d43_reason} if not d43_available else {}),
        },
        {
            "name": "V2-D3",
            "estimator": "elastic_net",
            "features": "q56_target_specific + d2_normalized_30",
            "raw_feature_count": 86,
            "construction_available": True,
        },
        {
            "name": "V2-D4",
            "estimator": "elastic_net",
            "features": (
                "q56_target_specific + d2_normalized_30 + d2_levels_5"
            ),
            "raw_feature_count": 91,
            "construction_available": True,
            "role": "provider-level sensitivity",
        },
        {
            "name": "V2-D5",
            "estimator": "shallow_xgboost",
            "features": "q56_target_specific + d2_normalized_30",
            "raw_feature_count": 86,
            "construction_available": True,
            "eligibility": (
                "validation gated after the corresponding linear endpoint"
            ),
        },
    ]
    return {
        "protocol_version": "2.0",
        "status": "locked_before_training",
        "experiment_id": "quant-deterministic-news-v2",
        "generated_at_utc": utc_now(),
        "claim_scope": CLAIM_SCOPE,
        "source_profile": SOURCE_PROFILE,
        "source_artifacts": {
            "panel_path": str(joined_path),
            "panel_sha256": joined_sha256,
            "panel_manifest_path": str(joined_manifest_path),
            "panel_manifest_sha256": joined_manifest_sha256,
            "row_count": row_count,
            "date_count": date_count,
            "stock_count": stock_count,
            "sector_count": sector_count,
            "first_date": start,
            "last_date": end,
        },
        "join_keys": JOIN_KEYS,
        "targets": list(TARGETS),
        "folds": [dict(value) for value in FOLDS],
        "target_row_eligibility": {
            "t1": "forecast_date lies within the current partition",
            "t2": (
                "forecast_date lies within the current partition and the "
                "target-specific end-date column is <= partition end"
            ),
            "t2_effect": (
                "purge the final four forecast sessions from each train, "
                "validation, and development-evaluation partition"
            ),
            "expected_rows": EXPECTED_SPLIT_ROWS,
        },
        "row_eligibility": {
            "required_equalities": {"d2_row_matched": 1},
        },
        "feature_blocks": {
            "q56_etf": q56_etf,
            "q56_loo": q56_loo,
            "d2_normalized_30": list(d2_features),
            "d2_levels_5": list(d2_level_features),
            "d43_recomputed_43": d43,
        },
        "feature_list_sha256": {
            "q56_etf": canonical_json_sha256(q56_etf),
            "q56_loo": canonical_json_sha256(q56_loo),
            "d2_normalized_30": canonical_json_sha256(list(d2_features)),
            "d2_levels_5": canonical_json_sha256(
                list(d2_level_features)
            ),
            "d43_recomputed_43": canonical_json_sha256(d43),
        },
        "bundle_availability": {
            "D0": {"available": True},
            "D1": {"available": True},
            "D2": {
                "available": d43_available,
                **({"reason": d43_reason} if not d43_available else {}),
            },
            "D3": {"available": True},
            "D4": {"available": True},
        },
        "deterministic_rungs": deterministic_rungs,
        "semantic_construction_status": {
            "wllm17": "not_constructed",
            "rllm70": "not_constructed",
            "semantic_rungs_locked": False,
        },
        "preprocessing": {
            "elastic_net_imputation": "training-partition median only",
            "elastic_net_scaling": (
                "training-partition mean and standard deviation only"
            ),
            "xgboost_missing_policy": "native missing branches",
            "d2_complete_case_contract": True,
            "fixed_log1p_features": list(QUANT_FIXED_LOG1P_FEATURES),
        },
        "model_tuning": {
            "linear_alpha_grid": [
                0.0001,
                0.0003,
                0.001,
                0.003,
                0.01,
                0.03,
                0.1,
            ],
            "elastic_net_l1_ratio_grid": [0.1, 0.5, 0.9, 1.0],
            "reuse_quant_v1_validation_only_boundary_rule": True,
            "reuse_quant_v1_shallow_xgboost_candidates": True,
        },
        "metrics": [
            "fisher_z_rmse",
            "fisher_z_mae",
            "raw_correlation_rmse",
            "raw_correlation_mae",
            "oos_r2_vs_persistence",
            "incremental_r2_vs_exact_matched_base",
        ],
        "inference": {
            "bootstrap_unit": "whole forecast dates with all stocks",
            "moving_block_sessions": 10,
            "bootstrap_resamples": 2000,
            "seed": 1729,
        },
    }


def schema_records(frame: pd.DataFrame) -> list[dict[str, str]]:
    return [
        {"name": column, "dtype": str(frame[column].dtype)}
        for column in frame.columns
    ]


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8", newline="\n")
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


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--quant", type=Path, default=DEFAULT_QUANT)
    result.add_argument(
        "--quant-manifest", type=Path, default=DEFAULT_QUANT_MANIFEST
    )
    result.add_argument("--news", type=Path, default=DEFAULT_NEWS)
    result.add_argument(
        "--news-manifest", type=Path, default=DEFAULT_NEWS_MANIFEST
    )
    result.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT
    )
    result.add_argument(
        "--protocol-output", type=Path, default=DEFAULT_PROTOCOL
    )
    result.add_argument(
        "--protocol-hash-output", type=Path, default=DEFAULT_PROTOCOL_HASH
    )
    result.add_argument("--start", default=DEFAULT_START)
    result.add_argument("--end", default=DEFAULT_END)
    result.add_argument(
        "--expected-row-count", type=int, default=EXPECTED_MODEL_ROWS
    )
    result.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace existing v2 join/protocol outputs.",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.expected_row_count <= 0:
        raise ValueError("expected-row-count must be positive")
    output_path = args.output_root / DEFAULT_OUTPUT_NAME
    manifest_path = args.output_root / MANIFEST_NAME
    manifest_hash_path = args.output_root / MANIFEST_HASH_NAME
    outputs = (
        output_path,
        manifest_path,
        manifest_hash_path,
        args.protocol_output,
        args.protocol_hash_output,
    )
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite v2 join/protocol artifacts without "
            "--overwrite: " + ", ".join(str(path) for path in existing)
        )

    quant_manifest, news_manifest, d2, levels, d43 = (
        validate_source_contracts(
            quant_path=args.quant,
            quant_manifest_path=args.quant_manifest,
            news_path=args.news,
            news_manifest_path=args.news_manifest,
            expected_rows=args.expected_row_count,
        )
    )
    quant = pd.read_parquet(args.quant)
    news = pd.read_parquet(args.news)
    joined, stats, news_audits = build_joined_panel(
        quant,
        news,
        d2_features=d2,
        d2_level_features=levels,
        d43_features=d43,
        start=pd.Timestamp(args.start),
        end=pd.Timestamp(args.end),
        expected_rows=args.expected_row_count,
    )
    quant_features = build_quant_feature_lists(
        quant_manifest, list(quant.columns)
    )
    atomic_write_parquet(output_path, joined)

    schema = schema_records(joined)
    manifest: dict[str, Any] = {
        "manifest_version": "q-plus-d2-massive-manifest-v2",
        "builder_version": BUILDER_VERSION,
        "builder_script_sha256": sha256_file(Path(__file__)),
        "status": "complete_exploratory_non_version_safe",
        "generated_at_utc": utc_now(),
        "warning": EXPLORATORY_WARNING,
        "claim_scope": {
            "development_only": True,
            "exploratory_only": True,
            "point_in_time_version_safe": False,
            "primary_training_eligible": False,
            "confirmatory_eligible": False,
        },
        "join_contract": {
            "keys": JOIN_KEYS,
            "key_set_equality_required": True,
            "quant_row_order_preserved": True,
            "quant_values_and_dtypes_preserved": True,
            "d2_missing_values_permitted": False,
            "d2_nonfinite_values_permitted": False,
            "requested_start": args.start,
            "requested_end": args.end,
        },
        "counts": stats,
        "ordered_feature_lists": {
            "d2_normalized_30": d2,
            "d2_levels_5": levels,
            "d43_recomputed_43": list(d43 or ()),
            "q56_by_target": quant_features,
        },
        "feature_list_sha256": {
            "d2_normalized_30": canonical_json_sha256(d2),
            "d2_levels_5": canonical_json_sha256(levels),
            "d43_recomputed_43": canonical_json_sha256(list(d43 or ())),
            "q56_by_target": {
                key: canonical_json_sha256(value)
                for key, value in quant_features.items()
            },
        },
        "column_roles": {
            "join_keys": JOIN_KEYS,
            "quant_columns": list(quant.columns),
            "d2_normalized_features": d2,
            "d2_levels_features": levels,
            "d43_recomputed_features": list(d43 or ()),
            "d2_audit_columns": news_audits,
            "join_audit_columns": ["d2_row_matched"],
        },
        "schema": schema,
        "schema_sha256": canonical_json_sha256(schema),
        "source_limitations": news_manifest["source_limitations"],
        "inputs": {
            "quant": {
                "path": str(args.quant),
                "sha256": sha256_file(args.quant),
                "manifest_path": str(args.quant_manifest),
                "manifest_sha256": sha256_file(args.quant_manifest),
            },
            "d2": {
                "path": str(args.news),
                "sha256": sha256_file(args.news),
                "manifest_path": str(args.news_manifest),
                "manifest_sha256": sha256_file(args.news_manifest),
            },
        },
        "output": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "bytes": output_path.stat().st_size,
            "row_count": len(joined),
            "column_count": len(joined.columns),
        },
    }
    atomic_write_json(manifest_path, manifest)
    manifest_sha = write_hash_sidecar(manifest_hash_path, manifest_path)

    protocol = build_locked_protocol(
        quant_features=quant_features,
        d2_features=d2,
        d2_level_features=levels,
        d43_features=d43,
        joined_path=output_path,
        joined_manifest_path=manifest_path,
        joined_sha256=manifest["output"]["sha256"],
        joined_manifest_sha256=manifest_sha,
        row_count=len(joined),
        date_count=stats["date_count"],
        stock_count=stats["stock_count"],
        sector_count=stats["sector_count"],
        start=args.start,
        end=args.end,
    )
    atomic_write_json(args.protocol_output, protocol)
    protocol_sha = write_hash_sidecar(
        args.protocol_hash_output, args.protocol_output
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "panel": manifest["output"],
                "manifest": str(manifest_path),
                "manifest_sha256": manifest_sha,
                "protocol": str(args.protocol_output),
                "protocol_sha256": protocol_sha,
                "counts": stats,
                "warning": EXPLORATORY_WARNING,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
