"""Build an audited, left-preserving quant + deterministic-news panel.

The Massive ordinary-news source used by the default inputs does not expose
first-seen timestamps, update timestamps, historical article versions, or
article bodies.  The resulting artifact is therefore deliberately marked as
exploratory and non-version-safe even when every modeling key matches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_bollerslev_core_features as quant_io  # noqa: E402


JOIN_KEYS = ["forecast_date", "sector", "stock", "benchmark"]
DEFAULT_QUANT = Path(
    "data/features/quant/training_v1/modeling_panel.parquet"
)
DEFAULT_QUANT_MANIFEST = Path(
    "data/features/quant/training_v1/modeling_panel.manifest.json"
)
DEFAULT_NEWS = Path(
    "data/features/news_deterministic/massive_v1/"
    "stock_day_features.csv.gz"
)
DEFAULT_NEWS_MANIFEST = Path(
    "data/features/news_deterministic/massive_v1/manifest.json"
)
DEFAULT_OUTPUT_ROOT = Path("data/features/q_plus_d/massive_v1")
DEFAULT_OUTPUT_NAME = "modeling_panel_q_plus_d.parquet"
MANIFEST_NAME = "manifest.json"
MANIFEST_HASH_NAME = "manifest.sha256"
BUILDER_VERSION = "q-plus-d-massive-join-v1.0.1"
EXPLORATORY_WARNING = (
    "EXPLORATORY ONLY: Massive ordinary news lacks first-seen timestamps, "
    "last-updated timestamps, historical article versions, and full article "
    "bodies. Publication time and the current description are proxies, so "
    "this panel is not point-in-time version safe and is not eligible to "
    "support primary causal or production-training claims."
)
PROTECTED_OUTPUT_PARTS = (
    ("experiments", "flan_t5_xl"),
    ("outputs", "flan_t5_xl"),
    ("experiments", "llama_3_1"),
    ("outputs", "llama_3_1"),
)
DECLARED_EXACT_REDUNDANCIES = {
    "timing_eligible_article_count": {
        "relationship": "equal",
        "other": "observed_relevant_article_count",
    },
    "timing_eligible_share": {
        "relationship": "one_minus",
        "other": "observed_no_relevant_news",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
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
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_output_root(path: Path) -> None:
    resolved = path.resolve()
    for parts in PROTECTED_OUTPUT_PARTS:
        protected = ROOT.joinpath(*parts).resolve()
        if _is_within(resolved, protected):
            raise ValueError(
                "Q+D output must not touch an active extractor path: "
                f"{resolved}"
            )


def _validate_manifest_hash(
    *,
    actual_path: Path,
    expected_hash: Any,
    label: str,
) -> str:
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError(f"{label} manifest has no valid SHA-256")
    actual_hash = sha256_file(actual_path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"{label} hash mismatch: expected {expected_hash}, "
            f"observed {actual_hash}"
        )
    return actual_hash


def validate_source_contracts(
    *,
    quant_path: Path,
    quant_manifest_path: Path,
    news_path: Path,
    news_manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Validate both source artifacts and return their declared news features."""

    quant_manifest = load_json_object(quant_manifest_path, "quant manifest")
    news_manifest = load_json_object(news_manifest_path, "news manifest")

    if quant_manifest.get("status") != "complete":
        raise ValueError("Quant manifest is not complete")
    quant_output = quant_manifest.get("output")
    if not isinstance(quant_output, dict):
        raise ValueError("Quant manifest has no output record")
    _validate_manifest_hash(
        actual_path=quant_path,
        expected_hash=quant_output.get("sha256"),
        label="quant artifact",
    )

    generated = news_manifest.get("generated_files")
    if not isinstance(generated, dict):
        raise ValueError("News manifest has no generated_files record")
    stock_day_record = generated.get("stock_day_features.csv.gz")
    if not isinstance(stock_day_record, dict):
        raise ValueError(
            "News manifest does not bind stock_day_features.csv.gz"
        )
    _validate_manifest_hash(
        actual_path=news_path,
        expected_hash=stock_day_record.get("sha256"),
        label="news artifact",
    )

    completeness = news_manifest.get("collection_completeness")
    if not isinstance(completeness, dict):
        raise ValueError("News manifest has no collection-completeness record")
    required_complete_flags = (
        "ordinary_ticker_collection_complete",
        "sector_benchmark_collection_complete",
        "control_collection_complete",
    )
    incomplete = [
        name for name in required_complete_flags if completeness.get(name) is not True
    ]
    if incomplete:
        raise ValueError(
            "News source collection is incomplete for: "
            + ", ".join(incomplete)
        )

    limitations = news_manifest.get("data_limitations")
    if not isinstance(limitations, dict):
        raise ValueError("News manifest has no data-limitations record")
    if limitations.get("point_in_time_version_safe") is not False:
        raise ValueError(
            "Expected the Massive ordinary-news source to be explicitly "
            "marked non-version-safe"
        )
    if limitations.get("primary_training_eligible") is not False:
        raise ValueError(
            "Expected the Massive ordinary-news source to be explicitly "
            "marked ineligible for primary training claims"
        )

    features = news_manifest.get("constructable_q_plus_d_features")
    if (
        not isinstance(features, list)
        or not features
        or any(not isinstance(item, str) or not item for item in features)
        or len(features) != len(set(features))
    ):
        raise ValueError(
            "News manifest has no valid unique constructable feature list"
        )
    return quant_manifest, news_manifest, features


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
    midnight = parsed.dt.normalize()
    if not parsed.equals(midnight):
        raise ValueError(f"{label} forecast_date values must be date-only")
    normalized["forecast_date"] = midnight

    for column in ("sector", "stock", "benchmark"):
        values = normalized[column].astype("string")
        if values.isna().any() or (values.str.len() == 0).any():
            raise ValueError(f"{label} contains an empty {column}")
        if not values.equals(values.str.strip()):
            raise ValueError(f"{label} contains non-canonical {column} values")
        if column in {"stock", "benchmark"} and not values.equals(
            values.str.upper()
        ):
            raise ValueError(f"{label} contains lowercase {column} values")

    if normalized.duplicated(JOIN_KEYS).any():
        duplicate = (
            normalized.loc[
                normalized.duplicated(JOIN_KEYS, keep=False), JOIN_KEYS
            ]
            .head(5)
            .astype(str)
            .to_dict("records")
        )
        raise ValueError(f"{label} has duplicate join keys: {duplicate}")
    return normalized


def build_joined_panel(
    quant_frame: pd.DataFrame,
    news_frame: pd.DataFrame,
    *,
    declared_news_features: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    require_full_match: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    """Left-join news to quant rows while preserving quant order and values."""

    quant = normalize_and_validate_keys(quant_frame, "quant panel")
    news = normalize_and_validate_keys(news_frame, "news panel")
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    if start > end:
        raise ValueError("start must not be after end")

    quant = quant[
        quant["forecast_date"].between(start, end, inclusive="both")
    ].copy()
    if quant.empty:
        raise ValueError("No quant rows remain in the requested date range")
    news = news[
        news["forecast_date"].between(start, end, inclusive="both")
    ].copy()

    missing_features = set(declared_news_features) - set(news.columns)
    if missing_features:
        raise ValueError(
            "News panel misses declared model features: "
            f"{sorted(missing_features)}"
        )
    news_payload = [column for column in news.columns if column not in JOIN_KEYS]
    collisions = set(news_payload) & set(quant.columns)
    if collisions:
        raise ValueError(
            "News payload collides with quant columns: "
            f"{sorted(collisions)}"
        )
    audit_columns = [
        column
        for column in news_payload
        if column not in set(declared_news_features)
    ]

    quant_columns = list(quant.columns)
    quant["__quant_row_order"] = range(len(quant))
    news["news_row_matched"] = 1
    joined = quant.merge(
        news[[*JOIN_KEYS, *news_payload, "news_row_matched"]],
        on=JOIN_KEYS,
        how="left",
        sort=False,
        validate="one_to_one",
    )
    joined = joined.sort_values("__quant_row_order", kind="stable")
    joined = joined.drop(columns="__quant_row_order").reset_index(drop=True)
    joined["news_row_matched"] = (
        joined["news_row_matched"].fillna(0).astype("int8")
    )

    expected_quant = quant.drop(columns="__quant_row_order").reset_index(
        drop=True
    )
    # Pandas may coerce key dtypes when a Parquet StringDtype key is merged
    # with an object-dtype CSV key. Restore every left column from the audited
    # left frame after order recovery so the output preserves the quant schema
    # as well as its values.
    for column in quant_columns:
        joined[column] = expected_quant[column].array
    pd.testing.assert_frame_equal(
        joined[quant_columns],
        expected_quant[quant_columns],
        check_dtype=True,
        check_like=False,
    )
    if len(joined) != len(expected_quant):
        raise AssertionError("Left join did not preserve the quant row count")

    matched = int(joined["news_row_matched"].sum())
    unmatched = len(joined) - matched
    if require_full_match and unmatched:
        examples = (
            joined.loc[joined["news_row_matched"].eq(0), JOIN_KEYS]
            .head(10)
            .astype(str)
            .to_dict("records")
        )
        raise ValueError(
            f"{unmatched} quant rows have no deterministic-news row: "
            f"{examples}"
        )

    stats: dict[str, Any] = {
        "quant_rows_in_requested_range": int(len(expected_quant)),
        "output_rows": int(len(joined)),
        "news_rows_in_requested_range": int(len(news)),
        "matched_news_rows": matched,
        "unmatched_quant_rows": int(unmatched),
        "match_fraction": float(matched / len(joined)),
        "first_date": joined["forecast_date"].min().date().isoformat(),
        "last_date": joined["forecast_date"].max().date().isoformat(),
        "date_count": int(joined["forecast_date"].nunique()),
        "stock_count": int(joined["stock"].nunique()),
        "sector_count": int(joined["sector"].nunique()),
    }
    return joined, stats, audit_columns


def schema_records(frame: pd.DataFrame) -> list[dict[str, str]]:
    return [
        {"name": column, "dtype": str(frame[column].dtype)}
        for column in frame.columns
    ]


def deterministic_feature_preflight(
    frame: pd.DataFrame,
    features: Sequence[str],
) -> dict[str, Any]:
    """Identify zero-variance and predeclared exact-redundant news columns."""

    constant = [
        feature
        for feature in features
        if frame[feature].nunique(dropna=False) <= 1
    ]
    verified_redundancies: list[dict[str, str]] = []
    for feature, contract in DECLARED_EXACT_REDUNDANCIES.items():
        other = contract["other"]
        if feature not in frame.columns or other not in frame.columns:
            continue
        left = pd.to_numeric(frame[feature], errors="raise")
        if contract["relationship"] == "equal":
            right = pd.to_numeric(frame[other], errors="raise")
        elif contract["relationship"] == "one_minus":
            right = 1 - pd.to_numeric(frame[other], errors="raise")
        else:  # pragma: no cover - static contract guard
            raise AssertionError(
                f"Unknown redundancy relationship {contract['relationship']!r}"
            )
        matches = bool(
            ((left == right) | (left.isna() & right.isna())).all()
        )
        if matches:
            verified_redundancies.append(
                {
                    "feature": feature,
                    "relationship": contract["relationship"],
                    "other": other,
                }
            )
    excluded = {
        *constant,
        *(record["feature"] for record in verified_redundancies),
    }
    recommended = [feature for feature in features if feature not in excluded]
    missingness = {
        feature: {
            "missing_count": int(frame[feature].isna().sum()),
            "missing_fraction": float(frame[feature].isna().mean()),
        }
        for feature in recommended
        if frame[feature].isna().any()
    }
    return {
        "materialized_feature_count": len(features),
        "nonconstant_feature_count": len(features) - len(constant),
        "constant_features": constant,
        "verified_exact_redundancies": verified_redundancies,
        "recommended_feature_count": len(recommended),
        "recommended_feature_columns": recommended,
        "recommended_feature_missingness": missingness,
        "recommended_fit_policy": (
            "Fit only the recommended columns unless a pipeline performs and "
            "records an equivalent training-fold-only zero-variance and exact-"
            "redundancy filter. Fit imputation values on each training fold "
            "only, and retain missingness indicators for nullable recency and "
            "history features."
        ),
    }


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_manifest_and_hash(
    manifest_path: Path,
    hash_path: Path,
    manifest: Mapping[str, Any],
) -> str:
    quant_io.write_json_atomic(manifest_path, manifest)
    manifest_hash = sha256_file(manifest_path)
    atomic_write_text(
        hash_path, f"{manifest_hash}  {manifest_path.name}\n"
    )
    return manifest_hash


def parser() -> argparse.ArgumentParser:
    output = argparse.ArgumentParser(description=__doc__)
    output.add_argument("--quant", type=Path, default=DEFAULT_QUANT)
    output.add_argument(
        "--quant-manifest", type=Path, default=DEFAULT_QUANT_MANIFEST
    )
    output.add_argument("--news", type=Path, default=DEFAULT_NEWS)
    output.add_argument(
        "--news-manifest", type=Path, default=DEFAULT_NEWS_MANIFEST
    )
    output.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT
    )
    output.add_argument("--start", default="2022-11-01")
    output.add_argument("--end", default="2026-06-30")
    output.add_argument(
        "--allow-unmatched-news",
        action="store_true",
        help=(
            "Preserve unmatched quant rows with news_row_matched=0 instead of "
            "failing. Missing news values are never zero-imputed."
        ),
    )
    output.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing Q+D artifact atomically.",
    )
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    validate_output_root(args.output_root)
    output_path = args.output_root / DEFAULT_OUTPUT_NAME
    manifest_path = args.output_root / MANIFEST_NAME
    manifest_hash_path = args.output_root / MANIFEST_HASH_NAME
    existing = [
        path
        for path in (output_path, manifest_path, manifest_hash_path)
        if path.exists()
    ]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing Q+D outputs without --overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    quant_manifest, news_manifest, news_features = validate_source_contracts(
        quant_path=args.quant,
        quant_manifest_path=args.quant_manifest,
        news_path=args.news,
        news_manifest_path=args.news_manifest,
    )
    quant = pd.read_parquet(args.quant)
    news = pd.read_csv(args.news, low_memory=False)
    joined, stats, news_audit_columns = build_joined_panel(
        quant,
        news,
        declared_news_features=news_features,
        start=pd.Timestamp(args.start),
        end=pd.Timestamp(args.end),
        require_full_match=not args.allow_unmatched_news,
    )
    quant_columns = list(quant.columns)
    output_columns = list(joined.columns)
    feature_preflight = deterministic_feature_preflight(joined, news_features)
    quant_io.write_frame_atomic(output_path, joined)

    schema = schema_records(joined)
    source_limitations = news_manifest["data_limitations"]
    manifest: dict[str, Any] = {
        "manifest_version": "q-plus-d-massive-manifest-v1",
        "builder_version": BUILDER_VERSION,
        "builder_script_sha256": sha256_file(Path(__file__)),
        "status": "complete_exploratory_non_version_safe",
        "generated_at_utc": datetime.now(UTC)
        .isoformat()
        .replace("+00:00", "Z"),
        "warning": EXPLORATORY_WARNING,
        "claim_scope": {
            "exploratory_only": True,
            "point_in_time_version_safe": False,
            "primary_training_eligible": False,
            "safe_uses": [
                "pipeline integration",
                "weak-data ablation",
                "sensitivity analysis",
                "provider feasibility analysis",
            ],
            "prohibited_primary_claims": [
                "causal news effects",
                "strict point-in-time news backtest",
                "production model readiness",
            ],
        },
        "join_contract": {
            "keys": JOIN_KEYS,
            "join_type": "left",
            "left_artifact": "quant modeling panel",
            "right_artifact": "Massive deterministic stock-day features",
            "requested_start": pd.Timestamp(args.start).date().isoformat(),
            "requested_end": pd.Timestamp(args.end).date().isoformat(),
            "require_full_news_match": not args.allow_unmatched_news,
            "missing_news_policy": (
                "preserve row, set news_row_matched=0, leave payload missing"
            ),
            "quant_row_order_preserved": True,
            "quant_values_preserved": True,
        },
        "counts": stats,
        "column_roles": {
            "join_keys": JOIN_KEYS,
            "quant_columns": quant_columns,
            "deterministic_news_model_features": news_features,
            "deterministic_news_audit_columns": news_audit_columns,
            "join_audit_columns": ["news_row_matched"],
            "total_output_columns": len(output_columns),
        },
        "modeling_preflight": feature_preflight,
        "schema": schema,
        "schema_sha256": canonical_json_sha256(schema),
        "source_limitations": source_limitations,
        "inputs": {
            "quant": {
                "path": str(args.quant),
                "sha256": sha256_file(args.quant),
                "manifest_path": str(args.quant_manifest),
                "manifest_sha256": sha256_file(args.quant_manifest),
                "source_status": quant_manifest["status"],
            },
            "deterministic_news": {
                "path": str(args.news),
                "sha256": sha256_file(args.news),
                "manifest_path": str(args.news_manifest),
                "manifest_sha256": sha256_file(args.news_manifest),
                "source_status": news_manifest["status"],
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
    manifest_hash = write_manifest_and_hash(
        manifest_path, manifest_hash_path, manifest
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "output": manifest["output"],
                "counts": stats,
                "manifest": str(manifest_path),
                "manifest_sha256": manifest_hash,
                "warning": EXPLORATORY_WARNING,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
