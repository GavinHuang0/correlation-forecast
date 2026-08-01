#!/usr/bin/env python
"""Build hash-bound daily W17-Lite features from completed FLAN inference.

This module is deliberately separate from ``flan_w17_lite.py``.  The latter
is an immutable input whose hash is part of the completed selection and
inference contracts.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import tempfile
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from scripts.semantic_news_v3 import flan_w17_lite as inference


ROOT = Path(__file__).resolve().parents[2]
PIPELINE_VERSION = "flan-w17-lite-daily-v1.0.0"
MANIFEST_VERSION = "flan-w17-lite-daily-manifest-v1"
MANIFEST_STATUS = "complete_exploratory_non_version_safe"
ARM_ID = "WL17__flan_t5_xl"
CONTRACT_ID = "weak-news-semantics-lite-v1"
SOURCE_PROFILE = "ordinary_massive_retrospective"
KEY_COLUMNS = ("forecast_date", "sector", "stock", "benchmark")

DEFAULT_WORK_ROOT = (
    ROOT
    / "data"
    / "features"
    / "news_semantic"
    / "massive_v3"
    / "flan_w17_lite_k16"
)
DEFAULT_PREDICTIONS = DEFAULT_WORK_ROOT / "predictions.jsonl"
DEFAULT_INFERENCE_MANIFEST = DEFAULT_WORK_ROOT / "predictions.jsonl.manifest.json"
DEFAULT_SELECTION_MANIFEST = DEFAULT_WORK_ROOT / "selection_manifest.json"
DEFAULT_SELECTED_ASSIGNMENTS = DEFAULT_WORK_ROOT / "selected_assignments.parquet"
DEFAULT_SOURCE_ASSIGNMENTS = (
    ROOT
    / "data"
    / "features"
    / "news_semantic"
    / "massive_v2"
    / "article_target_assignments.parquet"
)
DEFAULT_STOCK_DAY_SCOPE = DEFAULT_SOURCE_ASSIGNMENTS.parent / "stock_day_scope.jsonl.gz"
DEFAULT_CORPUS_MANIFEST = DEFAULT_SOURCE_ASSIGNMENTS.parent / "manifest.json"
DEFAULT_DESIGN = ROOT / "config" / "news_semantic_lite_design_v1.json"
DEFAULT_EVENT_SCHEMA = ROOT / "config" / "flan_w17_lite_event_schema_v1.json"
DEFAULT_AGGREGATION = ROOT / "config" / "flan_w17_lite_aggregation_v1.json"

VARIANT_FILES = {
    "canonical": "daily_features.parquet",
    "long_description": "daily_features_long_description.parquet",
    "permuted": "daily_features_permuted.parquet",
}
MANIFEST_FILES = {
    variant: filename.removesuffix(".parquet") + ".manifest.json"
    for variant, filename in VARIANT_FILES.items()
}

FEATURE_COLUMNS = (
    "wlite_event_share_firm_operating_financial",
    "wlite_event_share_policy_corporate",
    "wlite_event_share_macro_market",
    "wlite_route_share_target_idiosyncratic",
    "wlite_route_share_peer_idiosyncratic",
    "wlite_rule_status_cue_share_confirmed_action",
    "wlite_rule_status_cue_share_scheduled_expected",
    "wlite_rule_status_cue_share_rumor_unconfirmed",
    "wlite_rule_status_cue_share_analysis_opinion",
    "wlite_rule_status_cue_conflict_weight_share",
    "wlite_selection_weight_coverage",
    "wlite_selected_headline_only_weight_share",
    "wlite_selected_sub150_weight_share",
    "wlite_event_order_disagreement_weight_share",
    "wlite_observed_no_selected_article",
    "wlite_event_entropy_accepted",
    "wlite_selected_weight_hhi",
)

AUDIT_COLUMNS = (
    "wlite_full_assignment_count",
    "wlite_selected_assignment_count",
    "wlite_full_assignment_weight",
    "wlite_selected_assignment_weight",
    "wlite_route_share_common",
    "wlite_event_accepted_coverage",
    "wlite_event_other_or_unclear_weight_share",
    "wlite_rule_status_cue_all_zero_weight_share",
    "wlite_invalid_or_truncated_selected_article_count",
    "wlite_event_decomposition_residual",
)

SOURCE_COLUMNS = (
    "source_profile",
    "point_in_time_version_safe",
    "primary_training_eligible",
    "confirmatory_eligible",
    "exploratory_construction_eligible",
    "semantic_quality_gate_passed",
    "exploratory_quality_gate_override",
    "variant_id",
)

SOURCE_ASSIGNMENT_COLUMNS = (
    "assignment_id",
    "article_id",
    "forecast_date",
    "target_ticker",
    "sector",
    "benchmark",
    "role",
    "aggregation_weight",
    "source_profile",
    "source_query_scope_complete",
    "candidate_assignment_complete",
    "headline",
    "description",
    "model_text",
    "headline_sha256",
    "description_sha256",
    "text_sha256",
    "description_available",
)

SELECTED_COMPARISON_COLUMNS = (
    "assignment_id",
    "article_id",
    "forecast_date",
    "target_ticker",
    "sector",
    "role",
    "aggregation_weight",
    "source_profile",
    "source_query_scope_complete",
    "candidate_assignment_complete",
    "headline_sha256",
    "description_sha256",
    "text_sha256",
    "description_available",
)

EVENT_LABELS = (
    "firm_operating_financial",
    "policy_corporate",
    "macro_market",
)
ABSTENTION_LABEL = "other_or_unclear"
CUE_NAMES = (
    "confirmed_action",
    "scheduled_expected",
    "rumor_unconfirmed",
    "analysis_opinion",
)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{number}: expected a JSON object")
            yield value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii", newline="\n") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp.parquet"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_aggregation_config(config: Mapping[str, Any]) -> None:
    if (
        config.get("contract_id") != "weak-news-semantics-lite-daily-v1"
        or config.get("status") != "frozen_exploratory_aggregation_contract"
        or config.get("source_arm_id") != ARM_ID
        or config.get("source_contract_id") != CONTRACT_ID
        or config.get("source_profile") != SOURCE_PROFILE
        or int(config.get("selector_k", -1)) != 16
    ):
        raise ValueError("Unexpected W17-Lite aggregation contract")
    if tuple(config.get("ordered_features", ())) != FEATURE_COLUMNS:
        raise ValueError("W17-Lite ordered feature contract changed")
    if tuple(config.get("audit_only_columns", ())) != AUDIT_COLUMNS:
        raise ValueError("W17-Lite audit column contract changed")
    families = config.get("status_cues", {}).get("families")
    if not isinstance(families, Mapping) or tuple(families) != CUE_NAMES:
        raise ValueError("W17-Lite status cue family order changed")
    for name in CUE_NAMES:
        patterns = families[name]
        if not isinstance(patterns, list) or not patterns:
            raise ValueError(f"Status cue {name} has no patterns")
        for pattern in patterns:
            re.compile(str(pattern), re.IGNORECASE)
    variants = config.get("variants")
    if not isinstance(variants, Mapping) or tuple(variants) != tuple(
        VARIANT_FILES
    ):
        raise ValueError("W17-Lite variants changed")


def normalize_cue_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text)).casefold()
    return re.sub(r"\s+", " ", value).strip()


def status_cues(
    text: str, config: Mapping[str, Any]
) -> dict[str, bool]:
    normalized = normalize_cue_text(text)
    families = config["status_cues"]["families"]
    return {
        name: any(
            re.search(str(pattern), normalized, re.IGNORECASE) is not None
            for pattern in families[name]
        )
        for name in CUE_NAMES
    }


def _manifest_artifact(
    manifest: Mapping[str, Any],
    path: Path,
    *,
    section: str = "generated_files",
) -> Mapping[str, Any]:
    records = manifest.get(section)
    if not isinstance(records, Mapping):
        raise ValueError(f"Manifest lacks {section}")
    record = records.get(path.name)
    if not isinstance(record, Mapping):
        raise ValueError(f"Manifest does not bind {path.name}")
    recorded_path = Path(str(record.get("path", "")))
    if not recorded_path.is_absolute():
        recorded_path = ROOT / recorded_path
    if recorded_path.resolve() != path.resolve():
        raise ValueError(f"Manifest path differs for {path.name}")
    if record.get("sha256") != sha256_file(path):
        raise ValueError(f"Manifest hash differs for {path.name}")
    return record


def _ids_sha256(values: Sequence[str]) -> str:
    return sha256_text("\n".join(sorted(map(str, values))))


def _ordered_lf_sha256(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(map(str, values), key=lambda item: item.encode("utf-8")):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def validate_inputs(
    *,
    predictions_path: Path,
    inference_manifest_path: Path,
    selection_manifest_path: Path,
    selected_assignments_path: Path,
    source_assignments_path: Path,
    stock_day_scope_path: Path,
    corpus_manifest_path: Path,
    design_path: Path,
    event_schema_path: Path,
    aggregation_path: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    paths = (
        predictions_path,
        inference_manifest_path,
        selection_manifest_path,
        selected_assignments_path,
        source_assignments_path,
        stock_day_scope_path,
        corpus_manifest_path,
        design_path,
        event_schema_path,
        aggregation_path,
        Path(__file__).resolve(),
    )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    inference_manifest = load_json(inference_manifest_path)
    selection_manifest = load_json(selection_manifest_path)
    corpus_manifest = load_json(corpus_manifest_path)
    config = load_json(aggregation_path)
    validate_aggregation_config(config)

    if (
        inference_manifest.get("manifest_version")
        != "flan-w17-lite-inference-manifest-v1"
        or inference_manifest.get("status") != "complete"
        or inference_manifest.get("full_corpus_complete") is not True
        or inference_manifest.get("all_selected_articles_terminal") is not True
        or int(inference_manifest.get("failure_article_count", -1)) != 0
        or int(inference_manifest.get("remaining_article_count", -1)) != 0
        or inference_manifest.get("output_sha256") != sha256_file(predictions_path)
        or int(inference_manifest.get("terminal_record_count", -1)) != 50_488
    ):
        raise ValueError("W17-Lite inference manifest is not complete and valid")
    if (
        selection_manifest.get("manifest_version")
        != "flan-w17-lite-selection-v1"
        or selection_manifest.get("status") != "complete"
        or selection_manifest.get("arm_id") != ARM_ID
        or selection_manifest.get("contract_id") != CONTRACT_ID
        or int(selection_manifest.get("selector_k", -1)) != 16
        or inference_manifest.get("selection_manifest_sha256")
        != sha256_file(selection_manifest_path)
    ):
        raise ValueError("W17-Lite selection manifest is invalid")
    selection_record = _manifest_artifact(
        selection_manifest, selected_assignments_path
    )
    if int(selection_record.get("rows", -1)) != 438_522:
        raise ValueError("Selected assignment count changed")

    if (
        corpus_manifest.get("manifest_version")
        != "semantic-corpus-manifest-v1"
        or corpus_manifest.get("status")
        != "complete_exploratory_retrospective"
    ):
        raise ValueError("Semantic corpus manifest is invalid")
    sidecar = corpus_manifest_path.with_suffix(".sha256")
    if (
        not sidecar.is_file()
        or sidecar.read_text(encoding="ascii").strip()
        != sha256_file(corpus_manifest_path)
    ):
        raise ValueError("Semantic corpus manifest sidecar is invalid")
    source_record = _manifest_artifact(corpus_manifest, source_assignments_path)
    scope_record = _manifest_artifact(corpus_manifest, stock_day_scope_path)
    if int(source_record.get("rows", -1)) != 466_902:
        raise ValueError("Source assignment count changed")
    if int(scope_record.get("rows", -1)) != 27_510:
        raise ValueError("Stock-day scope count changed")

    expected_hashes = {
        design_path: selection_manifest.get("design_sha256"),
        event_schema_path: selection_manifest.get("schema_sha256"),
        Path(inference_manifest["preflight_manifest_path"]):
            inference_manifest.get("preflight_manifest_sha256"),
    }
    for path, expected in expected_hashes.items():
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Frozen inference input changed: {path}")
    if (
        inference_manifest.get("design_sha256")
        not in {None, sha256_file(design_path)}
        or inference_manifest.get("schema_sha256")
        not in {None, sha256_file(event_schema_path)}
    ):
        raise ValueError("Inference and selection design/schema hashes differ")

    inputs = {
        "predictions": predictions_path,
        "inference_manifest": inference_manifest_path,
        "selection_manifest": selection_manifest_path,
        "selected_assignments": selected_assignments_path,
        "source_assignments": source_assignments_path,
        "stock_day_scope": stock_day_scope_path,
        "semantic_corpus_manifest": corpus_manifest_path,
        "design": design_path,
        "event_schema": event_schema_path,
        "status_cue_rules": aggregation_path,
        "builder": Path(__file__).resolve(),
    }
    records = {
        name: {
            "path": display_path(path),
            "sha256": sha256_file(path),
        }
        for name, path in inputs.items()
    }
    records["selected_assignments"]["rows"] = int(selection_record["rows"])
    records["source_assignments"]["rows"] = int(source_record["rows"])
    records["stock_day_scope"]["rows"] = int(scope_record["rows"])
    return config, records


def load_predictions(
    path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for raw in iter_jsonl(path):
        inference.validate_terminal_record(raw)
        if raw.get("terminal_state") != "complete":
            raise ValueError("Daily construction requires every prediction complete")
        event = raw["event_group"]
        if event.get("input_truncated") is not False:
            raise ValueError("Truncated W17-Lite prediction is inadmissible")
        canonical = str(event["canonical_prediction"])
        reverse = str(event["reversed_prediction"])
        if canonical not in (*EVENT_LABELS, ABSTENTION_LABEL):
            raise ValueError("Unknown canonical event label")
        if reverse not in (*EVENT_LABELS, ABSTENTION_LABEL):
            raise ValueError("Unknown reversed event label")
        agreement = canonical == reverse
        reason = str(event["acceptance_reason"])
        accepted_label = event.get("accepted_label")
        expected_accepted = agreement and canonical != ABSTENTION_LABEL
        if bool(event["accepted"]) != expected_accepted:
            raise ValueError("Event acceptance does not replay")
        if accepted_label != (canonical if expected_accepted else None):
            raise ValueError("Accepted event label does not replay")
        expected_reason = (
            "accepted_order_agreement"
            if expected_accepted
            else (
                "agreed_other_or_unclear"
                if agreement
                else "order_disagreement"
            )
        )
        if reason != expected_reason:
            raise ValueError("Event acceptance reason does not replay")
        reasons[reason] += 1
        records.append(
            {
                "article_id": str(raw["article_id"]),
                "prediction_model_text_sha256": str(
                    raw["model_text_sha256"]
                ),
                "event_label": accepted_label,
                "event_order_disagreement": reason == "order_disagreement",
                "event_other_or_unclear": reason
                == "agreed_other_or_unclear",
            }
        )
    frame = pd.DataFrame(records)
    if frame.empty or frame["article_id"].duplicated().any():
        raise ValueError("Prediction ledger is empty or has duplicate articles")
    agreement_count = (
        len(frame) - int(reasons["order_disagreement"])
    )
    return frame, {
        "article_count": len(frame),
        "accepted_article_count": int(reasons["accepted_order_agreement"]),
        "agreed_other_or_unclear_count": int(
            reasons["agreed_other_or_unclear"]
        ),
        "order_disagreement_count": int(reasons["order_disagreement"]),
        "choice_order_agreement_rate": agreement_count / len(frame),
        "acceptance_rate": int(reasons["accepted_order_agreement"]) / len(frame),
    }


def load_stock_day_scope(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for value in iter_jsonl(path):
        rows.append(
            {
                "forecast_date": str(value["forecast_date"]),
                "sector": str(value["sector"]),
                "stock": str(value["stock"]),
                "benchmark": str(value["benchmark"]),
                "expected_assignment_count": int(
                    value["expected_assignment_count"]
                ),
                "expected_assignment_ids_sha256": str(
                    value["expected_assignment_ids_sha256"]
                ),
                "source_profile": str(value["source_profile"]),
                "source_query_scope_complete": bool(
                    value["source_query_scope_complete"]
                ),
                "candidate_assignment_complete": bool(
                    value["candidate_assignment_complete"]
                ),
            }
        )
    frame = pd.DataFrame(rows)
    if (
        len(frame) != 27_510
        or frame.duplicated(list(KEY_COLUMNS)).any()
        or set(frame["source_profile"]) != {SOURCE_PROFILE}
        or not frame["source_query_scope_complete"].all()
        or not frame["candidate_assignment_complete"].all()
    ):
        raise ValueError("Authoritative stock-day scope is incomplete")
    return frame.sort_values(list(KEY_COLUMNS), kind="mergesort").reset_index(
        drop=True
    )


def load_assignment_frames(
    *,
    source_path: Path,
    selected_path: Path,
    selection_manifest: Mapping[str, Any],
    scopes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_parquet(source_path, columns=list(SOURCE_ASSIGNMENT_COLUMNS))
    selected_file = pd.read_parquet(
        selected_path, columns=list(SELECTED_COMPARISON_COLUMNS)
    )
    if (
        len(source) != 466_902
        or source["assignment_id"].duplicated().any()
        or len(selected_file) != 438_522
        or selected_file["assignment_id"].duplicated().any()
    ):
        raise ValueError("Assignment ledgers have invalid counts or duplicate IDs")
    for frame, label in ((source, "source"), (selected_file, "selected")):
        if (
            set(frame["source_profile"]) != {SOURCE_PROFILE}
            or not frame["source_query_scope_complete"].all()
            or not frame["candidate_assignment_complete"].all()
            or not np.isfinite(frame["aggregation_weight"]).all()
            or not frame["aggregation_weight"].gt(0).all()
            or not frame["role"].isin(["I", "P", "C"]).all()
        ):
            raise ValueError(f"{label} assignment ledger is incomplete")
    selected_ids = set(selected_file["assignment_id"])
    selected = source[source["assignment_id"].isin(selected_ids)].copy()
    if len(selected) != len(selected_file):
        raise ValueError("Selected assignment IDs are not a source subset")
    left = selected_file.sort_values("assignment_id").reset_index(drop=True)
    right = selected[list(SELECTED_COMPARISON_COLUMNS)].sort_values(
        "assignment_id"
    ).reset_index(drop=True)
    for column in SELECTED_COMPARISON_COLUMNS:
        if column == "aggregation_weight":
            if not np.array_equal(
                left[column].to_numpy(dtype=float),
                right[column].to_numpy(dtype=float),
            ):
                raise ValueError("Selected assignment weights differ")
        elif not left[column].equals(right[column]):
            raise ValueError(f"Selected assignment field differs: {column}")

    audit = selection_manifest["audit"]
    if (
        _ordered_lf_sha256(selected["assignment_id"].tolist())
        != audit["ordered_selected_assignment_ids_sha256"]
        or _ordered_lf_sha256(
            selected["article_id"].drop_duplicates().tolist()
        )
        != audit["ordered_selected_article_ids_sha256"]
    ):
        raise ValueError("Selected assignment/article ID hash changed")

    source = source.rename(columns={"target_ticker": "stock"})
    selected = selected.rename(columns={"target_ticker": "stock"})
    grouped_ids = (
        source.groupby(list(KEY_COLUMNS), sort=False)["assignment_id"]
        .agg(lambda values: _ids_sha256(values.tolist()))
        .rename("observed_assignment_ids_sha256")
    )
    grouped_count = (
        source.groupby(list(KEY_COLUMNS), sort=False)
        .size()
        .rename("observed_assignment_count")
    )
    observed = pd.concat([grouped_ids, grouped_count], axis=1).reset_index()
    checked = scopes.merge(
        observed, on=list(KEY_COLUMNS), how="left", validate="one_to_one"
    )
    checked["observed_assignment_count"] = (
        checked["observed_assignment_count"].fillna(0).astype(int)
    )
    empty_hash = _ids_sha256([])
    checked["observed_assignment_ids_sha256"] = checked[
        "observed_assignment_ids_sha256"
    ].fillna(empty_hash)
    if not checked["expected_assignment_count"].equals(
        checked["observed_assignment_count"]
    ) or not checked["expected_assignment_ids_sha256"].equals(
        checked["observed_assignment_ids_sha256"]
    ):
        raise ValueError("Source assignments differ from stock-day scope")
    return source, selected


def enrich_selected(
    selected: pd.DataFrame,
    predictions: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    if set(selected["article_id"]) != set(predictions["article_id"]):
        raise ValueError("Selected article universe differs from predictions")
    article_text = (
        selected.sort_values("assignment_id", kind="mergesort")
        .drop_duplicates("article_id")
        .loc[
            :,
            [
                "article_id",
                "model_text",
                "text_sha256",
                "description",
                "description_available",
            ],
        ]
    )
    if article_text["article_id"].duplicated().any():
        raise AssertionError("Article table still has duplicates")
    text_consistency = selected.groupby("article_id", sort=False).agg(
        text_hashes=("text_sha256", "nunique"),
        descriptions=("description", "nunique"),
        availability=("description_available", "nunique"),
    )
    if (
        text_consistency["text_hashes"].max() != 1
        or text_consistency["descriptions"].max() != 1
        or text_consistency["availability"].max() != 1
    ):
        raise ValueError("Selected article text differs across assignments")
    cue_rows: list[dict[str, Any]] = []
    for record in article_text.itertuples(index=False):
        flags = status_cues(record.model_text, config)
        cue_rows.append(
            {
                "article_id": record.article_id,
                **{f"cue_{name}": bool(flags[name]) for name in CUE_NAMES},
                "cue_conflict": sum(flags.values()) >= 2,
                "cue_all_zero": not any(flags.values()),
                "headline_only": int(record.description_available) == 0,
                "description_sub150": (
                    int(record.description_available) == 1
                    and 0 < len(str(record.description)) < 150
                ),
                "description_ge150": (
                    int(record.description_available) == 1
                    and len(str(record.description)) >= 150
                ),
            }
        )
    article = (
        article_text.merge(
            predictions, on="article_id", how="inner", validate="one_to_one"
        )
        .merge(
            pd.DataFrame(cue_rows),
            on="article_id",
            how="inner",
            validate="one_to_one",
        )
    )
    if not article["prediction_model_text_sha256"].equals(
        article["text_sha256"]
    ):
        raise ValueError("Prediction text hash differs from selected article")
    output = selected.merge(
        article.drop(columns=["model_text", "text_sha256", "description",
                              "description_available"]),
        on="article_id",
        how="inner",
        validate="many_to_one",
    )
    return output


def permutation_mapping(
    selected: pd.DataFrame, *, seed: str
) -> tuple[dict[tuple[str, str, str], str], str]:
    triples = (
        selected.loc[
            selected["event_label"].notna(),
            ["forecast_date", "sector", "article_id"],
        ]
        .drop_duplicates()
        .sort_values(
            ["forecast_date", "sector", "article_id"], kind="mergesort"
        )
    )
    mapping: dict[tuple[str, str, str], str] = {}
    hash_lines: list[str] = []
    for (forecast_date, sector), group in triples.groupby(
        ["forecast_date", "sector"], sort=True
    ):
        ids = sorted(group["article_id"].tolist(), key=lambda value: value.encode("utf-8"))
        if len(ids) == 1:
            donors = ids
        else:
            digest = int(
                sha256_text(f"{seed}\n{forecast_date}\n{sector}"), 16
            )
            offset = 1 + digest % (len(ids) - 1)
            donors = ids[offset:] + ids[:offset]
        for recipient, donor in zip(ids, donors, strict=True):
            mapping[(str(forecast_date), str(sector), recipient)] = donor
            hash_lines.append(
                canonical_json(
                    {
                        "forecast_date": str(forecast_date),
                        "sector": str(sector),
                        "recipient_article_id": recipient,
                        "donor_article_id": donor,
                    }
                )
            )
    return mapping, sha256_text("\n".join(hash_lines))


def apply_event_permutation(
    selected: pd.DataFrame, *, seed: str
) -> tuple[pd.DataFrame, str]:
    mapping, mapping_hash = permutation_mapping(selected, seed=seed)
    accepted_labels = (
        selected[
            [
                "forecast_date",
                "sector",
                "article_id",
                "event_label",
            ]
        ]
        .drop_duplicates(["forecast_date", "sector", "article_id"])
        .loc[lambda frame: frame["event_label"].notna()]
        .set_index(["forecast_date", "sector", "article_id"])
    )
    donors = pd.DataFrame(
        [
            {
                "forecast_date": key[0],
                "sector": key[1],
                "article_id": key[2],
                "donor_article_id": donor,
            }
            for key, donor in mapping.items()
        ]
    )
    donor_state = accepted_labels.reset_index().rename(
        columns={
            "article_id": "donor_article_id",
            "event_label": "permuted_event_label",
        }
    )
    donors = donors.merge(
        donor_state,
        on=["forecast_date", "sector", "donor_article_id"],
        how="inner",
        validate="one_to_one",
    )
    output = selected.merge(
        donors.drop(columns=["donor_article_id"]),
        on=["forecast_date", "sector", "article_id"],
        how="left",
        validate="many_to_one",
    )
    accepted = output["event_label"].notna()
    if output.loc[accepted, "permuted_event_label"].isna().any():
        raise AssertionError("Accepted event label lacks a permutation donor")
    if output.loc[~accepted, "permuted_event_label"].notna().any():
        raise AssertionError("Abstained event acquired a permuted label")
    output.loc[accepted, "event_label"] = output.loc[
        accepted, "permuted_event_label"
    ]
    return output.drop(columns=["permuted_event_label"]), mapping_hash


def _weighted_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    weight = output["aggregation_weight"].astype(float)
    indicators: dict[str, pd.Series] = {
        **{
            f"event_{label}": output["event_label"].eq(label)
            for label in EVENT_LABELS
        },
        "route_I": output["role"].eq("I"),
        "route_P": output["role"].eq("P"),
        "route_C": output["role"].eq("C"),
        **{name: output[f"cue_{name}"].astype(bool) for name in CUE_NAMES},
        "cue_conflict": output["cue_conflict"].astype(bool),
        "cue_all_zero": output["cue_all_zero"].astype(bool),
        "headline_only": output["headline_only"].astype(bool),
        "description_sub150": output["description_sub150"].astype(bool),
        "event_disagreement": output["event_order_disagreement"].astype(bool),
        "event_other": output["event_other_or_unclear"].astype(bool),
    }
    indicators["event_accepted"] = output["event_label"].notna()
    for name, indicator in indicators.items():
        output[f"weighted_{name}"] = weight * indicator.astype(float)
    output["weight_squared"] = weight**2
    return output


def aggregate_variant(
    *,
    source: pd.DataFrame,
    selected: pd.DataFrame,
    scopes: pd.DataFrame,
    variant_id: str,
    enforce_expected_profile: bool = True,
) -> pd.DataFrame:
    if variant_id not in VARIANT_FILES:
        raise ValueError(f"Unknown variant {variant_id}")
    selected_weighted = _weighted_columns(selected)
    keys = list(KEY_COLUMNS)
    source_grouped = (
        source.groupby(keys, sort=False)
        .agg(
            wlite_full_assignment_count=("assignment_id", "size"),
            wlite_full_assignment_weight=("aggregation_weight", "sum"),
        )
        .reset_index()
    )
    weighted_names = [
        column for column in selected_weighted if column.startswith("weighted_")
    ]
    selected_grouped = (
        selected_weighted.groupby(keys, sort=False)
        .agg(
            wlite_selected_assignment_count=("assignment_id", "size"),
            wlite_selected_assignment_weight=("aggregation_weight", "sum"),
            selected_weight_square_sum=("weight_squared", "sum"),
            **{name: (name, "sum") for name in weighted_names},
        )
        .reset_index()
    )
    frame = (
        scopes[list(KEY_COLUMNS)]
        .merge(source_grouped, on=keys, how="left", validate="one_to_one")
        .merge(selected_grouped, on=keys, how="left", validate="one_to_one")
    )
    for column in (
        "wlite_full_assignment_count",
        "wlite_selected_assignment_count",
    ):
        frame[column] = frame[column].fillna(0).astype(int)
    numeric_fill = [
        "wlite_full_assignment_weight",
        "wlite_selected_assignment_weight",
        "selected_weight_square_sum",
        *weighted_names,
    ]
    frame[numeric_fill] = frame[numeric_fill].fillna(0.0)
    selected_weight = frame["wlite_selected_assignment_weight"]
    full_weight = frame["wlite_full_assignment_weight"]
    positive = selected_weight > 0
    full_positive = full_weight > 0
    if (positive & ~full_positive).any():
        raise AssertionError("Selected weight exists without full candidate weight")

    for column in (*FEATURE_COLUMNS, *AUDIT_COLUMNS):
        if column not in frame:
            frame[column] = np.nan
    frame["wlite_observed_no_selected_article"] = (~positive).astype(float)

    event_feature = {
        "firm_operating_financial":
            "wlite_event_share_firm_operating_financial",
        "policy_corporate": "wlite_event_share_policy_corporate",
        "macro_market": "wlite_event_share_macro_market",
    }
    for label, feature in event_feature.items():
        frame.loc[positive, feature] = (
            frame.loc[positive, f"weighted_event_{label}"]
            / selected_weight[positive]
        )
    direct_feature = {
        "I": "wlite_route_share_target_idiosyncratic",
        "P": "wlite_route_share_peer_idiosyncratic",
        "C": "wlite_route_share_common",
    }
    for role, feature in direct_feature.items():
        frame.loc[positive, feature] = (
            frame.loc[positive, f"weighted_route_{role}"]
            / selected_weight[positive]
        )
    for cue in CUE_NAMES:
        frame.loc[
            positive, f"wlite_rule_status_cue_share_{cue}"
        ] = (
            frame.loc[positive, f"weighted_{cue}"]
            / selected_weight[positive]
        )
    simple_weighted = {
        "cue_conflict": "wlite_rule_status_cue_conflict_weight_share",
        "cue_all_zero": "wlite_rule_status_cue_all_zero_weight_share",
        "headline_only": "wlite_selected_headline_only_weight_share",
        "description_sub150": "wlite_selected_sub150_weight_share",
        "event_disagreement":
            "wlite_event_order_disagreement_weight_share",
        "event_other": "wlite_event_other_or_unclear_weight_share",
        "event_accepted": "wlite_event_accepted_coverage",
    }
    for source_name, feature in simple_weighted.items():
        frame.loc[positive, feature] = (
            frame.loc[positive, f"weighted_{source_name}"]
            / selected_weight[positive]
        )
    if positive.any():
        frame.loc[positive, "wlite_selection_weight_coverage"] = (
            selected_weight[positive] / full_weight[positive]
        )
    if positive.any():
        frame.loc[positive, "wlite_selected_weight_hhi"] = (
            frame.loc[positive, "selected_weight_square_sum"]
            / selected_weight[positive] ** 2
        )
    event_shares = frame[list(event_feature.values())].fillna(0.0)
    accepted_mass = event_shares.sum(axis=1)
    frame.loc[positive, "wlite_event_entropy_accepted"] = 0.0
    entropy_rows = positive & accepted_mass.gt(0)
    if entropy_rows.any():
        probabilities = event_shares.loc[entropy_rows].div(
            accepted_mass[entropy_rows], axis=0
        )
        values = probabilities.to_numpy(dtype=float)
        terms = np.zeros_like(values)
        nonzero = values > 0
        terms[nonzero] = values[nonzero] * np.log(values[nonzero])
        frame.loc[entropy_rows, "wlite_event_entropy_accepted"] = (
            -terms.sum(axis=1) / math.log(3)
        )
    frame["wlite_invalid_or_truncated_selected_article_count"] = 0
    decomposition = (
        event_shares.sum(axis=1)
        + frame["wlite_event_other_or_unclear_weight_share"].fillna(0)
        + frame["wlite_event_order_disagreement_weight_share"].fillna(0)
    )
    if positive.any() and not np.allclose(
        decomposition[positive], 1.0, atol=1e-12, rtol=0
    ):
        raise AssertionError("Event decomposition failed before serialization")
    frame.loc[positive, "wlite_event_decomposition_residual"] = 0.0

    gate = False
    frame["source_profile"] = SOURCE_PROFILE
    frame["point_in_time_version_safe"] = False
    frame["primary_training_eligible"] = False
    frame["confirmatory_eligible"] = False
    frame["exploratory_construction_eligible"] = True
    frame["semantic_quality_gate_passed"] = gate
    frame["exploratory_quality_gate_override"] = True
    frame["variant_id"] = variant_id
    columns = [
        *KEY_COLUMNS,
        *SOURCE_COLUMNS,
        *AUDIT_COLUMNS,
        *FEATURE_COLUMNS,
    ]
    output = frame[columns].sort_values(
        list(KEY_COLUMNS), kind="mergesort"
    ).reset_index(drop=True)
    validate_daily_panel(
        output,
        variant_id=variant_id,
        enforce_expected_profile=enforce_expected_profile,
    )
    return output


def validate_daily_panel(
    frame: pd.DataFrame,
    *,
    variant_id: str,
    enforce_expected_profile: bool = True,
) -> None:
    expected_columns = (
        *KEY_COLUMNS,
        *SOURCE_COLUMNS,
        *AUDIT_COLUMNS,
        *FEATURE_COLUMNS,
    )
    if tuple(frame.columns) != expected_columns:
        raise ValueError("Daily W17-Lite column contract differs")
    profile_invalid = (
        len(frame) != 27_510
        or frame["forecast_date"].nunique() != 917
        or frame["stock"].nunique() != 30
    )
    if (
        (enforce_expected_profile and profile_invalid)
        or frame.empty
        or frame.duplicated(list(KEY_COLUMNS)).any()
        or set(frame["variant_id"]) != {variant_id}
    ):
        raise ValueError("Daily W17-Lite panel coverage differs")
    if (
        set(frame["source_profile"]) != {SOURCE_PROFILE}
        or frame["point_in_time_version_safe"].any()
        or frame["primary_training_eligible"].any()
        or frame["confirmatory_eligible"].any()
        or not frame["exploratory_construction_eligible"].all()
        or frame["semantic_quality_gate_passed"].any()
        or not frame["exploratory_quality_gate_override"].all()
    ):
        raise ValueError("Daily W17-Lite claim flags differ")
    numeric = frame[[*AUDIT_COLUMNS, *FEATURE_COLUMNS]].to_numpy(dtype=float)
    if np.isinf(numeric).any():
        raise ValueError("Daily W17-Lite panel contains infinity")
    no_selected = frame["wlite_observed_no_selected_article"].eq(1)
    if not frame["wlite_observed_no_selected_article"].isin([0.0, 1.0]).all():
        raise ValueError("No-selected indicator is not binary")
    if not frame.loc[no_selected, "wlite_selected_assignment_count"].eq(0).all():
        raise ValueError("No-selected rows retain selected assignments")
    bounded = [
        *FEATURE_COLUMNS[:14],
        "wlite_event_entropy_accepted",
        "wlite_selected_weight_hhi",
        "wlite_route_share_common",
        "wlite_event_accepted_coverage",
        "wlite_event_other_or_unclear_weight_share",
        "wlite_rule_status_cue_all_zero_weight_share",
    ]
    observed = frame[bounded].stack().dropna()
    if not observed.between(-1e-12, 1 + 1e-12).all():
        raise ValueError("Bounded W17-Lite feature exceeds [0,1]")
    residual = frame.loc[~no_selected, "wlite_event_decomposition_residual"]
    if not np.allclose(residual, 0.0, atol=1e-12, rtol=0):
        raise ValueError("Event decomposition identity failed")
    if (
        frame["wlite_invalid_or_truncated_selected_article_count"] != 0
    ).any():
        raise ValueError("Invalid/truncated selected articles entered panel")


def build_manifest(
    *,
    variant_id: str,
    output_path: Path,
    inputs: Mapping[str, Mapping[str, Any]],
    frame: pd.DataFrame,
    prediction_audit: Mapping[str, Any],
    config: Mapping[str, Any],
    permutation_mapping_sha256: str | None,
) -> dict[str, Any]:
    threshold = float(
        config["semantic_quality_gate"][
            "minimum_article_choice_order_agreement"
        ]
    )
    observed_agreement = float(
        prediction_audit["choice_order_agreement_rate"]
    )
    generated = {
        output_path.name: {
            "path": display_path(output_path),
            "rows": len(frame),
            "sha256": sha256_file(output_path),
        }
    }
    return {
        "manifest_version": MANIFEST_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "status": MANIFEST_STATUS,
        "generated_at_utc": utc_now(),
        "variant_id": variant_id,
        "arm_id": ARM_ID,
        "contract_id": CONTRACT_ID,
        "source_profile": SOURCE_PROFILE,
        "selector_k": 16,
        "ordered_feature_lists": {"wlite_17": list(FEATURE_COLUMNS)},
        "feature_list_sha256": sha256_text(canonical_json(list(FEATURE_COLUMNS))),
        "audit_column_list": list(AUDIT_COLUMNS),
        "inputs": dict(inputs),
        "generated_files": generated,
        "coverage": {
            "rows": len(frame),
            "dates": int(frame["forecast_date"].nunique()),
            "stocks": int(frame["stock"].nunique()),
            "sectors": int(frame["sector"].nunique()),
            "no_selected_article_rows": int(
                frame["wlite_observed_no_selected_article"].sum()
            ),
            "selected_assignment_rows": int(
                frame["wlite_selected_assignment_count"].sum()
            ),
            "source_assignment_rows": int(
                frame["wlite_full_assignment_count"].sum()
            ),
            "selected_weight_fraction": (
                float(frame["wlite_selected_assignment_weight"].sum())
                / float(frame["wlite_full_assignment_weight"].sum())
            ),
        },
        "prediction_audit": dict(prediction_audit),
        "semantic_quality_gate": {
            "metric": "article_choice_order_agreement_rate",
            "observed": observed_agreement,
            "required_minimum": threshold,
            "passed": observed_agreement >= threshold,
            "exploratory_override": True,
        },
        "permutation_mapping_sha256": permutation_mapping_sha256,
        "claim_flags": {
            "point_in_time_version_safe": False,
            "primary_training_eligible": False,
            "confirmatory_eligible": False,
            "exploratory_construction_eligible": True,
            "semantic_quality_gate_passed": observed_agreement >= threshold,
            "exploratory_quality_gate_override": True,
        },
    }


def build_all(
    *,
    output_root: Path = DEFAULT_WORK_ROOT,
    predictions_path: Path = DEFAULT_PREDICTIONS,
    inference_manifest_path: Path = DEFAULT_INFERENCE_MANIFEST,
    selection_manifest_path: Path = DEFAULT_SELECTION_MANIFEST,
    selected_assignments_path: Path = DEFAULT_SELECTED_ASSIGNMENTS,
    source_assignments_path: Path = DEFAULT_SOURCE_ASSIGNMENTS,
    stock_day_scope_path: Path = DEFAULT_STOCK_DAY_SCOPE,
    corpus_manifest_path: Path = DEFAULT_CORPUS_MANIFEST,
    design_path: Path = DEFAULT_DESIGN,
    event_schema_path: Path = DEFAULT_EVENT_SCHEMA,
    aggregation_path: Path = DEFAULT_AGGREGATION,
    overwrite: bool = False,
) -> dict[str, dict[str, Any]]:
    output_paths = {
        variant: output_root / filename
        for variant, filename in VARIANT_FILES.items()
    }
    manifest_paths = {
        variant: output_root / filename
        for variant, filename in MANIFEST_FILES.items()
    }
    all_outputs = [
        *output_paths.values(),
        *manifest_paths.values(),
        *(
            path.with_suffix(".sha256")
            for path in manifest_paths.values()
        ),
    ]
    if not overwrite and any(path.exists() for path in all_outputs):
        existing = [str(path) for path in all_outputs if path.exists()]
        raise FileExistsError(f"W17-Lite daily outputs exist: {existing}")

    config, inputs = validate_inputs(
        predictions_path=predictions_path,
        inference_manifest_path=inference_manifest_path,
        selection_manifest_path=selection_manifest_path,
        selected_assignments_path=selected_assignments_path,
        source_assignments_path=source_assignments_path,
        stock_day_scope_path=stock_day_scope_path,
        corpus_manifest_path=corpus_manifest_path,
        design_path=design_path,
        event_schema_path=event_schema_path,
        aggregation_path=aggregation_path,
    )
    immutable_hashes = {
        record["path"]: record["sha256"] for record in inputs.values()
    }
    predictions, prediction_audit = load_predictions(predictions_path)
    scopes = load_stock_day_scope(stock_day_scope_path)
    selection_manifest = load_json(selection_manifest_path)
    source, selected = load_assignment_frames(
        source_path=source_assignments_path,
        selected_path=selected_assignments_path,
        selection_manifest=selection_manifest,
        scopes=scopes,
    )
    selected = enrich_selected(selected, predictions, config)

    frames: dict[str, pd.DataFrame] = {}
    frames["canonical"] = aggregate_variant(
        source=source,
        selected=selected,
        scopes=scopes,
        variant_id="canonical",
    )
    long_selected = selected.loc[selected["description_ge150"]].copy()
    frames["long_description"] = aggregate_variant(
        source=source,
        selected=long_selected,
        scopes=scopes,
        variant_id="long_description",
    )
    permuted, mapping_hash = apply_event_permutation(
        selected,
        seed=str(config["variants"]["permuted"]["seed"]),
    )
    frames["permuted"] = aggregate_variant(
        source=source,
        selected=permuted,
        scopes=scopes,
        variant_id="permuted",
    )

    for name, expected in immutable_hashes.items():
        path = Path(name)
        if not path.is_absolute():
            path = ROOT / path
        if sha256_file(path) != expected:
            raise ValueError(f"Immutable W17-Lite input changed: {path}")

    manifests: dict[str, dict[str, Any]] = {}
    for variant, frame in frames.items():
        output_path = output_paths[variant]
        manifest_path = manifest_paths[variant]
        _atomic_parquet(output_path, frame)
        manifest = build_manifest(
            variant_id=variant,
            output_path=output_path,
            inputs=inputs,
            frame=frame,
            prediction_audit=prediction_audit,
            config=config,
            permutation_mapping_sha256=(
                mapping_hash if variant == "permuted" else None
            ),
        )
        _atomic_json(manifest_path, manifest)
        _atomic_text(
            manifest_path.with_suffix(".sha256"),
            sha256_file(manifest_path) + "\n",
        )
        manifests[variant] = manifest
    return manifests


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--output-root", type=Path, default=DEFAULT_WORK_ROOT)
    value.add_argument("--overwrite", action="store_true")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    manifests = build_all(
        output_root=args.output_root,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                variant: {
                    "rows": manifest["coverage"]["rows"],
                    "selected_weight_fraction": manifest["coverage"][
                        "selected_weight_fraction"
                    ],
                    "semantic_quality_gate": manifest[
                        "semantic_quality_gate"
                    ],
                }
                for variant, manifest in manifests.items()
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
