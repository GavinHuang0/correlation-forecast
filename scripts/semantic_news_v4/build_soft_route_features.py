#!/usr/bin/env python
"""Build v4 SoftRoute features from the completed FLAN W17-Lite ledger.

This is a cached-inference transformation.  It never imports or runs an LLM,
and it leaves the completed v3 feature and training artifacts untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PIPELINE_VERSION = "flan-w17-soft-route-v1.0.0"
DAILY_MANIFEST_VERSION = "flan-w17-soft-route-daily-manifest-v1"
JOIN_MANIFEST_VERSION = "q-plus-d2-soft-route-join-manifest-v1"
STATUS = "complete_exploratory_non_version_safe"
SOURCE_PROFILE = "ordinary_massive_retrospective"
ARM_ID = "WL17__flan_t5_xl"
SOURCE_CONTRACT_ID = "weak-news-semantics-lite-v1"
CONTRACT_ID = "flan-w17-soft-route-v1"
EXPECTED_ROWS = 27_510
EXPECTED_DATES = 917
EXPECTED_STOCKS = 30
KEY_COLUMNS = ("forecast_date", "sector", "stock", "benchmark")
PERMUTATION_SEED = "flan-w17-soft-route-probability-permutation-v1"

DEFAULT_SOURCE_ROOT = (
    ROOT / "data" / "features" / "news_semantic" / "massive_v2"
)
DEFAULT_V3_ROOT = (
    ROOT
    / "data"
    / "features"
    / "news_semantic"
    / "massive_v3"
    / "flan_w17_lite_k16"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "data"
    / "features"
    / "news_semantic"
    / "massive_v4"
    / "flan_w17_soft_route"
)
DEFAULT_JOIN_ROOT = ROOT / "data" / "features" / "q_plus_d" / "massive_v4"
DEFAULT_PREDICTIONS = DEFAULT_V3_ROOT / "predictions.jsonl"
DEFAULT_INFERENCE_MANIFEST = DEFAULT_V3_ROOT / "predictions.jsonl.manifest.json"
DEFAULT_SELECTION_MANIFEST = DEFAULT_V3_ROOT / "selection_manifest.json"
DEFAULT_SELECTED_ASSIGNMENTS = DEFAULT_V3_ROOT / "selected_assignments.parquet"
DEFAULT_SOURCE_ASSIGNMENTS = DEFAULT_SOURCE_ROOT / "article_target_assignments.parquet"
DEFAULT_SCOPE = DEFAULT_SOURCE_ROOT / "stock_day_scope.parquet"
DEFAULT_CORPUS_MANIFEST = DEFAULT_SOURCE_ROOT / "manifest.json"
DEFAULT_CONTRACT = ROOT / "config" / "flan_w17_soft_route_feature_contract_v1.json"
DEFAULT_BASE = (
    ROOT
    / "data"
    / "features"
    / "q_plus_d"
    / "massive_v3"
    / "modeling_panel_q_d2_wlite.parquet"
)
DEFAULT_BASE_MANIFEST = DEFAULT_BASE.parent / "manifest.json"

VARIANT_FILES = {
    "canonical": "daily_soft_route.parquet",
    "stale20": "daily_soft_route_stale20.parquet",
    "wrong_stock": "daily_soft_route_wrong_stock.parquet",
    "permuted": "daily_soft_route_permuted.parquet",
}
JOINED_FILES = {
    "canonical": "modeling_panel_q_d2_soft_route.parquet",
    "stale20": "modeling_panel_q_d2_soft_route_stale20.parquet",
    "wrong_stock": "modeling_panel_q_d2_soft_route_wrong_stock.parquet",
    "permuted": "modeling_panel_q_d2_soft_route_permuted.parquet",
}

ROLES = ("I", "P", "C")
ROLE_NAMES = {
    "I": "target_idiosyncratic",
    "P": "peer_idiosyncratic",
    "C": "common",
}
PREDICTIVE_LABELS = (
    "firm_operating_financial",
    "policy_corporate",
    "macro_market",
)
ALL_LABELS = (*PREDICTIVE_LABELS, "other_or_unclear")


def _mass_name(role: str, label: str) -> str:
    return f"lsoft_{ROLE_NAMES[role]}_event_{label}_joint_mass"


CURRENT9 = tuple(
    _mass_name(role, label) for role in ROLES for label in PREDICTIVE_LABELS
)
INNOVATION9 = tuple(
    f"{name}_innovation_ewma63_hl21" for name in CURRENT9
)
NO_SELECTED = "lsoft_observed_no_selected_article"
SOFT_ROUTE19 = (*CURRENT9, *INNOVATION9, NO_SELECTED)
COUPLING6 = tuple(
    name
    for label in PREDICTIVE_LABELS
    for name in (
        f"lsoft_coupling_{label}_c_minus_i_minus_p",
        f"lsoft_coupling_{label}_innovation_c_minus_i_minus_p",
    )
)
QUALITY4 = (
    "lsoft_score_order_js_divergence_normalized",
    "lsoft_score_consensus_entropy_normalized",
    "lsoft_score_other_or_unclear_mass",
    "lsoft_k16_selection_weight_coverage",
)
LEDGER_AUDIT_COLUMNS = (
    "lsoft_full_assignment_count",
    "lsoft_selected_assignment_count",
    "lsoft_full_assignment_weight",
    "lsoft_selected_assignment_weight",
    "lsoft_max_article_probability_sum_residual",
    "lsoft_daily_mass_decomposition_residual",
)
SOURCE_COLUMNS = (
    "source_profile",
    "point_in_time_version_safe",
    "primary_training_eligible",
    "confirmatory_eligible",
    "exploratory_construction_eligible",
    "variant_id",
)
DAILY_COLUMNS = (
    *KEY_COLUMNS,
    *SOURCE_COLUMNS,
    *LEDGER_AUDIT_COLUMNS,
    *QUALITY4,
    *SOFT_ROUTE19,
    *COUPLING6,
)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    with path.open("r", encoding="utf-8") as handle:
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
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
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


def _manifest_record(
    manifest: Mapping[str, Any], path: Path, *, section: str = "generated_files"
) -> Mapping[str, Any]:
    records = manifest.get(section)
    if not isinstance(records, Mapping):
        raise ValueError(f"Manifest has no {section}")
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


def _validate_sidecar(path: Path) -> None:
    sidecar = path.with_suffix(".sha256")
    if (
        not sidecar.is_file()
        or sidecar.read_text(encoding="ascii").strip().split()[0]
        != sha256_file(path)
    ):
        raise ValueError(f"Manifest sidecar differs: {sidecar}")


def validate_contract(path: Path) -> dict[str, Any]:
    value = load_json(path)
    if (
        value.get("contract_id") != CONTRACT_ID
        or value.get("status") != "frozen_exploratory_feature_contract"
        or value.get("source_arm_id") != ARM_ID
        or value.get("source_contract_id") != SOURCE_CONTRACT_ID
        or value.get("source_profile") != SOURCE_PROFILE
        or tuple(value.get("soft_route_19", ())) != SOFT_ROUTE19
        or tuple(value.get("coupling_6_residual_only", ())) != COUPLING6
        or tuple(value.get("score_quality_4_audit_or_control", ())) != QUALITY4
    ):
        raise ValueError("SoftRoute feature contract changed")
    probability = value.get("probability_pooling", {})
    if tuple(probability.get("labels", ())) != ALL_LABELS:
        raise ValueError("SoftRoute label order changed")
    aggregation = value.get("aggregation", {})
    if (
        tuple(aggregation.get("roles", ())) != ROLES
        or int(aggregation.get("innovation_warmup", -1)) != 63
        or int(aggregation.get("innovation_half_life_sessions", -1)) != 21
    ):
        raise ValueError("SoftRoute aggregation contract changed")
    controls = value.get("controls", {})
    if controls.get("permutation_seed") != PERMUTATION_SEED:
        raise ValueError("SoftRoute probability-permutation seed changed")
    return value


def softmax_scores(scores: Mapping[str, Any]) -> np.ndarray:
    if set(scores) != set(ALL_LABELS):
        raise ValueError("Candidate score labels differ")
    values = np.asarray([float(scores[label]) for label in ALL_LABELS])
    if not np.isfinite(values).all():
        raise ValueError("Candidate score contains a non-finite value")
    shifted = values - values.max()
    numerator = np.exp(shifted)
    return numerator / numerator.sum()


def consensus_probabilities(
    canonical: Mapping[str, Any], reversed_order: Mapping[str, Any]
) -> dict[str, Any]:
    first = softmax_scores(canonical)
    second = softmax_scores(reversed_order)
    consensus = (first + second) / 2.0
    entropy = float(-np.sum(consensus * np.log(consensus)) / math.log(4.0))
    midpoint = consensus
    js = 0.5 * np.sum(first * np.log(first / midpoint))
    js += 0.5 * np.sum(second * np.log(second / midpoint))
    js_normalized = float(js / math.log(2.0))
    residual = float(
        max(
            abs(first.sum() - 1.0),
            abs(second.sum() - 1.0),
            abs(consensus.sum() - 1.0),
        )
    )
    return {
        **{
            f"consensus_probability_{label}": float(consensus[index])
            for index, label in enumerate(ALL_LABELS)
        },
        "score_order_js_divergence_normalized": js_normalized,
        "score_consensus_entropy_normalized": entropy,
        "probability_sum_residual": residual,
    }


def load_predictions(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw in iter_jsonl(path):
        if (
            raw.get("terminal_state") != "complete"
            or raw.get("arm_id") != ARM_ID
            or raw.get("contract_id") != SOURCE_CONTRACT_ID
            or raw.get("source_profile") != SOURCE_PROFILE
        ):
            raise ValueError("Prediction record is not a complete source-arm record")
        event = raw.get("event_group")
        if not isinstance(event, Mapping) or event.get("input_truncated") is not False:
            raise ValueError("Truncated or malformed prediction record")
        pooled = consensus_probabilities(
            event["canonical_candidate_mean_log_probabilities"],
            event["reversed_candidate_mean_log_probabilities"],
        )
        records.append(
            {
                "article_id": str(raw["article_id"]),
                "prediction_model_text_sha256": str(raw["model_text_sha256"]),
                **pooled,
            }
        )
    frame = pd.DataFrame(records)
    if len(frame) != 50_488 or frame["article_id"].duplicated().any():
        raise ValueError("Prediction ledger count or uniqueness changed")
    probability_columns = [
        f"consensus_probability_{label}" for label in ALL_LABELS
    ]
    probability_sum = frame[probability_columns].sum(axis=1)
    if not np.allclose(probability_sum, 1.0, atol=1e-14, rtol=0):
        raise ValueError("Consensus probability vectors do not sum to one")
    return frame, {
        "article_count": len(frame),
        "maximum_probability_sum_residual": float(
            frame["probability_sum_residual"].max()
        ),
        "mean_normalized_js_divergence": float(
            frame["score_order_js_divergence_normalized"].mean()
        ),
        "mean_normalized_consensus_entropy": float(
            frame["score_consensus_entropy_normalized"].mean()
        ),
        "mean_consensus_probabilities": {
            label: float(frame[f"consensus_probability_{label}"].mean())
            for label in ALL_LABELS
        },
    }


def validate_frozen_inputs(
    *,
    predictions_path: Path,
    inference_manifest_path: Path,
    selection_manifest_path: Path,
    selected_assignments_path: Path,
    source_assignments_path: Path,
    scope_path: Path,
    corpus_manifest_path: Path,
    contract_path: Path,
    base_path: Path,
    base_manifest_path: Path,
) -> dict[str, dict[str, Any]]:
    paths = (
        predictions_path,
        inference_manifest_path,
        selection_manifest_path,
        selected_assignments_path,
        source_assignments_path,
        scope_path,
        corpus_manifest_path,
        contract_path,
        base_path,
        base_manifest_path,
        Path(__file__).resolve(),
    )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    validate_contract(contract_path)

    inference_manifest = load_json(inference_manifest_path)
    if (
        inference_manifest.get("manifest_version")
        != "flan-w17-lite-inference-manifest-v1"
        or inference_manifest.get("status") != "complete"
        or inference_manifest.get("full_corpus_complete") is not True
        or inference_manifest.get("all_selected_articles_terminal") is not True
        or int(inference_manifest.get("terminal_record_count", -1)) != 50_488
        or int(inference_manifest.get("failure_article_count", -1)) != 0
        or int(inference_manifest.get("remaining_article_count", -1)) != 0
        or inference_manifest.get("output_sha256")
        != sha256_file(predictions_path)
    ):
        raise ValueError("Completed W17-Lite inference manifest is invalid")

    selection_manifest = load_json(selection_manifest_path)
    if (
        selection_manifest.get("manifest_version")
        != "flan-w17-lite-selection-v1"
        or selection_manifest.get("status") != "complete"
        or selection_manifest.get("arm_id") != ARM_ID
        or selection_manifest.get("contract_id") != SOURCE_CONTRACT_ID
        or int(selection_manifest.get("selector_k", -1)) != 16
        or inference_manifest.get("selection_manifest_sha256")
        != sha256_file(selection_manifest_path)
    ):
        raise ValueError("W17-Lite selection manifest is invalid")
    selected_record = _manifest_record(
        selection_manifest, selected_assignments_path
    )
    if int(selected_record.get("rows", -1)) != 438_522:
        raise ValueError("Selected assignment row count changed")

    corpus_manifest = load_json(corpus_manifest_path)
    _validate_sidecar(corpus_manifest_path)
    if (
        corpus_manifest.get("manifest_version") != "semantic-corpus-manifest-v1"
        or corpus_manifest.get("status")
        != "complete_exploratory_retrospective"
        or corpus_manifest.get("source_profile") != SOURCE_PROFILE
    ):
        raise ValueError("Semantic corpus manifest is invalid")
    source_record = _manifest_record(corpus_manifest, source_assignments_path)
    scope_record = _manifest_record(corpus_manifest, scope_path)
    if (
        int(source_record.get("rows", -1)) != 466_902
        or int(scope_record.get("rows", -1)) != EXPECTED_ROWS
    ):
        raise ValueError("Semantic corpus row counts changed")

    base_manifest = load_json(base_manifest_path)
    _validate_sidecar(base_manifest_path)
    if (
        base_manifest.get("manifest_version")
        != "q-plus-d2-w17-lite-massive-manifest-v3"
        or base_manifest.get("status") != STATUS
    ):
        raise ValueError("v3 Q+D2+W17 base manifest is invalid")
    base_record = _manifest_record(base_manifest, base_path)
    if int(base_record.get("rows", -1)) != EXPECTED_ROWS:
        raise ValueError("v3 base panel row count changed")

    named = {
        "predictions": predictions_path,
        "inference_manifest": inference_manifest_path,
        "selection_manifest": selection_manifest_path,
        "selected_assignments": selected_assignments_path,
        "source_assignments": source_assignments_path,
        "stock_day_scope": scope_path,
        "semantic_corpus_manifest": corpus_manifest_path,
        "feature_contract": contract_path,
        "v3_base_panel": base_path,
        "v3_base_manifest": base_manifest_path,
        "builder": Path(__file__).resolve(),
    }
    return {
        name: {"path": display_path(path), "sha256": sha256_file(path)}
        for name, path in named.items()
    }


def load_scopes(path: Path, *, enforce_expected_profile: bool = True) -> pd.DataFrame:
    columns = [
        *KEY_COLUMNS,
        "source_profile",
        "source_query_scope_complete",
        "candidate_assignment_complete",
        "expected_assignment_count",
    ]
    frame = pd.read_parquet(path, columns=columns)
    frame["forecast_date"] = frame["forecast_date"].astype(str)
    profile_invalid = (
        len(frame) != EXPECTED_ROWS
        or frame["forecast_date"].nunique() != EXPECTED_DATES
        or frame["stock"].nunique() != EXPECTED_STOCKS
    )
    if (
        frame.empty
        or (enforce_expected_profile and profile_invalid)
        or frame.duplicated(list(KEY_COLUMNS)).any()
        or set(frame["source_profile"]) != {SOURCE_PROFILE}
        or not frame["source_query_scope_complete"].all()
        or not frame["candidate_assignment_complete"].all()
    ):
        raise ValueError("Authoritative stock-day scope is incomplete")
    return frame.sort_values(list(KEY_COLUMNS), kind="mergesort").reset_index(
        drop=True
    )


def load_assignments(
    *,
    source_path: Path,
    selected_path: Path,
    scopes: pd.DataFrame,
    enforce_expected_profile: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_columns = [
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
    ]
    selected_columns = [
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
        "text_sha256",
    ]
    source = pd.read_parquet(source_path, columns=source_columns)
    selected = pd.read_parquet(selected_path, columns=selected_columns)
    if enforce_expected_profile and (
        len(source) != 466_902 or len(selected) != 438_522
    ):
        raise ValueError("Assignment ledger row count changed")
    for frame, label in ((source, "source"), (selected, "selected")):
        if (
            frame["assignment_id"].duplicated().any()
            or set(frame["source_profile"]) != {SOURCE_PROFILE}
            or not frame["source_query_scope_complete"].all()
            or not frame["candidate_assignment_complete"].all()
            or not frame["role"].isin(ROLES).all()
            or not np.isfinite(frame["aggregation_weight"]).all()
            or not frame["aggregation_weight"].gt(0).all()
        ):
            raise ValueError(f"{label} assignment ledger is incomplete")
    source_ids = set(source["assignment_id"])
    if not set(selected["assignment_id"]).issubset(source_ids):
        raise ValueError("Selected assignments are not a source subset")
    source = source.rename(columns={"target_ticker": "stock"})
    selected = selected.rename(columns={"target_ticker": "stock"})
    for frame in (source, selected):
        frame["forecast_date"] = frame["forecast_date"].astype(str)
    benchmark = scopes[["forecast_date", "sector", "stock", "benchmark"]]
    selected = selected.merge(
        benchmark,
        on=["forecast_date", "sector", "stock"],
        how="left",
        validate="many_to_one",
    )
    if selected["benchmark"].isna().any():
        raise ValueError("Selected assignments do not map to stock-day scope")
    source_counts = (
        source.groupby(list(KEY_COLUMNS), sort=False)
        .size()
        .rename("observed_assignment_count")
        .reset_index()
    )
    checked = scopes.merge(
        source_counts, on=list(KEY_COLUMNS), how="left", validate="one_to_one"
    )
    checked["observed_assignment_count"] = (
        checked["observed_assignment_count"].fillna(0).astype(int)
    )
    if not checked["expected_assignment_count"].equals(
        checked["observed_assignment_count"]
    ):
        raise ValueError("Source assignments differ from authoritative scope")
    return source, selected


def attach_predictions(
    selected: pd.DataFrame, predictions: pd.DataFrame
) -> pd.DataFrame:
    if set(selected["article_id"]) != set(predictions["article_id"]):
        raise ValueError("Selected article universe differs from prediction ledger")
    hashes = selected.groupby("article_id", sort=False)["text_sha256"].nunique()
    if hashes.max() != 1:
        raise ValueError("Selected article text hashes differ across assignments")
    article_hash = selected[["article_id", "text_sha256"]].drop_duplicates(
        "article_id"
    )
    checked = predictions.merge(
        article_hash, on="article_id", how="inner", validate="one_to_one"
    )
    if not checked["prediction_model_text_sha256"].equals(
        checked["text_sha256"]
    ):
        raise ValueError("Prediction text hash differs from selected input")
    output = selected.merge(
        predictions.drop(columns=["prediction_model_text_sha256"]),
        on="article_id",
        how="inner",
        validate="many_to_one",
    )
    if len(output) != len(selected):
        raise AssertionError("Prediction join lost selected assignments")
    return output


def apply_probability_permutation(
    selected: pd.DataFrame, *, seed: str = PERMUTATION_SEED
) -> tuple[pd.DataFrame, str]:
    """Permute complete consensus vectors within date-sector article pools.

    Assignment roles, weights, and the recipient article's order-JSD remain
    fixed.  Consensus entropy moves with (and is determined by) the donated
    four-class probability vector.
    """

    probability_columns = [
        f"consensus_probability_{label}" for label in ALL_LABELS
    ]
    state_columns = [
        *probability_columns,
        "score_consensus_entropy_normalized",
    ]
    required = {
        "forecast_date",
        "sector",
        "article_id",
        "assignment_id",
        "score_order_js_divergence_normalized",
        *state_columns,
    }
    missing = required.difference(selected.columns)
    if missing:
        raise ValueError(f"Permutation input lacks columns: {sorted(missing)}")
    state = selected[
        ["forecast_date", "sector", "article_id", *state_columns]
    ].drop_duplicates()
    if state.duplicated(["forecast_date", "sector", "article_id"]).any():
        raise ValueError("An article has inconsistent probability state in a pool")

    mapping_rows: list[dict[str, str]] = []
    hash_lines: list[str] = []
    for (forecast_date, sector), group in state.groupby(
        ["forecast_date", "sector"], sort=True
    ):
        article_ids = sorted(
            group["article_id"].astype(str).tolist(),
            key=lambda value: value.encode("utf-8"),
        )
        if len(article_ids) <= 1:
            donors = article_ids
        else:
            digest = int(
                sha256_text(f"{seed}\n{forecast_date}\n{sector}"), 16
            )
            offset = 1 + digest % (len(article_ids) - 1)
            donors = article_ids[offset:] + article_ids[:offset]
        for recipient, donor in zip(article_ids, donors, strict=True):
            row = {
                "forecast_date": str(forecast_date),
                "sector": str(sector),
                "article_id": recipient,
                "donor_article_id": donor,
            }
            mapping_rows.append(row)
            hash_lines.append(canonical_json(row))
    mapping = pd.DataFrame(mapping_rows)
    donor_state = state.rename(
        columns={
            "article_id": "donor_article_id",
            **{name: f"donor__{name}" for name in state_columns},
        }
    )
    mapping = mapping.merge(
        donor_state,
        on=["forecast_date", "sector", "donor_article_id"],
        how="inner",
        validate="one_to_one",
    )
    original_js = selected.set_index("assignment_id")[
        "score_order_js_divergence_normalized"
    ].copy()
    output = selected.merge(
        mapping,
        on=["forecast_date", "sector", "article_id"],
        how="left",
        validate="many_to_one",
    )
    if output["donor_article_id"].isna().any():
        raise ValueError("Probability-permutation mapping is incomplete")
    for name in state_columns:
        output[name] = output.pop(f"donor__{name}")
    output["probability_sum_residual"] = (
        output[probability_columns].sum(axis=1) - 1.0
    ).abs()
    output = output.drop(columns=["donor_article_id"]).sort_values(
        "assignment_id", kind="mergesort"
    ).reset_index(drop=True)
    observed_js = output.set_index("assignment_id")[
        "score_order_js_divergence_normalized"
    ].reindex(original_js.index)
    if not np.array_equal(
        original_js.to_numpy(dtype=float), observed_js.to_numpy(dtype=float)
    ):
        raise AssertionError("Permutation changed recipient order-JSD quality")
    if not np.allclose(
        output[probability_columns].sum(axis=1), 1.0, atol=2e-14, rtol=0
    ):
        raise AssertionError("Permuted probability vector does not sum to one")
    return output, sha256_text("\n".join(hash_lines))


def finite_prior_ewma(
    values: Sequence[float], *, window: int = 63, half_life: float = 21.0
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    output = np.full(len(array), np.nan, dtype=float)
    if window <= 0 or half_life <= 0:
        raise ValueError("EWMA window and half-life must be positive")
    if len(array) <= window:
        return output
    # Input windows are oldest to newest.  The immediately prior observation
    # has age zero and therefore weight one.
    ages = np.arange(window - 1, -1, -1, dtype=float)
    weights = np.exp(-math.log(2.0) * ages / half_life)
    weights /= weights.sum()
    for index in range(window, len(array)):
        prior = array[index - window : index]
        if np.isfinite(prior).all():
            output[index] = float(np.dot(prior, weights))
    return output


def add_innovations(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.sort_values(
        ["stock", "forecast_date", "sector"], kind="mergesort"
    ).copy()
    for current, innovation in zip(CURRENT9, INNOVATION9, strict=True):
        output[innovation] = np.nan
        for _, indices in output.groupby("stock", sort=True).groups.items():
            positions = np.asarray(list(indices), dtype=int)
            values = output.loc[positions, current].to_numpy(dtype=float)
            expected = finite_prior_ewma(values, window=63, half_life=21.0)
            output.loc[positions, innovation] = values - expected
    return output.sort_values(list(KEY_COLUMNS), kind="mergesort").reset_index(
        drop=True
    )


def add_coupling(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.drop(columns=list(COUPLING6), errors="ignore").copy()
    for label in PREDICTIVE_LABELS:
        i_current = _mass_name("I", label)
        p_current = _mass_name("P", label)
        c_current = _mass_name("C", label)
        current_name = f"lsoft_coupling_{label}_c_minus_i_minus_p"
        innovation_name = (
            f"lsoft_coupling_{label}_innovation_c_minus_i_minus_p"
        )
        output[current_name] = (
            output[c_current] - output[i_current] - output[p_current]
        )
        output[innovation_name] = (
            output[f"{c_current}_innovation_ewma63_hl21"]
            - output[f"{i_current}_innovation_ewma63_hl21"]
            - output[f"{p_current}_innovation_ewma63_hl21"]
        )
    return output


def aggregate_daily(
    *,
    scopes: pd.DataFrame,
    source: pd.DataFrame,
    selected: pd.DataFrame,
    variant_id: str = "canonical",
    enforce_expected_profile: bool = True,
) -> pd.DataFrame:
    keys = list(KEY_COLUMNS)
    source_grouped = (
        source.groupby(keys, sort=False)
        .agg(
            lsoft_full_assignment_count=("assignment_id", "size"),
            lsoft_full_assignment_weight=("aggregation_weight", "sum"),
        )
        .reset_index()
    )
    weighted = selected.copy()
    weight = weighted["aggregation_weight"].to_numpy(dtype=float)
    weighted["weighted_entropy"] = (
        weight
        * weighted["score_consensus_entropy_normalized"].to_numpy(dtype=float)
    )
    weighted["weighted_js"] = (
        weight
        * weighted["score_order_js_divergence_normalized"].to_numpy(dtype=float)
    )
    weighted["weighted_other"] = (
        weight
        * weighted["consensus_probability_other_or_unclear"].to_numpy(dtype=float)
    )
    weighted_columns: list[str] = []
    for role in ROLES:
        role_mask = weighted["role"].eq(role).to_numpy(dtype=float)
        for label in PREDICTIVE_LABELS:
            name = f"weighted_{role}_{label}"
            weighted[name] = (
                weight
                * role_mask
                * weighted[f"consensus_probability_{label}"].to_numpy(dtype=float)
            )
            weighted_columns.append(name)
    aggregation: dict[str, tuple[str, str]] = {
        "lsoft_selected_assignment_count": ("assignment_id", "size"),
        "lsoft_selected_assignment_weight": ("aggregation_weight", "sum"),
        "weighted_entropy": ("weighted_entropy", "sum"),
        "weighted_js": ("weighted_js", "sum"),
        "weighted_other": ("weighted_other", "sum"),
        "lsoft_max_article_probability_sum_residual": (
            "probability_sum_residual",
            "max",
        ),
    }
    aggregation.update({name: (name, "sum") for name in weighted_columns})
    selected_grouped = (
        weighted.groupby(keys, sort=False).agg(**aggregation).reset_index()
    )
    frame = (
        scopes[list(KEY_COLUMNS)]
        .merge(source_grouped, on=keys, how="left", validate="one_to_one")
        .merge(selected_grouped, on=keys, how="left", validate="one_to_one")
    )
    for column in (
        "lsoft_full_assignment_count",
        "lsoft_selected_assignment_count",
    ):
        frame[column] = frame[column].fillna(0).astype(int)
    fill_zero = [
        "lsoft_full_assignment_weight",
        "lsoft_selected_assignment_weight",
        "weighted_entropy",
        "weighted_js",
        "weighted_other",
        "lsoft_max_article_probability_sum_residual",
        *weighted_columns,
    ]
    frame[fill_zero] = frame[fill_zero].fillna(0.0)
    selected_weight = frame["lsoft_selected_assignment_weight"]
    full_weight = frame["lsoft_full_assignment_weight"]
    positive = selected_weight.gt(0)
    full_positive = full_weight.gt(0)
    if not positive.equals(full_positive):
        raise ValueError("K16 selected and full candidate support differ by row")
    if (selected_weight > full_weight + 1e-12).any():
        raise ValueError("Selected weight exceeds full candidate weight")

    for role in ROLES:
        for label in PREDICTIVE_LABELS:
            feature = _mass_name(role, label)
            frame[feature] = 0.0
            frame.loc[positive, feature] = (
                frame.loc[positive, f"weighted_{role}_{label}"]
                / selected_weight[positive]
            )
    frame[NO_SELECTED] = (~positive).astype(float)
    frame[QUALITY4[0]] = np.nan
    frame[QUALITY4[1]] = np.nan
    frame[QUALITY4[2]] = 0.0
    frame[QUALITY4[3]] = np.nan
    frame.loc[positive, QUALITY4[0]] = (
        frame.loc[positive, "weighted_js"] / selected_weight[positive]
    )
    frame.loc[positive, QUALITY4[1]] = (
        frame.loc[positive, "weighted_entropy"] / selected_weight[positive]
    )
    frame.loc[positive, QUALITY4[2]] = (
        frame.loc[positive, "weighted_other"] / selected_weight[positive]
    )
    frame.loc[full_positive, QUALITY4[3]] = (
        selected_weight[full_positive] / full_weight[full_positive]
    )
    decomposition = frame[list(CURRENT9)].sum(axis=1) + frame[QUALITY4[2]]
    frame["lsoft_daily_mass_decomposition_residual"] = np.where(
        positive, decomposition - 1.0, decomposition
    )
    if not np.allclose(
        frame["lsoft_daily_mass_decomposition_residual"],
        0.0,
        atol=2e-14,
        rtol=0,
    ):
        raise AssertionError("SoftRoute daily mass decomposition failed")

    frame = add_innovations(frame)
    frame = add_coupling(frame)
    frame["source_profile"] = SOURCE_PROFILE
    frame["point_in_time_version_safe"] = False
    frame["primary_training_eligible"] = False
    frame["confirmatory_eligible"] = False
    frame["exploratory_construction_eligible"] = True
    frame["variant_id"] = variant_id
    output = frame[list(DAILY_COLUMNS)].sort_values(
        list(KEY_COLUMNS), kind="mergesort"
    ).reset_index(drop=True)
    validate_daily_panel(
        output,
        variant_id=variant_id,
        enforce_expected_profile=enforce_expected_profile,
    )
    return output


def _replace_coupling(frame: pd.DataFrame) -> pd.DataFrame:
    coupled = add_coupling(frame)
    return coupled[list(DAILY_COLUMNS)].sort_values(
        list(KEY_COLUMNS), kind="mergesort"
    ).reset_index(drop=True)


def make_stale20(
    canonical: pd.DataFrame, *, enforce_expected_profile: bool = True
) -> pd.DataFrame:
    output = canonical.sort_values(
        ["stock", "forecast_date", "sector"], kind="mergesort"
    ).copy()
    semantic18 = [*CURRENT9, *INNOVATION9]
    output[semantic18] = output.groupby("stock", sort=False)[semantic18].shift(20)
    output["variant_id"] = "stale20"
    output = _replace_coupling(output)
    output["lsoft_daily_mass_decomposition_residual"] = np.nan
    validate_daily_panel(
        output,
        variant_id="stale20",
        enforce_expected_profile=enforce_expected_profile,
    )
    return output


def wrong_stock_mapping(frame: pd.DataFrame) -> tuple[dict[str, str], str]:
    pairs = frame[["sector", "stock"]].drop_duplicates()
    if pairs["stock"].duplicated().any():
        raise ValueError("A stock appears in more than one sector")
    mapping: dict[str, str] = {}
    rows: list[dict[str, str]] = []
    for sector, group in pairs.groupby("sector", sort=True):
        stocks = sorted(group["stock"].tolist(), key=lambda value: value.encode())
        if len(stocks) < 2:
            raise ValueError("Wrong-stock control requires at least two peers")
        donors = stocks[1:] + stocks[:1]
        for recipient, donor in zip(stocks, donors, strict=True):
            mapping[recipient] = donor
            rows.append(
                {"sector": str(sector), "recipient": recipient, "donor": donor}
            )
    return mapping, sha256_text("\n".join(canonical_json(row) for row in rows))


def make_wrong_stock(
    canonical: pd.DataFrame, *, enforce_expected_profile: bool = True
) -> tuple[pd.DataFrame, str]:
    mapping, mapping_hash = wrong_stock_mapping(canonical)
    target_columns = [
        name
        for name in (*CURRENT9, *INNOVATION9)
        if "_target_idiosyncratic_" in name or "_peer_idiosyncratic_" in name
    ]
    recipient = canonical.copy()
    recipient["donor_stock"] = recipient["stock"].map(mapping)
    if recipient["donor_stock"].isna().any():
        raise ValueError("Wrong-stock donor mapping is incomplete")
    donor = canonical[["forecast_date", "sector", "stock", *target_columns]].copy()
    donor = donor.rename(
        columns={
            "stock": "donor_stock",
            **{name: f"donor__{name}" for name in target_columns},
        }
    )
    output = recipient.merge(
        donor,
        on=["forecast_date", "sector", "donor_stock"],
        how="left",
        validate="one_to_one",
    )
    for name in target_columns:
        output[name] = output.pop(f"donor__{name}")
    output = output.drop(columns=["donor_stock"])
    output["variant_id"] = "wrong_stock"
    output = _replace_coupling(output)
    output["lsoft_daily_mass_decomposition_residual"] = np.nan
    validate_daily_panel(
        output,
        variant_id="wrong_stock",
        enforce_expected_profile=enforce_expected_profile,
    )
    return output, mapping_hash


def validate_daily_panel(
    frame: pd.DataFrame,
    *,
    variant_id: str,
    enforce_expected_profile: bool = True,
) -> None:
    profile_invalid = (
        len(frame) != EXPECTED_ROWS
        or frame["forecast_date"].nunique() != EXPECTED_DATES
        or frame["stock"].nunique() != EXPECTED_STOCKS
    )
    if (
        tuple(frame.columns) != DAILY_COLUMNS
        or frame.empty
        or (enforce_expected_profile and profile_invalid)
        or frame.duplicated(list(KEY_COLUMNS)).any()
        or set(frame["variant_id"]) != {variant_id}
        or set(frame["source_profile"]) != {SOURCE_PROFILE}
        or frame["point_in_time_version_safe"].any()
        or frame["primary_training_eligible"].any()
        or frame["confirmatory_eligible"].any()
        or not frame["exploratory_construction_eligible"].all()
    ):
        raise ValueError("Daily SoftRoute coverage or claim contract differs")
    if not frame[NO_SELECTED].isin([0.0, 1.0]).all():
        raise ValueError("No-selected indicator is not binary")
    numeric = frame[
        [*LEDGER_AUDIT_COLUMNS, *QUALITY4, *SOFT_ROUTE19, *COUPLING6]
    ].to_numpy(dtype=float)
    if np.isinf(numeric).any():
        raise ValueError("Daily SoftRoute panel contains infinity")
    current = frame[list(CURRENT9)].stack().dropna()
    if not current.between(-1e-12, 1.0 + 1e-12).all():
        raise ValueError("Current joint mass is outside [0,1]")
    for name in QUALITY4:
        observed = frame[name].dropna()
        if not observed.between(-1e-12, 1.0 + 1e-12).all():
            raise ValueError(f"Score-quality field is outside [0,1]: {name}")
    residual = frame["lsoft_daily_mass_decomposition_residual"]
    if variant_id in {"stale20", "wrong_stock"}:
        if not residual.isna().all():
            raise ValueError(
                "Transformed control must mark mass decomposition not applicable"
            )
    elif residual.isna().any() or not np.allclose(
        residual, 0.0, atol=2e-14, rtol=0
    ):
        raise ValueError("Daily mass decomposition residual is invalid")
    probability = frame["lsoft_max_article_probability_sum_residual"]
    if not probability.between(0.0, 2e-14).all():
        raise ValueError("Article probability sum residual is too large")
    no_selected = frame[NO_SELECTED].eq(1.0)
    if (
        variant_id in {"canonical", "permuted"}
        and not frame.loc[no_selected, list(CURRENT9)].eq(0.0).all().all()
    ):
        raise ValueError("No-selected rows are not valid zero-mass rows")
    if (
        not frame.loc[no_selected, QUALITY4[2]].eq(0.0).all()
        or not frame.loc[no_selected, "lsoft_selected_assignment_weight"].eq(0.0).all()
    ):
        raise ValueError("Current no-selected audit state changed")
    if enforce_expected_profile:
        if int(no_selected.sum()) != 204:
            raise ValueError("No-selected stock-day count changed")
        expected_current_nulls = 0 if variant_id != "stale20" else 20 * 30
        expected_innovation_nulls = (
            63 * 30 if variant_id != "stale20" else 83 * 30
        )
        if any(frame[name].isna().sum() != expected_current_nulls for name in CURRENT9):
            raise ValueError("Current-mass control warmup differs")
        if any(
            frame[name].isna().sum() != expected_innovation_nulls
            for name in INNOVATION9
        ):
            raise ValueError("Innovation warmup differs")


def load_base_without_v3_wlite(path: Path) -> pd.DataFrame:
    raw = pd.read_parquet(path)
    wlite_columns = [name for name in raw if name.startswith("wlite_")]
    if len(raw) != EXPECTED_ROWS or len(wlite_columns) != 18:
        raise ValueError("v3 modeling panel coverage or W17 column count changed")
    base = raw.drop(columns=wlite_columns)
    if any(name.startswith("wlite_") for name in base):
        raise AssertionError("A v3 W17 column survived base projection")
    if base.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("Projected Q+D2 base has duplicate stock-days")
    return base


def build_joined_panel(
    base: pd.DataFrame,
    daily: pd.DataFrame,
    *,
    enforce_expected_profile: bool = True,
) -> pd.DataFrame:
    feature_columns = [*SOFT_ROUTE19, *COUPLING6, *QUALITY4]
    collision = set(feature_columns).intersection(base.columns)
    if collision:
        raise ValueError(f"SoftRoute columns already exist in base: {sorted(collision)}")
    right = daily[[*KEY_COLUMNS, *feature_columns]].copy()
    if pd.api.types.is_datetime64_any_dtype(base["forecast_date"]):
        right["forecast_date"] = pd.to_datetime(right["forecast_date"]).astype(
            base["forecast_date"].dtype
        )
    right["lsoft_row_matched"] = 1
    output = base.merge(
        right,
        on=list(KEY_COLUMNS),
        how="left",
        validate="one_to_one",
        sort=False,
    )
    profile_invalid = len(output) != EXPECTED_ROWS
    if (
        (enforce_expected_profile and profile_invalid)
        or output["lsoft_row_matched"].isna().any()
        or not output["lsoft_row_matched"].eq(1).all()
    ):
        raise ValueError("SoftRoute join was not complete and one-to-one")
    pd.testing.assert_frame_equal(
        output[list(base.columns)], base, check_dtype=True, check_exact=True
    )
    return output


def _schema_sha256(frame: pd.DataFrame) -> str:
    return sha256_text(
        canonical_json(
            [{"name": name, "dtype": str(frame[name].dtype)} for name in frame]
        )
    )


def _file_record(path: Path, frame: pd.DataFrame, variant_id: str) -> dict[str, Any]:
    return {
        "path": display_path(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "rows": len(frame),
        "columns": len(frame.columns),
        "schema_sha256": _schema_sha256(frame),
        "variant_id": variant_id,
    }


def _variant_stats(frame: pd.DataFrame) -> dict[str, Any]:
    decomposition = frame["lsoft_daily_mass_decomposition_residual"].dropna()
    return {
        "rows": len(frame),
        "dates": int(frame["forecast_date"].nunique()),
        "stocks": int(frame["stock"].nunique()),
        "sectors": int(frame["sector"].nunique()),
        "first_date": str(frame["forecast_date"].min()),
        "last_date": str(frame["forecast_date"].max()),
        "no_selected_rows": int(frame[NO_SELECTED].sum()),
        "selected_assignment_rows": int(
            frame["lsoft_selected_assignment_count"].sum()
        ),
        "full_assignment_rows": int(frame["lsoft_full_assignment_count"].sum()),
        "selected_weight_fraction": float(
            frame["lsoft_selected_assignment_weight"].sum()
            / frame["lsoft_full_assignment_weight"].sum()
        ),
        "null_counts": {
            name: int(frame[name].isna().sum())
            for name in (*SOFT_ROUTE19, *COUPLING6, *QUALITY4)
        },
        "maximum_probability_sum_residual": float(
            frame["lsoft_max_article_probability_sum_residual"].max()
        ),
        "mass_decomposition_audit_applicable": not decomposition.empty,
        "maximum_absolute_mass_decomposition_residual": (
            float(decomposition.abs().max()) if not decomposition.empty else None
        ),
    }


def build_daily_manifest(
    *,
    inputs: Mapping[str, Mapping[str, Any]],
    output_paths: Mapping[str, Path],
    frames: Mapping[str, pd.DataFrame],
    prediction_audit: Mapping[str, Any],
    wrong_stock_mapping_sha256: str,
    probability_permutation_mapping_sha256: str,
) -> dict[str, Any]:
    return {
        "manifest_version": DAILY_MANIFEST_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "status": STATUS,
        "contract_id": CONTRACT_ID,
        "source_arm_id": ARM_ID,
        "source_contract_id": SOURCE_CONTRACT_ID,
        "source_profile": SOURCE_PROFILE,
        "ordered_feature_lists": {
            "soft_route_19": list(SOFT_ROUTE19),
            "current_joint_mass_9": list(CURRENT9),
            "innovation_9": list(INNOVATION9),
            "coupling_6_residual_only": list(COUPLING6),
            "score_quality_4_audit_or_control": list(QUALITY4),
        },
        "feature_list_sha256": sha256_text(
            canonical_json(
                {
                    "soft_route_19": list(SOFT_ROUTE19),
                    "coupling_6": list(COUPLING6),
                    "score_quality_4": list(QUALITY4),
                }
            )
        ),
        "formula_contract": {
            "consensus_probability": "arithmetic mean of within-order softmax vectors",
            "current_denominator": "total selected stock-day assignment weight across I/P/C and all probability classes",
            "innovation": "current joint mass minus normalized finite EWMA of exactly 63 prior stock sessions",
            "innovation_half_life_sessions": 21,
            "stale20": "lag 18 current/innovation fields within stock by 20 sessions; retain current no-news and score-quality fields",
            "wrong_stock": "fixed same-date alphabetically-next within-sector donor for I/P current and innovation only; retain recipient C/no-news/quality",
            "permuted": "within date-sector, permute each unique article's complete four-class consensus vector; retain recipient roles/weights/order-JSD, then reaggregate",
            "transformed_control_mass_decomposition_audit": "null/not-applicable for stale20 and wrong_stock; recomputed and valid for canonical and permuted",
        },
        "inputs": dict(inputs),
        "prediction_audit": dict(prediction_audit),
        "wrong_stock_mapping_sha256": wrong_stock_mapping_sha256,
        "probability_permutation_mapping_sha256": (
            probability_permutation_mapping_sha256
        ),
        "generated_files": {
            output_paths[variant].name: _file_record(
                output_paths[variant], frames[variant], variant
            )
            for variant in VARIANT_FILES
        },
        "variants": {
            variant: _variant_stats(frames[variant]) for variant in VARIANT_FILES
        },
        "claim_scope": {
            "point_in_time_version_safe": False,
            "primary_training_eligible": False,
            "confirmatory_eligible": False,
            "exploratory_construction_eligible": True,
            "development_only": True,
        },
    }


def build_join_manifest(
    *,
    daily_manifest_path: Path,
    base_path: Path,
    base_manifest_path: Path,
    output_paths: Mapping[str, Path],
    frames: Mapping[str, pd.DataFrame],
    base_columns: Sequence[str],
) -> dict[str, Any]:
    return {
        "manifest_version": JOIN_MANIFEST_VERSION,
        "builder_version": PIPELINE_VERSION,
        "status": STATUS,
        "contract_id": CONTRACT_ID,
        "source_profile": SOURCE_PROFILE,
        "inputs": {
            "daily_manifest": {
                "path": display_path(daily_manifest_path),
                "sha256": sha256_file(daily_manifest_path),
            },
            "v3_base_panel": {
                "path": display_path(base_path),
                "sha256": sha256_file(base_path),
            },
            "v3_base_manifest": {
                "path": display_path(base_manifest_path),
                "sha256": sha256_file(base_manifest_path),
            },
            "builder": {
                "path": display_path(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
        "ordered_feature_lists": {
            "soft_route_19": list(SOFT_ROUTE19),
            "current_joint_mass_9": list(CURRENT9),
            "innovation_9": list(INNOVATION9),
            "coupling_6_residual_only": list(COUPLING6),
            "score_quality_4_audit_or_control": list(QUALITY4),
        },
        "column_roles": {
            "base_q_d2_columns": list(base_columns),
            "join_keys": list(KEY_COLUMNS),
            "join_audit_columns": ["lsoft_row_matched"],
        },
        "generated_files": {
            output_paths[variant].name: _file_record(
                output_paths[variant], frames[variant], variant
            )
            for variant in JOINED_FILES
        },
        "counts": {
            variant: {
                "rows": len(frames[variant]),
                "matched_rows": int(frames[variant]["lsoft_row_matched"].sum()),
                "match_fraction": float(frames[variant]["lsoft_row_matched"].mean()),
                "missing_soft_route_values": int(
                    frames[variant][list(SOFT_ROUTE19)].isna().sum().sum()
                ),
            }
            for variant in JOINED_FILES
        },
        "claim_scope": {
            "point_in_time_version_safe": False,
            "primary_training_eligible": False,
            "confirmatory_eligible": False,
            "exploratory_only": True,
            "development_only": True,
        },
    }


def build_all(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    join_root: Path = DEFAULT_JOIN_ROOT,
    predictions_path: Path = DEFAULT_PREDICTIONS,
    inference_manifest_path: Path = DEFAULT_INFERENCE_MANIFEST,
    selection_manifest_path: Path = DEFAULT_SELECTION_MANIFEST,
    selected_assignments_path: Path = DEFAULT_SELECTED_ASSIGNMENTS,
    source_assignments_path: Path = DEFAULT_SOURCE_ASSIGNMENTS,
    scope_path: Path = DEFAULT_SCOPE,
    corpus_manifest_path: Path = DEFAULT_CORPUS_MANIFEST,
    contract_path: Path = DEFAULT_CONTRACT,
    base_path: Path = DEFAULT_BASE,
    base_manifest_path: Path = DEFAULT_BASE_MANIFEST,
    overwrite: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    daily_paths = {
        variant: output_root / filename for variant, filename in VARIANT_FILES.items()
    }
    joined_paths = {
        variant: join_root / filename for variant, filename in JOINED_FILES.items()
    }
    daily_manifest_path = output_root / "manifest.json"
    join_manifest_path = join_root / "manifest.json"
    all_outputs = [
        *daily_paths.values(),
        *joined_paths.values(),
        daily_manifest_path,
        daily_manifest_path.with_suffix(".sha256"),
        join_manifest_path,
        join_manifest_path.with_suffix(".sha256"),
    ]
    existing = [path for path in all_outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "SoftRoute outputs exist: " + ", ".join(map(str, existing))
        )

    inputs = validate_frozen_inputs(
        predictions_path=predictions_path,
        inference_manifest_path=inference_manifest_path,
        selection_manifest_path=selection_manifest_path,
        selected_assignments_path=selected_assignments_path,
        source_assignments_path=source_assignments_path,
        scope_path=scope_path,
        corpus_manifest_path=corpus_manifest_path,
        contract_path=contract_path,
        base_path=base_path,
        base_manifest_path=base_manifest_path,
    )
    immutable_hashes = {
        record["path"]: record["sha256"] for record in inputs.values()
    }
    predictions, prediction_audit = load_predictions(predictions_path)
    scopes = load_scopes(scope_path)
    source, selected = load_assignments(
        source_path=source_assignments_path,
        selected_path=selected_assignments_path,
        scopes=scopes,
    )
    selected = attach_predictions(selected, predictions)
    canonical = aggregate_daily(
        scopes=scopes, source=source, selected=selected, variant_id="canonical"
    )
    stale = make_stale20(canonical)
    wrong, mapping_hash = make_wrong_stock(canonical)
    permuted_selected, permutation_mapping_hash = apply_probability_permutation(
        selected, seed=PERMUTATION_SEED
    )
    permuted = aggregate_daily(
        scopes=scopes,
        source=source,
        selected=permuted_selected,
        variant_id="permuted",
    )
    daily_frames = {
        "canonical": canonical,
        "stale20": stale,
        "wrong_stock": wrong,
        "permuted": permuted,
    }

    for recorded_path, expected_hash in immutable_hashes.items():
        path = Path(recorded_path)
        if not path.is_absolute():
            path = ROOT / path
        if sha256_file(path) != expected_hash:
            raise ValueError(f"Immutable source changed during construction: {path}")

    for variant, frame in daily_frames.items():
        _atomic_parquet(daily_paths[variant], frame)
    daily_manifest = build_daily_manifest(
        inputs=inputs,
        output_paths=daily_paths,
        frames=daily_frames,
        prediction_audit=prediction_audit,
        wrong_stock_mapping_sha256=mapping_hash,
        probability_permutation_mapping_sha256=permutation_mapping_hash,
    )
    _atomic_json(daily_manifest_path, daily_manifest)
    _atomic_text(
        daily_manifest_path.with_suffix(".sha256"),
        sha256_file(daily_manifest_path) + "\n",
    )

    base = load_base_without_v3_wlite(base_path)
    joined_frames = {
        variant: build_joined_panel(base, daily_frames[variant])
        for variant in JOINED_FILES
    }
    for variant, frame in joined_frames.items():
        _atomic_parquet(joined_paths[variant], frame)
    join_manifest = build_join_manifest(
        daily_manifest_path=daily_manifest_path,
        base_path=base_path,
        base_manifest_path=base_manifest_path,
        output_paths=joined_paths,
        frames=joined_frames,
        base_columns=list(base.columns),
    )
    _atomic_json(join_manifest_path, join_manifest)
    _atomic_text(
        join_manifest_path.with_suffix(".sha256"),
        sha256_file(join_manifest_path) + "\n",
    )
    return daily_manifest, join_manifest


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    value.add_argument("--join-root", type=Path, default=DEFAULT_JOIN_ROOT)
    value.add_argument("--overwrite", action="store_true")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    daily_manifest, join_manifest = build_all(
        output_root=args.output_root,
        join_root=args.join_root,
        overwrite=args.overwrite,
    )
    print(
        canonical_json(
            {
                "status": "complete",
                "daily_manifest_sha256": sha256_text(
                    canonical_json(daily_manifest)
                ),
                "join_manifest_sha256": sha256_text(
                    canonical_json(join_manifest)
                ),
                "rows": join_manifest["counts"]["canonical"]["rows"],
                "soft_route_feature_count": len(SOFT_ROUTE19),
                "coupling_feature_count": len(COUPLING6),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
