#!/usr/bin/env python
"""Build the hash-bound final v3 W17-Lite comparison bundle.

This program does not fit or alter a model.  It fails closed unless every
required saved bundle, status record, prediction panel, fit record, and
artifact hash agrees with the locked v3 protocol.  It then compares paired
outer-test losses and writes a separate exploratory evaluation bundle.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.correlation_training import training_common as quant_common
from scripts.quant_deterministic_news_training_v3 import (
    common,
    contract,
    run_xgboost,
)


OUTPUT_ROOT = contract.OUTPUT_ROOT / "comparisons" / "final"
TRACKED_ROOT = contract.EXPERIMENT_ROOT / "comparisons" / "final"
SCRIPT_PATH = Path(
    "scripts/quant_deterministic_news_training_v3/"
    "run_final_comparison.py"
)
REQUIRED_LINEAR_BUNDLES = (
    "S0",
    "S1",
    "S2",
    "S3",
    *contract.CONTROL_BUNDLES,
)
ELASTIC_NET_STABILITY_BUNDLES = ("S2", "S3")
PRIMARY_COMPARISONS = ("S2_vs_S0", "S3_vs_S1")
GATE_SUFFIXES = ("COV", "L20", "WS", "PERM")
EXPECTED_BOOTSTRAP_RESAMPLES = 2_000
EXPECTED_MOVING_BLOCK_SESSIONS = 10
EXPECTED_BOOTSTRAP_SEED = 1_729
NONZERO_COEFFICIENT_TOLERANCE = 1e-12


@dataclass(frozen=True)
class Comparison:
    key: str
    candidate: str
    base: str
    family: str
    targets: tuple[str, ...] | None = None
    primary: bool = False
    required_for_useful_gate: bool = False


def comparison_ladder(
    s4_targets: Sequence[str] = (),
) -> tuple[Comparison, ...]:
    """Return the complete, fixed v3 comparison ladder."""

    gated_targets = tuple(s4_targets)
    if (
        len(gated_targets) != len(set(gated_targets))
        or set(gated_targets) - set(contract.TARGETS)
    ):
        raise ValueError("S4 targets must be a unique subset of v3 targets")
    values = [
        Comparison(
            "S1_vs_S0",
            "S1",
            "S0",
            "deterministic_increment",
        ),
        Comparison(
            "S2_vs_S0",
            "S2",
            "S0",
            "q_plus_l_increment",
            primary=True,
            required_for_useful_gate=True,
        ),
        Comparison(
            "S3_vs_S1",
            "S3",
            "S1",
            "semantic_beyond_qd",
            primary=True,
            required_for_useful_gate=True,
        ),
        Comparison(
            "S3_vs_S2",
            "S3",
            "S2",
            "deterministic_after_semantics",
        ),
    ]
    for architecture in ("S2", "S3"):
        for suffix, family in (
            ("COV", "coverage_text_only_falsification"),
            ("L20", "stale_event_falsification"),
            ("WS", "wrong_stock_falsification"),
            ("PERM", "date_sector_permutation_falsification"),
        ):
            values.append(
                Comparison(
                    f"{architecture}_vs_C-{architecture}-{suffix}",
                    architecture,
                    f"C-{architecture}-{suffix}",
                    family,
                    required_for_useful_gate=True,
                )
            )
        values.append(
            Comparison(
                f"C-{architecture}-LONG_vs_{architecture}",
                f"C-{architecture}-LONG",
                architecture,
                "long_description_sensitivity",
            )
        )
    if gated_targets:
        values.extend(
            [
                Comparison(
                    "S4_vs_S1",
                    "S4",
                    "S1",
                    "validation_gated_xgb_vs_qd",
                    gated_targets,
                ),
                Comparison(
                    "S4_vs_S3",
                    "S4",
                    "S3",
                    "validation_gated_xgb_vs_linear",
                    gated_targets,
                ),
            ]
        )
    keys = [value.key for value in values]
    if len(keys) != len(set(keys)):
        raise AssertionError("Comparison keys must be unique")
    return tuple(values)


def _load_status_ledger(
    protocol: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    if not contract.STATUS_PATH.is_file():
        raise FileNotFoundError("The v3 status ledger is missing")
    status = common.load_json(contract.STATUS_PATH)
    if (
        status.get("experiment_id")
        != "quant-deterministic-news-v3-w17-lite"
        or status.get("protocol_sha256")
        != common.sha256_file(contract.PROTOCOL_PATH)
        or status.get("semantic_gate_passed") is not False
        or status.get("failed_gate_override") is not True
        or protocol["claim_flags"]["semantic_gate_passed"] is not False
    ):
        raise RuntimeError("The v3 status ledger is stale or malformed")
    rows = status.get("bundles")
    if not isinstance(rows, list) or not all(
        isinstance(row, Mapping) for row in rows
    ):
        raise TypeError("The v3 status ledger lacks bundle records")
    records = {str(row.get("key")): row for row in rows}
    if (
        len(records) != len(rows)
        or set(records) != set(contract.BUNDLE_BY_KEY)
    ):
        raise RuntimeError("The v3 status ledger bundle set changed")
    return records


def _artifact_record(
    bundle_key: str,
    artifact_name: str,
    *,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = manifest.get("artifacts", {}).get(artifact_name)
    if not isinstance(artifact, Mapping):
        raise RuntimeError(
            f"{bundle_key} manifest has no {artifact_name} artifact"
        )
    expected_path = (
        common.bundle_for(bundle_key).output_path / artifact_name
    )
    path = Path(str(artifact.get("path", "")))
    if path.resolve() != expected_path.resolve():
        raise RuntimeError(
            f"{bundle_key}/{artifact_name} has a noncanonical path"
        )
    expected_hash = str(artifact.get("sha256", ""))
    if (
        not path.is_file()
        or common.sha256_file(path) != expected_hash
    ):
        raise RuntimeError(
            f"{bundle_key}/{artifact_name} is missing or hash-invalid"
        )
    return {
        "path": path.as_posix(),
        "sha256": expected_hash,
        "bytes": int(path.stat().st_size),
    }


def _verify_tracked_bundle(
    bundle_key: str,
    verified: Mapping[str, Any],
) -> dict[str, Any]:
    bundle = common.bundle_for(bundle_key)
    tracked = bundle.experiment_path
    output = bundle.output_path
    required = (
        tracked / "summary.json",
        tracked / "review.json",
        tracked / "artifact_pointer.json",
        tracked / "RESULTS.md",
    )
    if not all(path.is_file() for path in required):
        raise RuntimeError(f"{bundle_key} tracked handoff is incomplete")
    for name in ("summary.json", "review.json"):
        if common.sha256_file(tracked / name) != common.sha256_file(
            output / name
        ):
            raise RuntimeError(
                f"{bundle_key} tracked {name} differs from its bundle"
            )
    pointer = common.load_json(tracked / "artifact_pointer.json")
    if (
        Path(str(pointer.get("manifest_path", ""))).resolve()
        != Path(str(verified["manifest_path"])).resolve()
        or pointer.get("manifest_sha256")
        != verified["manifest_sha256"]
        or pointer.get("protocol_sha256")
        != common.sha256_file(contract.PROTOCOL_PATH)
    ):
        raise RuntimeError(f"{bundle_key} tracked artifact pointer is stale")
    summary = common.load_json(output / "summary.json")
    review = common.load_json(output / "review.json")
    expected_results = common.result_markdown(
        bundle_key, summary, review
    )
    if (tracked / "RESULTS.md").read_text(
        encoding="utf-8"
    ) != expected_results:
        raise RuntimeError(f"{bundle_key} tracked RESULTS.md is stale")
    return {
        "summary_sha256": common.sha256_file(tracked / "summary.json"),
        "review_sha256": common.sha256_file(tracked / "review.json"),
        "artifact_pointer_sha256": common.sha256_file(
            tracked / "artifact_pointer.json"
        ),
        "results_sha256": common.sha256_file(tracked / "RESULTS.md"),
    }


def _verify_bundle(
    bundle_key: str,
    status: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if status.get("status") != "complete":
        raise RuntimeError(
            f"{bundle_key} status is {status.get('status')!r}, not complete"
        )
    verified = common.verify_completed_bundle(bundle_key)
    manifest = verified["manifest"]
    if (
        manifest.get("bundle") != bundle_key
        or manifest.get("semantic_gate_passed") is not False
        or manifest.get("failed_gate_override") is not True
        or manifest.get("claim_flags", {}).get(
            "primary_training_eligible"
        )
        is not False
        or manifest.get("claim_flags", {}).get("confirmatory_eligible")
        is not False
    ):
        raise RuntimeError(f"{bundle_key} manifest claim contract changed")
    required_artifacts = {
        "predictions.parquet",
        "validation_predictions.parquet",
        "fits.json",
        "fold_metrics.json",
        "summary.json",
        "review.json",
        "model_config.json",
        "source_preflight.json",
    }
    if bundle_key == "S4":
        required_artifacts.add("validation_gate.json")
    missing = required_artifacts - set(manifest.get("artifacts", {}))
    if missing:
        raise RuntimeError(
            f"{bundle_key} misses artifacts: {sorted(missing)}"
        )
    artifacts = {
        name: _artifact_record(
            bundle_key, name, manifest=manifest
        )
        for name in sorted(manifest["artifacts"])
    }
    tracked = _verify_tracked_bundle(bundle_key, verified)
    provenance = {
        "status": "complete",
        "manifest_path": verified["manifest_path"],
        "manifest_sha256": verified["manifest_sha256"],
        "artifacts": artifacts,
        "tracked_handoff": tracked,
    }
    return verified, provenance


def _validate_prediction_panel(
    bundle_key: str,
    frame: pd.DataFrame,
    *,
    protocol: Mapping[str, Any],
    split_index: int,
    targets: Sequence[str] | None = None,
) -> dict[str, Any]:
    required = {
        *contract.PREDICTION_KEYS,
        "model",
        "actual_fisher_z",
        "actual_correlation",
        "predicted_fisher_z",
        "predicted_correlation",
        "persistence_fisher_z",
        "persistence_correlation",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(
            f"{bundle_key} predictions miss columns: {sorted(missing)}"
        )
    review = common.validate_prediction_panel(
        frame,
        protocol=protocol,
        split_index=split_index,
        targets=targets,
    )
    numeric = frame[
        [
            "actual_fisher_z",
            "actual_correlation",
            "predicted_fisher_z",
            "predicted_correlation",
            "persistence_fisher_z",
            "persistence_correlation",
        ]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise RuntimeError(
            f"{bundle_key} predictions contain nonfinite values"
        )
    stock_count = int(protocol["coverage"]["stock_count"])
    date_panel = frame.groupby(
        ["target", "fold", "forecast_date"], sort=False
    ).agg(rows=("stock", "size"), stocks=("stock", "nunique"))
    if (
        date_panel.empty
        or not date_panel["rows"].eq(stock_count).all()
        or not date_panel["stocks"].eq(stock_count).all()
    ):
        raise RuntimeError(
            f"{bundle_key} does not retain all stocks per forecast date"
        )
    expected_models = {common.bundle_for(bundle_key).slug}
    if set(frame["model"].astype(str)) != expected_models:
        raise RuntimeError(f"{bundle_key} prediction model name changed")
    return {
        **review,
        "all_stocks_per_forecast_date": True,
        "required_prediction_columns_present": True,
        "actuals_and_persistence_finite": True,
    }


def _validation_gate(
    s3: pd.DataFrame,
    s1: pd.DataFrame,
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Recompute the S4 target gate from hash-verified validation outputs."""

    _validate_prediction_panel(
        "S3",
        s3,
        protocol=protocol,
        split_index=1,
    )
    _validate_prediction_panel(
        "S1",
        s1,
        protocol=protocol,
        split_index=1,
    )
    columns = [
        *contract.PREDICTION_KEYS,
        "actual_fisher_z",
        "predicted_fisher_z",
    ]
    merged = s3[columns].merge(
        s1[columns],
        on=list(contract.PREDICTION_KEYS),
        how="outer",
        validate="one_to_one",
        suffixes=("_s3", "_s1"),
        indicator=True,
    )
    if merged.empty or not merged["_merge"].eq("both").all():
        raise RuntimeError("S3 and S1 validation keys differ")
    actual_delta = (
        merged["actual_fisher_z_s3"]
        - merged["actual_fisher_z_s1"]
    ).abs()
    if actual_delta.isna().any() or float(actual_delta.max()) > 1e-12:
        raise RuntimeError("S3 and S1 validation targets differ")
    output: list[dict[str, Any]] = []
    expected_folds = tuple(
        record["name"] for record in protocol["folds"]
    )
    for target in contract.TARGETS:
        folds: list[dict[str, Any]] = []
        for fold_name in expected_folds:
            part = merged[
                merged["target"].eq(target)
                & merged["fold"].eq(fold_name)
            ]
            if part.empty:
                raise RuntimeError(
                    f"S4 validation gate lacks {target}/{fold_name}"
                )
            actual = part["actual_fisher_z_s3"].to_numpy(dtype=float)
            s3_mse = float(
                np.square(
                    actual
                    - part["predicted_fisher_z_s3"].to_numpy(dtype=float)
                ).mean()
            )
            s1_mse = float(
                np.square(
                    actual
                    - part["predicted_fisher_z_s1"].to_numpy(dtype=float)
                ).mean()
            )
            folds.append(
                {
                    "fold": fold_name,
                    "rows": int(len(part)),
                    "s1_validation_mse": s1_mse,
                    "s3_validation_mse": s3_mse,
                    "s3_minus_s1_validation_mse": s3_mse - s1_mse,
                    "s3_improved": s3_mse < s1_mse,
                }
            )
        improved = sum(bool(item["s3_improved"]) for item in folds)
        output.append(
            {
                "target": target,
                "required_improved_validation_folds": 2,
                "improved_validation_folds": improved,
                "eligible": improved >= 2,
                "folds": folds,
            }
        )
    return output


def _assert_gate_records_equal(
    observed: Sequence[Mapping[str, Any]],
    expected: Sequence[Mapping[str, Any]],
) -> None:
    if len(observed) != len(expected):
        raise RuntimeError("Saved S4 validation gate has wrong length")
    for left, right in zip(observed, expected, strict=True):
        for name in (
            "target",
            "required_improved_validation_folds",
            "improved_validation_folds",
            "eligible",
        ):
            if left.get(name) != right.get(name):
                raise RuntimeError(
                    f"Saved S4 validation gate differs at {name}"
                )
        left_folds = left.get("folds")
        right_folds = right.get("folds")
        if (
            not isinstance(left_folds, list)
            or len(left_folds) != len(right_folds)
        ):
            raise RuntimeError("Saved S4 validation fold set differs")
        for left_fold, right_fold in zip(
            left_folds, right_folds, strict=True
        ):
            for name in ("fold", "rows", "s3_improved"):
                if left_fold.get(name) != right_fold.get(name):
                    raise RuntimeError(
                        f"Saved S4 gate fold differs at {name}"
                    )
            for name in (
                "s1_validation_mse",
                "s3_validation_mse",
                "s3_minus_s1_validation_mse",
            ):
                if not math.isclose(
                    float(left_fold.get(name)),
                    float(right_fold.get(name)),
                    rel_tol=0.0,
                    abs_tol=1e-15,
                ):
                    raise RuntimeError(
                        f"Saved S4 gate fold differs at {name}"
                    )


def _verify_skipped_s4(
    protocol: Mapping[str, Any],
    gate: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Verify provenance for an executed all-ineligible S4 gate."""

    verified = run_xgboost.verify_s4_outcome(protocol)
    if verified.get("status") != "skipped_validation_gate":
        raise RuntimeError("S4 status says skipped but its outcome does not")
    saved_gate = verified.get("validation_gate")
    if not isinstance(saved_gate, list):
        raise TypeError("Verified S4 skip lacks its validation gate")
    _assert_gate_records_equal(saved_gate, gate)
    manifest_path = Path(str(verified.get("manifest_path", "")))
    manifest_hash = str(verified.get("manifest_sha256", ""))
    reason = str(verified.get("reason", "")).strip()
    if (
        not manifest_path.is_file()
        or len(manifest_hash) != 64
        or common.sha256_file(manifest_path) != manifest_hash
        or not reason
    ):
        raise RuntimeError("Verified S4 skip manifest provenance changed")
    return {
        "status": "skipped_validation_gate",
        "reason": reason,
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": manifest_hash,
        "recomputed_gate": list(gate),
    }


def _load_inputs(
    protocol: Mapping[str, Any],
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, Any],
    dict[str, list[dict[str, Any]]],
    tuple[str, ...],
    list[dict[str, Any]],
    str | None,
]:
    status = _load_status_ledger(protocol)
    predictions: dict[str, pd.DataFrame] = {}
    provenance: dict[str, Any] = {}
    fits: dict[str, list[dict[str, Any]]] = {}
    verified_by_key: dict[str, Mapping[str, Any]] = {}
    for bundle_key in REQUIRED_LINEAR_BUNDLES:
        verified, source = _verify_bundle(
            bundle_key, status[bundle_key]
        )
        frame, prediction_source = common.load_completed_predictions(
            bundle_key
        )
        source["prediction_review"] = _validate_prediction_panel(
            bundle_key,
            frame,
            protocol=protocol,
            split_index=2,
        )
        source["prediction_source"] = prediction_source
        predictions[bundle_key] = frame
        provenance[bundle_key] = source
        verified_by_key[bundle_key] = verified

    for bundle_key in ELASTIC_NET_STABILITY_BUNDLES:
        manifest = verified_by_key[bundle_key]["manifest"]
        record = _artifact_record(
            bundle_key, "fits.json", manifest=manifest
        )
        raw = json.loads(
            Path(record["path"]).read_text(encoding="utf-8")
        )
        if not isinstance(raw, list) or not all(
            isinstance(item, dict) for item in raw
        ):
            raise TypeError(f"{bundle_key} fits.json must be a list")
        fits[bundle_key] = raw

    s1_validation, s1_validation_source = (
        common.load_completed_predictions("S1", validation=True)
    )
    s3_validation, s3_validation_source = (
        common.load_completed_predictions("S3", validation=True)
    )
    gate = _validation_gate(s3_validation, s1_validation, protocol)
    eligible_targets = tuple(
        record["target"] for record in gate if record["eligible"]
    )
    provenance["S4_validation_inputs"] = {
        "S1": s1_validation_source,
        "S3": s3_validation_source,
        "recomputed_gate": gate,
    }

    s4_status = status["S4"]
    s4_skip_reason: str | None = None
    if s4_status.get("status") == "complete":
        if not eligible_targets:
            raise RuntimeError(
                "S4 is complete although no target passed its gate"
            )
        verified, source = _verify_bundle("S4", s4_status)
        frame, prediction_source = common.load_completed_predictions("S4")
        source["prediction_review"] = _validate_prediction_panel(
            "S4",
            frame,
            protocol=protocol,
            split_index=2,
            targets=eligible_targets,
        )
        source["prediction_source"] = prediction_source
        saved_gate_path = Path(
            _artifact_record(
                "S4",
                "validation_gate.json",
                manifest=verified["manifest"],
            )["path"]
        )
        saved_gate = json.loads(
            saved_gate_path.read_text(encoding="utf-8")
        )
        if not isinstance(saved_gate, list):
            raise TypeError("S4 validation_gate.json must contain a list")
        _assert_gate_records_equal(saved_gate, gate)
        summary_path = Path(
            _artifact_record(
                "S4", "summary.json", manifest=verified["manifest"]
            )["path"]
        )
        summary = common.load_json(summary_path)
        if (
            tuple(summary.get("eligible_targets", ()))
            != eligible_targets
            or set(frame["target"]) != set(eligible_targets)
        ):
            raise RuntimeError("S4 saved target subset differs from its gate")
        predictions["S4"] = frame
        provenance["S4"] = source
    elif s4_status.get("status") == "skipped":
        if eligible_targets:
            raise RuntimeError(
                "S4 was skipped although target(s) passed its gate"
            )
        verified_skip = _verify_skipped_s4(protocol, gate)
        s4_skip_reason = str(
            verified_skip.get("reason")
            or s4_status.get("summary")
            or "No target passed the validation gate"
        )
        provenance["S4"] = {
            **verified_skip,
            "reason": s4_skip_reason,
        }
    else:
        raise RuntimeError(
            "S4 must be complete or explicitly skipped before evaluation"
        )
    return (
        predictions,
        provenance,
        fits,
        eligible_targets,
        gate,
        s4_skip_reason,
    )


def _merge_pair(
    comparison: Comparison,
    frames: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    candidate = frames[comparison.candidate].copy()
    base = frames[comparison.base].copy()
    if comparison.targets is not None:
        candidate = candidate[
            candidate["target"].isin(comparison.targets)
        ]
        base = base[base["target"].isin(comparison.targets)]
    columns = [
        *contract.PREDICTION_KEYS,
        "actual_fisher_z",
        "actual_correlation",
        "predicted_fisher_z",
        "predicted_correlation",
    ]
    merged = candidate[columns].merge(
        base[columns],
        on=list(contract.PREDICTION_KEYS),
        how="outer",
        validate="one_to_one",
        suffixes=("_candidate", "_base"),
        indicator=True,
    )
    if merged.empty or not merged["_merge"].eq("both").all():
        counts = merged["_merge"].value_counts().to_dict()
        raise ValueError(
            f"{comparison.key} prediction keys differ: {counts}"
        )
    for name in ("actual_fisher_z", "actual_correlation"):
        delta = (
            merged[f"{name}_candidate"]
            - merged[f"{name}_base"]
        ).abs()
        if delta.isna().any() or float(delta.max()) > 1e-12:
            raise ValueError(f"{comparison.key} actual targets differ")
    merged = merged.drop(columns="_merge")
    merged["candidate_squared_loss"] = np.square(
        merged["actual_fisher_z_candidate"]
        - merged["predicted_fisher_z_candidate"]
    )
    merged["base_squared_loss"] = np.square(
        merged["actual_fisher_z_base"]
        - merged["predicted_fisher_z_base"]
    )
    merged["squared_loss_delta"] = (
        merged["candidate_squared_loss"]
        - merged["base_squared_loss"]
    )
    losses = merged[
        [
            "candidate_squared_loss",
            "base_squared_loss",
            "squared_loss_delta",
        ]
    ].to_numpy(dtype=float)
    if not np.isfinite(losses).all():
        raise ValueError(f"{comparison.key} has nonfinite losses")
    return merged


def _moving_block_samples(
    frame: pd.DataFrame,
    *,
    block_sessions: int,
    resamples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return row-weighted means from fold-contained moving date blocks."""

    if block_sessions <= 0 or resamples <= 0:
        raise ValueError("Bootstrap dimensions must be positive")
    sampled_sums = np.zeros(resamples, dtype=float)
    sampled_counts = np.zeros(resamples, dtype=float)
    for _, fold_frame in frame.groupby("fold", sort=True):
        by_date = (
            fold_frame.groupby("forecast_date", sort=True)[
                "squared_loss_delta"
            ]
            .agg(["sum", "count"])
            .sort_index()
        )
        n_dates = len(by_date)
        if n_dates < block_sessions:
            raise ValueError("A fold is shorter than one bootstrap block")
        blocks_needed = math.ceil(n_dates / block_sessions)
        starts = rng.integers(
            0,
            n_dates - block_sessions + 1,
            size=(resamples, blocks_needed),
        )
        offsets = np.arange(block_sessions)
        positions = (starts[:, :, None] + offsets).reshape(
            resamples, -1
        )[:, :n_dates]
        sampled_sums += by_date["sum"].to_numpy(dtype=float)[
            positions
        ].sum(axis=1)
        sampled_counts += by_date["count"].to_numpy(dtype=float)[
            positions
        ].sum(axis=1)
    if (sampled_counts <= 0).any():
        raise AssertionError("Bootstrap produced an empty resample")
    return sampled_sums / sampled_counts


def _validate_inference_contract(protocol: Mapping[str, Any]) -> None:
    inference = protocol.get("inference")
    if (
        not isinstance(inference, Mapping)
        or int(inference.get("bootstrap_resamples", -1))
        != EXPECTED_BOOTSTRAP_RESAMPLES
        or int(inference.get("moving_block_sessions", -1))
        != EXPECTED_MOVING_BLOCK_SESSIONS
        or int(inference.get("seed", -1)) != EXPECTED_BOOTSTRAP_SEED
        or inference.get("bootstrap_unit")
        != "whole forecast dates with all stocks"
    ):
        raise ValueError("The locked v3 bootstrap contract changed")


def evaluate(
    comparisons: Sequence[Comparison],
    frames: Mapping[str, pd.DataFrame],
    protocol: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _validate_inference_contract(protocol)
    summaries: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    samples: list[pd.DataFrame] = []
    inference = protocol["inference"]
    resamples = int(inference["bootstrap_resamples"])
    block = int(inference["moving_block_sessions"])
    seed = int(inference["seed"])
    expected_folds = tuple(
        value["name"] for value in protocol["folds"]
    )
    for comparison_index, comparison in enumerate(comparisons):
        merged = _merge_pair(comparison, frames)
        for target_index, (target, group) in enumerate(
            merged.groupby("target", sort=True)
        ):
            observed_folds = tuple(sorted(group["fold"].unique()))
            if observed_folds != tuple(sorted(expected_folds)):
                raise ValueError(
                    f"{comparison.key}/{target} fold set differs"
                )
            candidate_sse = float(
                group["candidate_squared_loss"].sum()
            )
            base_sse = float(group["base_squared_loss"].sum())
            delta = float(group["squared_loss_delta"].mean())
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [seed, comparison_index, target_index]
                )
            )
            draw = _moving_block_samples(
                group,
                block_sessions=block,
                resamples=resamples,
                rng=rng,
            )
            lower, upper = np.quantile(draw, [0.025, 0.975])
            improved_folds = 0
            for fold_name, fold_group in group.groupby(
                "fold", sort=True
            ):
                fold_candidate_sse = float(
                    fold_group["candidate_squared_loss"].sum()
                )
                fold_base_sse = float(
                    fold_group["base_squared_loss"].sum()
                )
                fold_delta = float(
                    fold_group["squared_loss_delta"].mean()
                )
                improved_folds += int(fold_delta < 0)
                folds.append(
                    {
                        "comparison": comparison.key,
                        "family": comparison.family,
                        "primary": comparison.primary,
                        "required_for_useful_gate": (
                            comparison.required_for_useful_gate
                        ),
                        "candidate": comparison.candidate,
                        "base": comparison.base,
                        "target": target,
                        "fold": fold_name,
                        "rows": int(len(fold_group)),
                        "dates": int(
                            fold_group["forecast_date"].nunique()
                        ),
                        "candidate_mse": (
                            fold_candidate_sse / len(fold_group)
                        ),
                        "base_mse": (
                            fold_base_sse / len(fold_group)
                        ),
                        "candidate_minus_base_mse": fold_delta,
                        "incremental_r2_vs_base": (
                            1.0 - fold_candidate_sse / fold_base_sse
                            if fold_base_sse > 0
                            else math.nan
                        ),
                        "candidate_improved": fold_delta < 0,
                    }
                )
            summaries.append(
                {
                    "comparison": comparison.key,
                    "family": comparison.family,
                    "primary": comparison.primary,
                    "required_for_useful_gate": (
                        comparison.required_for_useful_gate
                    ),
                    "candidate": comparison.candidate,
                    "base": comparison.base,
                    "target": target,
                    "rows": int(len(group)),
                    "dates": int(group["forecast_date"].nunique()),
                    "fold_count": len(expected_folds),
                    "candidate_mse": candidate_sse / len(group),
                    "base_mse": base_sse / len(group),
                    "candidate_fisher_z_rmse": math.sqrt(
                        candidate_sse / len(group)
                    ),
                    "base_fisher_z_rmse": math.sqrt(
                        base_sse / len(group)
                    ),
                    "candidate_minus_base_mse": delta,
                    "incremental_r2_vs_base": (
                        1.0 - candidate_sse / base_sse
                        if base_sse > 0
                        else math.nan
                    ),
                    "bootstrap_ci_lower": float(lower),
                    "bootstrap_ci_upper": float(upper),
                    "bootstrap_probability_candidate_better": float(
                        np.mean(draw < 0)
                    ),
                    "improved_development_folds": improved_folds,
                    "ci_excludes_zero_in_candidate_direction": bool(
                        upper < 0
                    ),
                }
            )
            samples.append(
                pd.DataFrame(
                    {
                        "comparison": comparison.key,
                        "target": target,
                        "resample": np.arange(
                            resamples, dtype=np.int32
                        ),
                        "candidate_minus_base_mse": draw,
                    }
                )
            )
    summary = pd.DataFrame(summaries)
    fold_frame = pd.DataFrame(folds)
    bootstrap = pd.concat(samples, ignore_index=True)
    finite = summary[
        [
            "candidate_mse",
            "base_mse",
            "candidate_minus_base_mse",
            "incremental_r2_vs_base",
            "bootstrap_ci_lower",
            "bootstrap_ci_upper",
        ]
    ].to_numpy(dtype=float)
    if summary.empty or not np.isfinite(finite).all():
        raise AssertionError("Final comparison metrics are incomplete")
    if len(bootstrap) != len(summary) * resamples:
        raise AssertionError("Bootstrap output count is incomplete")
    return summary, fold_frame, bootstrap


def _validate_gate_contract(protocol: Mapping[str, Any]) -> None:
    expected = {
        "S2": {
            "name": "q_plus_l_gate",
            "base": "S0",
        },
        "S3": {
            "name": "useful_semantic_gate",
            "base": "S1",
        },
    }
    for architecture, values in expected.items():
        gate = protocol.get("gates", {}).get(values["name"])
        must_beat = [
            values["base"],
            *[
                f"C-{architecture}-{suffix}"
                for suffix in GATE_SUFFIXES
            ],
        ]
        if (
            not isinstance(gate, Mapping)
            or gate.get("candidate") != architecture
            or gate.get("must_beat") != must_beat
            or gate.get("bootstrap_upper_bound_below_zero") is not True
            or int(
                gate.get("minimum_improved_development_folds", -1)
            )
            != 2
            or gate.get("long_description_is_sensitivity_only")
            is not True
        ):
            raise ValueError(
                f"The locked {architecture} useful gate changed"
            )


def useful_gate(
    summary: pd.DataFrame,
    *,
    architecture: str,
) -> list[dict[str, Any]]:
    if architecture not in {"S2", "S3"}:
        raise ValueError("Useful gate applies only to S2 and S3")
    matched_base = "S0" if architecture == "S2" else "S1"
    required = [
        f"{architecture}_vs_{matched_base}",
        *[
            f"{architecture}_vs_C-{architecture}-{suffix}"
            for suffix in GATE_SUFFIXES
        ],
    ]
    output: list[dict[str, Any]] = []
    for target in contract.TARGETS:
        rows = summary[
            summary["target"].eq(target)
            & summary["comparison"].isin(required)
        ]
        if set(rows["comparison"]) != set(required) or len(rows) != len(
            required
        ):
            raise ValueError(
                f"{architecture}/{target} misses useful-gate comparisons"
            )
        checks: dict[str, Any] = {}
        for row in rows.to_dict("records"):
            conditions = {
                "point_improved": (
                    float(row["candidate_minus_base_mse"]) < 0
                ),
                "bootstrap_upper_below_zero": (
                    float(row["bootstrap_ci_upper"]) < 0
                ),
                "improved_at_least_two_folds": (
                    int(row["improved_development_folds"]) >= 2
                ),
                "all_three_folds_present": (
                    int(row.get("fold_count", 3)) == 3
                ),
            }
            checks[str(row["comparison"])] = {
                **conditions,
                "passed": all(conditions.values()),
            }
        passed = all(value["passed"] for value in checks.values())
        output.append(
            {
                "architecture": architecture,
                "target": target,
                "matched_base": matched_base,
                "required_comparisons": required,
                "comparison_checks": checks,
                "matched_base_passed": checks[
                    f"{architecture}_vs_{matched_base}"
                ]["passed"],
                "all_falsification_controls_passed": all(
                    checks[
                        f"{architecture}_vs_C-{architecture}-{suffix}"
                    ]["passed"]
                    for suffix in GATE_SUFFIXES
                ),
                "passed_all_required": passed,
                "semantic_quality_gate_passed": False,
                "claim_if_passed": (
                    "exploratory downstream usefulness only; the FLAN "
                    "choice-order semantic-quality gate still failed"
                ),
            }
        )
    return output


def _wlite_feature_block(feature: str) -> str:
    if feature in contract.EVENT_CONTENT_4:
        return "flan_event_content_4"
    if feature in contract.COVERAGE_TEXT_6:
        return "coverage_text_6"
    if feature in contract.SEMANTIC_CONTENT_11:
        return "deterministic_routing_status_content_7"
    raise ValueError(f"Unknown W17-Lite feature {feature}")


def coefficient_stability(
    bundle_key: str,
    fits: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    *,
    tolerance: float = NONZERO_COEFFICIENT_TOLERANCE,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Summarize W17-Lite Elastic Net coefficient selection by fold."""

    if bundle_key not in ELASTIC_NET_STABILITY_BUNDLES:
        raise ValueError("Coefficient stability is limited to S2 and S3")
    expected_folds = tuple(
        record["name"] for record in protocol["folds"]
    )
    records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for target in contract.TARGETS:
        target_fits = [
            fit for fit in fits if str(fit.get("target")) == target
        ]
        observed_folds = tuple(
            sorted(str(fit.get("fold")) for fit in target_fits)
        )
        if observed_folds != tuple(sorted(expected_folds)):
            raise ValueError(
                f"{bundle_key}/{target} fit folds differ"
            )
        expected_features = common.model_features(
            bundle_key, target, protocol
        )
        coefficient_maps: dict[str, dict[str, float]] = {}
        for fit in target_fits:
            fold = str(fit["fold"])
            features = tuple(
                str(value) for value in fit.get("features", ())
            )
            coefficients = fit.get("coefficients")
            if not isinstance(coefficients, list):
                raise TypeError(
                    f"{bundle_key}/{target}/{fold} lacks coefficients"
                )
            coefficient_features = tuple(
                str(value.get("feature")) for value in coefficients
            )
            if (
                features != expected_features
                or coefficient_features != expected_features
                or int(fit.get("feature_count", -1))
                != len(expected_features)
                or fit.get("feature_ordered_sha256")
                != common.ordered_sha256(expected_features)
            ):
                raise ValueError(
                    f"{bundle_key}/{target}/{fold} feature contract differs"
                )
            values = np.asarray(
                [
                    float(value["coefficient_standardized"])
                    for value in coefficients
                ],
                dtype=float,
            )
            if not np.isfinite(values).all():
                raise ValueError(
                    f"{bundle_key}/{target}/{fold} has nonfinite coefficients"
                )
            coefficient_maps[fold] = dict(
                zip(expected_features, values, strict=True)
            )

        selected_sets: list[set[str]] = []
        for fold in expected_folds:
            selected_sets.append(
                {
                    feature
                    for feature in contract.WLITE_17
                    if abs(coefficient_maps[fold][feature]) > tolerance
                }
            )
        for feature in contract.WLITE_17:
            coefficients = np.asarray(
                [
                    coefficient_maps[fold][feature]
                    for fold in expected_folds
                ],
                dtype=float,
            )
            selected = np.abs(coefficients) > tolerance
            selected_values = coefficients[selected]
            sign_consistent: bool | None = None
            if len(selected_values):
                sign_consistent = bool(
                    np.all(selected_values > 0)
                    or np.all(selected_values < 0)
                )
            record: dict[str, Any] = {
                "bundle": bundle_key,
                "target": target,
                "feature": feature,
                "feature_block": _wlite_feature_block(feature),
                "folds_total": len(expected_folds),
                "folds_nonzero": int(selected.sum()),
                "selection_rate": float(selected.mean()),
                "selected_all_folds": bool(selected.all()),
                "sign_consistent_when_selected": sign_consistent,
                "stable_sign_all_folds": bool(
                    selected.all() and sign_consistent
                ),
                "mean_coefficient_standardized": float(
                    coefficients.mean()
                ),
                "median_coefficient_standardized": float(
                    np.median(coefficients)
                ),
                "mean_abs_coefficient_standardized": float(
                    np.abs(coefficients).mean()
                ),
                "max_abs_coefficient_standardized": float(
                    np.abs(coefficients).max()
                ),
            }
            for fold, coefficient, is_selected in zip(
                expected_folds,
                coefficients,
                selected,
                strict=True,
            ):
                record[f"{fold}_coefficient_standardized"] = float(
                    coefficient
                )
                record[f"{fold}_selected"] = bool(is_selected)
            records.append(record)
        pairwise_jaccard: list[float] = []
        for left, right in itertools.combinations(selected_sets, 2):
            union = left | right
            pairwise_jaccard.append(
                len(left & right) / len(union) if union else 1.0
            )
        union = set().union(*selected_sets)
        intersection = set.intersection(*selected_sets)
        stable = {
            record["feature"]
            for record in records
            if record["bundle"] == bundle_key
            and record["target"] == target
            and record["stable_sign_all_folds"]
        }
        summaries.append(
            {
                "bundle": bundle_key,
                "target": target,
                "wlite_feature_count": len(contract.WLITE_17),
                "wlite_union_selected_count": len(union),
                "wlite_selected_all_folds_count": len(intersection),
                "wlite_stable_sign_all_folds_count": len(stable),
                "wlite_never_selected_count": (
                    len(contract.WLITE_17) - len(union)
                ),
                "mean_pairwise_wlite_selected_set_jaccard": float(
                    np.mean(pairwise_jaccard)
                ),
                "nonzero_coefficient_tolerance": tolerance,
            }
        )
    frame = pd.DataFrame(records).sort_values(
        ["bundle", "target", "feature"], kind="mergesort"
    )
    return frame.reset_index(drop=True), summaries


def _model_metrics(
    predictions: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for bundle_key, frame in sorted(predictions.items()):
        metrics = quant_common.summarize_predictions(
            frame, group_columns=("target",)
        )
        records.extend(
            {"bundle": bundle_key, **record}
            for record in metrics.to_dict("records")
        )
    output = pd.DataFrame(records).sort_values(
        ["bundle", "target"], kind="mergesort"
    )
    numeric = output[
        [
            "fisher_z_rmse",
            "fisher_z_mae",
            "raw_correlation_rmse",
            "raw_correlation_mae",
            "oos_r2_vs_persistence",
        ]
    ].to_numpy(dtype=float)
    if output.empty or not np.isfinite(numeric).all():
        raise AssertionError("Model metrics are incomplete or nonfinite")
    return output.reset_index(drop=True)


def _markdown_table(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    digits: int = 6,
) -> list[str]:
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    lines = [header, separator]
    for row in frame[list(columns)].itertuples(index=False, name=None):
        values: list[str] = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.{digits}f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _render_report(
    comparisons: pd.DataFrame,
    gates: Sequence[Mapping[str, Any]],
    stability: pd.DataFrame,
    *,
    s4_targets: Sequence[str],
    s4_skip_reason: str | None,
    protocol: Mapping[str, Any],
) -> str:
    primary = comparisons[comparisons["primary"]].copy()
    gate_frame = pd.DataFrame(
        [
            {
                "architecture": row["architecture"],
                "target": row["target"],
                "matched_base_passed": row["matched_base_passed"],
                "controls_passed": row[
                    "all_falsification_controls_passed"
                ],
                "all_required_passed": row["passed_all_required"],
            }
            for row in gates
        ]
    )
    s4_note = (
        "S4 was evaluated only for validation-gated target(s): "
        + ", ".join(s4_targets)
        if s4_targets
        else "S4 was skipped: " + str(s4_skip_reason)
    )
    inference = protocol["inference"]
    return "\n".join(
        [
            "# V3 W17-Lite final comparison",
            "",
            (
                "Status: **exploratory failed-semantic-gate development "
                "diagnostic**."
            ),
            "",
            (
                "FLAN W17-Lite canonical/reversed choice-order agreement "
                "missed the frozen 85% semantic-quality threshold. The "
                "user-authorized fits and every comparison below remain "
                "diagnostic, even if a downstream usefulness gate passes."
            ),
            "",
            "## Primary matched comparisons",
            "",
            *_markdown_table(
                primary,
                (
                    "comparison",
                    "target",
                    "incremental_r2_vs_base",
                    "candidate_minus_base_mse",
                    "bootstrap_ci_lower",
                    "bootstrap_ci_upper",
                    "improved_development_folds",
                ),
                digits=8,
            ),
            "",
            "## All paired comparisons and controls",
            "",
            *_markdown_table(
                comparisons,
                (
                    "comparison",
                    "target",
                    "incremental_r2_vs_base",
                    "candidate_minus_base_mse",
                    "bootstrap_ci_lower",
                    "bootstrap_ci_upper",
                    "improved_development_folds",
                ),
                digits=8,
            ),
            "",
            "## Conservative useful-semantic gates",
            "",
            *_markdown_table(
                gate_frame,
                tuple(gate_frame.columns),
            ),
            "",
            (
                "Each S2 or S3 target must beat its matched base and all "
                "four coverage, stale-event, wrong-stock, and date-sector "
                "permutation controls, with a negative point loss delta, "
                "an upper 95% bound below zero, and improvement in at least "
                "two of three folds. Long-description is report-only."
            ),
            "",
            "## W17-Lite Elastic Net coefficient stability",
            "",
            *_markdown_table(
                stability,
                (
                    "bundle",
                    "target",
                    "wlite_union_selected_count",
                    "wlite_selected_all_folds_count",
                    "wlite_stable_sign_all_folds_count",
                    "mean_pairwise_wlite_selected_set_jaccard",
                ),
            ),
            "",
            "## Validation-gated nonlinear subset",
            "",
            s4_note + ".",
            "",
            "## Inference and claim boundary",
            "",
            (
                f"- Confidence intervals use "
                f"{int(inference['bootstrap_resamples']):,} paired, "
                f"fold-contained moving-block resamples of "
                f"{int(inference['moving_block_sessions'])} forecast "
                "sessions, retaining every stock on each sampled date."
            ),
            "- Loss is squared error in Fisher-z space.",
            "- No multiple-testing adjustment is applied.",
            "- These dates were previously inspected development periods.",
            (
                "- The retrospective ordinary Massive archive is not "
                "article-version safe."
            ),
            "- Semantic-quality gate passed: `false`.",
            (
                f"- Protocol SHA-256: "
                f"`{common.sha256_file(contract.PROTOCOL_PATH)}`."
            ),
            "",
        ]
    )


def _ensure_output_paths(*, overwrite: bool) -> None:
    paths = (
        (OUTPUT_ROOT, contract.OUTPUT_ROOT),
        (TRACKED_ROOT, contract.EXPERIMENT_ROOT),
    )
    occupied = any(
        path.exists() and any(path.iterdir())
        for path, _ in paths
    )
    if occupied and not overwrite:
        raise FileExistsError(
            "Final v3 comparison exists; pass --overwrite to replace it"
        )
    if overwrite:
        for path, root in paths:
            if not path.exists():
                continue
            resolved = path.resolve()
            resolved_root = root.resolve()
            if (
                resolved == resolved_root
                or resolved_root not in resolved.parents
            ):
                raise ValueError(
                    f"Unsafe final-comparison cleanup path: {path}"
                )
            shutil.rmtree(path)


def _write_outputs(
    *,
    protocol: Mapping[str, Any],
    provenance: Mapping[str, Any],
    ladder: Sequence[Comparison],
    model_metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
    folds: pd.DataFrame,
    bootstrap: pd.DataFrame,
    gates: Sequence[Mapping[str, Any]],
    stability: pd.DataFrame,
    stability_summary: Sequence[Mapping[str, Any]],
    xgboost_gate: Sequence[Mapping[str, Any]],
    s4_targets: Sequence[str],
    s4_skip_reason: str | None,
    report: str,
) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    TRACKED_ROOT.mkdir(parents=True, exist_ok=False)
    values: dict[str, object] = {
        "model_metrics.parquet": model_metrics,
        "paired_comparisons.parquet": comparisons,
        "fold_comparisons.parquet": folds,
        "bootstrap_loss_deltas.parquet": bootstrap,
        "useful_semantic_gates.json": list(gates),
        "elastic_net_wlite_coefficient_stability.parquet": stability,
        "elastic_net_wlite_stability_summary.json": list(
            stability_summary
        ),
        "xgboost_validation_gate_audit.json": {
            "status": "complete" if s4_targets else "skipped",
            "eligible_targets": list(s4_targets),
            "skip_reason": s4_skip_reason,
            "recomputed_gate": list(xgboost_gate),
        },
        "comparison_specs.json": [
            asdict(value) for value in ladder
        ],
        "dependencies.json": dict(provenance),
    }
    for name, value in values.items():
        path = OUTPUT_ROOT / name
        if isinstance(value, pd.DataFrame):
            common.atomic_parquet(path, value)
        else:
            common.atomic_json(path, value)

    primary = comparisons[
        comparisons["comparison"].isin(PRIMARY_COMPARISONS)
    ].to_dict("records")
    result = {
        "status": "complete",
        "generated_at_utc": common.utc_now(),
        "result_role": "exploratory_failed_semantic_gate_diagnostic",
        "semantic_gate_passed": False,
        "primary_comparison_keys": list(PRIMARY_COMPARISONS),
        "primary_comparisons": primary,
        "comparison_specs": [asdict(value) for value in ladder],
        "useful_semantic_gates": list(gates),
        "useful_gate_pass_count": sum(
            bool(record["passed_all_required"]) for record in gates
        ),
        "s4_targets": list(s4_targets),
        "s4_skip_reason": s4_skip_reason,
        "elastic_net_wlite_stability_summary": list(
            stability_summary
        ),
    }
    review = {
        "status": "passed",
        "all_required_bundle_statuses_complete_or_executed_skip": True,
        "all_bundle_manifests_and_artifact_hashes_verified": True,
        (
            "all_completed_bundle_handoffs_and_s4_skip_pointer_verified"
        ): True,
        "prediction_panels_preflighted": True,
        "prediction_keys_exact_in_every_pair": True,
        "actual_targets_exact_in_every_pair": True,
        "all_stocks_retained_per_forecast_date": True,
        "fold_sets_exact_in_every_comparison": True,
        "bootstrap_fold_contained": True,
        "bootstrap_unit": protocol["inference"]["bootstrap_unit"],
        "bootstrap_resamples": int(
            protocol["inference"]["bootstrap_resamples"]
        ),
        "moving_block_sessions": int(
            protocol["inference"]["moving_block_sessions"]
        ),
        "bootstrap_seed": int(protocol["inference"]["seed"]),
        "xgboost_gate_recomputed_from_validation_predictions": True,
        "xgboost_saved_gate_and_target_subset_verified": bool(
            s4_targets
        ),
        "xgboost_skip_verified_no_eligible_targets": not bool(
            s4_targets
        ),
        "elastic_net_wlite_fit_contracts_verified": True,
        "semantic_gate_passed": False,
        "failed_semantic_gate_override_confirmed": True,
        "primary_training_eligible": False,
        "confirmatory_eligible": False,
    }
    common.atomic_json(OUTPUT_ROOT / "summary.json", result)
    common.atomic_json(OUTPUT_ROOT / "review.json", review)
    common.atomic_text(OUTPUT_ROOT / "RESULTS.md", report)

    artifacts = {
        path.name: {
            "path": path.as_posix(),
            "sha256": common.sha256_file(path),
            "bytes": int(path.stat().st_size),
        }
        for path in sorted(OUTPUT_ROOT.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = {
        "manifest_version": (
            "quant-deterministic-news-v3-final-comparison-v1"
        ),
        "experiment_id": (
            "quant-deterministic-news-v3-w17-lite-final-comparison"
        ),
        "status": "complete",
        "generated_at_utc": common.utc_now(),
        "protocol_path": contract.PROTOCOL_PATH.as_posix(),
        "protocol_sha256": common.sha256_file(
            contract.PROTOCOL_PATH
        ),
        "semantic_gate_passed": False,
        "failed_gate_override": True,
        "claim_flags": {
            "exploratory": True,
            "primary_training_eligible": False,
            "confirmatory_eligible": False,
        },
        "comparison_script_path": SCRIPT_PATH.as_posix(),
        "comparison_script_sha256": common.sha256_file(SCRIPT_PATH),
        "comparisons": [asdict(value) for value in ladder],
        "primary_comparison_keys": list(PRIMARY_COMPARISONS),
        "inference": {
            **dict(protocol["inference"]),
            "paired_loss_delta": "candidate minus base",
            "loss": "Fisher-z squared error",
            "fold_contained": True,
            "all_stocks_per_sampled_date": True,
            "pooling": "stock-day row weighted",
        },
        "coefficient_nonzero_tolerance": (
            NONZERO_COEFFICIENT_TOLERANCE
        ),
        "s4_targets": list(s4_targets),
        "s4_skip_reason": s4_skip_reason,
        "dependencies": dict(provenance),
        "artifacts": artifacts,
    }
    common.atomic_json(OUTPUT_ROOT / "manifest.json", manifest)
    manifest_hash = common.sha256_file(
        OUTPUT_ROOT / "manifest.json"
    )
    common.atomic_text(
        OUTPUT_ROOT / "manifest.sha256",
        f"{manifest_hash}  manifest.json\n",
    )
    refs = {
        **artifacts,
        "manifest.json": {
            "path": (OUTPUT_ROOT / "manifest.json").as_posix(),
            "sha256": manifest_hash,
            "bytes": int(
                (OUTPUT_ROOT / "manifest.json").stat().st_size
            ),
        },
    }
    common.atomic_json(TRACKED_ROOT / "summary.json", result)
    common.atomic_json(TRACKED_ROOT / "review.json", review)
    common.atomic_json(TRACKED_ROOT / "artifact_refs.json", refs)
    common.atomic_json(
        TRACKED_ROOT / "coefficient_stability_summary.json",
        list(stability_summary),
    )
    common.atomic_text(TRACKED_ROOT / "RESULTS.md", report)


def parser() -> argparse.ArgumentParser:
    output = argparse.ArgumentParser(description=__doc__)
    output.add_argument(
        "--allow-failed-gate-exploratory", action="store_true"
    )
    output.add_argument("--overwrite", action="store_true")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not args.allow_failed_gate_exploratory:
        raise ValueError(
            "Pass --allow-failed-gate-exploratory to acknowledge the "
            "failed semantic-quality gate"
        )
    protocol = common.load_protocol()
    _validate_inference_contract(protocol)
    _validate_gate_contract(protocol)
    _ensure_output_paths(overwrite=args.overwrite)
    (
        predictions,
        provenance,
        fits,
        s4_targets,
        xgboost_gate,
        s4_skip_reason,
    ) = _load_inputs(protocol)
    ladder = comparison_ladder(s4_targets)
    comparisons, folds, bootstrap = evaluate(
        ladder, predictions, protocol
    )
    gates = [
        *useful_gate(comparisons, architecture="S2"),
        *useful_gate(comparisons, architecture="S3"),
    ]
    stability_frames: list[pd.DataFrame] = []
    stability_summary: list[dict[str, Any]] = []
    for bundle_key in ELASTIC_NET_STABILITY_BUNDLES:
        frame, records = coefficient_stability(
            bundle_key, fits[bundle_key], protocol
        )
        stability_frames.append(frame)
        stability_summary.extend(records)
    stability = pd.concat(stability_frames, ignore_index=True)
    stability_summary_frame = pd.DataFrame(stability_summary)
    model_metrics = _model_metrics(predictions)
    report = _render_report(
        comparisons,
        gates,
        stability_summary_frame,
        s4_targets=s4_targets,
        s4_skip_reason=s4_skip_reason,
        protocol=protocol,
    )
    _write_outputs(
        protocol=protocol,
        provenance=provenance,
        ladder=ladder,
        model_metrics=model_metrics,
        comparisons=comparisons,
        folds=folds,
        bootstrap=bootstrap,
        gates=gates,
        stability=stability,
        stability_summary=stability_summary,
        xgboost_gate=xgboost_gate,
        s4_targets=s4_targets,
        s4_skip_reason=s4_skip_reason,
        report=report,
    )
    print(
        comparisons[
            comparisons["comparison"].isin(PRIMARY_COMPARISONS)
        ].to_string(index=False)
    )
    print(pd.DataFrame(gates).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
