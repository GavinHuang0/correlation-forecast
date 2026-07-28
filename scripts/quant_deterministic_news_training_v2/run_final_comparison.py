"""Build the hash-bound deterministic-news v2 final comparison bundle.

This script does not fit or alter a model. It verifies the completed model
bundles, compares their saved outer-test predictions, and writes a separate
exploratory inference bundle.
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
from scripts.quant_deterministic_news_training_v2 import common


OUTPUT_ROOT = Path("outputs/quant_deterministic_news/v2/comparisons/final")
TRACKED_ROOT = Path(
    "experiments/quant_deterministic_news/v2/training/comparisons/final"
)
SCRIPT_PATH = Path(
    "scripts/quant_deterministic_news_training_v2/run_final_comparison.py"
)
REQUIRED_BUNDLES = ("D0", "D1", "D3", "D4", "C-D3-L20", "C-D3-WS")
ELASTIC_NET_STABILITY_BUNDLES = ("D3", "D4")
NONZERO_COEFFICIENT_TOLERANCE = 1e-12


@dataclass(frozen=True)
class Comparison:
    key: str
    candidate: str
    base: str
    family: str
    targets: tuple[str, ...] | None = None


def comparison_ladder(
    d5_targets: Sequence[str] | None = None,
) -> tuple[Comparison, ...]:
    """Return the fixed deterministic comparison ladder.

    D5 is validation gated, so its comparisons are restricted to the target
    subset present in its completed prediction artifact.
    """

    comparisons = [
        Comparison("D1_vs_D0", "D1", "D0", "d2_only_diagnostic"),
        Comparison("D3_vs_D0", "D3", "D0", "primary_d2_increment"),
        Comparison("D4_vs_D0", "D4", "D0", "d2_levels_sensitivity"),
        Comparison("D4_vs_D3", "D4", "D3", "d2_levels_increment"),
        Comparison(
            "D3_vs_C-D3-L20",
            "D3",
            "C-D3-L20",
            "stale_news_falsification",
        ),
        Comparison(
            "D3_vs_C-D3-WS",
            "D3",
            "C-D3-WS",
            "wrong_stock_falsification",
        ),
    ]
    gated_targets = tuple(d5_targets or ())
    if gated_targets:
        comparisons.extend(
            [
                Comparison(
                    "D5_vs_D0",
                    "D5",
                    "D0",
                    "validation_gated_xgboost_vs_quant",
                    targets=gated_targets,
                ),
                Comparison(
                    "D5_vs_D3",
                    "D5",
                    "D3",
                    "validation_gated_xgboost_vs_linear",
                    targets=gated_targets,
                ),
            ]
        )
    keys = [comparison.key for comparison in comparisons]
    if len(keys) != len(set(keys)):
        raise AssertionError("Comparison keys must be unique")
    return tuple(comparisons)


def _status_record(bundle_key: str) -> Mapping[str, Any]:
    status = common.load_json(common.STATUS_PATH)
    if status.get("protocol_sha256") != common.sha256_file(
        common.PROTOCOL_PATH
    ):
        raise RuntimeError("The v2 status ledger uses a stale protocol")
    try:
        return next(
            item for item in status["bundles"] if item["key"] == bundle_key
        )
    except StopIteration as error:
        raise RuntimeError(f"Status ledger has no {bundle_key} record") from error


def _artifact_record(
    bundle_key: str,
    artifact_name: str,
    *,
    manifest: Mapping[str, Any],
) -> dict[str, str]:
    artifact = manifest.get("artifacts", {}).get(artifact_name)
    if not isinstance(artifact, Mapping):
        raise RuntimeError(
            f"{bundle_key} manifest has no {artifact_name} artifact"
        )
    expected_path = common.bundle_for(bundle_key).output_path / artifact_name
    path = Path(str(artifact.get("path")))
    if path.resolve() != expected_path.resolve():
        raise RuntimeError(
            f"{bundle_key} {artifact_name} path is noncanonical"
        )
    expected_hash = str(artifact.get("sha256"))
    if not path.exists() or common.sha256_file(path) != expected_hash:
        raise RuntimeError(
            f"{bundle_key} {artifact_name} is missing or hash-invalid"
        )
    return {"path": path.as_posix(), "sha256": expected_hash}


def _validate_prediction_panel(
    bundle_key: str,
    frame: pd.DataFrame,
    *,
    expected_stock_count: int,
) -> None:
    required = {
        *common.PREDICTION_KEYS,
        "actual_fisher_z",
        "actual_correlation",
        "predicted_fisher_z",
        "predicted_correlation",
        "persistence_fisher_z",
        "persistence_correlation",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(
            f"{bundle_key} predictions miss columns: {sorted(missing)}"
        )
    if frame.empty or frame.duplicated(common.PREDICTION_KEYS).any():
        raise RuntimeError(f"{bundle_key} predictions are empty or duplicated")
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
        raise RuntimeError(f"{bundle_key} predictions contain nonfinite values")
    if not frame["predicted_correlation"].between(-1, 1).all():
        raise RuntimeError(f"{bundle_key} correlations exceed [-1, 1]")
    date_panel = frame.groupby(
        ["target", "fold", "forecast_date"], sort=False
    ).agg(rows=("stock", "size"), stocks=("stock", "nunique"))
    if (
        not date_panel["rows"].eq(expected_stock_count).all()
        or not date_panel["stocks"].eq(expected_stock_count).all()
    ):
        raise RuntimeError(
            f"{bundle_key} does not retain all stocks on every forecast date"
        )


def _load_inputs(
    protocol: Mapping[str, Any],
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, Any],
    dict[str, list[dict[str, Any]]],
    str | None,
]:
    predictions: dict[str, pd.DataFrame] = {}
    provenance: dict[str, Any] = {}
    fits: dict[str, list[dict[str, Any]]] = {}
    stock_count = int(protocol["source_artifacts"]["stock_count"])
    for bundle_key in REQUIRED_BUNDLES:
        frame, verified = common.load_completed_predictions(bundle_key)
        _validate_prediction_panel(
            bundle_key, frame, expected_stock_count=stock_count
        )
        manifest = common.load_json(Path(str(verified["manifest_path"])))
        prediction_artifact = _artifact_record(
            bundle_key, "predictions.parquet", manifest=manifest
        )
        provenance[bundle_key] = {
            **verified,
            "predictions": prediction_artifact,
        }
        predictions[bundle_key] = frame

    d5_skip_reason: str | None = None
    d5_status = _status_record("D5")
    if d5_status["status"] == "complete":
        frame, verified = common.load_completed_predictions("D5")
        _validate_prediction_panel("D5", frame, expected_stock_count=stock_count)
        manifest = common.load_json(Path(str(verified["manifest_path"])))
        provenance["D5"] = {
            **verified,
            "predictions": _artifact_record(
                "D5", "predictions.parquet", manifest=manifest
            ),
        }
        predictions["D5"] = frame
    elif d5_status["status"] == "skipped":
        d5_skip_reason = str(d5_status.get("summary") or "validation gate failed")
        provenance["D5"] = {
            "status": "skipped",
            "reason": d5_skip_reason,
        }
    else:
        raise RuntimeError(
            "D5 must be complete or explicitly skipped before comparison"
        )

    for bundle_key in ELASTIC_NET_STABILITY_BUNDLES:
        verified = common.verify_completed_bundle(bundle_key)
        manifest = verified["manifest"]
        artifact = _artifact_record(
            bundle_key, "fits.json", manifest=manifest
        )
        raw = json.loads(Path(artifact["path"]).read_text(encoding="utf-8"))
        if not isinstance(raw, list) or not all(
            isinstance(item, dict) for item in raw
        ):
            raise TypeError(f"{bundle_key} fits.json must contain a list")
        fits[bundle_key] = raw
        provenance[bundle_key]["fits"] = artifact
    return predictions, provenance, fits, d5_skip_reason


def _merge_pair(
    comparison: Comparison,
    predictions: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    candidate = predictions[comparison.candidate].copy()
    base = predictions[comparison.base].copy()
    if comparison.targets is not None:
        candidate = candidate[candidate["target"].isin(comparison.targets)]
        base = base[base["target"].isin(comparison.targets)]
    columns = [
        *common.PREDICTION_KEYS,
        "actual_fisher_z",
        "actual_correlation",
        "predicted_fisher_z",
        "predicted_correlation",
    ]
    merged = candidate[columns].merge(
        base[columns],
        on=common.PREDICTION_KEYS,
        how="outer",
        validate="one_to_one",
        suffixes=("_candidate", "_base"),
        indicator=True,
    )
    if merged.empty or not merged["_merge"].eq("both").all():
        counts = merged["_merge"].value_counts().to_dict()
        raise ValueError(f"{comparison.key} prediction keys differ: {counts}")
    for column in ("actual_fisher_z", "actual_correlation"):
        delta = (
            merged[f"{column}_candidate"] - merged[f"{column}_base"]
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
        merged["candidate_squared_loss"] - merged["base_squared_loss"]
    )
    if not np.isfinite(
        merged[
            [
                "candidate_squared_loss",
                "base_squared_loss",
                "squared_loss_delta",
            ]
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError(f"{comparison.key} contains nonfinite loss values")
    return merged


def _moving_block_samples(
    frame: pd.DataFrame,
    *,
    block_sessions: int,
    resamples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return row-weighted means from fold-contained moving date blocks."""

    sampled_sums = np.zeros(resamples, dtype=float)
    sampled_counts = np.zeros(resamples, dtype=float)
    for _, fold_frame in frame.groupby("fold", sort=True):
        dates = np.asarray(
            sorted(fold_frame["forecast_date"].unique()),
            dtype="datetime64[ns]",
        )
        if len(dates) < block_sessions:
            raise ValueError("A fold contains fewer dates than one block")
        by_date = (
            fold_frame.groupby("forecast_date", sort=True)[
                "squared_loss_delta"
            ]
            .agg(["sum", "count"])
            .reindex(pd.to_datetime(dates))
        )
        if by_date.isna().any().any():
            raise ValueError("Date aggregation lost a fold date")
        blocks_needed = math.ceil(len(dates) / block_sessions)
        starts = rng.integers(
            0,
            len(dates) - block_sessions + 1,
            size=(resamples, blocks_needed),
        )
        offsets = np.arange(block_sessions)
        indices = (starts[:, :, None] + offsets).reshape(resamples, -1)
        indices = indices[:, : len(dates)]
        sampled_sums += by_date["sum"].to_numpy(dtype=float)[indices].sum(
            axis=1
        )
        sampled_counts += by_date["count"].to_numpy(dtype=float)[indices].sum(
            axis=1
        )
    if (sampled_counts <= 0).any():
        raise AssertionError("Bootstrap produced an empty resample")
    return sampled_sums / sampled_counts


def _comparison_records(
    comparison: Comparison,
    merged: pd.DataFrame,
    *,
    block_sessions: int,
    resamples: int,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], pd.DataFrame]:
    pooled_records: list[dict[str, Any]] = []
    fold_records: list[dict[str, Any]] = []
    sample_frames: list[pd.DataFrame] = []
    for target, target_frame in merged.groupby("target", sort=True):
        candidate_sse = float(target_frame["candidate_squared_loss"].sum())
        base_sse = float(target_frame["base_squared_loss"].sum())
        observed_delta = float(target_frame["squared_loss_delta"].mean())
        samples = _moving_block_samples(
            target_frame,
            block_sessions=block_sessions,
            resamples=resamples,
            rng=rng,
        )
        lower, upper = np.quantile(samples, [0.025, 0.975])
        fold_deltas: list[float] = []
        for fold, fold_frame in target_frame.groupby("fold", sort=True):
            fold_candidate_sse = float(
                fold_frame["candidate_squared_loss"].sum()
            )
            fold_base_sse = float(fold_frame["base_squared_loss"].sum())
            fold_delta = float(fold_frame["squared_loss_delta"].mean())
            fold_deltas.append(fold_delta)
            fold_records.append(
                {
                    "comparison": comparison.key,
                    "family": comparison.family,
                    "candidate": comparison.candidate,
                    "base": comparison.base,
                    "target": target,
                    "fold": fold,
                    "rows": len(fold_frame),
                    "dates": int(fold_frame["forecast_date"].nunique()),
                    "candidate_fisher_z_rmse": math.sqrt(
                        fold_candidate_sse / len(fold_frame)
                    ),
                    "base_fisher_z_rmse": math.sqrt(
                        fold_base_sse / len(fold_frame)
                    ),
                    "incremental_r2": (
                        1 - fold_candidate_sse / fold_base_sse
                        if fold_base_sse > 0
                        else math.nan
                    ),
                    "mean_squared_loss_delta": fold_delta,
                    "candidate_better": fold_delta < 0,
                }
            )
        pooled_records.append(
            {
                "comparison": comparison.key,
                "family": comparison.family,
                "candidate": comparison.candidate,
                "base": comparison.base,
                "target": target,
                "rows": len(target_frame),
                "dates": int(target_frame["forecast_date"].nunique()),
                "folds": ",".join(
                    sorted(target_frame["fold"].unique().tolist())
                ),
                "candidate_fisher_z_rmse": math.sqrt(
                    candidate_sse / len(target_frame)
                ),
                "base_fisher_z_rmse": math.sqrt(
                    base_sse / len(target_frame)
                ),
                "incremental_r2": (
                    1 - candidate_sse / base_sse
                    if base_sse > 0
                    else math.nan
                ),
                "mean_squared_loss_delta": observed_delta,
                "bootstrap_ci_lower": float(lower),
                "bootstrap_ci_upper": float(upper),
                "bootstrap_probability_candidate_better": float(
                    np.mean(samples < 0)
                ),
                "folds_candidate_better": int(
                    sum(delta < 0 for delta in fold_deltas)
                ),
                "fold_count": len(fold_deltas),
                "ci_excludes_zero_in_candidate_direction": bool(upper < 0),
            }
        )
        sample_frames.append(
            pd.DataFrame(
                {
                    "comparison": comparison.key,
                    "target": target,
                    "resample": np.arange(resamples, dtype=np.int32),
                    "mean_squared_loss_delta": samples,
                }
            )
        )
    return (
        pooled_records,
        fold_records,
        pd.concat(sample_frames, ignore_index=True),
    )


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
            for record in metrics.to_dict(orient="records")
        )
    return pd.DataFrame(records).sort_values(
        ["bundle", "target"]
    ).reset_index(drop=True)


def _decision_gate(comparisons: pd.DataFrame) -> pd.DataFrame:
    """Apply the conservative D3 useful-news gate target by target."""

    indexed = comparisons.set_index(["comparison", "target"])

    def improvement(key: str, target: str) -> bool:
        try:
            row = indexed.loc[(key, target)]
        except KeyError as error:
            raise ValueError(f"Decision-gate comparison missing: {key}/{target}") from error
        return bool(
            row["mean_squared_loss_delta"] < 0
            and row["bootstrap_ci_upper"] < 0
        )

    records: list[dict[str, Any]] = []
    for target in common.TARGETS:
        matched = indexed.loc[("D3_vs_D0", target)]
        conditions = {
            "matched_quant_improvement_ci": improvement(
                "D3_vs_D0", target
            ),
            "beats_stale_d2_ci": improvement(
                "D3_vs_C-D3-L20", target
            ),
            "beats_wrong_stock_d2_ci": improvement(
                "D3_vs_C-D3-WS", target
            ),
            "improves_at_least_two_of_three_folds": int(
                matched["folds_candidate_better"]
            )
            >= 2,
        }
        records.append(
            {
                "candidate": "D3",
                "target": target,
                **conditions,
                "all_required_gates_passed": all(conditions.values()),
            }
        )
    return pd.DataFrame(records)


def _feature_block(
    feature: str, protocol: Mapping[str, Any]
) -> str:
    blocks = protocol["feature_blocks"]
    if feature in set(blocks["d2_normalized_30"]):
        return "d2_normalized_30"
    if feature in set(blocks["d2_levels_5"]):
        return "d2_levels_5"
    return "quant_q56"


def _selection_stability(
    bundle_key: str,
    fits: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    *,
    tolerance: float = NONZERO_COEFFICIENT_TOLERANCE,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    expected_folds = tuple(fold["name"] for fold in protocol["folds"])
    records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for target in common.TARGETS:
        target_fits = [
            fit for fit in fits if str(fit.get("target")) == target
        ]
        observed_folds = tuple(
            sorted(str(fit.get("fold")) for fit in target_fits)
        )
        if observed_folds != tuple(sorted(expected_folds)):
            raise ValueError(
                f"{bundle_key}/{target} fit folds differ: {observed_folds}"
            )
        coefficient_maps: list[dict[str, float]] = []
        feature_order: tuple[str, ...] | None = None
        for fit in sorted(target_fits, key=lambda item: str(item["fold"])):
            features = tuple(str(feature) for feature in fit.get("features", []))
            coefficients = fit.get("coefficients")
            if not isinstance(coefficients, list):
                raise TypeError(
                    f"{bundle_key}/{target}/{fit['fold']} coefficients missing"
                )
            coefficient_features = tuple(
                str(item["feature"]) for item in coefficients
            )
            if (
                len(features) != int(fit.get("feature_count", -1))
                or len(features) != len(set(features))
                or coefficient_features != features
            ):
                raise ValueError(
                    f"{bundle_key}/{target}/{fit['fold']} feature contract differs"
                )
            if feature_order is None:
                feature_order = features
            elif feature_order != features:
                raise ValueError(
                    f"{bundle_key}/{target} feature order varies by fold"
                )
            coefficient_maps.append(
                {
                    str(item["feature"]): float(
                        item["coefficient_standardized"]
                    )
                    for item in coefficients
                }
            )
        if feature_order is None:
            raise AssertionError(f"{bundle_key}/{target} has no fits")
        selected_sets: list[set[str]] = []
        for coefficient_map in coefficient_maps:
            selected_sets.append(
                {
                    feature
                    for feature, coefficient in coefficient_map.items()
                    if abs(coefficient) > tolerance
                }
            )
        for feature in feature_order:
            coefficients = np.asarray(
                [mapping[feature] for mapping in coefficient_maps], dtype=float
            )
            selected = np.abs(coefficients) > tolerance
            selected_values = coefficients[selected]
            sign_consistent: bool | None = None
            if len(selected_values):
                sign_consistent = bool(
                    np.all(selected_values > 0)
                    or np.all(selected_values < 0)
                )
            records.append(
                {
                    "bundle": bundle_key,
                    "target": target,
                    "feature": feature,
                    "feature_block": _feature_block(feature, protocol),
                    "folds_total": len(expected_folds),
                    "folds_nonzero": int(selected.sum()),
                    "selection_rate": float(selected.mean()),
                    "mean_coefficient_standardized": float(
                        coefficients.mean()
                    ),
                    "mean_abs_coefficient_standardized": float(
                        np.abs(coefficients).mean()
                    ),
                    "sign_consistent_when_selected": sign_consistent,
                }
            )
        pairwise_jaccard: list[float] = []
        for left, right in itertools.combinations(selected_sets, 2):
            union = left | right
            pairwise_jaccard.append(
                len(left & right) / len(union) if union else 1.0
            )
        d2_features = set(protocol["feature_blocks"]["d2_normalized_30"])
        level_features = set(protocol["feature_blocks"]["d2_levels_5"])
        summaries.append(
            {
                "bundle": bundle_key,
                "target": target,
                "raw_feature_count": len(feature_order),
                "union_selected_count": len(set().union(*selected_sets)),
                "stable_all_folds_count": len(
                    set.intersection(*selected_sets)
                ),
                "d2_union_selected_count": len(
                    set().union(*selected_sets) & d2_features
                ),
                "d2_stable_all_folds_count": len(
                    set.intersection(*selected_sets) & d2_features
                ),
                "d2_levels_union_selected_count": len(
                    set().union(*selected_sets) & level_features
                ),
                "d2_levels_stable_all_folds_count": len(
                    set.intersection(*selected_sets) & level_features
                ),
                "mean_pairwise_selected_set_jaccard": float(
                    np.mean(pairwise_jaccard)
                ),
                "nonzero_coefficient_tolerance": tolerance,
            }
        )
    return pd.DataFrame(records), summaries


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
    folds: pd.DataFrame,
    gate: pd.DataFrame,
    stability_summary: pd.DataFrame,
    *,
    protocol: Mapping[str, Any],
    d5_targets: Sequence[str],
    d5_skip_reason: str | None,
) -> str:
    primary_keys = (
        "D1_vs_D0",
        "D3_vs_D0",
        "D4_vs_D0",
        "D4_vs_D3",
        "D3_vs_C-D3-L20",
        "D3_vs_C-D3-WS",
        "D5_vs_D0",
        "D5_vs_D3",
    )
    primary = comparisons[
        comparisons["comparison"].isin(primary_keys)
    ].copy()
    fold_primary = folds[
        folds["comparison"].isin(
            ("D3_vs_D0", "D3_vs_C-D3-L20", "D3_vs_C-D3-WS")
        )
    ].copy()
    d5_note = (
        "D5 was compared only for validation-gated target(s): "
        + ", ".join(d5_targets)
        + "."
        if d5_targets
        else f"D5 was omitted: {d5_skip_reason or 'no eligible target'}."
    )
    any_gate = bool(gate["all_required_gates_passed"].any())
    inference = protocol["inference"]
    lines = [
        "# Deterministic-news v2 final comparison",
        "",
        "Status: **complete exploratory development inference**.",
        "",
        "Negative paired Fisher-z squared-loss deltas favor the candidate.",
        f"Confidence intervals use {int(inference['bootstrap_resamples']):,} "
        "moving-block resamples, with "
        f"{int(inference['moving_block_sessions'])} consecutive sessions per "
        "block. Blocks are sampled separately within each outer fold and retain "
        "all stocks observed on each sampled date.",
        "",
        "## Paired comparisons",
        "",
        *_markdown_table(
            primary,
            (
                "comparison",
                "target",
                "incremental_r2",
                "mean_squared_loss_delta",
                "bootstrap_ci_lower",
                "bootstrap_ci_upper",
                "folds_candidate_better",
            ),
        ),
        "",
        "## D3 fold consistency and falsification controls",
        "",
        *_markdown_table(
            fold_primary,
            (
                "comparison",
                "target",
                "fold",
                "incremental_r2",
                "mean_squared_loss_delta",
                "candidate_better",
            ),
        ),
        "",
        "## Conservative useful-news gate",
        "",
        "D3 must improve on D0, the 20-session-stale D2 control, and the "
        "wrong-stock D2 control with an upper 95% bootstrap bound below zero, "
        "and must beat D0 in at least two of three folds. This exact Boolean "
        "gate is a conservative final-stage operationalization, not a separately "
        "preregistered threshold.",
        "",
        *_markdown_table(gate, tuple(gate.columns)),
        "",
        (
            "At least one target passes every required gate."
            if any_gate
            else "No target passes every required gate."
        ),
        "",
        "## Elastic Net selection stability",
        "",
        *_markdown_table(
            stability_summary,
            (
                "bundle",
                "target",
                "union_selected_count",
                "stable_all_folds_count",
                "d2_union_selected_count",
                "d2_stable_all_folds_count",
                "mean_pairwise_selected_set_jaccard",
            ),
        ),
        "",
        d5_note,
        "",
        "## Claim boundary",
        "",
        "- The V2-D2 matched D43 comparator was not constructable under the "
        "v2 source contract and is not silently replaced; this does not refer "
        "to the completed D2-Normalized feature block.",
        "- D5 is validation gated and its target subset is not a full four-target "
        "model comparison.",
        "- No multiple-testing adjustment is applied across targets or ladder "
        "rungs.",
        "- All dates are previously inspected development periods.",
        "- Retrospective ordinary Massive news lacks historical article versions "
        "and local first-seen timestamps, so these results are not confirmatory "
        "point-in-time evidence.",
        f"- Protocol SHA-256: `{common.sha256_file(common.PROTOCOL_PATH)}`.",
        f"- Bootstrap seed: `{int(inference['seed'])}`.",
        "",
    ]
    return "\n".join(lines)


def _ensure_output_paths(*, overwrite: bool) -> None:
    paths = (OUTPUT_ROOT, TRACKED_ROOT)
    occupied = any(path.exists() and any(path.iterdir()) for path in paths)
    if occupied and not overwrite:
        raise FileExistsError(
            "Final comparison output exists; pass --overwrite to replace it"
        )
    if occupied:
        for path, root in (
            (OUTPUT_ROOT, common.OUTPUT_ROOT),
            (TRACKED_ROOT, common.EXPERIMENT_ROOT),
        ):
            resolved = path.resolve()
            resolved_root = root.resolve()
            if resolved == resolved_root or resolved_root not in resolved.parents:
                raise ValueError(f"Unsafe final-comparison cleanup path: {path}")
            if path.exists():
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
    gate: pd.DataFrame,
    stability: pd.DataFrame,
    stability_summary: Sequence[Mapping[str, Any]],
    report: str,
    d5_targets: Sequence[str],
    d5_skip_reason: str | None,
) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    TRACKED_ROOT.mkdir(parents=True, exist_ok=True)
    heavy: dict[str, pd.DataFrame | object] = {
        "model_metrics.parquet": model_metrics,
        "paired_comparisons.parquet": comparisons,
        "fold_comparisons.parquet": folds,
        "bootstrap_loss_deltas.parquet": bootstrap,
        "decision_gate.parquet": gate,
        "elastic_net_selection_stability.parquet": stability,
        "elastic_net_stability_summary.json": list(stability_summary),
        "comparison_specs.json": [asdict(item) for item in ladder],
    }
    for name, value in heavy.items():
        path = OUTPUT_ROOT / name
        if isinstance(value, pd.DataFrame):
            quant_common.write_parquet(path, value)
        else:
            quant_common.write_json(path, value)
    artifacts = {
        path.name: {
            "path": path.as_posix(),
            "sha256": common.sha256_file(path),
        }
        for path in sorted(OUTPUT_ROOT.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    sources = protocol["source_artifacts"]
    manifest = {
        "experiment_id": "quant-deterministic-news-v2-final-comparison",
        "generated_at_utc": common.utc_now(),
        "protocol_path": common.PROTOCOL_PATH.as_posix(),
        "protocol_sha256": common.sha256_file(common.PROTOCOL_PATH),
        "panel_path": common.PANEL_PATH.as_posix(),
        "panel_sha256": common.sha256_file(common.PANEL_PATH),
        "panel_manifest_path": str(sources["panel_manifest_path"]),
        "panel_manifest_sha256": str(sources["panel_manifest_sha256"]),
        "comparison_script_path": SCRIPT_PATH.as_posix(),
        "comparison_script_sha256": common.sha256_file(SCRIPT_PATH),
        "source_profile": protocol["source_profile"],
        "claim_scope": protocol["claim_scope"],
        "input_bundles": dict(provenance),
        "inference": {
            **dict(protocol["inference"]),
            "fold_contained": True,
            "all_stocks_per_sampled_date": True,
            "loss": "Fisher-z squared error",
            "paired_delta": "candidate minus base",
            "pooling": "stock-day row weighted",
        },
        "d5_targets": list(d5_targets),
        "d5_skip_reason": d5_skip_reason,
        "coefficient_nonzero_tolerance": NONZERO_COEFFICIENT_TOLERANCE,
        "artifacts": artifacts,
    }
    quant_common.write_json(OUTPUT_ROOT / "manifest.json", manifest)
    refs = {
        **artifacts,
        "manifest.json": {
            "path": (OUTPUT_ROOT / "manifest.json").as_posix(),
            "sha256": common.sha256_file(OUTPUT_ROOT / "manifest.json"),
        },
    }
    primary_keys = {
        "D3_vs_D0",
        "D3_vs_C-D3-L20",
        "D3_vs_C-D3-WS",
        "D4_vs_D3",
        "D5_vs_D0",
        "D5_vs_D3",
    }
    summary = {
        "status": "complete",
        "generated_at_utc": common.utc_now(),
        "primary_comparisons": comparisons[
            comparisons["comparison"].isin(primary_keys)
        ].to_dict(orient="records"),
        "decision_gate": gate.to_dict(orient="records"),
        "comparison_target_count": len(comparisons),
        "bootstrap_resample_rows": len(bootstrap),
        "d5_targets": list(d5_targets),
        "d5_skip_reason": d5_skip_reason,
        "elastic_net_stability_summary": list(stability_summary),
    }
    review = {
        "status": "passed",
        "all_input_bundle_manifests_hash_verified": True,
        "all_prediction_artifacts_hash_verified": True,
        "d3_d4_fit_artifacts_hash_verified": True,
        "prediction_keys_exactly_matched_in_every_comparison": True,
        "actual_targets_exactly_matched_in_every_comparison": True,
        "all_stocks_retained_per_forecast_date": True,
        "bootstrap_fold_contained": True,
        "bootstrap_resamples": int(
            protocol["inference"]["bootstrap_resamples"]
        ),
        "moving_block_sessions": int(
            protocol["inference"]["moving_block_sessions"]
        ),
        "bootstrap_seed": int(protocol["inference"]["seed"]),
        "multiple_testing_adjustment": None,
        "decision_gate_exact_boolean_threshold_prelocked": False,
        "decision_gate_any_target_passed": bool(
            gate["all_required_gates_passed"].any()
        ),
        "development_only_non_version_safe": True,
    }
    quant_common.write_json(TRACKED_ROOT / "summary.json", summary)
    quant_common.write_json(TRACKED_ROOT / "review.json", review)
    quant_common.write_json(TRACKED_ROOT / "artifact_refs.json", refs)
    temporary = (TRACKED_ROOT / "RESULTS.md").with_suffix(".md.tmp")
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(TRACKED_ROOT / "RESULTS.md")


def parser() -> argparse.ArgumentParser:
    output = argparse.ArgumentParser(description=__doc__)
    output.add_argument("--allow-exploratory", action="store_true")
    output.add_argument("--overwrite", action="store_true")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not args.allow_exploratory:
        raise ValueError("Pass --allow-exploratory for this news archive")
    protocol = common.load_protocol()
    common.load_panel_and_preflight(
        protocol,
        allow_exploratory=True,
        required_bundle_keys=REQUIRED_BUNDLES,
    )
    _ensure_output_paths(overwrite=args.overwrite)
    predictions, provenance, fits, d5_skip_reason = _load_inputs(protocol)
    d5_targets = (
        tuple(sorted(predictions["D5"]["target"].unique().tolist()))
        if "D5" in predictions
        else ()
    )
    ladder = comparison_ladder(d5_targets)
    inference = protocol["inference"]
    block_sessions = int(inference["moving_block_sessions"])
    resamples = int(inference["bootstrap_resamples"])
    rng = np.random.default_rng(int(inference["seed"]))
    pooled_records: list[dict[str, Any]] = []
    fold_records: list[dict[str, Any]] = []
    bootstrap_frames: list[pd.DataFrame] = []
    for comparison in ladder:
        merged = _merge_pair(comparison, predictions)
        pooled, fold_rows, samples = _comparison_records(
            comparison,
            merged,
            block_sessions=block_sessions,
            resamples=resamples,
            rng=rng,
        )
        pooled_records.extend(pooled)
        fold_records.extend(fold_rows)
        bootstrap_frames.append(samples)
    comparison_frame = pd.DataFrame(pooled_records)
    fold_frame = pd.DataFrame(fold_records)
    bootstrap_frame = pd.concat(bootstrap_frames, ignore_index=True)
    gate = _decision_gate(comparison_frame)
    stability_frames: list[pd.DataFrame] = []
    stability_summary: list[dict[str, Any]] = []
    for bundle_key in ELASTIC_NET_STABILITY_BUNDLES:
        frame, summary = _selection_stability(
            bundle_key, fits[bundle_key], protocol
        )
        stability_frames.append(frame)
        stability_summary.extend(summary)
    stability_frame = pd.concat(stability_frames, ignore_index=True)
    stability_summary_frame = pd.DataFrame(stability_summary)
    model_metrics = _model_metrics(predictions)
    if len(bootstrap_frame) != len(comparison_frame) * resamples:
        raise AssertionError("Bootstrap output count is incomplete")
    required_finite = comparison_frame[
        [
            "incremental_r2",
            "mean_squared_loss_delta",
            "bootstrap_ci_lower",
            "bootstrap_ci_upper",
        ]
    ].to_numpy(dtype=float)
    if comparison_frame.empty or not np.isfinite(required_finite).all():
        raise AssertionError("Final comparison metrics are empty or nonfinite")
    report = _render_report(
        comparison_frame,
        fold_frame,
        gate,
        stability_summary_frame,
        protocol=protocol,
        d5_targets=d5_targets,
        d5_skip_reason=d5_skip_reason,
    )
    _write_outputs(
        protocol=protocol,
        provenance=provenance,
        ladder=ladder,
        model_metrics=model_metrics,
        comparisons=comparison_frame,
        folds=fold_frame,
        bootstrap=bootstrap_frame,
        gate=gate,
        stability=stability_frame,
        stability_summary=stability_summary,
        report=report,
        d5_targets=d5_targets,
        d5_skip_reason=d5_skip_reason,
    )
    print(
        comparison_frame[
            comparison_frame["comparison"].isin(
                ("D3_vs_D0", "D3_vs_C-D3-L20", "D3_vs_C-D3-WS")
            )
        ].to_string(index=False)
    )
    print(gate.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
