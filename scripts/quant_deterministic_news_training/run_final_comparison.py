"""Build the locked Q+D comparison tables and date-block inference bundle."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from scripts.quant_deterministic_news_training import common


MODEL_BUNDLES = (
    "A0-L",
    "A0-T",
    "A1",
    "A2",
    "A3",
    "A4",
    "A5",
    "A6",
    "A7",
    "B0",
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "S-B0",
    "S-B1",
    "S-B2",
    "S-B3",
    "S-B4",
    "S-B5",
    "C-A5-L20",
    "C-A5-WS",
    "C-B4-L20",
    "C-B4-WS",
)


@dataclass(frozen=True)
class Comparison:
    key: str
    candidate: str
    base: str
    family: str
    folds: tuple[str, ...] | None = None
    targets: tuple[str, ...] | None = None
    controlled: bool = True


def comparison_ladder(
    a7_targets: Sequence[str] | None = None,
) -> tuple[Comparison, ...]:
    values: list[Comparison] = []
    for candidate in ("A1", "A2", "A3", "A4", "A5"):
        values.append(
            Comparison(
                f"{candidate}_vs_A0-L",
                candidate,
                "A0-L",
                "track_a_matched_elastic_net",
            )
        )
    for candidate, base in (
        ("A3", "A2"),
        ("A4", "A3"),
        ("A5", "A4"),
    ):
        values.append(
            Comparison(
                f"{candidate}_vs_{base}",
                candidate,
                base,
                "track_a_nested_news_block",
            )
        )
    values.append(
        Comparison(
            "A6_vs_A0-T",
            "A6",
            "A0-T",
            "track_a_matched_xgboost",
        )
    )
    included_a7 = tuple(a7_targets or ())
    if included_a7:
        for base in ("A0-L", "A0-T", "A5", "A6"):
            values.append(
                Comparison(
                    f"A7_vs_{base}",
                    "A7",
                    base,
                    "track_a_validation_gated_ensemble",
                    targets=included_a7,
                )
            )
    for candidate in ("B1", "B2", "B3", "B4", "B5"):
        values.append(
            Comparison(
                f"{candidate}_vs_B0",
                candidate,
                "B0",
                "track_b_primary",
            )
        )
    for candidate in ("B3", "B4", "B5"):
        for base in ("B1", "B2"):
            values.append(
                Comparison(
                    f"{candidate}_vs_{base}",
                    candidate,
                    base,
                    "track_b_vs_nonnews_calibration",
                )
            )
    for candidate in ("S-B1", "S-B2", "S-B3", "S-B4", "S-B5"):
        values.append(
            Comparison(
                f"{candidate}_vs_S-B0",
                candidate,
                "S-B0",
                "t1_etf_winner_base_sensitivity",
                targets=("t1_etf",),
            )
        )
    for control in ("C-A5-L20", "C-A5-WS"):
        values.append(
            Comparison(
                f"A5_vs_{control}",
                "A5",
                control,
                "track_a_placebo",
            )
        )
    for control in ("C-B4-L20", "C-B4-WS"):
        values.append(
            Comparison(
                f"B4_vs_{control}",
                "B4",
                control,
                "track_b_placebo",
            )
        )
    values.extend(
        [
            Comparison(
                "A5_vs_B4_common_folds",
                "A5",
                "B4",
                "joint_vs_residual_uncontrolled",
                folds=("fold_2", "fold_3"),
                controlled=False,
            ),
            Comparison(
                "A6_vs_B5_common_folds",
                "A6",
                "B5",
                "joint_vs_residual_uncontrolled",
                folds=("fold_2", "fold_3"),
                controlled=False,
            ),
        ]
    )
    if "t1_etf" in included_a7:
        values.append(
            Comparison(
                "A7_vs_S-B0_common_folds",
                "A7",
                "S-B0",
                "gated_joint_vs_quant_v1_winner",
                folds=("fold_2", "fold_3"),
                targets=("t1_etf",),
                controlled=False,
            )
        )
    keys = [item.key for item in values]
    if len(keys) != len(set(keys)):
        raise AssertionError("Comparison keys are not unique")
    return tuple(values)


def _load_predictions() -> tuple[
    dict[str, pd.DataFrame], dict[str, dict[str, Any]]
]:
    predictions: dict[str, pd.DataFrame] = {}
    provenance: dict[str, dict[str, Any]] = {}
    for bundle_key in MODEL_BUNDLES:
        if bundle_key == "A7":
            bundle_provenance = common.verify_completed_bundle(
                bundle_key, allowed_states=("complete", "skipped")
            )
            if bundle_provenance["state"] == "skipped":
                bundle_provenance.pop("manifest")
                provenance[bundle_key] = bundle_provenance
                continue
        path, artifact = common.verified_bundle_artifact(
            bundle_key, "predictions.parquet"
        )
        frame = pd.read_parquet(path)
        frame["forecast_date"] = pd.to_datetime(
            frame["forecast_date"]
        ).dt.normalize()
        if frame.duplicated(common.PREDICTION_KEYS).any():
            raise ValueError(f"{bundle_key} has duplicate prediction keys")
        predictions[bundle_key] = frame
        provenance[bundle_key] = artifact
    return predictions, provenance


def _merge_pair(
    comparison: Comparison,
    predictions: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    candidate = predictions[comparison.candidate].copy()
    base = predictions[comparison.base].copy()
    if comparison.folds is not None:
        candidate = candidate[candidate["fold"].isin(comparison.folds)]
        base = base[base["fold"].isin(comparison.folds)]
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
        raise ValueError(f"{comparison.key} keys differ: {counts}")
    for column in ("actual_fisher_z", "actual_correlation"):
        delta = (
            merged[f"{column}_candidate"] - merged[f"{column}_base"]
        ).abs()
        if delta.isna().any() or float(delta.max()) > 1e-12:
            raise ValueError(f"{comparison.key} actuals differ")
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
    return merged


def _moving_block_samples(
    frame: pd.DataFrame,
    *,
    block_sessions: int,
    resamples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return pooled row-mean loss deltas from fold-contained date blocks."""

    sampled_sums = np.zeros(resamples, dtype=float)
    sampled_counts = np.zeros(resamples, dtype=float)
    for _, fold_frame in frame.groupby("fold", sort=True):
        dates = np.asarray(
            sorted(fold_frame["forecast_date"].unique()),
            dtype="datetime64[ns]",
        )
        if len(dates) < block_sessions:
            raise ValueError("Fold contains fewer dates than one block")
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
        sampled_counts += by_date["count"].to_numpy(dtype=float)[
            indices
        ].sum(axis=1)
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
        fold_deltas = []
        for fold, fold_frame in target_frame.groupby("fold", sort=True):
            fold_candidate = float(
                fold_frame["candidate_squared_loss"].sum()
            )
            fold_base = float(fold_frame["base_squared_loss"].sum())
            fold_delta = float(fold_frame["squared_loss_delta"].mean())
            fold_deltas.append(fold_delta)
            fold_records.append(
                {
                    "comparison": comparison.key,
                    "family": comparison.family,
                    "candidate": comparison.candidate,
                    "base": comparison.base,
                    "controlled": comparison.controlled,
                    "target": target,
                    "fold": fold,
                    "rows": len(fold_frame),
                    "dates": fold_frame["forecast_date"].nunique(),
                    "candidate_fisher_z_rmse": math.sqrt(
                        fold_candidate / len(fold_frame)
                    ),
                    "base_fisher_z_rmse": math.sqrt(
                        fold_base / len(fold_frame)
                    ),
                    "incremental_r2": 1 - fold_candidate / fold_base,
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
                "controlled": comparison.controlled,
                "target": target,
                "rows": len(target_frame),
                "dates": target_frame["forecast_date"].nunique(),
                "folds": ",".join(
                    sorted(target_frame["fold"].unique().tolist())
                ),
                "candidate_fisher_z_rmse": math.sqrt(
                    candidate_sse / len(target_frame)
                ),
                "base_fisher_z_rmse": math.sqrt(
                    base_sse / len(target_frame)
                ),
                "incremental_r2": 1 - candidate_sse / base_sse,
                "mean_squared_loss_delta": observed_delta,
                "bootstrap_ci_lower": float(lower),
                "bootstrap_ci_upper": float(upper),
                "bootstrap_probability_candidate_better": float(
                    np.mean(samples < 0)
                ),
                "folds_candidate_better": int(
                    sum(value < 0 for value in fold_deltas)
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
    predictions: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    records = []
    for bundle_key in MODEL_BUNDLES:
        if bundle_key not in predictions:
            continue
        for record in common.metrics_records(predictions[bundle_key]):
            records.append({"bundle": bundle_key, **record})
    return pd.DataFrame(records).sort_values(
        ["bundle", "target"]
    ).reset_index(drop=True)


SLICE_COMPARISONS = (
    "A5_vs_A0-L",
    "A6_vs_A0-T",
    "B4_vs_B0",
    "B5_vs_B0",
    "S-B5_vs_S-B0",
)


def _slice_records(
    merged_by_comparison: dict[str, pd.DataFrame],
    panel: pd.DataFrame,
) -> pd.DataFrame:
    news = panel[
        [*common.PANEL_KEYS, "observed_no_relevant_news"]
    ].copy()
    records = []
    for key in SLICE_COMPARISONS:
        merged = merged_by_comparison[key].merge(
            news,
            on=common.PANEL_KEYS,
            how="left",
            validate="many_to_one",
        )
        if merged["observed_no_relevant_news"].isna().any():
            raise ValueError(f"{key} misses news-slice labels")
        merged["news_slice"] = np.where(
            merged["observed_no_relevant_news"].eq(1),
            "observed_no_relevant_news",
            "observed_relevant_news",
        )
        for (target, news_slice), frame in merged.groupby(
            ["target", "news_slice"], sort=True
        ):
            candidate_sse = float(frame["candidate_squared_loss"].sum())
            base_sse = float(frame["base_squared_loss"].sum())
            fold_row_counts = {
                str(fold): int(count)
                for fold, count in frame.groupby("fold", sort=True).size().items()
            }
            records.append(
                {
                    "comparison": key,
                    "target": target,
                    "news_slice": news_slice,
                    "rows": len(frame),
                    "dates": frame["forecast_date"].nunique(),
                    "folds_present": ",".join(fold_row_counts),
                    "fold_count": len(fold_row_counts),
                    "fold_row_counts": json.dumps(
                        fold_row_counts, sort_keys=True
                    ),
                    "candidate_fisher_z_rmse": math.sqrt(
                        candidate_sse / len(frame)
                    ),
                    "base_fisher_z_rmse": math.sqrt(
                        base_sse / len(frame)
                    ),
                    "incremental_r2": 1 - candidate_sse / base_sse,
                    "mean_squared_loss_delta": float(
                        frame["squared_loss_delta"].mean()
                    ),
                }
            )
    return pd.DataFrame(records)


def _decision_gate(comparisons: pd.DataFrame) -> pd.DataFrame:
    """Operationalize the protocol's qualitative useful-news requirements."""

    indexed = comparisons.set_index(["comparison", "target"])

    def passed(key: str, target: str) -> bool:
        row = indexed.loc[(key, target)]
        return bool(
            row["mean_squared_loss_delta"] < 0
            and row["bootstrap_ci_upper"] < 0
        )

    records = []
    targets = ("t1_etf", "t1_loo", "t2_etf", "t2_loo")
    for target in targets:
        matched = indexed.loc[("A5_vs_A0-L", target)]
        conditions = {
            "matched_quant_improvement_ci": passed(
                "A5_vs_A0-L", target
            ),
            "beats_stale_placebo_ci": passed(
                "A5_vs_C-A5-L20", target
            ),
            "beats_wrong_stock_placebo_ci": passed(
                "A5_vs_C-A5-WS", target
            ),
            "improves_at_least_two_of_three_folds": int(
                matched["folds_candidate_better"]
            )
            >= 2,
        }
        records.append(
            {
                "candidate": "A5",
                "target": target,
                **conditions,
                "all_required_gates_passed": all(conditions.values()),
            }
        )
    for target in targets:
        matched = indexed.loc[("B4_vs_B0", target)]
        conditions = {
            "matched_quant_improvement_ci": passed("B4_vs_B0", target),
            "beats_mean_error_calibration_ci": passed(
                "B4_vs_B1", target
            ),
            "beats_linear_calibration_ci": passed(
                "B4_vs_B2", target
            ),
            "beats_stale_placebo_ci": passed(
                "B4_vs_C-B4-L20", target
            ),
            "beats_wrong_stock_placebo_ci": passed(
                "B4_vs_C-B4-WS", target
            ),
            "improves_both_outer_folds": int(
                matched["folds_candidate_better"]
            )
            == 2,
        }
        records.append(
            {
                "candidate": "B4",
                "target": target,
                **conditions,
                "all_required_gates_passed": all(conditions.values()),
            }
        )
    return pd.DataFrame(records)


def _markdown_table(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    digits: int = 6,
) -> list[str]:
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join("---:" for _ in columns) + "|"
    lines = [header, separator]
    for row in frame[list(columns)].itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.{digits}f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _render_final_report(
    model_metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
    slices: pd.DataFrame,
    decision_gate: pd.DataFrame,
    claim: str,
    protocol: dict[str, Any],
) -> str:
    primary_keys = [
        "A5_vs_A0-L",
        "A6_vs_A0-T",
        "B4_vs_B0",
        "B5_vs_B0",
        "S-B5_vs_S-B0",
        "A5_vs_C-A5-L20",
        "A5_vs_C-A5-WS",
        "B4_vs_C-B4-L20",
        "B4_vs_C-B4-WS",
    ]
    primary = comparisons[
        comparisons["comparison"].isin(primary_keys)
    ].copy()
    lines = [
        "# Quant plus deterministic-news v1 results",
        "",
        "Status: **complete exploratory development experiment**.",
        "",
        claim,
        "",
        "Negative paired loss deltas favor the candidate. Confidence intervals",
        f"use {protocol['inference']['bootstrap_resamples']:,} moving-block "
        "resamples of whole dates, "
        f"{protocol['inference']['bootstrap_block_sessions']} sessions per "
        "block, constructed separately inside each outer fold.",
        "Pooled estimates are stock-day-row weighted; because every eligible",
        "target-date has 30 stocks, this is equivalent to equal date weighting",
        "with fold weights proportional to each fold's eligible date count.",
        "",
        "## Primary and placebo comparisons",
        "",
        *_markdown_table(
            primary[
                [
                    "comparison",
                    "target",
                    "incremental_r2",
                    "mean_squared_loss_delta",
                    "bootstrap_ci_lower",
                    "bootstrap_ci_upper",
                    "folds_candidate_better",
                    "fold_count",
                ]
            ],
            [
                "comparison",
                "target",
                "incremental_r2",
                "mean_squared_loss_delta",
                "bootstrap_ci_lower",
                "bootstrap_ci_upper",
                "folds_candidate_better",
                "fold_count",
            ],
        ),
        "",
        "## Useful-news decision gate",
        "",
        "This gate is a conservative final-stage operationalization of the",
        "locked protocol's qualitative requirement to beat the matched quant",
        "base, calibration/placebo controls, and more than one fold. The exact",
        "boolean threshold was not separately preregistered.",
        "",
        *_markdown_table(
            decision_gate,
            list(decision_gate.columns),
        ),
        "",
        "## Model metrics",
        "",
        *_markdown_table(
            model_metrics[
                [
                    "bundle",
                    "target",
                    "fisher_z_rmse",
                    "fisher_z_mae",
                    "oos_r2_vs_persistence",
                ]
            ],
            [
                "bundle",
                "target",
                "fisher_z_rmse",
                "fisher_z_mae",
                "oos_r2_vs_persistence",
            ],
        ),
        "",
        "## News/no-news slices",
        "",
        *_markdown_table(
            slices[
                [
                    "comparison",
                    "target",
                    "news_slice",
                    "rows",
                    "folds_present",
                    "fold_row_counts",
                    "incremental_r2",
                    "mean_squared_loss_delta",
                ]
            ],
            [
                "comparison",
                "target",
                "news_slice",
                "rows",
                "folds_present",
                "fold_row_counts",
                "incremental_r2",
                "mean_squared_loss_delta",
            ],
        ),
        "",
        "The observed-no-news slice is extremely small, has no fold-3 rows,",
        "and is descriptive only; it is not a stable across-fold sensitivity.",
        "",
        "## Interpretation",
        "",
        "- Track A and Track B are not a controlled architecture comparison;",
        "  Track B can train only on earlier saved out-of-sample residuals.",
        "- The stale-news and wrong-stock comparisons are required falsification",
        "  checks, not alternative production models.",
        "- No multiple-testing adjustment is applied across the ladder.",
        "- All results are development estimates because quant-v1 outer blocks",
        "  were previously inspected and the retrospective news archive lacks",
        "  historical article versions and first-seen timestamps.",
        f"- Protocol SHA-256: `{common.sha256_file(common.PROTOCOL_PATH)}`.",
        f"- Bootstrap seed: `{protocol['inference']['bootstrap_seed']}`.",
        "",
    ]
    return "\n".join(lines)


def parser() -> argparse.ArgumentParser:
    output = argparse.ArgumentParser(description=__doc__)
    output.add_argument("--allow-exploratory", action="store_true")
    output.add_argument("--overwrite", action="store_true")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not args.allow_exploratory:
        raise ValueError("Pass --allow-exploratory for this news archive")
    common.require_completed_preflight()
    common.ensure_no_existing_bundle("FINAL", overwrite=args.overwrite)
    common.update_status("FINAL", "running")
    try:
        protocol = common.load_protocol()
        panel, _ = common.load_panel_and_preflight(
            protocol, allow_exploratory=True
        )
        predictions, provenance = _load_predictions()
        a7_targets = (
            sorted(predictions["A7"]["target"].unique().tolist())
            if "A7" in predictions
            else []
        )
        ladder = comparison_ladder(a7_targets)
        model_metrics = _model_metrics(predictions)
        block_sessions = int(
            protocol["inference"]["bootstrap_block_sessions"]
        )
        resamples = int(protocol["inference"]["bootstrap_resamples"])
        rng = np.random.default_rng(
            int(protocol["inference"]["bootstrap_seed"])
        )
        pooled_records: list[dict[str, Any]] = []
        fold_records: list[dict[str, Any]] = []
        bootstrap_frames: list[pd.DataFrame] = []
        merged_by_comparison: dict[str, pd.DataFrame] = {}
        for comparison in ladder:
            merged = _merge_pair(comparison, predictions)
            merged_by_comparison[comparison.key] = merged
            pooled, folds, samples = _comparison_records(
                comparison,
                merged,
                block_sessions=block_sessions,
                resamples=resamples,
                rng=rng,
            )
            pooled_records.extend(pooled)
            fold_records.extend(folds)
            bootstrap_frames.append(samples)
        comparison_frame = pd.DataFrame(pooled_records)
        fold_frame = pd.DataFrame(fold_records)
        bootstrap_frame = pd.concat(bootstrap_frames, ignore_index=True)
        slice_frame = _slice_records(merged_by_comparison, panel)
        decision_gate = _decision_gate(comparison_frame)
        any_gate_passed = bool(
            decision_gate["all_required_gates_passed"].any()
        )
        claim = (
            "At least one of the two placebo-tested linear specifications "
            "(A5/B4) passes the conservative final-stage useful-news gate."
            if any_gate_passed
            else (
                "Neither of the two placebo-tested linear specifications "
                "(A5/B4) passes the conservative final-stage matched-base, "
                "calibration/placebo, and fold-consistency gate. Nonlinear "
                "A6/B5 and the winner-base sensitivity remain descriptive "
                "because matching nonlinear placebo fits were not "
                "preregistered."
            )
        )
        if comparison_frame.empty or fold_frame.empty or slice_frame.empty:
            raise AssertionError("Final comparison outputs are empty")
        if len(bootstrap_frame) != len(comparison_frame) * resamples:
            raise AssertionError("Bootstrap output count is incomplete")
        if not np.isfinite(
            comparison_frame[
                [
                    "incremental_r2",
                    "mean_squared_loss_delta",
                    "bootstrap_ci_lower",
                    "bootstrap_ci_upper",
                ]
            ].to_numpy(dtype=float)
        ).all():
            raise AssertionError("Final comparison contains nonfinite metrics")

        key_rows = comparison_frame[
            comparison_frame["comparison"].isin(
                [
                    "A5_vs_A0-L",
                    "A6_vs_A0-T",
                    "B4_vs_B0",
                    "B5_vs_B0",
                    "S-B5_vs_S-B0",
                ]
            )
        ].copy()
        metrics = model_metrics.to_dict(orient="records")
        summary = {
            "status": "complete",
            "metrics": metrics,
            "primary_comparisons": key_rows.to_dict(orient="records"),
            "decision_gate": decision_gate.to_dict(orient="records"),
            "comparison_count": len(comparison_frame),
            "bootstrap_resample_rows": len(bootstrap_frame),
            "claim": claim,
        }
        review = {
            "status": "passed",
            "prediction_keys_unique": True,
            "predictions_finite": True,
            "prediction_bounds_valid": True,
            "all_model_bundles_hash_verified": True,
            "fold_contained_date_blocks": True,
            "all_stocks_retained_per_sampled_date": True,
            "bootstrap_resamples": resamples,
            "bootstrap_block_sessions": block_sessions,
            "bootstrap_seed": int(
                protocol["inference"]["bootstrap_seed"]
            ),
            "track_a_vs_track_b_labeled_uncontrolled": True,
            "multiple_testing_adjustment": None,
            "pooling": (
                "stock-day-row weighted; equivalent to equal date weighting "
                "because every eligible target-date has 30 stocks; fold "
                "weights are proportional to eligible date counts"
            ),
            "decision_gate_exact_boolean_threshold_prelocked": False,
            "decision_gate_any_passed": any_gate_passed,
        }
        report = _render_final_report(
            model_metrics,
            comparison_frame,
            slice_frame,
            decision_gate,
            claim,
            protocol,
        )
        common.write_bundle(
            "FINAL",
            predictions=None,
            validation_predictions=None,
            fits=[item.__dict__ for item in ladder],
            fold_metrics=fold_frame.to_dict(orient="records"),
            summary=summary,
            review=review,
            model_config={
                "comparison_ladder": [
                    item.__dict__ for item in ladder
                ],
                "inference": protocol["inference"],
                "pooling": (
                    "stock-day-row weighted; fold weights proportional to "
                    "eligible date counts"
                ),
                "slice_comparisons": list(SLICE_COMPARISONS),
            },
            dependencies=provenance,
            extra_outputs={
                "model_metrics.parquet": model_metrics,
                "paired_comparisons.parquet": comparison_frame,
                "fold_comparisons.parquet": fold_frame,
                "bootstrap_loss_deltas.parquet": bootstrap_frame,
                "news_slice_comparisons.parquet": slice_frame,
                "decision_gate.parquet": decision_gate,
            },
            results_markdown=report,
        )
        common.update_status(
            "FINAL",
            "complete",
            summary=(
                f"{len(comparison_frame)} paired target comparisons; "
                f"{len(bootstrap_frame):,} block-bootstrap draws"
            ),
        )
        print(key_rows.to_string(index=False))
        print(json.dumps(review, indent=2))
        return 0
    except Exception as error:
        common.update_status("FINAL", "failed", summary=str(error))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
