#!/usr/bin/env python
"""Compare completed v4 rungs with paired whole-date moving bootstrap."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from scripts.quant_deterministic_news_training_v4 import common, contract


OUTPUT_ROOT = contract.OUTPUT_ROOT / "comparisons" / "final"
TRACKED_ROOT = contract.EXPERIMENT_ROOT / "comparisons" / "final"


@dataclass(frozen=True)
class Comparison:
    key: str
    candidate: str
    base: str
    family: str


COMPARISONS = (
    Comparison("J1_vs_v3_S0", "J1", "v3:S0", "short_semantic_increment"),
    Comparison("J2_vs_v3_S1", "J2", "v3:S1", "short_semantic_beyond_d2"),
    Comparison("J3_vs_v3_S0", "J3", "v3:S0", "short_current_mass_increment"),
    Comparison("J1_vs_stale20", "J1", "C-J1-L20", "short_stale_falsification"),
    Comparison("J2_vs_stale20", "J2", "C-J2-L20", "short_stale_falsification"),
    Comparison("J1_vs_wrong_stock", "J1", "C-J1-WS", "short_wrong_stock_falsification"),
    Comparison("J2_vs_wrong_stock", "J2", "C-J2-WS", "short_wrong_stock_falsification"),
    Comparison("J1_vs_permutation", "J1", "C-J1-PERM", "short_permutation_falsification"),
    Comparison("J2_vs_permutation", "J2", "C-J2-PERM", "short_permutation_falsification"),
    Comparison("J1_vs_quality", "J1", "C-J1-QUALITY", "short_quality_falsification"),
    Comparison("J2_vs_quality", "J2", "C-J2-QUALITY", "short_quality_falsification"),
    Comparison("RCAL_vs_R0", "RCAL", "R0", "long_base_calibration"),
    Comparison("RRES_C6_vs_R0", "RRES-C6", "R0", "long_residual_coupling"),
    Comparison("RRES_C6_vs_RCAL", "RRES-C6", "RCAL", "long_residual_coupling_beyond_calibration"),
    Comparison("RRES_L19_vs_R0", "RRES-L19", "R0", "long_residual_semantic"),
    Comparison("RRES_L19_vs_RCAL", "RRES-L19", "RCAL", "long_residual_semantic_beyond_calibration"),
    Comparison("RRES_L19_vs_C6", "RRES-L19", "RRES-C6", "long_full_vs_coupling"),
    Comparison("RSTACK_vs_R0", "RSTACK", "R0", "long_semantic_stack"),
    Comparison("RSTACK_vs_RCAL", "RSTACK", "RCAL", "long_semantic_stack_beyond_calibration"),
    Comparison("RRES_L19_vs_stale20", "RRES-L19", "C-RRES-L19-L20", "long_stale_falsification"),
    Comparison("RRES_L19_vs_wrong_stock", "RRES-L19", "C-RRES-L19-WS", "long_wrong_stock_falsification"),
    Comparison("RRES_L19_vs_permutation", "RRES-L19", "C-RRES-L19-PERM", "long_permutation_falsification"),
    Comparison("RRES_L19_vs_quality", "RRES-L19", "C-RRES-L19-QUALITY", "long_quality_falsification"),
)


USEFUL_GATE_REQUIREMENTS = {
    "J1": (
        "J1_vs_v3_S0",
        "J1_vs_stale20",
        "J1_vs_wrong_stock",
        "J1_vs_permutation",
        "J1_vs_quality",
    ),
    "J2": (
        "J2_vs_v3_S1",
        "J2_vs_stale20",
        "J2_vs_wrong_stock",
        "J2_vs_permutation",
        "J2_vs_quality",
    ),
    "RRES-L19": (
        "RRES_L19_vs_R0",
        "RRES_L19_vs_RCAL",
        "RRES_L19_vs_stale20",
        "RRES_L19_vs_wrong_stock",
        "RRES_L19_vs_permutation",
        "RRES_L19_vs_quality",
    ),
    "RSTACK": (
        "RSTACK_vs_R0",
        "RSTACK_vs_RCAL",
    ),
}

USEFUL_GATE_CONTROL_SET_COMPLETE = {
    "J1": True,
    "J2": True,
    "RRES-L19": True,
    # RSTACK has base comparisons but no architecture-matched stale,
    # wrong-stock, permutation, or quality stack controls.
    "RSTACK": False,
}


def moving_block_positions(
    n_dates: int,
    *,
    block_sessions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if n_dates <= 0:
        raise ValueError("n_dates must be positive")
    if not 1 <= block_sessions <= n_dates:
        raise ValueError("block_sessions must be within the date sample")
    maximum_start = n_dates - block_sessions
    values: list[int] = []
    while len(values) < n_dates:
        start = int(rng.integers(0, maximum_start + 1))
        values.extend(range(start, start + block_sessions))
    return np.asarray(values[:n_dates], dtype=int)


def paired_date_losses(
    candidate: pd.DataFrame,
    base: pd.DataFrame,
    *,
    target: str,
) -> pd.DataFrame:
    keys = list(contract.PREDICTION_KEYS)
    left = candidate[candidate["target"].eq(target)]
    right = base[base["target"].eq(target)]
    merged = left.merge(
        right,
        on=keys,
        suffixes=("_candidate", "_base"),
        validate="one_to_one",
    )
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError(f"Unmatched prediction rows for {target}")
    actual_gap = np.max(
        np.abs(
            merged["actual_fisher_z_candidate"].to_numpy(dtype=float)
            - merged["actual_fisher_z_base"].to_numpy(dtype=float)
        )
    )
    if actual_gap > 1e-12:
        raise ValueError(f"Actual target mismatch for {target}")
    merged["candidate_loss"] = np.square(
        merged["actual_fisher_z_candidate"]
        - merged["predicted_fisher_z_candidate"]
    )
    merged["base_loss"] = np.square(
        merged["actual_fisher_z_base"] - merged["predicted_fisher_z_base"]
    )
    daily = (
        merged.groupby(["fold", "forecast_date"], sort=True)
        .agg(
            rows=("stock", "size"),
            candidate_sse=("candidate_loss", "sum"),
            base_sse=("base_loss", "sum"),
        )
        .reset_index()
    )
    if daily["rows"].nunique() != 1:
        raise ValueError(f"Whole-date cross-sectional row count varies for {target}")
    return daily


def paired_fold_improvements(
    candidate: pd.DataFrame,
    base: pd.DataFrame,
    *,
    target: str,
    family: str,
) -> dict[str, Any]:
    """Count outer folds whose pooled squared loss favors the candidate."""

    keys = list(contract.PREDICTION_KEYS)
    left = candidate[candidate["target"].eq(target)]
    right = base[base["target"].eq(target)]
    merged = left.merge(
        right,
        on=keys,
        suffixes=("_candidate", "_base"),
        validate="one_to_one",
    )
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError(f"Unmatched prediction rows for {target}")
    merged["candidate_loss"] = np.square(
        merged["actual_fisher_z_candidate"]
        - merged["predicted_fisher_z_candidate"]
    )
    merged["base_loss"] = np.square(
        merged["actual_fisher_z_base"]
        - merged["predicted_fisher_z_base"]
    )
    folds = (
        merged.groupby("fold", sort=True)
        .agg(
            rows=("stock", "size"),
            candidate_sse=("candidate_loss", "sum"),
            base_sse=("base_loss", "sum"),
        )
        .reset_index()
    )
    expected_folds = 3 if family.startswith("short_") else 5
    required = 2 if expected_folds == 3 else 3
    if len(folds) != expected_folds:
        raise ValueError(
            f"{family}/{target} has {len(folds)} folds; expected {expected_folds}"
        )
    folds["candidate_improved"] = folds["candidate_sse"] < folds["base_sse"]
    improved = int(folds["candidate_improved"].sum())
    return {
        "folds_total": int(len(folds)),
        "folds_improved": improved,
        "fold_improvement_requirement": required,
        "fold_improvement_gate_pass": bool(improved >= required),
    }


def paired_bootstrap(
    daily: pd.DataFrame,
    *,
    resamples: int = 2000,
    block_sessions: int = 10,
    seed: int = 1729,
) -> dict[str, Any]:
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    candidate_sse = daily["candidate_sse"].to_numpy(dtype=float)
    base_sse = daily["base_sse"].to_numpy(dtype=float)
    rows = daily["rows"].to_numpy(dtype=float)
    denominator = float(base_sse.sum())
    if denominator <= 0:
        raise ValueError("Base SSE must be positive")
    point_r2 = float(1 - candidate_sse.sum() / denominator)
    point_delta = float((candidate_sse.sum() - base_sse.sum()) / rows.sum())
    rng = np.random.default_rng(seed)
    boot_r2 = np.empty(resamples, dtype=float)
    boot_delta = np.empty(resamples, dtype=float)
    if "fold" not in daily:
        raise ValueError("Daily paired losses must retain outer-fold identity")
    fold_positions = [
        group.index.to_numpy(dtype=int)
        for _, group in daily.reset_index(drop=True).groupby("fold", sort=True)
    ]
    if any(len(positions) < block_sessions for positions in fold_positions):
        raise ValueError("block_sessions exceeds at least one outer fold")
    for index in range(resamples):
        positions = np.concatenate(
            [
                fold_index[
                    moving_block_positions(
                        len(fold_index),
                        block_sessions=block_sessions,
                        rng=rng,
                    )
                ]
                for fold_index in fold_positions
            ]
        )
        sampled_candidate = float(candidate_sse[positions].sum())
        sampled_base = float(base_sse[positions].sum())
        sampled_rows = float(rows[positions].sum())
        boot_r2[index] = 1 - sampled_candidate / sampled_base
        boot_delta[index] = (
            sampled_candidate - sampled_base
        ) / sampled_rows
    return {
        "dates": len(daily),
        "rows": int(rows.sum()),
        "candidate_mse": float(candidate_sse.sum() / rows.sum()),
        "base_mse": float(base_sse.sum() / rows.sum()),
        "incremental_mse_r2": point_r2,
        "incremental_mse_r2_bootstrap_95pct": [
            float(np.quantile(boot_r2, 0.025)),
            float(np.quantile(boot_r2, 0.975)),
        ],
        "mean_squared_loss_delta_candidate_minus_base": point_delta,
        "mean_squared_loss_delta_bootstrap_95pct": [
            float(np.quantile(boot_delta, 0.025)),
            float(np.quantile(boot_delta, 0.975)),
        ],
        "bootstrap_probability_incremental_mse_r2_le_zero": float(
            np.mean(boot_r2 <= 0)
        ),
        "resamples": resamples,
        "block_sessions": block_sessions,
        "seed": seed,
        "bootstrap_blocks_fold_contained": True,
    }


def _load_prediction_reference(
    key: str,
    protocol: Mapping[str, Any],
    cache: dict[str, tuple[pd.DataFrame, dict[str, Any]]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if key not in cache:
        if key.startswith("v3:"):
            cache[key] = common.load_verified_v3_base(
                protocol, key.split(":", 1)[1]
            )
        else:
            cache[key] = common.load_completed_bundle(key)
    frame, provenance = cache[key]
    normalized = frame.copy()
    normalized["forecast_date"] = pd.to_datetime(
        normalized["forecast_date"]
    ).dt.normalize()
    return normalized, provenance


def build_comparisons(
    protocol: Mapping[str, Any],
    *,
    resamples: int,
    block_sessions: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cache: dict[str, tuple[pd.DataFrame, dict[str, Any]]] = {}
    provenance: dict[str, Any] = {}
    for comparison_index, comparison in enumerate(COMPARISONS):
        candidate, candidate_provenance = _load_prediction_reference(
            comparison.candidate, protocol, cache
        )
        base, base_provenance = _load_prediction_reference(
            comparison.base, protocol, cache
        )
        provenance[comparison.candidate] = candidate_provenance
        provenance[comparison.base] = base_provenance
        for target_index, target in enumerate(contract.TARGETS):
            daily = paired_date_losses(candidate, base, target=target)
            result = paired_bootstrap(
                daily,
                resamples=resamples,
                block_sessions=block_sessions,
                seed=seed + comparison_index * 101 + target_index,
            )
            fold_result = paired_fold_improvements(
                candidate,
                base,
                target=target,
                family=comparison.family,
            )
            point_pass = bool(
                result["mean_squared_loss_delta_candidate_minus_base"] < 0
            )
            ci_upper_pass = bool(
                result["mean_squared_loss_delta_bootstrap_95pct"][1] < 0
            )
            rows.append(
                {
                    "comparison": comparison.key,
                    "family": comparison.family,
                    "candidate": comparison.candidate,
                    "base": comparison.base,
                    "target": target,
                    **result,
                    **fold_result,
                    "point_loss_delta_gate_pass": point_pass,
                    "loss_delta_ci_upper_gate_pass": ci_upper_pass,
                    "comparison_gate_pass": bool(
                        point_pass
                        and ci_upper_pass
                        and fold_result["fold_improvement_gate_pass"]
                    ),
                }
            )
    return pd.DataFrame(rows), provenance


def build_useful_semantic_gates(comparisons: pd.DataFrame) -> pd.DataFrame:
    """Apply the predeclared all-comparator semantic-usefulness gate."""

    rows: list[dict[str, Any]] = []
    for candidate, requirements in USEFUL_GATE_REQUIREMENTS.items():
        control_set_complete = USEFUL_GATE_CONTROL_SET_COMPLETE[candidate]
        for target in contract.TARGETS:
            selected = comparisons[
                comparisons["comparison"].isin(requirements)
                & comparisons["target"].eq(target)
            ].copy()
            if set(selected["comparison"]) != set(requirements) or len(selected) != len(
                requirements
            ):
                raise ValueError(
                    f"Incomplete useful-semantic gate for {candidate}/{target}"
                )
            selected = selected.set_index("comparison").loc[list(requirements)]
            failed = selected.index[
                ~selected["comparison_gate_pass"].astype(bool)
            ].tolist()
            rows.append(
                {
                    "candidate": candidate,
                    "target": target,
                    "required_comparison_count": len(requirements),
                    "passed_comparison_count": int(
                        selected["comparison_gate_pass"].astype(bool).sum()
                    ),
                    "all_point_loss_deltas_negative": bool(
                        selected["point_loss_delta_gate_pass"].astype(bool).all()
                    ),
                    "all_loss_delta_ci_uppers_below_zero": bool(
                        selected["loss_delta_ci_upper_gate_pass"].astype(bool).all()
                    ),
                    "all_fold_improvement_requirements_met": bool(
                        selected["fold_improvement_gate_pass"].astype(bool).all()
                    ),
                    "required_comparisons": list(requirements),
                    "failed_comparisons": failed,
                    "control_set_complete": control_set_complete,
                    "useful_semantic_gate_evaluable": control_set_complete,
                    "useful_semantic_gate_pass": bool(
                        control_set_complete and not failed
                    ),
                }
            )
    return pd.DataFrame(rows)


def _markdown(frame: pd.DataFrame, gates: pd.DataFrame) -> str:
    lines = [
        "# V4 final paired comparisons",
        "",
        "These are exploratory retrospective results. Incremental MSE R2 is",
        "`1 - candidate MSE / base MSE`; positive values favor the candidate.",
        "Confidence intervals use paired moving blocks of whole forecast dates.",
        "",
        "| Comparison | Target | Incremental MSE R2 | 95% interval | Folds improved | Gate |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in frame.to_dict(orient="records"):
        low, high = row["incremental_mse_r2_bootstrap_95pct"]
        lines.append(
            f"| {row['comparison']} | {row['target']} | "
            f"{row['incremental_mse_r2']:.6f} | [{low:.6f}, {high:.6f}] | "
            f"{row['folds_improved']}/{row['folds_total']} | "
            f"{'pass' if row['comparison_gate_pass'] else 'fail'} |"
        )
    lines.extend(
        [
            "",
            "## Useful-semantic gates",
            "",
            "A live semantic candidate passes only when every required matched-base,",
            "stale, wrong-stock, permutation, and quality comparison passes its point,",
            "confidence-interval, and fold-count conditions.",
            "",
            "| Candidate | Target | Comparisons passed | Useful gate |",
            "|---|---|---:|---|",
        ]
    )
    for row in gates.to_dict(orient="records"):
        lines.append(
            f"| {row['candidate']} | {row['target']} | "
            f"{row['passed_comparison_count']}/{row['required_comparison_count']} | "
            f"{'pass' if row['useful_semantic_gate_pass'] else 'fail'} |"
        )
    return "\n".join(lines) + "\n"


def parser() -> argparse.ArgumentParser:
    output = argparse.ArgumentParser(description=__doc__)
    output.add_argument("--resamples", type=int, default=2000)
    output.add_argument("--block-sessions", type=int, default=10)
    output.add_argument("--seed", type=int, default=1729)
    output.add_argument("--overwrite", action="store_true")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    protocol = common.load_protocol()
    existing = [path for path in (OUTPUT_ROOT, TRACKED_ROOT) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"v4 final comparison exists: {existing}")
    if args.overwrite:
        for path in existing:
            shutil.rmtree(path)
    frame, provenance = build_comparisons(
        protocol,
        resamples=args.resamples,
        block_sessions=args.block_sessions,
        seed=args.seed,
    )
    gates = build_useful_semantic_gates(frame)
    OUTPUT_ROOT.mkdir(parents=True)
    common.atomic_parquet(OUTPUT_ROOT / "paired_comparisons.parquet", frame)
    common.atomic_parquet(
        OUTPUT_ROOT / "useful_semantic_gates.parquet", gates
    )
    common.atomic_json(
        OUTPUT_ROOT / "summary.json",
        {
            "status": "complete",
            "comparisons": frame.to_dict(orient="records"),
            "useful_semantic_gates": gates.to_dict(orient="records"),
            "claim_flags": protocol["claim_flags"],
        },
    )
    common.atomic_json(OUTPUT_ROOT / "dependencies.json", provenance)
    common.atomic_text(OUTPUT_ROOT / "RESULTS.md", _markdown(frame, gates))
    artifacts = {
        name: common.artifact_record(
            OUTPUT_ROOT / name,
            rows=(
                len(frame)
                if name == "paired_comparisons.parquet"
                else len(gates)
                if name == "useful_semantic_gates.parquet"
                else None
            ),
        )
        for name in (
            "paired_comparisons.parquet",
            "useful_semantic_gates.parquet",
            "summary.json",
            "dependencies.json",
            "RESULTS.md",
        )
    }
    common.atomic_json(
        OUTPUT_ROOT / "manifest.json",
        {
            "manifest_version": "quant-deterministic-news-v4-comparison-v1",
            "generated_at_utc": common.utc_now(),
            "protocol_sha256": common.sha256_file(contract.PROTOCOL_PATH),
            "bootstrap": {
                "resamples": args.resamples,
                "block_sessions": args.block_sessions,
                "seed": args.seed,
            },
            "artifacts": artifacts,
        },
    )
    common.atomic_text(
        OUTPUT_ROOT / "manifest.sha256",
        f"{common.sha256_file(OUTPUT_ROOT / 'manifest.json')}  manifest.json\n",
    )
    TRACKED_ROOT.mkdir(parents=True)
    for name in ("summary.json", "RESULTS.md"):
        shutil.copy2(OUTPUT_ROOT / name, TRACKED_ROOT / name)
    common.atomic_json(
        TRACKED_ROOT / "artifact_pointer.json",
        {
            "manifest_path": (OUTPUT_ROOT / "manifest.json").as_posix(),
            "manifest_sha256": common.sha256_file(OUTPUT_ROOT / "manifest.json"),
            "protocol_sha256": common.sha256_file(contract.PROTOCOL_PATH),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
