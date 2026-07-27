"""Gate and construct the optional A7 Elastic Net/XGBoost ensemble."""

from __future__ import annotations

import argparse
import json
from typing import Any, Sequence

import numpy as np
import pandas as pd

from scripts.quant_deterministic_news_training import common


def _load_component(
    bundle_key: str, filename: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path, provenance = common.verified_bundle_artifact(
        bundle_key, filename
    )
    frame = pd.read_parquet(path)
    frame["forecast_date"] = pd.to_datetime(frame["forecast_date"]).dt.normalize()
    return frame, provenance


def _merge_components(
    linear: pd.DataFrame, tree: pd.DataFrame
) -> pd.DataFrame:
    columns = [
        *common.PREDICTION_KEYS,
        "actual_fisher_z",
        "actual_correlation",
        "persistence_fisher_z",
        "persistence_correlation",
        "predicted_fisher_z",
    ]
    merged = linear[columns].merge(
        tree[columns],
        on=common.PREDICTION_KEYS,
        how="inner",
        validate="one_to_one",
        suffixes=("_linear", "_tree"),
    )
    if len(merged) != len(linear) or len(merged) != len(tree):
        raise ValueError("A5/A6 component keys differ")
    for column in (
        "actual_fisher_z",
        "actual_correlation",
        "persistence_fisher_z",
        "persistence_correlation",
    ):
        delta = (merged[f"{column}_linear"] - merged[f"{column}_tree"]).abs()
        if float(delta.max()) > 1e-12:
            raise ValueError(f"A5/A6 {column} differs")
    return merged


def _prediction_from_merged(
    frame: pd.DataFrame, *, weight: float
) -> pd.DataFrame:
    output = frame[common.PREDICTION_KEYS].copy()
    output["actual_fisher_z"] = frame["actual_fisher_z_linear"]
    output["actual_correlation"] = frame["actual_correlation_linear"]
    output["persistence_fisher_z"] = frame["persistence_fisher_z_linear"]
    output["persistence_correlation"] = frame[
        "persistence_correlation_linear"
    ]
    output["predicted_fisher_z"] = (
        weight * frame["predicted_fisher_z_tree"]
        + (1 - weight) * frame["predicted_fisher_z_linear"]
    )
    output["predicted_correlation"] = np.tanh(
        output["predicted_fisher_z"].to_numpy(dtype=float)
    )
    output["model"] = common.bundle_for("A7").slug
    output["xgboost_weight"] = weight
    return output


def run_ensemble() -> tuple[
    pd.DataFrame | None,
    pd.DataFrame | None,
    list[dict[str, Any]],
    dict[str, Any],
]:
    protocol = common.load_protocol()
    weights = [
        float(value)
        for value in protocol["model_tuning"]["ensemble_xgboost_weights"]
    ]
    a5_validation, a5_validation_provenance = _load_component(
        "A5", "validation_predictions.parquet"
    )
    a6_validation, a6_validation_provenance = _load_component(
        "A6", "validation_predictions.parquet"
    )
    a5_outer, a5_outer_provenance = _load_component(
        "A5", "predictions.parquet"
    )
    a6_outer, a6_outer_provenance = _load_component(
        "A6", "predictions.parquet"
    )
    validation = _merge_components(a5_validation, a6_validation)
    outer = _merge_components(a5_outer, a6_outer)
    component_provenance = {
        "A5_validation": a5_validation_provenance,
        "A6_validation": a6_validation_provenance,
        "A5_predictions": a5_outer_provenance,
        "A6_predictions": a6_outer_provenance,
    }
    diagnostics = []
    selected_by_target_fold: dict[tuple[str, str], float] = {}
    for (target, fold), frame in validation.groupby(
        ["target", "fold"], sort=True
    ):
        actual = frame["actual_fisher_z_linear"].to_numpy(dtype=float)
        linear = frame["predicted_fisher_z_linear"].to_numpy(dtype=float)
        tree = frame["predicted_fisher_z_tree"].to_numpy(dtype=float)
        candidates = []
        for weight in weights:
            prediction = weight * tree + (1 - weight) * linear
            candidates.append(
                {
                    "xgboost_weight": weight,
                    "validation_mse": float(
                        np.mean(np.square(actual - prediction))
                    ),
                }
            )
        selected = min(candidates, key=lambda item: item["validation_mse"])
        linear_mse = float(np.mean(np.square(actual - linear)))
        tree_mse = float(np.mean(np.square(actual - tree)))
        strict = selected["validation_mse"] < min(
            linear_mse, tree_mse
        ) - 1e-12
        interior = selected["xgboost_weight"] not in (0.0, 1.0)
        diagnostics.append(
            {
                "target": target,
                "fold": fold,
                "selected": selected,
                "linear_validation_mse": linear_mse,
                "tree_validation_mse": tree_mse,
                "strict_improvement": strict,
                "interior_weight": interior,
                "weight_candidates": candidates,
            }
        )
        selected_by_target_fold[(target, fold)] = float(
            selected["xgboost_weight"]
        )
    included = []
    for target in protocol["targets"]:
        rows = [item for item in diagnostics if item["target"] == target]
        if len(rows) == 3 and all(
            item["strict_improvement"] and item["interior_weight"]
            for item in rows
        ):
            included.append(target)
    if not included:
        return None, None, diagnostics, {
            "included_targets": [],
            "eligibility": "no target passed the all-fold interior strict-improvement gate",
            "component_provenance": component_provenance,
        }
    outer_outputs = []
    validation_outputs = []
    for (target, fold), frame in outer.groupby(
        ["target", "fold"], sort=True
    ):
        if target not in included:
            continue
        outer_outputs.append(
            _prediction_from_merged(
                frame, weight=selected_by_target_fold[(target, fold)]
            )
        )
    for (target, fold), frame in validation.groupby(
        ["target", "fold"], sort=True
    ):
        if target not in included:
            continue
        validation_outputs.append(
            _prediction_from_merged(
                frame, weight=selected_by_target_fold[(target, fold)]
            )
        )
    return (
        pd.concat(outer_outputs, ignore_index=True),
        pd.concat(validation_outputs, ignore_index=True),
        diagnostics,
        {
            "included_targets": included,
            "component_provenance": component_provenance,
        },
    )


def parser() -> argparse.ArgumentParser:
    output = argparse.ArgumentParser(description=__doc__)
    output.add_argument("--overwrite", action="store_true")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    common.require_completed_preflight()
    common.verify_completed_bundle("A5")
    common.verify_completed_bundle("A6")
    common.ensure_no_existing_bundle("A7", overwrite=args.overwrite)
    common.update_status("A7", "running")
    try:
        predictions, validation, diagnostics, details = run_ensemble()
        protocol = common.load_protocol()
        if predictions is None:
            review = {
                "status": "passed",
                "prediction_keys_unique": True,
                "predictions_finite": True,
                "prediction_bounds_valid": True,
                "all_fold_gate_applied": True,
                "included_targets": [],
            }
            summary = {
                "status": "skipped",
                "metrics": [],
                "included_targets": [],
                "gate_diagnostics": diagnostics,
            }
            common.write_bundle(
                "A7",
                predictions=None,
                validation_predictions=None,
                fits=[],
                fold_metrics=[],
                summary=summary,
                review=review,
                model_config={
                    "estimator": "validation_weighted_ensemble",
                    "weights": protocol["model_tuning"][
                        "ensemble_xgboost_weights"
                    ],
                    "eligibility": (
                        "interior weight strictly beats both components on "
                        "every validation fold for a target"
                    ),
                },
                dependencies={
                    **details["component_provenance"],
                },
                extra_outputs={"ensemble_diagnostics.json": diagnostics},
            )
            common.update_status(
                "A7",
                "skipped",
                summary="no target passed the all-fold ensemble gate",
            )
            print(json.dumps(summary, indent=2))
            return 0
        included = details["included_targets"]
        expected = common.track_a_expected_rows(protocol)
        review = common.validate_prediction_panel(
            predictions,
            expected_folds=["fold_1", "fold_2", "fold_3"],
            expected_targets=included,
            expected_rows={target: expected[target] for target in included},
        )
        review["status"] = "passed"
        review["all_fold_gate_applied"] = True
        review["included_targets"] = included
        metrics = common.metrics_records(predictions)
        fold_metrics = common.fold_metric_records(predictions)
        frozen_xgb = common.load_frozen_predictions(protocol, "xgboost")
        frozen_en = common.load_frozen_predictions(protocol, "elastic_net")
        summary = {
            "status": "complete",
            "metrics": metrics,
            "included_targets": included,
            "base_comparison_xgboost": common.summarize_with_base(
                predictions,
                frozen_xgb[frozen_xgb["target"].isin(included)],
            ),
            "base_comparison_elastic_net": common.summarize_with_base(
                predictions,
                frozen_en[frozen_en["target"].isin(included)],
            ),
            "gate_diagnostics": diagnostics,
        }
        common.write_bundle(
            "A7",
            predictions=predictions,
            validation_predictions=validation,
            fits=[],
            fold_metrics=fold_metrics,
            summary=summary,
            review=review,
            model_config={
                "estimator": "validation_weighted_ensemble",
                "weights": protocol["model_tuning"][
                    "ensemble_xgboost_weights"
                ],
                "eligibility": (
                    "interior weight strictly beats both components on "
                    "every validation fold for a target"
                ),
            },
            dependencies={
                **details["component_provenance"],
            },
            extra_outputs={"ensemble_diagnostics.json": diagnostics},
        )
        common.update_status(
            "A7", "complete", summary=common.summary_status_text(summary)
        )
        print(pd.DataFrame(metrics).to_string(index=False))
        print(json.dumps(review, indent=2))
        return 0
    except Exception as error:
        common.update_status("A7", "failed", summary=str(error))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
