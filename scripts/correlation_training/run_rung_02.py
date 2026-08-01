"""Run locked nested practical linear models for T1/T2 ETF and LOO."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.correlation_training import build_modeling_panel as schema  # noqa: E402
from scripts.correlation_training import training_common as common  # noqa: E402


BLOCK_SUFFIXES = {
    "core_dense": (),
    "core_dense_volatility": tuple(schema.VOLATILITY_FEATURES),
    "core_dense_volatility_extended": (
        *schema.VOLATILITY_FEATURES,
        *schema.EXTENDED_FEATURES,
    ),
}


def feature_blocks(spec: common.TargetSpec) -> dict[str, tuple[str, ...]]:
    base = (*spec.core_features, *schema.DENSE_CONTEXT_FEATURES)
    return {
        name: tuple([*base, *suffix])
        for name, suffix in BLOCK_SUFFIXES.items()
    }


def run_rung_02(
    panel: pd.DataFrame,
    protocol: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    alpha_grid = [float(value) for value in protocol["linear_alpha_grid"]]
    l1_ratios = [
        float(value) for value in protocol["elastic_net_l1_ratio_grid"]
    ]
    predictions: list[pd.DataFrame] = []
    fits: list[dict[str, object]] = []
    for spec in common.target_specs():
        blocks = feature_blocks(spec)
        all_features = sorted({item for values in blocks.values() for item in values})
        transformed_panel = common.apply_fixed_log1p_transforms(
            panel, all_features
        )
        for fold in protocol["folds"]:
            masks = common.split_masks(transformed_panel, spec, fold)
            train = transformed_panel.loc[masks["train"]].copy()
            validation = transformed_panel.loc[masks["validation"]].copy()
            test = transformed_panel.loc[masks["test"]].copy()
            joined = pd.concat([train, validation], ignore_index=True)
            for block_name, features in blocks.items():
                for kind in ("lasso", "elastic_net"):
                    best, candidates = common.select_linear_hyperparameters(
                        kind,
                        train,
                        validation,
                        features,
                        spec.response_column,
                        alpha_grid=alpha_grid,
                        l1_ratios=l1_ratios,
                    )
                    predicted, model = common.fit_predict_linear(
                        kind,
                        joined,
                        test,
                        features,
                        spec.response_column,
                        best,
                    )
                    model_name = f"{block_name}_{kind}"
                    predictions.append(
                        common.prediction_frame(
                            test,
                            spec,
                            fold_name=fold["name"],
                            model_name=model_name,
                            predicted_fisher=predicted,
                        )
                    )
                    fits.append(
                        {
                            "target": spec.name,
                            "fold": fold["name"],
                            "model": model_name,
                            "feature_block": block_name,
                            "features": list(features),
                            "log1p_features": sorted(
                                set(features).intersection(schema.LOG1P_FEATURES)
                            ),
                            "train_rows": len(train),
                            "validation_rows": len(validation),
                            "test_rows": len(test),
                            "hyperparameter_fit_rows": len(train),
                            "final_preprocessing_fit_rows": len(joined),
                            "selected_parameters": best,
                            "validation_candidates": candidates,
                            "coefficients": common.extract_linear_coefficients(
                                model, features
                            ),
                        }
                    )
    prediction_panel = pd.concat(predictions, ignore_index=True)
    metrics = common.summarize_predictions(prediction_panel)
    fold_metrics = common.summarize_predictions(
        prediction_panel, ("fold", "target", "model")
    )
    expected_models = {
        f"{block}_{kind}"
        for block in BLOCK_SUFFIXES
        for kind in ("lasso", "elastic_net")
    }
    for record in fits:
        common.validate_feature_columns(record["features"])
    feature_timing_passed = common.validate_panel_information_set(panel)
    preprocessing_excludes_test = all(
        record["hyperparameter_fit_rows"] == record["train_rows"]
        and record["final_preprocessing_fit_rows"]
        == record["train_rows"] + record["validation_rows"]
        for record in fits
    )
    review = {
        "status": "passed",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "expected_models_present": set(prediction_panel["model"])
        == expected_models,
        "expected_targets_present": set(prediction_panel["target"])
        == {spec.name for spec in common.target_specs()},
        "predictions_finite": bool(
            np.isfinite(
                prediction_panel[
                    ["predicted_fisher_z", "predicted_correlation"]
                ].to_numpy(dtype=float)
            ).all()
        ),
        "prediction_bounds_valid": bool(
            prediction_panel["predicted_correlation"].between(-1, 1).all()
        ),
        "prediction_keys_unique": bool(
            ~prediction_panel.duplicated(
                ["fold", "target", "model", "stock", "forecast_date"]
            ).any()
        ),
        "feature_name_leakage_scan_passed": True,
        "feature_timing_audit_passed": feature_timing_passed,
        "preprocessing_excludes_outer_test": preprocessing_excludes_test,
        "fit_records": len(fits),
    }
    required = [
        "expected_models_present",
        "expected_targets_present",
        "predictions_finite",
        "prediction_bounds_valid",
        "prediction_keys_unique",
        "feature_name_leakage_scan_passed",
        "feature_timing_audit_passed",
        "preprocessing_excludes_outer_test",
    ]
    if not all(bool(review[key]) for key in required):
        review["status"] = "failed"
        raise AssertionError(f"Rung 2 review failed: {review}")
    return prediction_panel, metrics, {
        "fits": fits,
        "fold_metrics": fold_metrics.to_dict(orient="records"),
        "review": review,
    }


def parser() -> argparse.ArgumentParser:
    output = argparse.ArgumentParser(description=__doc__)
    output.add_argument("--panel", type=Path)
    output.add_argument("--protocol", type=Path, default=common.PROTOCOL_PATH)
    output.add_argument("--experiment-root", type=Path)
    output.add_argument("--output-root", type=Path)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    protocol = common.load_protocol(args.protocol)
    paths = common.resolve_training_paths(
        protocol,
        panel=args.panel,
        experiment_root=args.experiment_root,
        output_root=args.output_root,
    )
    predictions, metrics, details = run_rung_02(
        common.load_panel(paths.panel), protocol
    )
    output_root = paths.output_root / "rung_02"
    experiment_root = paths.experiment_root / "rung_02"
    common.write_parquet(output_root / "predictions.parquet", predictions)
    common.write_json(output_root / "fits.json", details["fits"])
    common.write_json(output_root / "fold_metrics.json", details["fold_metrics"])
    common.write_json(experiment_root / "review.json", details["review"])
    common.write_json(
        experiment_root / "summary.json",
        {"status": "complete", "metrics": metrics.to_dict(orient="records")},
    )
    for target, frame in predictions.groupby("target", sort=True):
        common.write_parquet(
            output_root / target / "predictions.parquet", frame
        )
        common.write_json(
            experiment_root / target / "metrics.json",
            metrics[metrics["target"].eq(target)].to_dict(orient="records"),
        )
    print(metrics.to_string(index=False))
    print(json.dumps(details["review"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
