"""Run the locked published-logic linear ladder for all four targets."""

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

from scripts.correlation_training import training_common as common


def run_rung_01(
    panel: pd.DataFrame,
    protocol: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    predictions: list[pd.DataFrame] = []
    fit_records: list[dict[str, object]] = []
    alpha_grid = [float(value) for value in protocol["linear_alpha_grid"]]
    for spec in common.target_specs():
        model_features = {
            "har_ols": spec.har_features,
            "shar_ols": spec.shar_features,
            "core22_ols": spec.core_features,
            "core22_lasso": spec.core_features,
        }
        for fold in protocol["folds"]:
            masks = common.split_masks(panel, spec, fold)
            train = panel.loc[masks["train"]].copy()
            validation = panel.loc[masks["validation"]].copy()
            test = panel.loc[masks["test"]].copy()
            if min(len(train), len(validation), len(test)) == 0:
                raise ValueError(f"Empty split for {spec.name} {fold['name']}")
            joined = pd.concat([train, validation], ignore_index=True)

            persistence_z = common.correlation_to_fisher(
                test[spec.persistence_column]
            )
            predictions.append(
                common.prediction_frame(
                    test,
                    spec,
                    fold_name=fold["name"],
                    model_name="persistence",
                    predicted_fisher=persistence_z,
                )
            )
            fit_records.append(
                {
                    "target": spec.name,
                    "fold": fold["name"],
                    "model": "persistence",
                    "features": [spec.persistence_column],
                    "train_rows": len(train),
                    "validation_rows": len(validation),
                    "test_rows": len(test),
                }
            )

            for model_name in ("har_ols", "shar_ols", "core22_ols"):
                features = model_features[model_name]
                predicted, model = common.fit_predict_linear(
                    "ols",
                    joined,
                    test,
                    features,
                    spec.response_column,
                )
                predictions.append(
                    common.prediction_frame(
                        test,
                        spec,
                        fold_name=fold["name"],
                        model_name=model_name,
                        predicted_fisher=predicted,
                    )
                )
                fit_records.append(
                    {
                        "target": spec.name,
                        "fold": fold["name"],
                        "model": model_name,
                        "features": list(features),
                        "train_rows": len(train),
                        "validation_rows": len(validation),
                        "test_rows": len(test),
                        "coefficients": common.extract_linear_coefficients(
                            model, features
                        ),
                    }
                )

            features = model_features["core22_lasso"]
            best, candidates = common.select_linear_hyperparameters(
                "lasso",
                train,
                validation,
                features,
                spec.response_column,
                alpha_grid=alpha_grid,
            )
            predicted, model = common.fit_predict_linear(
                "lasso",
                joined,
                test,
                features,
                spec.response_column,
                best,
            )
            predictions.append(
                common.prediction_frame(
                    test,
                    spec,
                    fold_name=fold["name"],
                    model_name="core22_lasso",
                    predicted_fisher=predicted,
                )
            )
            fit_records.append(
                {
                    "target": spec.name,
                    "fold": fold["name"],
                    "model": "core22_lasso",
                    "features": list(features),
                    "train_rows": len(train),
                    "validation_rows": len(validation),
                    "test_rows": len(test),
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
        "persistence",
        "har_ols",
        "shar_ols",
        "core22_ols",
        "core22_lasso",
    }
    for record in fit_records:
        common.validate_feature_columns(record["features"])
    feature_timing_passed = common.validate_panel_information_set(panel)
    review = {
        "status": "passed",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "expected_models_present": (
            set(prediction_panel["model"]) == expected_models
        ),
        "expected_targets_present": (
            set(prediction_panel["target"])
            == {spec.name for spec in common.target_specs()}
        ),
        "prediction_bounds_valid": bool(
            prediction_panel["predicted_correlation"].between(-1, 1).all()
        ),
        "predictions_finite": bool(
            np.isfinite(
                prediction_panel[
                    ["predicted_fisher_z", "predicted_correlation"]
                ].to_numpy(dtype=float)
            ).all()
        ),
        "prediction_keys_unique": bool(
            ~prediction_panel.duplicated(
                ["fold", "target", "model", "stock", "forecast_date"]
            ).any()
        ),
        "feature_name_leakage_scan_passed": True,
        "feature_timing_audit_passed": feature_timing_passed,
        "fold_fit_records": len(fit_records),
    }
    required_checks = [
        "expected_models_present",
        "expected_targets_present",
        "prediction_bounds_valid",
        "predictions_finite",
        "prediction_keys_unique",
        "feature_name_leakage_scan_passed",
        "feature_timing_audit_passed",
    ]
    if not all(bool(review[key]) for key in required_checks):
        review["status"] = "failed"
        raise AssertionError(f"Rung 1 review failed: {review}")
    return prediction_panel, metrics, {
        "review": review,
        "fits": fit_records,
        "fold_metrics": fold_metrics.to_dict(orient="records"),
    }


def parser() -> argparse.ArgumentParser:
    output = argparse.ArgumentParser(description=__doc__)
    output.add_argument("--panel", type=Path, default=common.PANEL_PATH)
    output.add_argument("--protocol", type=Path, default=common.PROTOCOL_PATH)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    predictions, metrics, details = run_rung_01(
        common.load_panel(args.panel), common.load_protocol(args.protocol)
    )
    output_root = common.OUTPUT_ROOT / "rung_01"
    experiment_root = common.EXPERIMENT_ROOT / "rung_01"
    common.write_parquet(output_root / "predictions.parquet", predictions)
    common.write_json(output_root / "fits.json", details["fits"])
    common.write_json(output_root / "fold_metrics.json", details["fold_metrics"])
    common.write_json(experiment_root / "review.json", details["review"])
    common.write_json(
        experiment_root / "summary.json",
        {
            "status": "complete",
            "metrics": metrics.to_dict(orient="records"),
        },
    )
    for target, frame in predictions.groupby("target", sort=True):
        target_output = output_root / target
        target_experiment = experiment_root / target
        common.write_parquet(target_output / "predictions.parquet", frame)
        target_metrics = metrics[metrics["target"].eq(target)]
        common.write_json(
            target_experiment / "metrics.json",
            target_metrics.to_dict(orient="records"),
        )
    print(metrics.to_string(index=False))
    print(json.dumps(details["review"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
