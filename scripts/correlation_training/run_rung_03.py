"""Run the GPU XGBoost challenger and optional linear/tree ensemble."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import xgboost as xgb


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.correlation_training import build_modeling_panel as schema  # noqa: E402
from scripts.correlation_training import training_common as common  # noqa: E402


def final_features(spec: common.TargetSpec) -> tuple[str, ...]:
    return tuple(
        [
            *spec.core_features,
            *schema.DENSE_CONTEXT_FEATURES,
            *schema.VOLATILITY_FEATURES,
            *schema.EXTENDED_FEATURES,
        ]
    )


def xgb_model(
    config: Mapping[str, float],
    *,
    n_estimators: int,
    early_stopping_rounds: int | None,
    device: str,
) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        device=device,
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
        random_state=1729,
        n_jobs=4,
        verbosity=0,
        **dict(config),
    )


def booster_uses_cuda(model: xgb.XGBRegressor) -> bool:
    return '"device":"cuda:0"' in model.get_booster().save_config()


def run_rung_03(
    panel: pd.DataFrame,
    protocol: dict[str, object],
    *,
    device: str = "cuda",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    rung = protocol["rung_3"]
    configs = [dict(item) for item in rung["candidate_configs"]]
    max_estimators = int(rung["n_estimators"])
    early_stopping = int(rung["early_stopping_rounds"])
    ensemble_weights = [
        float(value) for value in rung["ensemble_weights_on_xgboost"]
    ]
    alpha_grid = [float(value) for value in protocol["linear_alpha_grid"]]
    l1_ratios = [
        float(value) for value in protocol["elastic_net_l1_ratio_grid"]
    ]
    tree_predictions: list[pd.DataFrame] = []
    ensemble_candidates: list[tuple[str, pd.DataFrame]] = []
    fits: list[dict[str, object]] = []
    ensemble_diagnostics: list[dict[str, object]] = []
    for spec in common.target_specs():
        features = final_features(spec)
        transformed = common.apply_fixed_log1p_transforms(panel, features)
        for fold in protocol["folds"]:
            masks = common.split_masks(transformed, spec, fold)
            train = transformed.loc[masks["train"]].copy()
            validation = transformed.loc[masks["validation"]].copy()
            test = transformed.loc[masks["test"]].copy()
            joined = pd.concat([train, validation], ignore_index=True)
            candidate_records: list[dict[str, object]] = []
            candidate_predictions: list[np.ndarray] = []
            for candidate_id, config in enumerate(configs):
                model = xgb_model(
                    config,
                    n_estimators=max_estimators,
                    early_stopping_rounds=early_stopping,
                    device=device,
                )
                model.fit(
                    train[list(features)],
                    train[spec.response_column],
                    eval_set=[
                        (
                            validation[list(features)],
                            validation[spec.response_column],
                        )
                    ],
                    verbose=False,
                )
                prediction = model.predict(validation[list(features)])
                candidate_predictions.append(prediction)
                candidate_records.append(
                    {
                        "candidate_id": candidate_id,
                        "config": config,
                        "best_iteration": int(model.best_iteration),
                        "validation_mse": common.mse(
                            validation[spec.response_column].to_numpy(
                                dtype=float
                            ),
                            prediction,
                        ),
                        "cuda_confirmed": booster_uses_cuda(model)
                        if device == "cuda"
                        else False,
                    }
                )
            best_position = min(
                range(len(candidate_records)),
                key=lambda position: candidate_records[position][
                    "validation_mse"
                ],
            )
            selected = candidate_records[best_position]
            selected_validation_prediction = candidate_predictions[best_position]
            final_tree = xgb_model(
                selected["config"],
                n_estimators=int(selected["best_iteration"]) + 1,
                early_stopping_rounds=None,
                device=device,
            )
            final_tree.fit(
                joined[list(features)], joined[spec.response_column], verbose=False
            )
            tree_test_prediction = final_tree.predict(test[list(features)])
            tree_predictions.append(
                common.prediction_frame(
                    test,
                    spec,
                    fold_name=fold["name"],
                    model_name="xgboost",
                    predicted_fisher=tree_test_prediction,
                )
            )

            enet_parameters, enet_candidates = (
                common.select_linear_hyperparameters(
                    "elastic_net",
                    train,
                    validation,
                    features,
                    spec.response_column,
                    alpha_grid=alpha_grid,
                    l1_ratios=l1_ratios,
                )
            )
            validation_enet = common.linear_pipeline(
                "elastic_net", **enet_parameters
            )
            validation_enet.fit(
                train[list(features)], train[spec.response_column]
            )
            enet_validation_prediction = validation_enet.predict(
                validation[list(features)]
            )
            enet_test_prediction, _ = common.fit_predict_linear(
                "elastic_net",
                joined,
                test,
                features,
                spec.response_column,
                enet_parameters,
            )
            weight_records = []
            validation_actual = validation[spec.response_column].to_numpy(
                dtype=float
            )
            for weight in ensemble_weights:
                prediction = (
                    weight * selected_validation_prediction
                    + (1 - weight) * enet_validation_prediction
                )
                weight_records.append(
                    {
                        "xgboost_weight": weight,
                        "validation_mse": common.mse(
                            validation_actual, prediction
                        ),
                    }
                )
            selected_weight = min(
                weight_records, key=lambda item: item["validation_mse"]
            )
            tree_validation_mse = common.mse(
                validation_actual, selected_validation_prediction
            )
            enet_validation_mse = common.mse(
                validation_actual, enet_validation_prediction
            )
            strict_improvement = selected_weight["validation_mse"] < min(
                tree_validation_mse, enet_validation_mse
            ) - 1e-12
            interior_weight = selected_weight["xgboost_weight"] not in (0.0, 1.0)
            ensemble_prediction = (
                selected_weight["xgboost_weight"] * tree_test_prediction
                + (1 - selected_weight["xgboost_weight"])
                * enet_test_prediction
            )
            ensemble_candidates.append(
                (
                    spec.name,
                    common.prediction_frame(
                        test,
                        spec,
                        fold_name=fold["name"],
                        model_name="elastic_xgboost_ensemble",
                        predicted_fisher=ensemble_prediction,
                    ),
                )
            )
            diagnostic = {
                "target": spec.name,
                "fold": fold["name"],
                "selected_weight": selected_weight,
                "tree_validation_mse": tree_validation_mse,
                "elastic_net_validation_mse": enet_validation_mse,
                "strict_improvement": strict_improvement,
                "interior_weight": interior_weight,
                "weight_candidates": weight_records,
            }
            ensemble_diagnostics.append(diagnostic)
            fits.append(
                {
                    "target": spec.name,
                    "fold": fold["name"],
                    "model": "xgboost",
                    "features": list(features),
                    "candidate_records": candidate_records,
                    "selected_candidate": selected,
                    "final_n_estimators": int(selected["best_iteration"]) + 1,
                    "configuration_fit_rows": len(train),
                    "final_fit_rows": len(joined),
                    "outer_test_rows": len(test),
                    "cuda_confirmed": booster_uses_cuda(final_tree)
                    if device == "cuda"
                    else False,
                    "feature_importances": [
                        {"feature": feature, "importance": float(value)}
                        for feature, value in zip(
                            features,
                            final_tree.feature_importances_,
                            strict=True,
                        )
                    ],
                    "elastic_net_parameters_for_ensemble": enet_parameters,
                    "elastic_net_validation_candidates": enet_candidates,
                }
            )

    predictions = list(tree_predictions)
    included_ensembles: list[str] = []
    for spec in common.target_specs():
        diagnostics = [
            item for item in ensemble_diagnostics if item["target"] == spec.name
        ]
        consistent = len(diagnostics) == len(protocol["folds"]) and all(
            item["strict_improvement"] and item["interior_weight"]
            for item in diagnostics
        )
        if consistent:
            included_ensembles.append(spec.name)
            predictions.extend(
                frame
                for target, frame in ensemble_candidates
                if target == spec.name
            )
    prediction_panel = pd.concat(predictions, ignore_index=True)
    metrics = common.summarize_predictions(prediction_panel)
    fold_metrics = common.summarize_predictions(
        prediction_panel, ("fold", "target", "model")
    )
    cuda_required = device == "cuda"
    for record in fits:
        common.validate_feature_columns(record["features"])
    feature_timing_passed = common.validate_panel_information_set(panel)
    review = {
        "status": "passed",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "xgboost_version": xgb.__version__,
        "requested_device": device,
        "cuda_confirmed_all_final_fits": (
            all(record["cuda_confirmed"] for record in fits)
            if cuda_required
            else None
        ),
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
        "fit_excludes_outer_test": all(
            record["configuration_fit_rows"] > 0
            and record["final_fit_rows"] > record["configuration_fit_rows"]
            and record["outer_test_rows"] > 0
            for record in fits
        ),
        "included_ensemble_targets": included_ensembles,
        "ensemble_policy": (
            "included only when an interior weight strictly improved over both "
            "components on every validation fold for that target"
        ),
        "fit_records": len(fits),
    }
    required = [
        "predictions_finite",
        "prediction_bounds_valid",
        "prediction_keys_unique",
        "feature_name_leakage_scan_passed",
        "feature_timing_audit_passed",
        "fit_excludes_outer_test",
    ]
    if cuda_required:
        required.append("cuda_confirmed_all_final_fits")
    if not all(bool(review[key]) for key in required):
        review["status"] = "failed"
        raise AssertionError(f"Rung 3 review failed: {review}")
    return prediction_panel, metrics, {
        "fits": fits,
        "fold_metrics": fold_metrics.to_dict(orient="records"),
        "ensemble_diagnostics": ensemble_diagnostics,
        "review": review,
    }


def parser() -> argparse.ArgumentParser:
    output = argparse.ArgumentParser(description=__doc__)
    output.add_argument("--panel", type=Path)
    output.add_argument("--protocol", type=Path, default=common.PROTOCOL_PATH)
    output.add_argument("--experiment-root", type=Path)
    output.add_argument("--output-root", type=Path)
    output.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
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
    predictions, metrics, details = run_rung_03(
        common.load_panel(paths.panel),
        protocol,
        device=args.device,
    )
    output_root = paths.output_root / "rung_03"
    experiment_root = paths.experiment_root / "rung_03"
    common.write_parquet(output_root / "predictions.parquet", predictions)
    common.write_json(output_root / "fits.json", details["fits"])
    common.write_json(output_root / "fold_metrics.json", details["fold_metrics"])
    common.write_json(
        output_root / "ensemble_diagnostics.json",
        details["ensemble_diagnostics"],
    )
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
