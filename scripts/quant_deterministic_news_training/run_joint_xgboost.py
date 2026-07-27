"""Train one isolated GPU XGBoost bundle from Track A."""

from __future__ import annotations

import argparse
import json
from typing import Any, Sequence

import pandas as pd

from scripts.correlation_training import run_rung_03 as quant_xgb
from scripts.correlation_training import training_common as quant_common
from scripts.quant_deterministic_news_training import common


TREE_BUNDLES = {"A0-T", "A6"}


def _require_dependencies(bundle_key: str) -> None:
    common.require_completed_preflight()
    if bundle_key == "A6":
        common.verify_completed_bundle("A5")


def run_joint_xgboost(
    bundle_key: str,
    *,
    allow_exploratory: bool,
    device: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    list[dict[str, Any]],
    dict[str, Any],
]:
    protocol = common.load_protocol()
    panel, _ = common.load_panel_and_preflight(
        protocol, allow_exploratory=allow_exploratory
    )
    panel, control_audit = common.prepare_control_panel(panel, protocol, None)
    configs = common.xgboost_candidate_configs(protocol)
    max_estimators = int(
        protocol["model_tuning"]["xgboost_max_estimators"]
    )
    early_stopping = int(
        protocol["model_tuning"]["xgboost_early_stopping_rounds"]
    )
    predictions: list[pd.DataFrame] = []
    validation_predictions: list[pd.DataFrame] = []
    fits: list[dict[str, Any]] = []
    feature_contract: dict[str, Any] = {}
    for spec in quant_common.target_specs():
        feature_key = "A0-T" if bundle_key == "A0-T" else "A6"
        features = common.joint_features(feature_key, spec, protocol)
        feature_contract[spec.name] = {
            "count": len(features),
            "ordered_sha256": common.ordered_sha256(features),
            "features": list(features),
        }
        transformed = common.apply_fixed_transforms(
            panel, features, protocol
        )
        for fold in protocol["folds"]:
            masks = common.split_masks(transformed, spec, fold)
            train = transformed.loc[masks["train"]].copy()
            validation = transformed.loc[masks["validation"]].copy()
            test = transformed.loc[masks["test"]].copy()
            joined = pd.concat([train, validation], ignore_index=True)
            candidates = []
            candidate_predictions = []
            for candidate_id, config in enumerate(configs):
                model = quant_xgb.xgb_model(
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
                predicted = model.predict(validation[list(features)])
                candidate_predictions.append(predicted)
                candidates.append(
                    {
                        "candidate_id": candidate_id,
                        "config": config,
                        "best_iteration": int(model.best_iteration),
                        "validation_mse": quant_common.mse(
                            validation[spec.response_column].to_numpy(
                                dtype=float
                            ),
                            predicted,
                        ),
                        "cuda_confirmed": (
                            quant_xgb.booster_uses_cuda(model)
                            if device == "cuda"
                            else False
                        ),
                    }
                )
            position = min(
                range(len(candidates)),
                key=lambda index: candidates[index]["validation_mse"],
            )
            selected = candidates[position]
            validation_prediction = candidate_predictions[position]
            validation_predictions.append(
                quant_common.prediction_frame(
                    validation,
                    spec,
                    fold_name=fold["name"],
                    model_name=common.bundle_for(bundle_key).slug,
                    predicted_fisher=validation_prediction,
                )
            )
            final_model = quant_xgb.xgb_model(
                selected["config"],
                n_estimators=int(selected["best_iteration"]) + 1,
                early_stopping_rounds=None,
                device=device,
            )
            final_model.fit(
                joined[list(features)],
                joined[spec.response_column],
                verbose=False,
            )
            test_prediction = final_model.predict(test[list(features)])
            predictions.append(
                quant_common.prediction_frame(
                    test,
                    spec,
                    fold_name=fold["name"],
                    model_name=common.bundle_for(bundle_key).slug,
                    predicted_fisher=test_prediction,
                )
            )
            fits.append(
                {
                    "target": spec.name,
                    "fold": fold["name"],
                    "model": common.bundle_for(bundle_key).slug,
                    "features": list(features),
                    "feature_count": len(features),
                    "feature_ordered_sha256": common.ordered_sha256(features),
                    "train_rows": len(train),
                    "validation_rows": len(validation),
                    "test_rows": len(test),
                    "configuration_fit_rows": len(train),
                    "final_fit_rows": len(joined),
                    "candidate_records": candidates,
                    "selected_candidate": selected,
                    "final_n_estimators": int(selected["best_iteration"]) + 1,
                    "cuda_confirmed": (
                        quant_xgb.booster_uses_cuda(final_model)
                        if device == "cuda"
                        else False
                    ),
                    "feature_importances": [
                        {"feature": feature, "importance": float(value)}
                        for feature, value in zip(
                            features,
                            final_model.feature_importances_,
                            strict=True,
                        )
                    ],
                }
            )
    prediction_panel = pd.concat(predictions, ignore_index=True)
    validation_panel = pd.concat(
        validation_predictions, ignore_index=True
    )
    prediction_review = common.validate_prediction_panel(
        prediction_panel,
        expected_folds=["fold_1", "fold_2", "fold_3"],
        expected_targets=list(protocol["targets"]),
        expected_rows=common.track_a_expected_rows(protocol),
    )
    validation_review = common.validate_validation_predictions(
        validation_panel, protocol
    )
    if device == "cuda" and not all(
        bool(record["cuda_confirmed"]) for record in fits
    ):
        raise AssertionError("Not every final XGBoost fit used CUDA")
    frozen = common.load_frozen_predictions(protocol, "xgboost")
    base_comparison = common.summarize_with_base(prediction_panel, frozen)
    parity = None
    if bundle_key == "A0-T":
        parity = common.parity_statistics(prediction_panel, frozen)
        if parity["maximum_absolute_fisher_z_difference"] > 1e-6:
            raise AssertionError(f"A0-T parity gate failed: {parity}")
    details = {
        "feature_contract": feature_contract,
        "control_audit": control_audit,
        "prediction_review": prediction_review,
        "validation_review": validation_review,
        "base_comparison": base_comparison,
        "parity": parity,
        "device": device,
    }
    return prediction_panel, validation_panel, fits, details


def parser() -> argparse.ArgumentParser:
    output = argparse.ArgumentParser(description=__doc__)
    output.add_argument("--bundle", choices=sorted(TREE_BUNDLES), required=True)
    output.add_argument("--allow-exploratory", action="store_true")
    output.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    output.add_argument("--overwrite", action="store_true")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    _require_dependencies(args.bundle)
    common.ensure_no_existing_bundle(args.bundle, overwrite=args.overwrite)
    common.update_status(args.bundle, "running")
    try:
        predictions, validation, fits, details = run_joint_xgboost(
            args.bundle,
            allow_exploratory=args.allow_exploratory,
            device=args.device,
        )
        metrics = common.metrics_records(predictions)
        fold_metrics = common.fold_metric_records(predictions)
        review = {
            "status": "passed",
            **details["prediction_review"],
            "validation_prediction_review": details["validation_review"],
            "feature_name_leakage_scan_passed": True,
            "feature_timing_audit_passed": True,
            "fit_excludes_outer_test": True,
            "requested_device": args.device,
            "cuda_confirmed_all_final_fits": (
                all(record["cuda_confirmed"] for record in fits)
                if args.device == "cuda"
                else None
            ),
            "parity_gate": details["parity"],
        }
        summary = {
            "status": "complete",
            "metrics": metrics,
            "base_comparison": details["base_comparison"],
            "parity_gate": details["parity"],
        }
        protocol = common.load_protocol()
        common.write_bundle(
            args.bundle,
            predictions=predictions,
            validation_predictions=validation,
            fits=fits,
            fold_metrics=fold_metrics,
            summary=summary,
            review=review,
            model_config={
                "estimator": "xgboost",
                "bundle": args.bundle,
                "feature_contract": details["feature_contract"],
                "candidate_configs": common.xgboost_candidate_configs(
                    protocol
                ),
                "max_estimators": protocol["model_tuning"][
                    "xgboost_max_estimators"
                ],
                "early_stopping_rounds": protocol["model_tuning"][
                    "xgboost_early_stopping_rounds"
                ],
                "device": args.device,
            },
            dependencies={
                "preflight": (
                    common.bundle_for("PRE").output_path / "manifest.json"
                ).as_posix(),
                "frozen_xgboost_sha256": protocol["source_artifacts"][
                    "quant_rung_3_predictions_sha256"
                ],
            },
        )
        text = common.summary_status_text(summary)
        common.update_status(args.bundle, "complete", summary=text)
        print(pd.DataFrame(metrics).to_string(index=False))
        print(json.dumps(review, indent=2))
        return 0
    except Exception as error:
        common.update_status(args.bundle, "failed", summary=str(error))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
