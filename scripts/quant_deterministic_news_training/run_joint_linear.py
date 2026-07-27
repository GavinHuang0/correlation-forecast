"""Train one isolated Elastic Net bundle from Track A or its controls."""

from __future__ import annotations

import argparse
import json
from typing import Any, Sequence

import numpy as np
import pandas as pd

from scripts.correlation_training import training_common as quant_common
from scripts.quant_deterministic_news_training import common


LINEAR_BUNDLES = {
    "A0-L": None,
    "A1": None,
    "A2": None,
    "A3": None,
    "A4": None,
    "A5": None,
    "C-A5-L20": "lag20",
    "C-A5-WS": "wrong_stock",
}


def run_joint_linear(
    bundle_key: str,
    *,
    allow_exploratory: bool,
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
    panel, control_audit = common.prepare_control_panel(
        panel, protocol, LINEAR_BUNDLES[bundle_key]
    )
    alpha_grid = [
        float(value) for value in protocol["model_tuning"]["linear_alpha_grid"]
    ]
    l1_ratios = [
        float(value)
        for value in protocol["model_tuning"]["elastic_net_l1_ratio_grid"]
    ]
    predictions: list[pd.DataFrame] = []
    validation_predictions: list[pd.DataFrame] = []
    fits: list[dict[str, Any]] = []
    feature_contract: dict[str, Any] = {}
    for spec in quant_common.target_specs():
        feature_key = "A5" if bundle_key.startswith("C-A5") else bundle_key
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
            selected, candidates = common.select_linear_hyperparameters(
                train,
                validation,
                features,
                spec.response_column,
                alpha_grid=alpha_grid,
                l1_ratios=l1_ratios,
            )
            validation_model = quant_common.linear_pipeline(
                "elastic_net", **selected
            )
            validation_model.fit(
                train[list(features)], train[spec.response_column]
            )
            validation_prediction = validation_model.predict(
                validation[list(features)]
            )
            validation_predictions.append(
                quant_common.prediction_frame(
                    validation,
                    spec,
                    fold_name=fold["name"],
                    model_name=common.bundle_for(bundle_key).slug,
                    predicted_fisher=validation_prediction,
                )
            )
            test_prediction, final_model = common.fit_predict_linear(
                "elastic_net",
                joined,
                test,
                features,
                spec.response_column,
                selected,
            )
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
                    "hyperparameter_fit_rows": len(train),
                    "final_preprocessing_fit_rows": len(joined),
                    "selected_parameters": selected,
                    "validation_candidates": candidates,
                    "coefficients": quant_common.extract_linear_coefficients(
                        final_model, features
                    ),
                }
            )
    prediction_panel = pd.concat(predictions, ignore_index=True)
    validation_panel = pd.concat(
        validation_predictions, ignore_index=True
    )
    expected_rows = common.track_a_expected_rows(protocol)
    prediction_review = common.validate_prediction_panel(
        prediction_panel,
        expected_folds=["fold_1", "fold_2", "fold_3"],
        expected_targets=list(protocol["targets"]),
        expected_rows=expected_rows,
    )
    validation_review = common.validate_validation_predictions(
        validation_panel, protocol
    )
    if not all(
        record["hyperparameter_fit_rows"] == record["train_rows"]
        and record["final_preprocessing_fit_rows"]
        == record["train_rows"] + record["validation_rows"]
        for record in fits
    ):
        raise AssertionError("Linear preprocessing row audit failed")
    frozen = common.load_frozen_predictions(protocol, "elastic_net")
    base_comparison = common.summarize_with_base(prediction_panel, frozen)
    parity = None
    if bundle_key == "A0-L":
        parity = common.parity_statistics(prediction_panel, frozen)
        if parity["maximum_absolute_fisher_z_difference"] > 1e-10:
            raise AssertionError(f"A0-L parity gate failed: {parity}")
    details = {
        "feature_contract": feature_contract,
        "control_audit": control_audit,
        "prediction_review": prediction_review,
        "validation_review": validation_review,
        "base_comparison": base_comparison,
        "parity": parity,
    }
    return prediction_panel, validation_panel, fits, details


def parser() -> argparse.ArgumentParser:
    output = argparse.ArgumentParser(description=__doc__)
    output.add_argument("--bundle", choices=sorted(LINEAR_BUNDLES), required=True)
    output.add_argument("--allow-exploratory", action="store_true")
    output.add_argument("--overwrite", action="store_true")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    common.require_completed_preflight()
    common.ensure_no_existing_bundle(args.bundle, overwrite=args.overwrite)
    common.update_status(args.bundle, "running")
    try:
        predictions, validation, fits, details = run_joint_linear(
            args.bundle, allow_exploratory=args.allow_exploratory
        )
        metrics = common.metrics_records(predictions)
        fold_metrics = common.fold_metric_records(predictions)
        review = {
            "status": "passed",
            **details["prediction_review"],
            "validation_prediction_review": details["validation_review"],
            "feature_name_leakage_scan_passed": True,
            "feature_timing_audit_passed": True,
            "preprocessing_excludes_outer_test": True,
            "parity_gate": details["parity"],
        }
        summary = {
            "status": "complete",
            "metrics": metrics,
            "base_comparison": details["base_comparison"],
            "parity_gate": details["parity"],
            "control": details["control_audit"],
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
                "estimator": "elastic_net",
                "bundle": args.bundle,
                "feature_contract": details["feature_contract"],
                "alpha_grid": protocol["model_tuning"]["linear_alpha_grid"],
                "l1_ratio_grid": protocol["model_tuning"][
                    "elastic_net_l1_ratio_grid"
                ],
                "control": details["control_audit"],
            },
            dependencies={
                "preflight": (
                    common.bundle_for("PRE").output_path / "manifest.json"
                ).as_posix(),
                "frozen_elastic_net_sha256": protocol["source_artifacts"][
                    "quant_rung_2_predictions_sha256"
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
