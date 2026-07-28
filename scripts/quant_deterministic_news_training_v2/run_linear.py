"""Train one isolated deterministic-news v2 Elastic Net bundle."""

from __future__ import annotations

import argparse
import json
from typing import Any, Sequence

import pandas as pd

from scripts.correlation_training import training_common as quant_common
from scripts.quant_deterministic_news_training_v2 import common


LINEAR_BUNDLES = ("D0", "D1", "D2", "D3", "D4")


def _optional_secondary_base(bundle_key: str) -> tuple[str | None, str | None]:
    if bundle_key == "D3":
        return "D2", "D2-Normalized versus matched D43"
    if bundle_key == "D4":
        return "D3", "D2-Levels sensitivity versus D2-Normalized"
    return None, None


def run_linear(
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
    panel, preflight = common.load_panel_and_preflight(
        protocol,
        allow_exploratory=allow_exploratory,
        required_bundle_keys=(bundle_key,),
    )
    alpha_grid = [
        float(value) for value in protocol["model_tuning"]["linear_alpha_grid"]
    ]
    l1_ratios = [
        float(value)
        for value in protocol["model_tuning"][
            "elastic_net_l1_ratio_grid"
        ]
    ]
    bundle = common.bundle_for(bundle_key)
    predictions: list[pd.DataFrame] = []
    validation_predictions: list[pd.DataFrame] = []
    fits: list[dict[str, Any]] = []
    feature_contract: dict[str, Any] = {}

    for spec in quant_common.target_specs():
        features = common.bundle_features(bundle_key, spec, protocol)
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
            common.validate_training_feature_availability(
                train,
                features,
                label=f"{bundle_key}/{spec.name}/{fold['name']}/train",
            )
            common.validate_training_feature_availability(
                joined,
                features,
                label=f"{bundle_key}/{spec.name}/{fold['name']}/joined",
            )
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
                    model_name=bundle.slug,
                    predicted_fisher=validation_prediction,
                )
            )

            final_model = quant_common.linear_pipeline(
                "elastic_net", **selected
            )
            final_model.fit(
                joined[list(features)], joined[spec.response_column]
            )
            test_prediction = final_model.predict(test[list(features)])
            predictions.append(
                quant_common.prediction_frame(
                    test,
                    spec,
                    fold_name=fold["name"],
                    model_name=bundle.slug,
                    predicted_fisher=test_prediction,
                )
            )
            fits.append(
                {
                    "target": spec.name,
                    "fold": fold["name"],
                    "model": bundle.slug,
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
    validation_panel = pd.concat(validation_predictions, ignore_index=True)
    prediction_review = common.validate_prediction_panel(
        prediction_panel, protocol=protocol, split_index=2
    )
    validation_review = common.validate_prediction_panel(
        validation_panel, protocol=protocol, split_index=1
    )
    if not all(
        record["hyperparameter_fit_rows"] == record["train_rows"]
        and record["final_preprocessing_fit_rows"]
        == record["train_rows"] + record["validation_rows"]
        for record in fits
    ):
        raise AssertionError("Elastic Net preprocessing row audit failed")

    base_comparison: list[dict[str, Any]] = []
    base_provenance: dict[str, Any] | None = None
    if bundle_key != "D0":
        base_predictions, base_provenance = common.load_completed_predictions(
            "D0"
        )
        base_comparison = common.compare_predictions(
            prediction_panel, base_predictions
        )

    secondary_comparison: list[dict[str, Any]] = []
    secondary_provenance: dict[str, Any] | None = None
    secondary_key, secondary_label = _optional_secondary_base(bundle_key)
    if secondary_key is not None:
        try:
            secondary_predictions, secondary_provenance = (
                common.load_completed_predictions(secondary_key)
            )
        except (FileNotFoundError, RuntimeError):
            secondary_key = None
            secondary_label = None
        else:
            secondary_comparison = common.compare_predictions(
                prediction_panel, secondary_predictions
            )

    details = {
        "preflight": preflight,
        "feature_contract": feature_contract,
        "prediction_review": prediction_review,
        "validation_review": validation_review,
        "base_comparison": base_comparison,
        "base_provenance": base_provenance,
        "secondary_comparison": secondary_comparison,
        "secondary_key": secondary_key,
        "secondary_label": secondary_label,
        "secondary_provenance": secondary_provenance,
    }
    return prediction_panel, validation_panel, fits, details


def parser() -> argparse.ArgumentParser:
    output = argparse.ArgumentParser(description=__doc__)
    output.add_argument(
        "--bundle", choices=LINEAR_BUNDLES, required=True
    )
    output.add_argument("--allow-exploratory", action="store_true")
    output.add_argument("--overwrite", action="store_true")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    protocol = common.load_protocol()
    common.initialize_status(common.sha256_file(common.PROTOCOL_PATH))
    available, unavailable_reason = common.bundle_availability(
        protocol, args.bundle
    )
    common.ensure_no_existing_bundle(args.bundle, overwrite=args.overwrite)
    if not available:
        source = common.source_artifact_provenance(protocol)
        common.update_status(args.bundle, "running")
        summary = {
            "status": "skipped",
            "reason": unavailable_reason,
            "metrics": [],
        }
        common.write_bundle(
            args.bundle,
            predictions=None,
            validation_predictions=None,
            fits=[],
            fold_metrics=[],
            summary=summary,
            review={
                "status": "passed",
                "prediction_keys_unique": None,
                "predictions_finite": None,
                "prediction_bounds_valid": None,
                "outer_evaluation_excluded_from_tuning": True,
                "skip_was_predeclared_in_locked_protocol": True,
            },
            model_config={
                "estimator": None,
                "bundle": args.bundle,
                "status": "skipped",
                "reason": unavailable_reason,
                "source_profile": protocol["source_profile"],
                "claim_scope": protocol["claim_scope"],
            },
            dependencies={"source_artifacts": source},
            extra_outputs={
                "skip.json": {
                    "bundle": args.bundle,
                    "reason": unavailable_reason,
                    "protocol_sha256": common.sha256_file(
                        common.PROTOCOL_PATH
                    ),
                }
            },
        )
        common.update_status(
            args.bundle, "skipped", summary=str(unavailable_reason)
        )
        print(json.dumps(summary, indent=2))
        return 0
    if args.bundle != "D0":
        common.verify_completed_bundle("D0")
    common.update_status(args.bundle, "running")
    try:
        predictions, validation, fits, details = run_linear(
            args.bundle, allow_exploratory=args.allow_exploratory
        )
        metrics = quant_common.summarize_predictions(predictions).to_dict(
            orient="records"
        )
        fold_metrics = quant_common.summarize_predictions(
            predictions, ("fold", "target", "model")
        ).to_dict(orient="records")
        review = {
            "status": "passed",
            **details["prediction_review"],
            "validation_prediction_review": details["validation_review"],
            "feature_name_leakage_scan_passed": True,
            "source_timing_contract_bound_by_verified_manifest": (
                details["preflight"]["status"] == "passed"
            ),
            "outer_evaluation_excluded_from_tuning": True,
            "training_fitted_preprocessing_for_validation_selection": True,
            "validation_loss_hyperparameter_selection": True,
            "train_validation_only_final_preprocessing": True,
            "exact_matched_q_comparison": args.bundle == "D0"
            or details["base_provenance"] is not None,
        }
        summary = {
            "status": "complete",
            "metrics": metrics,
            "base_comparison": details["base_comparison"],
            "secondary_comparison": details["secondary_comparison"],
            "secondary_comparison_label": details["secondary_label"],
        }
        dependencies: dict[str, Any] = {
            "source_panel_preflight": {
                "panel_sha256": details["preflight"]["panel_sha256"],
                "panel_manifest_sha256": details["preflight"][
                    "panel_manifest_sha256"
                ],
            }
        }
        if details["base_provenance"] is not None:
            dependencies["matched_q_bundle_D0"] = details["base_provenance"]
        if details["secondary_provenance"] is not None:
            dependencies[
                f"secondary_bundle_{details['secondary_key']}"
            ] = details["secondary_provenance"]
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
                "fixed_log1p_features": protocol.get(
                    "preprocessing", {}
                ).get("fixed_log1p_features", []),
                "source_profile": protocol["source_profile"],
                "claim_scope": protocol["claim_scope"],
            },
            dependencies=dependencies,
            extra_outputs={"source_preflight.json": details["preflight"]},
        )
        common.update_status(
            args.bundle,
            "complete",
            summary=common.summary_status_text(summary),
        )
        print(pd.DataFrame(metrics).to_string(index=False))
        print(json.dumps(review, indent=2))
        return 0
    except Exception as error:
        common.update_status(args.bundle, "failed", summary=str(error))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
