#!/usr/bin/env python
"""Train one isolated v3 W17-Lite Elastic Net rung or control."""

from __future__ import annotations

import argparse
import json
from typing import Any, Sequence

import pandas as pd

from scripts.correlation_training import training_common as quant_common
from scripts.quant_deterministic_news_training_v2 import common as v2_common
from scripts.quant_deterministic_news_training_v3 import common, contract


def required_predecessors(bundle_key: str) -> tuple[str, ...]:
    if bundle_key == "S0":
        return ()
    if bundle_key == "S1":
        return ("S0",)
    if bundle_key == "S2":
        return ("S0",)
    if bundle_key == "S3":
        return ("S0", "S1", "S2")
    bundle = common.bundle_for(bundle_key)
    if bundle.control is not None:
        live = "S2" if bundle.architecture.startswith("ql") else "S3"
        matched = "S0" if live == "S2" else "S1"
        return (matched, live)
    raise ValueError(f"{bundle_key} is not a linear bundle")


def primary_base(bundle_key: str) -> str | None:
    if bundle_key in {"S1", "S2"}:
        return "S0"
    if bundle_key == "S3":
        return "S1"
    bundle = common.bundle_for(bundle_key)
    if bundle.control is not None:
        return "S2" if bundle.architecture.startswith("ql") else "S3"
    return None


def train_bundle(
    bundle_key: str,
    *,
    allow_failed_gate_exploratory: bool,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    list[dict[str, Any]],
    dict[str, Any],
]:
    protocol = common.load_protocol()
    panel, preflight = common.load_panel_and_preflight(
        protocol,
        bundle_key=bundle_key,
        allow_failed_gate_exploratory=allow_failed_gate_exploratory,
    )
    alphas = [
        float(value)
        for value in protocol["model_tuning"]["linear_alpha_grid"]
    ]
    ratios = [
        float(value)
        for value in protocol["model_tuning"][
            "elastic_net_l1_ratio_grid"
        ]
    ]
    predictions: list[pd.DataFrame] = []
    validation_predictions: list[pd.DataFrame] = []
    fits: list[dict[str, Any]] = []
    feature_contract: dict[str, Any] = {}
    bundle = common.bundle_for(bundle_key)

    for spec in quant_common.target_specs():
        features = common.model_features(bundle_key, spec.name, protocol)
        feature_contract[spec.name] = {
            "count": len(features),
            "ordered_sha256": common.ordered_sha256(features),
            "features": list(features),
        }
        transformed = common.apply_fixed_transforms(
            panel, features, protocol
        )
        for fold in protocol["folds"]:
            masks = quant_common.split_masks(transformed, spec, fold)
            train = transformed.loc[masks["train"]].copy()
            validation = transformed.loc[masks["validation"]].copy()
            test = transformed.loc[masks["test"]].copy()
            joined = pd.concat([train, validation], ignore_index=True)
            common.validate_training_feature_availability(
                train,
                features,
                f"{bundle_key}/{spec.name}/{fold['name']}/train",
            )
            common.validate_training_feature_availability(
                joined,
                features,
                f"{bundle_key}/{spec.name}/{fold['name']}/joined",
            )
            selected, candidates = common.select_linear_hyperparameters(
                train,
                validation,
                features,
                spec.response_column,
                alpha_grid=alphas,
                l1_ratios=ratios,
            )
            validation_model = quant_common.linear_pipeline(
                "elastic_net", **selected
            )
            validation_model.fit(
                train[list(features)], train[spec.response_column]
            )
            validation_predictions.append(
                quant_common.prediction_frame(
                    validation,
                    spec,
                    fold_name=fold["name"],
                    model_name=bundle.slug,
                    predicted_fisher=validation_model.predict(
                        validation[list(features)]
                    ),
                )
            )
            final_model = quant_common.linear_pipeline(
                "elastic_net", **selected
            )
            final_model.fit(
                joined[list(features)], joined[spec.response_column]
            )
            predictions.append(
                quant_common.prediction_frame(
                    test,
                    spec,
                    fold_name=fold["name"],
                    model_name=bundle.slug,
                    predicted_fisher=final_model.predict(
                        test[list(features)]
                    ),
                )
            )
            fits.append(
                {
                    "target": spec.name,
                    "fold": fold["name"],
                    "model": bundle.slug,
                    "features": list(features),
                    "feature_count": len(features),
                    "feature_ordered_sha256": common.ordered_sha256(
                        features
                    ),
                    "train_rows": len(train),
                    "validation_rows": len(validation),
                    "test_rows": len(test),
                    "hyperparameter_fit_rows": len(train),
                    "final_preprocessing_fit_rows": len(joined),
                    "selected_parameters": selected,
                    "validation_candidates": candidates,
                    "coefficients": (
                        quant_common.extract_linear_coefficients(
                            final_model, features
                        )
                    ),
                }
            )

    prediction_panel = pd.concat(predictions, ignore_index=True)
    validation_panel = pd.concat(
        validation_predictions, ignore_index=True
    )
    prediction_review = common.validate_prediction_panel(
        prediction_panel, protocol=protocol, split_index=2
    )
    validation_review = common.validate_prediction_panel(
        validation_panel, protocol=protocol, split_index=1
    )
    base_key = primary_base(bundle_key)
    base_comparison: list[dict[str, Any]] = []
    base_provenance: dict[str, Any] | None = None
    if base_key is not None:
        base, base_provenance = common.load_completed_predictions(base_key)
        base_comparison = v2_common.compare_predictions(
            prediction_panel, base
        )
    return prediction_panel, validation_panel, fits, {
        "preflight": preflight,
        "feature_contract": feature_contract,
        "prediction_review": prediction_review,
        "validation_review": validation_review,
        "base_key": base_key,
        "base_comparison": base_comparison,
        "base_provenance": base_provenance,
    }


def parser() -> argparse.ArgumentParser:
    output = argparse.ArgumentParser(description=__doc__)
    output.add_argument(
        "--bundle", choices=contract.LINEAR_BUNDLES, required=True
    )
    output.add_argument(
        "--allow-failed-gate-exploratory", action="store_true"
    )
    output.add_argument("--overwrite", action="store_true")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    common.load_protocol()
    common.initialize_status()
    for predecessor in required_predecessors(args.bundle):
        common.verify_completed_bundle(predecessor)
    common.ensure_no_existing_bundle(
        args.bundle, overwrite=args.overwrite
    )
    common.update_status(args.bundle, "running")
    try:
        predictions, validation, fits, details = train_bundle(
            args.bundle,
            allow_failed_gate_exploratory=(
                args.allow_failed_gate_exploratory
            ),
        )
        metrics = quant_common.summarize_predictions(
            predictions
        ).to_dict(orient="records")
        fold_metrics = quant_common.summarize_predictions(
            predictions, ("fold", "target", "model")
        ).to_dict(orient="records")
        summary = {
            "status": "complete",
            "metrics": metrics,
            "base_key": details["base_key"],
            "base_comparison": details["base_comparison"],
            "semantic_gate_passed": False,
            "result_role": "exploratory_failed_semantic_gate_diagnostic",
        }
        review = {
            "status": "passed",
            **details["prediction_review"],
            "validation_prediction_review": details[
                "validation_review"
            ],
            "outer_evaluation_excluded_from_tuning": True,
            "training_fitted_preprocessing_for_validation_selection": True,
            "validation_loss_hyperparameter_selection": True,
            "train_validation_only_final_preprocessing": True,
            "source_hashes_verified": True,
            "semantic_gate_passed": False,
            "failed_semantic_gate_override_confirmed": True,
        }
        dependencies: dict[str, Any] = {
            "source_panel_preflight": details["preflight"]
        }
        if details["base_provenance"] is not None:
            dependencies[
                f"matched_base_{details['base_key']}"
            ] = details["base_provenance"]
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
                "alpha_grid": protocol["model_tuning"][
                    "linear_alpha_grid"
                ],
                "l1_ratio_grid": protocol["model_tuning"][
                    "elastic_net_l1_ratio_grid"
                ],
                "fixed_log1p_features": protocol["preprocessing"][
                    "fixed_log1p_features"
                ],
                "control": details["preflight"].get("control_audit"),
                "semantic_gate_passed": False,
                "result_role": (
                    "exploratory_failed_semantic_gate_diagnostic"
                ),
            },
            dependencies=dependencies,
            extra_outputs={
                "source_preflight.json": details["preflight"]
            },
        )
        common.update_status(
            args.bundle,
            "complete",
            common.summary_status_text(summary),
        )
        print(pd.DataFrame(metrics).to_string(index=False))
        print(json.dumps(review, indent=2))
        return 0
    except Exception as error:
        common.update_status(args.bundle, "failed", str(error))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
