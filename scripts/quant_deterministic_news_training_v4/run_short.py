#!/usr/bin/env python
"""Train one short-history v4 Elastic Net rung or falsification control."""

from __future__ import annotations

import argparse
from typing import Any, Sequence

import pandas as pd

from scripts.correlation_training import training_common as quant_common
from scripts.quant_deterministic_news_training_v4 import common, contract


SHORT_KEYS = tuple(bundle.key for bundle in contract.SHORT_BUNDLES)


def matched_base_key(bundle_key: str) -> str:
    bundle = common.bundle_for(bundle_key)
    return "S1" if bundle.features in {"q_d2_l19", "q_d2_quality5"} else "S0"


def train_short_bundle(
    bundle_key: str,
    protocol: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    bundle = common.bundle_for(bundle_key)
    if bundle.family != "short":
        raise ValueError(f"{bundle_key} is not a short-history bundle")
    panel, panel_audit = common.load_panel(protocol, bundle_key)
    predictions: list[pd.DataFrame] = []
    validation_predictions: list[pd.DataFrame] = []
    fits: list[dict[str, Any]] = []
    feature_contract: dict[str, Any] = {}

    for spec in quant_common.target_specs():
        features = common.model_features(bundle_key, spec.name, protocol)
        feature_contract[spec.name] = {
            "count": len(features),
            "features": list(features),
            "ordered_sha256": common.ordered_sha256(features),
        }
        transformed = common.apply_fixed_transforms(panel, features, protocol)
        for fold in contract.SHORT_FOLDS:
            masks = quant_common.split_masks(transformed, spec, fold)
            train = transformed.loc[masks["train"]].copy()
            validation = transformed.loc[masks["validation"]].copy()
            test = transformed.loc[masks["test"]].copy()
            joined = pd.concat([train, validation], ignore_index=True)
            common.validate_training_features(
                train, features, f"{bundle_key}/{spec.name}/{fold['name']}/train"
            )
            common.validate_training_features(
                joined, features, f"{bundle_key}/{spec.name}/{fold['name']}/joined"
            )
            selected, candidates = common.select_elastic_net(
                train,
                validation,
                features,
                spec.response_column,
                protocol,
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
                    predicted_fisher=final_model.predict(test[list(features)]),
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
                    "selected_parameters": selected,
                    "validation_candidates": candidates,
                    "coefficients": quant_common.extract_linear_coefficients(
                        final_model, features
                    ),
                    "preprocessing_tuning_fit_scope": "train_only",
                    "preprocessing_final_fit_scope": "train_plus_validation",
                }
            )

    prediction_panel = pd.concat(predictions, ignore_index=True)
    validation_panel = pd.concat(validation_predictions, ignore_index=True)
    prediction_review = common.validate_prediction_panel(
        prediction_panel,
        expected_folds=tuple(fold["name"] for fold in contract.SHORT_FOLDS),
    )
    validation_review = common.validate_prediction_panel(
        validation_panel,
        expected_folds=tuple(fold["name"] for fold in contract.SHORT_FOLDS),
    )
    base_key = matched_base_key(bundle_key)
    base, base_provenance = common.load_verified_v3_base(protocol, base_key)
    base_review = common.validate_matched_prediction_rows(prediction_panel, base)
    return prediction_panel, validation_panel, fits, {
        "panel": panel_audit,
        "feature_contract": feature_contract,
        "prediction_review": prediction_review,
        "validation_review": validation_review,
        "matched_base_key": base_key,
        "matched_base": base_provenance,
        "matched_base_review": base_review,
    }


def parser() -> argparse.ArgumentParser:
    output = argparse.ArgumentParser(description=__doc__)
    output.add_argument("--bundle", choices=SHORT_KEYS, required=True)
    output.add_argument("--overwrite", action="store_true")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    protocol = common.load_protocol()
    common.ensure_output_absent(args.bundle, overwrite=args.overwrite)
    predictions, validation, fits, details = train_short_bundle(
        args.bundle, protocol
    )
    bundle = common.bundle_for(args.bundle)
    common.write_bundle(
        args.bundle,
        predictions=predictions,
        validation_predictions=validation,
        fits=fits,
        model_config={
            "estimator": "elastic_net",
            "bundle": args.bundle,
            "architecture": bundle.features,
            "control": bundle.control,
            "feature_contract": details["feature_contract"],
            "alpha_grid": protocol["model_tuning"]["alpha_grid"],
            "l1_ratio_grid": protocol["model_tuning"]["l1_ratio_grid"],
            "matched_base_key": details["matched_base_key"],
        },
        dependencies={
            "source_panel": details["panel"],
            "matched_v3_base": details["matched_base"],
            "matched_base_review": details["matched_base_review"],
            "prediction_review": details["prediction_review"],
            "validation_review": details["validation_review"],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
