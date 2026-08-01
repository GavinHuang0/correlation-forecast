#!/usr/bin/env python
"""Train one long-history quant-anchored v4 stack or residual correction."""

from __future__ import annotations

import argparse
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from scripts.correlation_training import training_common as quant_common
from scripts.quant_deterministic_news_training_v4 import common, contract


LONG_KEYS = tuple(bundle.key for bundle in contract.LONG_BUNDLES)


def select_residual_parameters(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: Sequence[str],
    protocol: Mapping[str, Any],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Jointly select Elastic Net and correction shrinkage on validation."""

    candidates: list[dict[str, Any]] = []
    actual = validation["_actual_fisher_z"].to_numpy(dtype=float)
    base = validation["_base_fisher_z"].to_numpy(dtype=float)
    for alpha in protocol["model_tuning"]["alpha_grid"]:
        for l1_ratio in protocol["model_tuning"]["l1_ratio_grid"]:
            model = quant_common.linear_pipeline(
                "elastic_net",
                alpha=float(alpha),
                l1_ratio=float(l1_ratio),
            )
            model.fit(
                train[list(features)], train["_residual_fisher_z"]
            )
            correction = model.predict(validation[list(features)])
            for shrinkage in protocol["model_tuning"][
                "residual_correction_shrinkage_grid"
            ]:
                predicted = base + float(shrinkage) * correction
                candidates.append(
                    {
                        "alpha": float(alpha),
                        "l1_ratio": float(l1_ratio),
                        "correction_shrinkage": float(shrinkage),
                        "validation_mse": float(
                            np.mean(np.square(actual - predicted))
                        ),
                    }
                )
    candidates.sort(
        key=lambda row: (
            row["validation_mse"],
            row["alpha"],
            row["l1_ratio"],
            row["correction_shrinkage"],
        )
    )
    selected = {
        name: float(candidates[0][name])
        for name in ("alpha", "l1_ratio", "correction_shrinkage")
    }
    return selected, candidates


def _fit_one_split(
    bundle_key: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    features: Sequence[str],
    protocol: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], list[dict[str, Any]], Any]:
    bundle = common.bundle_for(bundle_key)
    joined = pd.concat([train, validation], ignore_index=True)
    if bundle_key == "R0":
        selected: dict[str, Any] = {"identity_anchor": True}
        candidates: list[dict[str, Any]] = []
        return (
            validation["_base_fisher_z"].to_numpy(dtype=float),
            test["_base_fisher_z"].to_numpy(dtype=float),
            selected,
            candidates,
            None,
        )

    common.validate_training_features(train, features, f"{bundle_key}/train")
    common.validate_training_features(joined, features, f"{bundle_key}/joined")
    if bundle_key == "RCAL":
        validation_model = quant_common.linear_pipeline("ols")
        validation_model.fit(
            train[list(features)], train["_actual_fisher_z"]
        )
        validation_prediction = validation_model.predict(
            validation[list(features)]
        )
        final_model = quant_common.linear_pipeline("ols")
        final_model.fit(joined[list(features)], joined["_actual_fisher_z"])
        test_prediction = final_model.predict(test[list(features)])
        selected = {"estimator": "ols"}
        candidates = []
    elif bundle.features in {"c6", "l19", "quality5"}:
        selected, candidates = select_residual_parameters(
            train, validation, features, protocol
        )
        model_parameters = {
            "alpha": selected["alpha"],
            "l1_ratio": selected["l1_ratio"],
        }
        validation_model = quant_common.linear_pipeline(
            "elastic_net", **model_parameters
        )
        validation_model.fit(
            train[list(features)], train["_residual_fisher_z"]
        )
        validation_prediction = (
            validation["_base_fisher_z"].to_numpy(dtype=float)
            + selected["correction_shrinkage"]
            * validation_model.predict(validation[list(features)])
        )
        final_model = quant_common.linear_pipeline(
            "elastic_net", **model_parameters
        )
        final_model.fit(
            joined[list(features)], joined["_residual_fisher_z"]
        )
        test_prediction = (
            test["_base_fisher_z"].to_numpy(dtype=float)
            + selected["correction_shrinkage"]
            * final_model.predict(test[list(features)])
        )
    elif bundle_key == "RSTACK":
        selected, candidates = common.select_elastic_net(
            train,
            validation,
            features,
            "_actual_fisher_z",
            protocol,
        )
        validation_model = quant_common.linear_pipeline(
            "elastic_net", **selected
        )
        validation_model.fit(
            train[list(features)], train["_actual_fisher_z"]
        )
        validation_prediction = validation_model.predict(
            validation[list(features)]
        )
        final_model = quant_common.linear_pipeline(
            "elastic_net", **selected
        )
        final_model.fit(
            joined[list(features)], joined["_actual_fisher_z"]
        )
        test_prediction = final_model.predict(test[list(features)])
    else:
        raise ValueError(f"Unsupported long bundle {bundle_key}")
    return (
        np.asarray(validation_prediction, dtype=float),
        np.asarray(test_prediction, dtype=float),
        selected,
        candidates,
        final_model,
    )


def train_long_bundle(
    bundle_key: str,
    protocol: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    bundle = common.bundle_for(bundle_key)
    if bundle.family != "long":
        raise ValueError(f"{bundle_key} is not a long-history bundle")
    panel, panel_audit = common.load_panel(protocol, bundle_key)
    base, base_audit = common.load_long_base(protocol, panel)
    features = common.model_features(bundle_key, contract.TARGETS[0], protocol)
    predictions: list[pd.DataFrame] = []
    validation_predictions: list[pd.DataFrame] = []
    fits: list[dict[str, Any]] = []

    for target in contract.TARGETS:
        for test_fold in contract.LONG_TEST_FOLDS:
            train, validation, test, split_audit = common.long_split(
                base, target, test_fold
            )
            (
                validation_prediction,
                test_prediction,
                selected,
                candidates,
                final_model,
            ) = _fit_one_split(
                bundle_key,
                train,
                validation,
                test,
                features,
                protocol,
            )
            validation_predictions.append(
                common.long_prediction_frame(
                    validation,
                    validation_prediction,
                    model_name=bundle.slug,
                    outer_fold=test_fold,
                )
            )
            predictions.append(
                common.long_prediction_frame(
                    test,
                    test_prediction,
                    model_name=bundle.slug,
                    outer_fold=test_fold,
                )
            )
            coefficient_rows: list[dict[str, float]] = []
            if final_model is not None:
                coefficient_rows = quant_common.extract_linear_coefficients(
                    final_model, features
                )
            fits.append(
                {
                    "target": target,
                    "fold": test_fold,
                    "model": bundle.slug,
                    "architecture": bundle.features,
                    "features": list(features),
                    "feature_count": len(features),
                    "feature_ordered_sha256": common.ordered_sha256(features),
                    **split_audit,
                    "selected_parameters": selected,
                    "validation_candidates": candidates,
                    "coefficients": coefficient_rows,
                    "base_coefficient_fixed_at_one": bundle.features in {"c6", "l19", "quality5"},
                    "preprocessing_tuning_fit_scope": "earlier_oos_train_folds_only",
                    "preprocessing_final_fit_scope": "earlier_oos_train_plus_immediately_prior_validation",
                }
            )

    prediction_panel = pd.concat(predictions, ignore_index=True)
    validation_panel = pd.concat(validation_predictions, ignore_index=True)
    prediction_review = common.validate_prediction_panel(
        prediction_panel, expected_folds=contract.LONG_TEST_FOLDS
    )
    validation_review = common.validate_prediction_panel(
        validation_panel, expected_folds=contract.LONG_TEST_FOLDS
    )
    return prediction_panel, validation_panel, fits, {
        "panel": panel_audit,
        "base": base_audit,
        "prediction_review": prediction_review,
        "validation_review": validation_review,
        "features": list(features),
        "feature_ordered_sha256": common.ordered_sha256(features),
    }


def parser() -> argparse.ArgumentParser:
    output = argparse.ArgumentParser(description=__doc__)
    output.add_argument("--bundle", choices=LONG_KEYS, required=True)
    output.add_argument("--overwrite", action="store_true")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    protocol = common.load_protocol()
    common.ensure_output_absent(args.bundle, overwrite=args.overwrite)
    predictions, validation, fits, details = train_long_bundle(
        args.bundle, protocol
    )
    bundle = common.bundle_for(args.bundle)
    common.write_bundle(
        args.bundle,
        predictions=predictions,
        validation_predictions=validation,
        fits=fits,
        model_config={
            "bundle": args.bundle,
            "architecture": bundle.features,
            "control": bundle.control,
            "features": details["features"],
            "feature_ordered_sha256": details["feature_ordered_sha256"],
            "alpha_grid": protocol["model_tuning"]["alpha_grid"],
            "l1_ratio_grid": protocol["model_tuning"]["l1_ratio_grid"],
            "correction_shrinkage_grid": protocol["model_tuning"][
                "residual_correction_shrinkage_grid"
            ],
            "prequential_test_folds": list(contract.LONG_TEST_FOLDS),
            "base_model": contract.LONG_ANCHOR_MODEL,
        },
        dependencies={
            "source_panel": details["panel"],
            "quant_v2_oos_anchor": details["base"],
            "prediction_review": details["prediction_review"],
            "validation_review": details["validation_review"],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
