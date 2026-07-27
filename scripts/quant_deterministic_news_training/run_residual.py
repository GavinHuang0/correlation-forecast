"""Train one prequential residual-correction bundle from Track B."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from scripts.correlation_training import run_rung_03 as quant_xgb
from scripts.correlation_training import training_common as quant_common
from scripts.quant_deterministic_news_training import common


@dataclass(frozen=True)
class ResidualBundle:
    method: str
    base_family: str
    control: str | None
    targets: tuple[str, ...]


ALL_TARGETS = ("t1_etf", "t1_loo", "t2_etf", "t2_loo")
RESIDUAL_BUNDLES = {
    **{
        f"B{index}": ResidualBundle(
            method=f"B{index}",
            base_family="xgboost",
            control=None,
            targets=ALL_TARGETS,
        )
        for index in range(6)
    },
    **{
        f"S-B{index}": ResidualBundle(
            method=f"B{index}",
            base_family="winner_t1_etf",
            control=None,
            targets=("t1_etf",),
        )
        for index in range(6)
    },
    "C-B4-L20": ResidualBundle(
        method="B4",
        base_family="xgboost",
        control="lag20",
        targets=ALL_TARGETS,
    ),
    "C-B4-WS": ResidualBundle(
        method="B4",
        base_family="xgboost",
        control="wrong_stock",
        targets=ALL_TARGETS,
    ),
}


def _build_residual_panel(
    protocol: dict[str, Any],
    panel: pd.DataFrame,
    bundle: ResidualBundle,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    panel, control_audit = common.prepare_control_panel(
        panel, protocol, bundle.control
    )
    base = common.load_frozen_predictions(
        protocol, bundle.base_family
    )
    base = base[base["target"].isin(bundle.targets)].copy()
    common.validate_frozen_actuals(base, panel)
    d43 = list(common.deterministic_features(protocol))
    payload = [
        *common.PANEL_KEYS,
        *d43,
        "_control_available",
        "target_etf_t2_end_date",
        "target_loo_t2_end_date",
        "observed_no_relevant_news",
    ]
    payload = list(dict.fromkeys(payload))
    merged = base.merge(
        panel[payload],
        on=common.PANEL_KEYS,
        how="left",
        validate="many_to_one",
    )
    if len(merged) != len(base):
        raise ValueError("Residual feature join changed base row count")
    merged = merged.rename(
        columns={"predicted_fisher_z": "base_predicted_fisher_z"}
    )
    merged["quant_residual_fisher_z"] = (
        merged["actual_fisher_z"] - merged["base_predicted_fisher_z"]
    )
    merged["predicted_fisher_z"] = merged["base_predicted_fisher_z"]
    return merged, control_audit


def _method_features(
    method: str, protocol: dict[str, Any]
) -> tuple[str, ...]:
    d43 = common.deterministic_features(protocol)
    if method in {"B0", "B1"}:
        return ()
    if method == "B2":
        return ("base_predicted_fisher_z",)
    if method == "B3":
        return d43
    if method in {"B4", "B5"}:
        return ("base_predicted_fisher_z", *d43)
    raise ValueError(f"Unknown residual method {method}")


def _development_masks(
    frame: pd.DataFrame, spec: quant_common.TargetSpec
) -> tuple[pd.Series, pd.Series]:
    fold1 = frame["fold"].eq("fold_1")
    available = frame["_control_available"].fillna(False)
    train = (
        fold1
        & available
        & frame["forecast_date"].between(
            "2025-01-01", "2025-03-31", inclusive="both"
        )
    )
    validation = (
        fold1
        & available
        & frame["forecast_date"].between(
            "2025-04-01", "2025-06-30", inclusive="both"
        )
    )
    if spec.target_end_column is not None:
        train &= frame[spec.target_end_column].notna()
        train &= frame[spec.target_end_column].le(pd.Timestamp("2025-03-31"))
        validation &= frame[spec.target_end_column].notna()
        validation &= frame[spec.target_end_column].le(
            pd.Timestamp("2025-06-30")
        )
    return train, validation


def _history_for_fold(
    frame: pd.DataFrame, prediction_fold: str
) -> pd.DataFrame:
    allowed = (
        ["fold_1"] if prediction_fold == "fold_2" else ["fold_1", "fold_2"]
    )
    return frame[
        frame["fold"].isin(allowed)
        & frame["_control_available"].fillna(False)
    ].copy()


def _correction_validation_frame(
    validation: pd.DataFrame,
    *,
    model_name: str,
    correction: np.ndarray,
) -> pd.DataFrame:
    return common.residual_prediction_frame(
        validation, model_name=model_name, correction=correction
    )


def _select_xgboost(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: Sequence[str],
    protocol: dict[str, Any],
    *,
    device: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    configs = common.xgboost_candidate_configs(protocol)
    candidates = []
    predictions = []
    for candidate_id, config in enumerate(configs):
        model = quant_xgb.xgb_model(
            config,
            n_estimators=int(
                protocol["model_tuning"]["xgboost_max_estimators"]
            ),
            early_stopping_rounds=int(
                protocol["model_tuning"]["xgboost_early_stopping_rounds"]
            ),
            device=device,
        )
        model.fit(
            train[list(features)],
            train["quant_residual_fisher_z"],
            eval_set=[
                (
                    validation[list(features)],
                    validation["quant_residual_fisher_z"],
                )
            ],
            verbose=False,
        )
        prediction = model.predict(validation[list(features)])
        predictions.append(prediction)
        candidates.append(
            {
                "candidate_id": candidate_id,
                "config": config,
                "best_iteration": int(model.best_iteration),
                "validation_mse": quant_common.mse(
                    validation["quant_residual_fisher_z"].to_numpy(
                        dtype=float
                    ),
                    prediction,
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
    return candidates[position], candidates, predictions[position]


def run_residual(
    bundle_key: str,
    *,
    allow_exploratory: bool,
    device: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    list[dict[str, Any]],
    pd.DataFrame,
    dict[str, Any],
]:
    protocol = common.load_protocol()
    panel, _ = common.load_panel_and_preflight(
        protocol, allow_exploratory=allow_exploratory
    )
    bundle = RESIDUAL_BUNDLES[bundle_key]
    residual_panel, control_audit = _build_residual_panel(
        protocol, panel, bundle
    )
    features = _method_features(bundle.method, protocol)
    residual_panel = common.apply_fixed_transforms(
        residual_panel, features, protocol
    )
    spec_by_name = {
        spec.name: spec for spec in quant_common.target_specs()
    }
    predictions = []
    validation_predictions = []
    fits = []
    training_index = []
    settings_by_target: dict[str, dict[str, Any]] = {}
    for target in bundle.targets:
        spec = spec_by_name[target]
        target_frame = residual_panel[
            residual_panel["target"].eq(target)
        ].copy()
        train_mask, validation_mask = _development_masks(target_frame, spec)
        development_train = target_frame.loc[train_mask].copy()
        development_validation = target_frame.loc[validation_mask].copy()
        expected_dev = protocol["track_b_residual_reuse"][
            "training_design"
        ]["hyperparameter_development"]["expected_rows_per_target"]
        horizon = spec.horizon
        expected_train, expected_validation = expected_dev[
            f"{horizon}_train_validation"
        ]
        if (
            len(development_train) != expected_train
            or len(development_validation) != expected_validation
        ):
            raise ValueError(
                f"Residual development counts failed for {target}: "
                f"{len(development_train)}/{len(development_validation)}"
            )

        selected: dict[str, Any] = {}
        candidates: list[dict[str, Any]] = []
        if bundle.method == "B0":
            validation_correction = np.zeros(len(development_validation))
        elif bundle.method == "B1":
            selected = {
                "development_train_mean_error": float(
                    development_train["quant_residual_fisher_z"].mean()
                )
            }
            validation_correction = np.full(
                len(development_validation),
                selected["development_train_mean_error"],
            )
        elif bundle.method == "B2":
            validation_model = quant_common.linear_pipeline("ols")
            validation_model.fit(
                development_train[list(features)],
                development_train["quant_residual_fisher_z"],
            )
            validation_correction = validation_model.predict(
                development_validation[list(features)]
            )
            selected = {"estimator": "ols"}
        elif bundle.method in {"B3", "B4"}:
            selected, candidates = common.select_linear_hyperparameters(
                development_train,
                development_validation,
                features,
                "quant_residual_fisher_z",
                alpha_grid=protocol["model_tuning"]["linear_alpha_grid"],
                l1_ratios=protocol["model_tuning"][
                    "elastic_net_l1_ratio_grid"
                ],
            )
            validation_model = quant_common.linear_pipeline(
                "elastic_net", **selected
            )
            validation_model.fit(
                development_train[list(features)],
                development_train["quant_residual_fisher_z"],
            )
            validation_correction = validation_model.predict(
                development_validation[list(features)]
            )
        elif bundle.method == "B5":
            selected, candidates, validation_correction = _select_xgboost(
                development_train,
                development_validation,
                features,
                protocol,
                device=device,
            )
        else:
            raise AssertionError(bundle.method)
        settings_by_target[target] = selected
        validation_predictions.append(
            _correction_validation_frame(
                development_validation,
                model_name=common.bundle_for(bundle_key).slug,
                correction=validation_correction,
            )
        )

        for prediction_fold in ("fold_2", "fold_3"):
            history = _history_for_fold(target_frame, prediction_fold)
            evaluation = target_frame[
                target_frame["fold"].eq(prediction_fold)
                & target_frame["_control_available"].fillna(False)
            ].copy()
            if bundle.method == "B0":
                correction = np.zeros(len(evaluation))
                coefficients = []
                cuda_confirmed = None
            elif bundle.method == "B1":
                correction = np.full(
                    len(evaluation),
                    float(history["quant_residual_fisher_z"].mean()),
                )
                coefficients = []
                cuda_confirmed = None
            elif bundle.method == "B2":
                correction, final_model = common.fit_predict_linear(
                    "ols",
                    history,
                    evaluation,
                    features,
                    "quant_residual_fisher_z",
                )
                coefficients = quant_common.extract_linear_coefficients(
                    final_model, features
                )
                cuda_confirmed = None
            elif bundle.method in {"B3", "B4"}:
                correction, final_model = common.fit_predict_linear(
                    "elastic_net",
                    history,
                    evaluation,
                    features,
                    "quant_residual_fisher_z",
                    selected,
                )
                coefficients = quant_common.extract_linear_coefficients(
                    final_model, features
                )
                cuda_confirmed = None
            else:
                final_model = quant_xgb.xgb_model(
                    selected["config"],
                    n_estimators=int(selected["best_iteration"]) + 1,
                    early_stopping_rounds=None,
                    device=device,
                )
                final_model.fit(
                    history[list(features)],
                    history["quant_residual_fisher_z"],
                    verbose=False,
                )
                correction = final_model.predict(evaluation[list(features)])
                coefficients = []
                cuda_confirmed = (
                    quant_xgb.booster_uses_cuda(final_model)
                    if device == "cuda"
                    else False
                )
            predictions.append(
                common.residual_prediction_frame(
                    evaluation,
                    model_name=common.bundle_for(bundle_key).slug,
                    correction=correction,
                )
            )
            if bundle.method != "B0":
                index = history[
                    [
                        "fold",
                        "target",
                        "forecast_date",
                        "sector",
                        "stock",
                        "benchmark",
                    ]
                ].copy()
                index = index.rename(columns={"fold": "source_outer_fold"})
                index["prediction_fold"] = prediction_fold
                training_index.append(index)
            fits.append(
                {
                    "target": target,
                    "prediction_fold": prediction_fold,
                    "method": bundle.method,
                    "base_family": bundle.base_family,
                    "features": list(features),
                    "feature_count": len(features),
                    "feature_ordered_sha256": common.ordered_sha256(features),
                    "development_train_rows": len(development_train),
                    "development_validation_rows": len(
                        development_validation
                    ),
                    "history_fit_rows": len(history),
                    "evaluation_rows": len(evaluation),
                    "selected_settings_frozen_from_q1_q2": selected,
                    "validation_candidates": candidates,
                    "coefficients": coefficients,
                    "cuda_confirmed": cuda_confirmed,
                }
            )
    prediction_panel = pd.concat(predictions, ignore_index=True)
    validation_panel = pd.concat(
        validation_predictions, ignore_index=True
    )
    validation_expected = {
        target: {
            "fold_1": 1_860 if target.startswith("t1_") else 1_740
        }
        for target in bundle.targets
    }
    validation_review = common.validate_prediction_panel(
        validation_panel,
        expected_folds=["fold_1"],
        expected_targets=list(bundle.targets),
        expected_rows=validation_expected,
    )
    if training_index:
        training_index_panel = pd.concat(training_index, ignore_index=True)
    else:
        training_index_panel = pd.DataFrame(
            columns=[
                "source_outer_fold",
                "target",
                "forecast_date",
                "sector",
                "stock",
                "benchmark",
                "prediction_fold",
            ]
        )
    expected = common.track_b_expected_rows()
    review = common.validate_prediction_panel(
        prediction_panel,
        expected_folds=["fold_2", "fold_3"],
        expected_targets=list(bundle.targets),
        expected_rows={target: expected[target] for target in bundle.targets},
    )
    if bundle.method == "B5" and device == "cuda":
        if not all(record["cuda_confirmed"] for record in fits):
            raise AssertionError("Not every B5 final fit used CUDA")
    base = common.load_frozen_predictions(protocol, bundle.base_family)
    base = base[
        base["target"].isin(bundle.targets)
        & base["fold"].isin(["fold_2", "fold_3"])
    ].copy()
    base_comparison = common.summarize_with_base(prediction_panel, base)
    parity = None
    if bundle.method == "B0":
        parity_frame = prediction_panel.copy()
        parity_frame["predicted_fisher_z"] = parity_frame[
            "base_predicted_fisher_z"
        ]
        parity_frame["predicted_correlation"] = np.tanh(
            parity_frame["predicted_fisher_z"]
        )
        parity = common.parity_statistics(parity_frame, base)
        if parity["maximum_absolute_fisher_z_difference"] > 0:
            raise AssertionError(f"B0 is not bit-identical: {parity}")
    chronology_passed = True
    if not training_index_panel.empty:
        order = {"fold_1": 1, "fold_2": 2, "fold_3": 3}
        chronology_passed = all(
            order[source] < order[prediction]
            for source, prediction in zip(
                training_index_panel["source_outer_fold"],
                training_index_panel["prediction_fold"],
                strict=True,
            )
        )
    if not chronology_passed:
        raise AssertionError("Residual training uses a non-earlier outer fold")
    details = {
        "bundle": bundle,
        "features": features,
        "feature_ordered_sha256": common.ordered_sha256(features),
        "control_audit": control_audit,
        "prediction_review": review,
        "validation_review": validation_review,
        "base_comparison": base_comparison,
        "parity": parity,
        "chronology_passed": chronology_passed,
        "settings_by_target": settings_by_target,
    }
    return (
        prediction_panel,
        validation_panel,
        fits,
        training_index_panel,
        details,
    )


def parser() -> argparse.ArgumentParser:
    output = argparse.ArgumentParser(description=__doc__)
    output.add_argument(
        "--bundle", choices=sorted(RESIDUAL_BUNDLES), required=True
    )
    output.add_argument("--allow-exploratory", action="store_true")
    output.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    output.add_argument("--overwrite", action="store_true")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    common.require_completed_preflight()
    common.ensure_no_existing_bundle(args.bundle, overwrite=args.overwrite)
    common.update_status(args.bundle, "running")
    try:
        (
            predictions,
            validation,
            fits,
            training_index,
            details,
        ) = run_residual(
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
            "earlier_out_of_sample_residuals_only": details[
                "chronology_passed"
            ],
            "settings_selected_on_fold1_q1_q2_only": True,
            "settings_fixed_for_fold2_and_fold3": True,
            "requested_device": args.device,
            "cuda_confirmed_all_final_fits": (
                all(record["cuda_confirmed"] for record in fits)
                if details["bundle"].method == "B5"
                and args.device == "cuda"
                else None
            ),
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
        base_path = (
            protocol["isolation"]["quant_rung_3_predictions"]
        )
        common.write_bundle(
            args.bundle,
            predictions=predictions,
            validation_predictions=validation,
            fits=fits,
            fold_metrics=fold_metrics,
            summary=summary,
            review=review,
            model_config={
                "method": details["bundle"].method,
                "base_family": details["bundle"].base_family,
                "targets": list(details["bundle"].targets),
                "features": list(details["features"]),
                "feature_ordered_sha256": details[
                    "feature_ordered_sha256"
                ],
                "settings_by_target": details["settings_by_target"],
                "control": details["control_audit"],
                "device": args.device,
            },
            dependencies={
                "preflight": (
                    common.bundle_for("PRE").output_path / "manifest.json"
                ).as_posix(),
                "base_predictions": base_path,
                "base_predictions_sha256": protocol["source_artifacts"][
                    "quant_rung_3_predictions_sha256"
                ],
            },
            extra_outputs={
                "residual_training_index.parquet": training_index,
                "base_prediction_reference.json": {
                    "path": base_path,
                    "sha256": protocol["source_artifacts"][
                        "quant_rung_3_predictions_sha256"
                    ],
                    "base_family": details["bundle"].base_family,
                },
            },
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
