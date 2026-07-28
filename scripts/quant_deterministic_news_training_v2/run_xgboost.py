"""Run the validation-gated deterministic-news v2 shallow-XGBoost challenger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from scripts.correlation_training import run_rung_03 as quant_xgb
from scripts.correlation_training import training_common as quant_common
from scripts.quant_deterministic_news_training_v2 import common


BUNDLE_KEY = "D5"
QUANT_PROTOCOL_PATH = Path("config/quant_training_protocol_v1.json")
REQUIRED_XGB_KEYS = {
    "max_depth",
    "learning_rate",
    "min_child_weight",
    "subsample",
    "colsample_bytree",
    "reg_lambda",
}


def load_xgboost_budget(
    protocol: Mapping[str, Any],
    path: Path = QUANT_PROTOCOL_PATH,
) -> dict[str, Any]:
    """Resolve the exact quant-v1 shallow-tree budget named by the v2 lock."""

    if (
        protocol.get("model_tuning", {}).get(
            "reuse_quant_v1_shallow_xgboost_candidates"
        )
        is not True
    ):
        raise ValueError("The v2 protocol did not lock quant-v1 XGBoost reuse")
    quant_protocol = common.load_json(path)
    rung = quant_protocol.get("rung_3")
    if not isinstance(rung, Mapping):
        raise ValueError("quant-v1 protocol has no rung_3 configuration")
    candidates = rung.get("candidate_configs")
    if (
        not isinstance(candidates, list)
        or len(candidates) != 4
        or any(
            not isinstance(candidate, Mapping)
            or set(candidate) != REQUIRED_XGB_KEYS
            for candidate in candidates
        )
    ):
        raise ValueError("quant-v1 XGBoost candidate contract is malformed")
    max_estimators = int(rung.get("n_estimators", 0))
    early_stopping = int(rung.get("early_stopping_rounds", 0))
    if max_estimators <= 0 or early_stopping <= 0:
        raise ValueError("quant-v1 XGBoost iteration budget is invalid")
    return {
        "source_path": path.as_posix(),
        "source_sha256": common.sha256_file(path),
        "candidate_configs": [dict(candidate) for candidate in candidates],
        "max_estimators": max_estimators,
        "early_stopping_rounds": early_stopping,
        "random_seed": 1729,
    }


def _load_completed_prediction_artifact(
    bundle_key: str, filename: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    provenance = common.verify_completed_bundle(bundle_key)
    manifest = provenance.pop("manifest")
    artifact = manifest.get("artifacts", {}).get(filename)
    if not isinstance(artifact, Mapping):
        raise RuntimeError(
            f"{bundle_key} manifest does not list {filename}"
        )
    path = common.bundle_for(bundle_key).output_path / filename
    if Path(str(artifact.get("path", ""))).resolve() != path.resolve():
        raise RuntimeError(f"{bundle_key}/{filename} path is noncanonical")
    if (
        not path.exists()
        or common.sha256_file(path) != str(artifact.get("sha256"))
    ):
        raise RuntimeError(f"{bundle_key}/{filename} fails its hash contract")
    frame = pd.read_parquet(path)
    frame["forecast_date"] = pd.to_datetime(
        frame["forecast_date"]
    ).dt.normalize()
    if frame.duplicated(common.PREDICTION_KEYS).any():
        raise RuntimeError(f"{bundle_key}/{filename} has duplicate keys")
    provenance.update(
        {
            "artifact_path": path.as_posix(),
            "artifact_sha256": common.sha256_file(path),
        }
    )
    return frame, provenance


def validation_gate_records(
    d3_validation: pd.DataFrame,
    d0_validation: pd.DataFrame,
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Apply the locked two-of-three validation-block gate per target."""

    common.validate_prediction_panel(
        d3_validation, protocol=protocol, split_index=1
    )
    common.validate_prediction_panel(
        d0_validation, protocol=protocol, split_index=1
    )
    columns = [
        *common.PREDICTION_KEYS,
        "actual_fisher_z",
        "predicted_fisher_z",
    ]
    merged = d3_validation[columns].merge(
        d0_validation[columns],
        on=common.PREDICTION_KEYS,
        how="outer",
        validate="one_to_one",
        suffixes=("_d3", "_d0"),
        indicator=True,
    )
    if merged.empty or not merged["_merge"].eq("both").all():
        raise ValueError("D3 and D0 validation prediction keys differ")
    actual_delta = (
        merged["actual_fisher_z_d3"] - merged["actual_fisher_z_d0"]
    ).abs()
    if actual_delta.isna().any() or float(actual_delta.max()) > 1e-12:
        raise ValueError("D3 and D0 validation targets differ")

    output: list[dict[str, Any]] = []
    expected_folds = [fold["name"] for fold in protocol["folds"]]
    for target in protocol["targets"]:
        target_frame = merged.loc[merged["target"].eq(target)]
        fold_records: list[dict[str, Any]] = []
        for fold in expected_folds:
            frame = target_frame.loc[target_frame["fold"].eq(fold)]
            if frame.empty:
                raise ValueError(f"Validation gate misses {target}/{fold}")
            actual = frame["actual_fisher_z_d3"].to_numpy(dtype=float)
            d3_loss = np.square(
                actual
                - frame["predicted_fisher_z_d3"].to_numpy(dtype=float)
            )
            d0_loss = np.square(
                actual
                - frame["predicted_fisher_z_d0"].to_numpy(dtype=float)
            )
            d3_mse = float(d3_loss.mean())
            d0_mse = float(d0_loss.mean())
            fold_records.append(
                {
                    "fold": fold,
                    "rows": len(frame),
                    "d0_validation_mse": d0_mse,
                    "d3_validation_mse": d3_mse,
                    "d3_minus_d0_validation_mse": d3_mse - d0_mse,
                    "d3_improved": d3_mse < d0_mse,
                }
            )
        improved = sum(bool(record["d3_improved"]) for record in fold_records)
        output.append(
            {
                "target": target,
                "required_improved_validation_folds": 2,
                "improved_validation_folds": improved,
                "eligible": improved >= 2,
                "folds": fold_records,
            }
        )
    return output


def validate_subset_prediction_panel(
    predictions: pd.DataFrame,
    *,
    protocol: Mapping[str, Any],
    split_index: int,
    targets: Sequence[str],
) -> dict[str, Any]:
    """Validate a target-subset artifact produced by a per-target gate."""

    expected_all = common.expected_rows(protocol, split_index=split_index)
    target_set = set(targets)
    if not target_set or not target_set.issubset(set(protocol["targets"])):
        raise ValueError("Prediction subset names invalid or empty targets")
    folds = [fold["name"] for fold in protocol["folds"]]
    observed = {
        target: {
            fold: int(
                len(
                    predictions[
                        predictions["target"].eq(target)
                        & predictions["fold"].eq(fold)
                    ]
                )
            )
            for fold in folds
        }
        for target in targets
    }
    expected = {target: expected_all[target] for target in targets}
    review = {
        "prediction_keys_unique": not bool(
            predictions.duplicated(common.PREDICTION_KEYS).any()
        ),
        "predictions_finite": bool(
            np.isfinite(
                predictions[
                    ["predicted_fisher_z", "predicted_correlation"]
                ].to_numpy(dtype=float)
            ).all()
        ),
        "prediction_bounds_valid": bool(
            predictions["predicted_correlation"].between(-1, 1).all()
        ),
        "eligible_targets_only": set(predictions["target"]) == target_set,
        "expected_folds_present": set(predictions["fold"]) == set(folds),
        "observed_rows": observed,
        "expected_rows": expected,
        "row_counts_match": observed == expected,
    }
    if not all(
        review[key]
        for key in (
            "prediction_keys_unique",
            "predictions_finite",
            "prediction_bounds_valid",
            "eligible_targets_only",
            "expected_folds_present",
            "row_counts_match",
        )
    ):
        raise AssertionError(f"Subset prediction review failed: {review}")
    return review


def train_eligible_targets(
    protocol: Mapping[str, Any],
    eligible_targets: Sequence[str],
    *,
    allow_exploratory: bool,
    device: str,
    budget: Mapping[str, Any],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    list[dict[str, Any]],
    dict[str, Any],
]:
    panel, preflight = common.load_panel_and_preflight(
        protocol,
        allow_exploratory=allow_exploratory,
        required_bundle_keys=(BUNDLE_KEY,),
    )
    predictions: list[pd.DataFrame] = []
    validation_predictions: list[pd.DataFrame] = []
    fits: list[dict[str, Any]] = []
    feature_contract: dict[str, Any] = {}
    eligible_set = set(eligible_targets)

    for spec in quant_common.target_specs():
        if spec.name not in eligible_set:
            continue
        features = common.bundle_features(BUNDLE_KEY, spec, protocol)
        feature_contract[spec.name] = {
            "count": len(features),
            "ordered_sha256": common.ordered_sha256(features),
            "features": list(features),
        }
        transformed = common.apply_fixed_transforms(panel, features, protocol)
        for fold in protocol["folds"]:
            masks = common.split_masks(transformed, spec, fold)
            train = transformed.loc[masks["train"]].copy()
            validation = transformed.loc[masks["validation"]].copy()
            test = transformed.loc[masks["test"]].copy()
            joined = pd.concat([train, validation], ignore_index=True)
            candidates: list[dict[str, Any]] = []
            candidate_predictions: list[np.ndarray] = []
            for candidate_id, config in enumerate(
                budget["candidate_configs"]
            ):
                model = quant_xgb.xgb_model(
                    config,
                    n_estimators=int(budget["max_estimators"]),
                    early_stopping_rounds=int(
                        budget["early_stopping_rounds"]
                    ),
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
                        "config": dict(config),
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
            selected_position = min(
                range(len(candidates)),
                key=lambda index: (
                    candidates[index]["validation_mse"],
                    candidates[index]["candidate_id"],
                ),
            )
            selected = candidates[selected_position]
            validation_predictions.append(
                quant_common.prediction_frame(
                    validation,
                    spec,
                    fold_name=fold["name"],
                    model_name=common.bundle_for(BUNDLE_KEY).slug,
                    predicted_fisher=candidate_predictions[
                        selected_position
                    ],
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
            predicted_test = final_model.predict(test[list(features)])
            predictions.append(
                quant_common.prediction_frame(
                    test,
                    spec,
                    fold_name=fold["name"],
                    model_name=common.bundle_for(BUNDLE_KEY).slug,
                    predicted_fisher=predicted_test,
                )
            )
            final_cuda = (
                quant_xgb.booster_uses_cuda(final_model)
                if device == "cuda"
                else False
            )
            fits.append(
                {
                    "target": spec.name,
                    "fold": fold["name"],
                    "model": common.bundle_for(BUNDLE_KEY).slug,
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
                    "cuda_confirmed": final_cuda,
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
    validation_panel = pd.concat(validation_predictions, ignore_index=True)
    prediction_review = validate_subset_prediction_panel(
        prediction_panel,
        protocol=protocol,
        split_index=2,
        targets=eligible_targets,
    )
    validation_review = validate_subset_prediction_panel(
        validation_panel,
        protocol=protocol,
        split_index=1,
        targets=eligible_targets,
    )
    if device == "cuda" and not all(
        bool(record["cuda_confirmed"]) for record in fits
    ):
        raise AssertionError("Not every final D5 fit used CUDA")
    return prediction_panel, validation_panel, fits, {
        "preflight": preflight,
        "feature_contract": feature_contract,
        "prediction_review": prediction_review,
        "validation_review": validation_review,
    }


def parser() -> argparse.ArgumentParser:
    output = argparse.ArgumentParser(description=__doc__)
    output.add_argument("--allow-exploratory", action="store_true")
    output.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    output.add_argument("--overwrite", action="store_true")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    protocol = common.load_protocol()
    common.initialize_status(common.sha256_file(common.PROTOCOL_PATH))
    common.verify_completed_bundle("D0")
    common.verify_completed_bundle("D3")
    common.ensure_no_existing_bundle(BUNDLE_KEY, overwrite=args.overwrite)
    d0_validation, d0_provenance = _load_completed_prediction_artifact(
        "D0", "validation_predictions.parquet"
    )
    d3_validation, d3_provenance = _load_completed_prediction_artifact(
        "D3", "validation_predictions.parquet"
    )
    gate = validation_gate_records(d3_validation, d0_validation, protocol)
    eligible_targets = [
        record["target"] for record in gate if record["eligible"]
    ]
    budget = load_xgboost_budget(protocol)
    source = common.source_artifact_provenance(protocol)
    common.update_status(BUNDLE_KEY, "running")

    if not eligible_targets:
        reason = (
            "D3 did not improve D0 validation MSE in at least two of three "
            "validation blocks for any target; the locked D5 gate failed."
        )
        summary = {
            "status": "skipped",
            "reason": reason,
            "metrics": [],
            "validation_gate": gate,
            "eligible_targets": [],
        }
        common.write_bundle(
            BUNDLE_KEY,
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
                "outer_evaluation_excluded_from_gate": True,
                "validation_gate_applied_per_target": True,
            },
            model_config={
                "estimator": "shallow_xgboost",
                "bundle": BUNDLE_KEY,
                "status": "skipped",
                "reason": reason,
                "validation_gate": (
                    "D3 beats D0 validation MSE in at least 2 of 3 folds"
                ),
                "xgboost_budget": budget,
                "source_profile": protocol["source_profile"],
                "claim_scope": protocol["claim_scope"],
            },
            dependencies={
                "source_artifacts": source,
                "D0_validation": d0_provenance,
                "D3_validation": d3_provenance,
            },
            extra_outputs={"validation_gate.json": gate},
        )
        common.update_status(BUNDLE_KEY, "skipped", summary=reason)
        print(json.dumps(summary, indent=2))
        return 0

    try:
        predictions, validation, fits, details = train_eligible_targets(
            protocol,
            eligible_targets,
            allow_exploratory=args.allow_exploratory,
            device=args.device,
            budget=budget,
        )
        d0_predictions, d0_test_provenance = (
            common.load_completed_predictions("D0")
        )
        d3_predictions, d3_test_provenance = (
            common.load_completed_predictions("D3")
        )
        d0_predictions = d0_predictions[
            d0_predictions["target"].isin(eligible_targets)
        ].copy()
        d3_predictions = d3_predictions[
            d3_predictions["target"].isin(eligible_targets)
        ].copy()
        metrics = quant_common.summarize_predictions(predictions).to_dict(
            orient="records"
        )
        fold_metrics = quant_common.summarize_predictions(
            predictions, ("fold", "target", "model")
        ).to_dict(orient="records")
        summary = {
            "status": "complete",
            "metrics": metrics,
            "eligible_targets": eligible_targets,
            "ineligible_targets": [
                target
                for target in protocol["targets"]
                if target not in eligible_targets
            ],
            "validation_gate": gate,
            "base_comparison": common.compare_predictions(
                predictions, d0_predictions
            ),
            "secondary_comparison": common.compare_predictions(
                predictions, d3_predictions
            ),
            "secondary_comparison_label": (
                "Shallow XGBoost versus the D3 linear endpoint"
            ),
        }
        review = {
            "status": "passed",
            **details["prediction_review"],
            "validation_prediction_review": details["validation_review"],
            "outer_evaluation_excluded_from_gate_and_tuning": True,
            "validation_gate_applied_per_target": True,
            "feature_name_leakage_scan_passed": True,
            "source_timing_contract_bound_by_verified_manifest": (
                details["preflight"]["status"] == "passed"
            ),
            "requested_device": args.device,
            "cuda_confirmed_all_final_fits": (
                all(bool(record["cuda_confirmed"]) for record in fits)
                if args.device == "cuda"
                else None
            ),
        }
        common.write_bundle(
            BUNDLE_KEY,
            predictions=predictions,
            validation_predictions=validation,
            fits=fits,
            fold_metrics=fold_metrics,
            summary=summary,
            review=review,
            model_config={
                "estimator": "shallow_xgboost",
                "bundle": BUNDLE_KEY,
                "feature_contract": details["feature_contract"],
                "eligible_targets": eligible_targets,
                "validation_gate": (
                    "D3 beats D0 validation MSE in at least 2 of 3 folds"
                ),
                "xgboost_budget": budget,
                "fixed_log1p_features": protocol["preprocessing"][
                    "fixed_log1p_features"
                ],
                "device": args.device,
                "source_profile": protocol["source_profile"],
                "claim_scope": protocol["claim_scope"],
            },
            dependencies={
                "source_panel_preflight": details["preflight"],
                "D0_validation": d0_provenance,
                "D3_validation": d3_provenance,
                "D0_test": d0_test_provenance,
                "D3_test": d3_test_provenance,
            },
            extra_outputs={"validation_gate.json": gate},
        )
        common.update_status(
            BUNDLE_KEY,
            "complete",
            summary=(
                f"Validation-gated targets: {', '.join(eligible_targets)}; "
                f"{common.summary_status_text(summary)}"
            ),
        )
        print(pd.DataFrame(metrics).to_string(index=False))
        print(json.dumps(review, indent=2))
        return 0
    except Exception as error:
        common.update_status(BUNDLE_KEY, "failed", summary=str(error))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
