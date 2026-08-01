#!/usr/bin/env python
"""Run validation-gated v3 W17-Lite shallow XGBoost."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import xgboost as xgb

from scripts.correlation_training import training_common as quant_common
from scripts.quant_deterministic_news_training_v3 import common, contract


BUNDLE_KEY = "S4"


def load_budget(protocol: Mapping[str, Any]) -> dict[str, Any]:
    tuning = protocol.get("model_tuning")
    locked = (
        tuning.get("shallow_xgboost")
        if isinstance(tuning, Mapping)
        else None
    )
    if not isinstance(locked, Mapping):
        raise ValueError("v3 did not lock shallow XGBoost tuning")
    source_protocol = locked.get("source_protocol")
    source_implementation = locked.get("source_implementation")
    for record, expected_path, label in (
        (
            source_protocol,
            contract.QUANT_V1_PROTOCOL,
            "quant-v1 protocol",
        ),
        (
            source_implementation,
            contract.QUANT_V1_XGBOOST_IMPLEMENTATION,
            "quant-v1 XGBoost implementation",
        ),
    ):
        if not isinstance(record, Mapping):
            raise ValueError(f"v3 lacks locked {label}")
        path = Path(str(record.get("path", "")))
        if (
            path != expected_path
            or not path.is_file()
            or common.sha256_file(path) != record.get("sha256")
        ):
            raise ValueError(f"Locked {label} hash changed")
    raw_candidates = locked.get("candidate_configs")
    if not isinstance(raw_candidates, list) or len(raw_candidates) != 4:
        raise ValueError("v3 must lock exactly four XGBoost candidates")
    candidates: list[dict[str, int | float]] = []
    for index, raw in enumerate(raw_candidates):
        if not isinstance(raw, Mapping) or not raw:
            raise ValueError(f"Locked XGBoost candidate {index} is malformed")
        candidate: dict[str, int | float] = {}
        for key, value in raw.items():
            if (
                not isinstance(key, str)
                or not key
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(
                    f"Locked XGBoost candidate {index}/{key!r} is invalid"
                )
            candidate[key] = value
        candidates.append(candidate)
    if common.sha256_json(candidates) != locked.get(
        "candidate_configs_ordered_sha256"
    ):
        raise ValueError("Locked XGBoost candidate order/hash changed")
    n_estimators = int(locked.get("n_estimators", 0))
    early_stopping = int(locked.get("early_stopping_rounds", 0))
    random_seed = locked.get("random_seed")
    if (
        n_estimators <= 0
        or early_stopping <= 0
        or isinstance(random_seed, bool)
        or not isinstance(random_seed, int)
    ):
        raise ValueError("Locked XGBoost budget/seed is invalid")
    return {
        "source_protocol": dict(source_protocol),
        "source_implementation": dict(source_implementation),
        "candidate_configs": candidates,
        "candidate_configs_ordered_sha256": common.sha256_json(candidates),
        "max_estimators": n_estimators,
        "early_stopping_rounds": early_stopping,
        "random_seed": random_seed,
    }


def xgb_model(
    config: Mapping[str, int | float],
    *,
    n_estimators: int,
    early_stopping_rounds: int | None,
    random_seed: int,
    device: str,
) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        device=device,
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
        random_state=random_seed,
        n_jobs=4,
        verbosity=0,
        **dict(config),
    )


def booster_uses_cuda(model: xgb.XGBRegressor) -> bool:
    return '"device":"cuda:0"' in model.get_booster().save_config()


def validation_gate(
    s3: pd.DataFrame,
    s1: pd.DataFrame,
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    common.validate_prediction_panel(
        s3, protocol=protocol, split_index=1
    )
    common.validate_prediction_panel(
        s1, protocol=protocol, split_index=1
    )
    keys = list(common.contract.PREDICTION_KEYS)
    columns = [*keys, "actual_fisher_z", "predicted_fisher_z"]
    merged = s3[columns].merge(
        s1[columns],
        on=keys,
        how="outer",
        validate="one_to_one",
        suffixes=("_s3", "_s1"),
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("S3 and S1 validation keys differ")
    if (
        merged["actual_fisher_z_s3"]
        .sub(merged["actual_fisher_z_s1"])
        .abs()
        .max()
        > 1e-12
    ):
        raise ValueError("S3 and S1 validation targets differ")
    records: list[dict[str, Any]] = []
    for target in protocol["targets"]:
        folds: list[dict[str, Any]] = []
        for fold in protocol["folds"]:
            part = merged[
                merged["target"].eq(target)
                & merged["fold"].eq(fold["name"])
            ]
            actual = part["actual_fisher_z_s3"].to_numpy(dtype=float)
            s3_mse = float(
                np.square(
                    actual
                    - part["predicted_fisher_z_s3"].to_numpy(dtype=float)
                ).mean()
            )
            s1_mse = float(
                np.square(
                    actual
                    - part["predicted_fisher_z_s1"].to_numpy(dtype=float)
                ).mean()
            )
            folds.append(
                {
                    "fold": fold["name"],
                    "rows": len(part),
                    "s1_validation_mse": s1_mse,
                    "s3_validation_mse": s3_mse,
                    "s3_minus_s1_validation_mse": s3_mse - s1_mse,
                    "s3_improved": s3_mse < s1_mse,
                }
            )
        improved = sum(item["s3_improved"] for item in folds)
        records.append(
            {
                "target": target,
                "required_improved_validation_folds": 2,
                "improved_validation_folds": improved,
                "eligible": improved >= 2,
                "folds": folds,
            }
        )
    return records


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "sha256": common.sha256_file(path),
        "bytes": path.stat().st_size,
    }


def write_skipped_bundle(
    protocol: Mapping[str, Any],
    *,
    gate: Sequence[Mapping[str, Any]],
    s1_validation_provenance: Mapping[str, Any],
    s3_validation_provenance: Mapping[str, Any],
    budget: Mapping[str, Any],
) -> dict[str, Any]:
    """Write an executed, hash-bound S4 validation-gate skip."""

    if any(bool(record.get("eligible")) for record in gate):
        raise ValueError("Cannot write an S4 skip with eligible targets")
    if {str(record.get("target")) for record in gate} != set(
        contract.TARGETS
    ):
        raise ValueError("S4 skip gate does not cover all targets")
    bundle = common.bundle_for(BUNDLE_KEY)
    output = bundle.output_path
    tracked = bundle.experiment_path
    output.mkdir(parents=True, exist_ok=False)
    tracked.mkdir(parents=True, exist_ok=False)
    reason = (
        "S3 failed the two-of-three validation-fold gate for every target"
    )
    summary = {
        "status": "skipped_validation_gate",
        "reason": reason,
        "eligible_targets": [],
        "metrics": [],
        "semantic_gate_passed": False,
        "result_role": "executed_validation_gate_skip",
    }
    review = {
        "status": "passed",
        "s4_execution_attempted": True,
        "validation_gate_executed": True,
        "all_targets_ineligible": True,
        "outer_evaluation_excluded_from_gate": True,
        "validation_gate_applied_per_target": True,
        "semantic_gate_passed": False,
        "failed_semantic_gate_override_confirmed": True,
    }
    model_config = {
        "estimator": "shallow_xgboost",
        "bundle": BUNDLE_KEY,
        "fit_executed": False,
        "skip_reason": reason,
        "locked_budget": dict(budget),
    }
    dependencies = {
        "S1_validation": dict(s1_validation_provenance),
        "S3_validation": dict(s3_validation_provenance),
    }
    files = {
        "validation_gate.json": output / "validation_gate.json",
        "summary.json": output / "summary.json",
        "review.json": output / "review.json",
        "model_config.json": output / "model_config.json",
        "dependencies.json": output / "dependencies.json",
    }
    common.atomic_json(files["validation_gate.json"], list(gate))
    common.atomic_json(files["summary.json"], summary)
    common.atomic_json(files["review.json"], review)
    common.atomic_json(files["model_config.json"], model_config)
    common.atomic_json(files["dependencies.json"], dependencies)
    manifest_path = output / "manifest.json"
    manifest = {
        "manifest_version": (
            "quant-deterministic-news-v3-s4-validation-skip-v1"
        ),
        "status": "skipped_validation_gate",
        "bundle": BUNDLE_KEY,
        "generated_at_utc": common.utc_now(),
        "protocol_path": contract.PROTOCOL_PATH.as_posix(),
        "protocol_sha256": common.sha256_file(contract.PROTOCOL_PATH),
        "reason": reason,
        "semantic_gate_passed": False,
        "claim_flags": {
            "exploratory": True,
            "primary_training_eligible": False,
            "confirmatory_eligible": False,
        },
        "artifacts": {
            name: _file_record(path) for name, path in files.items()
        },
        "dependencies": dependencies,
        "implementation_sha256": dict(protocol["implementation"]),
        "runtime": common.runtime_versions(),
    }
    common.atomic_json(manifest_path, manifest)
    manifest_hash = common.sha256_file(manifest_path)
    sidecar = manifest_path.with_suffix(".sha256")
    common.atomic_text(
        sidecar, f"{manifest_hash}  {manifest_path.name}\n"
    )
    common.atomic_json(tracked / "summary.json", summary)
    common.atomic_json(tracked / "review.json", review)
    common.atomic_json(
        tracked / "artifact_pointer.json",
        {
            "bundle": BUNDLE_KEY,
            "status": "skipped_validation_gate",
            "manifest_path": manifest_path.as_posix(),
            "manifest_sha256": manifest_hash,
            "protocol_sha256": common.sha256_file(
                contract.PROTOCOL_PATH
            ),
        },
    )
    common.atomic_text(
        tracked / "RESULTS.md",
        "\n".join(
            [
                "# V3-S4 validation-gated shallow XGBoost",
                "",
                "Status: **skipped after executing the locked gate**.",
                "",
                reason + ".",
                "",
                "No XGBoost fit or outer-test prediction was produced.",
                "The validation gate and its S1/S3 provenance are hash-bound.",
                "",
            ]
        ),
    )
    return {
        "status": "skipped_validation_gate",
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": manifest_hash,
        "validation_gate": list(gate),
    }


def verify_s4_outcome(
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify either a completed S4 fit or an executed gate-skip artifact."""

    locked = common.load_protocol() if protocol is None else protocol
    bundle = common.bundle_for(BUNDLE_KEY)
    manifest_path = bundle.output_path / "manifest.json"
    sidecar = manifest_path.with_suffix(".sha256")
    if not manifest_path.is_file() or not sidecar.is_file():
        raise FileNotFoundError(
            "S4 has neither a completed fit nor an executed skip artifact"
        )
    manifest_hash = common.sha256_file(manifest_path)
    if sidecar.read_text(encoding="ascii").strip().split()[0] != manifest_hash:
        raise RuntimeError("S4 manifest sidecar is stale")
    manifest = common.load_json(manifest_path)
    if manifest.get("implementation_sha256") != locked.get(
        "implementation"
    ):
        raise RuntimeError("S4 implementation provenance changed")
    if manifest.get("status") == "complete":
        verified = common.verify_completed_bundle(BUNDLE_KEY)
        summary_record = manifest.get("artifacts", {}).get("summary.json")
        if not isinstance(summary_record, Mapping):
            raise RuntimeError("Completed S4 lacks summary provenance")
        summary = common.load_json(Path(str(summary_record["path"])))
        return {
            "status": "complete",
            "eligible_targets": list(summary.get("eligible_targets", [])),
            "manifest_path": verified["manifest_path"],
            "manifest_sha256": verified["manifest_sha256"],
        }
    if (
        manifest.get("manifest_version")
        != "quant-deterministic-news-v3-s4-validation-skip-v1"
        or manifest.get("status") != "skipped_validation_gate"
        or manifest.get("bundle") != BUNDLE_KEY
        or manifest.get("protocol_sha256")
        != common.sha256_file(contract.PROTOCOL_PATH)
    ):
        raise RuntimeError("S4 skip manifest is stale or malformed")
    required_files = {
        "validation_gate.json",
        "summary.json",
        "review.json",
        "model_config.json",
        "dependencies.json",
    }
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != required_files:
        raise RuntimeError("S4 skip artifact set changed")
    for name, record in artifacts.items():
        if not isinstance(record, Mapping):
            raise RuntimeError(f"S4 skip {name} record is malformed")
        path = Path(str(record.get("path", "")))
        if (
            not path.is_file()
            or common.sha256_file(path) != record.get("sha256")
        ):
            raise RuntimeError(f"S4 skip {name} fails its hash")
    # atomic_json stores the gate as an array, so read it directly here.
    gate_value = json.loads(
        Path(str(artifacts["validation_gate.json"]["path"])).read_text(
            encoding="utf-8"
        )
    )
    if not isinstance(gate_value, list):
        raise RuntimeError("S4 validation gate artifact is not an array")
    if (
        {str(record.get("target")) for record in gate_value}
        != set(contract.TARGETS)
        or any(bool(record.get("eligible")) for record in gate_value)
    ):
        raise RuntimeError("S4 skip gate does not prove all-target rejection")
    dependencies = common.load_json(
        Path(str(artifacts["dependencies.json"]["path"]))
    )
    if manifest.get("dependencies") != dependencies:
        raise RuntimeError("S4 skip manifest dependency ledger changed")
    summary = common.load_json(
        Path(str(artifacts["summary.json"]["path"]))
    )
    review = common.load_json(
        Path(str(artifacts["review.json"]["path"]))
    )
    if (
        summary.get("status") != "skipped_validation_gate"
        or summary.get("eligible_targets") != []
        or summary.get("metrics") != []
        or review.get("status") != "passed"
        or review.get("validation_gate_executed") is not True
        or review.get("all_targets_ineligible") is not True
    ):
        raise RuntimeError("S4 skip summary/review contract changed")
    s1, s1_provenance = common.load_completed_predictions(
        "S1", validation=True
    )
    s3, s3_provenance = common.load_completed_predictions(
        "S3", validation=True
    )
    if dependencies != {
        "S1_validation": s1_provenance,
        "S3_validation": s3_provenance,
    }:
        raise RuntimeError("S4 skip validation dependencies changed")
    recomputed = validation_gate(s3, s1, locked)
    if common.canonical_json(recomputed) != common.canonical_json(gate_value):
        raise RuntimeError("S4 skip validation gate no longer recomputes")
    model_config = common.load_json(
        Path(str(artifacts["model_config.json"]["path"]))
    )
    if model_config.get("locked_budget") != load_budget(locked):
        raise RuntimeError("S4 skip budget provenance changed")
    tracked_pointer = bundle.experiment_path / "artifact_pointer.json"
    if not tracked_pointer.is_file():
        raise RuntimeError("S4 skip tracked pointer is missing")
    pointer = common.load_json(tracked_pointer)
    if (
        pointer.get("status") != "skipped_validation_gate"
        or pointer.get("manifest_sha256") != manifest_hash
        or pointer.get("protocol_sha256")
        != common.sha256_file(contract.PROTOCOL_PATH)
    ):
        raise RuntimeError("S4 skip tracked pointer is stale")
    return {
        "status": "skipped_validation_gate",
        "eligible_targets": [],
        "reason": str(summary["reason"]),
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": manifest_hash,
        "validation_gate": gate_value,
    }


def train(
    protocol: Mapping[str, Any],
    targets: Sequence[str],
    *,
    allow_failed_gate_exploratory: bool,
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
        bundle_key=BUNDLE_KEY,
        allow_failed_gate_exploratory=allow_failed_gate_exploratory,
    )
    target_set = set(targets)
    predictions: list[pd.DataFrame] = []
    validation_predictions: list[pd.DataFrame] = []
    fits: list[dict[str, Any]] = []
    feature_contract: dict[str, Any] = {}
    for spec in quant_common.target_specs():
        if spec.name not in target_set:
            continue
        features = common.model_features(BUNDLE_KEY, spec.name, protocol)
        feature_contract[spec.name] = {
            "count": len(features),
            "features": list(features),
            "ordered_sha256": common.ordered_sha256(features),
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
            candidates: list[dict[str, Any]] = []
            validation_outputs: list[np.ndarray] = []
            for candidate_id, config in enumerate(
                budget["candidate_configs"]
            ):
                model = xgb_model(
                    config,
                    n_estimators=int(budget["max_estimators"]),
                    early_stopping_rounds=int(
                        budget["early_stopping_rounds"]
                    ),
                    random_seed=int(budget["random_seed"]),
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
                record = {
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
                        booster_uses_cuda(model)
                        if device == "cuda"
                        else False
                    ),
                }
                candidates.append(record)
                validation_outputs.append(predicted)
            position = min(
                range(len(candidates)),
                key=lambda index: (
                    candidates[index]["validation_mse"],
                    candidates[index]["candidate_id"],
                ),
            )
            selected = candidates[position]
            validation_predictions.append(
                quant_common.prediction_frame(
                    validation,
                    spec,
                    fold_name=fold["name"],
                    model_name=common.bundle_for(BUNDLE_KEY).slug,
                    predicted_fisher=validation_outputs[position],
                )
            )
            final = xgb_model(
                selected["config"],
                n_estimators=int(selected["best_iteration"]) + 1,
                early_stopping_rounds=None,
                random_seed=int(budget["random_seed"]),
                device=device,
            )
            final.fit(
                joined[list(features)],
                joined[spec.response_column],
                verbose=False,
            )
            predictions.append(
                quant_common.prediction_frame(
                    test,
                    spec,
                    fold_name=fold["name"],
                    model_name=common.bundle_for(BUNDLE_KEY).slug,
                    predicted_fisher=final.predict(test[list(features)]),
                )
            )
            final_cuda = (
                booster_uses_cuda(final)
                if device == "cuda"
                else False
            )
            fits.append(
                {
                    "target": spec.name,
                    "fold": fold["name"],
                    "features": list(features),
                    "feature_count": len(features),
                    "feature_ordered_sha256": common.ordered_sha256(
                        features
                    ),
                    "train_rows": len(train),
                    "validation_rows": len(validation),
                    "test_rows": len(test),
                    "candidate_records": candidates,
                    "selected_candidate": selected,
                    "final_n_estimators": int(
                        selected["best_iteration"]
                    )
                    + 1,
                    "cuda_confirmed": final_cuda,
                    "feature_importances": [
                        {
                            "feature": feature,
                            "importance": float(value),
                        }
                        for feature, value in zip(
                            features,
                            final.feature_importances_,
                            strict=True,
                        )
                    ],
                }
            )
    prediction_panel = pd.concat(predictions, ignore_index=True)
    validation_panel = pd.concat(
        validation_predictions, ignore_index=True
    )
    return prediction_panel, validation_panel, fits, {
        "preflight": preflight,
        "feature_contract": feature_contract,
        "prediction_review": common.validate_prediction_panel(
            prediction_panel,
            protocol=protocol,
            split_index=2,
            targets=targets,
        ),
        "validation_review": common.validate_prediction_panel(
            validation_panel,
            protocol=protocol,
            split_index=1,
            targets=targets,
        ),
    }


def parser() -> argparse.ArgumentParser:
    output = argparse.ArgumentParser(description=__doc__)
    output.add_argument(
        "--allow-failed-gate-exploratory", action="store_true"
    )
    output.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    output.add_argument("--overwrite", action="store_true")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    protocol = common.load_protocol()
    common.initialize_status()
    common.verify_completed_bundle("S1")
    common.verify_completed_bundle("S3")
    common.ensure_no_existing_bundle(
        BUNDLE_KEY, overwrite=args.overwrite
    )
    s1_validation, s1_provenance = common.load_completed_predictions(
        "S1", validation=True
    )
    s3_validation, s3_provenance = common.load_completed_predictions(
        "S3", validation=True
    )
    gate = validation_gate(s3_validation, s1_validation, protocol)
    targets = [record["target"] for record in gate if record["eligible"]]
    budget = load_budget(protocol)
    if not targets:
        try:
            artifact = write_skipped_bundle(
                protocol,
                gate=gate,
                s1_validation_provenance=s1_provenance,
                s3_validation_provenance=s3_provenance,
                budget=budget,
            )
            verified = verify_s4_outcome(protocol)
            common.update_status(
                BUNDLE_KEY,
                "skipped",
                (
                    "Executed S3-vs-S1 validation gate; no target passed "
                    "two of three folds"
                ),
            )
            print(
                json.dumps(
                    {
                        **artifact,
                        "verified_manifest_sha256": verified[
                            "manifest_sha256"
                        ],
                    },
                    indent=2,
                )
            )
            return 0
        except Exception as error:
            common.update_status(BUNDLE_KEY, "failed", str(error))
            raise
    common.update_status(BUNDLE_KEY, "running")
    try:
        predictions, validation, fits, details = train(
            protocol,
            targets,
            allow_failed_gate_exploratory=(
                args.allow_failed_gate_exploratory
            ),
            device=args.device,
            budget=budget,
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
            "eligible_targets": targets,
            "validation_gate": gate,
            "semantic_gate_passed": False,
            "result_role": "exploratory_failed_semantic_gate_diagnostic",
        }
        review = {
            "status": "passed",
            **details["prediction_review"],
            "validation_prediction_review": details["validation_review"],
            "outer_evaluation_excluded_from_gate": True,
            "validation_gate_applied_per_target": True,
            "cuda_confirmed_all_final_fits": (
                args.device != "cuda"
                or all(record["cuda_confirmed"] for record in fits)
            ),
            "semantic_gate_passed": False,
            "failed_semantic_gate_override_confirmed": True,
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
                "validation_gate": gate,
                "budget": budget,
                "device": args.device,
                "semantic_gate_passed": False,
            },
            dependencies={
                "source_panel_preflight": details["preflight"],
                "S1_validation": s1_provenance,
                "S3_validation": s3_provenance,
            },
            extra_outputs={
                "source_preflight.json": details["preflight"],
                "validation_gate.json": gate,
            },
        )
        common.update_status(
            BUNDLE_KEY,
            "complete",
            common.summary_status_text(summary),
        )
        print(pd.DataFrame(metrics).to_string(index=False))
        return 0
    except Exception as error:
        common.update_status(BUNDLE_KEY, "failed", str(error))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
