#!/usr/bin/env python
"""Freeze v4 sources, features, folds, tuning, and implementation hashes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from scripts.quant_deterministic_news_training_v4 import common, contract


def _v3_base_record(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    manifest = common.load_json(manifest_path)
    artifact = manifest.get("artifacts", {}).get("predictions.parquet")
    if not isinstance(artifact, Mapping):
        raise ValueError(f"{manifest_path} does not bind predictions")
    prediction_path = Path(str(artifact.get("path", "")))
    if (
        not prediction_path.is_file()
        or common.sha256_file(prediction_path) != artifact.get("sha256")
    ):
        raise ValueError(f"{prediction_path} fails its v3 manifest")
    return {
        **common.artifact_record(
            prediction_path,
            rows=int(artifact.get("rows", len(pd.read_parquet(prediction_path)))),
        ),
        "manifest": common.artifact_record(manifest_path),
    }


def _panel_record(path: Path) -> dict[str, Any]:
    frame = pd.read_parquet(path)
    frame["forecast_date"] = pd.to_datetime(frame["forecast_date"]).dt.normalize()
    if frame.duplicated(list(contract.PANEL_KEYS)).any():
        raise ValueError(f"{path} has duplicate keys")
    expected = {
        *contract.PANEL_KEYS,
        *contract.SOFT_ROUTE_19,
        *contract.COUPLING_6,
        *contract.QUALITY_4,
    }
    missing = expected - set(frame.columns)
    if missing:
        raise ValueError(f"{path} misses v4 feature columns: {sorted(missing)}")
    return {
        **common.artifact_record(path, rows=len(frame)),
        "dates": int(frame["forecast_date"].nunique()),
        "stocks": int(frame["stock"].nunique()),
        "sectors": int(frame["sector"].nunique()),
        "date_min": frame["forecast_date"].min().date().isoformat(),
        "date_max": frame["forecast_date"].max().date().isoformat(),
        "manifest": common.artifact_record(contract.PANEL_MANIFEST_PATH),
    }


def build_protocol() -> dict[str, Any]:
    v3_sidecar = contract.V3_PROTOCOL_SIDECAR.read_text(encoding="ascii").split()[0]
    if v3_sidecar != common.sha256_file(contract.V3_PROTOCOL_PATH):
        raise ValueError("The v3 protocol sidecar is stale")
    v3 = common.load_json(contract.V3_PROTOCOL_PATH)
    if tuple(v3.get("folds", ())) != contract.SHORT_FOLDS:
        raise ValueError("The v4 short folds do not exactly match v3")
    blocks = v3.get("feature_blocks")
    if not isinstance(blocks, Mapping):
        raise ValueError("The v3 feature blocks are missing")
    q = blocks.get("q56_by_target")
    d = blocks.get("d2_normalized_30")
    if not isinstance(q, Mapping) or not isinstance(d, list):
        raise ValueError("The v3 Q56/D2 contracts are malformed")

    quant_predictions = pd.read_parquet(contract.QUANT_V2_PREDICTIONS_PATH)
    quant_anchor = quant_predictions[
        quant_predictions["model"].eq(contract.LONG_ANCHOR_MODEL)
        & quant_predictions["target"].isin(contract.TARGETS)
        & quant_predictions["fold"].isin(
            tuple(f"fold_{index:02d}" for index in range(6, 14))
        )
    ]
    if quant_anchor.empty or quant_anchor.duplicated(list(contract.PREDICTION_KEYS)).any():
        raise ValueError("The long-history XGBoost OOS anchor is invalid")

    return {
        "experiment_id": "quant-deterministic-news-v4-soft-route",
        "protocol_version": "4.0",
        "status": "locked_before_training",
        "locked_at_utc": common.utc_now(),
        "targets": list(contract.TARGETS),
        "join_keys": list(contract.PANEL_KEYS),
        "short_folds": list(contract.SHORT_FOLDS),
        "long_test_folds": list(contract.LONG_TEST_FOLDS),
        "long_prequential_rule": {
            "anchor_model": contract.LONG_ANCHOR_MODEL,
            "first_training_fold": "fold_06",
            "for_test_fold_j": "train fold_06 through j-2; validate j-1; refit train+validation",
            "base_predictions_must_be_out_of_sample": True,
            "t2_rows_are_the_already_purged_quant_v2_rows": True,
        },
        "feature_blocks": {
            "q56_by_target": {target: list(q[target]) for target in contract.TARGETS},
            "d2_normalized_30": list(d),
            "current_9": list(contract.CURRENT_9),
            "soft_route_19": list(contract.SOFT_ROUTE_19),
            "coupling_6": list(contract.COUPLING_6),
            "quality_4": list(contract.QUALITY_4),
            "quality_only_5": list(contract.QUALITY_ONLY_5),
        },
        "preprocessing": {
            "fixed_log1p_features": list(
                v3.get("preprocessing", {}).get("fixed_log1p_features", ())
            ),
            "median_imputation_fit_on_training_only": True,
            "standardization_fit_on_training_only": True,
            "final_preprocessing_refit_on_train_plus_validation": True,
        },
        "model_tuning": {
            "alpha_grid": list(contract.ALPHA_GRID),
            "l1_ratio_grid": list(contract.L1_RATIO_GRID),
            "residual_correction_shrinkage_grid": list(
                contract.CORRECTION_SHRINKAGE_GRID
            ),
            "all_hyperparameters_selected_on_validation_only": True,
        },
        "bootstrap": {
            "method": "paired_moving_block_whole_date_within_outer_fold",
            "block_sessions": 10,
            "resamples_default": 2000,
            "seed": 1729,
        },
        "source_artifacts": {
            "canonical_panel": _panel_record(contract.PANEL_PATH),
            "stale20_panel": _panel_record(contract.STALE20_PANEL_PATH),
            "wrong_stock_panel": _panel_record(contract.WRONG_STOCK_PANEL_PATH),
            "permuted_panel": _panel_record(contract.PERMUTED_PANEL_PATH),
            "v4_panel_manifest": common.artifact_record(contract.PANEL_MANIFEST_PATH),
            "v3_protocol": common.artifact_record(contract.V3_PROTOCOL_PATH),
            "v3_s0_predictions": _v3_base_record(contract.V3_S0_ROOT),
            "v3_s1_predictions": _v3_base_record(contract.V3_S1_ROOT),
            "quant_v2_protocol": common.artifact_record(contract.QUANT_V2_PROTOCOL_PATH),
            "quant_v2_rung03_predictions": {
                **common.artifact_record(
                    contract.QUANT_V2_PREDICTIONS_PATH,
                    rows=len(quant_predictions),
                ),
                "anchor_model": contract.LONG_ANCHOR_MODEL,
                "anchor_rows_fold06_to13": len(quant_anchor),
            },
        },
        "bundles": [
            {
                "key": bundle.key,
                "slug": bundle.slug,
                "label": bundle.label,
                "family": bundle.family,
                "features": bundle.features,
                "control": bundle.control,
            }
            for bundle in contract.BUNDLES
        ],
        "claim_flags": {
            "exploratory": True,
            "retrospective_non_version_safe": True,
            "primary_training_eligible": False,
            "confirmatory_eligible": False,
            "v3_artifacts_immutable": True,
        },
        "implementation": {
            path.as_posix(): common.sha256_file(path)
            for path in contract.BOUND_IMPLEMENTATION
        },
    }


def parser() -> argparse.ArgumentParser:
    output = argparse.ArgumentParser(description=__doc__)
    output.add_argument("--overwrite", action="store_true")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if contract.OUTPUT_ROOT.exists() and any(contract.OUTPUT_ROOT.iterdir()):
        raise RuntimeError("Refusing to re-lock after v4 outputs exist")
    if (contract.PROTOCOL_PATH.exists() or contract.PROTOCOL_SIDECAR.exists()) and not args.overwrite:
        raise FileExistsError("The v4 protocol already exists; pass --overwrite before training")
    protocol = build_protocol()
    common.atomic_json(contract.PROTOCOL_PATH, protocol)
    common.atomic_text(
        contract.PROTOCOL_SIDECAR,
        f"{common.sha256_file(contract.PROTOCOL_PATH)}  {contract.PROTOCOL_PATH.name}\n",
    )
    common.load_protocol()
    print(f"Locked {contract.PROTOCOL_PATH} at {common.sha256_file(contract.PROTOCOL_PATH)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
