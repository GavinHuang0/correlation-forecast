"""Freeze and audit every input before Q+D model fitting begins."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from scripts.correlation_training import training_common as quant_common
from scripts.quant_deterministic_news_training import common


def run_preflight(*, allow_exploratory: bool) -> dict[str, object]:
    protocol = common.load_protocol()
    panel, source_audit = common.load_panel_and_preflight(
        protocol, allow_exploratory=allow_exploratory
    )
    split_counts: dict[str, dict[str, dict[str, int]]] = {}
    expected = protocol["target_row_eligibility"]["expected_rows"]
    for spec in quant_common.target_specs():
        split_counts[spec.name] = {}
        horizon = spec.horizon
        for fold in protocol["folds"]:
            masks = common.split_masks(panel, spec, fold)
            observed = {
                block: int(mask.sum()) for block, mask in masks.items()
            }
            expected_values = expected[fold["name"]][
                f"{horizon}_train_validation_test"
            ]
            expected_record = dict(
                zip(
                    ("train", "validation", "test"),
                    map(int, expected_values),
                    strict=True,
                )
            )
            if observed != expected_record:
                raise ValueError(
                    f"Split count mismatch for {spec.name}/{fold['name']}: "
                    f"{observed} != {expected_record}"
                )
            split_counts[spec.name][fold["name"]] = observed

    frozen = {}
    for family in ("elastic_net", "xgboost", "winner_t1_etf"):
        predictions = common.load_frozen_predictions(protocol, family)
        common.validate_frozen_actuals(predictions, panel)
        frozen[family] = {
            "rows": len(predictions),
            "targets": sorted(predictions["target"].unique().tolist()),
            "folds": sorted(predictions["fold"].unique().tolist()),
            "base_model": sorted(predictions["base_model"].unique().tolist()),
            "artifact_sha256": sorted(
                predictions["base_artifact_sha256"].unique().tolist()
            ),
        }

    feature_contract = {}
    for spec in quant_common.target_specs():
        q = common.quant_features(spec, protocol)
        feature_contract[spec.name] = {
            "q_count": len(q),
            "q_ordered_sha256": common.ordered_sha256(q),
            "a1_count": len(common.joint_features("A1", spec, protocol)),
            "a2_count": len(common.joint_features("A2", spec, protocol)),
            "a3_count": len(common.joint_features("A3", spec, protocol)),
            "a4_count": len(common.joint_features("A4", spec, protocol)),
            "a5_count": len(common.joint_features("A5", spec, protocol)),
        }
    d43 = common.deterministic_features(protocol)
    audit = {
        **source_audit,
        "split_counts": split_counts,
        "frozen_prediction_contract": frozen,
        "feature_contract": feature_contract,
        "d43_ordered_sha256": common.ordered_sha256(d43),
        "d43": list(d43),
        "resolved_xgboost_candidates": common.xgboost_candidate_configs(
            protocol
        ),
    }
    return audit


def parser() -> argparse.ArgumentParser:
    output = argparse.ArgumentParser(description=__doc__)
    output.add_argument("--allow-exploratory", action="store_true")
    output.add_argument("--overwrite", action="store_true")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    protocol = common.load_protocol()
    common.initialize_status(common.sha256_file(common.PROTOCOL_PATH))
    common.ensure_no_existing_bundle("PRE", overwrite=args.overwrite)
    common.update_status("PRE", "running")
    try:
        audit = run_preflight(allow_exploratory=args.allow_exploratory)
        review = {
            "status": "passed",
            "prediction_keys_unique": True,
            "predictions_finite": True,
            "prediction_bounds_valid": True,
            "source_contract_passed": True,
            "split_contract_passed": True,
            "feature_contract_passed": True,
            "frozen_prediction_contract_passed": True,
        }
        summary = {
            "status": "complete",
            "rows": audit["rows"],
            "dates": audit["dates"],
            "stocks": audit["stocks"],
            "sectors": audit["sectors"],
            "metrics": [],
        }
        common.write_bundle(
            "PRE",
            predictions=None,
            validation_predictions=None,
            fits=[],
            fold_metrics=[],
            summary=summary,
            review=review,
            model_config={
                "role": "input and split preflight",
                "allow_exploratory": args.allow_exploratory,
            },
            extra_outputs={
                "preflight.json": audit,
                "protocol_lock.json": {
                    "path": common.PROTOCOL_PATH.as_posix(),
                    "sha256": common.sha256_file(common.PROTOCOL_PATH),
                    "protocol": protocol,
                },
            },
        )
        tracked_audit = (
            common.EXPERIMENT_ROOT / "construction" / "preflight.json"
        )
        quant_common.write_json(tracked_audit, audit)
        common.update_status(
            "PRE",
            "complete",
            summary=(
                f"{audit['rows']:,} rows; {audit['dates']} dates; "
                "all hashes/splits passed"
            ),
        )
        print(json.dumps(audit, indent=2))
        return 0
    except Exception as error:
        common.update_status("PRE", "failed", summary=str(error))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
