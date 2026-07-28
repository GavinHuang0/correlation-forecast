"""Train the two locked V2-D3 deterministic-news falsification controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from scripts.correlation_training import training_common as quant_common
from scripts.quant_deterministic_news_training_v2 import common


CONTROL_BY_BUNDLE = {
    "C-D3-L20": "lag20",
    "C-D3-WS": "wrong_stock",
}


def resolve_full_d2_source(
    protocol: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Resolve and verify the full-history D2 artifact behind the joined panel."""

    sources = protocol["source_artifacts"]
    joined_manifest_path = Path(str(sources["panel_manifest_path"]))
    if (
        not joined_manifest_path.exists()
        or common.sha256_file(joined_manifest_path)
        != str(sources["panel_manifest_sha256"])
    ):
        raise RuntimeError("Joined-panel manifest fails its hash contract")
    joined_manifest = common.load_json(joined_manifest_path)
    d2_input = joined_manifest.get("inputs", {}).get("d2")
    if not isinstance(d2_input, Mapping):
        raise RuntimeError("Joined-panel manifest does not bind its D2 input")
    d2_path = Path(str(d2_input.get("path", "")))
    d2_manifest_path = Path(str(d2_input.get("manifest_path", "")))
    expected_path_hash = str(d2_input.get("sha256", ""))
    expected_manifest_hash = str(d2_input.get("manifest_sha256", ""))
    for path, expected, label in (
        (d2_path, expected_path_hash, "full-history D2 panel"),
        (d2_manifest_path, expected_manifest_hash, "D2 source manifest"),
    ):
        if not path.exists() or common.sha256_file(path) != expected:
            raise RuntimeError(f"{label} fails its joined-manifest hash")
    d2_manifest = common.load_json(d2_manifest_path)
    generated = d2_manifest.get("generated_files", {}).get(
        "stock_day_features.parquet"
    )
    if (
        not isinstance(generated, Mapping)
        or Path(str(generated.get("path", ""))).resolve() != d2_path.resolve()
        or str(generated.get("sha256", "")) != expected_path_hash
    ):
        raise RuntimeError("D2 source manifest and joined manifest disagree")
    return d2_path, {
        "path": d2_path.as_posix(),
        "sha256": expected_path_hash,
        "manifest_path": d2_manifest_path.as_posix(),
        "manifest_sha256": expected_manifest_hash,
    }


def prepare_control_panel(
    panel: pd.DataFrame,
    protocol: Mapping[str, Any],
    control: str,
    *,
    full_d2_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Replace only D2-30 with a predeclared stale or wrong-stock block."""

    if control not in {"lag20", "wrong_stock"}:
        raise ValueError(f"Unknown D3 control {control}")
    d2 = list(common.deterministic_features("d2_normalized_30", protocol))
    original = panel.copy()
    original["forecast_date"] = pd.to_datetime(
        original["forecast_date"]
    ).dt.normalize()
    if original.duplicated(common.PANEL_KEYS).any():
        raise ValueError("Base panel has duplicate control keys")

    if control == "lag20":
        if full_d2_path is None:
            raise ValueError("lag20 requires the verified full-history D2 panel")
        required = ["forecast_date", "sector", "stock", "source_profile", *d2]
        source = pd.read_parquet(full_d2_path, columns=required)
        source["forecast_date"] = pd.to_datetime(
            source["forecast_date"]
        ).dt.normalize()
        if source.duplicated(["forecast_date", "sector", "stock"]).any():
            raise ValueError("Full-history D2 source has duplicate stock-days")
        if not source["source_profile"].eq(protocol["source_profile"]).all():
            raise ValueError("Full-history D2 source profile does not match")
        source = source.sort_values(
            ["stock", "forecast_date"], kind="mergesort"
        )
        shifted = source.groupby("stock", sort=False)[d2].shift(20)
        donor_date = source.groupby("stock", sort=False)[
            "forecast_date"
        ].shift(20)
        donor = source[["forecast_date", "sector", "stock"]].copy()
        donor["_control_source_date"] = donor_date
        for feature in d2:
            donor[f"_control_{feature}"] = shifted[feature]
        output = original.merge(
            donor,
            on=["forecast_date", "sector", "stock"],
            how="left",
            validate="one_to_one",
        )
        donor_columns = [f"_control_{feature}" for feature in d2]
        if (
            output["_control_source_date"].isna().any()
            or output[donor_columns].isna().any().any()
        ):
            raise ValueError(
                "Full D2 history does not cover every 20-session stale row"
            )
        if not (
            output["_control_source_date"] < output["forecast_date"]
        ).all():
            raise ValueError("Stale-D2 donor dates are not strictly earlier")
        for feature in d2:
            output[feature] = output.pop(f"_control_{feature}")
        audit = {
            "control": "20-session within-stock stale-D2 shift",
            "sessions": 20,
            "wrapped": False,
            "row_count": len(output),
            "all_rows_available": True,
            "first_recipient_date": output["forecast_date"].min().date().isoformat(),
            "last_recipient_date": output["forecast_date"].max().date().isoformat(),
            "first_source_date": output[
                "_control_source_date"
            ].min().date().isoformat(),
            "last_source_date": output[
                "_control_source_date"
            ].max().date().isoformat(),
        }
    else:
        sector_stocks = {
            sector: sorted(group["stock"].unique().tolist())
            for sector, group in original.groupby("sector", sort=True)
        }
        stock_count = int(protocol["source_artifacts"]["stock_count"])
        sector_count = int(protocol["source_artifacts"]["sector_count"])
        if sector_count <= 0 or stock_count % sector_count:
            raise ValueError("Protocol has an invalid balanced-sector contract")
        expected_sector_size = stock_count // sector_count
        mapping: dict[tuple[str, str], str] = {}
        for sector, stocks in sector_stocks.items():
            if len(stocks) != expected_sector_size or len(stocks) < 2:
                raise ValueError(
                    f"Expected {expected_sector_size} stocks in {sector}, "
                    f"found {len(stocks)}"
                )
            for position, stock in enumerate(stocks):
                mapping[(sector, stock)] = stocks[
                    (position + 1) % len(stocks)
                ]
        output = original.copy()
        output["_control_donor_stock"] = [
            mapping[(sector, stock)]
            for sector, stock in zip(
                output["sector"], output["stock"], strict=True
            )
        ]
        donor = original[
            ["forecast_date", "sector", "stock", *d2]
        ].rename(
            columns={
                "stock": "_control_donor_stock",
                **{feature: f"_control_{feature}" for feature in d2},
            }
        )
        output = output.merge(
            donor,
            on=["forecast_date", "sector", "_control_donor_stock"],
            how="left",
            validate="many_to_one",
        )
        donor_columns = [f"_control_{feature}" for feature in d2]
        if output[donor_columns].isna().any().any():
            raise ValueError("Wrong-stock D2 donor join produced missing values")
        for feature in d2:
            output[feature] = output.pop(f"_control_{feature}")
        audit = {
            "control": "fixed within-sector same-date wrong-stock D2 rotation",
            "row_count": len(output),
            "all_rows_available": True,
            "same_date": True,
            "mapping": {
                sector: {
                    stock: mapping[(sector, stock)]
                    for stock in stocks
                }
                for sector, stocks in sector_stocks.items()
            },
        }

    if len(output) != len(original):
        raise AssertionError("Control construction changed the panel row count")
    if output[d2].isna().any().any():
        raise AssertionError("Control construction produced incomplete D2")
    if np.isinf(output[d2].to_numpy(dtype=float)).any():
        raise AssertionError("Control construction produced infinite D2")
    original_keys = original[common.PANEL_KEYS].sort_values(
        common.PANEL_KEYS, kind="mergesort"
    ).reset_index(drop=True)
    output_keys = output[common.PANEL_KEYS].sort_values(
        common.PANEL_KEYS, kind="mergesort"
    ).reset_index(drop=True)
    if not original_keys.equals(output_keys):
        raise AssertionError("Control construction changed panel keys")
    output["_control_available"] = True
    output = output.sort_values(
        ["forecast_date", "sector", "stock"], kind="mergesort"
    ).reset_index(drop=True)
    return output, audit


def train_control(
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
    control = CONTROL_BY_BUNDLE[bundle_key]
    d2_source_provenance: dict[str, Any] | None = None
    d2_path: Path | None = None
    if control == "lag20":
        d2_path, d2_source_provenance = resolve_full_d2_source(protocol)
    panel, control_audit = prepare_control_panel(
        panel, protocol, control, full_d2_path=d2_path
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
        transformed = common.apply_fixed_transforms(panel, features, protocol)
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
    validation_panel = pd.concat(validation_predictions, ignore_index=True)
    prediction_review = common.validate_prediction_panel(
        prediction_panel, protocol=protocol, split_index=2
    )
    validation_review = common.validate_prediction_panel(
        validation_panel, protocol=protocol, split_index=1
    )
    return prediction_panel, validation_panel, fits, {
        "preflight": preflight,
        "control_audit": control_audit,
        "d2_source_provenance": d2_source_provenance,
        "feature_contract": feature_contract,
        "prediction_review": prediction_review,
        "validation_review": validation_review,
    }


def parser() -> argparse.ArgumentParser:
    output = argparse.ArgumentParser(description=__doc__)
    output.add_argument(
        "--bundle", choices=tuple(CONTROL_BY_BUNDLE), required=True
    )
    output.add_argument("--allow-exploratory", action="store_true")
    output.add_argument("--overwrite", action="store_true")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    protocol = common.load_protocol()
    common.initialize_status(common.sha256_file(common.PROTOCOL_PATH))
    common.verify_completed_bundle("D0")
    common.verify_completed_bundle("D3")
    common.ensure_no_existing_bundle(args.bundle, overwrite=args.overwrite)
    common.update_status(args.bundle, "running")
    try:
        predictions, validation, fits, details = train_control(
            args.bundle, allow_exploratory=args.allow_exploratory
        )
        d0_predictions, d0_provenance = common.load_completed_predictions("D0")
        d3_predictions, d3_provenance = common.load_completed_predictions("D3")
        metrics = quant_common.summarize_predictions(predictions).to_dict(
            orient="records"
        )
        fold_metrics = quant_common.summarize_predictions(
            predictions, ("fold", "target", "model")
        ).to_dict(orient="records")
        summary = {
            "status": "complete",
            "metrics": metrics,
            "control": details["control_audit"],
            "base_comparison": common.compare_predictions(
                predictions, d0_predictions
            ),
            "secondary_comparison": common.compare_predictions(
                predictions, d3_predictions
            ),
            "secondary_comparison_label": (
                "Falsification control versus contemporaneous V2-D3"
            ),
        }
        review = {
            "status": "passed",
            **details["prediction_review"],
            "validation_prediction_review": details["validation_review"],
            "outer_evaluation_excluded_from_tuning": True,
            "training_fitted_preprocessing_for_validation_selection": True,
            "validation_loss_hyperparameter_selection": True,
            "train_validation_only_final_preprocessing": True,
            "feature_name_leakage_scan_passed": True,
            "source_timing_contract_bound_by_verified_manifest": (
                details["preflight"]["status"] == "passed"
            ),
            "control_changes_only_d2_normalized": True,
            "control_row_keys_preserved": True,
        }
        dependencies: dict[str, Any] = {
            "source_panel_preflight": details["preflight"],
            "D0_test": d0_provenance,
            "D3_test": d3_provenance,
        }
        if details["d2_source_provenance"] is not None:
            dependencies["full_history_D2"] = details[
                "d2_source_provenance"
            ]
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
                "control": details["control_audit"],
                "alpha_grid": protocol["model_tuning"]["linear_alpha_grid"],
                "l1_ratio_grid": protocol["model_tuning"][
                    "elastic_net_l1_ratio_grid"
                ],
                "fixed_log1p_features": protocol["preprocessing"][
                    "fixed_log1p_features"
                ],
                "source_profile": protocol["source_profile"],
                "claim_scope": protocol["claim_scope"],
            },
            dependencies=dependencies,
            extra_outputs={"control_audit.json": details["control_audit"]},
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
