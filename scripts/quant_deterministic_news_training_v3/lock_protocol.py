#!/usr/bin/env python
"""Hash-lock the exploratory v3 W17-Lite training protocol.

Run this only after the canonical, long-description, and date-sector
permutation panels have been constructed.  The command performs no fitting.
It binds every source byte, the exact row intersection, feature order,
chronological folds, preprocessing, controls, and the failed semantic-gate
override before a model runner will execute.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from scripts.quant_deterministic_news_training_v3 import contract


V2_PROTOCOL = Path("config/quant_deterministic_news_protocol_v2.json")
EXPECTED_ROWS = 27_510
EXPECTED_DATES = 917
EXPECTED_STOCKS = 30
EXPECTED_SECTORS = 5
EXPECTED_SELECTED_ARTICLES = 50_488
REQUIRED_AGREEMENT = 0.85
EXPECTED_AGREEMENT_COUNT = 32_880
EXPECTED_DISAGREEMENT_COUNT = 17_608
EXPECTED_AGREED_ABSTENTION_COUNT = 390
EXPECTED_ACCEPTED_COUNT = 32_490
XGBOOST_RANDOM_SEED = 1729


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain one JSON object")
    return value


def _replace(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, target)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        _replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _manifest_for_panel(path: Path, *, base: bool) -> Path:
    candidates = (
        [contract.BASE_PANEL_MANIFEST, path.with_suffix(".manifest.json")]
        if base
        else [path.with_suffix(".manifest.json"), contract.BASE_PANEL_MANIFEST]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No manifest found for {path}; checked {candidates}"
    )


def _normalize_panel(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    missing = set(contract.PANEL_KEYS) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} misses join keys: {sorted(missing)}")
    frame = frame.copy()
    frame["forecast_date"] = pd.to_datetime(
        frame["forecast_date"], errors="raise"
    ).dt.normalize()
    if frame.duplicated(list(contract.PANEL_KEYS)).any():
        raise ValueError(f"{path} contains duplicate panel keys")
    return frame.sort_values(
        list(contract.PANEL_KEYS), kind="mergesort"
    ).reset_index(drop=True)


def _frame_profile(frame: pd.DataFrame, label: str) -> dict[str, Any]:
    profile = {
        "row_count": len(frame),
        "date_count": int(frame["forecast_date"].nunique()),
        "stock_count": int(frame["stock"].nunique()),
        "sector_count": int(frame["sector"].nunique()),
        "first_date": frame["forecast_date"].min().date().isoformat(),
        "last_date": frame["forecast_date"].max().date().isoformat(),
    }
    expected = {
        "row_count": EXPECTED_ROWS,
        "date_count": EXPECTED_DATES,
        "stock_count": EXPECTED_STOCKS,
        "sector_count": EXPECTED_SECTORS,
        "first_date": "2022-11-01",
        "last_date": "2026-06-30",
    }
    if profile != expected:
        raise ValueError(f"{label} coverage differs: {profile} != {expected}")
    return profile


def _assert_exact_keys(
    base: pd.DataFrame, candidate: pd.DataFrame, label: str
) -> None:
    left = base[list(contract.PANEL_KEYS)]
    right = candidate[list(contract.PANEL_KEYS)]
    if not left.equals(right):
        raise ValueError(f"{label} keys differ from the canonical panel")


def _assert_numeric_contract(
    frame: pd.DataFrame, columns: Sequence[str], label: str
) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{label} misses features: {sorted(missing)}")
    for column in columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.notna().sum() != frame[column].notna().sum():
            raise TypeError(f"{label}/{column} is not numeric")
        observed = numeric.dropna().to_numpy(dtype=float)
        if not np.isfinite(observed).all():
            raise ValueError(f"{label}/{column} contains infinities")


def _assert_shared_columns_equal(
    base: pd.DataFrame,
    candidate: pd.DataFrame,
    columns: Sequence[str],
    label: str,
) -> None:
    for column in columns:
        if column not in base or column not in candidate:
            raise ValueError(f"{label} misses shared column {column}")
        left = base[column]
        right = candidate[column]
        if pd.api.types.is_numeric_dtype(left):
            if not np.array_equal(
                left.to_numpy(), right.to_numpy(), equal_nan=True
            ):
                raise ValueError(f"{label} changed shared column {column}")
        elif not left.equals(right):
            raise ValueError(f"{label} changed shared column {column}")


def _artifact(
    path: Path,
    manifest_path: Path,
    *,
    rows: int | None = None,
) -> dict[str, Any]:
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Missing artifact or manifest: {path}")
    output: dict[str, Any] = {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
    }
    if rows is not None:
        output["rows"] = rows
    return output


def _permutation_provenance() -> dict[str, str]:
    """Validate and bind the builder's exact permutation identity."""

    aggregation = load_json(contract.AGGREGATION_CONFIG)
    variants = aggregation.get("variants")
    permuted = (
        variants.get("permuted")
        if isinstance(variants, Mapping)
        else None
    )
    if not isinstance(permuted, Mapping):
        raise ValueError("Aggregation config lacks variants.permuted")
    seed = permuted.get("seed")
    if not isinstance(seed, str) or not seed:
        raise ValueError("Permutation seed must be a nonempty string")
    if permuted.get("accepted_labels_only") is not True:
        raise ValueError("Permutation must be limited to accepted labels")
    if permuted.get("other_features_contemporaneous") is not True:
        raise ValueError("Permutation must retain other fields")
    if permuted.get("strata") != ["forecast_date", "sector"]:
        raise ValueError("Permutation strata changed")

    manifest = load_json(contract.PERMUTED_DAILY_MANIFEST)
    if (
        manifest.get("variant_id") != "permuted"
        or manifest.get("status")
        != "complete_exploratory_non_version_safe"
    ):
        raise ValueError("Permuted daily manifest is not complete")
    mapping_hash = manifest.get("permutation_mapping_sha256")
    if (
        not isinstance(mapping_hash, str)
        or len(mapping_hash) != 64
        or any(character not in "0123456789abcdef" for character in mapping_hash)
    ):
        raise ValueError("Permutation mapping SHA-256 is missing or malformed")
    config_input = manifest.get("inputs", {}).get("status_cue_rules")
    if (
        not isinstance(config_input, Mapping)
        or config_input.get("path")
        != contract.AGGREGATION_CONFIG.as_posix()
        or config_input.get("sha256")
        != sha256_file(contract.AGGREGATION_CONFIG)
    ):
        raise ValueError(
            "Permuted daily manifest does not bind the aggregation config"
        )
    generated = manifest.get("generated_files", {}).get(
        contract.PERMUTED_DAILY_PATH.name
    )
    if (
        not isinstance(generated, Mapping)
        or generated.get("path")
        != contract.PERMUTED_DAILY_PATH.as_posix()
        or int(generated.get("rows", -1)) != EXPECTED_ROWS
        or generated.get("sha256")
        != sha256_file(contract.PERMUTED_DAILY_PATH)
    ):
        raise ValueError(
            "Permuted daily manifest does not bind the generated panel"
        )
    return {
        "seed": seed,
        "permutation_mapping_sha256": mapping_hash,
        "aggregation_config_path": contract.AGGREGATION_CONFIG.as_posix(),
        "aggregation_config_sha256": sha256_file(
            contract.AGGREGATION_CONFIG
        ),
        "permuted_daily_manifest_sha256": sha256_file(
            contract.PERMUTED_DAILY_MANIFEST
        ),
    }


def _locked_xgboost_tuning(
    v2: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy the exact quant-v1 shallow-tree search into the v3 lock."""

    if (
        v2.get("model_tuning", {}).get(
            "reuse_quant_v1_shallow_xgboost_candidates"
        )
        is not True
    ):
        raise ValueError("v2 did not authorize quant-v1 XGBoost reuse")
    source = load_json(contract.QUANT_V1_PROTOCOL)
    rung = source.get("rung_3")
    if not isinstance(rung, Mapping):
        raise ValueError("quant-v1 rung_3 is missing")
    raw_candidates = rung.get("candidate_configs")
    if not isinstance(raw_candidates, list) or len(raw_candidates) != 4:
        raise ValueError("quant-v1 must define exactly four XGBoost candidates")
    candidates: list[dict[str, int | float]] = []
    for index, raw in enumerate(raw_candidates):
        if not isinstance(raw, Mapping) or not raw:
            raise ValueError(f"XGBoost candidate {index} is malformed")
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
                    f"XGBoost candidate {index}/{key!r} is invalid"
                )
            candidate[key] = value
        candidates.append(candidate)
    n_estimators = int(rung.get("n_estimators", 0))
    early_stopping = int(rung.get("early_stopping_rounds", 0))
    if n_estimators <= 0 or early_stopping <= 0:
        raise ValueError("quant-v1 XGBoost budgets must be positive")
    return {
        "source_protocol": {
            "path": contract.QUANT_V1_PROTOCOL.as_posix(),
            "sha256": sha256_file(contract.QUANT_V1_PROTOCOL),
        },
        "source_implementation": {
            "path": contract.QUANT_V1_XGBOOST_IMPLEMENTATION.as_posix(),
            "sha256": sha256_file(
                contract.QUANT_V1_XGBOOST_IMPLEMENTATION
            ),
        },
        "candidate_configs": candidates,
        "candidate_configs_ordered_sha256": sha256_json(candidates),
        "n_estimators": n_estimators,
        "early_stopping_rounds": early_stopping,
        "random_seed": XGBOOST_RANDOM_SEED,
    }


def _scan_inference_gate() -> dict[str, Any]:
    manifest = load_json(contract.PREDICTIONS_MANIFEST)
    if (
        manifest.get("status") != "complete"
        or manifest.get("all_selected_articles_terminal") is not True
        or int(manifest.get("selected_article_count", -1))
        != EXPECTED_SELECTED_ARTICLES
        or int(manifest.get("successful_article_count", -1))
        != EXPECTED_SELECTED_ARTICLES
        or int(manifest.get("failure_article_count", -1)) != 0
        or manifest.get("output_sha256")
        != sha256_file(contract.PREDICTIONS_PATH)
    ):
        raise ValueError("W17-Lite inference manifest is not complete")

    terminal = 0
    agreement = 0
    disagreement = 0
    agreed_abstention = 0
    accepted = 0
    seen: set[str] = set()
    with contract.PREDICTIONS_PATH.open("rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            article_id = str(record.get("article_id", ""))
            if not article_id or article_id in seen:
                raise ValueError(
                    f"Duplicate/empty inference ID at line {line_number}"
                )
            seen.add(article_id)
            if record.get("terminal_state") != "complete":
                raise ValueError("W17-Lite inference contains a failed row")
            event = record.get("event_group")
            if not isinstance(event, Mapping):
                raise ValueError("W17-Lite inference lacks event_group")
            terminal += 1
            same = (
                event.get("canonical_prediction")
                == event.get("reversed_prediction")
            )
            agreement += int(same)
            disagreement += int(not same)
            agreed_abstention += int(
                same
                and event.get("canonical_prediction")
                == "other_or_unclear"
            )
            accepted += int(event.get("accepted") is True)

    observed = {
        "terminal_count": terminal,
        "agreement_count": agreement,
        "disagreement_count": disagreement,
        "agreed_abstention_count": agreed_abstention,
        "accepted_predictive_count": accepted,
    }
    expected = {
        "terminal_count": EXPECTED_SELECTED_ARTICLES,
        "agreement_count": EXPECTED_AGREEMENT_COUNT,
        "disagreement_count": EXPECTED_DISAGREEMENT_COUNT,
        "agreed_abstention_count": EXPECTED_AGREED_ABSTENTION_COUNT,
        "accepted_predictive_count": EXPECTED_ACCEPTED_COUNT,
    }
    if observed != expected:
        raise ValueError(
            f"W17-Lite semantic-gate counts changed: {observed} != {expected}"
        )
    rate = agreement / terminal
    if not math.isclose(
        rate,
        EXPECTED_AGREEMENT_COUNT / EXPECTED_SELECTED_ARTICLES,
        rel_tol=0,
        abs_tol=1e-15,
    ):
        raise AssertionError("Agreement-rate computation is inconsistent")
    return {
        **observed,
        "observed_choice_order_agreement": rate,
        "observed_choice_order_agreement_percent": 100.0 * rate,
        "required_choice_order_agreement": REQUIRED_AGREEMENT,
        "passed": rate >= REQUIRED_AGREEMENT,
        "override": {
            "authorized": True,
            "authorization_basis": (
                "User explicitly requested feature construction and training "
                "after completing FLAN W17-Lite inference on 2026-07-30."
            ),
            "result_role": "exploratory_failed_semantic_gate_diagnostic",
            "primary_training_eligible": False,
            "confirmatory_eligible": False,
            "reported_approximation": "65.125% < 85%",
            "exact_observed_percent": 100.0 * rate,
        },
    }


def build_protocol() -> dict[str, Any]:
    v2 = load_json(V2_PROTOCOL)
    if v2.get("status") != "locked_before_training":
        raise ValueError("The source v2 protocol is not locked")

    base = _normalize_panel(contract.BASE_PANEL_PATH)
    long_panel = _normalize_panel(contract.LONG_PANEL_PATH)
    permuted = _normalize_panel(contract.PERMUTED_PANEL_PATH)
    profile = _frame_profile(base, "canonical joined panel")
    _frame_profile(long_panel, "long-description joined panel")
    _frame_profile(permuted, "permuted joined panel")
    _assert_exact_keys(base, long_panel, "long-description panel")
    _assert_exact_keys(base, permuted, "permuted panel")

    feature_blocks = v2.get("feature_blocks")
    if not isinstance(feature_blocks, Mapping):
        raise ValueError("v2 feature blocks are missing")
    d2 = tuple(feature_blocks.get("d2_normalized_30", ()))
    if len(d2) != 30 or len(set(d2)) != 30:
        raise ValueError("v2 D2-Normalized contract changed")
    q_by_target = {
        target: tuple(feature_blocks[f"q56_{target.split('_', 1)[1]}"])
        for target in contract.TARGETS
    }
    if any(len(values) != 56 for values in q_by_target.values()):
        raise ValueError("v2 Q56 contract changed")
    all_model_columns = sorted(
        set(d2)
        | set(contract.WLITE_17)
        | {name for values in q_by_target.values() for name in values}
    )
    _assert_numeric_contract(base, all_model_columns, "canonical panel")
    _assert_numeric_contract(
        long_panel, all_model_columns, "long-description panel"
    )
    _assert_numeric_contract(
        permuted, all_model_columns, "permuted panel"
    )
    shared = sorted(set(base.columns) - set(contract.WLITE_17))
    _assert_shared_columns_equal(
        base, long_panel, shared, "long-description panel"
    )
    permutation_shared = sorted(
        set(base.columns) - set(contract.EVENT_CONTENT_4)
    )
    _assert_shared_columns_equal(
        base, permuted, permutation_shared, "permuted panel"
    )

    no_selected = base["wlite_observed_no_selected_article"]
    if not no_selected.isin([0, 1]).all():
        raise ValueError("W17-Lite no-selected indicator is not binary")
    no_selected_rows = int(no_selected.sum())
    if no_selected_rows != 204:
        raise ValueError(
            f"Expected 204 no-candidate rows, observed {no_selected_rows}"
        )
    nullable = [
        column
        for column in contract.WLITE_17
        if base[column].isna().any()
    ]
    expected_nullable = sorted(
        set(contract.WLITE_17)
        - {"wlite_observed_no_selected_article"}
    )
    if sorted(nullable) != expected_nullable:
        raise ValueError(
            "W17-Lite nullable columns do not follow the no-selected contract"
        )
    missing_mask = base[expected_nullable].isna().all(axis=1)
    if not missing_mask.equals(no_selected.astype(bool)):
        raise ValueError(
            "W17-Lite missingness is not exactly identified by no-selected"
        )

    base_manifest = _manifest_for_panel(contract.BASE_PANEL_PATH, base=True)
    long_manifest = _manifest_for_panel(contract.LONG_PANEL_PATH, base=False)
    permuted_manifest = _manifest_for_panel(
        contract.PERMUTED_PANEL_PATH, base=False
    )
    permutation_provenance = _permutation_provenance()
    semantic_gate = _scan_inference_gate()
    if semantic_gate["passed"]:
        raise ValueError(
            "Expected the frozen W17-Lite choice-order gate to fail"
        )

    source_artifacts = {
        "canonical_joined_panel": _artifact(
            contract.BASE_PANEL_PATH, base_manifest, rows=len(base)
        ),
        "long_description_joined_panel": _artifact(
            contract.LONG_PANEL_PATH, long_manifest, rows=len(long_panel)
        ),
        "permuted_joined_panel": _artifact(
            contract.PERMUTED_PANEL_PATH,
            permuted_manifest,
            rows=len(permuted),
        ),
        "canonical_daily_wlite": _artifact(
            contract.DAILY_PATH, contract.DAILY_MANIFEST, rows=EXPECTED_ROWS
        ),
        "long_description_daily_wlite": _artifact(
            contract.LONG_DAILY_PATH,
            contract.LONG_DAILY_MANIFEST,
            rows=EXPECTED_ROWS,
        ),
        "permuted_daily_wlite": _artifact(
            contract.PERMUTED_DAILY_PATH,
            contract.PERMUTED_DAILY_MANIFEST,
            rows=EXPECTED_ROWS,
        ),
        "flan_predictions": _artifact(
            contract.PREDICTIONS_PATH,
            contract.PREDICTIONS_MANIFEST,
            rows=EXPECTED_SELECTED_ARTICLES,
        ),
    }
    fixed_log1p = v2.get("preprocessing", {}).get(
        "fixed_log1p_features"
    )
    if not isinstance(fixed_log1p, list):
        raise ValueError("v2 fixed-log1p contract is missing")
    model_tuning = dict(v2["model_tuning"])
    model_tuning["shallow_xgboost"] = _locked_xgboost_tuning(v2)

    return {
        "experiment_id": "quant-deterministic-news-v3-w17-lite",
        "protocol_version": "3.0",
        "status": "locked_before_training",
        "generated_at_utc": utc_now(),
        "claim_scope": (
            "Exploratory failed-semantic-gate development diagnostic only. "
            "The retrospective Massive text is not version-safe and FLAN "
            "W17-Lite choice-order agreement missed its frozen threshold."
        ),
        "source_profile": "ordinary_massive_retrospective",
        "arm_id": "WL17__flan_t5_xl",
        "contract_id": "weak-news-semantics-lite-v1",
        "extractor": {
            "model_id": "google/flan-t5-xl",
            "model_revision": (
                "7d6315df2c2fb742f0f5b556879d730926ca9001"
            ),
            "selector": "K16",
            "selected_articles": EXPECTED_SELECTED_ARTICLES,
            "selected_assignment_rows": 438_522,
            "selected_weight_fraction": 0.9732613743014124,
        },
        "semantic_gate": semantic_gate,
        "claim_flags": {
            "exploratory_fit_authorized": True,
            "semantic_gate_passed": False,
            "point_in_time_version_safe": False,
            "primary_training_eligible": False,
            "confirmatory_eligible": False,
            "human_reference_labels": False,
        },
        "targets": list(contract.TARGETS),
        "join_keys": list(contract.PANEL_KEYS),
        "coverage": {
            **profile,
            "candidate_bearing_stock_days": 27_306,
            "no_candidate_stock_days": no_selected_rows,
            "all_quant_rows_matched": True,
        },
        "feature_blocks": {
            "q56_by_target": {
                key: list(values) for key, values in q_by_target.items()
            },
            "d2_normalized_30": list(d2),
            "wlite_17": list(contract.WLITE_17),
            "event_content_4": list(contract.EVENT_CONTENT_4),
            "semantic_content_11": list(contract.SEMANTIC_CONTENT_11),
            "coverage_text_6": list(contract.COVERAGE_TEXT_6),
        },
        "rungs": {
            bundle.key: {
                "label": bundle.label,
                "architecture": bundle.architecture,
                "control": bundle.control,
            }
            for bundle in contract.BUNDLES
        },
        "controls": {
            "coverage_text_only": {
                "features": list(contract.COVERAGE_TEXT_6),
                "rule": "remove all 11 event/routing/status-content fields",
            },
            "stale_event_lag20": {
                "sessions": 20,
                "features_shifted_within_stock": list(
                    contract.EVENT_CONTENT_4
                ),
                "contemporaneous_features": [
                    name
                    for name in contract.WLITE_17
                    if name not in contract.EVENT_CONTENT_4
                ],
                "wrap": False,
                "initial_missing_values": (
                    "retained and imputed from the current training partition"
                ),
            },
            "wrong_stock": {
                "features_rotated": list(contract.SEMANTIC_CONTENT_11),
                "features_retained_contemporaneously": list(
                    contract.COVERAGE_TEXT_6
                ),
                "mapping": (
                    "alphabetically next stock within the same fixed sector, "
                    "with wraparound; constant across all dates"
                ),
            },
            "date_sector_permutation": {
                "features_permuted_together": list(
                    contract.EVENT_CONTENT_4
                ),
                "features_retained_contemporaneously": list(
                    name
                    for name in contract.WLITE_17
                    if name not in contract.EVENT_CONTENT_4
                ),
                "permutation_unit": (
                    "accepted predictive event-label identity within "
                    "forecast_date and sector"
                ),
                "invariants": (
                    "acceptance, disagreement, abstention/other mass, routing, "
                    "rule cues, selection, and text-quality fields remain "
                    "contemporaneous"
                ),
                **permutation_provenance,
                "panel": contract.PERMUTED_PANEL_PATH.as_posix(),
            },
            "long_description": {
                "selection": (
                    "selected articles with nonempty retained description "
                    "length at least 150 characters"
                ),
                "panel": contract.LONG_PANEL_PATH.as_posix(),
                "role": "sensitivity_not_required_to_beat",
            },
            "full_history_vs_quota": {
                "status": "skipped",
                "reason": (
                    "No full-55,197-article W17-Lite inference exists."
                ),
            },
        },
        "folds": v2["folds"],
        "target_row_eligibility": v2["target_row_eligibility"],
        "preprocessing": {
            "fixed_log1p_features": fixed_log1p,
            "elastic_net_imputation": "training-partition median only",
            "elastic_net_scaling": (
                "training-partition mean and standard deviation only"
            ),
            "wlite_missingness": (
                "Sixteen nullable fields are missing exactly on the 204 "
                "no-selected rows; preserve the observed-no-selected feature."
            ),
            "extra_missingness_indicators": [],
            "xgboost_missing_policy": "native missing branches",
        },
        "model_tuning": model_tuning,
        "metrics": v2["metrics"],
        "inference": v2["inference"],
        "gates": {
            "s4_validation_gate": (
                "S3 must beat S1 validation MSE in at least two of three "
                "validation blocks, applied separately by target."
            ),
            "useful_semantic_gate": {
                "candidate": "S3",
                "must_beat": [
                    "S1",
                    "C-S3-COV",
                    "C-S3-L20",
                    "C-S3-WS",
                    "C-S3-PERM",
                ],
                "bootstrap_upper_bound_below_zero": True,
                "minimum_improved_development_folds": 2,
                "long_description_is_sensitivity_only": True,
            },
            "q_plus_l_gate": {
                "candidate": "S2",
                "must_beat": [
                    "S0",
                    "C-S2-COV",
                    "C-S2-L20",
                    "C-S2-WS",
                    "C-S2-PERM",
                ],
                "bootstrap_upper_bound_below_zero": True,
                "minimum_improved_development_folds": 2,
                "long_description_is_sensitivity_only": True,
            },
        },
        "source_artifacts": source_artifacts,
        "source_protocol": {
            "path": V2_PROTOCOL.as_posix(),
            "sha256": sha256_file(V2_PROTOCOL),
        },
        "implementation": {
            path.as_posix(): sha256_file(path)
            for path in contract.BOUND_TRAINING_IMPLEMENTATION
        },
    }


def parser() -> argparse.ArgumentParser:
    output = argparse.ArgumentParser(description=__doc__)
    output.add_argument("--overwrite", action="store_true")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if (
        not args.overwrite
        and (
            contract.PROTOCOL_PATH.exists()
            or contract.PROTOCOL_SIDECAR.exists()
        )
    ):
        raise FileExistsError(
            "v3 protocol already exists; pass --overwrite only before fits"
        )
    protocol = build_protocol()
    payload = json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    atomic_text(contract.PROTOCOL_PATH, payload)
    protocol_hash = sha256_file(contract.PROTOCOL_PATH)
    atomic_text(
        contract.PROTOCOL_SIDECAR,
        f"{protocol_hash}  {contract.PROTOCOL_PATH.name}\n",
    )
    print(
        json.dumps(
            {
                "status": protocol["status"],
                "path": contract.PROTOCOL_PATH.as_posix(),
                "sha256": protocol_hash,
                "semantic_gate_passed": False,
                "semantic_gate_override": True,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
