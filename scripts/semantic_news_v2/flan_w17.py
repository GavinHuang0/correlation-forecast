#!/usr/bin/env python
"""Construct article predictions and daily WLLM17 features with FLAN-T5-XL.

The input is an upstream, deterministically routed v2 article-assignment
artifact.  This module deliberately does not run the legacy FLAN relevance
gate: every supplied C/I/P assignment is a WLLM candidate and all three W17
fields are structurally applicable.

Inference is sharded by ``article_id`` and appends one terminal JSONL record
at a time, making interrupted runs resumable. Article-consistent sharding lets
the byte-identical article-only event/status prompts be cached safely while
shock scope remains target-specific. No prompt is truncated; an over-limit
assignment receives a terminal ``prompt_too_long`` record.
Daily aggregation is fail-closed: incomplete source scope, an incomplete
assignment ledger, a missing/failed prediction, or a provenance mismatch
invalidates the entire W17 stock-day.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sqlite3
import sys
import tempfile
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Protocol, Sequence

import numpy as np
try:
    import pandas as pd
except ImportError:  # The CUDA inference environment intentionally omits pandas.
    pd = None


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIRECTORY = REPOSITORY_ROOT / "scripts"
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

import experiment_flan_t5_xl_v1_1 as active_calibration  # noqa: E402
import extract_flan_t5 as flan_base  # noqa: E402
import extract_flan_t5_coarse as coarse_runner  # noqa: E402
import run_flan_t5_xl_active as active_runner  # noqa: E402
import run_flan_t5_xl_coarse as xl_runner  # noqa: E402


PIPELINE_VERSION = "flan-w17-v2-construction-v1"
CONTRACT_ID = "weak-news-semantics-v1"
EXTRACTOR_ID = "flan-t5-xl-v1.1"
MODEL_ID = "google/flan-t5-xl"
MODEL_REVISION = "7d6315df2c2fb742f0f5b556879d730926ca9001"
MAX_INPUT_TOKENS = 512
W17_FIELDS = ("shock_scope", "event_family", "information_status")
ABSTENTION_LABELS = {
    "shock_scope": ("unclear",),
    "event_family": ("other_or_unclear",),
    "information_status": ("unclear",),
}
PREDICTIVE_LABELS = {
    "shock_scope": ("idiosyncratic", "common", "mixed"),
    "event_family": (
        "earnings_guidance",
        "product_demand",
        "supply_capacity",
        "regulation_legal",
        "corporate_analyst",
        "macro_market",
    ),
    "information_status": (
        "confirmed",
        "anticipated",
        "rumor_or_opinion",
    ),
}
ROLE_ALIASES = {
    "C": "common",
    "common": "common",
    "I": "target_idiosyncratic",
    "target_idiosyncratic": "target_idiosyncratic",
    "P": "peer_idiosyncratic",
    "peer_idiosyncratic": "peer_idiosyncratic",
}
KEY_COLUMNS = ("forecast_date", "sector", "stock", "benchmark")
WLLM17_COLUMNS = (
    "wllm_scope_share_idiosyncratic",
    "wllm_scope_share_common",
    "wllm_scope_share_mixed",
    "wllm_event_share_earnings_guidance",
    "wllm_event_share_product_demand",
    "wllm_event_share_supply_capacity",
    "wllm_event_share_regulation_legal",
    "wllm_event_share_corporate_analyst",
    "wllm_event_share_macro_market",
    "wllm_status_share_confirmed",
    "wllm_status_share_anticipated",
    "wllm_status_share_rumor_or_opinion",
    "wllm_scope_common_minus_idiosyncratic",
    "wllm_observed_no_eligible_semantic_article",
    "wllm_coverage_shock_scope",
    "wllm_coverage_event_family",
    "wllm_coverage_information_status",
)
FIELD_PREFIX = {
    "shock_scope": "wllm_scope_share_",
    "event_family": "wllm_event_share_",
    "information_status": "wllm_status_share_",
}
SUCCESS_TERMINAL_STATES = {"complete"}
FAILURE_TERMINAL_STATES = {
    "prompt_too_long",
    "inference_failed",
    "invalid_input",
}
ALL_TERMINAL_STATES = SUCCESS_TERMINAL_STATES | FAILURE_TERMINAL_STATES
DEFAULT_SEMANTIC_ROOT = (
    REPOSITORY_ROOT
    / "data"
    / "features"
    / "news_semantic"
    / "massive_v2"
)
DEFAULT_ASSIGNMENTS = (
    DEFAULT_SEMANTIC_ROOT / "article_target_assignments.jsonl.gz"
)
DEFAULT_STOCK_DAYS = DEFAULT_SEMANTIC_ROOT / "stock_day_scope.jsonl.gz"
DEFAULT_CORPUS_MANIFEST = DEFAULT_SEMANTIC_ROOT / "manifest.json"
DEFAULT_PREFLIGHT = DEFAULT_SEMANTIC_ROOT / "flan_w17" / "preflight.json"
DEFAULT_DAILY_OUTPUT = DEFAULT_SEMANTIC_ROOT / "flan_w17" / "daily_wllm17.parquet"
DEFAULT_PRIVATE_WORK_ROOT = DEFAULT_SEMANTIC_ROOT / "flan_w17" / "_private_work"
MAX_ASSIGNMENTS_PER_INFERENCE_SHARD = 25_000
DEFAULT_INFERENCE_SHARD_COUNT = 32
SYSTEMIC_FAILURE_ABORT_THRESHOLD = 3


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def runtime_contract() -> dict[str, Any]:
    """Return the immutable files that define prompt and score semantics."""

    _contract, schema_path, calibration_path = (
        active_runner.load_active_contract()
    )
    snapshot_manifest = xl_runner.SNAPSHOT_MANIFEST
    if not snapshot_manifest.is_file():
        raise FileNotFoundError(snapshot_manifest)
    packages: dict[str, str | None] = {}
    for package in (
        "numpy",
        "torch",
        "transformers",
        "tokenizers",
        "sentencepiece",
        "safetensors",
        "accelerate",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    cuda: dict[str, Any] = {
        "required_for_inference": True,
        "model_dtype": "float16",
        "available": False,
    }
    try:
        import torch
    except ImportError:
        pass
    else:
        cuda.update(
            {
                "available": bool(torch.cuda.is_available()),
                "torch_cuda_version": torch.version.cuda,
                "cudnn_version": (
                    torch.backends.cudnn.version()
                    if torch.backends.cudnn.is_available()
                    else None
                ),
            }
        )
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            cuda.update(
                {
                    "device_index": 0,
                    "device_name": properties.name,
                    "compute_capability": [
                        properties.major,
                        properties.minor,
                    ],
                    "total_memory_bytes": properties.total_memory,
                }
            )
    return {
        "pipeline_version": PIPELINE_VERSION,
        "extractor_id": EXTRACTOR_ID,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "maximum_input_tokens": MAX_INPUT_TOKENS,
        "active_pointer_sha256": sha256_file(active_runner.ACTIVE_POINTER),
        "schema_sha256": sha256_file(schema_path),
        "calibration_sha256": sha256_file(calibration_path),
        "snapshot_manifest_sha256": sha256_file(snapshot_manifest),
        "prompt_runner_sha256": sha256_file(
            Path(coarse_runner.__file__).resolve()
        ),
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
        "execution_environment": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "packages": packages,
            "cuda": cuda,
        },
    }


def runtime_contract_sha256() -> str:
    return sha256_json(runtime_contract())


def shared_model_text(headline: str, description: str) -> str:
    """Return the one canonical byte view shared by FLAN and GPT."""

    return headline if not description else f"{headline}\n\n{description}"


def _require_pandas() -> Any:
    if pd is None:
        raise RuntimeError(
            "Parquet/daily aggregation requires pandas; run that stage in "
            "the repository's data/training environment"
        )
    return pd


def default_acceptance_config() -> dict[str, Any]:
    """Return the explicitly exploratory, silver-only W17 acceptance lock.

    A zero margin floor intentionally adds no undocumented tuning.  It still
    applies a field/class-specific rule and excludes every ontology abstention.
    Later human calibration must be a separately versioned configuration.
    """

    return {
        "config_version": "flan-w17-acceptance-v1",
        "status": "frozen_exploratory_silver_fit",
        "contract_id": CONTRACT_ID,
        "extractor_id": EXTRACTOR_ID,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "score": "calibrated_adjusted_top1_top2_margin",
        "quality_weight": 1.0,
        "human_calibrated": False,
        "primary_training_eligible": False,
        "confirmatory_eligible": False,
        "result_role": "exploratory_silver_fit",
        "fields": {
            field: {
                "abstention_labels": list(ABSTENTION_LABELS[field]),
                "classes": {
                    label: {
                        "enabled_for_exploratory_fit": True,
                        "enabled_by_human_calibration": False,
                        "minimum_adjusted_margin": 0.0,
                    }
                    for label in PREDICTIVE_LABELS[field]
                },
            }
            for field in W17_FIELDS
        },
        "warning": (
            "This permissive acceptance lock is frozen for the requested "
            "silver-only exploratory fit. It is not human calibrated and "
            "does not make W17 primary or confirmatory evidence."
        ),
    }


def load_acceptance_config(path: Path | None) -> dict[str, Any]:
    config = (
        default_acceptance_config()
        if path is None
        else json.loads(path.read_text(encoding="utf-8"))
    )
    validate_acceptance_config(config)
    return config


def validate_acceptance_config(config: Mapping[str, Any]) -> None:
    if config.get("config_version") != "flan-w17-acceptance-v1":
        raise ValueError("Unsupported W17 acceptance config version")
    if config.get("status") != "frozen_exploratory_silver_fit":
        raise ValueError("W17 acceptance configuration is not frozen")
    expected = {
        "contract_id": CONTRACT_ID,
        "extractor_id": EXTRACTOR_ID,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    for name, value in expected.items():
        if config.get(name) != value:
            raise ValueError(f"W17 acceptance config has wrong {name}")
    quality = config.get("quality_weight")
    if not isinstance(quality, (int, float)) or float(quality) != 1.0:
        raise ValueError("Weak-contract quality_weight must equal one")
    fields = config.get("fields")
    if not isinstance(fields, Mapping) or set(fields) != set(W17_FIELDS):
        raise ValueError("Acceptance fields must match the ordered W17 fields")
    for field in W17_FIELDS:
        rule = fields[field]
        if tuple(rule.get("abstention_labels", ())) != ABSTENTION_LABELS[field]:
            raise ValueError(f"Acceptance abstentions differ for {field}")
        classes = rule.get("classes")
        if (
            not isinstance(classes, Mapping)
            or set(classes) != set(PREDICTIVE_LABELS[field])
        ):
            raise ValueError(f"Acceptance classes differ for {field}")
        for label, class_rule in classes.items():
            if not isinstance(class_rule, Mapping):
                raise TypeError(f"Invalid class rule for {field}/{label}")
            enabled = class_rule.get("enabled_for_exploratory_fit")
            if not isinstance(enabled, bool):
                raise TypeError(f"Missing enable flag for {field}/{label}")
            margin = class_rule.get("minimum_adjusted_margin")
            if (
                not isinstance(margin, (int, float))
                or not math.isfinite(float(margin))
                or float(margin) < 0
            ):
                raise ValueError(f"Invalid margin for {field}/{label}")
    if config.get("score") != "calibrated_adjusted_top1_top2_margin":
        raise ValueError("W17 acceptance score contract changed")
    if config.get("human_calibrated") is not False:
        raise ValueError("W17 v1 acceptance must remain silver-only")
    if config.get("primary_training_eligible") is not False:
        raise ValueError("Silver-only acceptance cannot be primary eligible")
    if config.get("confirmatory_eligible") is not False:
        raise ValueError("Silver-only acceptance cannot be confirmatory")
    if config.get("result_role") != "exploratory_silver_fit":
        raise ValueError("Silver-only acceptance has the wrong result role")
    for field in W17_FIELDS:
        for label in PREDICTIVE_LABELS[field]:
            if (
                config["fields"][field]["classes"][label].get(
                    "enabled_by_human_calibration"
                )
                is not False
            ):
                raise ValueError(
                    "Silver-only acceptance cannot claim human calibration"
                )
    if canonical_json(config) != canonical_json(default_acceptance_config()):
        raise ValueError(
            "W17 v1 accepts only the exact frozen silver-only configuration"
        )


def _bool(value: Any, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(value)
    raise TypeError(f"{name} must be boolean")


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _possibly_empty_string(value: Any, name: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value.strip()


def _date_string(value: Any, name: str = "forecast_date") -> str:
    try:
        return date.fromisoformat(str(value)[:10]).isoformat()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an ISO date") from exc


def _target_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    raw = record.get("target")
    if isinstance(raw, Mapping):
        ticker = raw.get("ticker", raw.get("stock"))
        company = raw.get("company")
        sector = raw.get("sector", record.get("sector"))
        benchmark = raw.get("benchmark", record.get("benchmark"))
        peers = raw.get("known_sector_peers", raw.get("peers"))
    else:
        ticker = raw or record.get("target_ticker") or record.get("stock")
        company = record.get("target_company")
        sector = record.get("sector")
        benchmark = record.get("benchmark")
        peers = record.get("known_sector_peers")
    if not isinstance(peers, (list, tuple)) or not all(
        isinstance(value, str) and value.strip() for value in peers
    ):
        raise ValueError("target known_sector_peers must be a string list")
    normalized_peers = [value.strip().upper() for value in peers]
    if len(normalized_peers) != len(set(normalized_peers)):
        raise ValueError("target known_sector_peers contain duplicates")
    target = {
        "ticker": _string(ticker, "target.ticker").upper(),
        "company": _string(company, "target.company"),
        "sector": _string(sector, "target.sector"),
        "benchmark": _string(benchmark, "target.benchmark").upper(),
        "known_sector_peers": normalized_peers,
    }
    if target["ticker"] in target["known_sector_peers"]:
        raise ValueError("target ticker cannot be one of its peers")
    return target


def _hash_from(record: Mapping[str, Any], *names: str) -> str | None:
    for name in names:
        value = record.get(name)
        if value is not None:
            return str(value).lower()
    return None


def normalize_assignment(record: Mapping[str, Any]) -> dict[str, Any]:
    headline = _string(record.get("headline"), "headline")
    description = _possibly_empty_string(
        record.get("description"), "description"
    )
    text = shared_model_text(headline, description)
    expected_hashes = {
        "headline_sha256": sha256_text(headline),
        "description_sha256": sha256_text(description),
        "text_sha256": sha256_text(text),
    }
    supplied_hashes = {
        "headline_sha256": _hash_from(record, "headline_sha256"),
        "description_sha256": _hash_from(record, "description_sha256"),
        "text_sha256": _hash_from(
            record, "text_sha256", "model_text_sha256", "shared_text_sha256"
        ),
    }
    for name, expected in expected_hashes.items():
        supplied = supplied_hashes[name]
        if supplied is None:
            raise ValueError(f"{name} is required")
        if supplied != expected:
            raise ValueError(f"{name} does not match supplied text")

    raw_role = _string(record.get("role"), "role")
    if raw_role not in ROLE_ALIASES:
        raise ValueError(f"role must be one of {sorted(ROLE_ALIASES)}")
    target = _target_payload(record)
    source_profile = _string(record.get("source_profile"), "source_profile")
    source_complete = _bool(
        record.get("source_query_scope_complete"),
        "source_query_scope_complete",
    )
    assignment_complete = _bool(
        record.get("candidate_assignment_complete"),
        "candidate_assignment_complete",
    )
    if "aggregation_weight" in record:
        weight = float(record["aggregation_weight"])
    else:
        age = float(record.get("article_age_hours"))
        group_size = int(record.get("duplication_group_size"))
        if age < 0:
            raise ValueError("article_age_hours cannot be negative")
        if group_size < 1:
            raise ValueError("duplication_group_size must be positive")
        weight = math.exp(-math.log(2.0) * age / 12.0) / group_size
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError("aggregation_weight must be positive and finite")

    normalized = {
        "assignment_id": _string(record.get("assignment_id"), "assignment_id"),
        "article_id": _string(
            record.get("article_id", record.get("provider_article_id")),
            "article_id",
        ),
        "forecast_date": _date_string(record.get("forecast_date")),
        "target": target,
        "role": ROLE_ALIASES[raw_role],
        "headline": headline,
        "description": description,
        "description_available": _bool(
            record.get("description_available", bool(description)),
            "description_available",
        ),
        **expected_hashes,
        "source_profile": source_profile,
        "source_query_scope_complete": source_complete,
        "candidate_assignment_complete": assignment_complete,
        "aggregation_weight": weight,
    }
    normalized["assignment_input_sha256"] = sha256_json(normalized)
    return normalized


def normalize_assignments(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized = [normalize_assignment(record) for record in records]
    ids = [record["assignment_id"] for record in normalized]
    if len(ids) != len(set(ids)):
        raise ValueError("Assignment input contains duplicate assignment_id values")
    normalized.sort(key=lambda record: record["assignment_id"])
    return normalized


def normalize_stock_day(record: Mapping[str, Any]) -> dict[str, Any]:
    target = _target_payload(record)
    expected_count = record.get("expected_assignment_count")
    if not isinstance(expected_count, (int, np.integer)) or int(expected_count) < 0:
        raise ValueError("expected_assignment_count must be a nonnegative integer")
    output = {
        "forecast_date": _date_string(record.get("forecast_date")),
        "sector": target["sector"],
        "stock": target["ticker"],
        "benchmark": target["benchmark"],
        "source_profile": _string(record.get("source_profile"), "source_profile"),
        "source_query_scope_complete": _bool(
            record.get("source_query_scope_complete"),
            "source_query_scope_complete",
        ),
        "candidate_assignment_complete": _bool(
            record.get("candidate_assignment_complete"),
            "candidate_assignment_complete",
        ),
        "expected_assignment_count": int(expected_count),
    }
    expected_hash = record.get("expected_assignment_ids_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError("expected_assignment_ids_sha256 is required")
    output["expected_assignment_ids_sha256"] = expected_hash.lower()
    return output


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".jsonl") or suffixes.endswith(".jsonl.gz"):
        opener = gzip.open if suffixes.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"{path}:{line_number}: expected an object")
                yield value
        return
    if suffixes.endswith(".parquet"):
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Reading Parquet requires pyarrow") from exc
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=4_096):
            for record in batch.to_pylist():
                yield record
        return
    raise ValueError(f"Unsupported input format for {path}; use JSONL or Parquet")


def read_records(path: Path) -> list[dict[str, Any]]:
    return list(iter_records(path))


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_records(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    suffixes = "".join(path.suffixes).lower()
    path.parent.mkdir(parents=True, exist_ok=True)
    if suffixes.endswith(".jsonl") or suffixes.endswith(".jsonl.gz"):
        if suffixes.endswith(".gz"):
            with tempfile.NamedTemporaryFile(
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=path.parent,
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
            try:
                with gzip.open(
                    temporary,
                    "wt",
                    encoding="utf-8",
                    newline="\n",
                ) as handle:
                    for record in records:
                        handle.write(canonical_json(record) + "\n")
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
        else:
            value = "".join(
                canonical_json(record) + "\n" for record in records
            )
            _atomic_text(path, value)
        return
    if suffixes.endswith(".parquet"):
        pandas = _require_pandas()
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        try:
            pandas.DataFrame(records).to_parquet(temporary, index=False)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return
    raise ValueError(f"Unsupported output format for {path}; use JSONL or Parquet")


def shard_for(assignment_id: str, shard_count: int) -> int:
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    digest = hashlib.sha256(assignment_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % shard_count


def assignment_shard(
    assignment: Mapping[str, Any],
    shard_count: int,
) -> int:
    """Keep all target assignments for one article in the same shard.

    Shock scope remains target-specific. Event family and information status
    are article-only prompts, so article-consistent sharding makes their exact
    results safely reusable without changing model inputs or outputs.
    """

    article_id = _string(assignment.get("article_id"), "article_id")
    return shard_for(article_id, shard_count)


def select_shard(
    assignments: Sequence[Mapping[str, Any]],
    shard_index: int,
    shard_count: int,
) -> list[dict[str, Any]]:
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must lie in [0, shard_count)")
    return [
        dict(record)
        for record in assignments
        if assignment_shard(record, shard_count) == shard_index
    ]


def extractor_record(assignment: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt a v2 assignment to the frozen prompt builder without re-gating."""

    target = assignment["target"]
    return {
        "article_id": assignment["assignment_id"],
        "headline": assignment["headline"],
        "article_text": assignment["description"],
        "target": {
            "ticker": target["ticker"],
            "company": target["company"],
            "sector": target["sector"],
            "sector_benchmark": target["benchmark"],
            "known_sector_peers": list(target["known_sector_peers"]),
        },
    }


def prompt_token_counts(
    assignment: Mapping[str, Any],
    tokenizer: Any,
    max_input_tokens: int = MAX_INPUT_TOKENS,
) -> dict[str, int]:
    record = extractor_record(assignment)
    counts: dict[str, int] = {}
    violations = []
    for field in W17_FIELDS:
        prompts = coarse_runner.prompt_variants_for_preflight(
            record,
            _active_schema(),
            field,
            "zero_shot",
        )
        lengths = [
            len(
                tokenizer(
                    prompt,
                    add_special_tokens=True,
                    truncation=False,
                )["input_ids"]
            )
            for prompt in prompts
        ]
        counts[field] = max(lengths)
        if counts[field] > max_input_tokens:
            violations.append(
                {
                    "field": field,
                    "tokens": counts[field],
                    "limit": max_input_tokens,
                }
            )
    if violations:
        raise PromptTooLong(violations)
    return counts


def _tokenizer_only() -> tuple[Any, dict[str, Any]]:
    """Load only the pinned tokenizer for the corpus-wide preflight gate."""

    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Tokenizer preflight requires the FLAN extraction environment"
        ) from exc
    snapshot_manifest = xl_runner.validate_snapshot_manifest()
    snapshot = Path(snapshot_manifest["snapshot_path"])
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True
    )
    tokenizer_files = {}
    for relative in (
        "tokenizer_config.json",
        "spiece.model",
        "special_tokens_map.json",
    ):
        path = snapshot / relative
        if path.is_file():
            tokenizer_files[relative] = sha256_file(path)
    return tokenizer, {
        "snapshot_path": str(snapshot),
        "tokenizer_files_sha256": tokenizer_files,
    }


def _token_lengths(tokenizer: Any, prompts: Sequence[str]) -> list[int]:
    encoded = tokenizer(
        list(prompts),
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )
    values = encoded["input_ids"]
    return [len(value) for value in values]


def preflight_corpus(
    *,
    input_path: Path,
    output_path: Path,
    prompt_batch_size: int = 512,
) -> dict[str, Any]:
    """Stream and tokenize every actual W17 prompt without loading the model."""

    if prompt_batch_size < 9:
        raise ValueError("prompt_batch_size must be at least 9")
    if output_path.exists():
        raise FileExistsError(output_path)
    startup_input_sha = sha256_file(input_path)
    startup_runtime = runtime_contract()
    startup_runtime_sha = sha256_json(startup_runtime)
    tokenizer, tokenizer_record = _tokenizer_only()
    schema = _active_schema()
    maxima = {field: 0 for field in W17_FIELDS}
    maximum_records: dict[str, dict[str, Any] | None] = {
        field: None for field in W17_FIELDS
    }
    violation_counts = {field: 0 for field in W17_FIELDS}
    violations_path = output_path.with_suffix(".violations.jsonl")
    violations_temporary = violations_path.with_suffix(
        violations_path.suffix + ".tmp"
    )
    violations_temporary.parent.mkdir(parents=True, exist_ok=True)
    prompt_buffer: list[str] = []
    metadata_buffer: list[tuple[str, str, int, tuple[str, str, str] | None]] = []
    token_length_cache: dict[tuple[str, str, str], tuple[int, ...]] = {}
    tokenized_prompt_count = 0
    assignment_ids: set[str] = set()
    ordered_ids = hashlib.sha256()
    assignment_count = 0
    headline_only_count = 0

    def consume_length(
        handle: Any,
        *,
        assignment_id: str,
        field: str,
        variant: int,
        tokens: int,
    ) -> None:
        if tokens > maxima[field]:
            maxima[field] = tokens
            maximum_records[field] = {
                "assignment_id": assignment_id,
                "variant": variant,
                "tokens": tokens,
            }
        if tokens > MAX_INPUT_TOKENS:
            violation_counts[field] += 1
            handle.write(
                canonical_json(
                    {
                        "assignment_id": assignment_id,
                        "field": field,
                        "variant": variant,
                        "tokens": tokens,
                        "limit": MAX_INPUT_TOKENS,
                    }
                )
                + "\n"
            )

    def flush(handle: Any) -> None:
        nonlocal tokenized_prompt_count
        if not prompt_buffer:
            return
        lengths = _token_lengths(tokenizer, prompt_buffer)
        tokenized_prompt_count += len(lengths)
        if len(lengths) != len(metadata_buffer):
            raise AssertionError("Tokenizer output length differs")
        grouped_cache_lengths: dict[
            tuple[str, str, str], dict[int, int]
        ] = {}
        for tokens, (assignment_id, field, variant, cache_key) in zip(
            lengths, metadata_buffer
        ):
            consume_length(
                handle,
                assignment_id=assignment_id,
                field=field,
                variant=variant,
                tokens=tokens,
            )
            if cache_key is not None:
                prior = grouped_cache_lengths.setdefault(cache_key, {}).get(
                    variant
                )
                if prior is not None and prior != tokens:
                    raise AssertionError(
                        "Byte-identical W17 prompts produced different "
                        "token lengths"
                    )
                grouped_cache_lengths[cache_key][variant] = tokens
        for cache_key, by_variant in grouped_cache_lengths.items():
            if set(by_variant) != {0, 1, 2}:
                raise AssertionError("W17 cache entry lacks a prompt variant")
            value = tuple(by_variant[index] for index in range(3))
            prior = token_length_cache.get(cache_key)
            if prior is not None and prior != value:
                raise AssertionError("W17 tokenizer cache is inconsistent")
            token_length_cache[cache_key] = value
        prompt_buffer.clear()
        metadata_buffer.clear()

    try:
        with violations_temporary.open(
            "w", encoding="utf-8", newline="\n"
        ) as violation_handle:
            for raw in iter_records(input_path):
                assignment = normalize_assignment(raw)
                assignment_id = assignment["assignment_id"]
                if assignment_id in assignment_ids:
                    raise ValueError(
                        f"Duplicate assignment_id {assignment_id!r}"
                    )
                assignment_ids.add(assignment_id)
                ordered_ids.update(assignment_id.encode("utf-8"))
                ordered_ids.update(b"\n")
                assignment_count += 1
                headline_only_count += int(
                    not assignment["description_available"]
                )
                record = extractor_record(assignment)
                for field in W17_FIELDS:
                    cache_key = (
                        (
                            assignment["article_id"],
                            assignment["text_sha256"],
                            field,
                        )
                        if field in {"event_family", "information_status"}
                        else None
                    )
                    cached_lengths = (
                        token_length_cache.get(cache_key)
                        if cache_key is not None
                        else None
                    )
                    if cached_lengths is not None:
                        for variant, tokens in enumerate(cached_lengths):
                            consume_length(
                                violation_handle,
                                assignment_id=assignment_id,
                                field=field,
                                variant=variant,
                                tokens=tokens,
                            )
                        continue
                    prompts = coarse_runner.prompt_variants_for_preflight(
                        record, schema, field, "zero_shot"
                    )
                    for variant, prompt in enumerate(prompts):
                        prompt_buffer.append(prompt)
                        metadata_buffer.append(
                            (assignment_id, field, variant, cache_key)
                        )
                    if len(prompt_buffer) >= prompt_batch_size:
                        flush(violation_handle)
            flush(violation_handle)
        os.replace(violations_temporary, violations_path)
    finally:
        violations_temporary.unlink(missing_ok=True)
    if assignment_count == 0:
        raise ValueError("W17 preflight input is empty")
    if sha256_file(input_path) != startup_input_sha:
        raise ValueError("W17 assignment input changed during preflight")
    if runtime_contract() != startup_runtime:
        raise ValueError("W17 runtime contract changed during preflight")
    violation_count = sum(violation_counts.values())
    manifest = {
        "manifest_version": "flan-w17-token-preflight-v1",
        "status": "passed" if violation_count == 0 else "failed",
        "input_path": str(input_path.resolve()),
        "input_sha256": startup_input_sha,
        "assignment_count": assignment_count,
        "ordered_assignment_ids_sha256": ordered_ids.hexdigest(),
        "headline_only_assignment_count": headline_only_count,
        "prompt_count": assignment_count * 3 * len(W17_FIELDS),
        "tokenized_prompt_count": tokenized_prompt_count,
        "target_invariant_prompt_cache_entries": len(token_length_cache),
        "maximum_input_tokens": MAX_INPUT_TOKENS,
        "maximum_observed_tokens_by_field": maxima,
        "maximum_records_by_field": maximum_records,
        "violation_count_by_field": violation_counts,
        "violation_count": violation_count,
        "violations": {
            "path": str(violations_path.resolve()),
            "sha256": sha256_file(violations_path),
            "rows": violation_count,
        },
        "runtime_contract": startup_runtime,
        "runtime_contract_sha256": startup_runtime_sha,
        "tokenizer": tokenizer_record,
        "silent_truncation_allowed": False,
        "model_loaded": False,
    }
    _atomic_text(
        output_path,
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    return manifest


def validate_preflight_manifest(
    path: Path,
    input_path: Path,
    *,
    require_current_runtime: bool = True,
) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("manifest_version") != "flan-w17-token-preflight-v1":
        raise ValueError("Unsupported W17 preflight manifest")
    if value.get("status") != "passed" or value.get("violation_count") != 0:
        raise ValueError("W17 tokenizer preflight did not pass")
    if value.get("input_sha256") != sha256_file(input_path):
        raise ValueError("W17 input differs from tokenizer preflight")
    if (
        require_current_runtime
        and value.get("runtime_contract_sha256") != runtime_contract_sha256()
    ):
        raise ValueError("W17 runtime differs from tokenizer preflight")
    return value


class PromptTooLong(RuntimeError):
    def __init__(self, violations: Sequence[Mapping[str, Any]]) -> None:
        self.violations = [dict(value) for value in violations]
        super().__init__(
            "Refusing to truncate W17 prompts: "
            + canonical_json(self.violations)
        )


_SCHEMA_CACHE: dict[str, Any] | None = None


def _active_schema() -> dict[str, Any]:
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is None:
        _, schema_path, _ = active_runner.load_active_contract()
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        coarse_runner.validate_schema(schema)
        for field in W17_FIELDS:
            labels = tuple(schema["closed_label_fields"][field])
            expected = PREDICTIVE_LABELS[field] + ABSTENTION_LABELS[field]
            if labels != expected:
                raise ValueError(f"Active schema labels differ for {field}")
        _SCHEMA_CACHE = schema
    return copy.deepcopy(_SCHEMA_CACHE)


def _finite_score_map(
    field: str,
    scores: Any,
    *,
    context: str,
) -> dict[str, float]:
    expected = set(PREDICTIVE_LABELS[field] + ABSTENTION_LABELS[field])
    if not isinstance(scores, Mapping) or set(scores) != expected:
        raise ValueError(f"{context} keys differ from the {field} ontology")
    output: dict[str, float] = {}
    for label, value in scores.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{context} contains a nonnumeric score")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{context} contains a nonfinite score")
        output[str(label)] = numeric
    return output


def adjusted_margin(field: str, scores: Mapping[str, Any]) -> float:
    checked = _finite_score_map(field, scores, context="adjusted score map")
    values = sorted(checked.values(), reverse=True)
    if len(values) < 2:
        raise ValueError("Adjusted score map must contain at least two labels")
    margin = values[0] - values[1]
    if not math.isfinite(margin):
        raise ValueError("Adjusted score margin is nonfinite")
    return margin


def apply_acceptance(
    *,
    field: str,
    label: str,
    adjusted_scores: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    validate_acceptance_config(config)
    valid_labels = set(PREDICTIVE_LABELS[field] + ABSTENTION_LABELS[field])
    if label not in valid_labels:
        raise ValueError(f"Invalid calibrated label for {field}: {label!r}")
    checked_scores = _finite_score_map(
        field,
        adjusted_scores,
        context="adjusted score map",
    )
    field_rule = config["fields"][field]
    if label in field_rule["abstention_labels"]:
        return {
            "accepted": False,
            "acceptance_reason": "unclear",
            "quality_weight": None,
            "adjusted_margin": adjusted_margin(field, checked_scores),
            "enabled_by_human_calibration": False,
        }
    class_rule = field_rule["classes"].get(label)
    if class_rule is None:
        return {
            "accepted": False,
            "acceptance_reason": "unvalidated_class",
            "quality_weight": None,
            "adjusted_margin": adjusted_margin(field, checked_scores),
            "enabled_by_human_calibration": False,
        }
    margin = adjusted_margin(field, checked_scores)
    if not class_rule["enabled_for_exploratory_fit"]:
        reason = "unvalidated_class"
        accepted = False
    elif margin < float(class_rule["minimum_adjusted_margin"]):
        reason = "low_margin"
        accepted = False
    else:
        reason = "accepted"
        accepted = True
    return {
        "accepted": accepted,
        "acceptance_reason": reason,
        "quality_weight": float(config["quality_weight"]) if accepted else None,
        "adjusted_margin": margin,
        "enabled_by_human_calibration": bool(
            class_rule["enabled_by_human_calibration"]
        ),
    }


class W17Engine(Protocol):
    def predict(self, assignment: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return successful terminal fields for one normalized assignment."""


class FlanT5XlW17Engine:
    """Pinned active FLAN-T5-XL scorer for three v2 W17 fields."""

    def __init__(
        self,
        acceptance: Mapping[str, Any],
        *,
        max_input_tokens: int = MAX_INPUT_TOKENS,
    ) -> None:
        validate_acceptance_config(acceptance)
        self.acceptance = copy.deepcopy(dict(acceptance))
        self.max_input_tokens = max_input_tokens
        contract, schema_path, calibration_path = active_runner.load_active_contract()
        if contract["extractor_id"] != EXTRACTOR_ID:
            raise ValueError("Active extractor ID changed")
        self.schema = json.loads(schema_path.read_text(encoding="utf-8"))
        coarse_runner.validate_schema(self.schema)
        self.calibration = json.loads(
            calibration_path.read_text(encoding="utf-8")
        )
        active_calibration.validate_apply_config(self.calibration)

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            import torch
            import transformers
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed
        except ImportError as exc:
            raise RuntimeError(
                "Install requirements-flan-t5-xl.txt and a CUDA PyTorch build"
            ) from exc
        manifest = xl_runner.validate_snapshot_manifest()
        snapshot = Path(manifest["snapshot_path"])
        self.snapshot_hashes = xl_runner.sharded_snapshot_file_hashes(snapshot)
        if not torch.cuda.is_available():
            raise RuntimeError("Pinned active W17 extraction requires CUDA")
        flan_base.configure_determinism(torch, set_seed)
        self.torch = torch
        self.transformers_version = transformers.__version__
        self.device = "cuda"
        self.dtype = torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(
            snapshot, local_files_only=True
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            snapshot,
            local_files_only=True,
            dtype=self.dtype,
            use_safetensors=True,
        )
        self.model.to(self.device)
        self.model.eval()
        self._target_invariant_cache: dict[
            tuple[str, str], dict[str, Any]
        ] = {}
        self._cache_hits = Counter()
        self._cache_misses = Counter()

    def diagnostics(self) -> dict[str, Any]:
        return {
            "target_invariant_fields": [
                "event_family",
                "information_status",
            ],
            "cache_hits_by_field": dict(sorted(self._cache_hits.items())),
            "cache_misses_by_field": dict(sorted(self._cache_misses.items())),
            "cache_entries": len(self._target_invariant_cache),
            "cache_key": (
                "field plus SHA-256 of the exact three prompt variants; "
                "reuse occurs only when extractor-visible prompts are "
                "byte-identical"
            ),
        }

    def predict(self, assignment: Mapping[str, Any]) -> Mapping[str, Any]:
        token_counts = prompt_token_counts(
            assignment, self.tokenizer, self.max_input_tokens
        )
        state = {
            "record": extractor_record(assignment),
            "labels": {field: None for field in W17_FIELDS},
        }
        fields: dict[str, Any] = {}
        original_scorer = coarse_runner.base.score_closed_label_batch
        coarse_runner.base.score_closed_label_batch = (
            xl_runner.low_memory_score_closed_label_batch
        )
        try:
            for field in W17_FIELDS:
                cache_key: tuple[str, str] | None = None
                if field in {"event_family", "information_status"}:
                    prompts = coarse_runner.prompt_variants_for_preflight(
                        state["record"],
                        self.schema,
                        field,
                        "zero_shot",
                    )
                    cache_key = (field, sha256_json(prompts))
                if (
                    cache_key is not None
                    and cache_key in self._target_invariant_cache
                ):
                    result = copy.deepcopy(
                        self._target_invariant_cache[cache_key]
                    )
                    self._cache_hits[field] += 1
                else:
                    result = coarse_runner.classify_active_batch(
                        field=field,
                        active_states=[state],
                        schema=self.schema,
                        decoding="order_averaged_letter_score",
                        tokenizer=self.tokenizer,
                        model=self.model,
                        torch=self.torch,
                        device=self.device,
                        max_input_tokens=self.max_input_tokens,
                        prompt_profile="zero_shot",
                    )[0]
                    if cache_key is not None:
                        self._target_invariant_cache[cache_key] = copy.deepcopy(
                            result
                        )
                        self._cache_misses[field] += 1
                raw_label = result["value"]
                rule = self.calibration["field_rules"][field]
                calibrated_label, adjusted_scores = (
                    active_calibration.calibrated_choice(
                        result,
                        source=rule["source"],
                        tau=float(rule["tau"]),
                        priors={
                            name: float(value)
                            for name, value in rule["priors"].items()
                        },
                        schema_order=list(
                            self.schema["closed_label_fields"][field]
                        ),
                    )
                )
                acceptance = apply_acceptance(
                    field=field,
                    label=calibrated_label,
                    adjusted_scores=adjusted_scores,
                    config=self.acceptance,
                )
                state["labels"][field] = calibrated_label
                fields[field] = {
                    "structurally_applicable": True,
                    "raw_label": raw_label,
                    "calibrated_label": calibrated_label,
                    "canonical_candidate_mean_log_probabilities": result[
                        "canonical_candidate_mean_log_probabilities"
                    ],
                    "reversed_candidate_mean_log_probabilities": result[
                        "reversed_candidate_mean_log_probabilities"
                    ],
                    "order_averaged_mean_log_probabilities": result[
                        "order_averaged_mean_log_probabilities"
                    ],
                    "raw_margin": float(result["top1_top2_margin"]),
                    "adjusted_scores": adjusted_scores,
                    "prompt_sha256": result["prompt_sha256"],
                    "reversed_prompt_sha256": result[
                        "reversed_prompt_sha256"
                    ],
                    "input_tokens_max": token_counts[field],
                    "input_truncated": False,
                    **acceptance,
                }
        finally:
            coarse_runner.base.score_closed_label_batch = original_scorer
        return {
            "terminal_state": "complete",
            "fields": fields,
            "no_input_truncation": True,
        }


def _terminal_base(
    assignment: Mapping[str, Any],
    acceptance_sha256: str,
    *,
    shard_index: int,
    shard_count: int,
    runtime_sha256: str,
    preflight_manifest_sha256: str,
) -> dict[str, Any]:
    target = assignment["target"]
    return {
        "record_version": "flan-w17-terminal-v2",
        "assignment_id": assignment["assignment_id"],
        "assignment_input_sha256": assignment["assignment_input_sha256"],
        "article_id": assignment["article_id"],
        "forecast_date": assignment["forecast_date"],
        "sector": target["sector"],
        "stock": target["ticker"],
        "benchmark": target["benchmark"],
        "role": assignment["role"],
        "source_profile": assignment["source_profile"],
        "headline_sha256": assignment["headline_sha256"],
        "description_sha256": assignment["description_sha256"],
        "description_available": assignment["description_available"],
        "text_sha256": assignment["text_sha256"],
        "extractor_id": EXTRACTOR_ID,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "contract_id": CONTRACT_ID,
        "acceptance_config_sha256": acceptance_sha256,
        "runtime_contract_sha256": runtime_sha256,
        "preflight_manifest_sha256": preflight_manifest_sha256,
        "shard_index": shard_index,
        "shard_count": shard_count,
    }


def terminal_prediction(
    assignment: Mapping[str, Any],
    engine: W17Engine,
    acceptance: Mapping[str, Any],
    *,
    shard_index: int,
    shard_count: int,
    preflight_manifest_sha256: str,
    runtime_sha256: str | None = None,
) -> dict[str, Any]:
    acceptance_sha = sha256_json(acceptance)
    base = _terminal_base(
        assignment,
        acceptance_sha,
        shard_index=shard_index,
        shard_count=shard_count,
        runtime_sha256=runtime_sha256 or runtime_contract_sha256(),
        preflight_manifest_sha256=preflight_manifest_sha256,
    )
    try:
        result = dict(engine.predict(assignment))
        if result.get("terminal_state") != "complete":
            raise ValueError("Engine did not return a successful terminal state")
        fields = result.get("fields")
        if (
            not isinstance(fields, Mapping)
            or len(fields) != len(W17_FIELDS)
            or set(fields) != set(W17_FIELDS)
        ):
            raise ValueError("Engine fields differ from W17")
        if any(
            field_record.get("input_truncated") is not False
            or field_record.get("structurally_applicable") is not True
            for field_record in fields.values()
        ):
            raise ValueError("Engine violated applicability/truncation contract")
        terminal = {
            **base,
            **result,
            "completed_at_utc": datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
        }
        validate_terminal_record(terminal)
        return terminal
    except PromptTooLong as exc:
        return {
            **base,
            "terminal_state": "prompt_too_long",
            "failure_reason": "truncated",
            "prompt_violations": exc.violations,
            "fields": {
                field: {
                    "structurally_applicable": True,
                    "accepted": False,
                    "acceptance_reason": "truncated",
                    "input_truncated": False,
                }
                for field in W17_FIELDS
            },
            "no_input_truncation": True,
        }
    except Exception as exc:
        return {
            **base,
            "terminal_state": "inference_failed",
            "failure_reason": "invalid",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "fields": {
                field: {
                    "structurally_applicable": True,
                    "accepted": False,
                    "acceptance_reason": "invalid",
                    "input_truncated": False,
                }
                for field in W17_FIELDS
            },
            "no_input_truncation": True,
        }


def validate_terminal_record(record: Mapping[str, Any]) -> None:
    if record.get("record_version") != "flan-w17-terminal-v2":
        raise ValueError("Unsupported W17 terminal record version")
    state = record.get("terminal_state")
    if state not in ALL_TERMINAL_STATES:
        raise ValueError(f"Unknown W17 terminal state {state!r}")
    if not isinstance(record.get("assignment_id"), str):
        raise ValueError("Terminal record has no assignment_id")
    if record.get("extractor_id") != EXTRACTOR_ID:
        raise ValueError("Terminal record has the wrong extractor")
    if record.get("contract_id") != CONTRACT_ID:
        raise ValueError("Terminal record has the wrong feature contract")
    if record.get("model_id") != MODEL_ID:
        raise ValueError("Terminal record has the wrong model")
    if record.get("model_revision") != MODEL_REVISION:
        raise ValueError("Terminal record has the wrong model revision")
    for name in (
        "assignment_input_sha256",
        "headline_sha256",
        "description_sha256",
        "text_sha256",
        "acceptance_config_sha256",
        "runtime_contract_sha256",
        "preflight_manifest_sha256",
    ):
        value = record.get(name)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"Terminal record has an invalid {name}")
    if record.get("acceptance_config_sha256") != sha256_json(
        default_acceptance_config()
    ):
        raise ValueError("Terminal record has the wrong acceptance lock")
    if not isinstance(record.get("description_available"), bool):
        raise ValueError("Terminal record lacks description availability")
    shard_index = record.get("shard_index")
    shard_count = record.get("shard_count")
    if (
        not isinstance(shard_index, int)
        or not isinstance(shard_count, int)
        or shard_count < 1
        or not 0 <= shard_index < shard_count
    ):
        raise ValueError("Terminal record has invalid shard metadata")
    fields = record.get("fields")
    if (
        not isinstance(fields, Mapping)
        or len(fields) != len(W17_FIELDS)
        or set(fields) != set(W17_FIELDS)
    ):
        raise ValueError("Terminal record fields differ from W17")
    if state == "complete":
        if record.get("no_input_truncation") is not True:
            raise ValueError("A successful terminal record permits truncation")
        for field in W17_FIELDS:
            item = fields[field]
            if not isinstance(item, Mapping):
                raise TypeError("A successful W17 field is not an object")
            if item.get("structurally_applicable") is not True:
                raise ValueError("A W17 field is not structurally applicable")
            if item.get("input_truncated") is not False:
                raise ValueError("A successful W17 field was truncated")
            if not isinstance(item.get("accepted"), bool):
                raise ValueError("A successful W17 field lacks acceptance")
            valid_labels = set(
                PREDICTIVE_LABELS[field] + ABSTENTION_LABELS[field]
            )
            if item.get("raw_label") not in valid_labels:
                raise ValueError(f"Invalid raw label for {field}")
            if item.get("calibrated_label") not in valid_labels:
                raise ValueError(f"Invalid calibrated label for {field}")
            score_maps: dict[str, dict[str, float]] = {}
            for name in (
                "canonical_candidate_mean_log_probabilities",
                "reversed_candidate_mean_log_probabilities",
                "order_averaged_mean_log_probabilities",
                "adjusted_scores",
            ):
                score_maps[name] = _finite_score_map(
                    field,
                    item.get(name),
                    context=name,
                )
            labels_in_order = list(
                PREDICTIVE_LABELS[field] + ABSTENTION_LABELS[field]
            )
            canonical_scores = score_maps[
                "canonical_candidate_mean_log_probabilities"
            ]
            reversed_scores = score_maps[
                "reversed_candidate_mean_log_probabilities"
            ]
            averaged_scores = score_maps[
                "order_averaged_mean_log_probabilities"
            ]
            for label in labels_in_order:
                expected_average = (
                    canonical_scores[label] + reversed_scores[label]
                ) / 2.0
                if not math.isclose(
                    averaged_scores[label],
                    expected_average,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        f"Inconsistent order-averaged score for "
                        f"{field}/{label}"
                    )
            expected_raw_label = max(
                labels_in_order,
                key=lambda label: averaged_scores[label],
            )
            if item.get("raw_label") != expected_raw_label:
                raise ValueError(f"Raw label is not the score argmax for {field}")
            raw_margin = item.get("raw_margin")
            ordered_raw_scores = sorted(
                averaged_scores.values(),
                reverse=True,
            )
            expected_raw_margin = (
                ordered_raw_scores[0] - ordered_raw_scores[1]
            )
            if (
                isinstance(raw_margin, bool)
                or not isinstance(raw_margin, (int, float))
                or not math.isfinite(float(raw_margin))
                or float(raw_margin) < 0
                or not math.isclose(
                    float(raw_margin),
                    expected_raw_margin,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(f"Invalid raw margin for {field}")
            adjusted_scores = score_maps["adjusted_scores"]
            expected_calibrated_label = max(
                labels_in_order,
                key=lambda label: adjusted_scores[label],
            )
            if item.get("calibrated_label") != expected_calibrated_label:
                raise ValueError(
                    f"Calibrated label is not the adjusted-score argmax for "
                    f"{field}"
                )
            expected_acceptance = apply_acceptance(
                field=field,
                label=str(item["calibrated_label"]),
                adjusted_scores=adjusted_scores,
                config=default_acceptance_config(),
            )
            for name in (
                "accepted",
                "acceptance_reason",
                "quality_weight",
                "enabled_by_human_calibration",
            ):
                if item.get(name) != expected_acceptance[name]:
                    raise ValueError(
                        f"Inconsistent {name} in successful {field} output"
                    )
            margin = item.get("adjusted_margin")
            if (
                isinstance(margin, bool)
                or not isinstance(margin, (int, float))
                or not math.isfinite(float(margin))
                or not math.isclose(
                    float(margin),
                    float(expected_acceptance["adjusted_margin"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(f"Invalid adjusted margin for {field}")
            for name in ("prompt_sha256", "reversed_prompt_sha256"):
                value = item.get(name)
                if (
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in value
                    )
                ):
                    raise ValueError(f"Invalid {name} for {field}")
            input_tokens = item.get("input_tokens_max")
            if (
                isinstance(input_tokens, bool)
                or not isinstance(input_tokens, int)
                or not 0 < input_tokens <= MAX_INPUT_TOKENS
            ):
                raise ValueError(f"Invalid input token count for {field}")
    else:
        if record.get("no_input_truncation") is not True:
            raise ValueError("A failed terminal record permits truncation")
        for item in fields.values():
            if (
                not isinstance(item, Mapping)
                or item.get("structurally_applicable") is not True
                or item.get("accepted") is not False
                or item.get("input_truncated") is not False
            ):
                raise ValueError("A failed terminal field is not fail-closed")


def load_terminal_records(paths: Sequence[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(read_records(path))
    for record in records:
        validate_terminal_record(record)
    ids = [record["assignment_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Prediction shards contain duplicate assignment IDs")
    return records


def recover_append_tail(path: Path) -> dict[str, Any]:
    """Repair only an unterminated final JSONL record after interruption."""

    if not path.exists() or path.stat().st_size == 0:
        return {"action": "not_needed", "bytes_removed": 0}
    with path.open("r+b") as handle:
        handle.seek(-1, os.SEEK_END)
        if handle.read(1) == b"\n":
            return {"action": "not_needed", "bytes_removed": 0}
        size = handle.tell()
        position = size
        previous_newline = -1
        while position > 0 and previous_newline < 0:
            block_start = max(0, position - 65_536)
            handle.seek(block_start)
            block = handle.read(position - block_start)
            offset = block.rfind(b"\n")
            if offset >= 0:
                previous_newline = block_start + offset
                break
            position = block_start
        tail_start = previous_newline + 1
        handle.seek(tail_start)
        tail = handle.read()
        try:
            value = json.loads(tail)
        except (UnicodeDecodeError, json.JSONDecodeError):
            handle.seek(tail_start)
            handle.truncate()
            handle.flush()
            os.fsync(handle.fileno())
            return {
                "action": "truncated_incomplete_tail",
                "bytes_removed": len(tail),
            }
        if not isinstance(value, dict):
            raise TypeError("Final unterminated JSONL value is not an object")
        handle.seek(0, os.SEEK_END)
        handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())
        return {"action": "terminated_valid_tail", "bytes_removed": 0}


def _append_terminal(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def inference_manifest(
    *,
    input_path: Path,
    output_path: Path,
    assignments: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    acceptance: Mapping[str, Any],
    shard_index: int,
    shard_count: int,
    preflight_manifest_path: Path,
    preflight_manifest_sha256_value: str,
    preflight_manifest: Mapping[str, Any],
    input_sha256: str,
    runtime_contract_value: Mapping[str, Any],
    runtime_contract_sha256_value: str,
    selection_limit: int | None,
    tail_recovery: Mapping[str, Any],
) -> dict[str, Any]:
    states = Counter(record["terminal_state"] for record in records)
    all_terminal = len(records) == len(assignments)
    failures = int(
        sum(states.get(name, 0) for name in FAILURE_TERMINAL_STATES)
    )
    if not all_terminal:
        status = "in_progress"
    elif failures:
        status = "complete_with_failures"
    elif selection_limit is not None:
        status = "smoke_complete"
    else:
        status = "complete"
    return {
        "manifest_version": "flan-w17-inference-manifest-v2",
        "pipeline_version": PIPELINE_VERSION,
        "status": status,
        "input_path": str(input_path.resolve()),
        "input_sha256": input_sha256,
        "output_path": str(output_path.resolve()),
        "output_sha256": sha256_file(output_path) if output_path.exists() else None,
        "extractor_id": EXTRACTOR_ID,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "active_pointer_path": str(active_runner.ACTIVE_POINTER.resolve()),
        "active_pointer_sha256": sha256_file(active_runner.ACTIVE_POINTER),
        "preflight_manifest_path": str(preflight_manifest_path.resolve()),
        "preflight_manifest_sha256": preflight_manifest_sha256_value,
        "preflight_contract": {
            "assignment_count": preflight_manifest["assignment_count"],
            "prompt_count": preflight_manifest["prompt_count"],
            "tokenized_prompt_count": preflight_manifest.get(
                "tokenized_prompt_count"
            ),
            "maximum_observed_tokens_by_field": preflight_manifest[
                "maximum_observed_tokens_by_field"
            ],
            "violation_count": preflight_manifest["violation_count"],
            "input_sha256": preflight_manifest["input_sha256"],
            "runtime_contract_sha256": preflight_manifest[
                "runtime_contract_sha256"
            ],
        },
        "selection_limit": selection_limit,
        "full_corpus_shard": selection_limit is None,
        "append_tail_recovery": dict(tail_recovery),
        "acceptance_config": acceptance,
        "acceptance_config_sha256": sha256_json(acceptance),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "sharding_key": "article_id_sha256_first8_modulo",
        "article_consistent_sharding": True,
        "selected_assignment_count": len(assignments),
        "selected_assignment_ids_sha256": sha256_text(
            "\n".join(record["assignment_id"] for record in assignments)
        ),
        "terminal_record_count": len(records),
        "terminal_state_counts": dict(sorted(states.items())),
        "all_selected_assignments_terminal": all_terminal,
        "successful_assignment_count": int(states.get("complete", 0)),
        "failure_assignment_count": failures,
        "legacy_relevance_gate_used": False,
        "w17_fields": list(W17_FIELDS),
        "maximum_input_tokens": MAX_INPUT_TOKENS,
        "silent_truncation_allowed": False,
        "runtime_contract": dict(runtime_contract_value),
        "runtime_contract_sha256": runtime_contract_sha256_value,
    }


def run_inference(
    *,
    input_path: Path,
    output_path: Path,
    acceptance: Mapping[str, Any],
    engine: W17Engine,
    shard_index: int = 0,
    shard_count: int = 1,
    retry_failed: bool = False,
    limit: int | None = None,
    preflight_manifest_path: Path,
    max_consecutive_failures: int = SYSTEMIC_FAILURE_ABORT_THRESHOLD,
) -> dict[str, Any]:
    acceptance_snapshot = copy.deepcopy(dict(acceptance))
    validate_acceptance_config(acceptance_snapshot)
    if "".join(output_path.suffixes).lower() != ".jsonl":
        raise ValueError(
            "Append-only W17 inference output must be uncompressed .jsonl"
        )
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must lie in [0, shard_count)")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if max_consecutive_failures < 1:
        raise ValueError("max_consecutive_failures must be positive")
    startup_input_sha = sha256_file(input_path)
    startup_runtime = runtime_contract()
    runtime_sha = sha256_json(startup_runtime)
    preflight = validate_preflight_manifest(
        preflight_manifest_path,
        input_path,
        require_current_runtime=True,
    )
    preflight_sha = sha256_file(preflight_manifest_path)
    if preflight["runtime_contract_sha256"] != runtime_sha:
        raise ValueError("W17 preflight runtime differs from inference startup")
    assignments: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for raw in iter_records(input_path):
        normalized = normalize_assignment(raw)
        if assignment_shard(normalized, shard_count) != shard_index:
            continue
        if normalized["assignment_id"] in selected_ids:
            raise ValueError("Assignment input contains a duplicate selected ID")
        selected_ids.add(normalized["assignment_id"])
        assignments.append(normalized)
        if limit is not None and len(assignments) >= limit:
            break
    if sha256_file(input_path) != startup_input_sha:
        raise ValueError("W17 assignment input changed while selecting a shard")
    if limit is None and len(assignments) > MAX_ASSIGNMENTS_PER_INFERENCE_SHARD:
        raise ValueError(
            "Selected W17 shard exceeds "
            f"{MAX_ASSIGNMENTS_PER_INFERENCE_SHARD:,} assignments; increase "
            "--shard-count before inference"
        )
    by_id = {record["assignment_id"]: record for record in assignments}

    tail_recovery = recover_append_tail(output_path)
    existing = read_records(output_path) if output_path.exists() else []
    retained: list[dict[str, Any]] = []
    for record in existing:
        validate_terminal_record(record)
        assignment_id = record["assignment_id"]
        assignment = by_id.get(assignment_id)
        if assignment is None:
            raise ValueError("Existing prediction is outside the selected shard")
        if (
            record.get("assignment_input_sha256")
            != assignment["assignment_input_sha256"]
        ):
            raise ValueError("Existing prediction input hash changed")
        if record.get("acceptance_config_sha256") != sha256_json(
            acceptance_snapshot
        ):
            raise ValueError("Existing prediction acceptance config changed")
        if record.get("runtime_contract_sha256") != runtime_sha:
            raise ValueError("Existing prediction runtime contract changed")
        if record.get("preflight_manifest_sha256") != preflight_sha:
            raise ValueError("Existing prediction preflight contract changed")
        if (
            record.get("shard_index") != shard_index
            or record.get("shard_count") != shard_count
        ):
            raise ValueError("Existing prediction shard contract changed")
        if retry_failed and record["terminal_state"] in FAILURE_TERMINAL_STATES:
            continue
        retained.append(record)
    retained_ids = [record["assignment_id"] for record in retained]
    if len(retained_ids) != len(set(retained_ids)):
        raise ValueError("Existing prediction output contains duplicate IDs")
    if len(retained) != len(existing):
        write_records(output_path, retained)

    completed = set(retained_ids)
    consecutive_failures = 0
    aborted_for_systemic_failures = False
    for assignment in assignments:
        if assignment["assignment_id"] in completed:
            continue
        terminal = terminal_prediction(
            assignment,
            engine,
            acceptance_snapshot,
            shard_index=shard_index,
            shard_count=shard_count,
            preflight_manifest_sha256=preflight_sha,
            runtime_sha256=runtime_sha,
        )
        validate_terminal_record(terminal)
        _append_terminal(output_path, terminal)
        retained.append(terminal)
        if terminal["terminal_state"] == "inference_failed":
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                aborted_for_systemic_failures = True
                break
        else:
            consecutive_failures = 0

    if sha256_file(input_path) != startup_input_sha:
        raise ValueError("W17 assignment input changed during inference")
    if runtime_contract() != startup_runtime:
        raise ValueError("W17 runtime contract changed during inference")
    if sha256_file(preflight_manifest_path) != preflight_sha:
        raise ValueError("W17 preflight manifest changed during inference")
    acceptance_sha = sha256_json(acceptance_snapshot)
    for record in retained:
        assignment = by_id[record["assignment_id"]]
        if (
            record.get("runtime_contract_sha256") != runtime_sha
            or record.get("preflight_manifest_sha256") != preflight_sha
            or record.get("acceptance_config_sha256") != acceptance_sha
            or record.get("assignment_input_sha256")
            != assignment["assignment_input_sha256"]
        ):
            raise ValueError("W17 terminal provenance differs from startup")

    manifest = inference_manifest(
        input_path=input_path,
        output_path=output_path,
        assignments=assignments,
        records=retained,
        acceptance=acceptance_snapshot,
        shard_index=shard_index,
        shard_count=shard_count,
        preflight_manifest_path=preflight_manifest_path,
        preflight_manifest_sha256_value=preflight_sha,
        preflight_manifest=preflight,
        input_sha256=startup_input_sha,
        runtime_contract_value=startup_runtime,
        runtime_contract_sha256_value=runtime_sha,
        selection_limit=limit,
        tail_recovery=tail_recovery,
    )
    diagnostics = getattr(engine, "diagnostics", None)
    if callable(diagnostics):
        manifest["engine_diagnostics"] = diagnostics()
    if aborted_for_systemic_failures:
        manifest["status"] = "aborted_systemic_failures"
        manifest["consecutive_failure_abort_threshold"] = (
            max_consecutive_failures
        )
    _atomic_text(
        output_path.with_suffix(output_path.suffix + ".manifest.json"),
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    if aborted_for_systemic_failures:
        raise RuntimeError(
            "Aborted W17 shard after "
            f"{max_consecutive_failures} consecutive inference failures"
        )
    return manifest


def _key_from_assignment(record: Mapping[str, Any]) -> tuple[str, str, str, str]:
    target = record["target"]
    return (
        record["forecast_date"],
        target["sector"],
        target["ticker"],
        target["benchmark"],
    )


def _ids_hash(ids: Sequence[str]) -> str:
    return sha256_text("\n".join(sorted(ids)))


def _invalid_daily_row(
    scope: Mapping[str, Any],
    reason: str,
    observed_count: int,
) -> dict[str, Any]:
    return {
        **{name: scope[name] for name in KEY_COLUMNS},
        "source_profile": scope["source_profile"],
        "semantic_row_complete": False,
        "semantic_ineligibility_reason": reason,
        "expected_assignment_count": scope["expected_assignment_count"],
        "observed_assignment_count": observed_count,
        **{column: np.nan for column in WLLM17_COLUMNS},
    }


def _aggregate_normalized_scope(
    *,
    scope: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    prediction_by_id: Mapping[str, Mapping[str, Any]],
    acceptance_sha256: str,
    expected_runtime_contract_sha256: str | None,
    expected_preflight_manifest_sha256: str | None,
) -> dict[str, Any]:
    """Aggregate one already-normalized stock-day after outer provenance checks."""

    pandas = _require_pandas()
    expected = int(scope["expected_assignment_count"])
    if not scope["source_query_scope_complete"]:
        return _invalid_daily_row(scope, "source_incomplete", len(candidates))
    if not scope["candidate_assignment_complete"]:
        return _invalid_daily_row(
            scope,
            "assignment_scope_incomplete",
            len(candidates),
        )
    if len(candidates) != expected:
        return _invalid_daily_row(
            scope,
            "assignment_count_mismatch",
            len(candidates),
        )
    if scope["expected_assignment_ids_sha256"] != _ids_hash(
        [str(record["assignment_id"]) for record in candidates]
    ):
        return _invalid_daily_row(
            scope,
            "assignment_id_hash_mismatch",
            len(candidates),
        )
    if any(
        not record["source_query_scope_complete"]
        or not record["candidate_assignment_complete"]
        or record["source_profile"] != scope["source_profile"]
        for record in candidates
    ):
        return _invalid_daily_row(
            scope,
            "candidate_provenance_incomplete",
            len(candidates),
        )

    paired: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    failure_reason: str | None = None
    key = tuple(scope[name] for name in KEY_COLUMNS)
    for assignment in candidates:
        prediction = prediction_by_id.get(str(assignment["assignment_id"]))
        if prediction is None:
            failure_reason = "missing_prediction"
            break
        if prediction["terminal_state"] != "complete":
            failure_reason = f"terminal_{prediction['terminal_state']}"
            break
        target = assignment["target"]
        if (
            (
                prediction.get("forecast_date"),
                prediction.get("sector"),
                prediction.get("stock"),
                prediction.get("benchmark"),
            )
            != key
            or prediction.get("article_id") != assignment["article_id"]
            or prediction.get("role") != assignment["role"]
            or prediction.get("source_profile")
            != assignment["source_profile"]
            or prediction.get("assignment_input_sha256")
            != assignment["assignment_input_sha256"]
            or prediction.get("headline_sha256")
            != assignment["headline_sha256"]
            or prediction.get("description_sha256")
            != assignment["description_sha256"]
            or prediction.get("text_sha256") != assignment["text_sha256"]
            or prediction.get("description_available")
            is not assignment["description_available"]
            or target["sector"] != scope["sector"]
            or target["ticker"] != scope["stock"]
            or target["benchmark"] != scope["benchmark"]
        ):
            failure_reason = "prediction_assignment_provenance_mismatch"
            break
        if prediction.get("acceptance_config_sha256") != acceptance_sha256:
            failure_reason = "acceptance_config_hash_mismatch"
            break
        if (
            expected_runtime_contract_sha256 is not None
            and prediction.get("runtime_contract_sha256")
            != expected_runtime_contract_sha256
        ):
            failure_reason = "runtime_contract_hash_mismatch"
            break
        if (
            expected_preflight_manifest_sha256 is not None
            and prediction.get("preflight_manifest_sha256")
            != expected_preflight_manifest_sha256
        ):
            failure_reason = "preflight_manifest_hash_mismatch"
            break
        paired.append((assignment, prediction))
    if failure_reason is not None:
        return _invalid_daily_row(scope, failure_reason, len(candidates))

    features = {column: np.nan for column in WLLM17_COLUMNS}
    if not candidates:
        features["wllm_observed_no_eligible_semantic_article"] = 1.0
    else:
        features["wllm_observed_no_eligible_semantic_article"] = 0.0
        for field in W17_FIELDS:
            applicable_weight = sum(
                float(assignment["aggregation_weight"])
                for assignment, _ in paired
            )
            if not math.isfinite(applicable_weight) or applicable_weight <= 0:
                failure_reason = "applicable_weight_invalid"
                break
            accepted: list[tuple[float, str]] = []
            for assignment, prediction in paired:
                field_record = prediction["fields"][field]
                if field_record.get("structurally_applicable") is not True:
                    failure_reason = "field_applicability_violation"
                    break
                if field_record.get("accepted"):
                    label = field_record.get("calibrated_label")
                    if label not in PREDICTIVE_LABELS[field]:
                        failure_reason = "accepted_label_invalid"
                        break
                    quality = field_record.get("quality_weight")
                    if (
                        isinstance(quality, bool)
                        or not isinstance(quality, (int, float))
                        or float(quality) != 1.0
                    ):
                        failure_reason = "quality_weight_invalid"
                        break
                    accepted.append(
                        (
                            float(assignment["aggregation_weight"])
                            * float(quality),
                            str(label),
                        )
                    )
            if failure_reason is not None:
                break
            coverage_name = f"wllm_coverage_{field}"
            accepted_mass = sum(weight for weight, _ in accepted)
            features[coverage_name] = accepted_mass / applicable_weight
            if accepted_mass > 0:
                for label in PREDICTIVE_LABELS[field]:
                    features[FIELD_PREFIX[field] + label] = (
                        sum(
                            weight
                            for weight, value in accepted
                            if value == label
                        )
                        / accepted_mass
                    )
        if failure_reason is None:
            common = features["wllm_scope_share_common"]
            idiosyncratic = features["wllm_scope_share_idiosyncratic"]
            if not pandas.isna(common) and not pandas.isna(idiosyncratic):
                features["wllm_scope_common_minus_idiosyncratic"] = (
                    common - idiosyncratic
                )
    if failure_reason is not None:
        return _invalid_daily_row(scope, failure_reason, len(candidates))
    return {
        **{name: scope[name] for name in KEY_COLUMNS},
        "source_profile": scope["source_profile"],
        "semantic_row_complete": True,
        "semantic_ineligibility_reason": None,
        "expected_assignment_count": expected,
        "observed_assignment_count": len(candidates),
        **features,
    }


def aggregate_daily_w17(
    *,
    assignments: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    stock_days: Sequence[Mapping[str, Any]],
    acceptance: Mapping[str, Any],
    expected_runtime_contract_sha256: str | None = None,
    expected_preflight_manifest_sha256: str | None = None,
) -> pd.DataFrame:
    """Aggregate exact WLLM17 features and invalidate incomplete stock-days."""

    pandas = _require_pandas()
    validate_acceptance_config(acceptance)
    normalized_assignments = normalize_assignments(assignments)
    normalized_scopes = [normalize_stock_day(record) for record in stock_days]
    scope_keys = [tuple(scope[name] for name in KEY_COLUMNS) for scope in normalized_scopes]
    if len(scope_keys) != len(set(scope_keys)):
        raise ValueError("Stock-day scope contains duplicate keys")
    source_profiles = {scope["source_profile"] for scope in normalized_scopes}
    if len(source_profiles) != 1:
        raise ValueError("Aggregate one source profile at a time")

    prediction_by_id: dict[str, Mapping[str, Any]] = {}
    for prediction in predictions:
        validate_terminal_record(prediction)
        assignment_id = prediction["assignment_id"]
        if assignment_id in prediction_by_id:
            raise ValueError("Predictions contain duplicate assignment IDs")
        prediction_by_id[assignment_id] = prediction
    assignment_ids = {record["assignment_id"] for record in normalized_assignments}
    if set(prediction_by_id) - assignment_ids:
        raise ValueError("Predictions contain IDs absent from the assignment ledger")

    assignments_by_key: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for assignment in normalized_assignments:
        assignments_by_key.setdefault(_key_from_assignment(assignment), []).append(
            assignment
        )
    assignment_keys = set(assignments_by_key)
    unknown_assignment_keys = assignment_keys - set(scope_keys)
    if unknown_assignment_keys:
        raise ValueError(
            "Assignment ledger contains keys absent from stock-day scope"
        )
    acceptance_sha = sha256_json(acceptance)
    rows: list[dict[str, Any]] = []
    for scope in normalized_scopes:
        key = tuple(scope[name] for name in KEY_COLUMNS)
        candidates = sorted(
            assignments_by_key.get(key, []),
            key=lambda record: record["assignment_id"],
        )
        rows.append(
            _aggregate_normalized_scope(
                scope=scope,
                candidates=candidates,
                prediction_by_id=prediction_by_id,
                acceptance_sha256=acceptance_sha,
                expected_runtime_contract_sha256=(
                    expected_runtime_contract_sha256
                ),
                expected_preflight_manifest_sha256=(
                    expected_preflight_manifest_sha256
                ),
            )
        )
    columns = [
        *KEY_COLUMNS,
        "source_profile",
        "semantic_row_complete",
        "semantic_ineligibility_reason",
        "expected_assignment_count",
        "observed_assignment_count",
        *WLLM17_COLUMNS,
    ]
    frame = pandas.DataFrame(rows, columns=columns)
    if tuple(frame.columns[-17:]) != WLLM17_COLUMNS:
        raise AssertionError("Daily output does not end with exact WLLM17 order")
    return frame.sort_values(list(KEY_COLUMNS), kind="mergesort").reset_index(
        drop=True
    )


def validate_complete_daily_panel(frame: pd.DataFrame) -> None:
    if tuple(frame.columns[-17:]) != WLLM17_COLUMNS:
        raise ValueError("Daily panel does not contain exact ordered WLLM17")
    if frame.empty:
        raise ValueError("Daily W17 panel is empty")
    incomplete = ~frame["semantic_row_complete"].astype(bool)
    if incomplete.any():
        reasons = frame.loc[
            incomplete, "semantic_ineligibility_reason"
        ].value_counts(dropna=False).to_dict()
        raise ValueError(
            "Daily W17 panel is incomplete and cannot enter training: "
            + canonical_json(reasons)
        )
    if frame.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("Daily W17 panel contains duplicate stock-day keys")
    feature_array = frame[list(WLLM17_COLUMNS)].to_numpy(dtype=float)
    finite_or_missing = np.isfinite(feature_array) | np.isnan(feature_array)
    if not finite_or_missing.all():
        raise ValueError("Daily W17 features contain infinities")
    no_eligible = frame["wllm_observed_no_eligible_semantic_article"]
    if not no_eligible.isin([0.0, 1.0]).all():
        raise ValueError("Complete rows need a binary no-eligible flag")


def _repository_data_output_check(path: Path) -> None:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError:
        return
    if not relative.parts or relative.parts[0].casefold() != "data":
        raise ValueError(
            "Semantic model inputs/outputs inside the repository must remain "
            "under data/"
        )


def _private_work_database(
    stage: str,
    output_path: Path,
    work_root: Path | None,
) -> Path:
    root = (work_root or DEFAULT_PRIVATE_WORK_ROOT).resolve()
    digest = sha256_text(str(output_path.resolve()))[:24]
    database_path = root / f"{stage}-{digest}.sqlite"
    _repository_data_output_check(database_path)
    root.mkdir(parents=True, exist_ok=True)
    return database_path


def _file_snapshot(paths: Sequence[Path]) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for path in paths:
        resolved = path.resolve()
        key = str(resolved)
        if key in snapshot:
            continue
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        snapshot[key] = sha256_file(resolved)
    return snapshot


def _verify_file_snapshot(snapshot: Mapping[str, str]) -> None:
    for name, expected_sha256 in snapshot.items():
        path = Path(name)
        if not path.is_file() or sha256_file(path) != expected_sha256:
            raise ValueError(
                f"W17 immutable aggregation input changed: {path}"
            )


def _manifest_artifact_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} manifest path is missing")
    path = Path(value)
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def validate_authoritative_corpus_manifest(
    *,
    corpus_manifest_path: Path,
    assignments_path: Path,
    stock_days_path: Path,
) -> dict[str, Any]:
    manifest = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("manifest_version") != "semantic-corpus-manifest-v1"
        or manifest.get("status") != "complete_exploratory_retrospective"
    ):
        raise ValueError(
            "W17 aggregation requires the complete authoritative semantic "
            "corpus manifest"
        )
    sidecar = corpus_manifest_path.with_suffix(".sha256")
    if (
        not sidecar.is_file()
        or sidecar.read_text(encoding="ascii").strip()
        != sha256_file(corpus_manifest_path)
    ):
        raise ValueError("Semantic corpus manifest sidecar hash is invalid")
    generated = manifest.get("generated_files")
    audit = manifest.get("audit")
    if not isinstance(generated, Mapping) or not isinstance(audit, Mapping):
        raise ValueError("Semantic corpus manifest lacks generated-file audit")

    def validate_file(
        path: Path,
        *,
        expected_rows_key: str,
    ) -> dict[str, Any]:
        record = generated.get(path.name)
        if not isinstance(record, Mapping):
            raise ValueError(f"Corpus manifest does not bind {path.name}")
        recorded_path = _manifest_artifact_path(
            record.get("path"),
            label=path.name,
        )
        if recorded_path != path.resolve():
            raise ValueError(f"Corpus manifest path differs for {path.name}")
        if record.get("sha256") != sha256_file(path):
            raise ValueError(f"Corpus manifest hash differs for {path.name}")
        if int(record.get("rows", -1)) != int(
            audit.get(expected_rows_key, -2)
        ):
            raise ValueError(f"Corpus manifest row count differs for {path.name}")
        return dict(record)

    assignment_record = validate_file(
        assignments_path,
        expected_rows_key="assignment_count",
    )
    stock_day_record = validate_file(
        stock_days_path,
        expected_rows_key="stock_day_scope_rows",
    )
    if manifest.get("claim_flags", {}).get(
        "exploratory_construction_eligible"
    ) is not True:
        raise ValueError("Semantic corpus is not exploratory-construction eligible")
    return {
        "manifest": manifest,
        "manifest_sha256": sha256_file(corpus_manifest_path),
        "assignment_record": assignment_record,
        "stock_day_record": stock_day_record,
    }


def validate_complete_inference_manifests(
    *,
    prediction_paths: Sequence[Path],
    assignments_path: Path,
    preflight_manifest_path: Path,
    acceptance: Mapping[str, Any],
) -> dict[str, Any]:
    if not prediction_paths:
        raise ValueError("At least one W17 inference shard is required")
    preflight = validate_preflight_manifest(
        preflight_manifest_path,
        assignments_path,
        require_current_runtime=False,
    )
    preflight_sha = sha256_file(preflight_manifest_path)
    acceptance_sha = sha256_json(acceptance)
    records: list[dict[str, Any]] = []
    seen_outputs: set[Path] = set()
    for prediction_path in prediction_paths:
        resolved_output = prediction_path.resolve()
        if resolved_output in seen_outputs:
            raise ValueError("A W17 prediction shard was supplied twice")
        seen_outputs.add(resolved_output)
        manifest_path = prediction_path.with_suffix(
            prediction_path.suffix + ".manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("manifest_version")
            != "flan-w17-inference-manifest-v2"
            or manifest.get("status") != "complete"
            or manifest.get("selection_limit") is not None
            or manifest.get("full_corpus_shard") is not True
            or manifest.get("all_selected_assignments_terminal") is not True
            or int(manifest.get("failure_assignment_count", -1)) != 0
        ):
            raise ValueError(
                "Daily W17 aggregation requires complete, non-smoke, "
                "failure-free inference manifests"
            )
        if (
            _manifest_artifact_path(
                manifest.get("input_path"),
                label="W17 inference input",
            )
            != assignments_path.resolve()
            or manifest.get("input_sha256") != sha256_file(assignments_path)
            or _manifest_artifact_path(
                manifest.get("output_path"),
                label="W17 inference output",
            )
            != resolved_output
            or manifest.get("output_sha256") != sha256_file(prediction_path)
            or _manifest_artifact_path(
                manifest.get("preflight_manifest_path"),
                label="W17 preflight",
            )
            != preflight_manifest_path.resolve()
            or manifest.get("preflight_manifest_sha256") != preflight_sha
            or manifest.get("runtime_contract_sha256")
            != preflight["runtime_contract_sha256"]
            or manifest.get("acceptance_config_sha256") != acceptance_sha
            or manifest.get("sharding_key")
            != "article_id_sha256_first8_modulo"
            or manifest.get("article_consistent_sharding") is not True
        ):
            raise ValueError("W17 inference manifest provenance differs")
        if (
            int(manifest.get("terminal_record_count", -1))
            != int(manifest.get("selected_assignment_count", -2))
            or int(manifest.get("successful_assignment_count", -1))
            != int(manifest.get("selected_assignment_count", -2))
        ):
            raise ValueError("W17 inference shard counts are inconsistent")
        records.append(
            {
                "prediction_path": prediction_path,
                "manifest_path": manifest_path,
                "manifest": manifest,
                "manifest_sha256": sha256_file(manifest_path),
            }
        )
    shard_counts = {int(item["manifest"]["shard_count"]) for item in records}
    if len(shard_counts) != 1:
        raise ValueError("W17 inference shards use different shard counts")
    shard_count = next(iter(shard_counts))
    shard_indexes = [int(item["manifest"]["shard_index"]) for item in records]
    if sorted(shard_indexes) != list(range(shard_count)):
        raise ValueError("W17 inference manifests do not cover every shard")
    records.sort(key=lambda item: int(item["manifest"]["shard_index"]))
    return {
        "preflight": preflight,
        "preflight_sha256": preflight_sha,
        "runtime_contract_sha256": preflight["runtime_contract_sha256"],
        "acceptance_config_sha256": acceptance_sha,
        "shard_count": shard_count,
        "shards": records,
    }


def _compact_prediction(record: Mapping[str, Any]) -> dict[str, Any]:
    output = {
        name: record[name]
        for name in (
            "assignment_id",
            "assignment_input_sha256",
            "article_id",
            "forecast_date",
            "sector",
            "stock",
            "benchmark",
            "role",
            "source_profile",
            "headline_sha256",
            "description_sha256",
            "description_available",
            "text_sha256",
            "terminal_state",
            "acceptance_config_sha256",
            "runtime_contract_sha256",
            "preflight_manifest_sha256",
        )
    }
    output["fields"] = {
        field: {
            name: record["fields"][field][name]
            for name in (
                "structurally_applicable",
                "calibrated_label",
                "accepted",
                "quality_weight",
            )
        }
        for field in W17_FIELDS
    }
    return output


def _daily_manifest(
    *,
    corpus_manifest_path: Path,
    corpus_contract: Mapping[str, Any],
    assignments_path: Path,
    inference_contract: Mapping[str, Any],
    stock_days_path: Path,
    preflight_manifest_path: Path,
    input_snapshot: Mapping[str, str],
    implementation_sha256: str,
    output_path: Path,
    frame: pd.DataFrame,
    acceptance: Mapping[str, Any],
) -> dict[str, Any]:
    source_claims = dict(
        corpus_contract["manifest"].get("claim_flags", {})
    )
    shard_files = {
        str(item["prediction_path"].resolve()): {
            "sha256": input_snapshot[
                str(item["prediction_path"].resolve())
            ],
            "manifest_path": str(item["manifest_path"].resolve()),
            "manifest_sha256": input_snapshot[
                str(item["manifest_path"].resolve())
            ],
            "shard_index": int(item["manifest"]["shard_index"]),
            "rows": int(item["manifest"]["terminal_record_count"]),
        }
        for item in inference_contract["shards"]
    }
    return {
        "manifest_version": "flan-w17-daily-manifest-v2",
        "pipeline_version": PIPELINE_VERSION,
        "status": "complete_exploratory_silver_fit",
        "contract_id": CONTRACT_ID,
        "extractor_id": EXTRACTOR_ID,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "result_role": acceptance["result_role"],
        "primary_training_eligible": False,
        "confirmatory_eligible": False,
        "point_in_time_version_safe": bool(
            source_claims.get("point_in_time_version_safe", False)
        ),
        "effective_dated_entity_map": bool(
            source_claims.get("effective_dated_entity_map", False)
        ),
        "complete_untickered_macro_coverage": bool(
            source_claims.get("complete_untickered_macro_coverage", False)
        ),
        "legacy_relevance_gate_used": False,
        "source_files": {
            "corpus_manifest": {
                "path": str(corpus_manifest_path.resolve()),
                "sha256": input_snapshot[
                    str(corpus_manifest_path.resolve())
                ],
            },
            "assignments": {
                "path": str(assignments_path.resolve()),
                "sha256": input_snapshot[str(assignments_path.resolve())],
                "rows": int(
                    corpus_contract["assignment_record"]["rows"]
                ),
            },
            "stock_days": {
                "path": str(stock_days_path.resolve()),
                "sha256": input_snapshot[str(stock_days_path.resolve())],
                "rows": int(corpus_contract["stock_day_record"]["rows"]),
            },
            "preflight": {
                "path": str(
                    preflight_manifest_path.resolve()
                ),
                "sha256": input_snapshot[
                    str(preflight_manifest_path.resolve())
                ],
            },
            "inference_shards": shard_files,
        },
        "acceptance_config": acceptance,
        "acceptance_config_sha256": sha256_json(acceptance),
        "runtime_contract_sha256": inference_contract[
            "runtime_contract_sha256"
        ],
        "shard_count": inference_contract["shard_count"],
        "ordered_feature_list": list(WLLM17_COLUMNS),
        "feature_list_sha256": sha256_json(list(WLLM17_COLUMNS)),
        "daily_rows": len(frame),
        "dates": int(frame["forecast_date"].nunique()),
        "stocks": int(frame["stock"].nunique()),
        "all_semantic_rows_complete": True,
        "join_memory_contract": (
            "disk-backed SQLite inputs; bounded 27,510-row output"
        ),
        "output_path": str(output_path.resolve()),
        "output_sha256": sha256_file(output_path),
        "implementation_sha256": implementation_sha256,
    }


def write_complete_daily_panel(
    *,
    assignments_path: Path,
    prediction_paths: Sequence[Path],
    stock_days_path: Path,
    corpus_manifest_path: Path,
    preflight_manifest_path: Path,
    output_path: Path,
    acceptance: Mapping[str, Any],
    work_root: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Materialize W17 with a hash-bound, disk-backed, fail-closed join."""

    acceptance_snapshot = copy.deepcopy(dict(acceptance))
    validate_acceptance_config(acceptance_snapshot)
    _repository_data_output_check(output_path)
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    if not overwrite and (output_path.exists() or manifest_path.exists()):
        raise FileExistsError(output_path)
    corpus_sidecar = corpus_manifest_path.with_suffix(".sha256")
    immutable_paths = [
        corpus_manifest_path,
        corpus_sidecar,
        assignments_path,
        stock_days_path,
        preflight_manifest_path,
        Path(__file__).resolve(),
    ]
    for prediction_path in prediction_paths:
        immutable_paths.extend(
            [
                prediction_path,
                prediction_path.with_suffix(
                    prediction_path.suffix + ".manifest.json"
                ),
            ]
        )
    input_snapshot = _file_snapshot(immutable_paths)
    implementation_sha256 = input_snapshot[str(Path(__file__).resolve())]
    corpus_contract = validate_authoritative_corpus_manifest(
        corpus_manifest_path=corpus_manifest_path,
        assignments_path=assignments_path,
        stock_days_path=stock_days_path,
    )
    inference_contract = validate_complete_inference_manifests(
        prediction_paths=prediction_paths,
        assignments_path=assignments_path,
        preflight_manifest_path=preflight_manifest_path,
        acceptance=acceptance_snapshot,
    )
    _verify_file_snapshot(input_snapshot)
    if int(
        inference_contract["preflight"]["assignment_count"]
    ) != int(corpus_contract["assignment_record"]["rows"]):
        raise ValueError("W17 preflight count differs from authoritative corpus")

    database_path = _private_work_database(
        "aggregate",
        output_path,
        work_root,
    )
    if database_path.exists():
        raise FileExistsError(
            "Stale W17 aggregation database exists; inspect it before retrying"
        )
    connection = sqlite3.connect(database_path)
    rows: list[dict[str, Any]] = []
    assignment_count = 0
    prediction_count = 0
    shard_count = int(inference_contract["shard_count"])
    shard_assignment_counts = Counter()
    shard_prediction_counts = Counter()
    shard_assignment_ids = {
        index: hashlib.sha256() for index in range(shard_count)
    }
    acceptance_sha = str(inference_contract["acceptance_config_sha256"])
    runtime_sha = str(inference_contract["runtime_contract_sha256"])
    preflight_sha = str(inference_contract["preflight_sha256"])
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.executescript(
            """
            CREATE TABLE universe (
                forecast_date TEXT NOT NULL,
                sector TEXT NOT NULL,
                stock TEXT NOT NULL,
                benchmark TEXT NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY (forecast_date, sector, stock, benchmark)
            );
            CREATE TABLE assignments (
                assignment_id TEXT PRIMARY KEY,
                forecast_date TEXT NOT NULL,
                sector TEXT NOT NULL,
                stock TEXT NOT NULL,
                benchmark TEXT NOT NULL,
                shard_index INTEGER NOT NULL,
                assignment_input_sha256 TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE INDEX assignments_key
            ON assignments (forecast_date, sector, stock, benchmark);
            CREATE TABLE predictions (
                assignment_id TEXT PRIMARY KEY,
                shard_index INTEGER NOT NULL,
                payload TEXT NOT NULL
            );
            """
        )
        universe_count = 0
        for raw in iter_records(stock_days_path):
            scope = normalize_stock_day(raw)
            key = tuple(scope[name] for name in KEY_COLUMNS)
            try:
                connection.execute(
                    "INSERT INTO universe VALUES (?, ?, ?, ?, ?)",
                    (*key, canonical_json(scope)),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("W17 stock-day universe has duplicate keys") from exc
            universe_count += 1
            if universe_count % 10_000 == 0:
                connection.commit()
        connection.commit()
        if universe_count != int(corpus_contract["stock_day_record"]["rows"]):
            raise ValueError(
                "W17 stock-day rows differ from authoritative corpus"
            )

        for raw in iter_records(assignments_path):
            assignment = normalize_assignment(raw)
            key = _key_from_assignment(assignment)
            known = connection.execute(
                "SELECT 1 FROM universe WHERE forecast_date = ? "
                "AND sector = ? AND stock = ? AND benchmark = ?",
                key,
            ).fetchone()
            if known is None:
                raise ValueError(
                    "W17 assignment ledger contains a key outside the "
                    "authoritative stock-day universe"
                )
            shard_index = assignment_shard(assignment, shard_count)
            compact_assignment = {
                name: assignment[name]
                for name in (
                    "assignment_id",
                    "assignment_input_sha256",
                    "article_id",
                    "forecast_date",
                    "target",
                    "role",
                    "description_available",
                    "headline_sha256",
                    "description_sha256",
                    "text_sha256",
                    "source_profile",
                    "source_query_scope_complete",
                    "candidate_assignment_complete",
                    "aggregation_weight",
                )
            }
            try:
                connection.execute(
                    "INSERT INTO assignments VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        assignment["assignment_id"],
                        *key,
                        shard_index,
                        assignment["assignment_input_sha256"],
                        canonical_json(compact_assignment),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(
                    "W17 assignment ledger contains duplicate IDs"
                ) from exc
            shard_assignment_counts[shard_index] += 1
            if shard_assignment_counts[shard_index] > 1:
                shard_assignment_ids[shard_index].update(b"\n")
            shard_assignment_ids[shard_index].update(
                assignment["assignment_id"].encode("utf-8")
            )
            assignment_count += 1
            if assignment_count % 10_000 == 0:
                connection.commit()
        connection.commit()
        if assignment_count != int(
            corpus_contract["assignment_record"]["rows"]
        ):
            raise ValueError("W17 assignment count differs from corpus manifest")

        for shard in inference_contract["shards"]:
            expected_shard = int(shard["manifest"]["shard_index"])
            expected_count = int(
                shard["manifest"]["selected_assignment_count"]
            )
            if shard_assignment_counts[expected_shard] != expected_count:
                raise ValueError(
                    f"W17 authoritative assignment count differs for shard "
                    f"{expected_shard}"
                )
            if (
                shard_assignment_ids[expected_shard].hexdigest()
                != shard["manifest"]["selected_assignment_ids_sha256"]
            ):
                raise ValueError(
                    f"W17 authoritative assignment IDs differ for shard "
                    f"{expected_shard}"
                )
            for record in iter_records(shard["prediction_path"]):
                validate_terminal_record(record)
                if (
                    int(record["shard_index"]) != expected_shard
                    or int(record["shard_count"]) != shard_count
                    or record["runtime_contract_sha256"] != runtime_sha
                    or record["preflight_manifest_sha256"] != preflight_sha
                    or record["acceptance_config_sha256"] != acceptance_sha
                ):
                    raise ValueError(
                        "W17 prediction terminal differs from its shard "
                        "manifest"
                    )
                expected = connection.execute(
                    "SELECT shard_index, assignment_input_sha256 "
                    "FROM assignments WHERE assignment_id = ?",
                    (record["assignment_id"],),
                ).fetchone()
                if expected is None:
                    raise ValueError(
                        "W17 prediction references an unknown assignment"
                    )
                if (
                    int(expected[0]) != expected_shard
                    or expected[1] != record["assignment_input_sha256"]
                ):
                    raise ValueError(
                        "W17 prediction differs from authoritative assignment"
                    )
                try:
                    connection.execute(
                        "INSERT INTO predictions VALUES (?, ?, ?)",
                        (
                            record["assignment_id"],
                            expected_shard,
                            canonical_json(_compact_prediction(record)),
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    raise ValueError(
                        "W17 predictions contain duplicate assignment IDs"
                    ) from exc
                shard_prediction_counts[expected_shard] += 1
                prediction_count += 1
                if prediction_count % 10_000 == 0:
                    connection.commit()
            connection.commit()
            if shard_prediction_counts[expected_shard] != expected_count:
                raise ValueError(
                    f"W17 prediction count differs for shard {expected_shard}"
                )
        if prediction_count != assignment_count:
            raise ValueError("W17 predictions do not cover every assignment")
        missing = connection.execute(
            "SELECT COUNT(*) FROM assignments AS a "
            "LEFT JOIN predictions AS p USING (assignment_id) "
            "WHERE p.assignment_id IS NULL"
        ).fetchone()[0]
        if int(missing) != 0:
            raise ValueError("W17 daily join is missing predictions")

        cursor = connection.execute(
            "SELECT forecast_date, sector, stock, benchmark, payload "
            "FROM universe ORDER BY forecast_date, sector, stock, benchmark"
        )
        for forecast_date, sector, stock, benchmark, payload in cursor:
            joined = connection.execute(
                """
                SELECT a.payload, p.payload
                FROM assignments AS a
                LEFT JOIN predictions AS p USING (assignment_id)
                WHERE a.forecast_date = ? AND a.sector = ?
                  AND a.stock = ? AND a.benchmark = ?
                ORDER BY a.assignment_id
                """,
                (forecast_date, sector, stock, benchmark),
            ).fetchall()
            candidates = [json.loads(value[0]) for value in joined]
            predictions = {
                value["assignment_id"]: value
                for value in (
                    json.loads(row[1])
                    for row in joined
                    if row[1] is not None
                )
            }
            rows.append(
                _aggregate_normalized_scope(
                    scope=json.loads(payload),
                    candidates=candidates,
                    prediction_by_id=predictions,
                    acceptance_sha256=acceptance_sha,
                    expected_runtime_contract_sha256=runtime_sha,
                    expected_preflight_manifest_sha256=preflight_sha,
                )
            )
    finally:
        connection.close()
        database_path.unlink(missing_ok=True)
        database_path.with_name(database_path.name + "-wal").unlink(
            missing_ok=True
        )
        database_path.with_name(database_path.name + "-shm").unlink(
            missing_ok=True
        )

    pandas = _require_pandas()
    columns = [
        *KEY_COLUMNS,
        "source_profile",
        "semantic_row_complete",
        "semantic_ineligibility_reason",
        "expected_assignment_count",
        "observed_assignment_count",
        *WLLM17_COLUMNS,
    ]
    frame = pandas.DataFrame(rows, columns=columns)
    frame = frame.sort_values(list(KEY_COLUMNS), kind="mergesort").reset_index(
        drop=True
    )
    validate_complete_daily_panel(frame)
    if len(frame) != int(corpus_contract["stock_day_record"]["rows"]):
        raise ValueError("W17 daily output differs from authoritative row count")
    _verify_file_snapshot(input_snapshot)
    if canonical_json(acceptance) != canonical_json(acceptance_snapshot):
        raise ValueError("W17 acceptance configuration changed during aggregation")
    staged_output = output_path.with_name(
        f".{output_path.stem}.staged{output_path.suffix}"
    )
    if staged_output.exists():
        raise FileExistsError(
            f"Stale staged W17 daily output exists: {staged_output}"
        )
    try:
        write_records(staged_output, frame.to_dict(orient="records"))
        _verify_file_snapshot(input_snapshot)
        if canonical_json(acceptance) != canonical_json(acceptance_snapshot):
            raise ValueError(
                "W17 acceptance configuration changed during output staging"
            )
        os.replace(staged_output, output_path)
    finally:
        staged_output.unlink(missing_ok=True)
    manifest = _daily_manifest(
        corpus_manifest_path=corpus_manifest_path,
        corpus_contract=corpus_contract,
        assignments_path=assignments_path,
        inference_contract=inference_contract,
        stock_days_path=stock_days_path,
        preflight_manifest_path=preflight_manifest_path,
        input_snapshot=input_snapshot,
        implementation_sha256=implementation_sha256,
        output_path=output_path,
        frame=frame,
        acceptance=acceptance_snapshot,
    )
    _atomic_text(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser("preflight")
    preflight.add_argument("--input", type=Path, default=DEFAULT_ASSIGNMENTS)
    preflight.add_argument("--output", type=Path, default=DEFAULT_PREFLIGHT)
    preflight.add_argument("--prompt-batch-size", type=int, default=512)

    infer = commands.add_parser("infer")
    infer.add_argument("--input", type=Path, default=DEFAULT_ASSIGNMENTS)
    infer.add_argument("--output", required=True, type=Path)
    infer.add_argument("--acceptance-config", type=Path)
    infer.add_argument(
        "--preflight-manifest",
        type=Path,
        default=DEFAULT_PREFLIGHT,
    )
    infer.add_argument("--shard-index", type=int, default=0)
    infer.add_argument(
        "--shard-count",
        type=int,
        default=DEFAULT_INFERENCE_SHARD_COUNT,
    )
    infer.add_argument("--retry-failed", action="store_true")
    infer.add_argument("--limit", type=int)
    infer.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=SYSTEMIC_FAILURE_ABORT_THRESHOLD,
    )

    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument(
        "--assignments",
        type=Path,
        default=DEFAULT_ASSIGNMENTS,
    )
    aggregate.add_argument("--predictions", required=True, type=Path, nargs="+")
    aggregate.add_argument(
        "--stock-days",
        type=Path,
        default=DEFAULT_STOCK_DAYS,
    )
    aggregate.add_argument(
        "--corpus-manifest",
        type=Path,
        default=DEFAULT_CORPUS_MANIFEST,
    )
    aggregate.add_argument(
        "--preflight-manifest",
        type=Path,
        default=DEFAULT_PREFLIGHT,
    )
    aggregate.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DAILY_OUTPUT,
    )
    aggregate.add_argument("--acceptance-config", type=Path)
    aggregate.add_argument("--work-root", type=Path)
    aggregate.add_argument("--overwrite", action="store_true")

    acceptance = commands.add_parser("print-default-acceptance")
    acceptance.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        manifest = preflight_corpus(
            input_path=args.input,
            output_path=args.output,
            prompt_batch_size=args.prompt_batch_size,
        )
        print(canonical_json(manifest))
        return 0 if manifest["status"] == "passed" else 2
    if args.command == "print-default-acceptance":
        value = json.dumps(default_acceptance_config(), indent=2) + "\n"
        if args.output is None:
            print(value, end="")
        else:
            _atomic_text(args.output, value)
        return 0
    acceptance = load_acceptance_config(args.acceptance_config)
    if args.command == "infer":
        validate_preflight_manifest(args.preflight_manifest, args.input)
        engine = FlanT5XlW17Engine(acceptance)
        manifest = run_inference(
            input_path=args.input,
            output_path=args.output,
            acceptance=acceptance,
            engine=engine,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            retry_failed=args.retry_failed,
            limit=args.limit,
            preflight_manifest_path=args.preflight_manifest,
            max_consecutive_failures=args.max_consecutive_failures,
        )
    else:
        manifest = write_complete_daily_panel(
            assignments_path=args.assignments,
            prediction_paths=args.predictions,
            stock_days_path=args.stock_days,
            corpus_manifest_path=args.corpus_manifest,
            preflight_manifest_path=args.preflight_manifest,
            output_path=args.output,
            acceptance=acceptance,
            work_root=args.work_root,
            overwrite=args.overwrite,
        )
    print(canonical_json(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
