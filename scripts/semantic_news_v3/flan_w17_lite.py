#!/usr/bin/env python
"""Select and infer the resumable K-16 FLAN W17-Lite article corpus.

This runner is intentionally separate from the v2 W17 runner.  It performs
one target-invariant four-way event classification per selected unique
article.  Selection and tokenizer preflight are immutable, hash-bound gates.
Inference appends and fsyncs one terminal JSON object per article; rerunning
the same command validates every retained record and skips completed work.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import signal
import socket
import sys
import tempfile
import time
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Protocol, Sequence

try:
    import pandas as pd
except ImportError:  # The CUDA inference environment intentionally omits it.
    pd = None


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIRECTORY = REPOSITORY_ROOT / "scripts"
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

import extract_flan_t5 as flan_base  # noqa: E402
import extract_flan_t5_coarse as coarse_runner  # noqa: E402
import run_flan_t5_xl_coarse as xl_runner  # noqa: E402

xl_runner.SNAPSHOT_MANIFEST = (
    REPOSITORY_ROOT / "outputs" / "flan_t5_xl" / "model_snapshot_manifest.json"
)


PIPELINE_VERSION = "flan-w17-lite-k16-construction-v1"
CONTRACT_ID = "weak-news-semantics-lite-v1"
ARM_ID = "WL17__flan_t5_xl"
MODEL_ID = "google/flan-t5-xl"
MODEL_REVISION = "7d6315df2c2fb742f0f5b556879d730926ca9001"
MAX_INPUT_TOKENS = 512
SELECTOR_K = 16
SOURCE_PROFILE = "ordinary_massive_retrospective"
EVENT_FIELD = "event_family"
EVENT_GROUP_LABELS = (
    "firm_operating_financial",
    "policy_corporate",
    "macro_market",
    "other_or_unclear",
)
ABSTENTION_LABEL = "other_or_unclear"
SUCCESS_TERMINAL_STATES = {"complete"}
FAILURE_TERMINAL_STATES = {
    "prompt_too_long",
    "inference_failed",
    "invalid_output",
}
ALL_TERMINAL_STATES = SUCCESS_TERMINAL_STATES | FAILURE_TERMINAL_STATES
SYSTEMIC_FAILURE_ABORT_THRESHOLD = 3
PROGRESS_INTERVAL = 25

DEFAULT_SOURCE_ROOT = (
    REPOSITORY_ROOT / "data" / "features" / "news_semantic" / "massive_v2"
)
DEFAULT_ASSIGNMENTS = DEFAULT_SOURCE_ROOT / "article_target_assignments.parquet"
DEFAULT_ARTICLES = DEFAULT_SOURCE_ROOT / "semantic_articles.parquet"
DEFAULT_CORPUS_MANIFEST = DEFAULT_SOURCE_ROOT / "manifest.json"
DEFAULT_DESIGN = REPOSITORY_ROOT / "config" / "news_semantic_lite_design_v1.json"
DEFAULT_SCHEMA = (
    REPOSITORY_ROOT / "config" / "flan_w17_lite_event_schema_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    REPOSITORY_ROOT
    / "data"
    / "features"
    / "news_semantic"
    / "massive_v3"
    / "flan_w17_lite_k16"
)
DEFAULT_QUEUE = DEFAULT_OUTPUT_ROOT / "selected_articles.jsonl.gz"
DEFAULT_SELECTION_MANIFEST = DEFAULT_OUTPUT_ROOT / "selection_manifest.json"
DEFAULT_PREFLIGHT = DEFAULT_OUTPUT_ROOT / "preflight.json"
DEFAULT_PREDICTIONS = DEFAULT_OUTPUT_ROOT / "predictions.jsonl"


EXPECTED_K16_PROFILE = {
    "source_assignment_rows": 466_902,
    "source_unique_articles": 55_197,
    "selected_unique_articles": 50_488,
    "omitted_unique_articles": 4_709,
    "selected_assignment_rows": 438_522,
    "eligible_group_article_rows": {
        "direct": 75_130,
        "sector": 15_170,
        "macro": 4_545,
    },
    "eligible_group_counts": {
        "direct": 18_534,
        "sector": 3_507,
        "macro": 863,
    },
    "selected_group_article_rows": {
        "direct": 67_474,
        "sector": 14_756,
        "macro": 4_537,
    },
    "pool_selected_unique_articles": {
        "direct": 47_830,
        "sector": 14_022,
        "macro": 4_537,
    },
    "selected_weight": 208_298.44822183,
    "source_weight": 214_021.07771033,
    "selected_weight_fraction": 0.973261374,
    "forecast_dates": 917,
    "target_tickers": 30,
    "sectors": 5,
    "candidate_bearing_stock_days": 27_306,
    "direct_bearing_stock_days": 18_534,
    "ordered_selected_article_ids_sha256": (
        "c676de4ada043e66d44497fdb693daebc04467303e689208cb2e0c802f568da2"
    ),
    "ordered_selected_assignment_ids_sha256": (
        "9a6158e10b97a400890cb8ceab41b56cebf93dc2a8da7fadafe721870079fba9"
    ),
}


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


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _require_pandas() -> Any:
    if pd is None:
        raise RuntimeError(
            "Selection preparation requires pandas and pyarrow; use "
            ".venv-training\\Scripts\\python.exe"
        )
    return pd


def _replace_with_retry(source: Path, target: Path) -> None:
    last_error: PermissionError | None = None
    for attempt in range(8):
        try:
            os.replace(source, target)
            return
        except PermissionError as error:
            last_error = error
            time.sleep(0.05 * (attempt + 1))
    if last_error is not None:
        raise last_error


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        _replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text(path: Path, value: str) -> None:
    _atomic_bytes(path, value.encode("utf-8"))


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _atomic_jsonl(
    path: Path,
    records: Iterable[Mapping[str, Any]],
    *,
    gzip_output: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w+b",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as raw:
        temporary = Path(raw.name)
        if gzip_output:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw,
                mtime=0,
            ) as compressed:
                with io.TextIOWrapper(
                    compressed,
                    encoding="utf-8",
                    newline="\n",
                ) as text:
                    for record in records:
                        text.write(canonical_json(record) + "\n")
        else:
            for record in records:
                raw.write((canonical_json(record) + "\n").encode("utf-8"))
        raw.flush()
        os.fsync(raw.fileno())
    try:
        _replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_parquet(path: Path, frame: Any) -> None:
    _require_pandas()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w+b",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False)
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        _replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    suffixes = "".join(path.suffixes).lower()
    opener = gzip.open if suffixes.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number}: expected an object")
            yield value


def _ordered_lf_hash(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _utf8_sorted(values: Iterable[str]) -> list[str]:
    return sorted(values, key=lambda value: value.encode("utf-8"))


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain one JSON object")
    return value


def validate_event_schema(schema: Mapping[str, Any]) -> None:
    labels = tuple(
        schema.get("closed_label_fields", {}).get(EVENT_FIELD, ())
    )
    if labels != EVENT_GROUP_LABELS:
        raise ValueError("W17-Lite event schema labels or order changed")
    definitions = schema.get("field_definitions", {}).get(EVENT_FIELD)
    if not isinstance(definitions, Mapping) or set(definitions) != set(labels):
        raise ValueError("W17-Lite event schema definitions differ")
    if schema.get("output_field") != "event_group":
        raise ValueError("W17-Lite output field must remain event_group")
    acceptance = schema.get("acceptance")
    if (
        not isinstance(acceptance, Mapping)
        or acceptance.get("requires_order_agreement") is not True
        or tuple(acceptance.get("abstention_labels", ()))
        != (ABSTENTION_LABEL,)
        or acceptance.get("inherits_v1_1_calibration") is not False
    ):
        raise ValueError("W17-Lite acceptance contract changed")


def load_event_schema(path: Path = DEFAULT_SCHEMA) -> dict[str, Any]:
    schema = _load_json(path)
    validate_event_schema(schema)
    return schema


def _source_generated_record(
    manifest: Mapping[str, Any], name: str
) -> Mapping[str, Any]:
    record = manifest.get("generated_files", {}).get(name)
    if not isinstance(record, Mapping):
        raise ValueError(f"Source manifest lacks {name}")
    return record


def validate_source_inputs(
    assignments_path: Path,
    articles_path: Path,
    corpus_manifest_path: Path,
) -> dict[str, Any]:
    manifest = _load_json(corpus_manifest_path)
    if manifest.get("manifest_version") != "semantic-corpus-manifest-v1":
        raise ValueError("Unsupported semantic source manifest")
    if manifest.get("status") != "complete_exploratory_retrospective":
        raise ValueError("Semantic source corpus is not complete")
    if manifest.get("source_profile") != SOURCE_PROFILE:
        raise ValueError("Semantic source profile changed")
    expected = {
        "article_target_assignments.parquet": assignments_path,
        "semantic_articles.parquet": articles_path,
    }
    snapshots: dict[str, Any] = {}
    for name, path in expected.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        record = _source_generated_record(manifest, name)
        actual_hash = sha256_file(path)
        if record.get("sha256") != actual_hash:
            raise ValueError(f"Source hash differs for {name}")
        snapshots[name] = {
            "path": str(path.resolve()),
            "sha256": actual_hash,
            "rows": int(record["rows"]),
            "bytes": path.stat().st_size,
        }
    return {
        "corpus_manifest": {
            "path": str(corpus_manifest_path.resolve()),
            "sha256": sha256_file(corpus_manifest_path),
        },
        "files": snapshots,
        "source_profile": SOURCE_PROFILE,
    }


ASSIGNMENT_COLUMNS = (
    "assignment_id",
    "article_id",
    "forecast_date",
    "published_at_utc",
    "cutoff_utc",
    "target_ticker",
    "sector",
    "role",
    "candidate_roles",
    "aggregation_weight",
    "source_profile",
    "source_query_scope_complete",
    "candidate_assignment_complete",
    "headline",
    "description",
    "model_text",
    "headline_sha256",
    "description_sha256",
    "text_sha256",
    "description_available",
)
ARTICLE_COLUMNS = (
    "provider_article_id",
    "published_at_utc",
    "forecast_date",
    "cutoff_utc",
    "headline",
    "description",
    "description_available",
    "retained_description_char_count",
    "description_was_bounded",
    "model_text",
    "headline_sha256",
    "description_sha256",
    "model_text_sha256",
    "source_profile",
)
MEMBERSHIP_COLUMNS = (
    "pool",
    "forecast_date",
    "target_ticker",
    "sector",
    "article_id",
    "rank",
    "aggregation_weight",
    "published_at_utc",
)


def _candidate_role(record: Any, name: str) -> bool:
    if not isinstance(record, Mapping):
        raise TypeError("candidate_roles must be a mapping")
    if name not in record or not isinstance(record[name], (bool,)):
        raise TypeError(f"candidate_roles.{name} must be boolean")
    return bool(record[name])


def _validate_assignment_frame(
    frame: Any, *, enforce_expected_profile: bool = True
) -> Any:
    pandas = _require_pandas()
    missing = set(ASSIGNMENT_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Assignment input lacks columns: {sorted(missing)}")
    if (
        enforce_expected_profile
        and len(frame) != EXPECTED_K16_PROFILE["source_assignment_rows"]
    ):
        raise ValueError("Assignment source row count changed")
    if frame["assignment_id"].isna().any() or frame["article_id"].isna().any():
        raise ValueError("Assignment IDs cannot be missing")
    if frame["assignment_id"].duplicated().any():
        raise ValueError("Assignment IDs are not unique")
    if (
        enforce_expected_profile
        and frame["article_id"].nunique()
        != EXPECTED_K16_PROFILE["source_unique_articles"]
    ):
        raise ValueError("Assigned unique-article count changed")
    if not frame["source_profile"].eq(SOURCE_PROFILE).all():
        raise ValueError("Assignment source_profile changed")
    if not frame["source_query_scope_complete"].eq(True).all():
        raise ValueError("Assignment source-query scope is incomplete")
    if not frame["candidate_assignment_complete"].eq(True).all():
        raise ValueError("Candidate assignment scope is incomplete")
    weights = pandas.to_numeric(frame["aggregation_weight"], errors="coerce")
    if (
        weights.isna().any()
        or not (weights > 0).all()
        or not all(math.isfinite(float(value)) for value in weights)
    ):
        raise ValueError("Assignment weights must be positive and finite")
    frame = frame.copy()
    frame["aggregation_weight"] = weights.astype(float)
    frame["forecast_date"] = pandas.to_datetime(
        frame["forecast_date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")
    frame["published_at_utc"] = pandas.to_datetime(
        frame["published_at_utc"], utc=True, errors="raise"
    )
    frame["cutoff_utc"] = pandas.to_datetime(
        frame["cutoff_utc"], utc=True, errors="raise"
    )
    if not (frame["published_at_utc"] < frame["cutoff_utc"]).all():
        raise ValueError("Every assignment must precede its forecast cutoff")
    required_roles = (
        "direct",
        "sector_common",
        "macro_common",
        "target_idio",
        "target_common",
        "peer_idio",
        "any_common",
    )
    for name in required_roles:
        frame[f"_candidate_{name}"] = frame["candidate_roles"].map(
            lambda value, field=name: _candidate_role(value, field)
        )
    if frame[["_candidate_direct", "_candidate_sector_common", "_candidate_macro_common"]].sum(
        axis=1
    ).eq(0).all():
        raise ValueError("No K-16 candidate pool is populated")
    text_columns = (
        "headline",
        "description",
        "model_text",
        "headline_sha256",
        "description_sha256",
        "text_sha256",
    )
    for name in text_columns:
        if frame[name].isna().any():
            raise ValueError(f"Assignment {name} contains missing values")
    per_article = frame.groupby("article_id", sort=False)[list(text_columns)].nunique(
        dropna=False
    )
    if (per_article > 1).any().any():
        raise ValueError("Assignments disagree on article text or hashes")
    return frame


def _validate_article_frame(
    frame: Any, *, enforce_expected_profile: bool = True
) -> Any:
    pandas = _require_pandas()
    missing = set(ARTICLE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Article input lacks columns: {sorted(missing)}")
    if (
        enforce_expected_profile
        and len(frame) != EXPECTED_K16_PROFILE["source_unique_articles"]
    ):
        raise ValueError("Semantic article source row count changed")
    if frame["provider_article_id"].isna().any():
        raise ValueError("Semantic article IDs cannot be missing")
    if frame["provider_article_id"].duplicated().any():
        raise ValueError("Semantic article IDs are not unique")
    if not frame["source_profile"].eq(SOURCE_PROFILE).all():
        raise ValueError("Semantic article source_profile changed")
    output = frame.copy()
    output["forecast_date"] = pandas.to_datetime(
        output["forecast_date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")
    output["published_at_utc"] = pandas.to_datetime(
        output["published_at_utc"], utc=True, errors="raise"
    )
    output["cutoff_utc"] = pandas.to_datetime(
        output["cutoff_utc"], utc=True, errors="raise"
    )
    if not (output["published_at_utc"] < output["cutoff_utc"]).all():
        raise ValueError("Every semantic article must precede its cutoff")
    for row in output.itertuples(index=False):
        headline = str(row.headline)
        description = str(row.description)
        model_text = (
            headline if not description else f"{headline}\n\n{description}"
        )
        if model_text != row.model_text:
            raise ValueError("Semantic article model_text contract changed")
        if sha256_text(headline) != row.headline_sha256:
            raise ValueError("Semantic article headline hash differs")
        if sha256_text(description) != row.description_sha256:
            raise ValueError("Semantic article description hash differs")
        if sha256_text(model_text) != row.model_text_sha256:
            raise ValueError("Semantic article model-text hash differs")
        if bool(row.description_available) is not bool(description):
            raise ValueError("description_available differs from text")
    return output


def _deduplicate_and_rank_pool(
    frame: Any,
    *,
    pool: str,
    flag_column: str,
    group_columns: Sequence[str],
    k: int,
) -> tuple[Any, int, int]:
    pandas = _require_pandas()
    columns = [
        *group_columns,
        "article_id",
        "aggregation_weight",
        "published_at_utc",
    ]
    eligible = frame.loc[frame[flag_column], columns].copy()
    if eligible.empty:
        raise ValueError(f"K-16 {pool} pool is empty")
    identity = [*group_columns, "article_id"]
    agreement = eligible.groupby(identity, dropna=False, sort=False)[
        ["aggregation_weight", "published_at_utc"]
    ].nunique(dropna=False)
    if (agreement > 1).any().any():
        raise ValueError(
            f"Duplicate {pool} group/article rows disagree on rank inputs"
        )
    eligible = eligible.drop_duplicates(identity, keep="first")
    eligible["_article_utf8"] = eligible["article_id"].map(
        lambda value: str(value).encode("utf-8").hex()
    )
    eligible = eligible.sort_values(
        [
            *group_columns,
            "aggregation_weight",
            "published_at_utc",
            "_article_utf8",
        ],
        ascending=[True] * len(group_columns) + [False, False, True],
        kind="mergesort",
    )
    eligible["rank"] = (
        eligible.groupby(list(group_columns), dropna=False, sort=False)
        .cumcount()
        .add(1)
    )
    group_count = int(
        eligible[list(group_columns)].drop_duplicates().shape[0]
    )
    eligible_count = len(eligible)
    selected = eligible.loc[eligible["rank"].le(k)].copy()
    selected["pool"] = pool
    selected["target_ticker"] = (
        selected["target_ticker"] if "target_ticker" in selected else None
    )
    selected["sector"] = (
        selected["sector"] if "sector" in selected else None
    )
    selected = selected[list(MEMBERSHIP_COLUMNS)]
    selected["published_at_utc"] = selected["published_at_utc"].dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    selected = selected.sort_values(
        ["pool", "forecast_date", "target_ticker", "sector", "rank"],
        na_position="first",
        kind="mergesort",
    ).reset_index(drop=True)
    return selected, eligible_count, group_count


def _queue_records(frame: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        article_id = str(row.provider_article_id)
        payload = {
            "record_version": "flan-w17-lite-selected-article-v1",
            "article_id": article_id,
            "forecast_date": str(row.forecast_date),
            "published_at_utc": row.published_at_utc.strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "cutoff_utc": row.cutoff_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "headline": str(row.headline),
            "description": str(row.description),
            "description_available": bool(row.description_available),
            "retained_description_char_count": int(
                row.retained_description_char_count
            ),
            "description_was_bounded": bool(row.description_was_bounded),
            "model_text": str(row.model_text),
            "headline_sha256": str(row.headline_sha256),
            "description_sha256": str(row.description_sha256),
            "model_text_sha256": str(row.model_text_sha256),
            "source_profile": str(row.source_profile),
        }
        payload["article_input_sha256"] = sha256_json(payload)
        records.append(payload)
    records.sort(key=lambda value: value["article_id"].encode("utf-8"))
    return records


def build_k16_selection(
    assignments: Any,
    articles: Any,
    *,
    k: int = SELECTOR_K,
    enforce_expected_profile: bool = True,
) -> tuple[Any, Any, list[dict[str, Any]], dict[str, Any]]:
    """Return membership, selected assignments, queue records, and audit."""

    pandas = _require_pandas()
    if k != SELECTOR_K:
        raise ValueError(f"This construction is pinned to K={SELECTOR_K}")
    assignments = _validate_assignment_frame(
        assignments, enforce_expected_profile=enforce_expected_profile
    )
    articles = _validate_article_frame(
        articles, enforce_expected_profile=enforce_expected_profile
    )
    if set(assignments["article_id"]) != set(articles["provider_article_id"]):
        raise ValueError("Assignment and semantic-article ID universes differ")

    specifications = (
        ("direct", "_candidate_direct", ("forecast_date", "target_ticker")),
        ("sector", "_candidate_sector_common", ("forecast_date", "sector")),
        ("macro", "_candidate_macro_common", ("forecast_date",)),
    )
    membership_frames = []
    eligible_rows: dict[str, int] = {}
    eligible_groups: dict[str, int] = {}
    selected_rows: dict[str, int] = {}
    pool_unique: dict[str, int] = {}
    pool_ids: dict[str, set[str]] = {}
    for pool, flag, groups in specifications:
        selected, row_count, group_count = _deduplicate_and_rank_pool(
            assignments,
            pool=pool,
            flag_column=flag,
            group_columns=groups,
            k=k,
        )
        membership_frames.append(selected)
        eligible_rows[pool] = row_count
        eligible_groups[pool] = group_count
        selected_rows[pool] = len(selected)
        pool_ids[pool] = set(selected["article_id"])
        pool_unique[pool] = len(pool_ids[pool])
    membership = pandas.concat(membership_frames, ignore_index=True)
    selected_ids = _utf8_sorted(set().union(*pool_ids.values()))
    selected_id_set = set(selected_ids)
    selected_assignments = assignments.loc[
        assignments["article_id"].isin(selected_id_set),
        list(ASSIGNMENT_COLUMNS),
    ].copy()
    selected_assignments = selected_assignments.sort_values(
        "assignment_id", kind="mergesort"
    ).reset_index(drop=True)
    selected_articles = articles.loc[
        articles["provider_article_id"].isin(selected_id_set),
        list(ARTICLE_COLUMNS),
    ].copy()
    selected_articles["_article_utf8"] = selected_articles[
        "provider_article_id"
    ].map(lambda value: str(value).encode("utf-8").hex())
    selected_articles = selected_articles.sort_values(
        "_article_utf8", kind="mergesort"
    ).drop(columns="_article_utf8")
    queue = _queue_records(selected_articles)
    if [record["article_id"] for record in queue] != selected_ids:
        raise AssertionError("Queue order differs from selected ID ledger")

    representative = assignments.drop_duplicates("article_id").set_index(
        "article_id"
    )
    article_index = selected_articles.set_index("provider_article_id")
    for article_id in selected_ids:
        left = representative.loc[article_id]
        right = article_index.loc[article_id]
        if (
            left["headline_sha256"] != right["headline_sha256"]
            or left["description_sha256"] != right["description_sha256"]
            or left["text_sha256"] != right["model_text_sha256"]
            or left["model_text"] != right["model_text"]
        ):
            raise ValueError(
                f"Assignment/article text provenance differs for {article_id}"
            )

    source_weight = float(assignments["aggregation_weight"].sum())
    selected_weight = float(selected_assignments["aggregation_weight"].sum())
    candidate_stock_days = int(
        assignments[["forecast_date", "target_ticker"]]
        .drop_duplicates()
        .shape[0]
    )
    direct_stock_days = int(
        assignments.loc[
            assignments["_candidate_direct"],
            ["forecast_date", "target_ticker"],
        ]
        .drop_duplicates()
        .shape[0]
    )
    description_lengths = selected_articles[
        "retained_description_char_count"
    ].astype(int)
    years = selected_articles["forecast_date"].str[:4].value_counts()
    audit = {
        "selector": {
            "k": k,
            "selection_family": (
                "symmetric_top_k_direct_stock_day_sector_common_sector_day_"
                "macro_common_date"
            ),
            "rank": [
                "aggregation_weight_desc",
                "published_at_utc_desc",
                "article_id_utf8_asc",
            ],
            "deduplicate_group_article_before_rank": True,
            "deduplicate_unique_article_union_before_inference": True,
        },
        "source_assignment_rows": len(assignments),
        "source_unique_articles": int(assignments["article_id"].nunique()),
        "selected_unique_articles": len(selected_ids),
        "omitted_unique_articles": (
            int(assignments["article_id"].nunique()) - len(selected_ids)
        ),
        "selected_assignment_rows": len(selected_assignments),
        "selected_assignment_fraction": (
            len(selected_assignments) / len(assignments)
        ),
        "source_weight": source_weight,
        "selected_weight": selected_weight,
        "selected_weight_fraction": selected_weight / source_weight,
        "eligible_group_article_rows": eligible_rows,
        "eligible_group_counts": eligible_groups,
        "selected_group_article_rows": selected_rows,
        "pool_selected_unique_articles": pool_unique,
        "pool_intersections": {
            "direct_and_sector": len(pool_ids["direct"] & pool_ids["sector"]),
            "direct_and_macro": len(pool_ids["direct"] & pool_ids["macro"]),
            "sector_and_macro": len(pool_ids["sector"] & pool_ids["macro"]),
            "all_three": len(
                pool_ids["direct"] & pool_ids["sector"] & pool_ids["macro"]
            ),
        },
        "forecast_dates": int(assignments["forecast_date"].nunique()),
        "target_tickers": int(assignments["target_ticker"].nunique()),
        "sectors": int(assignments["sector"].nunique()),
        "candidate_bearing_stock_days": candidate_stock_days,
        "direct_bearing_stock_days": direct_stock_days,
        "selected_text_strata": {
            "headline_only": int(
                selected_articles["description_available"].eq(0).sum()
            ),
            "description_1_149": int(
                (
                    selected_articles["description_available"].eq(1)
                    & description_lengths.lt(150)
                ).sum()
            ),
            "description_ge_150": int(description_lengths.ge(150).sum()),
            "bounded_description": int(
                selected_articles["description_was_bounded"].sum()
            ),
        },
        "selected_years": {
            str(year): int(count)
            for year, count in years.sort_index().items()
        },
        "role_rows": {
            str(role): {
                "selected": int(
                    selected_assignments["role"].eq(role).sum()
                ),
                "source": int(assignments["role"].eq(role).sum()),
            }
            for role in ("C", "I", "P")
        },
        "ordered_selected_article_ids_sha256": _ordered_lf_hash(selected_ids),
        "ordered_selected_assignment_ids_sha256": _ordered_lf_hash(
            _utf8_sorted(selected_assignments["assignment_id"].astype(str))
        ),
    }
    return membership, selected_assignments, queue, audit


def _assert_expected_k16_profile(audit: Mapping[str, Any]) -> None:
    scalar_names = (
        "source_assignment_rows",
        "source_unique_articles",
        "selected_unique_articles",
        "omitted_unique_articles",
        "selected_assignment_rows",
        "forecast_dates",
        "target_tickers",
        "sectors",
        "candidate_bearing_stock_days",
        "direct_bearing_stock_days",
        "ordered_selected_article_ids_sha256",
        "ordered_selected_assignment_ids_sha256",
    )
    for name in scalar_names:
        if audit.get(name) != EXPECTED_K16_PROFILE[name]:
            raise ValueError(
                f"K-16 audit {name} changed: {audit.get(name)!r} != "
                f"{EXPECTED_K16_PROFILE[name]!r}"
            )
    for name in (
        "eligible_group_article_rows",
        "eligible_group_counts",
        "selected_group_article_rows",
        "pool_selected_unique_articles",
    ):
        if audit.get(name) != EXPECTED_K16_PROFILE[name]:
            raise ValueError(f"K-16 audit mapping changed for {name}")
    for name in ("source_weight", "selected_weight"):
        if not math.isclose(
            float(audit[name]),
            float(EXPECTED_K16_PROFILE[name]),
            rel_tol=0,
            abs_tol=1e-6,
        ):
            raise ValueError(f"K-16 audit weight changed for {name}")
    if not math.isclose(
        float(audit["selected_weight_fraction"]),
        float(EXPECTED_K16_PROFILE["selected_weight_fraction"]),
        rel_tol=0,
        abs_tol=5e-10,
    ):
        raise ValueError("K-16 selected weight fraction changed")


def _selection_outputs(output_root: Path) -> dict[str, Path]:
    return {
        "selected_article_ids.txt": output_root / "selected_article_ids.txt",
        "selection_membership.parquet": (
            output_root / "selection_membership.parquet"
        ),
        "selected_assignments.parquet": (
            output_root / "selected_assignments.parquet"
        ),
        "selected_articles.jsonl.gz": (
            output_root / "selected_articles.jsonl.gz"
        ),
    }


def _validate_generated_file(
    name: str, record: Mapping[str, Any], expected_path: Path
) -> None:
    if Path(str(record.get("path", ""))).resolve() != expected_path.resolve():
        raise ValueError(f"Selection manifest path differs for {name}")
    if not expected_path.is_file():
        raise FileNotFoundError(expected_path)
    if sha256_file(expected_path) != record.get("sha256"):
        raise ValueError(f"Selection output hash differs for {name}")
    if expected_path.stat().st_size != int(record.get("bytes", -1)):
        raise ValueError(f"Selection output size differs for {name}")


def validate_selection_manifest(
    path: Path = DEFAULT_SELECTION_MANIFEST,
    *,
    queue_path: Path | None = None,
    require_current_implementation: bool = True,
) -> dict[str, Any]:
    manifest = _load_json(path)
    if manifest.get("manifest_version") != "flan-w17-lite-selection-v1":
        raise ValueError("Unsupported W17-Lite selection manifest")
    if manifest.get("status") != "complete":
        raise ValueError("W17-Lite selection is not complete")
    if manifest.get("selector_k") != SELECTOR_K:
        raise ValueError("W17-Lite selection K changed")
    audit = manifest.get("audit")
    if not isinstance(audit, Mapping):
        raise ValueError("W17-Lite selection audit is missing")
    _assert_expected_k16_profile(audit)
    outputs = manifest.get("generated_files")
    if not isinstance(outputs, Mapping):
        raise ValueError("W17-Lite generated-file records are missing")
    output_root = path.parent
    expected = _selection_outputs(output_root)
    for name, expected_path in expected.items():
        record = outputs.get(name)
        if not isinstance(record, Mapping):
            raise ValueError(f"Selection manifest lacks {name}")
        _validate_generated_file(name, record, expected_path)
    if queue_path is not None and queue_path.resolve() != expected[
        "selected_articles.jsonl.gz"
    ].resolve():
        raise ValueError("Requested queue differs from selection manifest")
    if require_current_implementation:
        if manifest.get("implementation_sha256") != sha256_file(
            Path(__file__).resolve()
        ):
            raise ValueError("W17-Lite implementation changed after selection")
        if manifest.get("design_sha256") != sha256_file(DEFAULT_DESIGN):
            raise ValueError("W17-Lite design config changed after selection")
        if manifest.get("schema_sha256") != sha256_file(DEFAULT_SCHEMA):
            raise ValueError("W17-Lite event schema changed after selection")
    return manifest


def prepare_selection(
    *,
    assignments_path: Path = DEFAULT_ASSIGNMENTS,
    articles_path: Path = DEFAULT_ARTICLES,
    corpus_manifest_path: Path = DEFAULT_CORPUS_MANIFEST,
    design_path: Path = DEFAULT_DESIGN,
    schema_path: Path = DEFAULT_SCHEMA,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    """Materialize or validate the exact current-corpus K-16 selection."""

    _require_pandas()
    load_event_schema(schema_path)
    source = validate_source_inputs(
        assignments_path, articles_path, corpus_manifest_path
    )
    manifest_path = output_root / "selection_manifest.json"
    if manifest_path.exists():
        manifest = validate_selection_manifest(
            manifest_path,
            queue_path=output_root / "selected_articles.jsonl.gz",
        )
        if manifest.get("source") != source:
            raise ValueError("Selection source snapshot changed")
        if manifest.get("design_sha256") != sha256_file(design_path):
            raise ValueError("Selection design snapshot changed")
        if manifest.get("schema_sha256") != sha256_file(schema_path):
            raise ValueError("Selection schema snapshot changed")
        return manifest

    assignments = pd.read_parquet(
        assignments_path, columns=list(ASSIGNMENT_COLUMNS)
    )
    articles = pd.read_parquet(articles_path, columns=list(ARTICLE_COLUMNS))
    membership, selected_assignments, queue, audit = build_k16_selection(
        assignments, articles, k=SELECTOR_K
    )
    _assert_expected_k16_profile(audit)
    output_root.mkdir(parents=True, exist_ok=True)
    outputs = _selection_outputs(output_root)
    selected_ids = [record["article_id"] for record in queue]
    _atomic_text(
        outputs["selected_article_ids.txt"],
        "".join(f"{value}\n" for value in selected_ids),
    )
    _atomic_parquet(
        outputs["selection_membership.parquet"], membership
    )
    _atomic_parquet(
        outputs["selected_assignments.parquet"], selected_assignments
    )
    _atomic_jsonl(
        outputs["selected_articles.jsonl.gz"],
        queue,
        gzip_output=True,
    )
    generated = {}
    for name, path in outputs.items():
        if name.endswith(".parquet"):
            rows = (
                len(membership)
                if name == "selection_membership.parquet"
                else len(selected_assignments)
            )
        elif name.endswith(".jsonl.gz"):
            rows = len(queue)
        else:
            rows = len(selected_ids)
        generated[name] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "rows": rows,
        }
    manifest = {
        "manifest_version": "flan-w17-lite-selection-v1",
        "pipeline_version": PIPELINE_VERSION,
        "status": "complete",
        "generated_at_utc": utc_now(),
        "arm_id": ARM_ID,
        "contract_id": CONTRACT_ID,
        "selector_k": SELECTOR_K,
        "source": source,
        "design_path": str(design_path.resolve()),
        "design_sha256": sha256_file(design_path),
        "schema_path": str(schema_path.resolve()),
        "schema_sha256": sha256_file(schema_path),
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
        "audit": audit,
        "generated_files": generated,
        "claim_flags": {
            "point_in_time_version_safe": False,
            "primary_or_confirmatory_eligible": False,
            "exploratory_construction_eligible": True,
        },
    }
    _atomic_json(manifest_path, manifest)
    validate_selection_manifest(
        manifest_path,
        queue_path=outputs["selected_articles.jsonl.gz"],
    )
    return manifest


def normalize_article(record: Mapping[str, Any]) -> dict[str, Any]:
    article_id = record.get("article_id")
    if not isinstance(article_id, str) or not article_id:
        raise ValueError("Selected article_id is required")
    headline = record.get("headline")
    description = record.get("description")
    if not isinstance(headline, str) or not headline.strip():
        raise ValueError("Selected article headline is required")
    if not isinstance(description, str):
        raise TypeError("Selected article description must be a string")
    expected_text = (
        headline if not description else f"{headline}\n\n{description}"
    )
    if record.get("model_text") != expected_text:
        raise ValueError("Selected article model_text changed")
    expected_hashes = {
        "headline_sha256": sha256_text(headline),
        "description_sha256": sha256_text(description),
        "model_text_sha256": sha256_text(expected_text),
    }
    for name, value in expected_hashes.items():
        if record.get(name) != value:
            raise ValueError(f"Selected article {name} changed")
    if record.get("source_profile") != SOURCE_PROFILE:
        raise ValueError("Selected article source profile changed")
    description_available = record.get("description_available")
    if not isinstance(description_available, bool):
        raise TypeError("description_available must be boolean")
    if description_available is not bool(description):
        raise ValueError("description_available differs from description")
    retained_count = record.get("retained_description_char_count")
    if (
        isinstance(retained_count, bool)
        or not isinstance(retained_count, int)
        or retained_count != len(description)
    ):
        raise ValueError("retained_description_char_count changed")
    normalized = {
        "record_version": "flan-w17-lite-selected-article-v1",
        "article_id": article_id,
        "forecast_date": str(record.get("forecast_date")),
        "published_at_utc": str(record.get("published_at_utc")),
        "cutoff_utc": str(record.get("cutoff_utc")),
        "headline": headline,
        "description": description,
        "description_available": description_available,
        "retained_description_char_count": retained_count,
        "description_was_bounded": bool(record.get("description_was_bounded")),
        "model_text": expected_text,
        **expected_hashes,
        "source_profile": SOURCE_PROFILE,
    }
    published = datetime.fromisoformat(
        normalized["published_at_utc"].replace("Z", "+00:00")
    )
    cutoff = datetime.fromisoformat(
        normalized["cutoff_utc"].replace("Z", "+00:00")
    )
    if (
        published.tzinfo is None
        or cutoff.tzinfo is None
        or not published < cutoff
    ):
        raise ValueError("Selected article timestamp contract changed")
    expected_input_sha = sha256_json(normalized)
    if record.get("article_input_sha256") != expected_input_sha:
        raise ValueError("Selected article input hash changed")
    normalized["article_input_sha256"] = expected_input_sha
    return normalized


def load_article_queue(path: Path) -> list[dict[str, Any]]:
    articles = [normalize_article(record) for record in iter_jsonl(path)]
    ids = [record["article_id"] for record in articles]
    if not articles:
        raise ValueError("W17-Lite article queue is empty")
    if len(ids) != len(set(ids)):
        raise ValueError("W17-Lite article queue contains duplicate IDs")
    if ids != _utf8_sorted(ids):
        raise ValueError("W17-Lite article queue is not UTF-8 ID sorted")
    return articles


def extractor_record(article: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt one target-invariant article to the existing prompt renderer."""

    return {
        "article_id": article["article_id"],
        "headline": article["headline"],
        "article_text": article["description"],
        "target": {
            "ticker": "",
            "company": "",
            "sector": "",
            "sector_benchmark": "",
            "known_sector_peers": [],
        },
    }


def event_prompts(
    article: Mapping[str, Any], schema: Mapping[str, Any]
) -> tuple[str, str]:
    validate_event_schema(schema)
    record = extractor_record(article)
    canonical, _ = coarse_runner.build_letter_prompt(
        record,
        dict(schema),
        EVENT_FIELD,
        list(EVENT_GROUP_LABELS),
        None,
        "zero_shot",
    )
    reversed_prompt, _ = coarse_runner.build_letter_prompt(
        record,
        dict(schema),
        EVENT_FIELD,
        list(reversed(EVENT_GROUP_LABELS)),
        None,
        "zero_shot",
    )
    return canonical, reversed_prompt


def _package_versions() -> dict[str, str | None]:
    values: dict[str, str | None] = {}
    for package in (
        "numpy",
        "torch",
        "transformers",
        "tokenizers",
        "sentencepiece",
        "safetensors",
        "accelerate",
        "psutil",
    ):
        try:
            values[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            values[package] = None
    return values


def runtime_contract(
    *,
    schema_path: Path = DEFAULT_SCHEMA,
    design_path: Path = DEFAULT_DESIGN,
) -> dict[str, Any]:
    schema = load_event_schema(schema_path)
    snapshot_manifest = xl_runner.validate_snapshot_manifest()
    recorded_files = snapshot_manifest.get("files")
    if not isinstance(recorded_files, Mapping) or not recorded_files:
        raise ValueError("Pinned FLAN snapshot file records are missing")
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
        "contract_id": CONTRACT_ID,
        "arm_id": ARM_ID,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "maximum_input_tokens": MAX_INPUT_TOKENS,
        "selector_k": SELECTOR_K,
        "event_group_labels": list(EVENT_GROUP_LABELS),
        "prompt_profile": "zero_shot",
        "decoding": "canonical_and_reversed_order_averaged_letter_score",
        "acceptance": {
            "requires_order_agreement": True,
            "abstention_label": ABSTENTION_LABEL,
            "calibration_used": False,
        },
        "schema": schema,
        "schema_path": str(schema_path.resolve()),
        "schema_sha256": sha256_file(schema_path),
        "design_path": str(design_path.resolve()),
        "design_sha256": sha256_file(design_path),
        "snapshot_manifest_path": str(
            xl_runner.SNAPSHOT_MANIFEST.resolve()
        ),
        "snapshot_manifest_sha256": sha256_file(
            xl_runner.SNAPSHOT_MANIFEST
        ),
        "snapshot_recorded_files": copy.deepcopy(dict(recorded_files)),
        "prompt_runner_sha256": sha256_file(
            Path(coarse_runner.__file__).resolve()
        ),
        "score_runner_sha256": sha256_file(Path(xl_runner.__file__).resolve()),
        "base_runner_sha256": sha256_file(Path(flan_base.__file__).resolve()),
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
        "execution_environment": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "packages": _package_versions(),
            "cuda": cuda,
        },
    }


def runtime_contract_sha256() -> str:
    return sha256_json(runtime_contract())


def _tokenizer_only() -> tuple[Any, dict[str, Any]]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Tokenizer preflight requires .venv-flan-t5-xl"
        ) from error
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
        "snapshot_path": str(snapshot.resolve()),
        "tokenizer_files_sha256": tokenizer_files,
    }


def _token_lengths(tokenizer: Any, prompts: Sequence[str]) -> list[int]:
    encoded = tokenizer(
        list(prompts),
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )
    return [len(value) for value in encoded["input_ids"]]


def validate_preflight_manifest(
    path: Path = DEFAULT_PREFLIGHT,
    *,
    queue_path: Path = DEFAULT_QUEUE,
    selection_manifest_path: Path = DEFAULT_SELECTION_MANIFEST,
    require_current_runtime: bool = True,
) -> dict[str, Any]:
    value = _load_json(path)
    if value.get("manifest_version") != "flan-w17-lite-token-preflight-v1":
        raise ValueError("Unsupported W17-Lite preflight manifest")
    if value.get("status") != "passed" or value.get("violation_count") != 0:
        raise ValueError("W17-Lite tokenizer preflight did not pass")
    selection = validate_selection_manifest(
        selection_manifest_path,
        queue_path=queue_path,
        require_current_implementation=require_current_runtime,
    )
    if value.get("selection_manifest_sha256") != sha256_file(
        selection_manifest_path
    ):
        raise ValueError("W17-Lite selection differs from preflight")
    if value.get("queue_sha256") != sha256_file(queue_path):
        raise ValueError("W17-Lite article queue differs from preflight")
    audit = selection["audit"]
    if value.get("article_count") != audit["selected_unique_articles"]:
        raise ValueError("W17-Lite preflight article count differs")
    if (
        value.get("ordered_article_ids_sha256")
        != audit["ordered_selected_article_ids_sha256"]
    ):
        raise ValueError("W17-Lite preflight article-ID hash differs")
    if require_current_runtime:
        current = runtime_contract()
        if value.get("runtime_contract_sha256") != sha256_json(current):
            raise ValueError("W17-Lite runtime differs from preflight")
    return value


def preflight_corpus(
    *,
    queue_path: Path = DEFAULT_QUEUE,
    selection_manifest_path: Path = DEFAULT_SELECTION_MANIFEST,
    output_path: Path = DEFAULT_PREFLIGHT,
    prompt_batch_size: int = 512,
) -> dict[str, Any]:
    """Tokenize every actual canonical/reversed prompt without loading FLAN."""

    if prompt_batch_size < 2:
        raise ValueError("prompt_batch_size must be at least two")
    selection = validate_selection_manifest(
        selection_manifest_path, queue_path=queue_path
    )
    if output_path.exists():
        return validate_preflight_manifest(
            output_path,
            queue_path=queue_path,
            selection_manifest_path=selection_manifest_path,
        )
    startup_queue_sha = sha256_file(queue_path)
    startup_selection_sha = sha256_file(selection_manifest_path)
    startup_runtime = runtime_contract()
    startup_runtime_sha = sha256_json(startup_runtime)
    articles = load_article_queue(queue_path)
    schema = load_event_schema()
    tokenizer, tokenizer_record = _tokenizer_only()
    prompts: list[str] = []
    metadata: list[tuple[str, str]] = []
    maximum_tokens = 0
    maximum_record: dict[str, Any] | None = None
    violations: list[dict[str, Any]] = []
    tokenized_prompt_count = 0

    def flush() -> None:
        nonlocal maximum_tokens, maximum_record, tokenized_prompt_count
        if not prompts:
            return
        lengths = _token_lengths(tokenizer, prompts)
        if len(lengths) != len(metadata):
            raise AssertionError("Tokenizer output length differs")
        tokenized_prompt_count += len(lengths)
        for tokens, (article_id, variant) in zip(lengths, metadata):
            if tokens > maximum_tokens:
                maximum_tokens = tokens
                maximum_record = {
                    "article_id": article_id,
                    "variant": variant,
                    "tokens": tokens,
                }
            if tokens > MAX_INPUT_TOKENS:
                violations.append(
                    {
                        "article_id": article_id,
                        "variant": variant,
                        "tokens": tokens,
                        "limit": MAX_INPUT_TOKENS,
                    }
                )
        prompts.clear()
        metadata.clear()

    for article in articles:
        canonical, reversed_prompt = event_prompts(article, schema)
        prompts.extend((canonical, reversed_prompt))
        metadata.extend(
            (
                (article["article_id"], "canonical"),
                (article["article_id"], "reversed"),
            )
        )
        if len(prompts) >= prompt_batch_size:
            flush()
    flush()
    if sha256_file(queue_path) != startup_queue_sha:
        raise ValueError("W17-Lite queue changed during tokenizer preflight")
    if sha256_file(selection_manifest_path) != startup_selection_sha:
        raise ValueError("W17-Lite selection changed during tokenizer preflight")
    if runtime_contract() != startup_runtime:
        raise ValueError("W17-Lite runtime changed during tokenizer preflight")
    violations_path = output_path.with_suffix(".violations.jsonl")
    _atomic_jsonl(violations_path, violations)
    ids = [record["article_id"] for record in articles]
    manifest = {
        "manifest_version": "flan-w17-lite-token-preflight-v1",
        "pipeline_version": PIPELINE_VERSION,
        "status": "passed" if not violations else "failed",
        "generated_at_utc": utc_now(),
        "queue_path": str(queue_path.resolve()),
        "queue_sha256": startup_queue_sha,
        "selection_manifest_path": str(selection_manifest_path.resolve()),
        "selection_manifest_sha256": startup_selection_sha,
        "article_count": len(articles),
        "ordered_article_ids_sha256": _ordered_lf_hash(ids),
        "prompt_count": len(articles) * 2,
        "tokenized_prompt_count": tokenized_prompt_count,
        "maximum_input_tokens": MAX_INPUT_TOKENS,
        "maximum_observed_tokens": maximum_tokens,
        "maximum_record": maximum_record,
        "violation_count": len(violations),
        "violations": {
            "path": str(violations_path.resolve()),
            "sha256": sha256_file(violations_path),
            "rows": len(violations),
        },
        "prompt_variants": ["canonical", "reversed"],
        "silent_truncation_allowed": False,
        "model_loaded": False,
        "tokenizer": tokenizer_record,
        "runtime_contract": startup_runtime,
        "runtime_contract_sha256": startup_runtime_sha,
        "source_selection": selection["audit"],
    }
    _atomic_json(output_path, manifest)
    if not violations:
        validate_preflight_manifest(
            output_path,
            queue_path=queue_path,
            selection_manifest_path=selection_manifest_path,
        )
    return manifest


class PromptTooLong(RuntimeError):
    def __init__(self, token_count: int) -> None:
        self.token_count = token_count
        super().__init__(
            f"W17-Lite prompt has {token_count} tokens; limit is "
            f"{MAX_INPUT_TOKENS}"
        )


class W17LiteEngine(Protocol):
    def predict(self, article: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return one valid target-invariant event result."""


def _best_label(scores: Mapping[str, float]) -> str:
    return max(EVENT_GROUP_LABELS, key=lambda label: float(scores[label]))


def _score_margin(scores: Mapping[str, float]) -> float:
    values = sorted((float(value) for value in scores.values()), reverse=True)
    return values[0] - values[1]


def _validate_score_map(value: Any, name: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(EVENT_GROUP_LABELS):
        raise ValueError(f"{name} does not cover the event ontology")
    output = {label: float(value[label]) for label in EVENT_GROUP_LABELS}
    if not all(math.isfinite(score) for score in output.values()):
        raise ValueError(f"{name} contains nonfinite scores")
    return output


class FlanT5XlW17LiteEngine:
    """Pinned CUDA FP16 scorer for the one W17-Lite event field."""

    def __init__(self, *, max_input_tokens: int = MAX_INPUT_TOKENS) -> None:
        self.max_input_tokens = max_input_tokens
        self.schema = load_event_schema()
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            import torch
            import transformers
            from transformers import (
                AutoModelForSeq2SeqLM,
                AutoTokenizer,
                set_seed,
            )
        except ImportError as error:
            raise RuntimeError(
                "Install requirements-flan-t5-xl.txt and a CUDA PyTorch build"
            ) from error
        manifest = xl_runner.validate_snapshot_manifest()
        snapshot = Path(manifest["snapshot_path"])
        self.snapshot_hashes = xl_runner.sharded_snapshot_file_hashes(snapshot)
        if not torch.cuda.is_available():
            raise RuntimeError("W17-Lite extraction requires CUDA")
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

    def diagnostics(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "dtype": str(self.dtype),
            "transformers_version": self.transformers_version,
            "snapshot_files_sha256": self.snapshot_hashes,
            "model_inputs_per_article": 2,
            "candidate_sequences_per_article": 8,
        }

    def predict(self, article: Mapping[str, Any]) -> Mapping[str, Any]:
        canonical_prompt, reversed_prompt = event_prompts(
            article, self.schema
        )
        lengths = _token_lengths(
            self.tokenizer, (canonical_prompt, reversed_prompt)
        )
        maximum = max(lengths)
        if maximum > self.max_input_tokens:
            raise PromptTooLong(maximum)
        state = {
            "record": extractor_record(article),
            "labels": {"shock_scope": None},
        }
        original_scorer = coarse_runner.base.score_closed_label_batch
        coarse_runner.base.score_closed_label_batch = (
            xl_runner.low_memory_score_closed_label_batch
        )
        try:
            result = coarse_runner.classify_active_batch(
                field=EVENT_FIELD,
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
        finally:
            coarse_runner.base.score_closed_label_batch = original_scorer
        if result.get("input_truncated") is not False:
            raise RuntimeError("W17-Lite scorer attempted input truncation")
        canonical_scores = _validate_score_map(
            result.get("canonical_candidate_mean_log_probabilities"),
            "canonical scores",
        )
        reversed_scores = _validate_score_map(
            result.get("reversed_candidate_mean_log_probabilities"),
            "reversed scores",
        )
        averaged_scores = _validate_score_map(
            result.get("order_averaged_mean_log_probabilities"),
            "averaged scores",
        )
        canonical_prediction = _best_label(canonical_scores)
        reversed_prediction = _best_label(reversed_scores)
        averaged_prediction = _best_label(averaged_scores)
        agreement = canonical_prediction == reversed_prediction
        accepted = agreement and canonical_prediction != ABSTENTION_LABEL
        if accepted:
            reason = "accepted_order_agreement"
            accepted_label: str | None = canonical_prediction
        elif agreement:
            reason = "agreed_other_or_unclear"
            accepted_label = None
        else:
            reason = "order_disagreement"
            accepted_label = None
        return {
            "terminal_state": "complete",
            "no_input_truncation": True,
            "event_group": {
                "canonical_prediction": canonical_prediction,
                "reversed_prediction": reversed_prediction,
                "order_averaged_prediction": averaged_prediction,
                "canonical_candidate_mean_log_probabilities": (
                    canonical_scores
                ),
                "reversed_candidate_mean_log_probabilities": reversed_scores,
                "order_averaged_mean_log_probabilities": averaged_scores,
                "order_averaged_top1_top2_margin": _score_margin(
                    averaged_scores
                ),
                "order_agreement": agreement,
                "accepted": accepted,
                "accepted_label": accepted_label,
                "acceptance_reason": reason,
                "prompt_sha256": result["prompt_sha256"],
                "reversed_prompt_sha256": result[
                    "reversed_prompt_sha256"
                ],
                "input_tokens_canonical": lengths[0],
                "input_tokens_reversed": lengths[1],
                "input_tokens_max": maximum,
                "input_truncated": False,
                "calibration_used": False,
            },
        }


def _run_id(
    *,
    queue_sha256: str,
    selection_manifest_sha256: str,
    preflight_manifest_sha256: str,
    runtime_contract_sha256_value: str,
    ordered_article_ids_sha256: str,
) -> str:
    return sha256_json(
        {
            "pipeline_version": PIPELINE_VERSION,
            "contract_id": CONTRACT_ID,
            "arm_id": ARM_ID,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "selector_k": SELECTOR_K,
            "queue_sha256": queue_sha256,
            "selection_manifest_sha256": selection_manifest_sha256,
            "preflight_manifest_sha256": preflight_manifest_sha256,
            "runtime_contract_sha256": runtime_contract_sha256_value,
            "ordered_article_ids_sha256": ordered_article_ids_sha256,
            "schema_sha256": sha256_file(DEFAULT_SCHEMA),
            "shard_count": 1,
        }
    )


def _terminal_base(
    article: Mapping[str, Any],
    *,
    run_id: str,
    queue_sha256: str,
    selection_manifest_sha256: str,
    preflight_manifest_sha256: str,
    runtime_contract_sha256_value: str,
    attempt: int,
) -> dict[str, Any]:
    return {
        "record_version": "flan-w17-lite-terminal-v1",
        "run_id": run_id,
        "article_id": article["article_id"],
        "article_input_sha256": article["article_input_sha256"],
        "headline_sha256": article["headline_sha256"],
        "description_sha256": article["description_sha256"],
        "model_text_sha256": article["model_text_sha256"],
        "source_profile": article["source_profile"],
        "arm_id": ARM_ID,
        "contract_id": CONTRACT_ID,
        "extractor_id": "flan-t5-xl-w17-lite-v1",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "selector_k": SELECTOR_K,
        "queue_sha256": queue_sha256,
        "selection_manifest_sha256": selection_manifest_sha256,
        "preflight_manifest_sha256": preflight_manifest_sha256,
        "runtime_contract_sha256": runtime_contract_sha256_value,
        "schema_sha256": sha256_file(DEFAULT_SCHEMA),
        "attempt": attempt,
    }


def terminal_prediction(
    article: Mapping[str, Any],
    engine: W17LiteEngine,
    *,
    run_id: str,
    queue_sha256: str,
    selection_manifest_sha256: str,
    preflight_manifest_sha256: str,
    runtime_contract_sha256_value: str,
    attempt: int,
) -> dict[str, Any]:
    base = _terminal_base(
        article,
        run_id=run_id,
        queue_sha256=queue_sha256,
        selection_manifest_sha256=selection_manifest_sha256,
        preflight_manifest_sha256=preflight_manifest_sha256,
        runtime_contract_sha256_value=runtime_contract_sha256_value,
        attempt=attempt,
    )
    try:
        result = dict(engine.predict(article))
    except PromptTooLong as error:
        return {
            **base,
            "terminal_state": "prompt_too_long",
            "failure_class": type(error).__name__,
            "failure_reason": str(error),
            "input_tokens_max": error.token_count,
            "no_input_truncation": True,
            "event_group": None,
        }
    except Exception as error:
        return {
            **base,
            "terminal_state": "inference_failed",
            "failure_class": type(error).__name__,
            "failure_reason": str(error)[:2_000],
            "no_input_truncation": True,
            "event_group": None,
        }
    terminal = {**base, **result}
    if set(base) & (set(result) - {"terminal_state"}):
        return {
            **base,
            "terminal_state": "invalid_output",
            "failure_class": "ContractOverride",
            "failure_reason": (
                "Engine output attempted to override immutable terminal fields"
            ),
            "no_input_truncation": True,
            "event_group": None,
        }
    try:
        validate_terminal_record(terminal)
    except Exception as error:
        return {
            **base,
            "terminal_state": "invalid_output",
            "failure_class": type(error).__name__,
            "failure_reason": str(error)[:2_000],
            "no_input_truncation": True,
            "event_group": None,
        }
    return terminal


def validate_terminal_record(record: Mapping[str, Any]) -> None:
    if record.get("record_version") != "flan-w17-lite-terminal-v1":
        raise ValueError("Unsupported W17-Lite terminal record")
    state = record.get("terminal_state")
    if state not in ALL_TERMINAL_STATES:
        raise ValueError("Invalid W17-Lite terminal state")
    for name in (
        "run_id",
        "article_id",
        "article_input_sha256",
        "queue_sha256",
        "selection_manifest_sha256",
        "preflight_manifest_sha256",
        "runtime_contract_sha256",
        "schema_sha256",
    ):
        value = record.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"W17-Lite terminal {name} is invalid")
    if (
        record.get("arm_id") != ARM_ID
        or record.get("contract_id") != CONTRACT_ID
        or record.get("model_id") != MODEL_ID
        or record.get("model_revision") != MODEL_REVISION
        or record.get("selector_k") != SELECTOR_K
        or record.get("source_profile") != SOURCE_PROFILE
        or record.get("no_input_truncation") is not True
    ):
        raise ValueError("W17-Lite terminal contract differs")
    attempt = record.get("attempt")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise ValueError("W17-Lite terminal attempt is invalid")
    if state != "complete":
        if record.get("event_group") is not None:
            raise ValueError("Failed W17-Lite result must be fail-closed")
        return
    field = record.get("event_group")
    if not isinstance(field, Mapping):
        raise ValueError("Complete W17-Lite result lacks event_group")
    canonical = _validate_score_map(
        field.get("canonical_candidate_mean_log_probabilities"),
        "canonical scores",
    )
    reversed_scores = _validate_score_map(
        field.get("reversed_candidate_mean_log_probabilities"),
        "reversed scores",
    )
    averaged = _validate_score_map(
        field.get("order_averaged_mean_log_probabilities"),
        "averaged scores",
    )
    expected_average = {
        label: (canonical[label] + reversed_scores[label]) / 2.0
        for label in EVENT_GROUP_LABELS
    }
    for label in EVENT_GROUP_LABELS:
        if not math.isclose(
            averaged[label],
            expected_average[label],
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError("W17-Lite averaged score map is inconsistent")
    canonical_prediction = _best_label(canonical)
    reversed_prediction = _best_label(reversed_scores)
    averaged_prediction = _best_label(averaged)
    if (
        field.get("canonical_prediction") != canonical_prediction
        or field.get("reversed_prediction") != reversed_prediction
        or field.get("order_averaged_prediction") != averaged_prediction
    ):
        raise ValueError("W17-Lite prediction differs from score-map argmax")
    agreement = canonical_prediction == reversed_prediction
    accepted = agreement and canonical_prediction != ABSTENTION_LABEL
    accepted_label = canonical_prediction if accepted else None
    reason = (
        "accepted_order_agreement"
        if accepted
        else (
            "agreed_other_or_unclear"
            if agreement
            else "order_disagreement"
        )
    )
    if (
        field.get("order_agreement") is not agreement
        or field.get("accepted") is not accepted
        or field.get("accepted_label") != accepted_label
        or field.get("acceptance_reason") != reason
        or field.get("input_truncated") is not False
        or field.get("calibration_used") is not False
    ):
        raise ValueError("W17-Lite acceptance rule is inconsistent")
    for name in ("prompt_sha256", "reversed_prompt_sha256"):
        value = field.get(name)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"W17-Lite {name} is invalid")
    expected_margin = _score_margin(averaged)
    if not math.isclose(
        float(field.get("order_averaged_top1_top2_margin")),
        expected_margin,
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise ValueError("W17-Lite score margin is inconsistent")
    token_values = []
    for name in (
        "input_tokens_canonical",
        "input_tokens_reversed",
        "input_tokens_max",
    ):
        value = field.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 < value <= MAX_INPUT_TOKENS
        ):
            raise ValueError(f"W17-Lite {name} is invalid")
        token_values.append(value)
    if token_values[2] != max(token_values[:2]):
        raise ValueError("W17-Lite maximum token count is inconsistent")


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


def _process_identity() -> dict[str, Any]:
    identity: dict[str, Any] = {
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "started_at_utc": utc_now(),
    }
    try:
        import psutil
    except ImportError:
        identity["process_create_time"] = None
    else:
        identity["process_create_time"] = psutil.Process().create_time()
    return identity


def _lock_owner_is_live(value: Mapping[str, Any]) -> bool:
    if value.get("hostname") != socket.gethostname():
        return True
    pid = value.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid < 1:
        return False
    try:
        import psutil
    except ImportError:
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True
    try:
        process = psutil.Process(pid)
        if not process.is_running():
            return False
        expected = value.get("process_create_time")
        if expected is not None and not math.isclose(
            float(expected),
            float(process.create_time()),
            rel_tol=0,
            abs_tol=1.0,
        ):
            return False
        return True
    except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError, TypeError):
        return False


@contextmanager
def inference_lock(
    path: Path,
    *,
    run_id: str,
    recover_stale: bool,
) -> Iterator[dict[str, Any]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        owner = {**_process_identity(), "run_id": run_id}
        try:
            descriptor = os.open(
                path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        except FileExistsError:
            prior = None
            for attempt in range(20):
                try:
                    prior = _load_json(path)
                    break
                except Exception:
                    if attempt < 19:
                        time.sleep(0.05)
            if isinstance(prior, Mapping) and _lock_owner_is_live(prior):
                raise RuntimeError(
                    "Another W17-Lite inference process holds the lock: "
                    f"{canonical_json(prior)}"
                )
            if prior is not None and prior.get("hostname") != socket.gethostname():
                if not recover_stale:
                    raise RuntimeError(
                        "A stale cross-host W17-Lite lock requires "
                        "--recover-stale-lock"
                    )
            elif prior is None and not recover_stale:
                raise RuntimeError(
                    "An unreadable W17-Lite lock requires "
                    "--recover-stale-lock"
                )
            path.unlink(missing_ok=True)
            continue
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(owner, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        break
    try:
        yield owner
    finally:
        try:
            current = _load_json(path)
        except Exception:
            current = None
        if isinstance(current, Mapping) and current.get("pid") == os.getpid():
            path.unlink(missing_ok=True)


class StopRequest:
    def __init__(self) -> None:
        self.requested = False
        self.signal_number: int | None = None

    def handler(self, signum: int, _frame: Any) -> None:
        self.requested = True
        self.signal_number = signum
        print(
            "\nStop requested; W17-Lite will checkpoint after the current "
            "article.",
            flush=True,
        )


@contextmanager
def stop_handlers(request: StopRequest) -> Iterator[None]:
    prior: dict[int, Any] = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            prior[signum] = signal.getsignal(signum)
            signal.signal(signum, request.handler)
        except (AttributeError, ValueError):
            pass
    try:
        yield
    finally:
        for signum, handler in prior.items():
            signal.signal(signum, handler)


def _output_companion(path: Path, suffix: str) -> Path:
    return path.with_suffix(path.suffix + suffix)


def _record_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return dict(
        sorted(Counter(str(record["terminal_state"]) for record in records).items())
    )


def _state_record(
    *,
    status: str,
    run_id: str,
    total_articles: int,
    records: Sequence[Mapping[str, Any]],
    output_path: Path,
    started_at_utc: str,
    new_articles_this_process: int,
    signal_number: int | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    states = _record_counts(records)
    successful = int(states.get("complete", 0))
    failures = int(
        sum(states.get(name, 0) for name in FAILURE_TERMINAL_STATES)
    )
    return {
        "state_version": "flan-w17-lite-run-state-v1",
        "status": status,
        "updated_at_utc": utc_now(),
        "started_at_utc": started_at_utc,
        "run_id": run_id,
        "output_path": str(output_path.resolve()),
        "output_bytes": output_path.stat().st_size if output_path.exists() else 0,
        "output_sha256": None,
        "output_sha256_note": (
            "Advisory state avoids repeatedly hashing the growing ledger; "
            "the inference manifest hashes it at every clean process exit."
        ),
        "selected_article_count": total_articles,
        "terminal_record_count": len(records),
        "successful_article_count": successful,
        "failure_article_count": failures,
        "remaining_article_count": total_articles - len(records),
        "new_articles_this_process": new_articles_this_process,
        "terminal_state_counts": states,
        "signal_number": signal_number,
        "message": message,
    }


def _inference_manifest(
    *,
    status: str,
    run_id: str,
    output_path: Path,
    queue_path: Path,
    selection_manifest_path: Path,
    preflight_manifest_path: Path,
    records: Sequence[Mapping[str, Any]],
    articles: Sequence[Mapping[str, Any]],
    runtime: Mapping[str, Any],
    tail_recovery: Mapping[str, Any],
    started_at_utc: str,
    new_articles_this_process: int,
    max_new_articles: int | None,
    engine_diagnostics: Mapping[str, Any] | None,
    signal_number: int | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    states = _record_counts(records)
    successful = int(states.get("complete", 0))
    failures = int(
        sum(states.get(name, 0) for name in FAILURE_TERMINAL_STATES)
    )
    return {
        "manifest_version": "flan-w17-lite-inference-manifest-v1",
        "pipeline_version": PIPELINE_VERSION,
        "status": status,
        "generated_at_utc": utc_now(),
        "started_at_utc": started_at_utc,
        "run_id": run_id,
        "arm_id": ARM_ID,
        "contract_id": CONTRACT_ID,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "selector_k": SELECTOR_K,
        "queue_path": str(queue_path.resolve()),
        "queue_sha256": sha256_file(queue_path),
        "selection_manifest_path": str(selection_manifest_path.resolve()),
        "selection_manifest_sha256": sha256_file(selection_manifest_path),
        "preflight_manifest_path": str(preflight_manifest_path.resolve()),
        "preflight_manifest_sha256": sha256_file(preflight_manifest_path),
        "output_path": str(output_path.resolve()),
        "output_sha256": (
            sha256_file(output_path) if output_path.exists() else None
        ),
        "output_bytes": output_path.stat().st_size if output_path.exists() else 0,
        "selected_article_count": len(articles),
        "ordered_selected_article_ids_sha256": _ordered_lf_hash(
            [article["article_id"] for article in articles]
        ),
        "terminal_record_count": len(records),
        "terminal_state_counts": states,
        "successful_article_count": successful,
        "failure_article_count": failures,
        "remaining_article_count": len(articles) - len(records),
        "all_selected_articles_terminal": len(records) == len(articles),
        "full_corpus_complete": (
            len(records) == len(articles) and failures == 0
        ),
        "append_tail_recovery": dict(tail_recovery),
        "new_articles_this_process": new_articles_this_process,
        "max_new_articles": max_new_articles,
        "per_article_flush_and_fsync": True,
        "runtime_contract": dict(runtime),
        "runtime_contract_sha256": sha256_json(runtime),
        "engine_diagnostics": (
            dict(engine_diagnostics) if engine_diagnostics is not None else None
        ),
        "signal_number": signal_number,
        "message": message,
        "claim_flags": {
            "point_in_time_version_safe": False,
            "primary_or_confirmatory_eligible": False,
            "exploratory_construction_eligible": True,
        },
    }


def _validate_record_provenance(
    record: Mapping[str, Any],
    article: Mapping[str, Any],
    *,
    run_id: str,
    queue_sha256: str,
    selection_manifest_sha256: str,
    preflight_manifest_sha256: str,
    runtime_contract_sha256_value: str,
) -> None:
    validate_terminal_record(record)
    if (
        record.get("run_id") != run_id
        or record.get("article_id") != article["article_id"]
        or record.get("article_input_sha256")
        != article["article_input_sha256"]
        or record.get("headline_sha256") != article["headline_sha256"]
        or record.get("description_sha256")
        != article["description_sha256"]
        or record.get("model_text_sha256")
        != article["model_text_sha256"]
        or record.get("queue_sha256") != queue_sha256
        or record.get("selection_manifest_sha256")
        != selection_manifest_sha256
        or record.get("preflight_manifest_sha256")
        != preflight_manifest_sha256
        or record.get("runtime_contract_sha256")
        != runtime_contract_sha256_value
        or record.get("schema_sha256") != sha256_file(DEFAULT_SCHEMA)
    ):
        raise ValueError(
            f"Existing W17-Lite record provenance differs for "
            f"{article['article_id']}"
        )
    if record["terminal_state"] == "complete":
        canonical_prompt, reversed_prompt = event_prompts(
            article, load_event_schema()
        )
        field = record["event_group"]
        if (
            field.get("prompt_sha256") != sha256_text(canonical_prompt)
            or field.get("reversed_prompt_sha256")
            != sha256_text(reversed_prompt)
        ):
            raise ValueError(
                f"Existing W17-Lite prompt hashes differ for "
                f"{article['article_id']}"
            )


def _read_existing_terminal_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return list(iter_jsonl(path))


def _validate_complete_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    output_path: Path,
    run_id: str,
    expected_count: int,
) -> None:
    if manifest.get("status") != "complete":
        return
    if manifest.get("run_id") != run_id:
        raise ValueError("Completed W17-Lite manifest run identity changed")
    if not output_path.is_file():
        raise FileNotFoundError(output_path)
    if manifest.get("output_sha256") != sha256_file(output_path):
        raise ValueError(
            "Completed W17-Lite output hash differs; refusing to rebless it"
        )
    if (
        manifest.get("selected_article_count") != expected_count
        or manifest.get("terminal_record_count") != expected_count
        or manifest.get("successful_article_count") != expected_count
        or manifest.get("failure_article_count") != 0
        or manifest.get("full_corpus_complete") is not True
    ):
        raise ValueError("Completed W17-Lite manifest is internally incomplete")


def run_inference(
    *,
    queue_path: Path = DEFAULT_QUEUE,
    selection_manifest_path: Path = DEFAULT_SELECTION_MANIFEST,
    preflight_manifest_path: Path = DEFAULT_PREFLIGHT,
    output_path: Path = DEFAULT_PREDICTIONS,
    engine: W17LiteEngine | None = None,
    engine_factory: Callable[[], W17LiteEngine] | None = None,
    max_new_articles: int | None = None,
    retry_failed: bool = True,
    recover_stale_lock: bool = False,
    max_consecutive_failures: int = SYSTEMIC_FAILURE_ABORT_THRESHOLD,
) -> dict[str, Any]:
    """Resume inference from the append-only authoritative terminal ledger."""

    if "".join(output_path.suffixes).lower() != ".jsonl":
        raise ValueError("W17-Lite predictions must be uncompressed JSONL")
    if max_new_articles is not None and max_new_articles < 1:
        raise ValueError("max_new_articles must be positive")
    if max_consecutive_failures < 1:
        raise ValueError("max_consecutive_failures must be positive")
    selection = validate_selection_manifest(
        selection_manifest_path, queue_path=queue_path
    )
    preflight = validate_preflight_manifest(
        preflight_manifest_path,
        queue_path=queue_path,
        selection_manifest_path=selection_manifest_path,
    )
    articles = load_article_queue(queue_path)
    expected_count = int(selection["audit"]["selected_unique_articles"])
    if len(articles) != expected_count:
        raise ValueError("W17-Lite queue count differs from selection")
    queue_sha = sha256_file(queue_path)
    selection_sha = sha256_file(selection_manifest_path)
    preflight_sha = sha256_file(preflight_manifest_path)
    startup_runtime = runtime_contract()
    runtime_sha = sha256_json(startup_runtime)
    if preflight.get("runtime_contract_sha256") != runtime_sha:
        raise ValueError("W17-Lite preflight/runtime contract differs")
    run_id = _run_id(
        queue_sha256=queue_sha,
        selection_manifest_sha256=selection_sha,
        preflight_manifest_sha256=preflight_sha,
        runtime_contract_sha256_value=runtime_sha,
        ordered_article_ids_sha256=selection["audit"][
            "ordered_selected_article_ids_sha256"
        ],
    )
    manifest_path = _output_companion(output_path, ".manifest.json")
    state_path = _output_companion(output_path, ".state.json")
    lock_path = _output_companion(output_path, ".lock")
    started_at = utc_now()
    with inference_lock(
        lock_path, run_id=run_id, recover_stale=recover_stale_lock
    ):
        if manifest_path.exists():
            prior_manifest = _load_json(manifest_path)
            _validate_complete_manifest(
                prior_manifest,
                manifest_path=manifest_path,
                output_path=output_path,
                run_id=run_id,
                expected_count=expected_count,
            )
            if prior_manifest.get("status") == "complete":
                existing = _read_existing_terminal_records(output_path)
                by_id = {
                    article["article_id"]: article for article in articles
                }
                if len(existing) != expected_count:
                    raise ValueError(
                        "Completed W17-Lite output row count changed"
                    )
                seen: set[str] = set()
                for record in existing:
                    article_id = str(record.get("article_id"))
                    if article_id in seen or article_id not in by_id:
                        raise ValueError(
                            "Completed W17-Lite output IDs changed"
                        )
                    seen.add(article_id)
                    _validate_record_provenance(
                        record,
                        by_id[article_id],
                        run_id=run_id,
                        queue_sha256=queue_sha,
                        selection_manifest_sha256=selection_sha,
                        preflight_manifest_sha256=preflight_sha,
                        runtime_contract_sha256_value=runtime_sha,
                    )
                return prior_manifest

        tail_recovery = recover_append_tail(output_path)
        existing = _read_existing_terminal_records(output_path)
        by_id = {article["article_id"]: article for article in articles}
        retained: list[dict[str, Any]] = []
        seen: set[str] = set()
        attempts: Counter[str] = Counter()
        for record in existing:
            article_id = str(record.get("article_id"))
            if article_id in seen:
                raise ValueError(
                    "W17-Lite output contains duplicate article IDs"
                )
            seen.add(article_id)
            article = by_id.get(article_id)
            if article is None:
                raise ValueError(
                    "W17-Lite output contains an out-of-selection article"
                )
            _validate_record_provenance(
                record,
                article,
                run_id=run_id,
                queue_sha256=queue_sha,
                selection_manifest_sha256=selection_sha,
                preflight_manifest_sha256=preflight_sha,
                runtime_contract_sha256_value=runtime_sha,
            )
            attempts[article_id] = max(
                attempts[article_id], int(record["attempt"])
            )
            if retry_failed and record["terminal_state"] in FAILURE_TERMINAL_STATES:
                continue
            retained.append(dict(record))
        if len(retained) != len(existing):
            _atomic_jsonl(output_path, retained)
        completed = {record["article_id"] for record in retained}
        new_count = 0
        consecutive_failures = 0
        stop_request = StopRequest()
        status = "running"
        message: str | None = None
        signal_number: int | None = None
        _atomic_json(
            state_path,
            _state_record(
                status="running",
                run_id=run_id,
                total_articles=len(articles),
                records=retained,
                output_path=output_path,
                started_at_utc=started_at,
                new_articles_this_process=0,
            ),
        )
        pending = [
            article
            for article in articles
            if article["article_id"] not in completed
        ]
        active_engine = engine
        if pending and active_engine is None:
            active_engine = (
                engine_factory()
                if engine_factory is not None
                else FlanT5XlW17LiteEngine()
            )
        process_start = time.monotonic()
        try:
            with stop_handlers(stop_request):
                for article in pending:
                    if stop_request.requested:
                        status = "interrupted"
                        signal_number = stop_request.signal_number
                        message = "Stop requested before next article"
                        break
                    if (
                        max_new_articles is not None
                        and new_count >= max_new_articles
                    ):
                        status = "in_progress"
                        message = "Reached --max-new-articles"
                        break
                    if active_engine is None:
                        raise AssertionError("Pending work lacks an engine")
                    terminal = terminal_prediction(
                        article,
                        active_engine,
                        run_id=run_id,
                        queue_sha256=queue_sha,
                        selection_manifest_sha256=selection_sha,
                        preflight_manifest_sha256=preflight_sha,
                        runtime_contract_sha256_value=runtime_sha,
                        attempt=attempts[article["article_id"]] + 1,
                    )
                    validate_terminal_record(terminal)
                    _append_terminal(output_path, terminal)
                    retained.append(terminal)
                    completed.add(article["article_id"])
                    new_count += 1
                    if (
                        terminal["terminal_state"]
                        in FAILURE_TERMINAL_STATES
                    ):
                        consecutive_failures += 1
                    else:
                        consecutive_failures = 0
                    if (
                        consecutive_failures
                        >= max_consecutive_failures
                    ):
                        status = "aborted_systemic_failures"
                        message = (
                            f"{consecutive_failures} consecutive inference "
                            "terminal failures"
                        )
                        break
                    if (
                        new_count % PROGRESS_INTERVAL == 0
                        or len(retained) == len(articles)
                    ):
                        elapsed = max(time.monotonic() - process_start, 1e-9)
                        rate = new_count / elapsed
                        remaining = len(articles) - len(retained)
                        eta_hours = (
                            remaining / rate / 3600 if rate > 0 else None
                        )
                        eta_text = (
                            f"{eta_hours:.2f}h"
                            if eta_hours is not None
                            else "unknown"
                        )
                        print(
                            f"W17-Lite {len(retained):,}/{len(articles):,} "
                            f"committed; {new_count:,} this run; "
                            f"ETA {eta_text}",
                            flush=True,
                        )
                        _atomic_json(
                            state_path,
                            _state_record(
                                status="running",
                                run_id=run_id,
                                total_articles=len(articles),
                                records=retained,
                                output_path=output_path,
                                started_at_utc=started_at,
                                new_articles_this_process=new_count,
                            ),
                        )
        except KeyboardInterrupt:
            status = "interrupted"
            signal_number = int(signal.SIGINT)
            message = "KeyboardInterrupt"

        states = _record_counts(retained)
        failures = int(
            sum(states.get(name, 0) for name in FAILURE_TERMINAL_STATES)
        )
        if status == "running":
            if len(retained) < len(articles):
                status = "in_progress"
            elif failures:
                status = "complete_with_failures"
            else:
                status = "complete"
        if sha256_file(queue_path) != queue_sha:
            raise ValueError("W17-Lite queue changed during inference")
        if sha256_file(selection_manifest_path) != selection_sha:
            raise ValueError("W17-Lite selection changed during inference")
        if sha256_file(preflight_manifest_path) != preflight_sha:
            raise ValueError("W17-Lite preflight changed during inference")
        if runtime_contract() != startup_runtime:
            raise ValueError("W17-Lite runtime changed during inference")
        diagnostics_method = (
            getattr(active_engine, "diagnostics", None)
            if active_engine is not None
            else None
        )
        diagnostics = (
            diagnostics_method() if callable(diagnostics_method) else None
        )
        manifest = _inference_manifest(
            status=status,
            run_id=run_id,
            output_path=output_path,
            queue_path=queue_path,
            selection_manifest_path=selection_manifest_path,
            preflight_manifest_path=preflight_manifest_path,
            records=retained,
            articles=articles,
            runtime=startup_runtime,
            tail_recovery=tail_recovery,
            started_at_utc=started_at,
            new_articles_this_process=new_count,
            max_new_articles=max_new_articles,
            engine_diagnostics=diagnostics,
            signal_number=signal_number,
            message=message,
        )
        _atomic_json(manifest_path, manifest)
        _atomic_json(
            state_path,
            _state_record(
                status=status,
                run_id=run_id,
                total_articles=len(articles),
                records=retained,
                output_path=output_path,
                started_at_utc=started_at,
                new_articles_this_process=new_count,
                signal_number=signal_number,
                message=message,
            ),
        )
        if status == "aborted_systemic_failures":
            raise RuntimeError(message)
        return manifest


def status_report(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    selection_path = output_root / "selection_manifest.json"
    preflight_path = output_root / "preflight.json"
    predictions_path = output_root / "predictions.jsonl"
    inference_manifest_path = _output_companion(
        predictions_path, ".manifest.json"
    )
    state_path = _output_companion(predictions_path, ".state.json")
    report: dict[str, Any] = {
        "output_root": str(output_root.resolve()),
        "selection": "missing",
        "preflight": "missing",
        "inference": "not_started",
    }
    if selection_path.exists():
        selection = validate_selection_manifest(
            selection_path,
            queue_path=output_root / "selected_articles.jsonl.gz",
            require_current_implementation=False,
        )
        report["selection"] = selection["status"]
        report["selected_article_count"] = selection["audit"][
            "selected_unique_articles"
        ]
    if preflight_path.exists():
        preflight = _load_json(preflight_path)
        report["preflight"] = preflight.get("status")
        report["preflight_violation_count"] = preflight.get(
            "violation_count"
        )
    manifest_is_newest = (
        inference_manifest_path.exists()
        and (
            not state_path.exists()
            or inference_manifest_path.stat().st_mtime_ns
            >= state_path.stat().st_mtime_ns
        )
    )
    if manifest_is_newest:
        manifest = _load_json(inference_manifest_path)
        report["inference"] = manifest.get("status")
        report["successful_article_count"] = manifest.get(
            "successful_article_count"
        )
        report["failure_article_count"] = manifest.get(
            "failure_article_count"
        )
        report["remaining_article_count"] = manifest.get(
            "remaining_article_count"
        )
        report["run_id"] = manifest.get("run_id")
    elif state_path.exists():
        state = _load_json(state_path)
        report["inference"] = state.get("status")
        report["successful_article_count"] = state.get(
            "successful_article_count"
        )
        report["failure_article_count"] = state.get("failure_article_count")
        report["remaining_article_count"] = state.get(
            "remaining_article_count"
        )
        report["run_id"] = state.get("run_id")
    if predictions_path.exists():
        report["prediction_ledger_bytes"] = predictions_path.stat().st_size
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser(
        "prepare",
        help="Materialize or validate the exact K-16 article selection.",
    )
    prepare.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    prepare.add_argument("--articles", type=Path, default=DEFAULT_ARTICLES)
    prepare.add_argument(
        "--corpus-manifest", type=Path, default=DEFAULT_CORPUS_MANIFEST
    )
    prepare.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    prepare.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    prepare.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)

    preflight = commands.add_parser(
        "preflight",
        help="Tokenize every actual prompt; idempotently validate if complete.",
    )
    preflight.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    preflight.add_argument(
        "--selection-manifest",
        type=Path,
        default=DEFAULT_SELECTION_MANIFEST,
    )
    preflight.add_argument("--output", type=Path, default=DEFAULT_PREFLIGHT)
    preflight.add_argument("--prompt-batch-size", type=int, default=512)

    infer = commands.add_parser(
        "infer",
        help="Resume append-only FLAN inference from the last committed article.",
    )
    infer.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    infer.add_argument(
        "--selection-manifest",
        type=Path,
        default=DEFAULT_SELECTION_MANIFEST,
    )
    infer.add_argument(
        "--preflight-manifest", type=Path, default=DEFAULT_PREFLIGHT
    )
    infer.add_argument("--output", type=Path, default=DEFAULT_PREDICTIONS)
    infer.add_argument("--max-new-articles", type=int)
    infer.add_argument(
        "--no-retry-failed",
        action="store_true",
        help="Retain prior failed terminals instead of atomically retrying them.",
    )
    infer.add_argument("--recover-stale-lock", action="store_true")
    infer.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=SYSTEMIC_FAILURE_ABORT_THRESHOLD,
    )

    status = commands.add_parser("status")
    status.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        manifest = prepare_selection(
            assignments_path=args.assignments,
            articles_path=args.articles,
            corpus_manifest_path=args.corpus_manifest,
            design_path=args.design,
            schema_path=args.schema,
            output_root=args.output_root,
        )
    elif args.command == "preflight":
        manifest = preflight_corpus(
            queue_path=args.queue,
            selection_manifest_path=args.selection_manifest,
            output_path=args.output,
            prompt_batch_size=args.prompt_batch_size,
        )
        print(canonical_json(manifest))
        return 0 if manifest["status"] == "passed" else 2
    elif args.command == "infer":
        manifest = run_inference(
            queue_path=args.queue,
            selection_manifest_path=args.selection_manifest,
            preflight_manifest_path=args.preflight_manifest,
            output_path=args.output,
            max_new_articles=args.max_new_articles,
            retry_failed=not args.no_retry_failed,
            recover_stale_lock=args.recover_stale_lock,
            max_consecutive_failures=args.max_consecutive_failures,
        )
        print(canonical_json(manifest))
        if manifest["status"] == "interrupted":
            signum = manifest.get("signal_number")
            return 143 if signum == int(signal.SIGTERM) else 130
        if manifest["status"] == "complete_with_failures":
            return 2
        return 0
    else:
        print(
            json.dumps(
                status_report(output_root=args.output_root),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    print(canonical_json(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
