#!/usr/bin/env python
"""Build the shared v2 target-article and immutable model-text corpus.

The builder reuses the exact D2 cutoff, entity, sector, macro, and C/I/P
routing helpers. It does not call an LLM. The resulting assignments are the
single licensed input ledger for both FLAN W17 and GPT R70.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts import build_v2_deterministic_news_features as d2


BUILDER_VERSION = "semantic-corpus-cip-v1.2.0"
SOURCE_PROFILE = d2.SOURCE_PROFILE
DEFAULT_ARTICLES = d2.DEFAULT_ARTICLES
DEFAULT_SOURCE_MANIFEST = d2.DEFAULT_SOURCE_MANIFEST
DEFAULT_CALENDAR = d2.DEFAULT_CALENDAR
DEFAULT_UNIVERSE = d2.DEFAULT_UNIVERSE
DEFAULT_RULES = d2.DEFAULT_RULES
DEFAULT_QUANT_PANEL = d2.DEFAULT_QUANT_PANEL
DEFAULT_OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT
    / "data"
    / "features"
    / "news_semantic"
    / "massive_v2"
)
KEY_COLUMNS = ("forecast_date", "sector", "stock", "benchmark")
ROLE_ORDER = ("C", "I", "P")
DESCRIPTION_EXCERPT_MAX_CHARS = 512
DESCRIPTION_EXCERPT_BOUNDARY_WINDOW = 64
GENERATED_FILES = (
    "semantic_articles.parquet",
    "article_target_assignments.parquet",
    "article_target_assignments.jsonl.gz",
    "stock_day_scope.parquet",
    "stock_day_scope.jsonl.gz",
    "manifest.json",
    "manifest.sha256",
)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(resolved)


def shared_model_text(headline: str, description: str) -> str:
    return headline if not description else f"{headline}\n\n{description}"


def bounded_description(description: str) -> tuple[str, dict[str, Any]]:
    """Return the explicit shared lede excerpt and its loss metadata.

    The bound is part of feature construction rather than hidden tokenizer
    truncation.  Prefer a whitespace boundary near the cap, but make progress
    with an exact Unicode-code-point cut when a very long token has no such
    boundary.  The pinned FLAN tokenizer must still preflight every resulting
    prompt before inference.
    """

    source_chars = len(description)
    if source_chars <= DESCRIPTION_EXCERPT_MAX_CHARS:
        retained = description
    else:
        candidate = description[:DESCRIPTION_EXCERPT_MAX_CHARS]
        floor = (
            DESCRIPTION_EXCERPT_MAX_CHARS
            - DESCRIPTION_EXCERPT_BOUNDARY_WINDOW
        )
        boundaries = [
            candidate.rfind(character)
            for character in (" ", "\t", "\n")
        ]
        boundary = max(boundaries)
        retained = candidate[:boundary] if boundary >= floor else candidate
        retained = retained.rstrip()
        if not retained:
            retained = candidate
    retained_chars = len(retained)
    return retained, {
        "source_description_sha256": sha256_text(description),
        "source_description_char_count": source_chars,
        "retained_description_char_count": retained_chars,
        "omitted_description_char_count": source_chars - retained_chars,
        "description_was_bounded": source_chars != retained_chars,
    }


def assignment_id(article_id: str, forecast_date: str, ticker: str) -> str:
    digest = sha256_text(
        "\x1f".join(
            (BUILDER_VERSION, article_id, forecast_date, ticker.upper())
        )
    )
    return f"sem-{digest}"


def assignment_ids_sha256(values: Sequence[str]) -> str:
    return sha256_text("\n".join(sorted(values)))


def _clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _atomic_jsonl(
    path: Path, records: Iterable[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(
        temporary,
        mode="wt",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        for record in records:
            handle.write(canonical_json(record) + "\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def normalize_row_universe(
    frame: pd.DataFrame,
    targets: Sequence[d2.Target],
    sessions: Sequence[d2.Session],
) -> pd.DataFrame:
    missing = set(KEY_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"row universe is missing {sorted(missing)}")
    result = frame[list(KEY_COLUMNS)].copy()
    result["forecast_date"] = pd.to_datetime(
        result["forecast_date"]
    ).dt.normalize()
    result["stock"] = result["stock"].astype(str).str.upper()
    result["benchmark"] = result["benchmark"].astype(str).str.upper()
    if result.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("row universe contains duplicate stock-day keys")
    expected_targets = {
        (target.sector, target.ticker, target.benchmark)
        for target in targets
    }
    for forecast_date, group in result.groupby("forecast_date", sort=False):
        observed = set(
            group[["sector", "stock", "benchmark"]].itertuples(
                index=False, name=None
            )
        )
        if observed != expected_targets:
            raise ValueError(
                f"{forecast_date.date()}: row universe differs from 30 targets"
            )
    session_dates = {pd.Timestamp(item.session_date) for item in sessions}
    missing_dates = set(result["forecast_date"]) - session_dates
    if missing_dates:
        raise ValueError(
            f"row universe has non-calendar dates: {sorted(missing_dates)[:3]}"
        )
    return result.sort_values(list(KEY_COLUMNS), kind="mergesort").reset_index(
        drop=True
    )


def _role(flags: Mapping[str, bool]) -> str:
    if flags["any_common"]:
        return "C"
    if flags["target_idio"]:
        return "I"
    if flags["peer_idio"]:
        return "P"
    raise AssertionError("Relevant assignment has no C/I/P role")


def build_semantic_corpus(
    *,
    articles: pd.DataFrame,
    targets: Sequence[d2.Target],
    static_rules: Mapping[str, Any],
    sessions: Sequence[d2.Session],
    row_universe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return unique text rows, target assignments, stock-day scope, and audit."""

    universe = normalize_row_universe(row_universe, targets, sessions)
    included_dates = set(universe["forecast_date"])
    target_by_ticker = {target.ticker: target for target in targets}
    targets_by_sector: dict[str, list[d2.Target]] = defaultdict(list)
    for target in targets:
        targets_by_sector[target.sector].append(target)
    sector_names = sorted(targets_by_sector)
    sector_target_sets = {
        sector: {target.ticker for target in values}
        for sector, values in targets_by_sector.items()
    }
    benchmark_by_sector = {
        sector: values[0].benchmark
        for sector, values in targets_by_sector.items()
    }
    aliases = d2.build_aliases(targets, static_rules)
    cutoff_times = [item.cutoff_utc for item in sessions]
    session_by_date = {
        pd.Timestamp(item.session_date): item for item in sessions
    }
    required = {
        "provider_article_id",
        "published_at_utc",
        "title",
        "description",
        "provider_tickers",
        "keywords",
    }
    if not required.issubset(articles.columns):
        raise ValueError(
            f"normalized articles missing {sorted(required - set(articles.columns))}"
        )
    if articles["provider_article_id"].duplicated().any():
        raise ValueError("normalized articles contain duplicate provider IDs")

    assignments: list[dict[str, Any]] = []
    shared_articles: dict[str, dict[str, Any]] = {}
    articles_in_window = 0
    descriptions_in_window = 0
    for row in articles.itertuples(index=False):
        published = pd.Timestamp(row.published_at_utc)
        if published.tzinfo is None:
            raise ValueError("published_at_utc must be timezone aware")
        published_dt = published.to_pydatetime().astimezone(d2.UTC)
        mapped_index = d2.assign_forecast_session_index(
            published_dt, cutoff_times
        )
        if mapped_index is None:
            continue
        mapped_session = sessions[mapped_index]
        forecast_timestamp = pd.Timestamp(mapped_session.session_date)
        if forecast_timestamp not in included_dates:
            continue
        articles_in_window += 1
        article_id = str(row.provider_article_id).strip()
        if not article_id:
            raise ValueError("provider_article_id must be nonempty")
        headline = _clean_text(row.title)
        if not headline:
            raise ValueError(f"{article_id}: headline is empty")
        source_description = _clean_text(row.description)
        descriptions_in_window += int(bool(source_description))
        description, description_bound = bounded_description(
            source_description
        )
        source_model_text = shared_model_text(headline, source_description)
        model_text = shared_model_text(headline, description)
        headline_sha = sha256_text(headline)
        description_sha = sha256_text(description)
        model_text_sha = sha256_text(model_text)
        provider_tickers = {
            value.upper()
            for value in d2.parse_semicolon_set(row.provider_tickers)
        }
        keywords = d2.parse_semicolon_set(row.keywords)
        detected = d2.detected_entities(
            source_model_text, provider_tickers, targets, aliases
        )
        macro_common = d2.macro_evidence(
            source_model_text, keywords, static_rules
        )
        visible_detected = d2.detected_entities(
            model_text, set(), targets, aliases
        )
        visible_macro_common = d2.macro_evidence(
            model_text, set(), static_rules
        )
        age_hours = (
            mapped_session.cutoff_utc - published_dt
        ).total_seconds() / 3600
        if age_hours < 0:
            raise AssertionError("strict cutoff mapper admitted a future article")
        recency_weight = math.exp(-math.log(2) * age_hours / 12)
        shared_articles[article_id] = {
            "provider_article_id": article_id,
            "published_at_utc": published_dt.isoformat().replace("+00:00", "Z"),
            "forecast_date": mapped_session.session_date.isoformat(),
            "cutoff_utc": mapped_session.cutoff_utc.isoformat().replace(
                "+00:00", "Z"
            ),
            "headline": headline,
            "description": description,
            "description_available": int(bool(source_description)),
            **description_bound,
            "model_text": model_text,
            "headline_sha256": headline_sha,
            "description_sha256": description_sha,
            "model_text_sha256": model_text_sha,
            "source_text_sha256": str(getattr(row, "text_sha256", "") or ""),
            "provider_tickers": sorted(provider_tickers),
            "keywords": sorted(keywords),
            "detected_tickers": sorted(detected),
            "extractor_visible_detected_tickers": sorted(visible_detected),
            "source_profile": SOURCE_PROFILE,
        }
        for sector in sector_names:
            sector_entities = detected & sector_target_sets[sector]
            sector_common = bool(
                benchmark_by_sector[sector] in provider_tickers
                or d2.sector_phrase_evidence(
                    source_model_text, sector, static_rules
                )
                or len(sector_entities) >= 2
            )
            if not macro_common and not sector_common and not sector_entities:
                continue
            visible_sector_entities = (
                visible_detected & sector_target_sets[sector]
            )
            visible_sector_common = bool(
                d2.sector_phrase_evidence(
                    model_text, sector, static_rules
                )
                or len(visible_sector_entities) >= 2
            )
            for target in targets_by_sector[sector]:
                flags = d2.role_flags(
                    target_ticker=target.ticker,
                    sector_entities=sector_entities,
                    sector_common=sector_common,
                    macro_common=macro_common,
                )
                if not flags["relevant"]:
                    continue
                visible_flags = d2.role_flags(
                    target_ticker=target.ticker,
                    sector_entities=visible_sector_entities,
                    sector_common=visible_sector_common,
                    macro_common=visible_macro_common,
                )
                role = _role(flags)
                forecast_date = mapped_session.session_date.isoformat()
                value = {
                    "assignment_id": assignment_id(
                        article_id, forecast_date, target.ticker
                    ),
                    "article_id": article_id,
                    "provider_article_id": article_id,
                    "forecast_date": forecast_date,
                    "cutoff_utc": mapped_session.cutoff_utc.isoformat().replace(
                        "+00:00", "Z"
                    ),
                    "published_at_utc": published_dt.isoformat().replace(
                        "+00:00", "Z"
                    ),
                    "target_ticker": target.ticker,
                    "target_company": target.company,
                    "sector": target.sector,
                    "benchmark": target.benchmark,
                    "known_sector_peers": list(target.peers),
                    "role": role,
                    "candidate_roles": {
                        name: bool(flags[name])
                        for name in (
                            "direct",
                            "target_idio",
                            "target_common",
                            "peer_idio",
                            "sector_common",
                            "macro_common",
                            "any_common",
                        )
                    },
                    "roles": {
                        name: bool(visible_flags[name])
                        for name in (
                            "direct",
                            "target_idio",
                            "target_common",
                            "peer_idio",
                            "sector_common",
                            "macro_common",
                            "any_common",
                        )
                    },
                    "assignment_reason": {
                        "C": "sector_or_macro_common",
                        "I": "direct_target_idiosyncratic",
                        "P": "single_peer_idiosyncratic",
                    }[role],
                    "routing_detected_entities": sorted(detected),
                    "routing_detected_sector_entities": sorted(
                        sector_entities
                    ),
                    "detected_entities": sorted(visible_detected),
                    "detected_sector_entities": sorted(
                        visible_sector_entities
                    ),
                    "provider_tickers": sorted(provider_tickers),
                    "headline": headline,
                    "description": description,
                    "description_available": int(bool(source_description)),
                    **description_bound,
                    "model_text": model_text,
                    "headline_sha256": headline_sha,
                    "description_sha256": description_sha,
                    "text_sha256": model_text_sha,
                    "model_text_sha256": model_text_sha,
                    "source_profile": SOURCE_PROFILE,
                    "source_query_scope_complete": True,
                    "candidate_assignment_complete": True,
                    "article_age_hours": age_hours,
                    "recency_weight_12h": recency_weight,
                    "duplication_group_id": article_id,
                    "duplication_group_size": 1,
                    "aggregation_weight": recency_weight,
                    "point_in_time_version_safe": False,
                    "entity_map_effective_dated": False,
                }
                value["corpus_assignment_sha256"] = sha256_text(
                    canonical_json(value)
                )
                assignments.append(value)

    assignment_frame = pd.DataFrame(assignments)
    if assignment_frame.empty:
        raise ValueError("semantic assignment corpus is empty")
    assignment_frame = assignment_frame.sort_values(
        ["forecast_date", "sector", "target_ticker", "assignment_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    if assignment_frame["assignment_id"].duplicated().any():
        raise AssertionError("semantic assignment IDs are not unique")
    grouped_ids: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
    grouped_roles: dict[tuple[str, str, str, str], Counter[str]] = defaultdict(
        Counter
    )
    for record in assignment_frame.to_dict(orient="records"):
        key = (
            record["forecast_date"],
            record["sector"],
            record["target_ticker"],
            record["benchmark"],
        )
        grouped_ids[key].append(record["assignment_id"])
        grouped_roles[key][record["role"]] += 1

    scopes = []
    for row in universe.itertuples(index=False):
        forecast_date = pd.Timestamp(row.forecast_date).date().isoformat()
        key = (forecast_date, row.sector, row.stock, row.benchmark)
        ids = grouped_ids.get(key, [])
        target = target_by_ticker[row.stock]
        counts = grouped_roles.get(key, Counter())
        session = session_by_date[pd.Timestamp(row.forecast_date)]
        scopes.append(
            {
                "forecast_date": forecast_date,
                "sector": row.sector,
                "stock": row.stock,
                "benchmark": row.benchmark,
                "target_ticker": row.stock,
                "target_company": target.company,
                "known_sector_peers": list(target.peers),
                "cutoff_utc": session.cutoff_utc.isoformat().replace(
                    "+00:00", "Z"
                ),
                "source_profile": SOURCE_PROFILE,
                "source_query_scope_complete": True,
                "candidate_assignment_complete": True,
                "expected_assignment_count": len(ids),
                "expected_assignment_ids_sha256": assignment_ids_sha256(ids),
                "expected_common_count": counts["C"],
                "expected_target_idiosyncratic_count": counts["I"],
                "expected_peer_idiosyncratic_count": counts["P"],
            }
        )
    scope_frame = pd.DataFrame(scopes).sort_values(
        list(KEY_COLUMNS), kind="mergesort"
    ).reset_index(drop=True)
    if scope_frame.duplicated(list(KEY_COLUMNS)).any():
        raise AssertionError("stock-day semantic scope is not unique")
    if int(scope_frame["expected_assignment_count"].sum()) != len(
        assignment_frame
    ):
        raise AssertionError("stock-day assignment ledger is incomplete")

    used_ids = set(assignment_frame["provider_article_id"])
    article_frame = pd.DataFrame(
        [shared_articles[value] for value in sorted(used_ids)]
    )
    article_frame = article_frame.sort_values(
        ["forecast_date", "published_at_utc", "provider_article_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    audit = {
        "articles_in_information_window": articles_in_window,
        "descriptions_in_information_window": descriptions_in_window,
        "assigned_unique_articles": len(article_frame),
        "assignment_count": len(assignment_frame),
        "stock_day_scope_rows": len(scope_frame),
        "stock_days_without_candidates": int(
            scope_frame["expected_assignment_count"].eq(0).sum()
        ),
        "role_counts": {
            role: int(assignment_frame["role"].eq(role).sum())
            for role in ROLE_ORDER
        },
        "maximum_assignments_per_stock_day": int(
            scope_frame["expected_assignment_count"].max()
        ),
        "mean_assignments_per_stock_day": float(
            scope_frame["expected_assignment_count"].mean()
        ),
        "maximum_target_assignments_per_article": int(
            assignment_frame.groupby("provider_article_id").size().max()
        ),
        "description_available_assignment_fraction": float(
            assignment_frame["description_available"].mean()
        ),
        "bounded_description_unique_article_count": int(
            article_frame["description_was_bounded"].sum()
        ),
        "bounded_description_assignment_count": int(
            assignment_frame["description_was_bounded"].sum()
        ),
        "maximum_source_description_chars": int(
            article_frame["source_description_char_count"].max()
        ),
        "maximum_retained_description_chars": int(
            article_frame["retained_description_char_count"].max()
        ),
        "total_omitted_description_chars": int(
            article_frame["omitted_description_char_count"].sum()
        ),
    }
    return article_frame, assignment_frame, scope_frame, audit


def _generated_record(path: Path, rows: int, columns: int) -> dict[str, Any]:
    return {
        "path": display_path(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "rows": rows,
        "columns": columns,
    }


def build_and_write(
    *,
    articles_path: Path = DEFAULT_ARTICLES,
    source_manifest_path: Path = DEFAULT_SOURCE_MANIFEST,
    calendar_path: Path = DEFAULT_CALENDAR,
    universe_path: Path = DEFAULT_UNIVERSE,
    rules_path: Path = DEFAULT_RULES,
    quant_panel_path: Path = DEFAULT_QUANT_PANEL,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    overwrite: bool = False,
) -> dict[str, Any]:
    d2.validate_source_manifest(source_manifest_path, articles_path)
    output_directory.mkdir(parents=True, exist_ok=True)
    existing = [
        output_directory / name
        for name in GENERATED_FILES
        if (output_directory / name).exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            "semantic corpus outputs exist; pass --overwrite to replace them"
        )
    if overwrite:
        for path in existing:
            path.unlink()
    targets = d2.load_targets(universe_path)
    rules = d2.load_static_rules(rules_path, targets)
    sessions = d2.load_sessions(calendar_path)
    articles = pd.read_csv(articles_path, compression="gzip")
    quant_panel = pd.read_parquet(quant_panel_path)
    article_frame, assignment_frame, scope_frame, audit = (
        build_semantic_corpus(
            articles=articles,
            targets=targets,
            static_rules=rules,
            sessions=sessions,
            row_universe=quant_panel,
        )
    )
    article_path = output_directory / "semantic_articles.parquet"
    assignment_path = output_directory / "article_target_assignments.parquet"
    assignment_jsonl = (
        output_directory / "article_target_assignments.jsonl.gz"
    )
    scope_path = output_directory / "stock_day_scope.parquet"
    scope_jsonl = output_directory / "stock_day_scope.jsonl.gz"
    _atomic_parquet(article_path, article_frame)
    _atomic_parquet(assignment_path, assignment_frame)
    _atomic_jsonl(
        assignment_jsonl, assignment_frame.to_dict(orient="records")
    )
    _atomic_parquet(scope_path, scope_frame)
    _atomic_jsonl(scope_jsonl, scope_frame.to_dict(orient="records"))
    manifest = {
        "manifest_version": "semantic-corpus-manifest-v1",
        "builder_version": BUILDER_VERSION,
        "status": "complete_exploratory_retrospective",
        "generated_at_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "source_profile": SOURCE_PROFILE,
        "information_time_contract": {
            "cutoff_et": "09:00",
            "timezone": "America/New_York",
            "assignment": "first official-session cutoff strictly after published_at_utc",
            "availability_proxy": "published_at_utc",
        },
        "routing_contract": {
            "roles": list(ROLE_ORDER),
            "role_precedence": "C then I then P",
            "helpers_reused_from": display_path(
                REPOSITORY_ROOT
                / "scripts"
                / "build_v2_deterministic_news_features.py"
            ),
            "static_entity_and_peer_map": True,
        },
        "text_contract": {
            "view": (
                "trimmed headline; append two LF bytes and an explicit "
                "bounded leading-description excerpt only when nonempty"
            ),
            "encoding": "utf-8",
            "full_body_used": False,
            "description_excerpt_max_unicode_codepoints": (
                DESCRIPTION_EXCERPT_MAX_CHARS
            ),
            "preferred_whitespace_boundary_window_codepoints": (
                DESCRIPTION_EXCERPT_BOUNDARY_WINDOW
            ),
            "routing_uses_complete_source_description": True,
            "candidate_roles_may_use_provider_and_complete_source_metadata": (
                True
            ),
            "extractors_use_only_bounded_shared_view": True,
            "extractor_visible_entities_and_roles_recomputed_from_shared_view": (
                True
            ),
            "provider_tickers_keywords_and_candidate_roles_exposed_to_extractors": (
                False
            ),
            "runner_side_or_silent_truncation_allowed": False,
            "flan_tokenizer_preflight_required": True,
        },
        "audit": audit,
        "input_files": {
            display_path(path): sha256_file(path)
            for path in (
                articles_path,
                source_manifest_path,
                calendar_path,
                universe_path,
                rules_path,
                quant_panel_path,
                REPOSITORY_ROOT
                / "scripts"
                / "build_v2_deterministic_news_features.py",
                Path(__file__).resolve(),
            )
        },
        "generated_files": {
            article_path.name: _generated_record(
                article_path, len(article_frame), len(article_frame.columns)
            ),
            assignment_path.name: _generated_record(
                assignment_path,
                len(assignment_frame),
                len(assignment_frame.columns),
            ),
            assignment_jsonl.name: _generated_record(
                assignment_jsonl,
                len(assignment_frame),
                len(assignment_frame.columns),
            ),
            scope_path.name: _generated_record(
                scope_path, len(scope_frame), len(scope_frame.columns)
            ),
            scope_jsonl.name: _generated_record(
                scope_jsonl, len(scope_frame), len(scope_frame.columns)
            ),
        },
        "claim_flags": {
            "point_in_time_version_safe": False,
            "effective_dated_entity_map": False,
            "complete_untickered_macro_coverage": False,
            "primary_or_confirmatory_eligible": False,
            "exploratory_construction_eligible": True,
        },
    }
    manifest_path = output_directory / "manifest.json"
    _atomic_json(manifest_path, manifest)
    sha_path = output_directory / "manifest.sha256"
    temporary = sha_path.with_name(sha_path.name + ".tmp")
    temporary.write_text(sha256_file(manifest_path) + "\n", encoding="ascii")
    os.replace(temporary, sha_path)
    return manifest


def parser() -> argparse.ArgumentParser:
    output = argparse.ArgumentParser(description=__doc__)
    output.add_argument("--articles", type=Path, default=DEFAULT_ARTICLES)
    output.add_argument(
        "--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST
    )
    output.add_argument("--calendar", type=Path, default=DEFAULT_CALENDAR)
    output.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    output.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    output.add_argument("--quant-panel", type=Path, default=DEFAULT_QUANT_PANEL)
    output.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    output.add_argument("--overwrite", action="store_true")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    manifest = build_and_write(
        articles_path=args.articles,
        source_manifest_path=args.source_manifest,
        calendar_path=args.calendar,
        universe_path=args.universe,
        rules_path=args.rules,
        quant_panel_path=args.quant_panel,
        output_directory=args.output_directory,
        overwrite=args.overwrite,
    )
    audit = manifest["audit"]
    print(
        f"Built {audit['assignment_count']:,} assignments over "
        f"{audit['stock_day_scope_rows']:,} stock-days"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
