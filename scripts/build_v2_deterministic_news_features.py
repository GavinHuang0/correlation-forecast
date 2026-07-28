#!/usr/bin/env python
"""Build the exploratory v2 deterministic-news feature panel.

This builder intentionally starts from provider-ID-deduplicated Massive
articles, not the v1 article-target assignments.  It applies the v2 strict
cutoff and role rules, materializes auditable daily components, and then
constructs the exact ordered D2-Normalized (30) and D2-Levels (5) blocks.

The current entity registry is a static research-peer fallback.  Outputs are
therefore exploratory and retrospective, never historical-version-safe.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

import build_deterministic_news_features as v1_rules  # noqa: E402
import coarse_news_features as coarse_rules  # noqa: E402


BUILDER_VERSION = "d2-normalized-static-research-peer-v1.0.0"
SOURCE_PROFILE = "ordinary_massive_retrospective"
ENTITY_MAP_MODE = "static_research_peer_alias_fallback"
NEW_YORK = ZoneInfo("America/New_York")
UTC = timezone.utc

DEFAULT_ARTICLES = (
    REPOSITORY_ROOT
    / "data"
    / "features"
    / "news_deterministic"
    / "massive_v1"
    / "normalized_articles.csv.gz"
)
DEFAULT_SOURCE_MANIFEST = DEFAULT_ARTICLES.parent / "manifest.json"
DEFAULT_CALENDAR = (
    REPOSITORY_ROOT
    / "data"
    / "prices"
    / "alpaca"
    / "calendar"
    / "2016-01-01_2026-06-30.json"
)
DEFAULT_UNIVERSE = REPOSITORY_ROOT / "config" / "news_target_universe_30.json"
DEFAULT_RULES = (
    REPOSITORY_ROOT / "config" / "news_deterministic_v2_static_rules.json"
)
DEFAULT_QUANT_PANEL = (
    REPOSITORY_ROOT
    / "data"
    / "features"
    / "quant"
    / "training_v1"
    / "modeling_panel.parquet"
)
DEFAULT_OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT
    / "data"
    / "features"
    / "news_deterministic"
    / "massive_v2"
)

KEY_COLUMNS = ["forecast_date", "sector", "stock", "benchmark"]

D2_NORMALIZED_30 = [
    "d2_target_direct_intensity_midrank_126",
    "d2_peer_idio_intensity_midrank_126",
    "d2_common_intensity_midrank_126",
    "d2_observed_no_direct_target_article",
    "d2_common_article_share",
    "d2_target_common_share",
    "d2_observed_no_peer_set_entity_mention",
    "d2_peer_set_entity_hhi",
    "d2_peer_coverage_ratio",
    "d2_target_peer_co_mention_share",
    "d2_target_attention_share_delta_63",
    "d2_macro_share_of_common",
    "d2_target_mean_recency_weight_12h",
    "d2_common_mean_recency_weight_12h",
    "d2_target_premarket_article_share",
    "d2_common_premarket_article_share",
    "d2_target_earnings_guidance_article_share",
    "d2_common_earnings_guidance_article_share",
    "d2_target_product_demand_article_share",
    "d2_common_product_demand_article_share",
    "d2_target_supply_capacity_article_share",
    "d2_common_supply_capacity_article_share",
    "d2_target_regulation_legal_article_share",
    "d2_common_regulation_legal_article_share",
    "d2_target_corporate_analyst_article_share",
    "d2_common_corporate_analyst_article_share",
    "d2_target_positive_surprise_article_share",
    "d2_target_negative_surprise_article_share",
    "d2_common_positive_surprise_article_share",
    "d2_common_negative_surprise_article_share",
]

D2_LEVELS_5 = [
    "d2_log1p_target_idio_article_count",
    "d2_log1p_target_common_article_count",
    "d2_log1p_peer_idio_article_count",
    "d2_log1p_sector_common_article_count",
    "d2_log1p_macro_common_article_count",
]

ROLE_COUNT_COLUMNS = [
    "n_direct",
    "n_target_idio",
    "n_target_common",
    "n_peer_idio",
    "n_sector_common",
    "n_macro_common",
    "n_any_common",
]

CUE_NAMES = [
    "earnings_guidance",
    "product_demand",
    "supply_capacity",
    "regulation_legal",
    "corporate_analyst",
    "positive_surprise",
    "negative_surprise",
]

AUDIT_COLUMNS = [
    "source_profile",
    "entity_map_mode",
    "research_peer_set",
    "source_query_scope_complete",
    "untickered_macro_coverage_complete",
    "point_in_time_version_safe",
    "primary_training_eligible",
    "exploratory_fit_eligible",
]


@dataclass(frozen=True)
class Session:
    session_date: date
    cutoff_utc: datetime


@dataclass(frozen=True)
class Target:
    ticker: str
    company: str
    sector: str
    benchmark: str
    peers: tuple[str, ...]


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(resolved)


def literal_present(text: str, phrase: str, *, case_sensitive: bool = False) -> bool:
    flags = 0 if case_sensitive else re.IGNORECASE
    pattern = (
        r"(?<![A-Za-z0-9])"
        + re.escape(phrase)
        + r"(?![A-Za-z0-9])"
    )
    return re.search(pattern, text, flags) is not None


def ticker_text_present(text: str, ticker: str) -> bool:
    if literal_present(text, f"${ticker}", case_sensitive=True):
        return True
    return len(ticker) >= 3 and literal_present(
        text, ticker, case_sensitive=True
    )


def normalize_keyword(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def parse_semicolon_set(value: Any) -> set[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return set()
    return {
        part.strip()
        for part in str(value).split(";")
        if part.strip()
    }


def load_sessions(path: Path, cutoff_et: time = time(9, 0)) -> list[Session]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_sessions = payload.get("sessions")
    if not isinstance(raw_sessions, list):
        raise ValueError(f"{path}: missing sessions array")
    sessions: list[Session] = []
    for raw in raw_sessions:
        session_date = date.fromisoformat(str(raw["date"]))
        cutoff_local = datetime.combine(
            session_date, cutoff_et, tzinfo=NEW_YORK
        )
        sessions.append(
            Session(
                session_date=session_date,
                cutoff_utc=cutoff_local.astimezone(UTC),
            )
        )
    sessions.sort(key=lambda item: item.session_date)
    if len({item.session_date for item in sessions}) != len(sessions):
        raise ValueError(f"{path}: duplicate session dates")
    return sessions


def assign_forecast_session_index(
    available_at: datetime,
    cutoff_times_utc: Sequence[datetime],
) -> int | None:
    """Return the first session whose cutoff is strictly after availability."""

    index = bisect.bisect_right(
        cutoff_times_utc, available_at.astimezone(UTC)
    )
    return index if index < len(cutoff_times_utc) else None


def load_targets(path: Path) -> list[Target]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_targets = payload.get("targets")
    if not isinstance(raw_targets, Mapping) or len(raw_targets) != 30:
        raise ValueError(f"{path}: expected exactly 30 targets")
    targets: list[Target] = []
    for ticker, raw in raw_targets.items():
        if not isinstance(raw, Mapping):
            raise ValueError(f"{path}: invalid target {ticker!r}")
        targets.append(
            Target(
                ticker=str(ticker).upper(),
                company=str(raw["company"]).strip(),
                sector=str(raw["sector"]).strip(),
                benchmark=str(raw["benchmark"]).upper(),
                peers=tuple(str(item).upper() for item in raw["peers"]),
            )
        )
    targets.sort(key=lambda item: (item.sector, item.ticker))
    by_sector: dict[str, set[str]] = {}
    for target in targets:
        by_sector.setdefault(target.sector, set()).add(target.ticker)
    if any(len(values) != 6 for values in by_sector.values()):
        raise ValueError(f"{path}: every research sector must have six stocks")
    for target in targets:
        expected = by_sector[target.sector] - {target.ticker}
        if set(target.peers) != expected:
            raise ValueError(f"{path}: inconsistent peers for {target.ticker}")
    return targets


def load_static_rules(
    path: Path, targets: Sequence[Target]
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != (
        "static_research_peer_alias_fallback_non_effective_dated"
    ):
        raise ValueError(f"{path}: unexpected static-rule status")
    aliases = payload.get("aliases")
    sector_phrases = payload.get("sector_phrases")
    macro_keywords = payload.get("macro_provider_keywords")
    if not isinstance(aliases, Mapping) or set(aliases) != {
        target.ticker for target in targets
    }:
        raise ValueError(f"{path}: aliases must cover exactly the target universe")
    sectors = {target.sector for target in targets}
    if not isinstance(sector_phrases, Mapping) or set(sector_phrases) != sectors:
        raise ValueError(f"{path}: sector phrases must cover exactly five sectors")
    if not isinstance(macro_keywords, list):
        raise ValueError(f"{path}: macro_provider_keywords must be a list")
    return payload


def build_aliases(
    targets: Sequence[Target], static_rules: Mapping[str, Any]
) -> dict[str, tuple[str, ...]]:
    raw_aliases = static_rules["aliases"]
    aliases: dict[str, tuple[str, ...]] = {}
    for target in targets:
        values = [target.company, *raw_aliases[target.ticker]]
        aliases[target.ticker] = tuple(
            dict.fromkeys(str(value).strip() for value in values if str(value).strip())
        )
    return aliases


def detected_entities(
    text: str,
    provider_tickers: set[str],
    targets: Sequence[Target],
    aliases: Mapping[str, Sequence[str]],
) -> set[str]:
    detected: set[str] = set()
    for target in targets:
        ticker = target.ticker
        if (
            ticker in provider_tickers
            or ticker_text_present(text, ticker)
            or any(literal_present(text, alias) for alias in aliases[ticker])
        ):
            detected.add(ticker)
    return detected


def macro_evidence(
    text: str,
    keywords: set[str],
    static_rules: Mapping[str, Any],
) -> bool:
    normalized_keywords = {normalize_keyword(value) for value in keywords}
    configured = {
        normalize_keyword(value)
        for value in static_rules["macro_provider_keywords"]
    }
    keyword_match = bool(normalized_keywords & configured)
    text_match = any(
        literal_present(text, phrase)
        for phrase in coarse_rules.MACRO_TEXT_TERMS
    )
    return bool(keyword_match or text_match)


def sector_phrase_evidence(
    text: str, sector: str, static_rules: Mapping[str, Any]
) -> bool:
    return any(
        literal_present(text, phrase)
        for phrase in static_rules["sector_phrases"][sector]
    )


def lexical_cues(text: str) -> dict[str, bool]:
    result = {
        name: any(
            re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            for pattern in v1_rules.EVENT_FAMILY_PATTERNS[name]
        )
        for name in CUE_NAMES[:5]
    }
    _, positive, negative = coarse_rules.explicit_surprise_from_text(text)
    result["positive_surprise"] = bool(positive)
    result["negative_surprise"] = bool(negative)
    return result


def role_flags(
    *,
    target_ticker: str,
    sector_entities: set[str],
    sector_common: bool,
    macro_common: bool,
) -> dict[str, bool]:
    direct = target_ticker in sector_entities
    common = bool(sector_common or macro_common)
    target_idio = bool(direct and not common)
    target_common = bool(direct and common)
    peer_idio = bool(
        not direct and not common and len(sector_entities) == 1
    )
    return {
        "direct": direct,
        "target_idio": target_idio,
        "target_common": target_common,
        "peer_idio": peer_idio,
        "sector_common": bool(sector_common),
        "macro_common": bool(macro_common),
        "any_common": common,
        "relevant": bool(common or target_idio or peer_idio),
    }


def prior_midrank(values: Sequence[float], window: int = 126) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    result = np.full(array.shape, np.nan, dtype=float)
    for index in range(window, len(array)):
        history = array[index - window : index]
        current = array[index]
        result[index] = (
            np.count_nonzero(history < current)
            + 0.5 * np.count_nonzero(history == current)
        ) / window
    return result


def prior_median_delta(values: Sequence[float], window: int = 63) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    result = np.full(array.shape, np.nan, dtype=float)
    for index in range(window, len(array)):
        result[index] = array[index] - np.median(
            array[index - window : index]
        )
    return result


def validate_source_manifest(
    path: Path, articles_path: Path
) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    generated = manifest.get("generated_files", {})
    record = generated.get("normalized_articles.csv.gz", {})
    actual_hash = sha256_file(articles_path)
    if record.get("sha256") != actual_hash:
        raise ValueError(f"{articles_path}: hash does not match {path}")
    completeness = manifest.get("collection_completeness", {})
    required_true = [
        "collector_manifest_available",
        "all_collector_manifests_available",
        "ordinary_ticker_collection_complete",
        "sector_benchmark_collection_complete",
        "control_collection_complete",
        "benchmark_and_control_collection_complete",
    ]
    if not all(bool(completeness.get(name)) for name in required_true):
        raise ValueError(f"{path}: provider query collection is incomplete")
    roots = completeness.get("collector_roots", [])
    if not roots or any(
        root.get("status") != "complete"
        or not root.get("scope_covers_build_dates")
        or not root.get("all_manifest_pages_present")
        or not root.get("all_page_hashes_match")
        for root in roots
    ):
        raise ValueError(f"{path}: collector-root provenance is incomplete")
    return manifest


def _empty_arrays(
    session_count: int,
    target_count: int,
    sector_count: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        name: np.zeros((session_count, target_count), dtype=np.int32)
        for name in ROLE_COUNT_COLUMNS
    }
    result.update(
        {
            "target_recency_sum": np.zeros(
                (session_count, target_count), dtype=np.float64
            ),
            "common_recency_sum": np.zeros(
                (session_count, target_count), dtype=np.float64
            ),
            "target_premarket_count": np.zeros(
                (session_count, target_count), dtype=np.int32
            ),
            "common_premarket_count": np.zeros(
                (session_count, target_count), dtype=np.int32
            ),
            "target_peer_co_mention_count": np.zeros(
                (session_count, target_count), dtype=np.int32
            ),
            "entity_mentions": np.zeros(
                (session_count, sector_count, 6), dtype=np.int32
            ),
        }
    )
    for cue in CUE_NAMES:
        result[f"target_{cue}_count"] = np.zeros(
            (session_count, target_count), dtype=np.int32
        )
        result[f"common_{cue}_count"] = np.zeros(
            (session_count, target_count), dtype=np.int32
        )
    return result


def build_daily_components(
    articles: pd.DataFrame,
    targets: Sequence[Target],
    static_rules: Mapping[str, Any],
    all_sessions: Sequence[Session],
    complete_sessions: Sequence[Session],
) -> pd.DataFrame:
    cutoff_times = [item.cutoff_utc for item in all_sessions]
    complete_index = {
        item.session_date: index
        for index, item in enumerate(complete_sessions)
    }
    target_index = {target.ticker: index for index, target in enumerate(targets)}
    sector_names = sorted({target.sector for target in targets})
    sector_index = {sector: index for index, sector in enumerate(sector_names)}
    targets_by_sector = {
        sector: [target for target in targets if target.sector == sector]
        for sector in sector_names
    }
    sector_entity_order = {
        sector: sorted(target.ticker for target in sector_targets)
        for sector, sector_targets in targets_by_sector.items()
    }
    sector_target_sets = {
        sector: set(values) for sector, values in sector_entity_order.items()
    }
    benchmark_by_sector = {
        sector: sector_targets[0].benchmark
        for sector, sector_targets in targets_by_sector.items()
    }
    aliases = build_aliases(targets, static_rules)
    arrays = _empty_arrays(
        len(complete_sessions), len(targets), len(sector_names)
    )

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

    for row in articles.itertuples(index=False):
        published = pd.Timestamp(row.published_at_utc)
        if published.tzinfo is None:
            raise ValueError("published_at_utc must be timezone aware")
        published_dt = published.to_pydatetime().astimezone(UTC)
        mapped_index = assign_forecast_session_index(
            published_dt, cutoff_times
        )
        if mapped_index is None:
            continue
        mapped_session = all_sessions[mapped_index]
        local_session_index = complete_index.get(mapped_session.session_date)
        if local_session_index is None:
            continue

        title = "" if pd.isna(row.title) else str(row.title)
        description = "" if pd.isna(row.description) else str(row.description)
        text = "\n".join(part for part in (title, description) if part)
        provider_tickers = {
            value.upper() for value in parse_semicolon_set(row.provider_tickers)
        }
        keywords = parse_semicolon_set(row.keywords)
        entities = detected_entities(
            text, provider_tickers, targets, aliases
        )
        is_macro = macro_evidence(text, keywords, static_rules)
        cues = lexical_cues(text)
        age_hours = (
            mapped_session.cutoff_utc - published_dt
        ).total_seconds() / 3600
        if age_hours < 0:
            raise AssertionError("strict cutoff mapper produced a future article")
        recency_weight = math.exp(-math.log(2) * age_hours / 12)
        local_published = published_dt.astimezone(NEW_YORK)
        is_premarket = bool(
            local_published.date() == mapped_session.session_date
            and time(4, 0) <= local_published.time() < time(9, 0)
        )

        for sector in sector_names:
            sector_entities = entities & sector_target_sets[sector]
            sector_common = bool(
                benchmark_by_sector[sector] in provider_tickers
                or sector_phrase_evidence(text, sector, static_rules)
                or len(sector_entities) >= 2
            )
            if not is_macro and not sector_common and not sector_entities:
                continue

            sector_targets = targets_by_sector[sector]
            sector_entity_position = {
                ticker: position
                for position, ticker in enumerate(sector_entity_order[sector])
            }
            # Every article admitted to a sector candidate set contributes at
            # most one mention per detected peer-set entity.
            for entity in sector_entities:
                arrays["entity_mentions"][
                    local_session_index,
                    sector_index[sector],
                    sector_entity_position[entity],
                ] += 1

            for target in sector_targets:
                flags = role_flags(
                    target_ticker=target.ticker,
                    sector_entities=sector_entities,
                    sector_common=sector_common,
                    macro_common=is_macro,
                )
                if not flags["relevant"]:
                    continue
                column = target_index[target.ticker]
                for source_name, flag_name in (
                    ("n_direct", "direct"),
                    ("n_target_idio", "target_idio"),
                    ("n_target_common", "target_common"),
                    ("n_peer_idio", "peer_idio"),
                    ("n_sector_common", "sector_common"),
                    ("n_macro_common", "macro_common"),
                    ("n_any_common", "any_common"),
                ):
                    arrays[source_name][local_session_index, column] += int(
                        flags[flag_name]
                    )
                if flags["direct"]:
                    arrays["target_recency_sum"][
                        local_session_index, column
                    ] += recency_weight
                    arrays["target_premarket_count"][
                        local_session_index, column
                    ] += int(is_premarket)
                    arrays["target_peer_co_mention_count"][
                        local_session_index, column
                    ] += int(bool(sector_entities - {target.ticker}))
                    for cue, detected in cues.items():
                        arrays[f"target_{cue}_count"][
                            local_session_index, column
                        ] += int(detected)
                if flags["any_common"]:
                    arrays["common_recency_sum"][
                        local_session_index, column
                    ] += recency_weight
                    arrays["common_premarket_count"][
                        local_session_index, column
                    ] += int(is_premarket)
                    for cue, detected in cues.items():
                        arrays[f"common_{cue}_count"][
                            local_session_index, column
                        ] += int(detected)

    rows: list[dict[str, Any]] = []
    sector_entity_positions = {
        sector: {
            ticker: position
            for position, ticker in enumerate(
                sorted(target.ticker for target in targets_by_sector[sector])
            )
        }
        for sector in sector_names
    }
    for session_index, session in enumerate(complete_sessions):
        for target in targets:
            column = target_index[target.ticker]
            sector_number = sector_index[target.sector]
            entity_counts = arrays["entity_mentions"][
                session_index, sector_number
            ].astype(float)
            total_entity_mentions = float(entity_counts.sum())
            target_position = sector_entity_positions[target.sector][target.ticker]
            target_mentions = float(entity_counts[target_position])
            n_direct = int(arrays["n_direct"][session_index, column])
            n_common = int(arrays["n_any_common"][session_index, column])
            row: dict[str, Any] = {
                "forecast_date": pd.Timestamp(session.session_date),
                "sector": target.sector,
                "stock": target.ticker,
                "benchmark": target.benchmark,
                "source_query_scope_complete": 1,
                **{
                    name: int(arrays[name][session_index, column])
                    for name in ROLE_COUNT_COLUMNS
                },
                "target_recency_sum": float(
                    arrays["target_recency_sum"][session_index, column]
                ),
                "common_recency_sum": float(
                    arrays["common_recency_sum"][session_index, column]
                ),
                "target_premarket_count": int(
                    arrays["target_premarket_count"][session_index, column]
                ),
                "common_premarket_count": int(
                    arrays["common_premarket_count"][session_index, column]
                ),
                "target_peer_co_mention_count": int(
                    arrays["target_peer_co_mention_count"][
                        session_index, column
                    ]
                ),
                "peer_set_entity_mention_total": int(total_entity_mentions),
                "peer_set_entity_hhi": (
                    float(np.square(entity_counts / total_entity_mentions).sum())
                    if total_entity_mentions
                    else 0.0
                ),
                "peer_coverage_ratio": (
                    float(
                        np.count_nonzero(
                            np.delete(entity_counts, target_position) > 0
                        )
                        / 5
                    )
                ),
                "target_entity_attention_share": (
                    target_mentions / total_entity_mentions
                    if total_entity_mentions
                    else 0.0
                ),
                "target_mean_recency_weight_12h": (
                    float(arrays["target_recency_sum"][session_index, column])
                    / max(1, n_direct)
                ),
                "common_mean_recency_weight_12h": (
                    float(arrays["common_recency_sum"][session_index, column])
                    / max(1, n_common)
                ),
            }
            for cue in CUE_NAMES:
                row[f"target_{cue}_count"] = int(
                    arrays[f"target_{cue}_count"][session_index, column]
                )
                row[f"common_{cue}_count"] = int(
                    arrays[f"common_{cue}_count"][session_index, column]
                )
            rows.append(row)
    result = pd.DataFrame(rows)
    if result.duplicated(KEY_COLUMNS).any():
        raise AssertionError("daily components contain duplicate stock-days")
    return result.sort_values(KEY_COLUMNS, kind="mergesort").reset_index(drop=True)


def build_d2_panel(components: pd.DataFrame) -> pd.DataFrame:
    frame = components.copy()
    frame = frame.sort_values(["stock", "forecast_date"], kind="mergesort")
    frame["d2_target_direct_intensity_midrank_126"] = frame.groupby(
        "stock", sort=False
    )["n_direct"].transform(lambda values: prior_midrank(values.to_numpy()))
    frame["d2_peer_idio_intensity_midrank_126"] = frame.groupby(
        "stock", sort=False
    )["n_peer_idio"].transform(lambda values: prior_midrank(values.to_numpy()))
    frame["d2_common_intensity_midrank_126"] = frame.groupby(
        "stock", sort=False
    )["n_any_common"].transform(lambda values: prior_midrank(values.to_numpy()))
    frame["d2_target_attention_share_delta_63"] = frame.groupby(
        "stock", sort=False
    )["target_entity_attention_share"].transform(
        lambda values: prior_median_delta(values.to_numpy())
    )

    frame["d2_observed_no_direct_target_article"] = (
        frame["n_direct"] == 0
    ).astype(np.int8)
    frame["d2_common_article_share"] = frame["n_any_common"] / np.maximum(
        1,
        frame["n_any_common"] + frame["n_target_idio"] + frame["n_peer_idio"],
    )
    frame["d2_target_common_share"] = frame["n_target_common"] / np.maximum(
        1, frame["n_direct"]
    )
    frame["d2_observed_no_peer_set_entity_mention"] = (
        frame["peer_set_entity_mention_total"] == 0
    ).astype(np.int8)
    frame["d2_peer_set_entity_hhi"] = frame["peer_set_entity_hhi"]
    frame["d2_peer_coverage_ratio"] = frame["peer_coverage_ratio"]
    frame["d2_target_peer_co_mention_share"] = frame[
        "target_peer_co_mention_count"
    ] / np.maximum(1, frame["n_direct"])
    frame["d2_macro_share_of_common"] = frame["n_macro_common"] / np.maximum(
        1, frame["n_any_common"]
    )
    frame["d2_target_mean_recency_weight_12h"] = frame[
        "target_mean_recency_weight_12h"
    ]
    frame["d2_common_mean_recency_weight_12h"] = frame[
        "common_mean_recency_weight_12h"
    ]
    frame["d2_target_premarket_article_share"] = frame[
        "target_premarket_count"
    ] / np.maximum(1, frame["n_direct"])
    frame["d2_common_premarket_article_share"] = frame[
        "common_premarket_count"
    ] / np.maximum(1, frame["n_any_common"])

    for cue in CUE_NAMES:
        frame[f"d2_target_{cue}_article_share"] = frame[
            f"target_{cue}_count"
        ] / np.maximum(1, frame["n_direct"])
        frame[f"d2_common_{cue}_article_share"] = frame[
            f"common_{cue}_count"
        ] / np.maximum(1, frame["n_any_common"])

    frame["d2_log1p_target_idio_article_count"] = np.log1p(
        frame["n_target_idio"]
    )
    frame["d2_log1p_target_common_article_count"] = np.log1p(
        frame["n_target_common"]
    )
    frame["d2_log1p_peer_idio_article_count"] = np.log1p(
        frame["n_peer_idio"]
    )
    frame["d2_log1p_sector_common_article_count"] = np.log1p(
        frame["n_sector_common"]
    )
    frame["d2_log1p_macro_common_article_count"] = np.log1p(
        frame["n_macro_common"]
    )

    eligible = frame[D2_NORMALIZED_30].notna().all(axis=1)
    frame = frame.loc[eligible].copy()
    frame["source_profile"] = SOURCE_PROFILE
    frame["entity_map_mode"] = ENTITY_MAP_MODE
    frame["research_peer_set"] = 1
    frame["source_query_scope_complete"] = 1
    frame["untickered_macro_coverage_complete"] = 0
    frame["point_in_time_version_safe"] = 0
    frame["primary_training_eligible"] = 0
    frame["exploratory_fit_eligible"] = 1
    ordered = KEY_COLUMNS + AUDIT_COLUMNS + D2_NORMALIZED_30 + D2_LEVELS_5
    result = frame[ordered].sort_values(KEY_COLUMNS, kind="mergesort")
    result = result.reset_index(drop=True)
    if result.duplicated(KEY_COLUMNS).any():
        raise AssertionError("D2 panel contains duplicate stock-days")
    numeric = result[D2_NORMALIZED_30 + D2_LEVELS_5]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise AssertionError("D2 panel contains non-finite fitting values")
    return result


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
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


def atomic_text(value: str, path: Path) -> None:
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


def schema_records(frame: pd.DataFrame) -> list[dict[str, str]]:
    return [
        {"name": column, "dtype": str(frame[column].dtype)}
        for column in frame.columns
    ]


def quant_match_audit(
    d2_panel: pd.DataFrame, quant_panel_path: Path
) -> dict[str, Any]:
    if not quant_panel_path.exists():
        return {
            "quant_panel_available": False,
            "full_match": False,
            "reason": "quant panel path does not exist",
        }
    quant = pd.read_parquet(quant_panel_path, columns=KEY_COLUMNS)
    quant["forecast_date"] = pd.to_datetime(quant["forecast_date"])
    d2_keys = d2_panel[KEY_COLUMNS].copy()
    d2_keys["forecast_date"] = pd.to_datetime(d2_keys["forecast_date"])
    merged = quant.merge(
        d2_keys.assign(_d2_match=1),
        how="left",
        on=KEY_COLUMNS,
        validate="one_to_one",
    )
    matched = int(merged["_d2_match"].fillna(0).sum())
    return {
        "quant_panel_available": True,
        "quant_panel_path": display_path(quant_panel_path),
        "quant_panel_sha256": sha256_file(quant_panel_path),
        "quant_rows": int(len(quant)),
        "matched_rows": matched,
        "unmatched_rows": int(len(quant) - matched),
        "match_fraction": float(matched / len(quant)) if len(quant) else 0.0,
        "full_match": bool(matched == len(quant)),
        "first_date": str(quant["forecast_date"].min().date()),
        "last_date": str(quant["forecast_date"].max().date()),
    }


def build(
    *,
    articles_path: Path = DEFAULT_ARTICLES,
    source_manifest_path: Path = DEFAULT_SOURCE_MANIFEST,
    calendar_path: Path = DEFAULT_CALENDAR,
    universe_path: Path = DEFAULT_UNIVERSE,
    static_rules_path: Path = DEFAULT_RULES,
    quant_panel_path: Path = DEFAULT_QUANT_PANEL,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    output_end: date = date(2026, 6, 30),
) -> dict[str, Any]:
    source_manifest = validate_source_manifest(
        source_manifest_path, articles_path
    )
    targets = load_targets(universe_path)
    static_rules = load_static_rules(static_rules_path, targets)
    all_sessions = load_sessions(calendar_path)
    if output_end > all_sessions[-1].session_date:
        raise ValueError("output_end exceeds the available official calendar")

    # The raw request starts at 2016-06-22. The 2016-06-22 stock-day window
    # starts at the prior session's cutoff and is therefore not fully covered.
    # The next official session is the first complete strict-cutoff window.
    collection_start = date.fromisoformat(
        str(source_manifest["forecast_contract"]["start"])
    )
    start_position = next(
        index
        for index, session in enumerate(all_sessions)
        if session.session_date >= collection_start
    )
    if start_position + 1 >= len(all_sessions):
        raise ValueError("collection start has no following complete session")
    first_complete_date = all_sessions[start_position + 1].session_date
    complete_sessions = [
        session
        for session in all_sessions
        if first_complete_date <= session.session_date <= output_end
    ]
    if len(complete_sessions) <= 126:
        raise ValueError("fewer than 127 complete source sessions")

    articles = pd.read_csv(articles_path, low_memory=False)
    components = build_daily_components(
        articles,
        targets,
        static_rules,
        all_sessions,
        complete_sessions,
    )
    d2_panel = build_d2_panel(components)

    expected_component_rows = len(complete_sessions) * len(targets)
    expected_d2_rows = (len(complete_sessions) - 126) * len(targets)
    if len(components) != expected_component_rows:
        raise AssertionError("unexpected daily-component row count")
    if len(d2_panel) != expected_d2_rows:
        raise AssertionError("unexpected D2 row count")

    component_path = output_directory / "daily_role_components.parquet"
    panel_path = output_directory / "stock_day_features.parquet"
    atomic_parquet(components, component_path)
    atomic_parquet(d2_panel, panel_path)
    match_audit = quant_match_audit(d2_panel, quant_panel_path)

    source_collection = source_manifest["collection_completeness"]
    manifest: dict[str, Any] = {
        "manifest_version": "d2-normalized-manifest-v1",
        "builder_version": BUILDER_VERSION,
        "status": "complete_exploratory_non_version_safe",
        "generated_at_utc": datetime.now(UTC)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "source_profile": SOURCE_PROFILE,
        "information_time_contract": {
            "cutoff_et": "09:00",
            "timezone": "America/New_York",
            "window": "[previous official-session 09:00 ET, current 09:00 ET)",
            "right_boundary_strict": True,
            "exact_cutoff_assignment": "next official session",
            "availability_proxy": "published_at_utc",
        },
        "claim_scope": {
            "exploratory_only": True,
            "point_in_time_version_safe": False,
            "primary_training_eligible": False,
            "exploratory_fit_eligible": True,
            "entity_map_mode": ENTITY_MAP_MODE,
            "research_peer_set": True,
            "historical_index_membership_claimed": False,
        },
        "source_completeness": {
            "collection_manifest_complete": True,
            "all_page_hashes_match": True,
            "all_configured_query_roots_complete": True,
            "ordinary_ticker_queries_complete": bool(
                source_collection["ordinary_ticker_collection_complete"]
            ),
            "sector_benchmark_queries_complete": bool(
                source_collection["sector_benchmark_collection_complete"]
            ),
            "spy_control_query_complete": bool(
                source_collection["control_collection_complete"]
            ),
            "source_profile_complete_for_requested_interval": True,
            "untickered_macro_coverage_complete": False,
            "first_raw_query_date": collection_start.isoformat(),
            "first_complete_strict_window_session": (
                first_complete_date.isoformat()
            ),
            "output_end": output_end.isoformat(),
            "complete_session_count": len(complete_sessions),
            "d2_history_requirement_sessions": 126,
            "d2_first_eligible_session": str(
                d2_panel["forecast_date"].min().date()
            ),
            "all_model_rows_eligible": bool(match_audit.get("full_match")),
            "expected_model_row_count": int(
                match_audit.get("quant_rows", 0)
            ),
        },
        "counts": {
            "normalized_provider_articles": int(len(articles)),
            "daily_component_rows": int(len(components)),
            "d2_stock_day_rows": int(len(d2_panel)),
            "d2_dates": int(d2_panel["forecast_date"].nunique()),
            "targets": int(d2_panel["stock"].nunique()),
            "sectors": int(d2_panel["sector"].nunique()),
        },
        "ordered_feature_lists": {
            "keys": KEY_COLUMNS,
            "audit_columns": AUDIT_COLUMNS,
            "d2_normalized_30": D2_NORMALIZED_30,
            "d2_levels_5": D2_LEVELS_5,
        },
        "input_files": {
            display_path(articles_path): sha256_file(articles_path),
            display_path(source_manifest_path): sha256_file(
                source_manifest_path
            ),
            display_path(calendar_path): sha256_file(calendar_path),
            display_path(universe_path): sha256_file(universe_path),
            display_path(static_rules_path): sha256_file(static_rules_path),
            display_path(Path(__file__)): sha256_file(Path(__file__)),
            display_path(Path(v1_rules.__file__)): sha256_file(
                Path(v1_rules.__file__)
            ),
            display_path(Path(coarse_rules.__file__)): sha256_file(
                Path(coarse_rules.__file__)
            ),
        },
        "generated_files": {
            "daily_role_components.parquet": {
                "path": display_path(component_path),
                "sha256": sha256_file(component_path),
                "bytes": component_path.stat().st_size,
                "rows": int(len(components)),
                "columns": int(len(components.columns)),
            },
            "stock_day_features.parquet": {
                "path": display_path(panel_path),
                "sha256": sha256_file(panel_path),
                "bytes": panel_path.stat().st_size,
                "rows": int(len(d2_panel)),
                "columns": int(len(d2_panel.columns)),
            },
        },
        "schema": schema_records(d2_panel),
        "quant_panel_match_audit": match_audit,
        "source_limitations": {
            "point_in_time_version_safe": False,
            "primary_training_eligible": False,
            "first_seen_available": False,
            "last_updated_available": False,
            "historical_article_versions_available": False,
            "untickered_macro_coverage_complete": False,
            "effective_dated_entity_map_available": False,
        },
        "d43_recomputed": {
            "status": "omitted",
            "reason": (
                "The v1 D43 artifact uses a different exact-cutoff rule and "
                "broad article assignment. Reusing it would not be a matched "
                "comparator; a separate strict-window rebuild of the exact "
                "v1 contract is required."
            ),
        },
        "limitations": {
            "static_aliases_not_effective_dated": True,
            "static_peer_membership_not_effective_dated": True,
            "article_versions_recoverable": False,
            "last_updated_available": False,
            "first_seen_available": False,
            "full_public_macro_universe_observed": False,
            "spy_query_or_tag_used_as_macro_evidence": False,
            "saved_v1_article_target_roles_reused": False,
            "saved_v1_macro_cue_reused": False,
            "family_cues_recomputed_from_text_only": True,
        },
    }
    if not match_audit.get("full_match"):
        raise AssertionError("D2 panel does not fully match the quant panel")

    manifest_path = output_directory / "manifest.json"
    manifest_text = json.dumps(
        manifest, indent=2, ensure_ascii=False, sort_keys=True
    ) + "\n"
    atomic_text(manifest_text, manifest_path)
    atomic_text(
        sha256_bytes(manifest_text.encode("utf-8")) + "\n",
        output_directory / "manifest.sha256",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--articles", type=Path, default=DEFAULT_ARTICLES)
    parser.add_argument(
        "--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST
    )
    parser.add_argument("--calendar", type=Path, default=DEFAULT_CALENDAR)
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--static-rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--quant-panel", type=Path, default=DEFAULT_QUANT_PANEL)
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    parser.add_argument(
        "--output-end",
        type=date.fromisoformat,
        default=date(2026, 6, 30),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build(
        articles_path=args.articles,
        source_manifest_path=args.source_manifest,
        calendar_path=args.calendar,
        universe_path=args.universe,
        static_rules_path=args.static_rules,
        quant_panel_path=args.quant_panel,
        output_directory=args.output_directory,
        output_end=args.output_end,
    )
    print(
        canonical_json(
            {
                "status": manifest["status"],
                "d2_stock_day_rows": manifest["counts"]["d2_stock_day_rows"],
                "quant_full_match": manifest["quant_panel_match_audit"][
                    "full_match"
                ],
                "output": manifest["generated_files"][
                    "stock_day_features.parquet"
                ]["path"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
