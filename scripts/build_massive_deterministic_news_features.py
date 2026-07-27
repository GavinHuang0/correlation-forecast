"""Build deterministic daily features from Massive ordinary-news pages.

The builder is a resume-free, network-free transformation of
``pages/<query-ticker>/*.json.gz`` produced by ``fetch_massive_news.py``.  It
deduplicates by Massive's provider article ID, retains every query-page
provenance link, assigns publication timestamps to the exact 09:00
America/New_York forecast cutoff, and reuses the repository's frozen
deterministic feature rules.

Ordinary Massive news has a stable article ID, second-precision publication
time, headline, description, publisher, ticker tags, keywords, and optional
insights.  It does not provide historical first-seen/update versions, article
bodies, or causal event clusters.  Outputs therefore remain ineligible for
strict point-in-time primary training even when all 30 ticker queries are
complete.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import defaultdict
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

import build_deterministic_news_features as base  # noqa: E402


UTC = timezone.utc
BUILDER_VERSION = "massive-deterministic-news-v1.0.2"
DEFAULT_RAW_DIRECTORY = (
    REPOSITORY_ROOT
    / "data"
    / "raw"
    / "massive"
    / "ordinary_news"
    / "free_v1"
)
DEFAULT_UNIVERSE = (
    REPOSITORY_ROOT / "config" / "news_target_universe_30.json"
)
DEFAULT_CALENDAR = (
    REPOSITORY_ROOT
    / "data"
    / "prices"
    / "alpaca"
    / "calendar"
    / "2016-01-01_2026-06-30.json"
)
DEFAULT_OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT
    / "data"
    / "features"
    / "news_deterministic"
    / "massive_v1"
)
DEFAULT_START = date(2016, 6, 22)
DEFAULT_END = date(2026, 6, 30)
OPTIONAL_BENCHMARK_TICKERS = frozenset({"SOXX", "XLF", "XLE", "XLV", "XLI"})
OPTIONAL_CONTROL_TICKERS = frozenset({"SPY"})

EXPECTED_SECTOR_BENCHMARKS = {
    "Semiconductors": "SOXX",
    "Financials": "XLF",
    "Energy": "XLE",
    "Health Care": "XLV",
    "Industrials": "XLI",
}

PROTECTED_RELATIVE_ROOTS = (
    Path("outputs/flan_t5"),
    Path("outputs/flan_t5_xl"),
    Path("outputs/llama_2"),
    Path("outputs/llama_3_1"),
    Path("experiments/flan_t5"),
    Path("experiments/flan_t5_xl"),
    Path("experiments/llama_2"),
    Path("experiments/llama_3_1"),
    Path(".venv-flan-t5-xl"),
    Path(".venv-llama2"),
    Path(".venv-llama31"),
)

NORMALIZED_ARTICLE_FIELDS = (
    "article_id",
    "provider_article_id",
    "published_at_utc",
    "conservative_available_at_utc",
    "timestamp_precision",
    "title",
    "description",
    "article_url",
    "amp_url",
    "author",
    "source",
    "publisher_name",
    "publisher_homepage_url",
    "publisher_host",
    "url_sha256",
    "text_sha256",
    "normalized_title_cluster_id",
    "query_copy_count",
    "query_tickers",
    "query_files",
    "provider_tickers",
    "insight_tickers",
    "vendor_tickers",
    "keywords",
    "insights_json",
    "vendor_ticker_count",
    "positive_surprise_cue",
    "negative_surprise_cue",
    "earnings_guidance_cue",
    "product_demand_cue",
    "supply_capacity_cue",
    "regulation_legal_cue",
    "corporate_analyst_cue",
    "macro_market_cue",
    "government_primary_source",
    "press_release_wire_source",
)

COVERAGE_FIELDS = (
    "raw_root",
    "file_name",
    "root_relative_file",
    "query_kind",
    "query_value",
    "page_number",
    "raw_row_count",
    "unique_provider_id_count",
    "duplicate_provider_id_rows_within_page",
    "unique_url_count",
    "hit_1000_result_limit",
    "has_next_page",
    "earliest_time_published",
    "latest_time_published",
    "request_manifest_available",
    "collector_page_record_available",
    "collector_page_hash_matches",
    "response_sha256",
    "json_sha256",
)

CONSTRUCTABLE_Q_PLUS_D_FEATURES = (
    "observed_relevant_article_count",
    "observed_direct_target_article_count",
    "observed_target_only_article_count",
    "observed_peer_article_count",
    "observed_peer_specific_article_count",
    "observed_common_proxy_article_count",
    "observed_macro_article_count",
    "observed_mixed_target_common_article_count",
    "observed_no_relevant_news",
    "observed_multi_ticker_article_share",
    "observed_target_peer_co_mention_share",
    "observed_unique_peer_count",
    "observed_peer_coverage_ratio",
    "observed_sector_firms_with_news_share",
    "observed_sector_entity_hhi",
    "observed_target_share_of_sector_entity_mentions",
    "observed_common_shock_balance",
    "observed_firm_common_imbalance",
    "positive_surprise_cue_article_count",
    "negative_surprise_cue_article_count",
    "earnings_guidance_cue_article_count",
    "product_demand_cue_article_count",
    "supply_capacity_cue_article_count",
    "regulation_legal_cue_article_count",
    "corporate_analyst_cue_article_count",
    "macro_market_cue_article_count",
    "timing_eligible_article_count",
    "timing_imprecise_article_count",
    "timing_eligible_share",
    "hours_since_latest_precise_target_article",
    "hours_since_latest_precise_common_article",
    "recency_weighted_precise_target_count_12h",
    "recency_weighted_precise_common_count_12h",
    "same_day_premarket_precise_article_share",
    "prior_session_afterhours_precise_article_share",
    "unique_source_count",
    "source_entropy",
    "normalized_title_cluster_count",
    "normalized_title_duplicate_ratio",
    "max_normalized_title_cluster_size",
    "max_normalized_title_cluster_source_count",
    "government_primary_source_share",
    "press_release_wire_source_share",
    "observed_target_news_burst_60_session",
)

STRICTLY_UNCONSTRUCTABLE_FEATURES = (
    "version_safe_no_news",
    "first_seen_based_recency",
    "last_updated_based_recency",
    "historical_article_version_features",
    "full_body_deterministic_cues",
    "causal_event_count",
    "causal_event_cluster_size",
    "causal_cross_source_propagation",
    "complete_untagged_macro_news_count",
)


def parse_massive_timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Invalid Massive published_utc: {value!r}")
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"Invalid Massive published_utc: {value!r}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"Massive published_utc has no timezone: {value!r}")
    return parsed.astimezone(UTC).replace(microsecond=0)


def _string_list(value: Any, *, field: str, path: Path) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{path}: article {field} must be an array")
    return sorted(
        {
            str(item).strip()
            for item in value
            if isinstance(item, (str, int, float)) and str(item).strip()
        }
    )


def normalize_insights(value: Any, *, path: Path) -> list[dict[str, str]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{path}: article insights must be an array")
    by_json: dict[str, dict[str, str]] = {}
    for raw in value:
        if not isinstance(raw, Mapping):
            raise ValueError(f"{path}: article insight must be an object")
        normalized = {
            "ticker": str(raw.get("ticker", "")).strip().upper(),
            "sentiment": str(raw.get("sentiment", "")).strip().lower(),
            "sentiment_reasoning": str(
                raw.get("sentiment_reasoning", "")
            ).strip(),
        }
        key = base.canonical_json(normalized)
        by_json[key] = normalized
    return [by_json[key] for key in sorted(by_json)]


def normalize_keyword_topic(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def normalize_massive_article(
    raw: Mapping[str, Any],
    *,
    path: Path,
    query_ticker: str,
    relative_path: str,
) -> dict[str, Any]:
    provider_id = str(raw.get("id", "")).strip()
    title = str(raw.get("title", "")).strip()
    if not provider_id or not title:
        raise ValueError(f"{path}: every article requires id and title")
    published_at = parse_massive_timestamp(raw.get("published_utc"))
    description = str(raw.get("description") or "").strip()
    article_url = str(raw.get("article_url") or "").strip()
    amp_url = str(raw.get("amp_url") or "").strip()
    author = str(raw.get("author") or "").strip()

    publisher_raw = raw.get("publisher")
    if publisher_raw is None:
        publisher_raw = {}
    if not isinstance(publisher_raw, Mapping):
        raise ValueError(f"{path}: article publisher must be an object")
    publisher_name = str(publisher_raw.get("name") or "").strip()
    publisher_homepage = str(
        publisher_raw.get("homepage_url") or ""
    ).strip()
    host = base.publisher_host(publisher_homepage or article_url)
    source = publisher_name or host

    provider_tickers = {
        ticker.upper()
        for ticker in _string_list(raw.get("tickers"), field="tickers", path=path)
    }
    keywords = set(
        _string_list(raw.get("keywords"), field="keywords", path=path)
    )
    insights = normalize_insights(raw.get("insights"), path=path)
    insight_tickers = {
        insight["ticker"] for insight in insights if insight["ticker"]
    }
    all_tickers = provider_tickers | insight_tickers
    topics = {
        normalized
        for keyword in keywords
        if (normalized := normalize_keyword_topic(keyword))
    }
    if "SPY" in all_tickers or query_ticker == "SPY":
        topics.add("economy_macro")
    combined_text = "\n".join(
        part for part in (title, description) if part
    )
    _, positive, negative = base.coarse.explicit_surprise_from_text(
        combined_text
    )
    family_cues = base.event_family_cues(combined_text, topics)
    normalized_title = base.normalize_title(title)
    published_text = base.format_utc(published_at)
    return {
        "article_id": f"massive_{provider_id}",
        "provider_article_id": provider_id,
        "url": article_url,
        "article_url": article_url,
        "amp_url": amp_url,
        "title": title,
        "description": description,
        "summary": description,
        "text": combined_text,
        "author": author,
        "published_at": published_at,
        "published_at_utc": published_text,
        "available_at": published_at,
        "conservative_available_at_utc": published_text,
        "timestamp_precision": "second",
        "source": source,
        "publisher_name": publisher_name,
        "publisher_homepage_url": publisher_homepage,
        "publisher_host": host,
        "topics": set(topics),
        "keywords": set(keywords),
        "provider_tickers": set(provider_tickers),
        "insight_tickers": set(insight_tickers),
        "tickers": set(all_tickers),
        "insights_by_json": {
            base.canonical_json(insight): insight for insight in insights
        },
        "query_tickers": {query_ticker},
        "query_files": {relative_path},
        "query_copy_count": 1,
        "url_sha256": base.sha256_bytes(article_url.encode("utf-8")),
        "text_sha256": base.sha256_bytes(combined_text.encode("utf-8")),
        "normalized_title_cluster_id": (
            "title_"
            + base.sha256_bytes(normalized_title.encode("utf-8"))[:20]
        ),
        "positive_surprise_cue": positive,
        "negative_surprise_cue": negative,
        "family_cues": family_cues,
        "government_primary_source": base.host_matches(
            host, base.GOVERNMENT_PRIMARY_HOST_SUFFIXES
        ),
        "press_release_wire_source": base.host_matches(
            host, base.PRESS_RELEASE_WIRE_HOSTS
        ),
    }


def merge_duplicate_article(
    existing: dict[str, Any],
    incoming: Mapping[str, Any],
    *,
    path: Path,
) -> None:
    immutable_fields = (
        "title",
        "description",
        "article_url",
        "amp_url",
        "author",
        "published_at_utc",
        "publisher_name",
        "publisher_homepage_url",
    )
    for field in immutable_fields:
        if existing[field] != incoming[field]:
            raise ValueError(
                f"{path}: conflicting {field} for Massive ID "
                f"{existing['provider_article_id']!r}"
            )
    for field in (
        "topics",
        "keywords",
        "provider_tickers",
        "insight_tickers",
        "tickers",
        "query_tickers",
        "query_files",
    ):
        existing[field].update(incoming[field])
    existing["insights_by_json"].update(incoming["insights_by_json"])
    existing["query_copy_count"] += 1


def _page_number(path: Path) -> int:
    match = re.fullmatch(r"page_(\d+)\.json\.gz", path.name)
    return int(match.group(1)) if match else 0


def load_collector_manifest(raw_root: Path) -> dict[str, Any] | None:
    path = raw_root / "manifest.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: collector manifest must be an object")
    return payload


def _coerce_raw_roots(raw_roots: Path | Sequence[Path]) -> list[Path]:
    roots = [raw_roots] if isinstance(raw_roots, Path) else list(raw_roots)
    if not roots:
        raise ValueError("At least one Massive raw root is required")
    resolved = [root.resolve() for root in roots]
    if len(resolved) != len(set(resolved)):
        raise ValueError("Massive raw roots must be unique")
    return roots


def load_massive_articles(
    raw_roots: Path | Sequence[Path],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[Path],
    list[dict[str, Any]],
]:
    roots = _coerce_raw_roots(raw_roots)
    by_id: dict[str, dict[str, Any]] = {}
    coverage: list[dict[str, Any]] = []
    all_paths: list[Path] = []
    collector_sources: list[dict[str, Any]] = []
    for raw_root in roots:
        paths = sorted((raw_root / "pages").glob("*/*.json.gz"))
        if not paths:
            raise FileNotFoundError(
                f"No Massive JSON gzip pages found under {raw_root / 'pages'}"
            )
        all_paths.extend(paths)
        root_label = base.display_path(raw_root)
        collector_manifest = load_collector_manifest(raw_root)
        manifest_pages: dict[str, Mapping[str, Any]] = {}
        if collector_manifest is not None:
            raw_manifest_pages = collector_manifest.get("pages", [])
            if isinstance(raw_manifest_pages, list):
                manifest_pages = {
                    str(record.get("path", "")): record
                    for record in raw_manifest_pages
                    if isinstance(record, Mapping)
                }
        root_coverage_paths: list[str] = []
        for path in paths:
            query_ticker = path.parent.name.strip().upper()
            if not re.fullmatch(r"[A-Z][A-Z0-9.\-]*", query_ticker):
                raise ValueError(f"{path}: invalid query-ticker directory")
            local_relative = path.relative_to(raw_root).as_posix()
            provenance = f"{root_label}::{local_relative}"
            root_coverage_paths.append(local_relative)
            compressed = path.read_bytes()
            try:
                raw_json = gzip.decompress(compressed)
                payload = json.loads(raw_json.decode("utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path}: invalid gzip JSON page") from exc
            if not isinstance(payload, Mapping):
                raise ValueError(f"{path}: page payload must be an object")
            results = payload.get("results")
            if not isinstance(results, list):
                raise ValueError(f"{path}: page payload has no results array")
            provider_ids: list[str] = []
            urls: set[str] = set()
            timestamps: list[str] = []
            for raw_article in results:
                if not isinstance(raw_article, Mapping):
                    raise ValueError(f"{path}: result row must be an object")
                article = normalize_massive_article(
                    raw_article,
                    path=path,
                    query_ticker=query_ticker,
                    relative_path=provenance,
                )
                provider_id = str(article["provider_article_id"])
                provider_ids.append(provider_id)
                if article["article_url"]:
                    urls.add(str(article["article_url"]))
                timestamps.append(str(article["published_at_utc"]))
                existing = by_id.get(provider_id)
                if existing is None:
                    by_id[provider_id] = article
                else:
                    merge_duplicate_article(existing, article, path=path)

            manifest_record = manifest_pages.get(local_relative)
            compressed_hash = base.sha256_bytes(compressed)
            manifest_hash = (
                str(manifest_record.get("gzip_sha256", ""))
                if manifest_record is not None
                else ""
            )
            hash_matches = bool(
                manifest_record is not None and manifest_hash == compressed_hash
            )
            if manifest_record is not None and not hash_matches:
                raise ValueError(f"{path}: collector manifest hash mismatch")
            coverage.append(
                {
                    "raw_root": root_label,
                    "file_name": provenance,
                    "root_relative_file": local_relative,
                    "query_kind": "ticker",
                    "query_value": query_ticker,
                    "page_number": _page_number(path),
                    "raw_row_count": len(results),
                    "unique_provider_id_count": len(set(provider_ids)),
                    "duplicate_provider_id_rows_within_page": (
                        len(provider_ids) - len(set(provider_ids))
                    ),
                    "unique_url_count": len(urls),
                    # A full Massive page is complete when its recorded cursor
                    # was followed, unlike Alpha's unpageable ceiling.
                    "hit_1000_result_limit": 0,
                    "has_next_page": int(bool(payload.get("next_url"))),
                    "earliest_time_published": (
                        min(timestamps) if timestamps else ""
                    ),
                    "latest_time_published": (
                        max(timestamps) if timestamps else ""
                    ),
                    "request_manifest_available": int(
                        collector_manifest is not None
                    ),
                    "collector_page_record_available": int(
                        manifest_record is not None
                    ),
                    "collector_page_hash_matches": int(hash_matches),
                    "response_sha256": compressed_hash,
                    "json_sha256": base.sha256_bytes(raw_json),
                }
            )
        collector_sources.append(
            {
                "raw_root": raw_root,
                "raw_root_label": root_label,
                "manifest": collector_manifest,
                "coverage_paths": sorted(root_coverage_paths),
            }
        )

    articles = sorted(
        by_id.values(),
        key=lambda article: (
            article["published_at"],
            article["provider_article_id"],
        ),
    )
    for article in articles:
        article["topics"] = sorted(article["topics"])
        article["keywords"] = sorted(article["keywords"])
        article["provider_tickers"] = sorted(article["provider_tickers"])
        article["insight_tickers"] = sorted(article["insight_tickers"])
        article["tickers"] = sorted(article["tickers"])
        article["query_tickers"] = sorted(article["query_tickers"])
        article["query_files"] = sorted(article["query_files"])
        article["insights"] = [
            article["insights_by_json"][key]
            for key in sorted(article["insights_by_json"])
        ]
    return articles, coverage, all_paths, collector_sources


def load_targets(path: Path) -> list[base.Target]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_targets = payload.get("targets") if isinstance(payload, Mapping) else None
    if not isinstance(raw_targets, Mapping):
        raise ValueError(f"{path}: targets must be an object")
    if len(raw_targets) != 30:
        raise ValueError(
            f"{path}: Massive deterministic news requires exactly 30 targets"
        )
    targets: list[base.Target] = []
    for raw_ticker, metadata in raw_targets.items():
        if not isinstance(metadata, Mapping):
            raise ValueError(f"{path}: target metadata must be an object")
        ticker = str(raw_ticker).strip().upper()
        company = str(metadata.get("company", "")).strip()
        sector = str(metadata.get("sector", "")).strip()
        benchmark = str(metadata.get("benchmark", "")).strip().upper()
        peers_raw = metadata.get("peers")
        if (
            not ticker
            or not company
            or sector not in EXPECTED_SECTOR_BENCHMARKS
            or benchmark != EXPECTED_SECTOR_BENCHMARKS.get(sector)
            or not isinstance(peers_raw, list)
        ):
            raise ValueError(f"{path}: invalid target metadata for {ticker!r}")
        peers = tuple(str(peer).strip().upper() for peer in peers_raw)
        targets.append(
            base.Target(
                ticker=ticker,
                company=company,
                sector=sector,
                benchmark=benchmark,
                peers=peers,
            )
        )

    by_sector: dict[str, set[str]] = defaultdict(set)
    for target in targets:
        by_sector[target.sector].add(target.ticker)
    if set(by_sector) != set(EXPECTED_SECTOR_BENCHMARKS):
        raise ValueError(f"{path}: target sectors do not match the fixed five")
    for sector, tickers in by_sector.items():
        if len(tickers) != 6:
            raise ValueError(f"{path}: sector {sector!r} must contain six stocks")
    for target in targets:
        expected_peers = by_sector[target.sector] - {target.ticker}
        if set(target.peers) != expected_peers or len(target.peers) != 5:
            raise ValueError(
                f"{path}: {target.ticker} peers must be the other five "
                f"{target.sector} stocks"
            )
    tickers = [target.ticker for target in targets]
    if len(tickers) != len(set(tickers)):
        raise ValueError(f"{path}: duplicate target ticker")
    return sorted(targets, key=lambda target: target.ticker)


def assess_collection(
    *,
    collector_sources: Sequence[Mapping[str, Any]],
    coverage: Sequence[Mapping[str, Any]],
    targets: Sequence[base.Target],
    start: date,
    end: date,
) -> dict[str, Any]:
    target_tickers = {target.ticker for target in targets}
    complete_query_tickers: set[str] = set()
    all_completed_tickers: set[str] = set()
    root_assessments: list[dict[str, Any]] = []
    for source in collector_sources:
        root_label = str(source["raw_root_label"])
        collector_manifest = source.get("manifest")
        root_rows = [
            row for row in coverage if str(row["raw_root"]) == root_label
        ]
        page_tickers = {str(row["query_value"]) for row in root_rows}
        if not isinstance(collector_manifest, Mapping):
            root_assessments.append(
                {
                    "raw_root": root_label,
                    "manifest_available": False,
                    "status": "missing",
                    "scope_covers_build_dates": False,
                    "all_manifest_pages_present": False,
                    "all_page_hashes_match": False,
                    "complete_query_tickers": [],
                }
            )
            continue
        scope = collector_manifest.get("scope", {})
        scope_tickers = (
            {str(value).upper() for value in scope.get("tickers", [])}
            if isinstance(scope, Mapping)
            else set()
        )
        try:
            scope_start = date.fromisoformat(str(scope.get("start", "")))
            scope_end = date.fromisoformat(str(scope.get("end", "")))
            dates_covered = scope_start <= start and scope_end >= end
        except ValueError:
            dates_covered = False
        completed = {
            str(value).upper()
            for value in collector_manifest.get("completed_tickers", [])
        }
        all_completed_tickers.update(completed)
        manifest_pages = collector_manifest.get("pages", [])
        manifest_paths = (
            {
                str(record.get("path", ""))
                for record in manifest_pages
                if isinstance(record, Mapping)
            }
            if isinstance(manifest_pages, list)
            else set()
        )
        coverage_paths = set(source["coverage_paths"])
        pages_present = bool(coverage_paths) and manifest_paths == coverage_paths
        page_hashes_match = bool(root_rows) and all(
            bool(row["collector_page_hash_matches"]) for row in root_rows
        )
        # ``completed_tickers`` is resumable per-query state.  Tickers already
        # completed in an otherwise in-progress root are individually usable.
        root_complete = bool(
            dates_covered and pages_present and page_hashes_match
        )
        root_complete_tickers = (
            scope_tickers & completed & page_tickers if root_complete else set()
        )
        complete_query_tickers.update(root_complete_tickers)
        root_assessments.append(
            {
                "raw_root": root_label,
                "manifest_available": True,
                "status": str(collector_manifest.get("status", "")),
                "scope_covers_build_dates": dates_covered,
                "all_manifest_pages_present": pages_present,
                "all_page_hashes_match": page_hashes_match,
                "complete_query_tickers": sorted(root_complete_tickers),
            }
        )

    collection_complete = target_tickers.issubset(complete_query_tickers)
    benchmark_complete = OPTIONAL_BENCHMARK_TICKERS.issubset(
        complete_query_tickers
    )
    control_complete = OPTIONAL_CONTROL_TICKERS.issubset(
        complete_query_tickers
    )
    return {
        "collector_manifest_available": any(
            assessment["manifest_available"]
            for assessment in root_assessments
        ),
        "all_collector_manifests_available": all(
            assessment["manifest_available"]
            for assessment in root_assessments
        ),
        "collector_roots": root_assessments,
        "completed_tickers": sorted(all_completed_tickers),
        "complete_query_tickers": sorted(complete_query_tickers),
        "complete_tickers": sorted(target_tickers & complete_query_tickers),
        "ordinary_ticker_collection_complete": collection_complete,
        "sector_benchmark_collection_complete": benchmark_complete,
        "control_collection_complete": control_complete,
        "benchmark_and_control_collection_complete": (
            benchmark_complete and control_complete
        ),
    }


def normalized_article_rows(
    articles: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for article in articles:
        rows.append(
            {
                "article_id": article["article_id"],
                "provider_article_id": article["provider_article_id"],
                "published_at_utc": article["published_at_utc"],
                "conservative_available_at_utc": article[
                    "conservative_available_at_utc"
                ],
                "timestamp_precision": article["timestamp_precision"],
                "title": article["title"],
                "description": article["description"],
                "article_url": article["article_url"],
                "amp_url": article["amp_url"],
                "author": article["author"],
                "source": article["source"],
                "publisher_name": article["publisher_name"],
                "publisher_homepage_url": article["publisher_homepage_url"],
                "publisher_host": article["publisher_host"],
                "url_sha256": article["url_sha256"],
                "text_sha256": article["text_sha256"],
                "normalized_title_cluster_id": article[
                    "normalized_title_cluster_id"
                ],
                "query_copy_count": article["query_copy_count"],
                "query_tickers": ";".join(article["query_tickers"]),
                "query_files": ";".join(article["query_files"]),
                "provider_tickers": ";".join(article["provider_tickers"]),
                "insight_tickers": ";".join(article["insight_tickers"]),
                "vendor_tickers": ";".join(article["tickers"]),
                "keywords": ";".join(article["keywords"]),
                "insights_json": base.canonical_json(article["insights"]),
                "vendor_ticker_count": len(article["tickers"]),
                "positive_surprise_cue": article["positive_surprise_cue"],
                "negative_surprise_cue": article["negative_surprise_cue"],
                **{
                    f"{family}_cue": article["family_cues"][family]
                    for family in base.EVENT_FAMILY_PATTERNS
                },
                "government_primary_source": article[
                    "government_primary_source"
                ],
                "press_release_wire_source": article[
                    "press_release_wire_source"
                ],
            }
        )
    return rows


def _group_targets(
    targets: Sequence[base.Target],
) -> dict[str, list[base.Target]]:
    grouped: dict[str, list[base.Target]] = defaultdict(list)
    for target in targets:
        grouped[target.sector].append(target)
    return {
        sector: sorted(values, key=lambda target: target.ticker)
        for sector, values in sorted(grouped.items())
    }


def build_daily_features(
    *,
    articles: Sequence[dict[str, Any]],
    coverage: Sequence[Mapping[str, Any]],
    targets: Sequence[base.Target],
    sessions: Sequence[base.Session],
    collection: Mapping[str, Any],
    start: date,
    end: date,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_pairs: list[dict[str, Any]] = []
    all_stock_days: list[dict[str, Any]] = []
    complete_tickers = set(collection["complete_tickers"])
    complete_queries = set(collection["complete_query_tickers"])
    for sector, sector_targets in _group_targets(targets).items():
        pair_rows, mapped_articles = base.build_pair_rows(
            articles,
            sector_targets,
            sessions,
            start,
            end,
        )
        stock_days = base.aggregate_stock_days(
            pair_rows,
            mapped_articles,
            sector_targets,
            sessions,
            coverage,
            start,
            end,
        )
        sector_stock_tickers_complete = all(
            target.ticker in complete_queries for target in sector_targets
        )
        sector_benchmark = sector_targets[0].benchmark
        sector_benchmark_complete = sector_benchmark in complete_queries
        for row in stock_days:
            row["archive_scope"] = (
                "massive_ordinary_news_fixed_30_stock_v1"
            )
            row["point_in_time_version_safe"] = 0
            row["common_news_coverage_complete"] = int(
                sector_stock_tickers_complete and sector_benchmark_complete
            )
            row["macro_news_coverage_complete"] = 0
            row["primary_training_eligible"] = 0
            row["query_manifest_available"] = int(
                collection["all_collector_manifests_available"]
            )
            row["ordinary_ticker_news_collection_complete"] = int(
                collection["ordinary_ticker_collection_complete"]
            )
            row["target_ticker_query_complete"] = int(
                str(row["stock"]) in complete_tickers
            )
            row["sector_stock_ticker_queries_complete"] = int(
                sector_stock_tickers_complete
            )
            row["sector_benchmark_query_complete"] = int(
                sector_benchmark_complete
            )
            row["control_query_complete"] = int(
                OPTIONAL_CONTROL_TICKERS.issubset(complete_queries)
            )
            row.pop("target_ticker_query_below_1000_limit", None)
        all_pairs.extend(pair_rows)
        all_stock_days.extend(stock_days)

    all_pairs.sort(
        key=lambda row: (
            str(row["forecast_date"]),
            str(row["target_ticker"]),
            str(row["article_id"]),
        )
    )
    all_stock_days.sort(
        key=lambda row: (str(row["forecast_date"]), str(row["stock"]))
    )
    return all_pairs, all_stock_days


def validate_output_directory(path: Path) -> None:
    resolved = path.resolve()
    for relative in PROTECTED_RELATIVE_ROOTS:
        protected = (REPOSITORY_ROOT / relative).resolve()
        try:
            resolved.relative_to(protected)
        except ValueError:
            continue
        raise ValueError(
            "Refusing to write Massive deterministic news output under "
            f"protected model path {protected}"
        )


def existing_output_paths(output_directory: Path) -> list[Path]:
    return [
        output_directory / "normalized_articles.csv.gz",
        output_directory / "article_target_features.csv.gz",
        output_directory / "stock_day_features.csv.gz",
        output_directory / "coverage_audit.csv",
        output_directory / "manifest.json",
    ]


def _source_file_hashes(
    paths: Iterable[Path],
) -> dict[str, str]:
    return {
        base.display_path(path): base.sha256_file(path)
        for path in sorted({path.resolve() for path in paths}, key=str)
    }


def build(
    *,
    raw_roots: Sequence[Path] = (DEFAULT_RAW_DIRECTORY,),
    universe_path: Path = DEFAULT_UNIVERSE,
    calendar_path: Path = DEFAULT_CALENDAR,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    start: date = DEFAULT_START,
    end: date = DEFAULT_END,
    cutoff_et: time = time(9, 0),
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    if start > end:
        raise ValueError("start must not be after end")
    validate_output_directory(output_directory)
    if not overwrite and not dry_run:
        conflicts = [
            path for path in existing_output_paths(output_directory) if path.exists()
        ]
        if conflicts:
            raise FileExistsError(
                "Output already exists; pass --overwrite for a deterministic "
                "full rebuild: " + ", ".join(str(path) for path in conflicts)
            )

    targets = load_targets(universe_path)
    normalized_raw_roots = _coerce_raw_roots(raw_roots)
    articles, coverage, raw_paths, collector_sources = load_massive_articles(
        normalized_raw_roots
    )
    sessions = base.load_sessions(calendar_path, cutoff_et)
    if not any(start <= session.date <= end for session in sessions):
        raise ValueError("Calendar has no sessions in the requested date range")
    collection = assess_collection(
        collector_sources=collector_sources,
        coverage=coverage,
        targets=targets,
        start=start,
        end=end,
    )
    pair_rows, stock_days = build_daily_features(
        articles=articles,
        coverage=coverage,
        targets=targets,
        sessions=sessions,
        collection=collection,
        start=start,
        end=end,
    )
    article_rows = normalized_article_rows(articles)
    if stock_days:
        missing_features = set(CONSTRUCTABLE_Q_PLUS_D_FEATURES) - set(
            stock_days[0]
        )
        if missing_features:
            raise AssertionError(
                f"Constructable feature declaration is stale: {missing_features}"
            )

    source_paths = [*raw_paths, universe_path, calendar_path]
    for raw_root in normalized_raw_roots:
        collector_manifest_path = raw_root / "manifest.json"
        if collector_manifest_path.exists():
            source_paths.append(collector_manifest_path)
    manifest: dict[str, Any] = {
        "builder_version": BUILDER_VERSION,
        "build_mode": "resume_free_deterministic_full_rebuild",
        "builder_script_sha256": base.sha256_file(Path(__file__)),
        "reused_deterministic_builder_sha256": base.sha256_file(
            Path(base.__file__)
        ),
        "coarse_rule_version": base.coarse.DETERMINISTIC_RULE_VERSION,
        "coarse_rules_sha256": base.coarse.DETERMINISTIC_RULES_SHA256,
        "status": (
            "ordinary_collection_complete_but_not_version_safe"
            if collection["ordinary_ticker_collection_complete"]
            else "partial_ordinary_collection_not_version_safe"
        ),
        "forecast_contract": {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "cutoff_et": cutoff_et.strftime("%H:%M"),
            "timezone": "America/New_York",
            "news_window": "(previous trading-session cutoff, current cutoff]",
            "availability_proxy": "published_utc truncated to whole seconds",
            "exact_cutoff_policy": "published exactly at 09:00 ET is included",
        },
        "collection_completeness": collection,
        "data_limitations": {
            "stable_provider_article_id_available": True,
            "published_utc_second_precision_available": True,
            "description_available": True,
            "publisher_metadata_available": True,
            "ticker_tags_available": True,
            "keywords_available": True,
            "insights_available_when_returned": True,
            "first_seen_available": False,
            "last_updated_available": False,
            "historical_version_history_available": False,
            "full_article_body_available": False,
            "causal_event_clustering_available": False,
            "untagged_macro_news_coverage_complete": False,
            "point_in_time_version_safe": False,
            "primary_training_eligible": False,
        },
        "counts": {
            "raw_roots": len(normalized_raw_roots),
            "raw_query_rows": sum(int(row["raw_row_count"]) for row in coverage),
            "unique_provider_articles": len(articles),
            "provider_id_duplicate_rows_removed": (
                sum(int(row["raw_row_count"]) for row in coverage) - len(articles)
            ),
            "cross_query_articles": sum(
                len(article["query_tickers"]) > 1 for article in articles
            ),
            "article_target_rows": len(pair_rows),
            "stock_day_rows": len(stock_days),
            "targets": len(targets),
            "sectors": len({target.sector for target in targets}),
        },
        "constructable_q_plus_d_features": list(
            CONSTRUCTABLE_Q_PLUS_D_FEATURES
        ),
        "strictly_unconstructable_features": list(
            STRICTLY_UNCONSTRUCTABLE_FEATURES
        ),
        "feature_interpretation": {
            "article_counts": (
                "provider-ID-deduplicated ordinary-news articles, not events"
            ),
            "description_text": (
                "deterministic lexical cues use headline plus description only"
            ),
            "timing": (
                "publication-time proxy; first-seen/update history is unavailable"
            ),
            "commonality": (
                "fixed six-stock within-sector peer-basket proxy; a separate "
                "sector-benchmark query root improves tagged common-shock coverage"
            ),
            "macro": (
                "observed ticker-tagged macro articles only; untagged macro "
                "coverage is absent"
            ),
            "normalized_title_clusters": (
                "exact normalized-title duplication proxy, not causal clustering"
            ),
            "insights": (
                "preserved in normalized articles but not promoted to frozen "
                "Q+D features"
            ),
            "primary_training_eligibility": (
                "always false for strict version-safe claims, independently "
                "of ordinary ticker-query collection completeness"
            ),
        },
        "source_files": _source_file_hashes(source_paths),
        "generated_files": {},
    }
    if dry_run:
        manifest["dry_run"] = True
        return manifest

    output_directory.mkdir(parents=True, exist_ok=True)
    article_path = output_directory / "normalized_articles.csv.gz"
    pair_path = output_directory / "article_target_features.csv.gz"
    stock_day_path = output_directory / "stock_day_features.csv.gz"
    coverage_path = output_directory / "coverage_audit.csv"
    manifest_path = output_directory / "manifest.json"

    base.atomic_write_csv_gz(
        article_path, article_rows, NORMALIZED_ARTICLE_FIELDS
    )
    base.atomic_write_csv_gz(
        pair_path, pair_rows, base.PAIR_OUTPUT_FIELDS
    )
    stock_day_fields = tuple(stock_days[0]) if stock_days else ()
    base.atomic_write_csv_gz(stock_day_path, stock_days, stock_day_fields)
    base.atomic_write_csv(coverage_path, coverage, COVERAGE_FIELDS)
    manifest["generated_files"] = {
        path.name: {
            "sha256": base.sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in (article_path, pair_path, stock_day_path, coverage_path)
    }
    base.atomic_write_json(manifest_path, manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        dest="raw_roots",
        action="append",
        type=Path,
        help=(
            "Massive collector root containing pages/ and manifest.json; "
            "repeat for stock and benchmark/control scopes"
        ),
    )
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--calendar", type=Path, default=DEFAULT_CALENDAR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--start", type=date.fromisoformat, default=DEFAULT_START)
    parser.add_argument("--end", type=date.fromisoformat, default=DEFAULT_END)
    parser.add_argument(
        "--cutoff-et", type=base.parse_time_of_day, default=time(9, 0)
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build(
        raw_roots=args.raw_roots or [DEFAULT_RAW_DIRECTORY],
        universe_path=args.universe,
        calendar_path=args.calendar,
        output_directory=args.output_dir,
        start=args.start,
        end=args.end,
        cutoff_et=args.cutoff_et,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
