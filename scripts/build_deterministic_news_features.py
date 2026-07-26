"""Build an isolated deterministic-news pilot from cached Alpha Vantage data.

This builder is deliberately CPU-only and never imports Torch or Transformers.
It reads canonical raw-news responses, exact-deduplicates cross-query overlap,
and creates article metadata, article-target features, stock-day aggregates,
and a coverage audit under a dedicated output directory.

The current repository archive is incomplete.  Generated stock-day features
therefore use an ``observed_*`` prefix and are explicitly marked ineligible for
the primary training experiment.  The output is useful for validating the
deterministic-news pipeline, not as a point-in-time production news panel.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import hashlib
import html
import io
import json
import math
import os
import re
import statistics
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse
from zoneinfo import ZoneInfo


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

import coarse_news_features as coarse  # noqa: E402


NEW_YORK = ZoneInfo("America/New_York")
UTC = timezone.utc
BUILDER_VERSION = "deterministic-news-pilot-v1.0.0"
DEFAULT_RAW_DIRECTORY = REPOSITORY_ROOT / "data" / "raw" / "alpha_vantage_news"
DEFAULT_UNIVERSE = REPOSITORY_ROOT / "config" / "target_universe.json"
DEFAULT_CALENDAR = (
    REPOSITORY_ROOT
    / "data"
    / "prices"
    / "alpaca"
    / "calendar"
    / "2016-01-01_2026-06-30.json"
)
DEFAULT_OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT / "data" / "features" / "news_deterministic" / "v1_0"
)

PROTECTED_RELATIVE_ROOTS = (
    Path("outputs/flan_t5"),
    Path("outputs/flan_t5_xl"),
    Path("experiments/flan_t5_xl"),
    Path(".venv-flan-t5-xl"),
)

MACRO_TOPICS = frozenset({"economy_macro", "economy_monetary"})
PRESS_RELEASE_WIRE_HOSTS = frozenset(
    {
        "businesswire.com",
        "globenewswire.com",
        "prnewswire.com",
    }
)
GOVERNMENT_PRIMARY_HOST_SUFFIXES = (
    "bea.gov",
    "bls.gov",
    "commerce.gov",
    "congress.gov",
    "federalreserve.gov",
    "ftc.gov",
    "justice.gov",
    "sec.gov",
    "treasury.gov",
    "whitehouse.gov",
)

EVENT_FAMILY_PATTERNS: Mapping[str, tuple[str, ...]] = {
    "earnings_guidance": (
        r"\bearnings\b",
        r"\brevenue\b",
        r"\bprofit\b",
        r"\bquarterly results?\b",
        r"\bguidance\b",
        r"\boutlook\b",
        r"\bforecast\b",
    ),
    "product_demand": (
        r"\bproduct\b",
        r"\btechnology\b",
        r"\bcustomer\b",
        r"\bcontract\b",
        r"\bdemand\b",
        r"\borders?\b",
        r"\blaunch(?:es|ed|ing)?\b",
    ),
    "supply_capacity": (
        r"\bsupply chain\b",
        r"\bcapacity\b",
        r"\binventory\b",
        r"\bproduction\b",
        r"\bmanufactur(?:e|ing)\b",
        r"\bfoundr(?:y|ies)\b",
        r"\bfabs?\b",
        r"\bshortage\b",
    ),
    "regulation_legal": (
        r"\bregulat(?:ion|ory|or)\b",
        r"\bexport controls?\b",
        r"\btrade polic(?:y|ies)\b",
        r"\btariffs?\b",
        r"\blawsuits?\b",
        r"\blitigation\b",
        r"\binvestigation\b",
        r"\bantitrust\b",
        r"\blegal\b",
    ),
    "corporate_analyst": (
        r"\banalysts?\b",
        r"\bupgrad(?:e|ed|es)\b",
        r"\bdowngrad(?:e|ed|es)\b",
        r"\bprice target\b",
        r"\bmerger\b",
        r"\bacquisition\b",
        r"\bbuyback\b",
        r"\bdividend\b",
        r"\brestructur(?:e|ed|ing)\b",
        r"\bchief executive\b",
        r"\bceo\b",
    ),
    "macro_market": (
        r"\bfederal reserve\b",
        r"\bfomc\b",
        r"\binflation\b",
        r"\binterest rates?\b",
        r"\btreasury yields?\b",
        r"\bunemployment\b",
        r"\bnonfarm payrolls?\b",
        r"\bgross domestic product\b",
        r"\brecession\b",
    ),
}

TOPIC_TO_EVENT_FAMILY: Mapping[str, str] = {
    "earnings": "earnings_guidance",
    "technology": "product_demand",
    "manufacturing": "supply_capacity",
    "mergers_and_acquisitions": "corporate_analyst",
    "economy_macro": "macro_market",
    "economy_monetary": "macro_market",
}

ARTICLE_OUTPUT_FIELDS = (
    "article_id",
    "published_at_utc",
    "conservative_available_at_utc",
    "timestamp_precision",
    "source",
    "publisher_host",
    "url_sha256",
    "text_sha256",
    "normalized_title_cluster_id",
    "query_copy_count",
    "query_files",
    "vendor_tickers",
    "vendor_topics",
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

PAIR_OUTPUT_FIELDS = (
    "article_id",
    "forecast_date",
    "target_ticker",
    "sector",
    "benchmark",
    "published_at_utc",
    "conservative_available_at_utc",
    "timestamp_precision",
    "direct_target_evidence",
    "peer_evidence",
    "sector_or_peer_evidence",
    "explicit_sector_evidence",
    "macro_evidence",
    "common_proxy",
    "peer_specific_proxy",
    "target_only_proxy",
    "mixed_target_common_proxy",
    "vendor_multi_ticker",
    "target_peer_co_mention",
    "peer_entities",
    "sector_entities",
    "positive_surprise_cue",
    "negative_surprise_cue",
    "earnings_guidance_cue",
    "product_demand_cue",
    "supply_capacity_cue",
    "regulation_legal_cue",
    "corporate_analyst_cue",
    "macro_market_cue",
    "source",
    "publisher_host",
    "normalized_title_cluster_id",
    "government_primary_source",
    "press_release_wire_source",
    "timing_eligible",
    "hours_before_cutoff",
    "same_day_premarket",
    "prior_session_afterhours",
)


@dataclass(frozen=True)
class Target:
    ticker: str
    company: str
    sector: str
    benchmark: str
    peers: tuple[str, ...]


@dataclass(frozen=True)
class Session:
    date: date
    open_time: time
    close_time: time
    cutoff_utc: datetime


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


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def format_utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_time_of_day(value: str) -> time:
    return datetime.strptime(value, "%H:%M").time()


def parse_alpha_timestamp(value: Any) -> tuple[datetime, str]:
    if not isinstance(value, str) or not re.fullmatch(r"\d{8}T\d{6}", value):
        raise ValueError(f"Invalid Alpha Vantage timestamp: {value!r}")
    parsed = datetime.strptime(value, "%Y%m%dT%H%M%S").replace(tzinfo=UTC)
    precision = "date_only_proxy" if parsed.time() == time(0, 0) else "second"
    return parsed, precision


def conservative_available_at(
    published_at: datetime, timestamp_precision: str
) -> datetime:
    if timestamp_precision == "date_only_proxy":
        return datetime.combine(
            published_at.date() + timedelta(days=1),
            time(0, 0),
            tzinfo=UTC,
        )
    return published_at


def publisher_host(url: str) -> str:
    host = urlparse(url).hostname or ""
    host = host.lower().rstrip(".")
    return host.removeprefix("www.")


def host_matches(host: str, suffixes: Iterable[str]) -> bool:
    return any(host == suffix or host.endswith(f".{suffix}") for suffix in suffixes)


def normalize_title(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", html.unescape(value)).lower()
    normalized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


def text_contains_literal(text: str, value: str, *, case_sensitive: bool) -> bool:
    flags = 0 if case_sensitive else re.IGNORECASE
    escaped = re.escape(value.strip())
    if not escaped:
        return False
    return re.search(
        rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])",
        text,
        flags,
    ) is not None


def event_family_cues(text: str, topics: Iterable[str]) -> dict[str, bool]:
    topic_set = set(topics)
    result = {
        family: any(re.search(pattern, text, re.IGNORECASE | re.DOTALL) for pattern in patterns)
        for family, patterns in EVENT_FAMILY_PATTERNS.items()
    }
    for topic in topic_set:
        family = TOPIC_TO_EVENT_FAMILY.get(topic)
        if family is not None:
            result[family] = True
    return result


def query_metadata(path: Path, feed: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ticker_match = re.search(r"_ticker_([A-Za-z0-9.\-]+)\.json$", path.name)
    topic_match = re.search(r"_topic_([A-Za-z0-9_.\-]+)\.json$", path.name)
    query_kind = "ticker" if ticker_match else "topic" if topic_match else "unknown"
    query_value = (
        ticker_match.group(1).upper()
        if ticker_match
        else topic_match.group(1)
        if topic_match
        else ""
    )
    timestamps = sorted(
        str(record.get("time_published"))
        for record in feed
        if record.get("time_published")
    )
    return {
        "file_name": path.name,
        "query_kind": query_kind,
        "query_value": query_value,
        "raw_row_count": len(feed),
        "unique_url_count": len({str(record.get("url", "")).strip() for record in feed}),
        "hit_1000_result_limit": int(len(feed) >= 1000),
        "earliest_time_published": timestamps[0] if timestamps else "",
        "latest_time_published": timestamps[-1] if timestamps else "",
        "request_manifest_available": 0,
        "response_sha256": sha256_file(path),
    }


def load_raw_articles(
    raw_directory: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[Path]]:
    paths = sorted(raw_directory.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No JSON news payloads found under {raw_directory}")

    by_url: dict[str, dict[str, Any]] = {}
    coverage: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        feed = payload.get("feed")
        if not isinstance(feed, list):
            raise ValueError(f"{path} has no feed array")
        coverage.append(query_metadata(path, feed))
        for raw in feed:
            if not isinstance(raw, Mapping):
                raise ValueError(f"{path} contains a non-object feed row")
            url = str(raw.get("url", "")).strip()
            title = str(raw.get("title", "")).strip()
            timestamp = raw.get("time_published")
            if not url or not title:
                raise ValueError(f"{path} contains an article without URL/title")
            published_at, precision = parse_alpha_timestamp(timestamp)
            summary_value = raw.get("summary")
            summary = summary_value.strip() if isinstance(summary_value, str) else ""
            source = str(raw.get("source", "")).strip()
            topics = sorted(
                {
                    str(item.get("topic", "")).strip()
                    for item in raw.get("topics", [])
                    if isinstance(item, Mapping) and str(item.get("topic", "")).strip()
                }
            )
            tickers = sorted(
                {
                    str(item.get("ticker", "")).strip().upper()
                    for item in raw.get("ticker_sentiment", [])
                    if isinstance(item, Mapping) and str(item.get("ticker", "")).strip()
                }
            )

            existing = by_url.get(url)
            if existing is None:
                combined_text = "\n".join(part for part in (title, summary) if part)
                surprise, positive, negative = coarse.explicit_surprise_from_text(
                    combined_text
                )
                del surprise
                family_cues = event_family_cues(combined_text, topics)
                host = publisher_host(url)
                available_at = conservative_available_at(published_at, precision)
                title_cluster = normalize_title(title)
                by_url[url] = {
                    "article_id": f"avurl_{sha256_bytes(url.encode('utf-8'))[:20]}",
                    "url": url,
                    "title": title,
                    "summary": summary,
                    "text": combined_text,
                    "published_at": published_at,
                    "published_at_utc": format_utc(published_at),
                    "available_at": available_at,
                    "conservative_available_at_utc": format_utc(available_at),
                    "timestamp_precision": precision,
                    "source": source,
                    "publisher_host": host,
                    "topics": set(topics),
                    "tickers": set(tickers),
                    "query_files": {path.name},
                    "query_copy_count": 1,
                    "url_sha256": sha256_bytes(url.encode("utf-8")),
                    "text_sha256": sha256_bytes(combined_text.encode("utf-8")),
                    "normalized_title_cluster_id": (
                        f"title_{sha256_bytes(title_cluster.encode('utf-8'))[:20]}"
                    ),
                    "positive_surprise_cue": positive,
                    "negative_surprise_cue": negative,
                    "family_cues": family_cues,
                    "government_primary_source": host_matches(
                        host, GOVERNMENT_PRIMARY_HOST_SUFFIXES
                    ),
                    "press_release_wire_source": host_matches(
                        host, PRESS_RELEASE_WIRE_HOSTS
                    ),
                }
                continue

            core_values = (
                ("title", title),
                ("summary", summary),
                ("published_at_utc", format_utc(published_at)),
                ("source", source),
            )
            for field, value in core_values:
                if existing[field] != value:
                    raise ValueError(
                        f"Conflicting {field} for duplicate URL {url!r} "
                        f"between query payloads"
                    )
            existing["topics"].update(topics)
            existing["tickers"].update(tickers)
            existing["query_files"].add(path.name)
            existing["query_copy_count"] += 1

    articles = sorted(by_url.values(), key=lambda row: (row["published_at"], row["url"]))
    for article in articles:
        article["topics"] = sorted(article["topics"])
        article["tickers"] = sorted(article["tickers"])
        article["query_files"] = sorted(article["query_files"])
    return articles, coverage, paths


def load_targets(path: Path) -> list[Target]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sector = str(payload["sector"])
    benchmark = str(payload["sector_benchmark"]).upper()
    targets: list[Target] = []
    for ticker, metadata in payload["targets"].items():
        targets.append(
            Target(
                ticker=str(ticker).upper(),
                company=str(metadata["company"]),
                sector=sector,
                benchmark=benchmark,
                peers=tuple(str(value).upper() for value in metadata["peers"]),
            )
        )
    if not targets:
        raise ValueError("Target universe is empty")
    return sorted(targets, key=lambda target: target.ticker)


def load_sessions(path: Path, cutoff_et: time) -> list[Session]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_sessions = payload.get("sessions")
    if not isinstance(raw_sessions, list):
        raise ValueError(f"{path} has no sessions array")
    sessions: list[Session] = []
    for raw in raw_sessions:
        session_date = date.fromisoformat(str(raw["date"]))
        cutoff_local = datetime.combine(session_date, cutoff_et, tzinfo=NEW_YORK)
        sessions.append(
            Session(
                date=session_date,
                open_time=parse_time_of_day(str(raw["open"])),
                close_time=parse_time_of_day(str(raw["close"])),
                cutoff_utc=cutoff_local.astimezone(UTC),
            )
        )
    sessions.sort(key=lambda session: session.date)
    return sessions


def assign_forecast_session(
    available_at: datetime,
    sessions: Sequence[Session],
) -> Session | None:
    cutoffs = [session.cutoff_utc for session in sessions]
    index = bisect.bisect_left(cutoffs, available_at.astimezone(UTC))
    return sessions[index] if index < len(sessions) else None


def build_aliases(targets: Sequence[Target]) -> dict[str, tuple[str, ...]]:
    all_tickers = sorted(
        {
            target.ticker
            for target in targets
        }
        | {
            peer
            for target in targets
            for peer in target.peers
        }
    )
    company_by_ticker = {target.ticker: target.company for target in targets}
    aliases: dict[str, tuple[str, ...]] = {}
    for ticker in all_tickers:
        values = []
        if ticker in company_by_ticker:
            values.append(company_by_ticker[ticker])
        values.extend(coarse.COMPANY_ALIASES_BY_TICKER.get(ticker, ()))
        aliases[ticker] = tuple(dict.fromkeys(value for value in values if value))
    return aliases


def detected_sector_entities(
    text: str,
    vendor_tickers: Iterable[str],
    aliases: Mapping[str, Sequence[str]],
) -> set[str]:
    tagged = set(vendor_tickers)
    detected: set[str] = set()
    for ticker, names in aliases.items():
        if ticker in tagged or text_contains_literal(text, ticker, case_sensitive=True):
            detected.add(ticker)
            continue
        if any(
            text_contains_literal(text, name, case_sensitive=False)
            for name in names
        ):
            detected.add(ticker)
    return detected


def previous_session_by_date(sessions: Sequence[Session]) -> dict[date, Session | None]:
    result: dict[date, Session | None] = {}
    prior: Session | None = None
    for session in sessions:
        result[session.date] = prior
        prior = session
    return result


def build_pair_rows(
    articles: Sequence[dict[str, Any]],
    targets: Sequence[Target],
    sessions: Sequence[Session],
    start: date,
    end: date,
) -> tuple[list[dict[str, Any]], dict[date, list[dict[str, Any]]]]:
    aliases = build_aliases(targets)
    previous_by_date = previous_session_by_date(sessions)
    pair_rows: list[dict[str, Any]] = []
    mapped_articles: dict[date, list[dict[str, Any]]] = defaultdict(list)

    for article in articles:
        session = assign_forecast_session(article["available_at"], sessions)
        if session is None or session.date < start or session.date > end:
            continue
        sector_entities = detected_sector_entities(
            article["text"],
            article["tickers"],
            aliases,
        )
        article["forecast_date"] = session.date
        article["sector_entities"] = sorted(sector_entities)
        mapped_articles[session.date].append(article)

        local_published = article["published_at"].astimezone(NEW_YORK)
        prior_session = previous_by_date.get(session.date)
        timing_eligible = article["timestamp_precision"] == "second"
        hours_before_cutoff = (
            (session.cutoff_utc - article["published_at"]).total_seconds() / 3600
            if timing_eligible
            else None
        )
        same_day_premarket = bool(
            timing_eligible
            and local_published.date() == session.date
            and time(4, 0) <= local_published.time() < time(9, 0)
        )
        prior_afterhours = bool(
            timing_eligible
            and prior_session is not None
            and local_published.date() == prior_session.date
            and prior_session.close_time <= local_published.time() < time(20, 0)
        )

        macro_topic = bool(set(article["topics"]) & MACRO_TOPICS)
        for target in targets:
            input_record = {
                "article_id": article["article_id"],
                "headline": article["title"],
                "article_text": article["summary"],
                "vendor_tickers": article["tickers"],
                "target": {
                    "company": target.company,
                    "ticker": target.ticker,
                    "sector": target.sector,
                    "sector_benchmark": target.benchmark,
                    "known_sector_peers": list(target.peers),
                },
            }
            deterministic = coarse.deterministic_features(input_record)
            direct = bool(deterministic["direct_target_evidence"])
            peer_entities = sorted(
                set(deterministic["peer_vendor_tickers"])
                | set(deterministic["peer_entities_in_text"])
            )
            peer_evidence = bool(peer_entities)
            sector_or_peer = bool(deterministic["sector_or_peer_evidence"])
            explicit_sector = bool(
                deterministic["benchmark_vendor_tagged"]
                or deterministic["sector_benchmark_in_text"]
                or deterministic["sector_terms_in_text"]
            )
            macro = bool(deterministic["macro_evidence"] or macro_topic)
            common = bool(
                macro
                or explicit_sector
                or len(sector_entities) >= 2
            )
            peer_specific = bool(peer_evidence and not common)
            relevant = bool(direct or peer_evidence or common)
            if not relevant:
                continue
            target_only = bool(
                direct
                and not common
                and sector_entities.issubset({target.ticker})
            )
            family_cues = article["family_cues"]
            pair_rows.append(
                {
                    "article_id": article["article_id"],
                    "forecast_date": session.date.isoformat(),
                    "target_ticker": target.ticker,
                    "sector": target.sector,
                    "benchmark": target.benchmark,
                    "published_at_utc": article["published_at_utc"],
                    "conservative_available_at_utc": article[
                        "conservative_available_at_utc"
                    ],
                    "timestamp_precision": article["timestamp_precision"],
                    "direct_target_evidence": direct,
                    "peer_evidence": peer_evidence,
                    "sector_or_peer_evidence": sector_or_peer,
                    "explicit_sector_evidence": explicit_sector,
                    "macro_evidence": macro,
                    "common_proxy": common,
                    "peer_specific_proxy": peer_specific,
                    "target_only_proxy": target_only,
                    "mixed_target_common_proxy": bool(direct and common),
                    "vendor_multi_ticker": len(article["tickers"]) >= 2,
                    "target_peer_co_mention": bool(direct and peer_evidence),
                    "peer_entities": ";".join(peer_entities),
                    "sector_entities": ";".join(sorted(sector_entities)),
                    "positive_surprise_cue": article["positive_surprise_cue"],
                    "negative_surprise_cue": article["negative_surprise_cue"],
                    **{
                        f"{family}_cue": family_cues[family]
                        for family in EVENT_FAMILY_PATTERNS
                    },
                    "source": article["source"],
                    "publisher_host": article["publisher_host"],
                    "normalized_title_cluster_id": article[
                        "normalized_title_cluster_id"
                    ],
                    "government_primary_source": article[
                        "government_primary_source"
                    ],
                    "press_release_wire_source": article[
                        "press_release_wire_source"
                    ],
                    "timing_eligible": timing_eligible,
                    "hours_before_cutoff": hours_before_cutoff,
                    "same_day_premarket": same_day_premarket,
                    "prior_session_afterhours": prior_afterhours,
                }
            )
    pair_rows.sort(
        key=lambda row: (
            row["forecast_date"],
            row["target_ticker"],
            row["article_id"],
        )
    )
    return pair_rows, mapped_articles


def safe_share(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def source_entropy(rows: Sequence[Mapping[str, Any]]) -> float:
    counts = Counter(str(row["source"]) for row in rows)
    total = sum(counts.values())
    if not total:
        return 0.0
    return -sum(
        (count / total) * math.log(count / total)
        for count in counts.values()
        if count
    )


def aggregate_stock_days(
    pair_rows: Sequence[dict[str, Any]],
    mapped_articles: Mapping[date, Sequence[dict[str, Any]]],
    targets: Sequence[Target],
    sessions: Sequence[Session],
    coverage: Sequence[Mapping[str, Any]],
    start: date,
    end: date,
) -> list[dict[str, Any]]:
    rows_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        rows_by_key[(str(row["forecast_date"]), str(row["target_ticker"]))].append(
            row
        )

    ticker_coverage = {
        str(row["query_value"]): row
        for row in coverage
        if row["query_kind"] == "ticker"
    }
    all_sector_tickers = sorted(
        {target.ticker for target in targets}
        | {peer for target in targets for peer in target.peers}
    )
    output: list[dict[str, Any]] = []
    selected_sessions = [
        session for session in sessions if start <= session.date <= end
    ]
    for session in selected_sessions:
        date_key = session.date.isoformat()
        day_articles = list(mapped_articles.get(session.date, ()))
        sector_entity_counts: Counter[str] = Counter()
        for article in day_articles:
            for ticker in article.get("sector_entities", ()):
                sector_entity_counts[str(ticker)] += 1
        total_sector_entity_mentions = sum(sector_entity_counts.values())
        sector_hhi = (
            sum(
                (count / total_sector_entity_mentions) ** 2
                for count in sector_entity_counts.values()
            )
            if total_sector_entity_mentions
            else 0.0
        )
        sector_firm_share = safe_share(
            len(sector_entity_counts),
            len(all_sector_tickers),
        )

        for target in targets:
            relevant = rows_by_key.get((date_key, target.ticker), [])
            count = len(relevant)
            direct_count = sum(bool(row["direct_target_evidence"]) for row in relevant)
            target_only_count = sum(bool(row["target_only_proxy"]) for row in relevant)
            peer_count = sum(bool(row["peer_evidence"]) for row in relevant)
            peer_specific_count = sum(
                bool(row["peer_specific_proxy"]) for row in relevant
            )
            common_count = sum(bool(row["common_proxy"]) for row in relevant)
            macro_count = sum(bool(row["macro_evidence"]) for row in relevant)
            mixed_count = sum(
                bool(row["mixed_target_common_proxy"]) for row in relevant
            )
            multi_ticker_count = sum(
                bool(row["vendor_multi_ticker"]) for row in relevant
            )
            co_mention_count = sum(
                bool(row["target_peer_co_mention"]) for row in relevant
            )
            peer_entities = {
                entity
                for row in relevant
                for entity in str(row["peer_entities"]).split(";")
                if entity
            }
            precise = [row for row in relevant if row["timing_eligible"]]
            precise_target = [
                row for row in precise if row["direct_target_evidence"]
            ]
            precise_common = [row for row in precise if row["common_proxy"]]
            positive_cue_count = sum(
                bool(row["positive_surprise_cue"]) for row in relevant
            )
            negative_cue_count = sum(
                bool(row["negative_surprise_cue"]) for row in relevant
            )
            clusters = Counter(
                str(row["normalized_title_cluster_id"]) for row in relevant
            )
            cluster_sources: dict[str, set[str]] = defaultdict(set)
            for row in relevant:
                cluster_sources[str(row["normalized_title_cluster_id"])].add(
                    str(row["source"])
                )
            query = ticker_coverage.get(target.ticker)
            target_query_present = query is not None
            target_query_below_limit = bool(
                query is not None and not query["hit_1000_result_limit"]
            )

            output.append(
                {
                    "forecast_date": date_key,
                    "sector": target.sector,
                    "stock": target.ticker,
                    "benchmark": target.benchmark,
                    "archive_scope": "2024_semiconductor_pilot",
                    "point_in_time_version_safe": 0,
                    "common_news_coverage_complete": 0,
                    "macro_news_coverage_complete": 0,
                    "primary_training_eligible": 0,
                    "target_ticker_query_present": int(target_query_present),
                    "target_ticker_query_below_1000_limit": int(
                        target_query_below_limit
                    ),
                    "query_manifest_available": 0,
                    "observed_relevant_article_count": count,
                    "observed_direct_target_article_count": direct_count,
                    "observed_target_only_article_count": target_only_count,
                    "observed_peer_article_count": peer_count,
                    "observed_peer_specific_article_count": peer_specific_count,
                    "observed_common_proxy_article_count": common_count,
                    "observed_macro_article_count": macro_count,
                    "observed_mixed_target_common_article_count": mixed_count,
                    "observed_no_relevant_news": int(count == 0),
                    "observed_multi_ticker_article_share": safe_share(
                        multi_ticker_count, count
                    ),
                    "observed_target_peer_co_mention_share": safe_share(
                        co_mention_count, count
                    ),
                    "observed_unique_peer_count": len(peer_entities),
                    "observed_peer_coverage_ratio": safe_share(
                        len(peer_entities), len(target.peers)
                    ),
                    "observed_sector_firms_with_news_share": sector_firm_share,
                    "observed_sector_entity_hhi": sector_hhi,
                    "observed_target_share_of_sector_entity_mentions": safe_share(
                        sector_entity_counts[target.ticker],
                        total_sector_entity_mentions,
                    ),
                    "observed_common_shock_balance": safe_share(
                        common_count - target_only_count,
                        1 + count,
                    ),
                    "observed_firm_common_imbalance": (
                        math.log1p(target_only_count) - math.log1p(common_count)
                    ),
                    "positive_surprise_cue_article_count": positive_cue_count,
                    "negative_surprise_cue_article_count": negative_cue_count,
                    **{
                        f"{family}_cue_article_count": sum(
                            bool(row[f"{family}_cue"]) for row in relevant
                        )
                        for family in EVENT_FAMILY_PATTERNS
                    },
                    "timing_eligible_article_count": len(precise),
                    "timing_imprecise_article_count": count - len(precise),
                    "timing_eligible_share": safe_share(len(precise), count),
                    "hours_since_latest_precise_target_article": (
                        min(float(row["hours_before_cutoff"]) for row in precise_target)
                        if precise_target
                        else None
                    ),
                    "hours_since_latest_precise_common_article": (
                        min(float(row["hours_before_cutoff"]) for row in precise_common)
                        if precise_common
                        else None
                    ),
                    "recency_weighted_precise_target_count_12h": sum(
                        math.exp(
                            -math.log(2)
                            * float(row["hours_before_cutoff"])
                            / 12.0
                        )
                        for row in precise_target
                    ),
                    "recency_weighted_precise_common_count_12h": sum(
                        math.exp(
                            -math.log(2)
                            * float(row["hours_before_cutoff"])
                            / 12.0
                        )
                        for row in precise_common
                    ),
                    "same_day_premarket_precise_article_share": safe_share(
                        sum(bool(row["same_day_premarket"]) for row in precise),
                        len(precise),
                    ),
                    "prior_session_afterhours_precise_article_share": safe_share(
                        sum(
                            bool(row["prior_session_afterhours"])
                            for row in precise
                        ),
                        len(precise),
                    ),
                    "unique_source_count": len(
                        {str(row["source"]) for row in relevant}
                    ),
                    "source_entropy": source_entropy(relevant),
                    "normalized_title_cluster_count": len(clusters),
                    "normalized_title_duplicate_ratio": safe_share(
                        count - len(clusters), count
                    ),
                    "max_normalized_title_cluster_size": (
                        max(clusters.values()) if clusters else 0
                    ),
                    "max_normalized_title_cluster_source_count": (
                        max(map(len, cluster_sources.values()))
                        if cluster_sources
                        else 0
                    ),
                    "government_primary_source_share": safe_share(
                        sum(
                            bool(row["government_primary_source"])
                            for row in relevant
                        ),
                        count,
                    ),
                    "press_release_wire_source_share": safe_share(
                        sum(
                            bool(row["press_release_wire_source"])
                            for row in relevant
                        ),
                        count,
                    ),
                    "observed_target_news_burst_60_session": None,
                    "target_news_burst_history_count": 0,
                }
            )

    apply_target_news_burst(output)
    output.sort(key=lambda row: (str(row["forecast_date"]), str(row["stock"])))
    return output


def apply_target_news_burst(
    rows: Sequence[dict[str, Any]],
    *,
    history_length: int = 60,
) -> None:
    """Add a trailing-history robust burst score in place.

    A value is emitted only when all ``history_length`` preceding stock
    sessions are available and their log-count median absolute deviation is
    nonzero.
    """

    if history_length <= 0:
        raise ValueError("history_length must be positive")
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_target[str(row["stock"])].append(row)
    for target_rows in by_target.values():
        target_rows.sort(key=lambda row: str(row["forecast_date"]))
        for index, row in enumerate(target_rows):
            history = target_rows[max(0, index - history_length) : index]
            row["target_news_burst_history_count"] = len(history)
            if len(history) < history_length:
                continue
            values = [
                math.log1p(float(item["observed_direct_target_article_count"]))
                for item in history
            ]
            median = statistics.median(values)
            mad = statistics.median(abs(value - median) for value in values)
            if mad <= 0:
                continue
            current = math.log1p(float(row["observed_direct_target_article_count"]))
            row["observed_target_news_burst_60_session"] = (
                current - median
            ) / (1.4826 * mad)


def article_output_rows(articles: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for article in articles:
        rows.append(
            {
                "article_id": article["article_id"],
                "published_at_utc": article["published_at_utc"],
                "conservative_available_at_utc": article[
                    "conservative_available_at_utc"
                ],
                "timestamp_precision": article["timestamp_precision"],
                "source": article["source"],
                "publisher_host": article["publisher_host"],
                "url_sha256": article["url_sha256"],
                "text_sha256": article["text_sha256"],
                "normalized_title_cluster_id": article[
                    "normalized_title_cluster_id"
                ],
                "query_copy_count": article["query_copy_count"],
                "query_files": ";".join(article["query_files"]),
                "vendor_tickers": ";".join(article["tickers"]),
                "vendor_topics": ";".join(article["topics"]),
                "vendor_ticker_count": len(article["tickers"]),
                "positive_surprise_cue": article["positive_surprise_cue"],
                "negative_surprise_cue": article["negative_surprise_cue"],
                **{
                    f"{family}_cue": article["family_cues"][family]
                    for family in EVENT_FAMILY_PATTERNS
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


def validate_output_directory(path: Path) -> None:
    resolved = path.resolve()
    for relative in PROTECTED_RELATIVE_ROOTS:
        protected = (REPOSITORY_ROOT / relative).resolve()
        try:
            resolved.relative_to(protected)
        except ValueError:
            continue
        raise ValueError(
            f"Refusing to write deterministic news output under protected "
            f"FLAN path {protected}"
        )


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    return value


def atomic_write_csv_gz(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as raw_handle:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw_handle,
                mtime=0,
            ) as compressed:
                with io.TextIOWrapper(
                    compressed,
                    encoding="utf-8",
                    newline="",
                ) as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=fieldnames,
                        extrasaction="raise",
                    )
                    writer.writeheader()
                    for row in rows:
                        writer.writerow(
                            {
                                field: csv_value(row.get(field))
                                for field in fieldnames
                            }
                        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {field: csv_value(row.get(field)) for field in fieldnames}
                )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def existing_output_paths(output_directory: Path) -> list[Path]:
    return [
        output_directory / "normalized_articles.csv.gz",
        output_directory / "article_target_features.csv.gz",
        output_directory / "stock_day_features.csv.gz",
        output_directory / "coverage_audit.csv",
        output_directory / "manifest.json",
    ]


def build(
    *,
    raw_directory: Path,
    universe_path: Path,
    calendar_path: Path,
    output_directory: Path,
    start: date,
    end: date,
    cutoff_et: time = time(9, 0),
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    if start > end:
        raise ValueError("start must not be after end")
    validate_output_directory(output_directory)
    if not overwrite and not dry_run:
        conflicts = [path for path in existing_output_paths(output_directory) if path.exists()]
        if conflicts:
            raise FileExistsError(
                "Output already exists; pass --overwrite to replace it atomically: "
                + ", ".join(str(path) for path in conflicts)
            )

    articles, coverage, raw_paths = load_raw_articles(raw_directory)
    targets = load_targets(universe_path)
    sessions = load_sessions(calendar_path, cutoff_et)
    pair_rows, mapped_articles = build_pair_rows(
        articles,
        targets,
        sessions,
        start,
        end,
    )
    stock_days = aggregate_stock_days(
        pair_rows,
        mapped_articles,
        targets,
        sessions,
        coverage,
        start,
        end,
    )
    normalized_rows = article_output_rows(articles)

    manifest: dict[str, Any] = {
        "builder_version": BUILDER_VERSION,
        "builder_script_sha256": sha256_file(Path(__file__)),
        "coarse_rule_version": coarse.DETERMINISTIC_RULE_VERSION,
        "coarse_rules_sha256": coarse.DETERMINISTIC_RULES_SHA256,
        "coarse_module_sha256": sha256_file(Path(coarse.__file__)),
        "event_family_patterns_sha256": sha256_bytes(
            canonical_json(
                {
                    "patterns": EVENT_FAMILY_PATTERNS,
                    "topic_mapping": TOPIC_TO_EVENT_FAMILY,
                }
            ).encode("utf-8")
        ),
        "status": "pilot_only_not_primary_training_eligible",
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "forecast_contract": {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "cutoff_et": cutoff_et.strftime("%H:%M"),
            "news_window": "(previous trading-session cutoff, current cutoff]",
            "timezone_assumption": "Alpha Vantage time_published interpreted as UTC",
            "midnight_policy": (
                "T000000 is treated as date-only; available_at is delayed to "
                "the next UTC midnight and the record is excluded from "
                "intraday timing features"
            ),
        },
        "data_limitations": {
            "point_in_time_version_safe": False,
            "request_manifests_available": False,
            "stable_provider_article_id_available": False,
            "first_seen_or_updated_at_available": False,
            "full_text_available": False,
            "entity_level_cue_attribution_available": False,
            "causal_event_clustering_available": False,
            "topic_query_coverage_complete": False,
            "multi_sector_coverage_complete": False,
            "primary_training_eligible": False,
        },
        "counts": {
            "raw_query_rows": sum(int(row["raw_row_count"]) for row in coverage),
            "unique_exact_url_articles": len(articles),
            "cross_query_duplicate_rows_removed": (
                sum(int(row["raw_row_count"]) for row in coverage) - len(articles)
            ),
            "date_only_proxy_articles": sum(
                article["timestamp_precision"] == "date_only_proxy"
                for article in articles
            ),
            "article_target_rows": len(pair_rows),
            "stock_day_rows": len(stock_days),
            "targets": len(targets),
        },
        "source_files": {
            display_path(path): sha256_file(path)
            for path in [*raw_paths, universe_path, calendar_path]
        },
        "generated_files": {},
        "feature_interpretation": {
            "article_counts": "exact-URL-deduplicated observed articles, not event counts",
            "commonality": (
                "fixed semiconductor-universe proxy; single-peer-only articles "
                "are separated from common articles"
            ),
            "propagation": "normalized-title proxy, not causal event clustering",
            "timing": "publication-time proxy; historical summary revisions are unavailable",
            "surprise_cues": (
                "un-attributed article-level lexical cues; no target/common "
                "directional alignment is emitted"
            ),
            "column_prefix": (
                "observed_* marks values that are computable from the local "
                "archive but not complete enough for production claims"
            ),
        },
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

    atomic_write_csv_gz(article_path, normalized_rows, ARTICLE_OUTPUT_FIELDS)
    atomic_write_csv_gz(pair_path, pair_rows, PAIR_OUTPUT_FIELDS)
    stock_day_fields = tuple(stock_days[0]) if stock_days else ()
    atomic_write_csv_gz(stock_day_path, stock_days, stock_day_fields)
    coverage_fields = tuple(coverage[0]) if coverage else ()
    atomic_write_csv(coverage_path, coverage, coverage_fields)
    manifest["generated_files"] = {
        path.name: {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in (article_path, pair_path, stock_day_path, coverage_path)
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIRECTORY)
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--calendar", type=Path, default=DEFAULT_CALENDAR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2024, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2024, 12, 31))
    parser.add_argument("--cutoff-et", type=parse_time_of_day, default=time(9, 0))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build(
        raw_directory=args.raw_dir,
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
