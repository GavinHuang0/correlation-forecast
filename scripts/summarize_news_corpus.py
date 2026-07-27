"""Summarize cached Massive or Alpha Vantage news without network access.

The provider collectors retain raw gzip-compressed API responses.  This
script turns those immutable pages into a compact, reproducible coverage
audit while avoiding licensed article text in the report itself.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


UTC = timezone.utc


def parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    if len(text) == 15 and text[8] == "T":
        try:
            return datetime.strptime(text, "%Y%m%dT%H%M%S").replace(tzinfo=UTC)
        except ValueError:
            return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def format_timestamp(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json_gzip(path: Path) -> Mapping[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def top_counts(counter: Counter[str], limit: int = 20) -> list[dict[str, Any]]:
    return [
        {"value": value, "count": count}
        for value, count in sorted(
            counter.items(), key=lambda item: (-item[1], item[0])
        )[:limit]
    ]


def alpha_response_at_ceiling(payload: Mapping[str, Any]) -> bool:
    records = payload.get("feed")
    if not isinstance(records, list):
        return False
    try:
        reported_items = int(str(payload.get("items", len(records))))
    except ValueError:
        reported_items = len(records)
    return len(records) >= 1_000 or reported_items >= 1_000


def _massive_identity(record: Mapping[str, Any]) -> str:
    article_id = record.get("id")
    if article_id not in (None, ""):
        return f"id:{article_id}"
    url = record.get("article_url")
    if isinstance(url, str) and url.strip():
        return f"url:{url.strip()}"
    stable = json.dumps(
        {
            "title": record.get("title"),
            "published_utc": record.get("published_utc"),
            "publisher": record.get("publisher"),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return "fallback:" + hashlib.sha256(stable.encode("utf-8")).hexdigest()


def summarize_massive(root: Path, cutoff: datetime) -> dict[str, Any]:
    paths = sorted((root / "pages").glob("*/*.json.gz"))
    query_rows = 0
    unique: dict[str, Mapping[str, Any]] = {}
    query_tickers_by_id: dict[str, set[str]] = defaultdict(set)
    query_counts: Counter[str] = Counter()
    for path in paths:
        payload = load_json_gzip(path)
        records = payload.get("results")
        if not isinstance(records, list):
            raise ValueError(f"{path}: expected a results array")
        query_ticker = path.parent.name.upper()
        for record in records:
            if not isinstance(record, Mapping):
                continue
            query_rows += 1
            query_counts[query_ticker] += 1
            identity = _massive_identity(record)
            query_tickers_by_id[identity].add(query_ticker)
            unique.setdefault(identity, record)

    timestamps: list[datetime] = []
    post_cutoff = 0
    description_count = 0
    keyword_count = 0
    insight_count = 0
    url_count = 0
    publisher_count = 0
    tagged_ticker_count = 0
    publishers: Counter[str] = Counter()
    post_cutoff_by_ticker: Counter[str] = Counter()
    unique_by_query_ticker: Counter[str] = Counter()

    for identity, record in unique.items():
        timestamp = parse_timestamp(record.get("published_utc"))
        if timestamp is not None:
            timestamps.append(timestamp)
        is_post_cutoff = timestamp is not None and timestamp > cutoff
        if is_post_cutoff:
            post_cutoff += 1
        if isinstance(record.get("description"), str) and record["description"].strip():
            description_count += 1
        if isinstance(record.get("keywords"), list) and record["keywords"]:
            keyword_count += 1
        if isinstance(record.get("insights"), list) and record["insights"]:
            insight_count += 1
        if isinstance(record.get("article_url"), str) and record["article_url"].strip():
            url_count += 1
        publisher = record.get("publisher")
        if isinstance(publisher, Mapping):
            name = publisher.get("name")
            if isinstance(name, str) and name.strip():
                publisher_count += 1
                publishers[name.strip()] += 1
        tagged = record.get("tickers")
        if isinstance(tagged, list) and tagged:
            tagged_ticker_count += 1
        for query_ticker in query_tickers_by_id[identity]:
            unique_by_query_ticker[query_ticker] += 1
            if is_post_cutoff:
                post_cutoff_by_ticker[query_ticker] += 1

    unique_count = len(unique)
    duplicate_query_rows = query_rows - unique_count
    manifest = root / "manifest.json"
    return {
        "provider": "Massive",
        "dataset": "ordinary_news",
        "root": str(root),
        "manifest_sha256": sha256_file(manifest) if manifest.exists() else None,
        "cached_page_count": len(paths),
        "query_result_rows": query_rows,
        "unique_documents": unique_count,
        "duplicate_query_rows": duplicate_query_rows,
        "duplicate_query_row_rate": safe_ratio(duplicate_query_rows, query_rows),
        "earliest_published_utc": format_timestamp(min(timestamps) if timestamps else None),
        "latest_published_utc": format_timestamp(max(timestamps) if timestamps else None),
        "post_cutoff": {
            "cutoff_relation": "strictly_after",
            "cutoff_utc": format_timestamp(cutoff),
            "unique_documents": post_cutoff,
            "by_query_ticker": dict(sorted(post_cutoff_by_ticker.items())),
        },
        "field_coverage": {
            "description": {
                "count": description_count,
                "rate": safe_ratio(description_count, unique_count),
            },
            "article_url": {
                "count": url_count,
                "rate": safe_ratio(url_count, unique_count),
            },
            "keywords_nonempty": {
                "count": keyword_count,
                "rate": safe_ratio(keyword_count, unique_count),
            },
            "insights_nonempty": {
                "count": insight_count,
                "rate": safe_ratio(insight_count, unique_count),
            },
            "publisher_name": {
                "count": publisher_count,
                "rate": safe_ratio(publisher_count, unique_count),
            },
            "tagged_tickers_nonempty": {
                "count": tagged_ticker_count,
                "rate": safe_ratio(tagged_ticker_count, unique_count),
            },
        },
        "query_rows_by_ticker": dict(sorted(query_counts.items())),
        "unique_documents_by_query_ticker": dict(sorted(unique_by_query_ticker.items())),
        "top_publishers": top_counts(publishers),
    }


def _alpha_identity(record: Mapping[str, Any]) -> str:
    url = record.get("url")
    if isinstance(url, str) and url.strip():
        return f"url:{url.strip()}"
    stable = json.dumps(
        {
            "title": record.get("title"),
            "time_published": record.get("time_published"),
            "source": record.get("source"),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return "fallback:" + hashlib.sha256(stable.encode("utf-8")).hexdigest()


def summarize_alpha(root: Path, cutoff: datetime) -> dict[str, Any]:
    paths = sorted((root / "responses").glob("*/*.json.gz"))
    query_rows = 0
    accepted_query_rows = 0
    unique: dict[str, Mapping[str, Any]] = {}
    query_tickers_by_id: dict[str, set[str]] = defaultdict(set)
    query_counts: Counter[str] = Counter()
    ceiling_responses = 0
    for path in paths:
        payload = load_json_gzip(path)
        records = payload.get("feed")
        if not isinstance(records, list):
            raise ValueError(f"{path}: expected a feed array")
        at_ceiling = alpha_response_at_ceiling(payload)
        ceiling_responses += int(at_ceiling)
        query_ticker = path.parent.name.upper()
        for record in records:
            if not isinstance(record, Mapping):
                continue
            query_rows += 1
            query_counts[query_ticker] += 1
            if at_ceiling:
                continue
            accepted_query_rows += 1
            identity = _alpha_identity(record)
            query_tickers_by_id[identity].add(query_ticker)
            unique.setdefault(identity, record)

    timestamps: list[datetime] = []
    summary_count = 0
    url_count = 0
    source_count = 0
    post_cutoff = 0
    sources: Counter[str] = Counter()
    post_cutoff_by_ticker: Counter[str] = Counter()
    unique_by_query_ticker: Counter[str] = Counter()
    for identity, record in unique.items():
        timestamp = parse_timestamp(record.get("time_published"))
        if timestamp is not None:
            timestamps.append(timestamp)
        is_post_cutoff = timestamp is not None and timestamp > cutoff
        if is_post_cutoff:
            post_cutoff += 1
        if isinstance(record.get("summary"), str) and record["summary"].strip():
            summary_count += 1
        if isinstance(record.get("url"), str) and record["url"].strip():
            url_count += 1
        source = record.get("source")
        if isinstance(source, str) and source.strip():
            source_count += 1
            sources[source.strip()] += 1
        for query_ticker in query_tickers_by_id[identity]:
            unique_by_query_ticker[query_ticker] += 1
            if is_post_cutoff:
                post_cutoff_by_ticker[query_ticker] += 1

    unique_count = len(unique)
    duplicate_query_rows = accepted_query_rows - unique_count
    manifest = root / "manifest.json"
    return {
        "provider": "Alpha Vantage",
        "dataset": "NEWS_SENTIMENT",
        "root": str(root),
        "manifest_sha256": sha256_file(manifest) if manifest.exists() else None,
        "cached_response_count": len(paths),
        "ceiling_response_count": ceiling_responses,
        "query_result_rows": query_rows,
        "accepted_query_result_rows": accepted_query_rows,
        "unique_documents": unique_count,
        "duplicate_accepted_query_rows": duplicate_query_rows,
        "duplicate_accepted_query_row_rate": safe_ratio(
            duplicate_query_rows, accepted_query_rows
        ),
        "earliest_published_utc": format_timestamp(min(timestamps) if timestamps else None),
        "latest_published_utc": format_timestamp(max(timestamps) if timestamps else None),
        "post_cutoff": {
            "cutoff_relation": "strictly_after",
            "cutoff_utc": format_timestamp(cutoff),
            "unique_documents": post_cutoff,
            "by_query_ticker": dict(sorted(post_cutoff_by_ticker.items())),
        },
        "field_coverage": {
            "summary": {
                "count": summary_count,
                "rate": safe_ratio(summary_count, unique_count),
            },
            "url": {
                "count": url_count,
                "rate": safe_ratio(url_count, unique_count),
            },
            "source": {
                "count": source_count,
                "rate": safe_ratio(source_count, unique_count),
            },
        },
        "query_rows_by_ticker": dict(sorted(query_counts.items())),
        "unique_documents_by_query_ticker": dict(sorted(unique_by_query_ticker.items())),
        "top_sources": top_counts(sources),
    }


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("massive", "alpha"), required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument(
        "--cutoff",
        default="2026-03-01T00:00:00Z",
        help="UTC timestamp used for post-knowledge-cutoff eligibility counts.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cutoff = parse_timestamp(args.cutoff)
    if cutoff is None:
        raise ValueError(f"Invalid cutoff timestamp: {args.cutoff!r}")
    if args.provider == "massive":
        report = summarize_massive(args.input_root, cutoff)
    else:
        report = summarize_alpha(args.input_root, cutoff)
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
