"""Build a balanced, post-cutoff full-text retrieval queue from Massive pages.

No network access occurs here.  The queue contains provider metadata and URLs
under the ignored ``data/`` tree so the retrieval worker can run independently
and resume safely.  Each unique article is emitted once, with round-robin
primary-ticker assignment to avoid exhausting the full-text budget on the
highest-news-volume companies first.

The input may contain separate stock-query and sector/market-query roots.
Eligibility is intentionally broader than direct vendor tags: an article about
a same-sector peer is a valid peer-specific document, a sector-benchmark query
is valid for every configured stock in that sector, and a market-control query
is valid for every configured target.  Every expansion is retained as explicit
per-target provenance in the queue.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_INPUT = Path("data/raw/massive/ordinary_news/free_v1")
DEFAULT_UNIVERSE = Path("config/price_universe.json")
DEFAULT_OUTPUT = Path(
    "data/external/news_provider_comparison/v1_0/fulltext_candidates.jsonl"
)
DEFAULT_CUTOFF = "2026-03-01T00:00:00Z"
SEED = "fulltext-candidate-queue-v1"


def parse_timestamp(value: str) -> datetime:
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise ValueError("timestamp must contain a timezone")
    return parsed.astimezone(timezone.utc)


def load_tickers(path: Path) -> list[str]:
    return list(load_universe(path)["tickers"])


def load_universe(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sectors = payload.get("sectors") if isinstance(payload, Mapping) else None
    if not isinstance(sectors, list):
        raise ValueError(f"{path}: expected a sectors array")
    result: list[str] = []
    benchmark_by_ticker: dict[str, str] = {}
    stocks_by_benchmark: dict[str, list[str]] = {}
    for sector in sectors:
        if not isinstance(sector, Mapping) or not isinstance(sector.get("stocks"), list):
            raise ValueError(f"{path}: invalid sector record")
        benchmark = str(sector.get("benchmark", "")).strip().upper()
        if not benchmark or benchmark in stocks_by_benchmark:
            raise ValueError(f"{path}: sector benchmarks must be nonempty and unique")
        stocks = [str(value).strip().upper() for value in sector["stocks"]]
        if not stocks or any(not ticker for ticker in stocks):
            raise ValueError(f"{path}: sector stocks must be nonempty")
        stocks_by_benchmark[benchmark] = stocks
        result.extend(stocks)
        benchmark_by_ticker.update({ticker: benchmark for ticker in stocks})
    if len(result) != len(set(result)) or not result:
        raise ValueError(f"{path}: tickers must be nonempty and unique")
    raw_controls = payload.get("controls", []) if isinstance(payload, Mapping) else []
    if not isinstance(raw_controls, list):
        raise ValueError(f"{path}: controls must be an array")
    controls = [str(value).strip().upper() for value in raw_controls]
    if any(not ticker for ticker in controls) or len(controls) != len(set(controls)):
        raise ValueError(f"{path}: controls must be nonempty and unique")
    return {
        "tickers": result,
        "benchmark_by_ticker": benchmark_by_ticker,
        "stocks_by_benchmark": stocks_by_benchmark,
        "controls": controls,
    }


def stable_hash(*values: str) -> str:
    return hashlib.sha256("\x1f".join(values).encode("utf-8")).hexdigest()


def identity(record: Mapping[str, Any]) -> str:
    if record.get("id") not in (None, ""):
        return "id:" + str(record["id"])
    url = record.get("article_url")
    if isinstance(url, str) and url.strip():
        return "url:" + url.strip()
    return "fallback:" + stable_hash(
        str(record.get("title", "")),
        str(record.get("published_utc", "")),
        json.dumps(record.get("publisher"), sort_keys=True, ensure_ascii=False),
    )


def read_candidates(
    input_roots: Path | Sequence[Path],
    *,
    tickers: Sequence[str],
    cutoff: datetime,
    benchmark_by_ticker: Mapping[str, str] | None = None,
    stocks_by_benchmark: Mapping[str, Sequence[str]] | None = None,
    controls: Sequence[str] = (),
) -> list[dict[str, Any]]:
    known = set(tickers)
    benchmark_for = {
        str(ticker).strip().upper(): str(benchmark).strip().upper()
        for ticker, benchmark in (benchmark_by_ticker or {}).items()
    }
    sector_stocks = {
        str(benchmark).strip().upper(): tuple(
            str(ticker).strip().upper() for ticker in stocks
        )
        for benchmark, stocks in (stocks_by_benchmark or {}).items()
    }
    control_tickers = {str(value).strip().upper() for value in controls}
    supplied_roots = (
        [input_roots] if isinstance(input_roots, Path) else list(input_roots)
    )
    roots: list[Path] = []
    seen_roots: set[str] = set()
    for supplied_root in supplied_roots:
        root = Path(supplied_root)
        root_key = str(root.resolve()).casefold()
        if root_key not in seen_roots:
            roots.append(root)
            seen_roots.add(root_key)
    if not roots:
        raise ValueError("at least one input root is required")

    unique: dict[str, dict[str, Any]] = {}
    query_tickers: dict[str, set[str]] = defaultdict(set)
    tagged_tickers: dict[str, set[str]] = defaultdict(set)
    query_provenance: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for input_root in roots:
        for path in sorted((input_root / "pages").glob("*/*.json.gz")):
            query_ticker = path.parent.name.upper()
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
            records = payload.get("results") if isinstance(payload, Mapping) else None
            if not isinstance(records, list):
                raise ValueError(f"{path}: expected a results array")
            for raw in records:
                if not isinstance(raw, Mapping):
                    continue
                published = raw.get("published_utc")
                if not isinstance(published, str):
                    continue
                try:
                    published_time = parse_timestamp(published)
                except ValueError:
                    continue
                if published_time <= cutoff:
                    continue
                url = raw.get("article_url")
                title = raw.get("title")
                description = raw.get("description")
                if not all(
                    isinstance(value, str) and value.strip()
                    for value in (url, title, description)
                ):
                    continue
                key = identity(raw)
                record_tickers = raw.get("tickers")
                tagged = (
                    {str(value).strip().upper() for value in record_tickers}
                    if isinstance(record_tickers, list)
                    else set()
                )
                query_tickers[key].add(query_ticker)
                tagged_tickers[key].update(tagged)
                query_provenance[key].add((query_ticker, str(input_root)))
                candidate = {
                    "id": str(raw.get("id") or stable_hash(key)[:24]),
                    "article_url": url.strip(),
                    "title": title.strip(),
                    "description": description.strip(),
                    "published_utc": published_time.isoformat(
                        timespec="seconds"
                    ).replace("+00:00", "Z"),
                    "tickers": sorted(tagged),
                    "keywords": raw.get("keywords")
                    if isinstance(raw.get("keywords"), list)
                    else [],
                    "publisher": raw.get("publisher")
                    if isinstance(raw.get("publisher"), Mapping)
                    else {},
                    "_identity": key,
                }
                existing = unique.get(key)
                if existing is None or (
                    len(candidate["description"]),
                    json.dumps(candidate, sort_keys=True, ensure_ascii=False),
                ) > (
                    len(existing["description"]),
                    json.dumps(existing, sort_keys=True, ensure_ascii=False),
                ):
                    unique[key] = candidate

    result: list[dict[str, Any]] = []
    for key, row in unique.items():
        row["tickers"] = sorted(tagged_tickers[key])
        provenance: dict[
            str, set[tuple[str, str, str, str]]
        ] = defaultdict(set)

        def add_eligibility(
            target: str,
            *,
            reason: str,
            source_ticker: str,
            source_basis: str,
            sector_benchmark: str = "",
        ) -> None:
            if target in known:
                provenance[target].add(
                    (reason, source_ticker, source_basis, sector_benchmark)
                )

        tagged_targets = tagged_tickers[key] & known
        queried_targets = query_tickers[key] & known
        for source_ticker in sorted(tagged_targets):
            benchmark = benchmark_for.get(source_ticker, "")
            add_eligibility(
                source_ticker,
                reason="direct_vendor_tag",
                source_ticker=source_ticker,
                source_basis="vendor_tag",
                sector_benchmark=benchmark,
            )
            for target in sector_stocks.get(benchmark, ()):
                if target != source_ticker:
                    add_eligibility(
                        target,
                        reason="same_sector_peer",
                        source_ticker=source_ticker,
                        source_basis="vendor_tag",
                        sector_benchmark=benchmark,
                    )
        for source_ticker in sorted(queried_targets):
            benchmark = benchmark_for.get(source_ticker, "")
            add_eligibility(
                source_ticker,
                reason="direct_stock_query",
                source_ticker=source_ticker,
                source_basis="query",
                sector_benchmark=benchmark,
            )
            for target in sector_stocks.get(benchmark, ()):
                if target != source_ticker:
                    add_eligibility(
                        target,
                        reason="same_sector_peer",
                        source_ticker=source_ticker,
                        source_basis="query",
                        sector_benchmark=benchmark,
                    )
        for benchmark in sorted(query_tickers[key] & set(sector_stocks)):
            for target in sector_stocks[benchmark]:
                add_eligibility(
                    target,
                    reason="sector_benchmark_query",
                    source_ticker=benchmark,
                    source_basis="query",
                    sector_benchmark=benchmark,
                )
        for control in sorted(query_tickers[key] & control_tickers):
            for target in tickers:
                add_eligibility(
                    target,
                    reason="market_control_query",
                    source_ticker=control,
                    source_basis="query",
                )

        if not provenance:
            continue
        row["eligible_target_tickers"] = sorted(provenance)
        row["eligibility_provenance"] = {
            target: [
                {
                    **{
                        "reason": reason,
                        "source_ticker": source_ticker,
                        "source_basis": source_basis,
                    },
                    **(
                        {"sector_benchmark": sector_benchmark}
                        if sector_benchmark
                        else {}
                    ),
                }
                for reason, source_ticker, source_basis, sector_benchmark in sorted(
                    entries
                )
            ]
            for target, entries in sorted(provenance.items())
        }
        row["query_tickers"] = sorted(query_tickers[key])
        row["query_provenance"] = [
            {"query_ticker": query_ticker, "input_root": input_root}
            for query_ticker, input_root in sorted(query_provenance[key])
        ]
        result.append(row)
    return result


def balanced_order(
    candidates: Sequence[Mapping[str, Any]], tickers: Sequence[str]
) -> list[dict[str, Any]]:
    by_ticker: dict[str, list[Mapping[str, Any]]] = {}
    for ticker in tickers:
        rows = [
            row
            for row in candidates
            if ticker in set(row.get("eligible_target_tickers", []))
        ]
        rows.sort(
            key=lambda row: stable_hash(
                SEED, ticker, str(row.get("_identity", ""))
            )
        )
        by_ticker[ticker] = rows

    emitted: set[str] = set()
    positions = {ticker: 0 for ticker in tickers}
    ordered: list[dict[str, Any]] = []
    while True:
        progress = False
        for ticker in tickers:
            rows = by_ticker[ticker]
            while positions[ticker] < len(rows):
                row = rows[positions[ticker]]
                positions[ticker] += 1
                key = str(row["_identity"])
                if key in emitted:
                    continue
                output = dict(row)
                output.pop("_identity", None)
                output["queue_primary_ticker"] = ticker
                output["queue_position"] = len(ordered) + 1
                ordered.append(output)
                emitted.add(key)
                progress = True
                break
        if not progress:
            break
    return ordered


def write_queue(
    output: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    input_roots: Sequence[Path],
    cutoff: datetime,
    tickers: Sequence[str],
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            )
    counts = Counter(str(row["queue_primary_ticker"]) for row in rows)
    eligible_counts = Counter(
        ticker
        for row in rows
        for ticker in row.get("eligible_target_tickers", [])
    )
    reason_counts = Counter(
        str(item["reason"])
        for row in rows
        for entries in row.get("eligibility_provenance", {}).values()
        for item in entries
    )
    manifest = {
        "schema_version": 2,
        "input_roots": [str(path) for path in input_roots],
        "cutoff_relation": "strictly_after",
        "cutoff_utc": cutoff.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "configured_tickers": list(tickers),
        "candidate_count": len(rows),
        "queue_primary_ticker_counts": {
            ticker: counts[ticker] for ticker in tickers
        },
        "eligible_article_counts": {
            ticker: eligible_counts[ticker] for ticker in tickers
        },
        "eligibility_reason_counts": dict(sorted(reason_counts.items())),
        "ordering": "deterministic round-robin over stable-hash ticker queues",
        "output_file": str(output),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        action="append",
        type=Path,
        default=[],
        help=(
            "Massive collector root containing pages/<QUERY>/*.json.gz; "
            "repeat for separate stock and sector/market query collections."
        ),
    )
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cutoff", default=DEFAULT_CUTOFF)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cutoff = parse_timestamp(args.cutoff)
    universe = load_universe(args.universe)
    tickers = universe["tickers"]
    input_roots = args.input_root or [DEFAULT_INPUT]
    candidates = read_candidates(
        input_roots,
        tickers=tickers,
        cutoff=cutoff,
        benchmark_by_ticker=universe["benchmark_by_ticker"],
        stocks_by_benchmark=universe["stocks_by_benchmark"],
        controls=universe["controls"],
    )
    rows = balanced_order(candidates, tickers)
    manifest = write_queue(
        args.output,
        rows,
        input_roots=input_roots,
        cutoff=cutoff,
        tickers=tickers,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
