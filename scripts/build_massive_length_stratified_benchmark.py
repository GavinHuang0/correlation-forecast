"""Build a private length-stratified Massive-news robustness benchmark.

The benchmark uses the currently cached Massive ordinary-news snapshot. It
selects one target-conditioned record per provider article, balances every
length stratum across the five configured sectors, and adds a headline-only
counterfactual for every article that has a description.

This script does not call Massive or an LLM. Provider text is written below
``data/`` by default and must remain private.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOTS = (
    ROOT / "data" / "raw" / "massive" / "ordinary_news" / "free_v1",
)
DEFAULT_TARGETS = ROOT / "config" / "news_target_universe_30.json"
DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "data"
    / "benchmarks"
    / "massive_length_stratified_current"
    / "v1"
)
DEFAULT_CUTOFF = "2022-11-01T00:00:00Z"
DEFAULT_PER_STRATUM = 40
SELECTION_SEED = "massive-length-stratified-current-v1"
MANIFEST_VERSION = "massive-length-stratified-current-v1"
DESCRIPTION_EXCERPT_MAX_CHARS = 512
DESCRIPTION_EXCERPT_BOUNDARY_WINDOW = 64
STRATA = (
    "headline_only",
    "description_001_149",
    "description_150_299",
    "description_300_599",
    "description_600_plus",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        action="append",
        type=Path,
        help="Cached Massive ordinary-news root; repeat if needed.",
    )
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cutoff", default=DEFAULT_CUTOFF)
    parser.add_argument(
        "--per-stratum", type=int, default=DEFAULT_PER_STRATUM
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(*parts: str) -> str:
    return sha256_bytes("\x1f".join(parts).encode("utf-8"))


def clean_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def length_stratum(description: str) -> str:
    count = len(description)
    if count == 0:
        return "headline_only"
    if count < 150:
        return "description_001_149"
    if count < 300:
        return "description_150_299"
    if count < 600:
        return "description_300_599"
    return "description_600_plus"


def bounded_description(description: str) -> tuple[str, dict[str, Any]]:
    """Apply the repository's explicit deployment-time lede bound."""

    source_count = len(description)
    if source_count <= DESCRIPTION_EXCERPT_MAX_CHARS:
        retained = description
    else:
        candidate = description[:DESCRIPTION_EXCERPT_MAX_CHARS]
        floor = (
            DESCRIPTION_EXCERPT_MAX_CHARS
            - DESCRIPTION_EXCERPT_BOUNDARY_WINDOW
        )
        boundary = max(
            candidate.rfind(character) for character in (" ", "\t", "\n")
        )
        retained = candidate[:boundary] if boundary >= floor else candidate
        retained = retained.rstrip() or candidate
    return retained, {
        "source_description_char_count": source_count,
        "retained_description_char_count": len(retained),
        "omitted_description_char_count": source_count - len(retained),
        "description_was_bounded": retained != description,
    }


def discover_page_files(raw_roots: Sequence[Path]) -> list[Path]:
    files: set[Path] = set()
    for root in raw_roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
        files.update(path.resolve() for path in root.glob("pages/*/*.json.gz"))
    if not files:
        raise FileNotFoundError("No Massive page files were found")
    return sorted(files, key=lambda path: str(path).casefold())


def load_targets(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_targets = payload.get("targets")
    if not isinstance(raw_targets, Mapping) or not raw_targets:
        raise ValueError(f"{path}: targets must be a nonempty object")
    targets: dict[str, dict[str, Any]] = {}
    for raw_ticker, raw in raw_targets.items():
        ticker = str(raw_ticker).strip().upper()
        if not isinstance(raw, Mapping):
            raise ValueError(f"{path}: target {ticker} must be an object")
        record = {
            "ticker": ticker,
            "company": str(raw["company"]).strip(),
            "sector": str(raw["sector"]).strip(),
            "sector_benchmark": str(raw["benchmark"]).strip().upper(),
            "known_sector_peers": [
                str(value).strip().upper() for value in raw["peers"]
            ],
        }
        if not all(
            isinstance(value, str) and value
            for key, value in record.items()
            if key != "known_sector_peers"
        ):
            raise ValueError(f"{path}: target {ticker} has an empty field")
        targets[ticker] = record
    return targets


def read_massive_records(page_files: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in page_files:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        rows = payload.get("results")
        if not isinstance(rows, list):
            raise ValueError(f"{path}: results must be an array")
        for index, row in enumerate(rows, start=1):
            if not isinstance(row, Mapping):
                raise ValueError(f"{path}: result {index} must be an object")
            yield dict(row)


def merge_articles(
    rows: Iterable[dict[str, Any]], cutoff: str
) -> dict[str, dict[str, Any]]:
    articles: dict[str, dict[str, Any]] = {}
    for row in rows:
        published = clean_text(row.get("published_utc"))
        if published < cutoff:
            continue
        article_id = clean_text(row.get("id"))
        headline = clean_text(row.get("title"))
        if not article_id or not headline:
            continue
        description = clean_text(row.get("description"))
        tickers = sorted(
            {
                str(value).strip().upper()
                for value in row.get("tickers", [])
                if isinstance(value, str) and value.strip()
            }
        )
        publisher = row.get("publisher")
        source = (
            clean_text(publisher.get("name"))
            if isinstance(publisher, Mapping)
            else ""
        )
        candidate = {
            "provider_article_id": article_id,
            "published_utc": published,
            "source": source or "unknown",
            "headline": headline,
            "description": description,
            "vendor_tickers": tickers,
            "article_url": clean_text(row.get("article_url")),
        }
        existing = articles.get(article_id)
        if existing is None:
            articles[article_id] = candidate
            continue
        for field in ("published_utc", "headline", "description"):
            if existing[field] != candidate[field]:
                raise ValueError(
                    f"Provider article {article_id} has conflicting {field}"
                )
        existing["vendor_tickers"] = sorted(
            set(existing["vendor_tickers"]) | set(tickers)
        )
        if existing["source"] == "unknown" and candidate["source"] != "unknown":
            existing["source"] = candidate["source"]
        if not existing["article_url"] and candidate["article_url"]:
            existing["article_url"] = candidate["article_url"]
    return articles


def assign_target(
    article: Mapping[str, Any], targets: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any] | None:
    eligible = [
        ticker for ticker in article["vendor_tickers"] if ticker in targets
    ]
    if not eligible:
        return None
    ticker = min(
        eligible,
        key=lambda value: (
            stable_hash(
                SELECTION_SEED,
                str(article["provider_article_id"]),
                value,
            ),
            value,
        ),
    )
    return dict(targets[ticker])


def select_diverse(
    candidates: Sequence[dict[str, Any]], count: int, stratum: str, sector: str
) -> list[dict[str, Any]]:
    """Select deterministically while giving every available stock one turn."""

    by_ticker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in candidates:
        by_ticker[record["target"]["ticker"]].append(record)
    for ticker, rows in by_ticker.items():
        rows.sort(
            key=lambda row: (
                stable_hash(
                    SELECTION_SEED,
                    stratum,
                    sector,
                    ticker,
                    row["provider_article_id"],
                ),
                row["provider_article_id"],
            )
        )
    selected: list[dict[str, Any]] = []
    depth = 0
    tickers = sorted(by_ticker)
    while len(selected) < count:
        added = False
        for ticker in tickers:
            if depth < len(by_ticker[ticker]):
                selected.append(by_ticker[ticker][depth])
                added = True
                if len(selected) == count:
                    break
        if not added:
            break
        depth += 1
    if len(selected) != count:
        raise ValueError(
            f"Need {count} records for {stratum}/{sector}, found {len(selected)}"
        )
    return selected


def build_sample(
    articles: Mapping[str, dict[str, Any]],
    targets: Mapping[str, Mapping[str, Any]],
    per_stratum: int,
) -> list[dict[str, Any]]:
    sectors = sorted({str(target["sector"]) for target in targets.values()})
    if per_stratum <= 0 or per_stratum % len(sectors):
        raise ValueError(
            f"per-stratum must be positive and divisible by {len(sectors)}"
        )
    per_sector = per_stratum // len(sectors)
    pools: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for article in articles.values():
        target = assign_target(article, targets)
        if target is None:
            continue
        candidate = dict(article)
        candidate["target"] = target
        candidate["description_char_count"] = len(article["description"])
        candidate["length_stratum"] = length_stratum(article["description"])
        pools[(candidate["length_stratum"], target["sector"])].append(candidate)

    selected: list[dict[str, Any]] = []
    for stratum in STRATA:
        for sector in sectors:
            selected.extend(
                select_diverse(
                    pools[(stratum, sector)], per_sector, stratum, sector
                )
            )
    if len({record["provider_article_id"] for record in selected}) != len(
        selected
    ):
        raise ValueError("Selected provider articles are not unique")
    selected.sort(
        key=lambda row: (
            STRATA.index(row["length_stratum"]),
            row["target"]["sector"],
            row["target"]["ticker"],
            row["provider_article_id"],
        )
    )
    return selected


def parent_record(record: Mapping[str, Any], row_number: int) -> dict[str, Any]:
    retained, bound = bounded_description(record["description"])
    return {
        "row_number": row_number,
        "provider_article_id": record["provider_article_id"],
        "time_published_utc": record["published_utc"],
        "source": record["source"],
        "headline": record["headline"],
        "description": retained,
        "description_char_count": record["description_char_count"],
        **bound,
        "length_stratum": record["length_stratum"],
        "vendor_tickers": list(record["vendor_tickers"]),
        "target": dict(record["target"]),
        "article_url": record["article_url"],
    }


def model_input(
    parent: Mapping[str, Any], row_number: int, variant: str
) -> dict[str, Any]:
    description = (
        parent["description"] if variant == "native_description" else ""
    )
    article_id = (
        f"massive-length-v1:{parent['provider_article_id']}:{variant}"
    )
    return {
        "row_number": row_number,
        "article_id": article_id,
        "parent_article_id": parent["provider_article_id"],
        "time_published_utc": parent["time_published_utc"],
        "source": parent["source"],
        "headline": parent["headline"],
        "article_text": description,
        "vendor_tickers": list(parent["vendor_tickers"]),
        "target": dict(parent["target"]),
        "text_variant": variant,
        "native_length_stratum": parent["length_stratum"],
        "native_description_char_count": parent["description_char_count"],
    }


def write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(canonical_json(record) + "\n")


def source_manifest_records(raw_roots: Sequence[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for root in raw_roots:
        manifest = root / "manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        records.append(
            {
                "path": str(manifest),
                "sha256": sha256_file(manifest),
                "status": payload.get("status"),
                "scope": payload.get("scope"),
                "totals": payload.get("totals"),
            }
        )
    return records


def main() -> int:
    args = parse_args()
    raw_roots = tuple(args.raw_root or DEFAULT_RAW_ROOTS)
    targets = load_targets(args.targets)
    pages = discover_page_files(raw_roots)
    articles = merge_articles(read_massive_records(pages), args.cutoff)
    selected = build_sample(articles, targets, args.per_stratum)
    parents = [
        parent_record(record, row_number)
        for row_number, record in enumerate(selected, start=1)
    ]
    inputs: list[dict[str, Any]] = []
    for parent in parents:
        inputs.append(model_input(parent, len(inputs) + 1, "native_description"))
        if parent["description_char_count"] > 0:
            inputs.append(
                model_input(
                    parent, len(inputs) + 1, "headline_only_ablation"
                )
            )

    output_root = args.output_root
    parent_path = output_root / "parent_sample.jsonl"
    input_path = output_root / "model_inputs.jsonl"
    manifest_path = output_root / "manifest.json"
    for path in (parent_path, input_path, manifest_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite")
    output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(parent_path, parents)
    write_jsonl(input_path, inputs)

    stratum_counts = Counter(row["length_stratum"] for row in parents)
    sector_counts = Counter(row["target"]["sector"] for row in parents)
    source_by_stratum: dict[str, Counter[str]] = {
        stratum: Counter() for stratum in STRATA
    }
    dates_by_stratum: dict[str, list[str]] = {
        stratum: [] for stratum in STRATA
    }
    for row in parents:
        source_by_stratum[row["length_stratum"]][row["source"]] += 1
        dates_by_stratum[row["length_stratum"]].append(
            row["time_published_utc"]
        )

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "status": "complete_private_benchmark",
        "purpose": (
            "Length-stratified operational and paired-ablation validation of "
            "the frozen FLAN-T5-XL active research extractor."
        ),
        "claim_limit": (
            "No independent labels are included. This benchmark measures "
            "runtime validity, abstention, confidence, distributional "
            "behavior, and paired text sensitivity—not objective accuracy."
        ),
        "source_snapshot": source_manifest_records(raw_roots),
        "selection": {
            "cutoff_inclusive_utc": args.cutoff,
            "seed": SELECTION_SEED,
            "strata_in_order": list(STRATA),
            "per_stratum": args.per_stratum,
            "sector_balance_per_stratum": (
                args.per_stratum
                // len({target["sector"] for target in targets.values()})
            ),
            "one_provider_tagged_target_per_unique_article": True,
            "counterfactual": (
                "Every nonempty-description article is rerun with article_text "
                "empty and the headline unchanged."
            ),
            "model_text_contract": {
                "length_stratum_uses": "original provider description length",
                "extractor_receives": (
                    "headline plus an explicit leading description excerpt"
                ),
                "description_excerpt_max_unicode_codepoints": (
                    DESCRIPTION_EXCERPT_MAX_CHARS
                ),
                "preferred_whitespace_boundary_window_codepoints": (
                    DESCRIPTION_EXCERPT_BOUNDARY_WINDOW
                ),
                "runner_side_truncation_allowed": False,
            },
        },
        "counts": {
            "parent_articles": len(parents),
            "model_inputs": len(inputs),
            "native_views": len(parents),
            "headline_only_ablation_views": sum(
                row["description_char_count"] > 0 for row in parents
            ),
            "by_stratum": dict(stratum_counts),
            "by_sector": dict(sector_counts),
        },
        "coverage_by_stratum": {
            stratum: {
                "count": stratum_counts[stratum],
                "published_min_utc": min(dates_by_stratum[stratum]),
                "published_max_utc": max(dates_by_stratum[stratum]),
                "source_counts": dict(source_by_stratum[stratum]),
            }
            for stratum in STRATA
        },
        "files": {
            "parent_sample": {
                "path": str(parent_path),
                "sha256": sha256_file(parent_path),
                "rows": len(parents),
            },
            "model_inputs": {
                "path": str(input_path),
                "sha256": sha256_file(input_path),
                "rows": len(inputs),
            },
            "target_metadata": {
                "path": str(args.targets),
                "sha256": sha256_file(args.targets),
            },
            "builder": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
