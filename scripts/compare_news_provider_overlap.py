"""Measure exact document overlap between cached Massive and Alpha news.

The report contains counts and hashes only, never licensed descriptions,
summaries, headlines, or URLs. Alpha responses at the 1,000-result ceiling are
excluded because the collector treats them as incomplete parent slices.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


TRACKING_NAMES = {"fbclid", "gclid", "mc_cid", "mc_eid"}


def parse_time(value: str) -> datetime:
    text = value.strip()
    if re.fullmatch(r"\d{8}T\d{6}", text):
        return datetime.strptime(text, "%Y%m%dT%H%M%S").replace(
            tzinfo=timezone.utc
        )
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def normalize_url(value: str) -> str:
    parsed = urlsplit(html.unescape(value).strip())
    if not parsed.netloc:
        return value.strip()
    query = [
        (key, item)
        for key, item in parse_qsl(parsed.query, keep_blank_values=True)
        if key.casefold() not in TRACKING_NAMES
        and not key.casefold().startswith("utm_")
    ]
    path = re.sub(r"/+", "/", parsed.path or "/")
    if path != "/":
        path = path.rstrip("/")
    return urlunsplit(
        (
            (parsed.scheme or "https").casefold(),
            parsed.netloc.casefold(),
            path,
            urlencode(sorted(query)),
            "",
        )
    )


def normalize_title(value: str) -> str:
    text = unicodedata.normalize("NFKC", html.unescape(value)).casefold()
    return " ".join(re.sub(r"[^\w]+", " ", text).split())


def load_gzip(path: Path) -> Mapping[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: expected an object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def input_root_fingerprint(root: Path, response_directory: str) -> dict[str, Any]:
    """Hash the cached corpus itself without exposing licensed record content."""

    files = sorted(
        (root / response_directory).glob("*/*.json.gz"),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    manifest = root / "manifest.json"
    entries = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    canonical = json.dumps(
        entries, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "cached_file_count": len(files),
        "cached_files_sha256": hashlib.sha256(canonical).hexdigest(),
        "manifest_sha256": sha256_file(manifest) if manifest.is_file() else None,
    }


def alpha_response_at_ceiling(payload: Mapping[str, Any]) -> bool:
    records = payload.get("feed")
    if not isinstance(records, list):
        return False
    try:
        reported_items = int(str(payload.get("items", len(records))))
    except ValueError:
        reported_items = len(records)
    return len(records) >= 1_000 or reported_items >= 1_000


def load_massive(root: Path, cutoff: datetime) -> dict[str, dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    query_tickers: dict[str, set[str]] = defaultdict(set)
    for path in sorted((root / "pages").glob("*/*.json.gz")):
        records = load_gzip(path).get("results")
        if not isinstance(records, list):
            raise ValueError(f"{path}: expected results")
        for raw in records:
            if not isinstance(raw, Mapping):
                continue
            published = raw.get("published_utc")
            if not isinstance(published, str) or parse_time(published) <= cutoff:
                continue
            key = str(raw.get("id") or raw.get("article_url") or "")
            if not key:
                continue
            query_tickers[key].add(path.parent.name.upper())
            if key not in unique:
                unique[key] = dict(raw)
    for key, record in unique.items():
        record["_query_tickers"] = sorted(query_tickers[key])
    return unique


def load_alpha(root: Path, cutoff: datetime) -> dict[str, dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "responses").glob("*/*.json.gz")):
        payload = load_gzip(path)
        records = payload.get("feed")
        if not isinstance(records, list):
            raise ValueError(f"{path}: expected feed")
        if alpha_response_at_ceiling(payload):
            continue
        for raw in records:
            if not isinstance(raw, Mapping):
                continue
            published = raw.get("time_published")
            if not isinstance(published, str) or parse_time(published) <= cutoff:
                continue
            url = raw.get("url")
            if not isinstance(url, str) or not url.strip():
                continue
            unique.setdefault(normalize_url(url), dict(raw))
    return unique


def length_summary(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "minimum": None, "median": None, "maximum": None}
    return {
        "count": len(values),
        "minimum": min(values),
        "median": median(values),
        "maximum": max(values),
    }


def compare(
    massive: Mapping[str, Mapping[str, Any]],
    alpha: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    massive_by_url: dict[str, Mapping[str, Any]] = {}
    massive_title_time: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    for record in massive.values():
        url = record.get("article_url")
        title = record.get("title")
        published = record.get("published_utc")
        if isinstance(url, str) and url.strip():
            massive_by_url.setdefault(normalize_url(url), record)
        if isinstance(title, str) and isinstance(published, str):
            key = (
                normalize_title(title),
                parse_time(published).isoformat(timespec="seconds"),
            )
            massive_title_time[key].append(record)

    exact_url: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    title_time_only: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    used_massive: set[int] = set()
    for alpha_url, alpha_record in alpha.items():
        massive_record = massive_by_url.get(alpha_url)
        if massive_record is not None:
            exact_url.append((massive_record, alpha_record))
            used_massive.add(id(massive_record))
            continue
        title = alpha_record.get("title")
        published = alpha_record.get("time_published")
        if not isinstance(title, str) or not isinstance(published, str):
            continue
        key = (
            normalize_title(title),
            parse_time(published).isoformat(timespec="seconds"),
        )
        candidates = [
            row for row in massive_title_time.get(key, []) if id(row) not in used_massive
        ]
        if len(candidates) == 1:
            title_time_only.append((candidates[0], alpha_record))
            used_massive.add(id(candidates[0]))

    pairs = [*exact_url, *title_time_only]
    by_ticker: Counter[str] = Counter()
    massive_lengths: list[int] = []
    alpha_lengths: list[int] = []
    for massive_record, alpha_record in pairs:
        tags = massive_record.get("tickers")
        if isinstance(tags, list):
            by_ticker.update(str(value) for value in tags)
        description = massive_record.get("description")
        summary = alpha_record.get("summary")
        if isinstance(description, str):
            massive_lengths.append(len(description))
        if isinstance(summary, str):
            alpha_lengths.append(len(summary))
    return {
        "massive_unique_post_cutoff": len(massive),
        "alpha_unique_post_cutoff_from_complete_slices": len(alpha),
        "matched_unique_documents": len(pairs),
        "match_methods": {
            "normalized_url": len(exact_url),
            "normalized_title_and_exact_second": len(title_time_only),
        },
        "matched_by_massive_ticker_tag": dict(sorted(by_ticker.items())),
        "matched_text_character_lengths": {
            "massive_description": length_summary(massive_lengths),
            "alpha_summary": length_summary(alpha_lengths),
        },
    }


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--massive-root", type=Path, required=True)
    parser.add_argument("--alpha-root", type=Path, required=True)
    parser.add_argument("--cutoff", default="2026-03-01T00:00:00Z")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cutoff = parse_time(args.cutoff)
    massive = load_massive(args.massive_root, cutoff)
    alpha = load_alpha(args.alpha_root, cutoff)
    report = compare(massive, alpha)
    report.update(
        {
            "cutoff_relation": "strictly_after",
            "cutoff_utc": cutoff.isoformat(timespec="seconds").replace(
                "+00:00", "Z"
            ),
            "input_fingerprints": {
                "massive": input_root_fingerprint(
                    args.massive_root, "pages"
                ),
                "alpha": input_root_fingerprint(
                    args.alpha_root, "responses"
                ),
            },
        }
    )
    write_report(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
