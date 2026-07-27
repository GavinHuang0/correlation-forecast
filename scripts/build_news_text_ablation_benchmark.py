"""Build a private, post-cutoff news-text ablation benchmark.

The builder is intentionally isolated from the active FLAN/Llama benchmark
directories.  It only reads cached provider exports and writes licensed text
below ``data/benchmarks/news_text_ablation_300/v1`` by default, a path covered
by the repository's ``data/`` ignore rule.  It does not call a provider or a
model.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UNIVERSE = REPOSITORY_ROOT / "config" / "price_universe.json"
DEFAULT_TARGET_METADATA = (
    REPOSITORY_ROOT / "config" / "news_target_universe_30.json"
)
DEFAULT_SCHEMA = REPOSITORY_ROOT / "config" / "news_feature_schema_coarse.json"
DEFAULT_OUTPUT_ROOT = (
    REPOSITORY_ROOT / "data" / "benchmarks" / "news_text_ablation_300" / "v1"
)
DEFAULT_CUTOFF = "2026-03-01T00:00:00Z"
DEFAULT_PER_STOCK = 10
DEFAULT_DEVELOPMENT_COUNT = 72
DEFAULT_MAX_CHUNK_CHARACTERS = 650
SELECTION_SEED = "news-text-ablation-300-v1-selection"
SPLIT_SEED = "news-text-ablation-300-v1-split"
MANIFEST_VERSION = "news-text-ablation-300-v1.1"
VARIANT_FILES = {
    "massive_description": "massive_description.jsonl",
    "alpha_summary": "alpha_summary.jsonl",
    "fulltext_evidence_chunks": "fulltext_evidence_chunks.jsonl",
}
ASSIGNMENT_BASIS_VALUES = (
    "direct_target_tag",
    "peer_only_tag",
    "expanded_sector_or_market_only",
)

ID_FIELDS = (
    "id",
    "article_id",
    "provider_article_id",
    "source_article_id",
    "massive_article_id",
    "massive_id",
    "benzinga_id",
    "news_id",
)
URL_FIELDS = (
    "article_url",
    "url",
    "canonical_url",
    "amp_url",
    "requested_url",
    "final_url",
)
TIMESTAMP_FIELDS = (
    "published_utc",
    "published_at",
    "published",
    "time_published_utc",
    "time_published",
    "publication_time",
)
TITLE_FIELDS = ("title", "headline")
FULLTEXT_FIELDS = (
    "full_text",
    "fulltext",
    "body",
    "article_text",
    "content",
    "text",
    "main_text",
    "extracted_text",
    "markdown",
    "paragraphs",
)
TRACKING_QUERY_PREFIXES = ("utm_",)
TRACKING_QUERY_NAMES = frozenset({"fbclid", "gclid", "mc_cid", "mc_eid"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a balanced private 300-document benchmark from cached "
            "Massive descriptions, optional Alpha summaries, and full text."
        )
    )
    parser.add_argument(
        "--massive",
        action="append",
        required=True,
        type=Path,
        help="Massive JSON/JSONL file or directory; repeat for multiple inputs.",
    )
    parser.add_argument(
        "--fulltext",
        action="append",
        required=True,
        type=Path,
        help="Full-text retrieval JSON/JSONL file or directory; repeat as needed.",
    )
    parser.add_argument(
        "--alpha",
        action="append",
        default=[],
        type=Path,
        help="Optional Alpha Vantage JSON/JSONL input; repeat as needed.",
    )
    parser.add_argument(
        "--require-alpha-match",
        action="store_true",
        help=(
            "Provider-comparison mode: require a nonempty matched Alpha summary "
            "before balanced assignment. Requires a separate --output-root."
        ),
    )
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cutoff", default=DEFAULT_CUTOFF)
    parser.add_argument("--per-stock", type=int, default=DEFAULT_PER_STOCK)
    parser.add_argument(
        "--development-count", type=int, default=DEFAULT_DEVELOPMENT_COUNT
    )
    parser.add_argument(
        "--max-chunk-characters",
        type=int,
        default=DEFAULT_MAX_CHUNK_CHARACTERS,
    )
    parser.add_argument(
        "--fixed-assignment-manifest",
        type=Path,
        help=(
            "Reuse article_id-to-target assignments from an earlier benchmark "
            "manifest while rebuilding text chunks. Every assigned article must "
            "remain eligible, and the assignment must still be exactly balanced."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only the four known benchmark output files if they exist.",
    )
    return parser.parse_args()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(*parts: str) -> str:
    return sha256_bytes("\x1f".join(parts).encode("utf-8"))


def format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def parse_timestamp(value: Any) -> datetime:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        if numeric > 10_000_000_000:
            numeric /= 1000
        return datetime.fromtimestamp(numeric, tz=timezone.utc)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Invalid publication timestamp {value!r}")
    text = value.strip()
    if re.fullmatch(r"\d{8}T\d{6}", text):
        return datetime.strptime(text, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    normalized = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"Invalid publication timestamp {value!r}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"Publication timestamp must include a timezone: {value!r}")
    return parsed.astimezone(timezone.utc)


def first_string(record: Mapping[str, Any], fields: Iterable[str]) -> str:
    for field in fields:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def publication_time(record: Mapping[str, Any]) -> datetime:
    for field in TIMESTAMP_FIELDS:
        value = record.get(field)
        if value not in (None, ""):
            return parse_timestamp(value)
    raise ValueError("Record has no recognized publication timestamp")


def normalize_title(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", html.unescape(value)).casefold()
    normalized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


def normalize_url(value: str) -> str:
    text = html.unescape(value).strip()
    if not text:
        return ""
    try:
        parsed = urlsplit(text)
    except ValueError:
        return text
    if not parsed.netloc:
        return text.rstrip("/")
    scheme = parsed.scheme.lower() or "https"
    netloc = parsed.netloc.casefold()
    if scheme == "https" and netloc.endswith(":443"):
        netloc = netloc[:-4]
    if scheme == "http" and netloc.endswith(":80"):
        netloc = netloc[:-3]
    path = re.sub(r"/+", "/", parsed.path or "/")
    if path != "/":
        path = path.rstrip("/")
    query_items = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.casefold() not in TRACKING_QUERY_NAMES
        and not key.casefold().startswith(TRACKING_QUERY_PREFIXES)
    ]
    query = urlencode(sorted(query_items))
    return urlunsplit((scheme, netloc, path, query, ""))


def record_ids(record: Mapping[str, Any]) -> tuple[str, ...]:
    values = {
        str(record[field]).strip()
        for field in ID_FIELDS
        if record.get(field) not in (None, "")
    }
    return tuple(sorted(value for value in values if value))


def record_urls(record: Mapping[str, Any]) -> tuple[str, ...]:
    values = {
        normalized
        for field in URL_FIELDS
        if isinstance(record.get(field), str)
        and (normalized := normalize_url(str(record[field])))
    }
    return tuple(sorted(values))


def title_time_key(record: Mapping[str, Any]) -> str:
    title = normalize_title(first_string(record, TITLE_FIELDS))
    if not title:
        return ""
    try:
        timestamp = format_utc(publication_time(record))
    except ValueError:
        return ""
    return f"{title}\x1f{timestamp}"


def _looks_like_record(payload: Mapping[str, Any]) -> bool:
    recognized = set(ID_FIELDS + URL_FIELDS + TIMESTAMP_FIELDS + TITLE_FIELDS)
    return bool(recognized & set(payload))


def records_from_payload(payload: Any, *, path: Path) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, Mapping):
        rows = None
        for key in ("results", "feed", "articles", "data", "items"):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                rows = candidate
                break
            if isinstance(candidate, Mapping):
                rows = list(candidate.values())
                break
        if rows is None and _looks_like_record(payload):
            rows = [payload]
        if rows is None:
            mapped_rows = []
            for key, value in payload.items():
                if isinstance(value, Mapping) and _looks_like_record(value):
                    row = dict(value)
                    row.setdefault("id", str(key))
                    mapped_rows.append(row)
            rows = mapped_rows
    else:
        rows = None
    if not rows:
        raise ValueError(f"{path} contains no recognized article records")
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise ValueError(f"{path} record {index} is not an object")
        result.append(dict(row))
    return result


def discover_input_files(paths: Sequence[Path]) -> list[Path]:
    discovered: set[Path] = set()
    for supplied in paths:
        if supplied.is_file():
            discovered.add(supplied.resolve())
            continue
        if supplied.is_dir():
            for pattern in (
                "*.json",
                "*.jsonl",
                "*.ndjson",
                "*.json.gz",
                "*.jsonl.gz",
                "*.ndjson.gz",
            ):
                for path in supplied.rglob(pattern):
                    lower = path.name.casefold()
                    if lower in {"manifest.json", "state.json"} or lower.endswith(
                        ".manifest.json"
                    ):
                        continue
                    discovered.add(path.resolve())
            continue
        raise FileNotFoundError(f"Input path does not exist: {supplied}")
    if not discovered:
        raise FileNotFoundError("No JSON or JSONL input files were found")
    return sorted(discovered, key=lambda path: str(path).casefold())


def _read_text(path: Path) -> str:
    if path.name.casefold().endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8-sig") as handle:
            return handle.read()
    return path.read_text(encoding="utf-8-sig")


def read_input_files(paths: Sequence[Path]) -> tuple[list[dict[str, Any]], list[Path]]:
    files = discover_input_files(paths)
    records: list[dict[str, Any]] = []
    for path in files:
        lower = path.name.casefold()
        is_lines = lower.endswith(
            (".jsonl", ".ndjson", ".jsonl.gz", ".ndjson.gz")
        )
        text = _read_text(path)
        if is_lines:
            for line_number, line in enumerate(text.splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}:{line_number}: invalid JSON: {exc}"
                    ) from exc
                records.extend(records_from_payload(payload, path=path))
        else:
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: invalid JSON: {exc}") from exc
            records.extend(records_from_payload(payload, path=path))
    return records, files


def read_fulltext_inputs(
    paths: Sequence[Path],
) -> tuple[list[dict[str, Any]], list[Path]]:
    """Read ordinary inline records or the retriever's audit-plus-text layout."""

    files = discover_input_files(paths)
    materialized_files = [
        path for path in files if path.name.casefold() == "records.jsonl"
    ]
    # records.jsonl is atomically materialized by retrieve_article_fulltext.py;
    # prefer it over the append-only audit, whose final line can be interrupted.
    if materialized_files:
        records, _ = read_input_files(materialized_files)
    else:
        records, _ = read_input_files(paths)
    input_files = set(files)
    private_roots = sorted(
        {
            path.parent.resolve()
            for path in files
            if path.name.casefold() in {"manifest.jsonl", "records.jsonl"}
        },
        key=lambda path: str(path).casefold(),
    )

    def body_candidates(
        text_file: str, expected_hash: Any
    ) -> list[tuple[Path, bytes, str]]:
        candidates: list[tuple[Path, bytes, str]] = []
        for root in private_roots:
            body_path = (root / text_file).resolve()
            try:
                body_path.relative_to(root)
            except ValueError as exc:
                raise ValueError(
                    f"Full-text path escapes its private root: {text_file!r}"
                ) from exc
            if body_path.is_file():
                body_bytes = body_path.read_bytes()
                candidates.append(
                    (body_path, body_bytes, sha256_bytes(body_bytes))
                )
        if isinstance(expected_hash, str):
            candidates = [
                candidate
                for candidate in candidates
                if candidate[2] == expected_hash
            ]
        return candidates

    enriched: list[dict[str, Any]] = []
    for record in records:
        text_file = record.get("text_file")
        expected_hash = record.get("content_sha256")
        inline_text = full_text(record)
        if inline_text:
            if (
                isinstance(expected_hash, str)
                and sha256_bytes(inline_text.encode("utf-8")) != expected_hash
            ):
                raise ValueError(
                    "Inline full-text hash mismatch for "
                    f"{record.get('article_id')!r}"
                )
            if isinstance(text_file, str):
                candidates = body_candidates(text_file, expected_hash)
                if not candidates:
                    raise ValueError(
                        "Materialized full-text record has no matching private body: "
                        f"{record.get('article_id')!r}"
                    )
                input_files.add(candidates[0][0])
            enriched.append(record)
            continue
        if record.get("status") != "retrieved" or not isinstance(text_file, str):
            enriched.append(record)
            continue
        # Audit entries deliberately omit the raw URL/title.  Exact provider ID
        # remains sufficient for matching, and the referenced body stays below
        # the retriever's private output root.
        candidates = body_candidates(text_file, expected_hash)
        if not candidates:
            raise ValueError(
                "Full-text audit record has no body matching its path and hash: "
                f"{record.get('article_id')!r}"
            )
        body_path, body_bytes, _ = candidates[0]
        try:
            body = body_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Full-text body is not UTF-8: {body_path}") from exc
        enriched.append({**record, "full_text": body})
        input_files.add(body_path)
    return enriched, sorted(input_files, key=lambda path: str(path).casefold())


def _normalized_tickers(raw_values: Iterable[Any]) -> set[str]:
    result = set()
    for value in raw_values:
        if not isinstance(value, str):
            continue
        ticker = value.strip().upper().split(":")[-1]
        if ticker:
            result.add(ticker)
    return result


def provider_ticker_values(record: Mapping[str, Any]) -> set[str]:
    """Return only ticker metadata supplied by the provider record."""

    raw_values: list[Any] = []
    for field in ("tickers", "symbols", "vendor_tickers"):
        value = record.get(field)
        if isinstance(value, list):
            raw_values.extend(value)
        elif value not in (None, ""):
            raw_values.append(value)
    if record.get("ticker") not in (None, ""):
        raw_values.append(record["ticker"])
    return _normalized_tickers(raw_values)


def eligible_target_values(
    record: Mapping[str, Any], universe: set[str]
) -> set[str]:
    """Return assignment eligibility without presenting it as provider metadata."""

    raw = record.get("eligible_target_tickers")
    values = raw if isinstance(raw, list) else [raw]
    return _normalized_tickers(values) & universe


def _text_value(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [_text_value(item) for item in value]
        return "\n\n".join(part for part in parts if part)
    if isinstance(value, Mapping):
        for key in ("text", "content", "body", "value", "paragraphs"):
            if key in value and (text := _text_value(value[key])):
                return text
    return ""


def full_text(record: Mapping[str, Any]) -> str:
    for field in FULLTEXT_FIELDS:
        if field in record and (text := _text_value(record[field])):
            return text
    return ""


MOJIBAKE_MARKERS = (
    "\ufffd",
    "ï¿½",
    "Ã",
    "Â",
    "â€",
    "â€™",
    "â€œ",
    "â€˜",
    "ðŸ",
)


def assess_fulltext_quality(
    text: str,
) -> tuple[bool, str | None, dict[str, int | float]]:
    """Conservatively reject only clearly non-English or unusable bodies.

    This is a benchmark-selection guard, not a language classifier.  It is
    deliberately permissive of Latin-script foreign words and isolated
    mojibake.  Its main language check catches bodies dominated by CJK,
    Cyrillic, Arabic, or another non-Latin script even when an English Massive
    description was used to discover the article.
    """

    characters = list(text)
    non_whitespace = [character for character in characters if not character.isspace()]
    letters = [
        character
        for character in characters
        if unicodedata.category(character).startswith("L")
    ]
    latin_letters = [
        character
        for character in letters
        if "LATIN" in unicodedata.name(character, "")
    ]
    non_latin_letters = len(letters) - len(latin_letters)
    controls = [
        character
        for character in characters
        if unicodedata.category(character) in {"Cc", "Cs"}
        and character not in "\t\n\r"
    ]
    replacement_count = text.count("\ufffd") + text.count("ï¿½")
    mojibake_count = sum(text.count(marker) for marker in MOJIBAKE_MARKERS)
    non_whitespace_count = len(non_whitespace)
    letter_count = len(letters)
    metrics: dict[str, int | float] = {
        "character_count": len(characters),
        "non_whitespace_count": non_whitespace_count,
        "letter_count": letter_count,
        "latin_letter_count": len(latin_letters),
        "non_latin_letter_count": non_latin_letters,
        "non_latin_letter_ratio": round(
            non_latin_letters / max(1, letter_count), 6
        ),
        "control_character_count": len(controls),
        "replacement_character_count": replacement_count,
        "mojibake_marker_count": mojibake_count,
    }

    if (
        letter_count >= 80
        and non_latin_letters >= 60
        and non_latin_letters / letter_count >= 0.55
    ):
        return False, "clearly_non_english_fulltext", metrics

    severe_control_corruption = (
        len(controls) >= 5
        and len(controls) / max(1, non_whitespace_count) >= 0.005
    )
    severe_replacement_corruption = (
        replacement_count >= 5
        and replacement_count / max(1, non_whitespace_count) >= 0.005
    )
    severe_mojibake = (
        mojibake_count >= 20
        and mojibake_count / max(1, non_whitespace_count) >= 0.03
    )
    prose_absent = (
        non_whitespace_count >= 200
        and letter_count / non_whitespace_count < 0.03
    )
    if (
        severe_control_corruption
        or severe_replacement_corruption
        or severe_mojibake
        or prose_absent
    ):
        return False, "severely_garbled_fulltext", metrics
    return True, None, metrics


def _source_name(record: Mapping[str, Any]) -> str:
    publisher = record.get("publisher")
    if isinstance(publisher, Mapping):
        return first_string(publisher, ("name", "homepage_url"))
    if isinstance(publisher, str) and publisher.strip():
        return publisher.strip()
    return first_string(record, ("source", "author"))


def _record_match_keys(record: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    fallback = title_time_key(record)
    return {
        "ids": record_ids(record),
        "urls": record_urls(record),
        "title_time": (fallback,) if fallback else (),
    }


def deduplicate_massive(
    records: Sequence[Mapping[str, Any]], universe: set[str]
) -> tuple[list[dict[str, Any]], int]:
    """Collapse query copies that share any ID, canonical URL, or fallback key."""

    parents = list(range(len(records)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    first_by_key: dict[tuple[str, str], int] = {}
    keys_by_record: list[dict[str, tuple[str, ...]]] = []
    for index, record in enumerate(records):
        keys = _record_match_keys(record)
        keys_by_record.append(keys)
        for key_type, values in keys.items():
            for value in values:
                compound = (key_type, value)
                if compound in first_by_key:
                    union(index, first_by_key[compound])
                else:
                    first_by_key[compound] = index

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        groups[find(index)].append(index)

    deduplicated: list[dict[str, Any]] = []
    for indices in groups.values():
        group_records = [dict(records[index]) for index in indices]
        chosen = min(
            group_records,
            key=lambda row: (
                -int(bool(first_string(row, ("description",)))),
                -len(first_string(row, ("description",))),
                canonical_json(row),
            ),
        )
        ids = sorted(
            {value for index in indices for value in keys_by_record[index]["ids"]}
        )
        urls = sorted(
            {value for index in indices for value in keys_by_record[index]["urls"]}
        )
        fallback = sorted(
            {
                value
                for index in indices
                for value in keys_by_record[index]["title_time"]
            }
        )
        provider_tickers = sorted(
            {
                ticker
                for row in group_records
                for ticker in provider_ticker_values(row)
            }
        )
        eligible_target_tickers = sorted(
            {
                ticker
                for row in group_records
                for ticker in eligible_target_values(row, universe)
            }
        )
        assignment_tickers = sorted(
            (set(provider_tickers) & universe) | set(eligible_target_tickers)
        )
        identity_kind, identity_value = (
            ("id", ids[0])
            if ids
            else ("url", urls[0])
            if urls
            else ("title_time", fallback[0])
            if fallback
            else ("payload", stable_hash(canonical_json(chosen)))
        )
        deduplicated.append(
            {
                "record": chosen,
                "ids": tuple(ids),
                "urls": tuple(urls),
                "title_time": tuple(fallback),
                "provider_tickers": tuple(provider_tickers),
                "eligible_target_tickers": tuple(eligible_target_tickers),
                "assignment_tickers": tuple(assignment_tickers),
                "identity": f"{identity_kind}:{identity_value}",
            }
        )
    deduplicated.sort(key=lambda row: row["identity"])
    return deduplicated, len(records) - len(deduplicated)


class MatchIndex:
    def __init__(self, records: Sequence[Mapping[str, Any]]) -> None:
        unique: dict[str, dict[str, Any]] = {}
        for record in records:
            unique.setdefault(canonical_json(record), dict(record))
        self.records = list(unique.values())
        self.by_id: dict[str, list[int]] = defaultdict(list)
        self.by_url: dict[str, list[int]] = defaultdict(list)
        self.by_title_time: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(self.records):
            for value in record_ids(record):
                self.by_id[value].append(index)
            for value in record_urls(record):
                self.by_url[value].append(index)
            if fallback := title_time_key(record):
                self.by_title_time[fallback].append(index)

    def match(
        self, keys: Mapping[str, Sequence[str]], *, content_getter: Any
    ) -> tuple[dict[str, Any] | None, str | None]:
        for method, index, key_name in (
            ("exact_id", self.by_id, "ids"),
            ("exact_url", self.by_url, "urls"),
            (
                "normalized_title_timestamp",
                self.by_title_time,
                "title_time",
            ),
        ):
            candidate_indices = sorted(
                {
                    candidate
                    for key in keys.get(key_name, ())
                    for candidate in index.get(key, [])
                }
            )
            if not candidate_indices:
                continue
            candidates = [self.records[candidate] for candidate in candidate_indices]
            content_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for candidate in candidates:
                content_groups[content_getter(candidate)].append(candidate)
            nonempty = {
                content: rows for content, rows in content_groups.items() if content
            }
            if len(nonempty) > 1:
                raise ValueError(
                    f"Ambiguous {method} match has {len(nonempty)} conflicting texts"
                )
            if not nonempty:
                # A failed retrieval row can share an exact ID with a later
                # successful row indexed by URL or title/time.  Empty content
                # is not a usable match, so continue down the frozen priority.
                continue
            rows = next(iter(nonempty.values()))
            return min(rows, key=canonical_json), method
        return None, None


def load_universe(path: Path) -> tuple[list[str], dict[str, dict[str, Any]], Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    target_payload = json.loads(
        DEFAULT_TARGET_METADATA.read_text(encoding="utf-8")
    )
    target_rows = target_payload.get("targets")
    if not isinstance(target_rows, Mapping):
        raise ValueError("Canonical target metadata must contain a targets object")
    company_names = {
        str(ticker).upper(): str(record.get("company", "")).strip()
        for ticker, record in target_rows.items()
        if isinstance(record, Mapping) and str(record.get("company", "")).strip()
    }
    sectors = payload.get("sectors")
    if not isinstance(sectors, list) or not sectors:
        raise ValueError("Universe must contain a nonempty sectors array")
    tickers: list[str] = []
    target_by_ticker: dict[str, dict[str, Any]] = {}
    manifest_sectors = []
    for sector in sectors:
        if not isinstance(sector, Mapping):
            raise ValueError("Universe sector rows must be objects")
        name = str(sector.get("name", "")).strip()
        benchmark = str(sector.get("benchmark", "")).strip().upper()
        stocks = [
            str(value).strip().upper()
            for value in sector.get("stocks", [])
            if str(value).strip()
        ]
        if not name or not benchmark or not stocks:
            raise ValueError("Each sector requires name, benchmark, and stocks")
        if len(stocks) != len(set(stocks)):
            raise ValueError(f"Sector {name!r} contains duplicate stocks")
        manifest_sectors.append(
            {"name": name, "benchmark": benchmark, "stocks": stocks}
        )
        for ticker in stocks:
            if ticker in target_by_ticker:
                raise ValueError(f"Ticker {ticker!r} occurs in multiple sectors")
            peers = [stock for stock in stocks if stock != ticker]
            target_by_ticker[ticker] = {
                "company": company_names.get(ticker, ticker),
                "ticker": ticker,
                "sector": name,
                "sector_benchmark": benchmark,
                "known_sector_peers": peers,
            }
            tickers.append(ticker)
    return tickers, target_by_ticker, {
        "source_version": payload.get("version"),
        "target_metadata_version": target_payload.get("version"),
        "target_metadata_path": str(DEFAULT_TARGET_METADATA.resolve()),
        "target_metadata_sha256": sha256_file(DEFAULT_TARGET_METADATA),
        "sectors": manifest_sectors,
    }


def _candidate_keys(candidate: Mapping[str, Any]) -> dict[str, Sequence[str]]:
    return {
        "ids": candidate["ids"],
        "urls": candidate["urls"],
        "title_time": candidate["title_time"],
    }


def eligible_candidates(
    massive: Sequence[Mapping[str, Any]],
    fulltext_index: MatchIndex,
    *,
    cutoff: datetime,
) -> tuple[
    list[dict[str, Any]],
    dict[str, int],
    list[dict[str, Any]],
]:
    eligible = []
    reasons: Counter[str] = Counter()
    quality_exclusions: list[dict[str, Any]] = []
    for candidate in massive:
        record = candidate["record"]
        description = first_string(record, ("description",))
        if not description:
            reasons["missing_massive_description"] += 1
            continue
        try:
            published = publication_time(record)
        except ValueError:
            reasons["invalid_or_missing_publication_timestamp"] += 1
            continue
        if published <= cutoff:
            reasons["not_post_cutoff"] += 1
            continue
        if not candidate["assignment_tickers"]:
            reasons["no_universe_ticker"] += 1
            continue
        try:
            matched, method = fulltext_index.match(
                _candidate_keys(candidate), content_getter=full_text
            )
        except ValueError:
            reasons["ambiguous_fulltext_match"] += 1
            continue
        if matched is None:
            reasons["no_fulltext_match"] += 1
            continue
        text = full_text(matched)
        if not text:
            reasons["matched_fulltext_empty"] += 1
            continue
        article_id = "massive_" + stable_hash(candidate["identity"])[:24]
        accepted, quality_reason, quality_metrics = assess_fulltext_quality(text)
        if not accepted:
            if quality_reason is None:
                raise AssertionError("A rejected body must have a quality reason")
            reasons[quality_reason] += 1
            quality_exclusions.append(
                {
                    "article_id": article_id,
                    "reason": quality_reason,
                    "fulltext_match_method": method,
                    "metrics": quality_metrics,
                }
            )
            continue
        eligible.append(
            {
                **candidate,
                "article_id": article_id,
                "published": published,
                "description": description,
                "headline": first_string(record, TITLE_FIELDS),
                "source": _source_name(record),
                "fulltext": text,
                "fulltext_match_method": method,
            }
        )
        reasons["eligible"] += 1
    eligible.sort(key=lambda row: row["article_id"])
    quality_exclusions.sort(key=lambda row: row["article_id"])
    return (
        eligible,
        dict(sorted(reasons.items())),
        quality_exclusions,
    )


def attach_alpha_summaries(
    candidates: Sequence[Mapping[str, Any]],
    alpha_index: MatchIndex | None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Attach usable Alpha summaries without changing candidate order."""

    attached: list[dict[str, Any]] = []
    methods: Counter[str] = Counter()
    for candidate in candidates:
        alpha_record: dict[str, Any] | None = None
        alpha_method: str | None
        if alpha_index is None:
            alpha_method = "not_provided"
        else:
            try:
                alpha_record, alpha_method = alpha_index.match(
                    _candidate_keys(candidate),
                    content_getter=lambda row: first_string(row, ("summary",)),
                )
            except ValueError:
                alpha_record, alpha_method = None, "ambiguous"
            if alpha_record is None:
                alpha_method = alpha_method or "unmatched"
        summary = (
            first_string(alpha_record, ("summary",))
            if alpha_record is not None
            else ""
        )
        if not summary and alpha_method not in {"ambiguous", "not_provided"}:
            alpha_method = "unmatched"
        methods[alpha_method] += 1
        attached.append(
            {
                **candidate,
                "alpha_summary": summary,
                "alpha_match_method": alpha_method,
            }
        )
    return attached, dict(sorted(methods.items()))


def availability_by_ticker(
    candidates: Sequence[Mapping[str, Any]], tickers: Sequence[str]
) -> dict[str, int]:
    return {
        ticker: sum(
            ticker in candidate["assignment_tickers"]
            for candidate in candidates
        )
        for ticker in tickers
    }


def balanced_assignment(
    candidates: Sequence[Mapping[str, Any]],
    tickers: Sequence[str],
    *,
    per_stock: int,
) -> dict[str, str]:
    """Return article_id -> ticker using deterministic maximum bipartite matching."""

    if per_stock <= 0:
        raise ValueError("per_stock must be positive")
    known = set(tickers)
    candidates_by_ticker: dict[str, list[str]] = {}
    candidate_tickers: dict[str, set[str]] = {}
    for candidate in candidates:
        article_id = str(candidate["article_id"])
        candidate_tickers[article_id] = (
            set(candidate["assignment_tickers"]) & known
        )
    for ticker in tickers:
        candidates_by_ticker[ticker] = sorted(
            (
                article_id
                for article_id, mentioned in candidate_tickers.items()
                if ticker in mentioned
            ),
            key=lambda article_id: stable_hash(SELECTION_SEED, ticker, article_id),
        )

    article_to_slot: dict[str, tuple[str, int]] = {}

    def augment(slot: tuple[str, int], seen: set[str]) -> bool:
        ticker, _ = slot
        for article_id in candidates_by_ticker[ticker]:
            if article_id in seen:
                continue
            seen.add(article_id)
            previous = article_to_slot.get(article_id)
            if previous is None or augment(previous, seen):
                article_to_slot[article_id] = slot
                return True
        return False

    unmatched = []
    for slot_number in range(per_stock):
        for ticker in tickers:
            slot = (ticker, slot_number)
            if not augment(slot, set()):
                unmatched.append(slot)
    if unmatched:
        available = {
            ticker: len(candidates_by_ticker[ticker]) for ticker in tickers
        }
        raise ValueError(
            "Insufficient distinct eligible documents for balanced assignment; "
            f"unmatched_slots={unmatched!r}, eligible_by_ticker={available!r}"
        )
    assignment = {
        article_id: slot[0] for article_id, slot in article_to_slot.items()
    }
    expected = len(tickers) * per_stock
    if len(assignment) != expected:
        raise AssertionError(
            f"Balanced matching selected {len(assignment)} records, expected {expected}"
        )
    return assignment


def deterministic_split(
    article_to_ticker: Mapping[str, str],
    tickers: Sequence[str],
    *,
    development_count: int,
) -> tuple[set[str], set[str], dict[str, int]]:
    total = len(article_to_ticker)
    if development_count < 0 or development_count > total:
        raise ValueError("development_count must fall between zero and benchmark size")
    base, remainder = divmod(development_count, len(tickers))
    extra_tickers = set(
        sorted(tickers, key=lambda ticker: stable_hash(SPLIT_SEED, ticker))[:remainder]
    )
    quotas = {
        ticker: base + int(ticker in extra_tickers) for ticker in tickers
    }
    development: set[str] = set()
    for ticker in tickers:
        ids = sorted(
            (
                article_id
                for article_id, assigned in article_to_ticker.items()
                if assigned == ticker
            ),
            key=lambda article_id: stable_hash(SPLIT_SEED, ticker, article_id),
        )
        if quotas[ticker] > len(ids):
            raise ValueError(
                f"Development quota {quotas[ticker]} exceeds {ticker} count {len(ids)}"
            )
        development.update(ids[: quotas[ticker]])
    evaluation = set(article_to_ticker) - development
    if len(development) != development_count:
        raise AssertionError("Deterministic split produced the wrong development count")
    return development, evaluation, quotas


def _split_oversized_unit(text: str, maximum: int) -> list[str]:
    text = " ".join(text.split())
    if len(text) <= maximum:
        return [text] if text else []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    units: list[str] = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= maximum:
            units.append(sentence)
            continue
        words = sentence.split()
        current = ""
        for word in words:
            if len(word) > maximum:
                if current:
                    units.append(current)
                    current = ""
                units.extend(
                    word[index : index + maximum]
                    for index in range(0, len(word), maximum)
                )
                continue
            proposed = word if not current else f"{current} {word}"
            if len(proposed) <= maximum:
                current = proposed
            else:
                units.append(current)
                current = word
        if current:
            units.append(current)
    return units


def paragraph_chunks(text: str, maximum: int = DEFAULT_MAX_CHUNK_CHARACTERS) -> list[str]:
    if maximum < 64:
        raise ValueError("max chunk characters must be at least 64")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    paragraphs = [
        " ".join(paragraph.split())
        for paragraph in re.split(r"\n\s*\n+", normalized)
        if paragraph.strip()
    ]
    units = [
        unit
        for paragraph in paragraphs
        for unit in _split_oversized_unit(paragraph, maximum)
    ]
    chunks: list[str] = []
    current = ""
    for unit in units:
        proposed = unit if not current else f"{current}\n\n{unit}"
        if len(proposed) <= maximum:
            current = proposed
        else:
            if current:
                chunks.append(current)
            current = unit
    if current:
        chunks.append(current)
    if not chunks or any(len(chunk) > maximum for chunk in chunks):
        raise ValueError("Full text could not be split into nonempty bounded chunks")
    return chunks


def _common_record(
    candidate: Mapping[str, Any],
    *,
    row_number: int,
    target: Mapping[str, Any],
    split: str,
    variant: str,
    article_text: str,
    match_method: str,
) -> dict[str, Any]:
    return {
        "row_number": row_number,
        "article_id": candidate["article_id"],
        "time_published_utc": format_utc(candidate["published"]),
        "source": candidate["source"],
        "headline": candidate["headline"],
        "article_text": article_text,
        "vendor_tickers": list(candidate["provider_tickers"]),
        "target": dict(target),
        "text_variant": variant,
        "benchmark_split": split,
        "source_match_method": match_method,
    }


def _private_output_check(output_root: Path) -> None:
    resolved = output_root.resolve()
    try:
        relative = resolved.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError:
        return
    if not relative.parts or relative.parts[0].casefold() != "data":
        raise ValueError(
            "Licensed benchmark outputs inside the repository must remain under data/"
        )


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(canonical_json(record) + "\n")
    os.replace(temporary, path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _input_manifest(paths: Sequence[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in paths
    ]


def build_benchmark(
    *,
    massive_paths: Sequence[Path],
    fulltext_paths: Sequence[Path],
    alpha_paths: Sequence[Path] = (),
    universe_path: Path = DEFAULT_UNIVERSE,
    schema_path: Path = DEFAULT_SCHEMA,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    cutoff: str | datetime = DEFAULT_CUTOFF,
    per_stock: int = DEFAULT_PER_STOCK,
    development_count: int = DEFAULT_DEVELOPMENT_COUNT,
    max_chunk_characters: int = DEFAULT_MAX_CHUNK_CHARACTERS,
    require_alpha_match: bool = False,
    fixed_assignment_manifest: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    _private_output_check(output_root)
    if (
        require_alpha_match
        and output_root.resolve() == DEFAULT_OUTPUT_ROOT.resolve()
    ):
        raise ValueError(
            "--require-alpha-match requires a caller-specified output_root "
            "separate from the default ablation benchmark"
        )
    cutoff_time = parse_timestamp(cutoff) if isinstance(cutoff, str) else cutoff
    cutoff_time = cutoff_time.astimezone(timezone.utc)
    tickers, targets, universe_manifest = load_universe(universe_path)
    universe_set = set(tickers)
    if len(tickers) != len(universe_set):
        raise ValueError("Universe stock tickers must be unique")

    massive_records, massive_files = read_input_files(massive_paths)
    fulltext_records, fulltext_files = read_fulltext_inputs(fulltext_paths)
    if alpha_paths:
        alpha_records, alpha_files = read_input_files(alpha_paths)
    else:
        alpha_records, alpha_files = [], []

    deduplicated, duplicate_count = deduplicate_massive(
        massive_records, universe_set
    )
    fulltext_index = MatchIndex(fulltext_records)
    eligible, eligibility, quality_exclusions = eligible_candidates(
        deduplicated, fulltext_index, cutoff=cutoff_time
    )
    alpha_index = MatchIndex(alpha_records) if alpha_records else None
    eligible, alpha_eligible_method_counts = attach_alpha_summaries(
        eligible, alpha_index
    )
    alpha_matched = [
        candidate for candidate in eligible if candidate["alpha_summary"]
    ]
    selection_pool = alpha_matched if require_alpha_match else eligible
    available_before_alpha = availability_by_ticker(eligible, tickers)
    available_with_alpha = availability_by_ticker(alpha_matched, tickers)
    available_for_assignment = availability_by_ticker(selection_pool, tickers)
    if require_alpha_match and any(
        available_for_assignment[ticker] < per_stock for ticker in tickers
    ):
        raise ValueError(
            "Insufficient nonempty Alpha-matched documents for balanced "
            f"assignment; required_per_ticker={per_stock}, "
            f"alpha_matched_eligible_by_ticker={available_for_assignment!r}"
        )
    fixed_assignment_source: dict[str, Any] | None = None
    if fixed_assignment_manifest is not None:
        fixed_path = fixed_assignment_manifest.resolve()
        output_manifest_path = (output_root / "manifest.json").resolve()
        if fixed_path == output_manifest_path:
            raise ValueError(
                "The fixed-assignment manifest must be a preserved source "
                "copy outside the output manifest path; otherwise --overwrite "
                "would invalidate its recorded SHA-256 provenance."
            )
        fixed_payload = json.loads(fixed_path.read_text(encoding="utf-8-sig"))
        raw_assignment = (
            fixed_payload.get("target_metadata", {})
            .get("article_target_tickers")
        )
        if not isinstance(raw_assignment, dict) or not raw_assignment:
            raise ValueError(
                "Fixed-assignment manifest has no "
                "target_metadata.article_target_tickers mapping"
            )
        assignment = {
            str(article_id): str(ticker).upper()
            for article_id, ticker in raw_assignment.items()
        }
        expected = len(tickers) * per_stock
        if len(assignment) != expected:
            raise ValueError(
                "Fixed assignment has wrong document count: "
                f"{len(assignment)} != {expected}"
            )
        assignment_counts = Counter(assignment.values())
        invalid_tickers = sorted(set(assignment_counts) - set(tickers))
        if invalid_tickers:
            raise ValueError(
                f"Fixed assignment contains unknown tickers: {invalid_tickers!r}"
            )
        unbalanced = {
            ticker: assignment_counts[ticker]
            for ticker in tickers
            if assignment_counts[ticker] != per_stock
        }
        if unbalanced:
            raise ValueError(
                f"Fixed assignment is not balanced: {unbalanced!r}"
            )
        eligible_by_id = {
            candidate["article_id"]: candidate for candidate in selection_pool
        }
        eligible_ids = set(eligible_by_id)
        missing_ids = sorted(set(assignment) - eligible_ids)
        if missing_ids:
            raise ValueError(
                "Fixed assignment contains articles that are no longer eligible: "
                f"{missing_ids[:10]!r}"
            )
        ineligible_assignments = sorted(
            (article_id, ticker)
            for article_id, ticker in assignment.items()
            if ticker
            not in set(eligible_by_id[article_id].get("assignment_tickers", ()))
        )
        if ineligible_assignments:
            raise ValueError(
                "Fixed assignment maps articles to tickers for which they are "
                "no longer eligible: "
                f"{ineligible_assignments[:10]!r}"
            )
        fixed_assignment_source = {
            "path": str(fixed_path),
            "sha256": sha256_file(fixed_path),
        }
    else:
        try:
            assignment = balanced_assignment(
                selection_pool, tickers, per_stock=per_stock
            )
        except ValueError as exc:
            if require_alpha_match:
                raise ValueError(
                    "Alpha-match provider comparison cannot satisfy distinct "
                    "balanced assignment despite per-ticker availability; "
                    f"alpha_matched_eligible_by_ticker={available_for_assignment!r}; "
                    f"{exc}"
                ) from exc
            raise
    selected_by_id = {
        candidate["article_id"]: candidate
        for candidate in selection_pool
        if candidate["article_id"] in assignment
    }
    development_ids, evaluation_ids, development_quotas = deterministic_split(
        assignment, tickers, development_count=development_count
    )
    ordered_ids = sorted(
        selected_by_id,
        key=lambda article_id: (
            tickers.index(assignment[article_id]),
            stable_hash(SELECTION_SEED, assignment[article_id], article_id),
        ),
    )
    description_rows: list[dict[str, Any]] = []
    alpha_rows: list[dict[str, Any]] = []
    chunk_rows: list[dict[str, Any]] = []
    chunk_mapping: list[dict[str, Any]] = []
    alpha_match_methods: Counter[str] = Counter()
    fulltext_match_methods: Counter[str] = Counter()
    selected_counts_by_ticker: Counter[str] = Counter()
    selected_counts_by_sector: Counter[str] = Counter()
    target_provider_tagged_count = 0
    target_or_peer_provider_tagged_count = 0
    no_provider_ticker_count = 0
    assignment_basis_by_article: dict[str, str] = {}
    assignment_basis_counts: Counter[str] = Counter()

    for row_number, article_id in enumerate(ordered_ids, start=1):
        candidate = selected_by_id[article_id]
        ticker = assignment[article_id]
        target = targets[ticker]
        split = "development" if article_id in development_ids else "evaluation"
        selected_counts_by_ticker[ticker] += 1
        selected_counts_by_sector[target["sector"]] += 1
        provider_tickers = set(candidate["provider_tickers"])
        target_provider_tagged_count += int(ticker in provider_tickers)
        target_or_peer_provider_tagged_count += int(
            bool(
                provider_tickers
                & {
                    ticker,
                    *target["known_sector_peers"],
                }
            )
        )
        no_provider_ticker_count += int(not provider_tickers)
        if ticker in provider_tickers:
            assignment_basis = "direct_target_tag"
        elif provider_tickers & set(target["known_sector_peers"]):
            assignment_basis = "peer_only_tag"
        else:
            assignment_basis = "expanded_sector_or_market_only"
        assignment_basis_by_article[article_id] = assignment_basis
        assignment_basis_counts[assignment_basis] += 1
        fulltext_match_methods[str(candidate["fulltext_match_method"])] += 1
        description_rows.append(
            _common_record(
                candidate,
                row_number=row_number,
                target=target,
                split=split,
                variant="massive_description",
                article_text=candidate["description"],
                match_method="native_massive_description",
            )
        )

        alpha_method = str(candidate["alpha_match_method"])
        alpha_match_methods[alpha_method] += 1
        if candidate["alpha_summary"]:
            alpha_rows.append(
                _common_record(
                    candidate,
                    row_number=row_number,
                    target=target,
                    split=split,
                    variant="alpha_summary",
                    article_text=str(candidate["alpha_summary"]),
                    match_method=alpha_method,
                )
            )

        chunks = paragraph_chunks(
            candidate["fulltext"], maximum=max_chunk_characters
        )
        reconstructed_text = "\n\n".join(chunks)
        source_text_sha256 = sha256_bytes(
            candidate["fulltext"].encode("utf-8")
        )
        reconstructed_text_sha256 = sha256_bytes(
            reconstructed_text.encode("utf-8")
        )
        chunk_count = len(chunks)
        for chunk_index, chunk in enumerate(chunks):
            chunk_id = f"{article_id}__chunk_{chunk_index:04d}"
            row = _common_record(
                candidate,
                row_number=len(chunk_rows) + 1,
                target=target,
                split=split,
                variant="fulltext_evidence_chunks",
                article_text=chunk,
                match_method=str(candidate["fulltext_match_method"]),
            )
            row["article_id"] = chunk_id
            row["chunk_id"] = chunk_id
            row["parent_article_id"] = article_id
            row["chunk_index"] = chunk_index
            row["chunk_count"] = chunk_count
            row["source_fulltext_sha256"] = source_text_sha256
            row["reconstructed_fulltext_sha256"] = (
                reconstructed_text_sha256
            )
            chunk_rows.append(row)
            chunk_mapping.append(
                {
                    "chunk_id": chunk_id,
                    "article_id": article_id,
                    "chunk_index": chunk_index,
                    "chunk_count": chunk_count,
                    "source_fulltext_sha256": source_text_sha256,
                    "reconstructed_fulltext_sha256": (
                        reconstructed_text_sha256
                    ),
                }
            )

    expected_total = len(tickers) * per_stock
    if len(description_rows) != expected_total:
        raise AssertionError("Description variant does not satisfy the balanced contract")
    if require_alpha_match and len(alpha_rows) != expected_total:
        raise AssertionError(
            "Alpha-required provider comparison lacks a summary for a selected article"
        )
    if {row["parent_article_id"] for row in chunk_rows} != set(ordered_ids):
        raise AssertionError("Every selected article must have at least one full-text chunk")

    output_root.mkdir(parents=True, exist_ok=True)
    output_paths = {
        variant: output_root / file_name
        for variant, file_name in VARIANT_FILES.items()
    }
    manifest_path = output_root / "manifest.json"
    existing = [
        path
        for path in [*output_paths.values(), manifest_path]
        if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            "Benchmark output already exists; pass --overwrite to replace known files: "
            + ", ".join(str(path) for path in existing)
        )

    _write_jsonl(output_paths["massive_description"], description_rows)
    _write_jsonl(output_paths["alpha_summary"], alpha_rows)
    _write_jsonl(output_paths["fulltext_evidence_chunks"], chunk_rows)

    publication_values = [selected_by_id[value]["published"] for value in ordered_ids]
    alpha_article_ids = [row["article_id"] for row in alpha_rows]
    chunk_lengths = [len(row["article_text"]) for row in chunk_rows]
    variant_ids = {
        "massive_description": ordered_ids,
        "alpha_summary": alpha_article_ids,
        "fulltext_evidence_chunks": ordered_ids,
    }
    source_inputs = {
        "massive": _input_manifest(massive_files),
        "fulltext": _input_manifest(fulltext_files),
        "alpha": _input_manifest(alpha_files),
    }
    manifest: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "status": "complete",
        "builder_script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__)),
        },
        "private_licensed_text": True,
        "privacy": {
            "default_output_root": "data/benchmarks/news_text_ablation_300/v1",
            "repository_outputs_required_under_ignored_data_directory": True,
            "tracked_source_files_contain_only_code_and_synthetic_test_text": True,
        },
        "selection": {
            "cutoff_utc": format_utc(cutoff_time),
            "cutoff_relation": "strictly_after",
            "seed": SELECTION_SEED,
            "per_stock": per_stock,
            "stock_count": len(tickers),
            "sector_count": len(universe_manifest["sectors"]),
            "document_count": expected_total,
            "method": (
                "fixed article-to-ticker assignment from prior manifest"
                if fixed_assignment_source
                else (
                    "deterministic maximum bipartite matching of unique eligible "
                    "articles to equal-capacity ticker slots"
                )
            ),
            "fixed_assignment_source": fixed_assignment_source,
            "massive_raw_record_count": len(massive_records),
            "massive_deduplicated_record_count": len(deduplicated),
            "massive_duplicate_query_copy_count": duplicate_count,
            "eligibility_counts": eligibility,
            "require_alpha_match": require_alpha_match,
            "eligible_before_alpha_requirement": len(eligible),
            "eligible_with_nonempty_alpha_summary": len(alpha_matched),
            "eligible_for_balanced_assignment": len(selection_pool),
        },
        "ticker_provenance_contract": {
            "provider_tickers": (
                "Only ticker metadata present in provider fields; emitted to "
                "local models as vendor_tickers."
            ),
            "eligible_target_tickers": (
                "Expanded target-assignment eligibility from the candidate "
                "queue; never emitted as vendor metadata."
            ),
            "assignment_tickers": (
                "Union of in-universe provider tickers and expanded eligibility; "
                "used only for balanced assignment."
            ),
            "selected_target_provider_tagged_count": (
                target_provider_tagged_count
            ),
            "selected_target_or_peer_provider_tagged_count": (
                target_or_peer_provider_tagged_count
            ),
            "selected_no_provider_ticker_count": no_provider_ticker_count,
            "assignment_basis_definitions": {
                "direct_target_tag": (
                    "The assigned target ticker is present in provider ticker "
                    "metadata."
                ),
                "peer_only_tag": (
                    "The assigned target is absent, but at least one known "
                    "same-sector peer is present in provider ticker metadata."
                ),
                "expanded_sector_or_market_only": (
                    "Neither the assigned target nor a known same-sector peer is "
                    "provider-tagged; assignment relies on separately recorded "
                    "sector or market eligibility."
                ),
            },
            "selected_assignment_basis_counts": dict(
                sorted(assignment_basis_counts.items())
            ),
            "article_assignment_basis": {
                article_id: assignment_basis_by_article[article_id]
                for article_id in ordered_ids
            },
        },
        "provider_comparison": {
            "require_alpha_match": require_alpha_match,
            "alpha_input_provided": bool(alpha_files),
            "nonempty_alpha_summary_required_before_assignment": (
                require_alpha_match
            ),
            "eligible_before_alpha_requirement": len(eligible),
            "eligible_with_nonempty_alpha_summary": len(alpha_matched),
            "eligible_after_requirement": len(selection_pool),
            "eligible_by_ticker_before_alpha_requirement": (
                available_before_alpha
            ),
            "alpha_matched_eligible_by_ticker": available_with_alpha,
            "assignment_pool_by_ticker": available_for_assignment,
        },
        "fulltext_quality_gate": {
            "applied_before_balanced_assignment": True,
            "raw_retrieval_modified": False,
            "policy_version": 1,
            "policy": (
                "Conservative Unicode/script and corruption checks reject "
                "clearly non-Latin-script bodies and severely garbled text; "
                "isolated mojibake is retained."
            ),
            "excluded_count": len(quality_exclusions),
            "exclusion_counts": dict(
                sorted(
                    Counter(
                        str(row["reason"]) for row in quality_exclusions
                    ).items()
                )
            ),
            "excluded_records": quality_exclusions,
        },
        "publication_range": {
            "earliest_utc": format_utc(min(publication_values)),
            "latest_utc": format_utc(max(publication_values)),
        },
        "coverage": {
            "massive_description_articles": len(description_rows),
            "massive_description_rate": len(description_rows) / expected_total,
            "fulltext_articles": len(
                {row["parent_article_id"] for row in chunk_rows}
            ),
            "fulltext_rate": len(
                {row["parent_article_id"] for row in chunk_rows}
            )
            / expected_total,
            "alpha_summary_articles": len(alpha_rows),
            "alpha_summary_rate": len(alpha_rows) / expected_total,
        },
        "matching": {
            "priority": [
                "exact_id",
                "exact_url",
                "normalized_title_timestamp",
            ],
            "timestamp_normalization": "UTC with one-second precision",
            "title_normalization": "HTML-unescaped NFKC casefolded alphanumeric words",
            "url_normalization": (
                "lowercase origin, normalized path, fragment/tracking-query removal"
            ),
            "fulltext_method_counts": dict(sorted(fulltext_match_methods.items())),
            "alpha_method_counts": dict(sorted(alpha_match_methods.items())),
            "alpha_eligible_method_counts": alpha_eligible_method_counts,
            "alpha_selected_method_counts": dict(
                sorted(alpha_match_methods.items())
            ),
        },
        "balance": {
            "selected_by_ticker": {
                ticker: selected_counts_by_ticker[ticker] for ticker in tickers
            },
            "selected_by_sector": {
                sector["name"]: selected_counts_by_sector[sector["name"]]
                for sector in universe_manifest["sectors"]
            },
        },
        "source_metadata": {
            "article_sources": {
                article_id: selected_by_id[article_id]["source"]
                for article_id in ordered_ids
            },
            "selected_by_source": dict(
                sorted(
                    Counter(
                        selected_by_id[article_id]["source"]
                        for article_id in ordered_ids
                    ).items()
                )
            ),
        },
        "target_metadata": {
            "universe_path": str(universe_path.resolve()),
            "universe_sha256": sha256_file(universe_path),
            **universe_manifest,
            "targets": {ticker: targets[ticker] for ticker in tickers},
            "article_target_tickers": {
                article_id: assignment[article_id] for article_id in ordered_ids
            },
        },
        "evaluation_schema": {
            "path": str(schema_path.resolve()),
            "sha256": sha256_file(schema_path),
        },
        "split": {
            "seed": SPLIT_SEED,
            "method": (
                "ticker-stratified SHA-256 order; equal base quota plus "
                "deterministically ranked remainder tickers"
            ),
            "development_count": len(development_ids),
            "evaluation_count": len(evaluation_ids),
            "development_quota_by_ticker": development_quotas,
            "development_article_ids": [
                value for value in ordered_ids if value in development_ids
            ],
            "evaluation_article_ids": [
                value for value in ordered_ids if value in evaluation_ids
            ],
            "development_article_ids_sha256": stable_hash(
                *[value for value in ordered_ids if value in development_ids]
            ),
            "evaluation_article_ids_sha256": stable_hash(
                *[value for value in ordered_ids if value in evaluation_ids]
            ),
        },
        "variants": {
            variant: {
                "file": VARIANT_FILES[variant],
                "article_count": len(article_ids),
                "article_ids": article_ids,
                "article_ids_sha256": stable_hash(*article_ids),
            }
            for variant, article_ids in variant_ids.items()
        },
        "fulltext_chunking": {
            "algorithm": (
                "paragraph packing with sentence/word/hard splitting only for "
                "oversized paragraphs"
            ),
            "max_characters": max_chunk_characters,
            "chunk_count": len(chunk_rows),
            "minimum_chunk_characters": min(chunk_lengths),
            "maximum_chunk_characters_observed": max(chunk_lengths),
            "mean_chunk_characters": sum(chunk_lengths) / len(chunk_lengths),
            "reconstruction": {
                "join_separator": "\\n\\n",
                "source_hash_recorded": True,
                "normalized_reconstruction_hash_recorded": True,
                "byte_identity_with_source_claimed": False,
            },
            "chunk_id_to_article_id": chunk_mapping,
            "aggregation": {
                "group_key": "parent article_id from chunk_id_to_article_id",
                "field_method_priority": [
                    (
                        "arithmetic mean across chunks with complete per-label "
                        "candidate scores, when available"
                    ),
                    "plurality of valid chunk labels",
                ],
                "tie_break": "evaluation schema label order",
                "missing_result": None,
            },
        },
        "source_inputs": source_inputs,
    }
    for variant, path in output_paths.items():
        manifest["variants"][variant].update(
            {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "contains_licensed_text": True,
            }
        )
    manifest["source_inputs_sha256"] = sha256_bytes(
        canonical_json(source_inputs).encode("utf-8")
    )
    _write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    args = parse_args()
    manifest = build_benchmark(
        massive_paths=args.massive,
        fulltext_paths=args.fulltext,
        alpha_paths=args.alpha,
        universe_path=args.universe,
        schema_path=args.schema,
        output_root=args.output_root,
        cutoff=args.cutoff,
        per_stock=args.per_stock,
        development_count=args.development_count,
        max_chunk_characters=args.max_chunk_characters,
        require_alpha_match=args.require_alpha_match,
        fixed_assignment_manifest=args.fixed_assignment_manifest,
        overwrite=args.overwrite,
    )
    print(
        "Wrote "
        f"{manifest['selection']['document_count']} balanced articles, "
        f"{manifest['fulltext_chunking']['chunk_count']} full-text chunks, and "
        f"{manifest['coverage']['alpha_summary_articles']} Alpha-summary rows "
        f"to {args.output_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
