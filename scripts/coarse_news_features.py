"""Coarse-label mapping and deterministic gates for FLAN-T5 v0.3.

The reference mapping in this module deliberately coarsens the existing
GPT-5.6 annotations.  It does not create new human ground truth.  The
production gate uses only point-in-time article input and target metadata;
it never reads the reference labels or future market outcomes.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA_PATH = REPOSITORY_ROOT / "config" / "news_feature_schema_coarse.json"

COARSE_FIELDS = (
    "shock_scope",
    "event_family",
    "information_status",
    "directional_alignment",
)

APPLICABLE_RELEVANCE = frozenset({"direct_target", "sector_or_peer", "macro_relevant"})
INAPPLICABLE_RELEVANCE = frozenset({"irrelevant", "insufficient"})

SHOCK_SCOPE_MAP = {
    "firm_specific": "idiosyncratic",
    "peer_specific": "idiosyncratic",
    "sector_wide": "common",
    "macro_market": "common",
    "mixed": "mixed",
    "unclear": "unclear",
}

EVENT_FAMILY_MAP = {
    "earnings": "earnings_guidance",
    "guidance": "earnings_guidance",
    "product_technology": "product_demand",
    "demand_customer_contract": "product_demand",
    "supply_chain_capacity": "supply_capacity",
    "regulation_trade_policy": "regulation_legal",
    "legal_governance_operations": "regulation_legal",
    "analyst_action": "corporate_analyst",
    "corporate_action": "corporate_analyst",
    "macro_market": "macro_market",
    "other": "other_or_unclear",
    "unclear": "other_or_unclear",
}

INFORMATION_STATUS_MAP = {
    "confirmed": "confirmed",
    "scheduled_or_expected": "anticipated",
    "rumor_or_unconfirmed": "rumor_or_opinion",
    "analysis_or_opinion": "rumor_or_opinion",
    "unclear": "unclear",
}

_DIRECTIONAL_VALUES = frozenset({"positive", "negative", "neutral"})
_OPPOSITE_DIRECTION_PAIRS = frozenset(
    {("positive", "negative"), ("negative", "positive")}
)

# These terms are intentionally narrow.  The gate should abstain rather than
# turn generic business language into a macro-relevant article.
MACRO_TEXT_TERMS = (
    "central bank",
    "consumer price index",
    "economic growth",
    "federal funds rate",
    "federal reserve",
    "fomc",
    "gross domestic product",
    "inflation",
    "interest rate",
    "interest rates",
    "jobs report",
    "monetary policy",
    "nonfarm payroll",
    "nonfarm payrolls",
    "recession",
    "treasury yield",
    "treasury yields",
    "unemployment rate",
)

# Conservative aliases for the fixed semiconductor pilot universe.  Ticker
# matching remains available for any future universe; these names merely avoid
# missing obvious prose such as "Nvidia and Intel" when the metadata uses NVDA
# and INTC.  No company relationships are inferred beyond the supplied peer
# list in each record.
COMPANY_ALIASES_BY_TICKER = {
    "AMD": ("Advanced Micro Devices",),
    "NVDA": ("Nvidia", "NVIDIA Corporation"),
    "INTC": ("Intel", "Intel Corporation"),
    "MU": ("Micron", "Micron Technology"),
    "AVGO": ("Broadcom", "Broadcom Inc"),
    "QCOM": ("Qualcomm", "Qualcomm Incorporated"),
}

POSITIVE_SURPRISE_PATTERNS = (
    r"\b(?:beat|beats|beating|topped|exceeded|surpassed)\b.{0,50}\b(?:estimate|estimates|expectation|expectations|consensus)\b",
    r"\b(?:above|ahead of|better than|stronger than|higher than)\b.{0,50}\b(?:estimate|estimates|expectation|expectations|consensus|guidance|forecast|outlook)\b",
    r"\b(?:raised|raises|lifted|increased|upgraded)\b.{0,35}\b(?:guidance|outlook|forecast)\b",
    r"\bupward revision\b",
)

NEGATIVE_SURPRISE_PATTERNS = (
    r"\b(?:missed|misses|fell short of)\b.{0,50}\b(?:estimate|estimates|expectation|expectations|consensus)\b",
    r"\b(?:below|worse than|weaker than|lower than)\b.{0,50}\b(?:estimate|estimates|expectation|expectations|consensus|guidance|forecast|outlook)\b",
    r"\b(?:cut|cuts|lowered|reduced|downgraded)\b.{0,35}\b(?:guidance|outlook|forecast)\b",
    r"\bdownward revision\b",
)

DETERMINISTIC_RULE_VERSION = "coarse-news-gate-v0.1.0"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


_RULE_PAYLOAD = {
    "version": DETERMINISTIC_RULE_VERSION,
    "route_priority": ["direct_target", "sector_or_peer", "macro_relevant", "none"],
    "ticker_text_matching": "case-sensitive ASCII token boundary",
    "company_sector_macro_matching": "case-insensitive literal token/phrase boundary",
    "macro_text_terms": list(MACRO_TEXT_TERMS),
    "company_aliases_by_ticker": COMPANY_ALIASES_BY_TICKER,
    "positive_surprise_patterns": list(POSITIVE_SURPRISE_PATTERNS),
    "negative_surprise_patterns": list(NEGATIVE_SURPRISE_PATTERNS),
}
DETERMINISTIC_RULES_SHA256 = hashlib.sha256(
    _canonical_json(_RULE_PAYLOAD).encode("utf-8")
).hexdigest()


class CoarseFeatureError(ValueError):
    """Raised when an input or label record violates the coarse contract."""


def load_schema(path: Path | str = DEFAULT_SCHEMA_PATH) -> dict[str, Any]:
    schema = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_schema(schema)
    return schema


def validate_schema(schema: Mapping[str, Any]) -> None:
    if schema.get("schema_name") != "stock_sector_news_semantics_coarse":
        raise CoarseFeatureError("Unexpected coarse schema_name")
    closed = schema.get("closed_label_fields")
    if not isinstance(closed, Mapping):
        raise CoarseFeatureError("closed_label_fields must be an object")
    if tuple(closed) != COARSE_FIELDS:
        raise CoarseFeatureError(
            f"closed_label_fields must be ordered exactly as {COARSE_FIELDS!r}"
        )
    for field, values in closed.items():
        if not isinstance(values, list) or not values or not all(
            isinstance(value, str) and value for value in values
        ):
            raise CoarseFeatureError(f"{field} must have a nonempty string label list")
        if len(values) != len(set(values)):
            raise CoarseFeatureError(f"{field} contains duplicate labels")


def _require_fine_value(
    fine_labels: Mapping[str, Any], field: str, allowed: Iterable[str]
) -> str:
    if not isinstance(fine_labels, Mapping):
        raise CoarseFeatureError("fine_labels must be an object")
    value = fine_labels.get(field)
    allowed_set = frozenset(allowed)
    if value not in allowed_set:
        raise CoarseFeatureError(
            f"Invalid or missing fine label {field}={value!r}; "
            f"expected one of {sorted(allowed_set)!r}"
        )
    return str(value)


def semantic_applicable(fine_labels: Mapping[str, Any]) -> bool:
    """Return whether the reference labels permit semantic classification.

    This helper is for mapping/evaluation only.  A production extractor must
    use :func:`deterministic_features`, because reference relevance is not
    available at inference time.
    """

    relevance = _require_fine_value(
        fine_labels,
        "relevance",
        APPLICABLE_RELEVANCE | INAPPLICABLE_RELEVANCE,
    )
    return relevance in APPLICABLE_RELEVANCE


def _map_directional_alignment(
    fine_labels: Mapping[str, Any], coarse_scope: str, applicable: bool
) -> str:
    peer_effect = _require_fine_value(
        fine_labels,
        "peer_effect",
        {
            "same_direction",
            "opposite_direction",
            "mixed",
            "none_stated",
            "unknown",
            "not_applicable",
        },
    )
    target_direction = _require_fine_value(
        fine_labels,
        "target_direction",
        {"positive", "negative", "neutral", "mixed", "unknown", "not_applicable"},
    )
    sector_direction = _require_fine_value(
        fine_labels,
        "sector_direction",
        {"positive", "negative", "neutral", "mixed", "unknown", "not_applicable"},
    )

    if not applicable or coarse_scope == "unclear":
        return "unclear"

    # Scope takes precedence over target/sector direction.  This prevents a
    # firm-specific article with incidental sector-direction text from being
    # relabeled as a common shock.  An explicit peer comparison remains useful.
    if coarse_scope == "idiosyncratic":
        if peer_effect in {"same_direction", "opposite_direction"}:
            return peer_effect
        return "single_firm_only"

    if peer_effect in {"same_direction", "opposite_direction"}:
        return peer_effect
    if target_direction in _DIRECTIONAL_VALUES and sector_direction in _DIRECTIONAL_VALUES:
        if target_direction == sector_direction:
            return "same_direction"
        if (target_direction, sector_direction) in _OPPOSITE_DIRECTION_PAIRS:
            return "opposite_direction"
    return "common_direction_unclear"


def map_fine_labels(fine_labels: Mapping[str, Any]) -> dict[str, str]:
    """Deterministically coarsen one v0.1 GPT-5.6 silver label object."""

    applicable = semantic_applicable(fine_labels)
    fine_scope = _require_fine_value(fine_labels, "event_scope", SHOCK_SCOPE_MAP)
    event_type = _require_fine_value(fine_labels, "event_type", EVENT_FAMILY_MAP)
    fine_status = _require_fine_value(
        fine_labels, "information_status", INFORMATION_STATUS_MAP
    )

    shock_scope = SHOCK_SCOPE_MAP[fine_scope] if applicable else "unclear"
    coarse = {
        "shock_scope": shock_scope,
        "event_family": EVENT_FAMILY_MAP[event_type],
        "information_status": INFORMATION_STATUS_MAP[fine_status],
        "directional_alignment": _map_directional_alignment(
            fine_labels, shock_scope, applicable
        ),
    }
    validate_coarse_labels(coarse)
    return coarse


def validate_coarse_labels(
    labels: Mapping[str, Any], schema: Mapping[str, Any] | None = None
) -> None:
    """Validate a complete four-field coarse label object."""

    if not isinstance(labels, Mapping):
        raise CoarseFeatureError("coarse labels must be an object")
    schema_object = dict(schema) if schema is not None else load_schema()
    validate_schema(schema_object)
    unexpected = set(labels) - set(COARSE_FIELDS)
    missing = set(COARSE_FIELDS) - set(labels)
    if missing or unexpected:
        raise CoarseFeatureError(
            f"coarse label fields mismatch; missing={sorted(missing)!r}, "
            f"unexpected={sorted(unexpected)!r}"
        )
    for field in COARSE_FIELDS:
        allowed = schema_object["closed_label_fields"][field]
        if labels[field] not in allowed:
            raise CoarseFeatureError(
                f"Invalid coarse label {field}={labels[field]!r}; expected {allowed!r}"
            )


def _normalize_ticker(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CoarseFeatureError(f"Ticker must be a nonempty string, received {value!r}")
    ticker = value.strip().upper()
    if not re.fullmatch(r"[A-Z0-9.\-]{1,15}", ticker):
        raise CoarseFeatureError(f"Invalid ticker syntax: {value!r}")
    return ticker


def _literal_phrase_present(text: str, phrase: str, *, case_sensitive: bool) -> bool:
    flags = 0 if case_sensitive else re.IGNORECASE
    escaped = re.escape(phrase.strip())
    if not escaped:
        return False
    return re.search(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", text, flags) is not None


def _ticker_present(text: str, ticker: str) -> bool:
    # Tickers such as MU and AMD are matched case-sensitively to avoid ordinary
    # lower-case words and substrings (for example, "much") becoming entities.
    return _literal_phrase_present(text, ticker, case_sensitive=True)


def explicit_surprise_from_text(text: str) -> tuple[str, bool, bool]:
    """Return a conservative explicit-surprise label and its positive/negative cues."""

    positive = any(re.search(pattern, text, re.IGNORECASE | re.DOTALL) for pattern in POSITIVE_SURPRISE_PATTERNS)
    negative = any(re.search(pattern, text, re.IGNORECASE | re.DOTALL) for pattern in NEGATIVE_SURPRISE_PATTERNS)
    if positive and negative:
        label = "mixed"
    elif positive:
        label = "positive"
    elif negative:
        label = "negative"
    else:
        label = "none"
    return label, positive, negative


def _sector_terms(sector: str) -> tuple[str, ...]:
    normalized = sector.strip()
    if not normalized:
        return ()
    terms = [normalized]
    if normalized.lower().endswith("s") and len(normalized) > 3:
        terms.append(normalized[:-1])
    return tuple(dict.fromkeys(terms))


def validate_input_record(record: Mapping[str, Any]) -> None:
    if not isinstance(record, Mapping):
        raise CoarseFeatureError("input record must be an object")
    for field in ("article_id", "headline", "article_text", "vendor_tickers", "target"):
        if field not in record:
            raise CoarseFeatureError(f"input record is missing {field}")
    if not isinstance(record["article_id"], str) or not record["article_id"].strip():
        raise CoarseFeatureError("article_id must be a nonempty string")
    if not isinstance(record["headline"], str) or not isinstance(record["article_text"], str):
        raise CoarseFeatureError("headline and article_text must be strings")
    if not isinstance(record["vendor_tickers"], list):
        raise CoarseFeatureError("vendor_tickers must be a list")
    target = record["target"]
    if not isinstance(target, Mapping):
        raise CoarseFeatureError("target must be an object")
    for field in ("company", "ticker", "sector", "sector_benchmark", "known_sector_peers"):
        if field not in target:
            raise CoarseFeatureError(f"target is missing {field}")
    if not all(isinstance(target[field], str) and target[field].strip() for field in (
        "company", "ticker", "sector", "sector_benchmark"
    )):
        raise CoarseFeatureError("target company/ticker/sector/benchmark must be nonempty strings")
    if not isinstance(target["known_sector_peers"], list):
        raise CoarseFeatureError("known_sector_peers must be a list")
    _normalize_ticker(target["ticker"])
    _normalize_ticker(target["sector_benchmark"])
    for value in record["vendor_tickers"]:
        _normalize_ticker(value)
    for value in target["known_sector_peers"]:
        _normalize_ticker(value)


def deterministic_features(record: Mapping[str, Any]) -> dict[str, Any]:
    """Build conservative point-in-time metadata/text gate features.

    The returned ``semantic_applicable`` is a production gate decision, not the
    GPT-reference applicability returned by :func:`semantic_applicable`.
    ``gate_route`` is intentionally coarse and should be logged in predictions.
    """

    validate_input_record(record)
    target = record["target"]
    headline = record["headline"].strip()
    article_text = record["article_text"].strip()
    combined_text = "\n".join(part for part in (headline, article_text) if part)

    target_ticker = _normalize_ticker(target["ticker"])
    benchmark = _normalize_ticker(target["sector_benchmark"])
    vendor_tickers = sorted({_normalize_ticker(value) for value in record["vendor_tickers"]})
    peers = sorted({_normalize_ticker(value) for value in target["known_sector_peers"]})
    peer_set = set(peers)

    target_vendor_tagged = target_ticker in vendor_tickers
    peer_vendor_tickers = sorted(set(vendor_tickers) & peer_set)
    benchmark_vendor_tagged = benchmark in vendor_tickers
    target_ticker_in_text = _ticker_present(combined_text, target_ticker)
    target_aliases = tuple(
        dict.fromkeys((target["company"], *COMPANY_ALIASES_BY_TICKER.get(target_ticker, ())))
    )
    target_company_in_text = any(
        _literal_phrase_present(combined_text, alias, case_sensitive=False)
        for alias in target_aliases
    )
    peer_tickers_in_text = [ticker for ticker in peers if _ticker_present(combined_text, ticker)]
    peer_companies_in_text = [
        ticker
        for ticker in peers
        if any(
            _literal_phrase_present(combined_text, alias, case_sensitive=False)
            for alias in COMPANY_ALIASES_BY_TICKER.get(ticker, ())
        )
    ]
    peer_entities_in_text = sorted(set(peer_tickers_in_text) | set(peer_companies_in_text))
    sector_terms_in_text = [
        term
        for term in _sector_terms(target["sector"])
        if _literal_phrase_present(combined_text, term, case_sensitive=False)
    ]
    sector_benchmark_in_text = _ticker_present(combined_text, benchmark)
    macro_terms_in_text = [
        term
        for term in MACRO_TEXT_TERMS
        if _literal_phrase_present(combined_text, term, case_sensitive=False)
    ]
    explicit_surprise_rule, positive_surprise_cue, negative_surprise_cue = (
        explicit_surprise_from_text(combined_text)
    )

    direct_target_evidence = bool(
        target_vendor_tagged or target_ticker_in_text or target_company_in_text
    )
    sector_or_peer_evidence = bool(
        peer_vendor_tickers
        or peer_entities_in_text
        or benchmark_vendor_tagged
        or sector_benchmark_in_text
        or sector_terms_in_text
    )
    macro_evidence = bool(macro_terms_in_text)

    if direct_target_evidence:
        gate_route = "direct_target"
    elif sector_or_peer_evidence:
        gate_route = "sector_or_peer"
    elif macro_evidence:
        gate_route = "macro_relevant"
    else:
        gate_route = "no_semantic_evidence"

    text_target_mentioned = bool(target_ticker_in_text or target_company_in_text)
    if gate_route == "no_semantic_evidence":
        deterministic_relevance_route = "no_relevance_evidence"
    elif (
        (target_vendor_tagged or peer_vendor_tickers or benchmark_vendor_tagged)
        and not text_target_mentioned
        and not peer_entities_in_text
        and not sector_terms_in_text
        and not sector_benchmark_in_text
        and not macro_terms_in_text
    ):
        deterministic_relevance_route = "vendor_only_review"
    elif text_target_mentioned:
        deterministic_relevance_route = "target_text"
    elif peer_entities_in_text or sector_terms_in_text or sector_benchmark_in_text:
        deterministic_relevance_route = "sector_or_peer_text"
    elif macro_terms_in_text:
        deterministic_relevance_route = "macro_text"
    else:
        deterministic_relevance_route = "metadata_evidence"

    return {
        "rule_version": DETERMINISTIC_RULE_VERSION,
        "rules_sha256": DETERMINISTIC_RULES_SHA256,
        "target_ticker": target_ticker,
        "normalized_vendor_tickers": vendor_tickers,
        "normalized_known_peers": peers,
        "target_vendor_tagged": target_vendor_tagged,
        "peer_vendor_tickers": peer_vendor_tickers,
        "benchmark_vendor_tagged": benchmark_vendor_tagged,
        "target_ticker_in_text": target_ticker_in_text,
        "target_company_in_text": target_company_in_text,
        "peer_tickers_in_text": peer_tickers_in_text,
        "peer_companies_in_text": peer_companies_in_text,
        "peer_entities_in_text": peer_entities_in_text,
        "sector_terms_in_text": sector_terms_in_text,
        "sector_benchmark_in_text": sector_benchmark_in_text,
        "macro_terms_in_text": macro_terms_in_text,
        "positive_surprise_cue": positive_surprise_cue,
        "negative_surprise_cue": negative_surprise_cue,
        "explicit_surprise_rule": explicit_surprise_rule,
        "direct_target_evidence": direct_target_evidence,
        "sector_or_peer_evidence": sector_or_peer_evidence,
        "macro_evidence": macro_evidence,
        "semantic_applicable": gate_route != "no_semantic_evidence",
        "gate_route": gate_route,
        "deterministic_relevance_route": deterministic_relevance_route,
        "text_target_mentioned": text_target_mentioned,
        "text_peer_count": len(peer_entities_in_text),
    }


def _record_identity(record: Mapping[str, Any]) -> str:
    article_id = record.get("article_id")
    target_ticker = record.get("target_ticker")
    if target_ticker is None and isinstance(record.get("target"), Mapping):
        target_ticker = record["target"].get("ticker")
    if not isinstance(article_id, str) or not article_id:
        raise CoarseFeatureError("Every split record needs a nonempty article_id")
    if not isinstance(target_ticker, str) or not target_ticker:
        raise CoarseFeatureError("Every split record needs target_ticker or target.ticker")
    return f"{article_id}\0{target_ticker.upper()}"


def _default_stratum(record: Mapping[str, Any], field: str) -> str:
    labels = record.get("labels")
    if not isinstance(labels, Mapping) or labels.get(field) is None:
        raise CoarseFeatureError(f"Split record lacks labels.{field}")
    return str(labels[field])


def stratified_split(
    reference_records: Sequence[Mapping[str, Any]],
    *,
    development_fraction: float = 0.24,
    seed: str = "flan-t5-coarse-v0.3-development",
    stratify_field: str = "shock_scope",
    stratum_getter: Callable[[Mapping[str, Any]], str] | None = None,
) -> dict[str, Any]:
    """Return a stable, chronological-agnostic development/evaluation split.

    Within each stratum, identities are ordered by a seeded SHA-256 digest.
    The split is therefore stable under input reordering and across Python
    processes.  The helper is intended only for the already assembled 300-row
    extraction benchmark; forecasting data must still be split chronologically.
    """

    if not 0.0 < development_fraction < 1.0:
        raise CoarseFeatureError("development_fraction must lie strictly between 0 and 1")
    if not isinstance(seed, str) or not seed:
        raise CoarseFeatureError("seed must be a nonempty string")
    getter = stratum_getter or (lambda record: _default_stratum(record, stratify_field))

    groups: dict[str, list[tuple[str, Mapping[str, Any]]]] = defaultdict(list)
    seen_identities: set[str] = set()
    for record in reference_records:
        identity = _record_identity(record)
        if identity in seen_identities:
            raise CoarseFeatureError(f"Duplicate split identity: {identity!r}")
        seen_identities.add(identity)
        stratum = getter(record)
        if not isinstance(stratum, str) or not stratum:
            raise CoarseFeatureError(f"Invalid split stratum for {identity!r}: {stratum!r}")
        digest = hashlib.sha256(f"{seed}\0{stratum}\0{identity}".encode("utf-8")).hexdigest()
        groups[stratum].append((digest, record))

    assignments: dict[str, str] = {}
    development: list[Mapping[str, Any]] = []
    evaluation: list[Mapping[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}
    for stratum in sorted(groups):
        ordered = sorted(groups[stratum], key=lambda item: (item[0], _record_identity(item[1])))
        size = len(ordered)
        development_size = int(math.floor(size * development_fraction + 0.5))
        if size >= 2:
            development_size = min(max(development_size, 1), size - 1)
        else:
            development_size = size
        development_identities = {
            _record_identity(record) for _, record in ordered[:development_size]
        }
        counts[stratum] = {
            "total": size,
            "development": development_size,
            "evaluation": size - development_size,
        }
        for _, record in ordered:
            identity = _record_identity(record)
            split = "development" if identity in development_identities else "evaluation"
            assignments[identity] = split
            (development if split == "development" else evaluation).append(record)

    # Sort outputs by stable identity so serialized files do not depend on input order.
    development.sort(key=_record_identity)
    evaluation.sort(key=_record_identity)
    return {
        "development": development,
        "evaluation": evaluation,
        "assignments": dict(sorted(assignments.items())),
        "counts_by_stratum": counts,
        "development_fraction": development_fraction,
        "seed": seed,
        "stratify_field": stratify_field,
    }


def coarse_distribution(records: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """Return deterministic per-field counts for diagnostics and tests."""

    counts = {field: Counter() for field in COARSE_FIELDS}
    for record in records:
        labels = record.get("labels", record)
        validate_coarse_labels(labels)
        for field in COARSE_FIELDS:
            counts[field][labels[field]] += 1
    return {
        field: dict(sorted(field_counts.items()))
        for field, field_counts in counts.items()
    }


__all__ = [
    "APPLICABLE_RELEVANCE",
    "COARSE_FIELDS",
    "CoarseFeatureError",
    "DETERMINISTIC_RULE_VERSION",
    "DETERMINISTIC_RULES_SHA256",
    "coarse_distribution",
    "deterministic_features",
    "explicit_surprise_from_text",
    "load_schema",
    "map_fine_labels",
    "semantic_applicable",
    "stratified_split",
    "validate_coarse_labels",
    "validate_input_record",
    "validate_schema",
]
