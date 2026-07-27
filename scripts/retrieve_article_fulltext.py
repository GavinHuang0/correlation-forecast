"""Retrieve public article text while respecting access controls.

This command is intentionally conservative:

* only public HTTP(S) URLs are requested;
* every article URL, including redirect destinations, is checked against that
  origin's ``robots.txt`` before it is fetched;
* login, subscription, and paywall pages are rejected rather than worked
  around;
* response size, request time, redirects, and per-domain request frequency are
  bounded; and
* only extracted UTF-8 text and a sanitized JSONL audit trail are retained.

The command has no non-stdlib dependencies.  Its output location is fixed at
``data/external/news_provider_comparison/v1_0/fulltext`` so raw extracted text
cannot accidentally be written to a tracked experiment directory.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import ipaddress
import json
import os
import re
import socket
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = (
    REPOSITORY_ROOT
    / "data"
    / "external"
    / "news_provider_comparison"
    / "v1_0"
    / "fulltext"
)
MANIFEST_NAME = "manifest.jsonl"
MATERIALIZED_RECORDS_NAME = "records.jsonl"
TEXT_DIRECTORY_NAME = "articles"
USER_AGENT = (
    "CorrelationForecastResearchBot/1.0 "
    "(public-news research; respects robots and access controls)"
)

DEFAULT_MIN_CHARS = 500
DEFAULT_MAX_BYTES = 2_000_000
DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_DOMAIN_INTERVAL_SECONDS = 1.0
DEFAULT_MAX_REDIRECTS = 5
MAX_ROBOTS_BYTES = 512_000

REDIRECT_STATUSES = {301, 302, 303, 307, 308}
HTML_MEDIA_TYPES = {"text/html", "application/xhtml+xml"}
RECORD_CONTAINER_KEYS = (
    "articles",
    "results",
    "items",
    "feed",
    "news",
    "data",
)
ID_FIELDS = ("article_id", "provider_article_id", "id", "uuid")
URL_FIELDS = ("url", "article_url", "link", "web_url")
TITLE_FIELDS = ("title", "headline", "name")
PUBLISHED_FIELDS = (
    "published",
    "published_at",
    "published_at_utc",
    "published_utc",
    "time_published",
    "published_time",
    "date_published",
    "datetime",
    "created_at",
    "date",
)
SENSITIVE_QUERY_KEYS = {
    "access_token",
    "api_key",
    "apikey",
    "auth",
    "authorization",
    "credential",
    "key",
    "password",
    "signature",
    "signed",
    "token",
    "x_amz_credential",
    "x_amz_security_token",
    "x_amz_signature",
    "x_goog_credential",
    "x_goog_signature",
}
PERMANENT_REASONS = {
    "access_denied",
    "below_min_chars",
    "http_not_found",
    "max_redirects",
    "no_article_text",
    "non_html",
    "paywall_detected",
    "redirect_without_location",
    "robots_denied",
    "sensitive_query",
    "unsupported_url",
}

_BOILERPLATE_TOKEN = re.compile(
    r"(?:^|[\s_-])(?:"
    r"ad|ads|advert|advertisement|aside|author-bio|banner|breadcrumb|caption|"
    r"cookie|footer|header|menu|nav|newsletter|paywall|promo|recommend|"
    r"related|share|social|sponsor|subscribe|toolbar"
    r")(?:$|[\s_-])",
    re.IGNORECASE,
)
_PAYWALL_ATTRIBUTE = re.compile(
    r"""(?:class|id|data-[\w:-]+)\s*=\s*["'][^"']*
        (?:paywall|subscription[-_\s]?wall|subscriber[-_\s]?only|
        premium[-_\s]?content|metered[-_\s]?content)[^"']*["']""",
    re.IGNORECASE | re.VERBOSE,
)
_NOT_ACCESSIBLE_JSON_LD = re.compile(
    r"""["']isAccessibleForFree["']\s*:\s*(?:false|["']false["'])""",
    re.IGNORECASE,
)
_PAYWALL_PHRASES = (
    "already a subscriber",
    "log in to continue",
    "register to continue",
    "sign in to continue",
    "subscribe to continue",
    "subscription required",
    "this article is for subscribers",
    "this content is for subscribers",
    "unlock this article",
    "you have reached your article limit",
)
_IGNORED_ELEMENTS = {
    "aside",
    "button",
    "dialog",
    "footer",
    "form",
    "header",
    "iframe",
    "nav",
    "noscript",
    "script",
    "style",
    "svg",
    "template",
}


class RetrievalFailure(Exception):
    """A categorized failure safe to include in the audit manifest."""

    def __init__(self, reason: str, *, http_status: int | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.http_status = http_status


@dataclass(frozen=True)
class ArticleRecord:
    article_id: str
    url: str
    title: str
    published: str


@dataclass(frozen=True)
class HttpResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes
    final_url: str


@dataclass(frozen=True)
class ExtractionResult:
    text: str
    source_method: str


@dataclass(frozen=True)
class RetrievalConfig:
    min_chars: int = DEFAULT_MIN_CHARS
    max_bytes: int = DEFAULT_MAX_BYTES
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    domain_interval_seconds: float = DEFAULT_DOMAIN_INTERVAL_SECONDS
    max_redirects: int = DEFAULT_MAX_REDIRECTS

    def validate(self) -> None:
        if self.min_chars < 1:
            raise ValueError("min_chars must be at least 1")
        if self.max_bytes < 1:
            raise ValueError("max_bytes must be at least 1")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.domain_interval_seconds < 0:
            raise ValueError("domain_interval_seconds cannot be negative")
        if self.max_redirects < 0:
            raise ValueError("max_redirects cannot be negative")


@dataclass
class RetrievalOutcome:
    status: str
    reason: str
    http_status: int | None = None
    response_bytes: int | None = None
    content_length: int | None = None
    content_sha256: str | None = None
    source_method: str | None = None
    final_domain: str | None = None
    text: str | None = field(default=None, repr=False)


@dataclass(frozen=True)
class RobotsDecision:
    allowed: bool
    reason: str
    interval_seconds: float = 0.0


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    """Hash text deterministically as UTF-8 without platform newline changes."""

    return sha256_bytes(value.encode("utf-8"))


def article_text_relative_path(article_id: str) -> Path:
    return Path(TEXT_DIRECTORY_NAME) / f"{sha256_text(article_id)}.txt"


def _scalar_text(value: Any) -> str | None:
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return None


def _first_field(record: Mapping[str, Any], names: Sequence[str]) -> str | None:
    for name in names:
        value = _scalar_text(record.get(name))
        if value is not None:
            return value
    return None


def normalize_record(record: Mapping[str, Any], *, source: str = "record") -> ArticleRecord:
    url = _first_field(record, URL_FIELDS)
    title = _first_field(record, TITLE_FIELDS)
    published = _first_field(record, PUBLISHED_FIELDS)
    article_id = _first_field(record, ID_FIELDS)

    missing = [
        name
        for name, value in (("url", url), ("title", title), ("published", published))
        if value is None
    ]
    if missing:
        raise ValueError(f"{source} is missing required field(s): {', '.join(missing)}")
    assert url is not None and title is not None and published is not None
    if article_id is None:
        # Some provider feeds expose no ID.  A URL-derived ID is stable without
        # putting the potentially sensitive URL in the output filename.
        article_id = f"url_{sha256_text(url)[:24]}"
    return ArticleRecord(
        article_id=article_id,
        url=url,
        title=title,
        published=published,
    )


def _looks_like_record(value: Mapping[str, Any]) -> bool:
    return any(name in value for name in URL_FIELDS) and any(
        name in value for name in TITLE_FIELDS
    )


def _records_from_json(value: Any, *, source: str) -> Iterator[Mapping[str, Any]]:
    if isinstance(value, list):
        for index, item in enumerate(value, start=1):
            if not isinstance(item, Mapping):
                raise ValueError(f"{source} item {index} is not a JSON object")
            yield item
        return
    if not isinstance(value, Mapping):
        raise ValueError(f"{source} must contain a JSON object or array")
    if _looks_like_record(value):
        yield value
        return

    found_container = False
    for key in RECORD_CONTAINER_KEYS:
        nested = value.get(key)
        if isinstance(nested, list):
            found_container = True
            yield from _records_from_json(nested, source=f"{source}.{key}")
        elif isinstance(nested, Mapping):
            try:
                nested_records = list(
                    _records_from_json(nested, source=f"{source}.{key}")
                )
            except ValueError:
                continue
            if nested_records:
                found_container = True
                yield from nested_records
    if not found_container:
        raise ValueError(f"{source} has no article record or recognized record array")


def _logical_suffix(path: Path) -> str:
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if suffixes and suffixes[-1] == ".gz":
        suffixes.pop()
    return suffixes[-1] if suffixes else ""


def iter_input_records(path: Path) -> Iterator[Mapping[str, Any]]:
    """Yield objects from JSON, JSONL/NDJSON, and gzip-compressed variants."""

    is_gzip = path.suffix.lower() == ".gz"
    opener: Callable[..., Any] = gzip.open if is_gzip else open
    suffix = _logical_suffix(path)
    with opener(path, "rt", encoding="utf-8") as handle:
        if suffix in {".jsonl", ".ndjson"}:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path} line {line_number} is invalid JSON"
                    ) from exc
                if not isinstance(value, Mapping):
                    raise ValueError(
                        f"{path} line {line_number} is not a JSON object"
                    )
                yield value
            return
        if suffix not in {"", ".json"}:
            raise ValueError(
                f"{path} must be .json, .jsonl, .ndjson, or a gzip variant"
            )
        try:
            value = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} is invalid JSON") from exc
        yield from _records_from_json(value, source=str(path))


def iter_articles(paths: Sequence[Path]) -> Iterator[ArticleRecord]:
    for path in paths:
        for index, record in enumerate(iter_input_records(path), start=1):
            yield normalize_record(record, source=f"{path} record {index}")


def _default_resolver(host: str, port: int) -> Iterable[str]:
    return {
        address[4][0]
        for address in socket.getaddrinfo(
            host, port, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM
        )
    }


Resolver = Callable[[str, int], Iterable[str]]


def public_url_rejection_reason(
    url: str, *, resolver: Resolver = _default_resolver
) -> str | None:
    """Return a safe rejection category, or ``None`` for a public HTTP(S) URL."""

    try:
        parsed = urllib.parse.urlsplit(url)
        port = parsed.port
    except (TypeError, ValueError):
        return "unsupported_url"
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        return "unsupported_url"
    if parsed.username is not None or parsed.password is not None:
        return "unsupported_url"
    query_keys = [
        re.sub(r"[^a-z0-9]+", "_", key.casefold()).strip("_")
        for key, _ in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    ]
    if any(
        key in SENSITIVE_QUERY_KEYS
        or any(
            token in key.split("_")
            for token in ("credential", "password", "secret", "signature", "token")
        )
        for key in query_keys
    ):
        return "sensitive_query"

    host = parsed.hostname.rstrip(".").casefold()
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
        return "private_host"
    try:
        ascii_host = host.encode("idna").decode("ascii")
    except UnicodeError:
        return "unsupported_url"
    effective_port = port or (443 if parsed.scheme.lower() == "https" else 80)

    try:
        literal = ipaddress.ip_address(ascii_host)
    except ValueError:
        try:
            addresses = list(resolver(ascii_host, effective_port))
        except (OSError, socket.gaierror):
            return "dns_error"
        if not addresses:
            return "dns_error"
        for address in addresses:
            try:
                if not ipaddress.ip_address(address).is_global:
                    return "private_host"
            except ValueError:
                return "dns_error"
    else:
        if not literal.is_global:
            return "private_host"
    return None


def _domain_for_url(url: str) -> str | None:
    try:
        host = urllib.parse.urlsplit(url).hostname
    except ValueError:
        return None
    return host.rstrip(".").casefold() if host else None


class DomainRateLimiter:
    def __init__(
        self,
        interval_seconds: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.interval_seconds = interval_seconds
        self.clock = clock
        self.sleeper = sleeper
        self._last_request: dict[str, float] = {}

    def wait(self, domain: str, *, minimum_interval: float = 0.0) -> None:
        interval = max(self.interval_seconds, minimum_interval)
        previous = self._last_request.get(domain)
        if previous is not None:
            remaining = interval - (self.clock() - previous)
            if remaining > 0:
                self.sleeper(remaining)
        self._last_request[domain] = self.clock()


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


_DEFAULT_OPENER = urllib.request.build_opener(_NoRedirectHandler())


def _headers_dict(headers: Any) -> dict[str, str]:
    return {str(key).lower(): str(value) for key, value in headers.items()}


def _bounded_response_body(
    response: Any,
    *,
    max_bytes: int,
    timeout_seconds: float,
    http_status: int,
) -> bytes:
    headers = _headers_dict(response.headers)
    declared = headers.get("content-length")
    if declared:
        try:
            if int(declared) > max_bytes:
                raise RetrievalFailure(
                    "response_too_large", http_status=http_status
                )
        except ValueError:
            pass
    started = time.monotonic()
    body = response.read(max_bytes + 1)
    if time.monotonic() - started > timeout_seconds:
        raise RetrievalFailure("timeout", http_status=http_status)
    if len(body) > max_bytes:
        raise RetrievalFailure("response_too_large", http_status=http_status)
    return body


def default_fetcher(
    url: str,
    *,
    timeout: float,
    max_bytes: int,
    user_agent: str,
) -> HttpResponse:
    """Fetch one URL without following redirects."""

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.1",
            "Accept-Encoding": "identity",
        },
        method="GET",
    )
    try:
        response = _DEFAULT_OPENER.open(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        # With redirects disabled, urllib exposes 3xx responses as HTTPError.
        try:
            status = int(exc.code)
            body = (
                b""
                if status in REDIRECT_STATUSES
                else _bounded_response_body(
                    exc,
                    max_bytes=max_bytes,
                    timeout_seconds=timeout,
                    http_status=status,
                )
            )
            return HttpResponse(
                status=status,
                headers=_headers_dict(exc.headers),
                body=body,
                final_url=exc.geturl(),
            )
        finally:
            exc.close()
    except (TimeoutError, socket.timeout) as exc:
        raise RetrievalFailure("timeout") from exc
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, (TimeoutError, socket.timeout)):
            raise RetrievalFailure("timeout") from exc
        raise RetrievalFailure("network_error") from exc
    try:
        status = int(getattr(response, "status", response.getcode()))
        body = _bounded_response_body(
            response,
            max_bytes=max_bytes,
            timeout_seconds=timeout,
            http_status=status,
        )
        return HttpResponse(
            status=status,
            headers=_headers_dict(response.headers),
            body=body,
            final_url=response.geturl(),
        )
    finally:
        response.close()


Fetcher = Callable[..., HttpResponse]


def robots_allows(
    url: str, robots_text: str, *, user_agent: str = USER_AGENT
) -> bool:
    parser = urllib.robotparser.RobotFileParser()
    parsed = urllib.parse.urlsplit(url)
    parser.set_url(
        urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "/robots.txt", "", ""))
    )
    parser.parse(robots_text.splitlines())
    return parser.can_fetch(user_agent, url)


class RobotsCache:
    def __init__(
        self,
        raw_fetch: Callable[..., HttpResponse],
        *,
        user_agent: str,
        max_redirects: int,
    ) -> None:
        self.raw_fetch = raw_fetch
        self.user_agent = user_agent
        self.max_redirects = max_redirects
        self._cache: dict[
            str, tuple[urllib.robotparser.RobotFileParser | None, str]
        ] = {}

    @staticmethod
    def _origin(url: str) -> str:
        parsed = urllib.parse.urlsplit(url)
        return urllib.parse.urlunsplit(
            (parsed.scheme.lower(), parsed.netloc.lower(), "", "", "")
        )

    def _load(
        self, origin: str
    ) -> tuple[urllib.robotparser.RobotFileParser | None, str]:
        robots_url = origin + "/robots.txt"
        current = robots_url
        seen: set[str] = set()
        try:
            for _ in range(self.max_redirects + 1):
                if current in seen:
                    return None, "robots_unavailable"
                seen.add(current)
                response = self.raw_fetch(current, max_bytes=MAX_ROBOTS_BYTES)
                if response.status in REDIRECT_STATUSES:
                    location = response.headers.get("location")
                    if not location:
                        return None, "robots_unavailable"
                    current = urllib.parse.urljoin(current, location)
                    continue
                parser = urllib.robotparser.RobotFileParser()
                parser.set_url(robots_url)
                if response.status in {404, 410}:
                    parser.parse(["User-agent: *", "Allow: /"])
                    return parser, "ok"
                if not 200 <= response.status < 300:
                    return None, "robots_unavailable"
                parser.parse(response.body.decode("utf-8", errors="replace").splitlines())
                return parser, "ok"
        except RetrievalFailure:
            return None, "robots_unavailable"
        return None, "robots_unavailable"

    def check(self, url: str) -> RobotsDecision:
        origin = self._origin(url)
        if origin not in self._cache:
            self._cache[origin] = self._load(origin)
        parser, reason = self._cache[origin]
        if parser is None:
            return RobotsDecision(False, reason)
        if not parser.can_fetch(self.user_agent, url):
            return RobotsDecision(False, "robots_denied")
        delay = parser.crawl_delay(self.user_agent)
        if delay is None:
            delay = parser.crawl_delay("*")
        interval = float(delay or 0.0)
        request_rate = parser.request_rate(self.user_agent) or parser.request_rate("*")
        if request_rate and request_rate.requests > 0:
            interval = max(interval, request_rate.seconds / request_rate.requests)
        return RobotsDecision(True, "ok", interval)


@dataclass
class _Paragraph:
    chunks: list[str]
    in_article: bool
    in_main: bool


@dataclass
class _Element:
    tag: str
    blocked: bool
    in_article: bool
    in_main: bool
    paragraph: _Paragraph | None = None
    json_ld: list[str] | None = None


def _attrs_dict(attrs: Sequence[tuple[str, str | None]]) -> dict[str, str]:
    return {
        name.casefold(): (value or "")
        for name, value in attrs
        if isinstance(name, str)
    }


def _attrs_are_boilerplate(attrs: Mapping[str, str]) -> bool:
    if "hidden" in attrs or attrs.get("aria-hidden", "").casefold() == "true":
        return True
    style = attrs.get("style", "").replace(" ", "").casefold()
    if "display:none" in style or "visibility:hidden" in style:
        return True
    role = attrs.get("role", "").casefold()
    if role in {"banner", "complementary", "contentinfo", "navigation"}:
        return True
    tokens = " ".join((attrs.get("id", ""), attrs.get("class", "")))
    return bool(_BOILERPLATE_TOKEN.search(tokens))


def _normalize_paragraph(value: str) -> str:
    return " ".join(html.unescape(value).split())


def _deduplicate_paragraphs(paragraphs: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for paragraph in paragraphs:
        normalized = _normalize_paragraph(paragraph)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


class _ArticleHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[_Element] = []
        self.article_paragraphs: list[str] = []
        self.main_paragraphs: list[str] = []
        self.json_ld_blocks: list[str] = []
        self.page_text: list[str] = []
        self.has_password_input = False

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        tag = tag.casefold()
        attributes = _attrs_dict(attrs)
        parent = self.stack[-1] if self.stack else None
        blocked = (
            bool(parent and parent.blocked)
            or tag in _IGNORED_ELEMENTS
            or _attrs_are_boilerplate(attributes)
        )
        in_article = bool(parent and parent.in_article) or tag == "article"
        in_main = bool(parent and parent.in_main) or tag == "main"
        paragraph = (
            _Paragraph([], in_article, in_main)
            if tag == "p" and not blocked and (in_article or in_main)
            else None
        )
        json_ld = (
            []
            if tag == "script"
            and attributes.get("type", "").casefold().split(";", 1)[0].strip()
            == "application/ld+json"
            else None
        )
        if tag == "input" and attributes.get("type", "").casefold() == "password":
            self.has_password_input = True
        self.stack.append(
            _Element(
                tag=tag,
                blocked=blocked,
                in_article=in_article,
                in_main=in_main,
                paragraph=paragraph,
                json_ld=json_ld,
            )
        )
        if tag == "br":
            active = self._active_paragraph()
            if active is not None and not blocked:
                active.chunks.append(" ")

    def handle_startendtag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def _active_paragraph(self) -> _Paragraph | None:
        for element in reversed(self.stack):
            if element.paragraph is not None:
                return element.paragraph
        return None

    def handle_data(self, data: str) -> None:
        if not self.stack:
            if data.strip():
                self.page_text.append(data)
            return
        for element in reversed(self.stack):
            if element.json_ld is not None:
                element.json_ld.append(data)
                return
        if self.stack[-1].blocked:
            return
        if data.strip():
            self.page_text.append(data)
        paragraph = self._active_paragraph()
        if paragraph is not None:
            paragraph.chunks.append(data)

    def _finish(self, element: _Element) -> None:
        if element.json_ld is not None:
            value = "".join(element.json_ld).strip()
            if value:
                self.json_ld_blocks.append(value)
        if element.paragraph is not None:
            value = _normalize_paragraph("".join(element.paragraph.chunks))
            if not value:
                return
            if element.paragraph.in_article:
                self.article_paragraphs.append(value)
            if element.paragraph.in_main:
                self.main_paragraphs.append(value)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        match = next(
            (
                index
                for index in range(len(self.stack) - 1, -1, -1)
                if self.stack[index].tag == tag
            ),
            None,
        )
        if match is None:
            return
        for element in reversed(self.stack[match:]):
            self._finish(element)
        del self.stack[match:]

    def close(self) -> None:
        super().close()
        for element in reversed(self.stack):
            self._finish(element)
        self.stack.clear()


class _FragmentTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag.casefold() in _IGNORED_ELEMENTS:
            self.skip_depth += 1
        elif not self.skip_depth and tag.casefold() in {"br", "div", "li", "p"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() in _IGNORED_ELEMENTS and self.skip_depth:
            self.skip_depth -= 1
        elif not self.skip_depth and tag.casefold() in {"div", "li", "p"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.skip_depth:
            self.parts.append(data)

    def text(self) -> str:
        paragraphs = _deduplicate_paragraphs("".join(self.parts).splitlines())
        return "\n\n".join(paragraphs)


def _clean_json_ld_body(value: str) -> str:
    parser = _FragmentTextParser()
    parser.feed(value)
    parser.close()
    return parser.text()


def _json_ld_article_bodies(value: Any) -> Iterator[str]:
    if isinstance(value, Mapping):
        body = value.get("articleBody")
        if isinstance(body, str) and body.strip():
            yield body
        elif isinstance(body, list) and all(isinstance(item, str) for item in body):
            joined = "\n\n".join(item for item in body if item.strip())
            if joined:
                yield joined
        for nested in value.values():
            if isinstance(nested, (Mapping, list)):
                yield from _json_ld_article_bodies(nested)
    elif isinstance(value, list):
        for item in value:
            yield from _json_ld_article_bodies(item)


def _parse_json_ld_block(block: str) -> Any | None:
    cleaned = block.strip()
    if cleaned.startswith("<!--"):
        cleaned = cleaned[4:]
    if cleaned.endswith("-->"):
        cleaned = cleaned[:-3]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def extract_article_text(document: str) -> ExtractionResult:
    """Extract the best JSON-LD or visible primary-content text candidate."""

    parser = _ArticleHTMLParser()
    parser.feed(document)
    parser.close()
    candidates: list[tuple[str, str, int]] = []

    for block in parser.json_ld_blocks:
        parsed = _parse_json_ld_block(block)
        if parsed is None:
            continue
        for body in _json_ld_article_bodies(parsed):
            cleaned = _clean_json_ld_body(body)
            if cleaned:
                candidates.append((cleaned, "json_ld_articleBody", 2))

    article_paragraphs = _deduplicate_paragraphs(parser.article_paragraphs)
    if article_paragraphs:
        candidates.append(
            ("\n\n".join(article_paragraphs), "article_paragraphs", 1)
        )
    main_paragraphs = _deduplicate_paragraphs(parser.main_paragraphs)
    if main_paragraphs:
        candidates.append(("\n\n".join(main_paragraphs), "main_paragraphs", 0))

    if not candidates:
        return ExtractionResult("", "none")
    text, method, _ = max(candidates, key=lambda candidate: (len(candidate[0]), candidate[2]))
    return ExtractionResult(text, method)


def has_paywall_marker(document: str) -> bool:
    """Conservatively detect pages that require payment or authentication."""

    if _PAYWALL_ATTRIBUTE.search(document) or _NOT_ACCESSIBLE_JSON_LD.search(
        document
    ):
        return True
    parser = _ArticleHTMLParser()
    parser.feed(document)
    parser.close()
    visible = " ".join(" ".join(parser.page_text).casefold().split())
    if any(phrase in visible for phrase in _PAYWALL_PHRASES):
        return True
    return parser.has_password_input and (
        "sign in" in visible or "log in" in visible or "login" in visible
    )


def _decode_html(response: HttpResponse) -> str:
    content_type = response.headers.get("content-type", "")
    match = re.search(r"charset\s*=\s*[\"']?([^;\"'\s]+)", content_type, re.I)
    charset = match.group(1) if match else "utf-8"
    try:
        return response.body.decode(charset, errors="replace")
    except LookupError:
        return response.body.decode("utf-8", errors="replace")


class ArticleRetriever:
    def __init__(
        self,
        config: RetrievalConfig,
        *,
        fetcher: Fetcher = default_fetcher,
        resolver: Resolver = _default_resolver,
        limiter: DomainRateLimiter | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.fetcher = fetcher
        self.resolver = resolver
        self.limiter = limiter or DomainRateLimiter(
            config.domain_interval_seconds
        )
        self.robots = RobotsCache(
            self._raw_fetch,
            user_agent=USER_AGENT,
            max_redirects=config.max_redirects,
        )

    def _raw_fetch(
        self,
        url: str,
        *,
        max_bytes: int,
        minimum_interval: float = 0.0,
    ) -> HttpResponse:
        rejection = public_url_rejection_reason(url, resolver=self.resolver)
        if rejection is not None:
            raise RetrievalFailure(rejection)
        domain = _domain_for_url(url)
        assert domain is not None
        self.limiter.wait(domain, minimum_interval=minimum_interval)
        return self.fetcher(
            url,
            timeout=self.config.timeout_seconds,
            max_bytes=max_bytes,
            user_agent=USER_AGENT,
        )

    def retrieve(self, record: ArticleRecord) -> RetrievalOutcome:
        current = record.url
        seen: set[str] = set()
        response: HttpResponse | None = None
        try:
            for _ in range(self.config.max_redirects + 1):
                if current in seen:
                    return RetrievalOutcome(
                        "rejected", "max_redirects", final_domain=_domain_for_url(current)
                    )
                seen.add(current)
                rejection = public_url_rejection_reason(
                    current, resolver=self.resolver
                )
                if rejection is not None:
                    status = "error" if rejection == "dns_error" else "rejected"
                    return RetrievalOutcome(
                        status, rejection, final_domain=_domain_for_url(current)
                    )

                decision = self.robots.check(current)
                if not decision.allowed:
                    status = (
                        "rejected"
                        if decision.reason == "robots_denied"
                        else "error"
                    )
                    return RetrievalOutcome(
                        status,
                        decision.reason,
                        final_domain=_domain_for_url(current),
                    )
                response = self._raw_fetch(
                    current,
                    max_bytes=self.config.max_bytes,
                    minimum_interval=decision.interval_seconds,
                )
                if response.status not in REDIRECT_STATUSES:
                    break
                location = response.headers.get("location")
                if not location:
                    return RetrievalOutcome(
                        "rejected",
                        "redirect_without_location",
                        http_status=response.status,
                        response_bytes=len(response.body),
                        final_domain=_domain_for_url(current),
                    )
                current = urllib.parse.urljoin(current, location)
            else:
                return RetrievalOutcome(
                    "rejected",
                    "max_redirects",
                    http_status=response.status if response else None,
                    final_domain=_domain_for_url(current),
                )
        except RetrievalFailure as exc:
            status = (
                "rejected"
                if exc.reason
                in {
                    "private_host",
                    "sensitive_query",
                    "unsupported_url",
                }
                else "error"
            )
            return RetrievalOutcome(
                status,
                exc.reason,
                http_status=exc.http_status,
                final_domain=_domain_for_url(current),
            )

        assert response is not None
        outcome_base = {
            "http_status": response.status,
            "response_bytes": len(response.body),
            "final_domain": _domain_for_url(current),
        }
        if response.status in {401, 402, 403}:
            return RetrievalOutcome("rejected", "access_denied", **outcome_base)
        if response.status in {404, 410}:
            return RetrievalOutcome("rejected", "http_not_found", **outcome_base)
        if not 200 <= response.status < 300:
            return RetrievalOutcome("error", "http_error", **outcome_base)

        media_type = response.headers.get("content-type", "").split(";", 1)[0]
        media_type = media_type.strip().casefold()
        if media_type and media_type not in HTML_MEDIA_TYPES:
            return RetrievalOutcome("rejected", "non_html", **outcome_base)
        document = _decode_html(response)
        if has_paywall_marker(document):
            return RetrievalOutcome("rejected", "paywall_detected", **outcome_base)
        extracted = extract_article_text(document)
        if not extracted.text:
            return RetrievalOutcome(
                "rejected",
                "no_article_text",
                source_method=extracted.source_method,
                **outcome_base,
            )
        content = extracted.text.encode("utf-8")
        if len(extracted.text) < self.config.min_chars:
            return RetrievalOutcome(
                "rejected",
                "below_min_chars",
                content_length=len(content),
                content_sha256=sha256_bytes(content),
                source_method=extracted.source_method,
                **outcome_base,
            )
        return RetrievalOutcome(
            "retrieved",
            "ok",
            content_length=len(content),
            content_sha256=sha256_bytes(content),
            source_method=extracted.source_method,
            text=extracted.text,
            **outcome_base,
        )


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=path.name + ".",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _materialized_record(
    record: ArticleRecord,
    *,
    text: str,
    content_sha256: str,
    source_method: str,
) -> dict[str, Any]:
    """Return the private, builder-compatible representation of one success."""

    return {
        "article_id": record.article_id,
        "article_url": record.url,
        "title": record.title,
        "published": record.published,
        "full_text": text,
        "text_file": article_text_relative_path(record.article_id).as_posix(),
        "content_sha256": content_sha256,
        "source_method": source_method,
    }


def _load_materialized_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path} line {line_number} is invalid JSON"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path} line {line_number} is not an object")
            article_id = record.get("article_id")
            if not isinstance(article_id, str) or not article_id:
                raise ValueError(
                    f"{path} line {line_number} has no valid article_id"
                )
            records[article_id] = record
    return records


def _write_materialized_records_atomic(
    path: Path, records: Mapping[str, Mapping[str, Any]]
) -> None:
    lines = [
        json.dumps(
            records[article_id],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for article_id in sorted(records)
    ]
    _write_text_atomic(path, "".join(line + "\n" for line in lines))


def _append_audit(path: Path, entry: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        entry, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _repair_trailing_audit_fragment(path: Path) -> bool:
    """Drop an interrupted final JSONL fragment before the next append."""

    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open("r+b") as handle:
        handle.seek(-1, os.SEEK_END)
        if handle.read(1) == b"\n":
            return False
        handle.seek(0)
        data = handle.read()
        final_newline = data.rfind(b"\n")
        handle.seek(0)
        handle.truncate(final_newline + 1 if final_newline >= 0 else 0)
        handle.flush()
        os.fsync(handle.fileno())
    return True


def _latest_audit_entries(path: Path) -> dict[str, Mapping[str, Any]]:
    if not path.exists():
        return {}
    latest: dict[str, Mapping[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                # A process can be interrupted during its last append.  Earlier
                # complete entries remain usable for resume.
                continue
            article_id = entry.get("article_id")
            if isinstance(article_id, str):
                latest[article_id] = entry
    return latest


def _resume_status(
    record: ArticleRecord,
    entry: Mapping[str, Any] | None,
    *,
    output_root: Path,
    config: RetrievalConfig,
) -> str | None:
    if not entry or entry.get("url_sha256") != sha256_text(record.url):
        return None
    status = entry.get("status")
    if status == "retrieved":
        expected_hash = entry.get("content_sha256")
        text_path = output_root / article_text_relative_path(record.article_id)
        if (
            isinstance(expected_hash, str)
            and text_path.is_file()
            and sha256_bytes(text_path.read_bytes()) == expected_hash
        ):
            return "retrieved"
        return None
    reason = entry.get("reason")
    if status == "rejected" and reason in PERMANENT_REASONS:
        if reason == "below_min_chars" and entry.get("min_chars") != config.min_chars:
            return None
        return "rejected"
    return None


def _audit_entry(
    record: ArticleRecord,
    outcome: RetrievalOutcome,
    config: RetrievalConfig,
) -> dict[str, Any]:
    # Deliberately omit the raw URL, title, headers, exceptions, and response
    # body.  Query strings can contain credentials even when input handling
    # rejects common credential names.
    return {
        "schema_version": "1.0",
        "recorded_at_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "article_id": record.article_id,
        "published": record.published,
        "domain": outcome.final_domain or _domain_for_url(record.url),
        "url_sha256": sha256_text(record.url),
        "status": outcome.status,
        "reason": outcome.reason,
        "http_status": outcome.http_status,
        "response_bytes": outcome.response_bytes,
        "content_length": outcome.content_length,
        "content_sha256": outcome.content_sha256,
        "source_method": outcome.source_method,
        "text_file": (
            article_text_relative_path(record.article_id).as_posix()
            if outcome.status == "retrieved"
            else None
        ),
        "min_chars": config.min_chars,
        "max_bytes": config.max_bytes,
        "timeout_seconds": config.timeout_seconds,
        "user_agent": USER_AGENT,
    }


def run(
    input_paths: Sequence[Path],
    *,
    config: RetrievalConfig,
    max_documents: int | None,
    output_root: Path = OUTPUT_ROOT,
    retriever: ArticleRetriever | None = None,
) -> dict[str, int]:
    """Retrieve until ``max_documents`` valid stored documents are encountered."""

    config.validate()
    if max_documents is not None and max_documents < 0:
        raise ValueError("max_documents cannot be negative")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / MANIFEST_NAME
    materialized_path = output_root / MATERIALIZED_RECORDS_NAME
    _repair_trailing_audit_fragment(manifest_path)
    latest = _latest_audit_entries(manifest_path)
    materialized = _load_materialized_records(materialized_path)
    worker = retriever or ArticleRetriever(config)
    summary = {
        "stored": 0,
        "retrieved_now": 0,
        "resumed": 0,
        "rejected": 0,
        "errors": 0,
        "duplicate_records": 0,
    }
    seen: dict[str, str] = {}

    for record in iter_articles(input_paths):
        if max_documents is not None and summary["stored"] >= max_documents:
            break
        url_hash = sha256_text(record.url)
        previous_url_hash = seen.get(record.article_id)
        if previous_url_hash is not None:
            if previous_url_hash != url_hash:
                raise ValueError(
                    f"article_id {record.article_id!r} is associated with multiple URLs"
                )
            summary["duplicate_records"] += 1
            continue
        seen[record.article_id] = url_hash

        prior_entry = latest.get(record.article_id)
        if (
            prior_entry is not None
            and isinstance(prior_entry.get("url_sha256"), str)
            and prior_entry["url_sha256"] != url_hash
        ):
            raise ValueError(
                f"article_id {record.article_id!r} conflicts with the "
                "URL recorded by an earlier retrieval run"
            )
        resume = _resume_status(
            record,
            prior_entry,
            output_root=output_root,
            config=config,
        )
        if resume is not None:
            summary["resumed"] += 1
            if resume == "retrieved":
                audit = latest[record.article_id]
                text_path = output_root / article_text_relative_path(
                    record.article_id
                )
                text = text_path.read_text(encoding="utf-8")
                expected = str(audit["content_sha256"])
                source_method = str(audit.get("source_method") or "unknown")
                rebuilt = _materialized_record(
                    record,
                    text=text,
                    content_sha256=expected,
                    source_method=source_method,
                )
                if materialized.get(record.article_id) != rebuilt:
                    materialized[record.article_id] = rebuilt
                    _write_materialized_records_atomic(
                        materialized_path, materialized
                    )
                summary["stored"] += 1
            else:
                summary["rejected"] += 1
            continue

        outcome = worker.retrieve(record)
        if outcome.status == "retrieved":
            assert outcome.text is not None
            text_path = output_root / article_text_relative_path(record.article_id)
            _write_text_atomic(text_path, outcome.text)
            summary["stored"] += 1
            summary["retrieved_now"] += 1
        elif outcome.status == "rejected":
            summary["rejected"] += 1
        else:
            summary["errors"] += 1
        entry = _audit_entry(record, outcome, config)
        _append_audit(manifest_path, entry)
        latest[record.article_id] = entry
        if outcome.status == "retrieved":
            assert (
                outcome.text is not None
                and outcome.content_sha256 is not None
                and outcome.source_method is not None
            )
            materialized[record.article_id] = _materialized_record(
                record,
                text=outcome.text,
                content_sha256=outcome.content_sha256,
                source_method=outcome.source_method,
            )
            _write_materialized_records_atomic(materialized_path, materialized)
    return summary


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value cannot be negative")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value cannot be negative")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="JSON/JSONL/NDJSON article records, optionally gzip-compressed",
    )
    parser.add_argument(
        "--min-chars",
        type=_positive_int,
        default=DEFAULT_MIN_CHARS,
        help=f"minimum extracted character count (default: {DEFAULT_MIN_CHARS})",
    )
    parser.add_argument(
        "--max-bytes",
        type=_positive_int,
        default=DEFAULT_MAX_BYTES,
        help=f"maximum bytes per article response (default: {DEFAULT_MAX_BYTES})",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=_positive_float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"per-request timeout (default: {DEFAULT_TIMEOUT_SECONDS})",
    )
    parser.add_argument(
        "--rate-limit-seconds",
        type=_nonnegative_float,
        default=DEFAULT_DOMAIN_INTERVAL_SECONDS,
        help=(
            "minimum interval between requests to one domain; a robots crawl-delay "
            "can increase it"
        ),
    )
    parser.add_argument(
        "--max-documents",
        "--max-docs",
        type=_nonnegative_int,
        default=None,
        help=(
            "stop after this many valid stored documents, counting resumable "
            "documents already present"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    for path in args.inputs:
        if not path.is_file():
            print(f"Input file does not exist: {path}", file=sys.stderr)
            return 2
    config = RetrievalConfig(
        min_chars=args.min_chars,
        max_bytes=args.max_bytes,
        timeout_seconds=args.timeout_seconds,
        domain_interval_seconds=args.rate_limit_seconds,
    )
    try:
        summary = run(
            args.inputs,
            config=config,
            max_documents=args.max_documents,
        )
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
