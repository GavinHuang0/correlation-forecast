from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import hmac
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "data"
    / "external"
    / "news_provider_comparison"
    / "v1_0"
    / "flat_access_audit.json"
)
ENDPOINT = "https://files.massive.com"
BUCKET = "flatfiles"
REGION = "us-east-1"
SERVICE = "s3"
USER_AGENT = "correlation-forecast-provider-audit/1.0"
MINUTE_AGGREGATE_ROOT = "us_stocks_sip/minute_aggs_v1"
HISTORICAL_MINUTE_PREFIX = f"{MINUTE_AGGREGATE_ROOT}/2016/01/"
EXPECTED_MINUTE_COLUMNS = (
    "ticker",
    "volume",
    "open",
    "close",
    "high",
    "low",
    "window_start",
    "transactions",
)
DEFAULT_SAMPLE_BYTES = 64 * 1024
MAX_SAMPLE_BYTES = 1024 * 1024
MAX_SCHEMA_BYTES = 64 * 1024
MAX_ERROR_BYTES = 8 * 1024
MAX_LIST_RESPONSE_BYTES = 2 * 1024 * 1024
MINUTE_OBJECT_PATTERN = re.compile(
    r"^us_stocks_sip/minute_aggs_v1/"
    r"(?P<year>[0-9]{4})/(?P<month>[0-9]{2})/"
    r"(?P<date>[0-9]{4}-[0-9]{2}-[0-9]{2})\.csv\.gz$"
)


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Never forward a signed request to another origin."""

    def redirect_request(
        self,
        request: urllib.request.Request,
        file_pointer: Any,
        code: int,
        message: str,
        headers: Mapping[str, str],
        new_url: str,
    ) -> None:
        return None


def open_no_redirect(
    request: urllib.request.Request, *, timeout: float
) -> Any:
    opener = urllib.request.build_opener(NoRedirectHandler())
    return opener.open(request, timeout=timeout)


def read_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[name.strip()] = value
    return values


def credential(
    name: str, *, dotenv_path: Path = ROOT / ".env"
) -> str:
    value = os.environ.get(name)
    if value:
        return value
    if not dotenv_path.is_file():
        raise RuntimeError(f"{name} is not set and {dotenv_path} is missing")
    value = read_dotenv(dotenv_path).get(name)
    if not value:
        raise RuntimeError(f"{name} is missing from the environment and .env")
    return value


def _sign(key: bytes, message: str) -> bytes:
    return hmac.new(key, message.encode("utf-8"), hashlib.sha256).digest()


def _signed_request(
    *,
    access_key: str,
    secret_key: str,
    method: str,
    canonical_uri: str,
    query_items: Mapping[str, str] | None,
    now: dt.datetime,
    additional_signed_headers: Mapping[str, str] | None = None,
) -> urllib.request.Request:
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("Signing time must be timezone-aware")
    for name, value in (
        ("access_key", access_key),
        ("secret_key", secret_key),
    ):
        if not value:
            raise ValueError(f"{name} must not be empty")
        if any(ord(character) < 32 or ord(character) == 127 for character in value):
            raise ValueError(f"{name} contains a control character")
    timestamp = now.astimezone(dt.timezone.utc)
    amz_date = timestamp.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = timestamp.strftime("%Y%m%d")
    host = urllib.parse.urlparse(ENDPOINT).netloc
    canonical_query = urllib.parse.urlencode(
        sorted((query_items or {}).items()),
        quote_via=urllib.parse.quote,
        safe="-_.~",
    )
    payload_hash = hashlib.sha256(b"").hexdigest()
    canonical_header_values = {
        "host": host,
        "x-amz-content-sha256": payload_hash,
        "x-amz-date": amz_date,
    }
    for raw_name, raw_value in (additional_signed_headers or {}).items():
        name = raw_name.strip().lower()
        if name in {"authorization", "host", "x-amz-content-sha256", "x-amz-date"}:
            raise ValueError(f"Reserved signed header: {raw_name}")
        if not name or name != raw_name.lower():
            raise ValueError(f"Signed header names must be lowercase: {raw_name}")
        canonical_header_values[name] = " ".join(str(raw_value).split())
    signed_header_names = sorted(canonical_header_values)
    canonical_headers = "".join(
        f"{name}:{canonical_header_values[name]}\n"
        for name in signed_header_names
    )
    signed_headers = ";".join(signed_header_names)
    canonical_request = "\n".join(
        [
            method,
            canonical_uri,
            canonical_query,
            canonical_headers,
            signed_headers,
            payload_hash,
        ]
    )
    scope = f"{date_stamp}/{REGION}/{SERVICE}/aws4_request"
    string_to_sign = "\n".join(
        [
            "AWS4-HMAC-SHA256",
            amz_date,
            scope,
            hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
        ]
    )
    date_key = _sign(("AWS4" + secret_key).encode("utf-8"), date_stamp)
    region_key = _sign(date_key, REGION)
    service_key = _sign(region_key, SERVICE)
    signing_key = _sign(service_key, "aws4_request")
    signature = hmac.new(
        signing_key, string_to_sign.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    authorization = (
        "AWS4-HMAC-SHA256 "
        f"Credential={access_key}/{scope}, "
        f"SignedHeaders={signed_headers}, "
        f"Signature={signature}"
    )
    url = f"{ENDPOINT}{canonical_uri}"
    if canonical_query:
        url += f"?{canonical_query}"
    request_headers = {
        "Authorization": authorization,
        "User-Agent": USER_AGENT,
        "x-amz-content-sha256": payload_hash,
        "x-amz-date": amz_date,
    }
    request_headers.update(additional_signed_headers or {})
    return urllib.request.Request(
        url,
        method=method,
        headers=request_headers,
    )


def signed_list_request(
    *,
    access_key: str,
    secret_key: str,
    prefix: str,
    max_keys: int,
    now: dt.datetime,
) -> urllib.request.Request:
    if max_keys < 1 or max_keys > 1000:
        raise ValueError("max_keys must be in [1, 1000]")
    query_items = {
        "list-type": "2",
        "max-keys": str(max_keys),
        "prefix": prefix,
    }
    return _signed_request(
        access_key=access_key,
        secret_key=secret_key,
        method="GET",
        canonical_uri=f"/{BUCKET}",
        query_items=query_items,
        now=now,
    )


def validate_object_key(object_key: str) -> str:
    if not object_key or object_key.startswith(("/", "\\")):
        raise ValueError("object_key must be a relative S3 key")
    if "\\" in object_key or "\x00" in object_key:
        raise ValueError("object_key contains an unsafe character")
    if any(part in {"", ".", ".."} for part in object_key.split("/")):
        raise ValueError("object_key contains an unsafe path segment")
    return object_key


def signed_object_request(
    *,
    access_key: str,
    secret_key: str,
    object_key: str,
    method: str,
    now: dt.datetime,
    byte_range: tuple[int, int] | None = None,
) -> urllib.request.Request:
    object_key = validate_object_key(object_key)
    if method not in {"HEAD", "GET"}:
        raise ValueError("Object validation only supports HEAD and GET")
    additional_headers: dict[str, str] = {"accept-encoding": "identity"}
    if byte_range is not None:
        start, end = byte_range
        if method != "GET" or start < 0 or end < start:
            raise ValueError("byte_range requires a valid GET byte interval")
        additional_headers["range"] = f"bytes={start}-{end}"
    encoded_key = urllib.parse.quote(object_key, safe="/-_.~")
    return _signed_request(
        access_key=access_key,
        secret_key=secret_key,
        method=method,
        canonical_uri=f"/{BUCKET}/{encoded_key}",
        query_items=None,
        now=now,
        additional_signed_headers=additional_headers,
    )


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_list_response(body: bytes) -> dict[str, object]:
    root = ET.fromstring(body)
    keys = [
        element.text or ""
        for element in root.iter()
        if local_name(element.tag) == "Key"
    ]
    truncated = next(
        (
            (element.text or "").lower() == "true"
            for element in root.iter()
            if local_name(element.tag) == "IsTruncated"
        ),
        False,
    )
    return {
        "object_count_returned": len(keys),
        "sample_keys": keys,
        "is_truncated": truncated,
    }


def redact_credentials(value: str, *credentials: str) -> str:
    """Remove credential values and their URL encodings from audit text."""

    redacted = value
    for credential_value in credentials:
        if not credential_value:
            continue
        variants = {
            credential_value,
            urllib.parse.quote(credential_value, safe=""),
            urllib.parse.quote_plus(credential_value),
        }
        for variant in variants:
            if variant:
                redacted = redacted.replace(variant, "[REDACTED]")
    return redacted


def _response_status(response: Any) -> int:
    status = getattr(response, "status", None)
    if status is None:
        status = response.getcode()
    return int(status)


def _header(response: Any, name: str) -> str | None:
    value = response.headers.get(name)
    if value is None:
        return None
    return str(value)


def _nonnegative_header_int(response: Any, name: str) -> int | None:
    value = _header(response, name)
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def _read_error_body(error: urllib.error.HTTPError) -> bytes:
    body = error.read(MAX_ERROR_BYTES + 1)
    return body[:MAX_ERROR_BYTES]


def _allowlisted_s3_error_code(body: bytes) -> str | None:
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return None
    code = next(
        (
            (element.text or "").strip()
            for element in root.iter()
            if local_name(element.tag) == "Code"
        ),
        "",
    )
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9]{0,63}", code):
        return code
    return None


def _request_error_audit(
    error: urllib.error.HTTPError | urllib.error.URLError,
    *,
    access_key: str,
    secret_key: str,
) -> dict[str, object]:
    if isinstance(error, urllib.error.HTTPError):
        body = _read_error_body(error)
        return {
            "http_status": error.code,
            "access": "redirect_refused"
            if 300 <= error.code < 400
            else "denied",
            "error_category": "redirect_refused"
            if 300 <= error.code < 400
            else "http_error",
            "s3_error_code": _allowlisted_s3_error_code(body),
            "redirect_followed": False,
        }
    return {
        "http_status": None,
        "access": "network_error",
        "error_category": "network_error",
        "redirect_followed": False,
    }


def inspect_gzip_csv_schema(sample: bytes) -> dict[str, object]:
    result: dict[str, object] = {
        "validation_scope": "prefix_only",
        "full_gzip_integrity_verified": False,
        "format": "gzip_csv" if sample.startswith(b"\x1f\x8b") else "unknown",
        "header_complete": False,
        "columns": [],
        "column_count": 0,
        "schema_sha256": None,
        "matches_expected_minute_aggregate_schema": False,
        "schema_status": "invalid_gzip",
    }
    if result["format"] != "gzip_csv":
        result["schema_error"] = "sample does not begin with a gzip header"
        return result
    try:
        decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
        decompressed = decompressor.decompress(sample, MAX_SCHEMA_BYTES)
    except zlib.error:
        result["schema_error"] = "gzip prefix could not be decompressed"
        return result
    result["decompressed_prefix_bytes"] = len(decompressed)
    newline = decompressed.find(b"\n")
    if newline < 0:
        result["schema_status"] = "insufficient_prefix"
        result["schema_error"] = "CSV header was not complete within the sample"
        return result
    try:
        header = decompressed[:newline].decode("utf-8-sig").rstrip("\r")
        columns = tuple(next(csv.reader([header])))
    except (UnicodeDecodeError, csv.Error, StopIteration):
        result["schema_status"] = "schema_mismatch"
        result["schema_error"] = "CSV header could not be parsed"
        return result
    if not columns or any(not column for column in columns):
        result["schema_status"] = "schema_mismatch"
        result["schema_error"] = "CSV header has an empty column"
        return result
    canonical_schema = ",".join(columns).encode("utf-8")
    result.update(
        {
            "header_complete": True,
            "columns": list(columns),
            "column_count": len(columns),
            "schema_sha256": hashlib.sha256(canonical_schema).hexdigest(),
            "matches_expected_minute_aggregate_schema": (
                columns == EXPECTED_MINUTE_COLUMNS
            ),
            "schema_status": (
                "expected_schema"
                if columns == EXPECTED_MINUTE_COLUMNS
                else "schema_mismatch"
            ),
        }
    )
    return result


def _normalized_etag(response: Any) -> str | None:
    value = _header(response, "ETag")
    if value is None:
        return None
    normalized = value.strip().strip('"')
    if not normalized or len(normalized) > 256:
        return None
    return normalized


def _etag_sha256(etag: str | None) -> str | None:
    if etag is None:
        return None
    return hashlib.sha256(etag.encode("utf-8")).hexdigest()


def _content_type_class(response: Any) -> str:
    value = (_header(response, "Content-Type") or "").lower()
    if "gzip" in value:
        return "gzip"
    if "csv" in value:
        return "csv"
    if "octet-stream" in value:
        return "binary"
    return "unspecified_or_other"


def _parse_content_range(value: str | None) -> tuple[int, int, int] | None:
    if value is None:
        return None
    match = re.fullmatch(r"bytes ([0-9]+)-([0-9]+)/([0-9]+)", value.strip())
    if match is None:
        return None
    start, end, total = (int(part) for part in match.groups())
    if start > end or end >= total:
        return None
    return start, end, total


def run_object_validation(
    *,
    access_key: str,
    secret_key: str,
    object_key: str,
    sample_bytes: int = DEFAULT_SAMPLE_BYTES,
    now: dt.datetime | None = None,
) -> dict[str, object]:
    if sample_bytes < 1 or sample_bytes > MAX_SAMPLE_BYTES:
        raise ValueError(
            f"sample_bytes must be in [1, {MAX_SAMPLE_BYTES}]"
        )
    object_key = validate_object_key(object_key)
    timestamp = now or dt.datetime.now(dt.timezone.utc)
    result: dict[str, object] = {
        "object_key": object_key,
        "credential_names": ["MASSIVE_FLAT_KEY", "MASSIVE_FLAT_SECRET"],
        "credentials_recorded": False,
        "sample_byte_limit": sample_bytes,
        "maximum_network_bytes_read": sample_bytes + 1,
        "redirect_followed": False,
        "full_object_download_performed": False,
        "retrieval_succeeded": False,
    }

    head_request = signed_object_request(
        access_key=access_key,
        secret_key=secret_key,
        object_key=object_key,
        method="HEAD",
        now=timestamp,
    )
    head_etag: str | None = None
    full_length: int | None = None
    head_succeeded = False
    try:
        with open_no_redirect(head_request, timeout=30) as response:
            head_etag = _normalized_etag(response)
            full_length = _nonnegative_header_int(response, "Content-Length")
            head_succeeded = True
            result["head"] = {
                "http_status": _response_status(response),
                "content_length_bytes": full_length,
                "content_type_class": _content_type_class(response),
                "server_etag_sha256": _etag_sha256(head_etag),
            }
    except (urllib.error.HTTPError, urllib.error.URLError) as error:
        result["head"] = _request_error_audit(
            error, access_key=access_key, secret_key=secret_key
        )
        # Some S3-compatible entitlements may deny HEAD while still allowing
        # a ranged GetObject. Continue with the independently signed, bounded
        # GET so the audit distinguishes that case from complete object denial.
        result["head_access_failed"] = True

    range_request = signed_object_request(
        access_key=access_key,
        secret_key=secret_key,
        object_key=object_key,
        method="GET",
        byte_range=(0, sample_bytes - 1),
        now=timestamp,
    )
    try:
        with open_no_redirect(range_request, timeout=30) as response:
            received = response.read(sample_bytes + 1)
            exceeded_bound = len(received) > sample_bytes
            sample = received[:sample_bytes]
            status = _response_status(response)
            content_range = _parse_content_range(
                _header(response, "Content-Range")
            )
            sample_etag = _normalized_etag(response)
            etag_match = (
                None
                if head_etag is None or sample_etag is None
                else head_etag == sample_etag
            )
            range_valid = bool(
                status == 206
                and content_range is not None
                and content_range[0] == 0
                and content_range[1] + 1 == len(sample)
                and (
                    full_length is None
                    or content_range[2] == full_length
                )
                and not exceeded_bound
            )
            known_total = (
                full_length
                if full_length is not None
                else content_range[2]
                if content_range is not None
                else None
            )
            full_download = (
                isinstance(known_total, int)
                and not exceeded_bound
                and len(sample) >= known_total
            )
            schema = inspect_gzip_csv_schema(sample)
            result["sample"] = {
                "http_status": status,
                "request_range": f"bytes=0-{sample_bytes - 1}",
                "content_range_start": (
                    content_range[0] if content_range is not None else None
                ),
                "content_range_end": (
                    content_range[1] if content_range is not None else None
                ),
                "content_range_total_bytes": (
                    content_range[2] if content_range is not None else None
                ),
                "range_honored_and_valid": range_valid,
                "response_bytes_read": len(received),
                "sample_bytes_hashed": len(sample),
                "response_exceeded_bound": exceeded_bound,
                "sample_sha256": hashlib.sha256(sample).hexdigest(),
                "head_get_etag_match": etag_match,
                "sample_etag_sha256": _etag_sha256(sample_etag),
                "schema": schema,
            }
            result["full_object_download_performed"] = full_download
            result["retrieval_succeeded"] = (
                range_valid
                and not full_download
                and etag_match is not False
                and schema["schema_status"] == "expected_schema"
            )
            if result["retrieval_succeeded"]:
                result["access"] = (
                    "allowed_range_get_head_denied"
                    if not head_succeeded
                    else "allowed"
                )
            elif status != 206:
                result["access"] = "range_not_honored"
            elif etag_match is False:
                result["access"] = "object_changed_between_head_and_get"
            else:
                result["access"] = "invalid_sample"
    except (urllib.error.HTTPError, urllib.error.URLError) as error:
        result["failed_phases"] = (
            ["head", "range_get"] if not head_succeeded else ["range_get"]
        )
        result["sample"] = _request_error_audit(
            error, access_key=access_key, secret_key=secret_key
        )
        result["access"] = result["sample"]["access"]
    return result


def run_probe(
    *,
    access_key: str,
    secret_key: str,
    prefix: str,
    max_keys: int,
    now: dt.datetime | None = None,
) -> dict[str, object]:
    request = signed_list_request(
        access_key=access_key,
        secret_key=secret_key,
        prefix=prefix,
        max_keys=max_keys,
        now=now or dt.datetime.now(dt.timezone.utc),
    )
    result: dict[str, object] = {
        "endpoint": ENDPOINT,
        "bucket": BUCKET,
        "prefix": prefix,
        "max_keys": max_keys,
        "credential_names": ["MASSIVE_FLAT_KEY", "MASSIVE_FLAT_SECRET"],
        "credentials_recorded": False,
        "purpose": (
            "Entitlement-only probe. Massive flat files contain market data, "
            "not news, and are not used as a news source."
        ),
    }
    try:
        with open_no_redirect(request, timeout=30) as response:
            body = response.read(MAX_LIST_RESPONSE_BYTES + 1)
            if len(body) > MAX_LIST_RESPONSE_BYTES:
                raise ValueError("S3 list response exceeded the audit byte cap")
            result.update(
                {
                    "http_status": _response_status(response),
                    "access": "allowed",
                    "redirect_followed": False,
                    **parse_list_response(body),
                }
            )
    except (urllib.error.HTTPError, urllib.error.URLError) as error:
        result.update(
            _request_error_audit(
                error, access_key=access_key, secret_key=secret_key
            )
        )
    return result


def validate_minute_object_for_role(
    object_key: str,
    *,
    role: str,
    reference_time: dt.datetime,
) -> dt.date:
    object_key = validate_object_key(object_key)
    match = MINUTE_OBJECT_PATTERN.fullmatch(object_key)
    if match is None:
        raise ValueError(
            f"{role} object must be a stock minute-aggregate .csv.gz key"
        )
    try:
        object_date = dt.date.fromisoformat(match.group("date"))
    except ValueError as exc:
        raise ValueError(f"{role} object has an invalid calendar date") from exc
    if (
        object_date.year != int(match.group("year"))
        or object_date.month != int(match.group("month"))
    ):
        raise ValueError(f"{role} object date does not match its path")
    if object_date.weekday() >= 5:
        raise ValueError(f"{role} object date is not a weekday")
    reference_date = reference_time.astimezone(dt.timezone.utc).date()
    if role == "current":
        age_days = (reference_date - object_date).days
        if age_days < 0 or age_days > 45:
            raise ValueError(
                "current object must be no more than 45 days old"
            )
    elif role == "historical_2016":
        if object_date.year != 2016:
            raise ValueError("historical object must be from 2016")
    else:
        raise ValueError(f"Unknown object role: {role}")
    return object_date


def assert_credentials_absent(
    value: Mapping[str, object], *credentials: str
) -> None:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True)
    for credential_value in credentials:
        variants = {
            credential_value,
            urllib.parse.quote(credential_value, safe=""),
            urllib.parse.quote_plus(credential_value),
        }
        for variant in variants:
            if variant and variant in serialized:
                raise RuntimeError(
                    "Credential material was detected in audit output"
                )


def run_minute_object_validations(
    *,
    access_key: str,
    secret_key: str,
    current_object_key: str,
    historical_object_key: str,
    sample_bytes: int = DEFAULT_SAMPLE_BYTES,
    now: dt.datetime | None = None,
) -> dict[str, object]:
    timestamp = now or dt.datetime.now(dt.timezone.utc)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("Validation time must be timezone-aware")
    if current_object_key == historical_object_key:
        raise ValueError("Current and historical object keys must differ")
    roles = (
        ("current", current_object_key),
        ("historical_2016", historical_object_key),
    )
    objects: list[dict[str, object]] = []
    for role, object_key in roles:
        object_date = validate_minute_object_for_role(
            object_key,
            role=role,
            reference_time=timestamp,
        )
        validation = run_object_validation(
            access_key=access_key,
            secret_key=secret_key,
            object_key=object_key,
            sample_bytes=sample_bytes,
            now=timestamp,
        )
        objects.append(
            {
                "role": role,
                "object_date": object_date.isoformat(),
                **validation,
            }
        )
    result: dict[str, object] = {
        "validation_version": "massive-flat-minute-object-validation-v1",
        "endpoint": ENDPOINT,
        "bucket": BUCKET,
        "validation_scope": "head_and_bounded_prefix",
        "request_budget": {
            "objects": 2,
            "requests_per_object": 2,
            "methods": ["HEAD", "GET_RANGE"],
            "maximum_sample_bytes_per_object": sample_bytes,
            "redirects_allowed": 0,
            "retries": 0,
        },
        "credential_names": ["MASSIVE_FLAT_KEY", "MASSIVE_FLAT_SECRET"],
        "credentials_recorded": False,
        "objects": objects,
        "actual_object_retrieval_succeeded": all(
            bool(item["retrieval_succeeded"]) for item in objects
        ),
        "full_object_download_performed": any(
            bool(item["full_object_download_performed"]) for item in objects
        ),
    }
    assert_credentials_absent(result, access_key, secret_key)
    return result


def write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "List one Massive flat-file key, or validate exactly one current "
            "and one 2016 minute-aggregate object with HEAD plus bounded Range "
            "GET requests. The result never records credentials."
        )
    )
    parser.add_argument(
        "--prefix",
        default="us_stocks_sip/day_aggs_v1/2026/07/",
    )
    parser.add_argument("--max-keys", type=int, default=1)
    parser.add_argument(
        "--current-minute-object",
        help=(
            "Exact current us_stocks_sip/minute_aggs_v1 CSV.GZ object key. "
            "Must be paired with --historical-minute-object."
        ),
    )
    parser.add_argument(
        "--historical-minute-object",
        help=(
            "Exact 2016 us_stocks_sip/minute_aggs_v1 CSV.GZ object key. "
            "Must be paired with --current-minute-object."
        ),
    )
    parser.add_argument(
        "--sample-bytes",
        type=int,
        default=DEFAULT_SAMPLE_BYTES,
        help=(
            f"Compressed prefix byte limit for each Range GET "
            f"(maximum {MAX_SAMPLE_BYTES})."
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dotenv", type=Path, default=ROOT / ".env")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if bool(args.current_minute_object) != bool(
        args.historical_minute_object
    ):
        raise ValueError(
            "Both minute-object arguments must be supplied together"
        )
    access_key = credential("MASSIVE_FLAT_KEY", dotenv_path=args.dotenv)
    secret_key = credential("MASSIVE_FLAT_SECRET", dotenv_path=args.dotenv)
    timestamp = dt.datetime.now(dt.timezone.utc)
    if args.current_minute_object:
        result = run_minute_object_validations(
            access_key=access_key,
            secret_key=secret_key,
            current_object_key=args.current_minute_object,
            historical_object_key=args.historical_minute_object,
            sample_bytes=args.sample_bytes,
            now=timestamp,
        )
    else:
        result = run_probe(
            access_key=access_key,
            secret_key=secret_key,
            prefix=args.prefix,
            max_keys=args.max_keys,
            now=timestamp,
        )
    result["probed_at_utc"] = timestamp.isoformat()
    assert_credentials_absent(result, access_key, secret_key)
    write_json_atomic(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "http_status": result.get("http_status"),
                "access": result.get("access"),
                "object_count_returned": result.get("object_count_returned", 0),
                "actual_object_retrieval_succeeded": result.get(
                    "actual_object_retrieval_succeeded"
                ),
            }
        )
    )
    if args.current_minute_object:
        return 0 if result["actual_object_retrieval_succeeded"] else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
