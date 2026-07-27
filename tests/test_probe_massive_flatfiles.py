from __future__ import annotations

import datetime as dt
import gzip
import io
import importlib.util
import json
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "probe_massive_flatfiles.py"
SPEC = importlib.util.spec_from_file_location("flat_probe_for_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


class FakeResponse:
    def __init__(
        self,
        body: bytes = b"",
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._body = io.BytesIO(body)
        self.status = status
        self.headers = headers or {}

    def read(self, size: int = -1) -> bytes:
        return self._body.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def gzip_sample(size: int) -> bytes:
    raw = (
        b"ticker,volume,open,close,high,low,window_start,transactions\n"
        b"AAPL,10,1,2,3,1,123,4\n"
    )
    compressed = gzip.compress(raw, mtime=0)
    if len(compressed) > size:
        raise AssertionError("Synthetic gzip fixture exceeds sample size")
    return compressed + (b"\x00" * (size - len(compressed)))


class MassiveFlatProbeTests(unittest.TestCase):
    def test_dotenv_parser_strips_balanced_quotes(self) -> None:
        path = ROOT / "tests" / "fixtures" / "nonexistent.env"
        self.assertEqual(
            probe.read_dotenv.__name__,
            "read_dotenv",
        )
        lines = 'A="quoted"\nB=\'single\'\nC=plain\n# ignored\n'
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory) / ".env"
            temporary.write_text(lines, encoding="utf-8")
            self.assertEqual(
                probe.read_dotenv(temporary),
                {"A": "quoted", "B": "single", "C": "plain"},
            )
        self.assertFalse(path.exists())

    def test_signing_is_deterministic_and_does_not_put_secret_in_url(self) -> None:
        request = probe.signed_list_request(
            access_key="ACCESS",
            secret_key="SECRET",
            prefix="us_stocks_sip/day_aggs_v1/2026/07/",
            max_keys=1,
            now=dt.datetime(2026, 7, 26, 12, 0, tzinfo=dt.timezone.utc),
        )
        self.assertIn("list-type=2", request.full_url)
        self.assertIn("max-keys=1", request.full_url)
        self.assertNotIn("ACCESS", request.full_url)
        self.assertNotIn("SECRET", request.full_url)
        self.assertTrue(request.headers["Authorization"].startswith("AWS4-HMAC-SHA256"))

    def test_head_and_range_requests_are_signed_without_redirectable_urls(
        self,
    ) -> None:
        now = dt.datetime(
            2026, 7, 26, 12, 0, tzinfo=dt.timezone.utc
        )
        key = (
            "us_stocks_sip/minute_aggs_v1/"
            "2026/07/2026-07-24.csv.gz"
        )
        head = probe.signed_object_request(
            access_key="ACCESS",
            secret_key="SECRET",
            object_key=key,
            method="HEAD",
            now=now,
        )
        ranged = probe.signed_object_request(
            access_key="ACCESS",
            secret_key="SECRET",
            object_key=key,
            method="GET",
            byte_range=(0, 65535),
            now=now,
        )
        range_headers = {
            name.lower(): value for name, value in ranged.headers.items()
        }

        self.assertEqual(head.get_method(), "HEAD")
        self.assertEqual(ranged.get_method(), "GET")
        self.assertEqual(range_headers["range"], "bytes=0-65535")
        self.assertEqual(range_headers["accept-encoding"], "identity")
        self.assertIn(
            "accept-encoding;host;range;x-amz-content-sha256;x-amz-date",
            ranged.headers["Authorization"],
        )
        for request in (head, ranged):
            self.assertNotIn("ACCESS", request.full_url)
            self.assertNotIn("SECRET", request.full_url)

    def test_signing_rejects_naive_time_and_credential_control_chars(
        self,
    ) -> None:
        arguments = {
            "access_key": "ACCESS",
            "secret_key": "SECRET",
            "prefix": "fixture/",
            "max_keys": 1,
            "now": dt.datetime(2026, 7, 26, 12, 0),
        }
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            probe.signed_list_request(**arguments)
        arguments["now"] = dt.datetime(
            2026, 7, 26, 12, 0, tzinfo=dt.timezone.utc
        )
        arguments["secret_key"] = "SECRET\nINJECT"
        with self.assertRaisesRegex(ValueError, "control character"):
            probe.signed_list_request(**arguments)

    def test_parse_list_response_handles_s3_namespace(self) -> None:
        payload = b"""<?xml version="1.0" encoding="UTF-8"?>
        <ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
          <IsTruncated>false</IsTruncated>
          <Contents><Key>us_stocks_sip/day_aggs_v1/2026/07/a.csv.gz</Key></Contents>
        </ListBucketResult>"""
        parsed = probe.parse_list_response(payload)
        self.assertEqual(parsed["object_count_returned"], 1)
        self.assertFalse(parsed["is_truncated"])

    def test_http_error_audit_redacts_echoed_credentials(self) -> None:
        access_key = "ACCESS+KEY"
        secret_key = "SECRET/VALUE"
        body = (
            b"<Error><Code>AccessDenied</Code>"
            b"<AccessKeyId>ACCESS+KEY</AccessKeyId>"
            b"<Encoded>ACCESS%2BKEY</Encoded>"
            b"<Detail>SECRET/VALUE</Detail></Error>"
        )
        error = urllib.error.HTTPError(
            "https://files.massive.com/flatfiles",
            403,
            "Forbidden",
            {},
            io.BytesIO(body),
        )
        with mock.patch.object(probe, "open_no_redirect", side_effect=error):
            result = probe.run_probe(
                access_key=access_key,
                secret_key=secret_key,
                prefix="fixture/",
                max_keys=1,
                now=dt.datetime(
                    2026, 7, 26, 12, 0, tzinfo=dt.timezone.utc
                ),
            )

        serialized = json.dumps(result)
        self.assertNotIn(access_key, serialized)
        self.assertNotIn("ACCESS%2BKEY", serialized)
        self.assertNotIn(secret_key, serialized)
        self.assertNotIn("error_message", result)
        self.assertEqual(result["s3_error_code"], "AccessDenied")

    def test_partial_gzip_prefix_reports_expected_minute_schema(self) -> None:
        complete = gzip.compress(
            (
                b"ticker,volume,open,close,high,low,"
                b"window_start,transactions\n"
                b"AAPL,10,1,2,3,1,123,4\n"
            ),
            mtime=0,
        )
        result = probe.inspect_gzip_csv_schema(complete[:-8])

        self.assertEqual(result["schema_status"], "expected_schema")
        self.assertFalse(result["full_gzip_integrity_verified"])
        self.assertEqual(
            result["columns"], list(probe.EXPECTED_MINUTE_COLUMNS)
        )

    def test_object_validation_uses_head_then_bounded_range(self) -> None:
        key = (
            "us_stocks_sip/minute_aggs_v1/"
            "2026/07/2026-07-24.csv.gz"
        )
        sample = gzip_sample(256)
        responses = [
            FakeResponse(
                status=200,
                headers={
                    "Content-Length": "10000",
                    "Content-Type": "application/gzip",
                    "ETag": '"fixture-etag"',
                },
            ),
            FakeResponse(
                sample,
                status=206,
                headers={
                    "Content-Range": "bytes 0-255/10000",
                    "Content-Length": "256",
                    "ETag": '"fixture-etag"',
                },
            ),
        ]
        with mock.patch.object(
            probe, "open_no_redirect", side_effect=responses
        ) as opened:
            result = probe.run_object_validation(
                access_key="ACCESS+KEY",
                secret_key="SECRET/VALUE",
                object_key=key,
                sample_bytes=256,
                now=dt.datetime(
                    2026, 7, 26, 12, 0, tzinfo=dt.timezone.utc
                ),
            )

        self.assertEqual(
            [call.args[0].get_method() for call in opened.call_args_list],
            ["HEAD", "GET"],
        )
        ranged_headers = {
            name.lower(): value
            for name, value in opened.call_args_list[1].args[0].headers.items()
        }
        self.assertEqual(ranged_headers["range"], "bytes=0-255")
        self.assertTrue(result["retrieval_succeeded"])
        self.assertFalse(result["full_object_download_performed"])
        self.assertEqual(result["head"]["content_length_bytes"], 10000)
        self.assertTrue(result["sample"]["range_honored_and_valid"])
        self.assertEqual(
            result["sample"]["schema"]["schema_status"], "expected_schema"
        )
        serialized = json.dumps(result)
        self.assertNotIn("ACCESS+KEY", serialized)
        self.assertNotIn("SECRET/VALUE", serialized)

    def test_ignored_range_is_bounded_and_fails_closed(self) -> None:
        key = (
            "us_stocks_sip/minute_aggs_v1/"
            "2026/07/2026-07-24.csv.gz"
        )
        sample = gzip_sample(257)
        responses = [
            FakeResponse(
                status=200,
                headers={"Content-Length": "10000"},
            ),
            FakeResponse(
                sample,
                status=200,
                headers={"Content-Length": "10000"},
            ),
        ]
        with mock.patch.object(
            probe, "open_no_redirect", side_effect=responses
        ):
            result = probe.run_object_validation(
                access_key="ACCESS",
                secret_key="SECRET",
                object_key=key,
                sample_bytes=256,
                now=dt.datetime(
                    2026, 7, 26, 12, 0, tzinfo=dt.timezone.utc
                ),
            )

        self.assertFalse(result["retrieval_succeeded"])
        self.assertEqual(result["access"], "range_not_honored")
        self.assertEqual(result["sample"]["response_bytes_read"], 257)
        self.assertEqual(result["sample"]["sample_bytes_hashed"], 256)
        self.assertTrue(result["sample"]["response_exceeded_bound"])

    def test_range_get_is_attempted_when_head_is_denied(self) -> None:
        key = (
            "us_stocks_sip/minute_aggs_v1/"
            "2026/07/2026-07-24.csv.gz"
        )
        head_error = urllib.error.HTTPError(
            f"https://files.massive.com/flatfiles/{key}",
            403,
            "Forbidden",
            {},
            io.BytesIO(b"<Error><Code>AccessDenied</Code></Error>"),
        )
        sample = gzip_sample(256)
        responses = [
            head_error,
            FakeResponse(
                sample,
                status=206,
                headers={
                    "Content-Range": "bytes 0-255/10000",
                    "Content-Length": "256",
                    "ETag": '"fixture-etag"',
                },
            ),
        ]
        with mock.patch.object(
            probe, "open_no_redirect", side_effect=responses
        ) as opened:
            result = probe.run_object_validation(
                access_key="ACCESS",
                secret_key="SECRET",
                object_key=key,
                sample_bytes=256,
                now=dt.datetime(
                    2026, 7, 26, 12, 0, tzinfo=dt.timezone.utc
                ),
            )

        self.assertEqual(opened.call_count, 2)
        self.assertTrue(result["head_access_failed"])
        self.assertTrue(result["retrieval_succeeded"])
        self.assertEqual(result["access"], "allowed_range_get_head_denied")

    def test_two_object_validation_is_exact_and_credential_safe(self) -> None:
        current = (
            "us_stocks_sip/minute_aggs_v1/"
            "2026/07/2026-07-24.csv.gz"
        )
        historical = (
            "us_stocks_sip/minute_aggs_v1/"
            "2016/01/2016-01-04.csv.gz"
        )
        sample = gzip_sample(256)
        responses = []
        for etag in ('"current"', '"historical"'):
            responses.extend(
                [
                    FakeResponse(
                        status=200,
                        headers={
                            "Content-Length": "10000",
                            "ETag": etag,
                        },
                    ),
                    FakeResponse(
                        sample,
                        status=206,
                        headers={
                            "Content-Range": "bytes 0-255/10000",
                            "ETag": etag,
                        },
                    ),
                ]
        )
        with mock.patch.object(
            probe, "open_no_redirect", side_effect=responses
        ) as opened:
            access_key = "flat-access-123456789"
            secret_key = "flat-secret-987654321"
            result = probe.run_minute_object_validations(
                access_key=access_key,
                secret_key=secret_key,
                current_object_key=current,
                historical_object_key=historical,
                sample_bytes=256,
                now=dt.datetime(
                    2026, 7, 26, 12, 0, tzinfo=dt.timezone.utc
                ),
            )

        self.assertEqual(opened.call_count, 4)
        self.assertEqual(
            [call.args[0].get_method() for call in opened.call_args_list],
            ["HEAD", "GET", "HEAD", "GET"],
        )
        self.assertTrue(result["actual_object_retrieval_succeeded"])
        self.assertFalse(result["full_object_download_performed"])
        self.assertNotIn(access_key, json.dumps(result))
        self.assertNotIn(secret_key, json.dumps(result))

    def test_object_roles_reject_wrong_dataset_or_year(self) -> None:
        now = dt.datetime(
            2026, 7, 26, 12, 0, tzinfo=dt.timezone.utc
        )
        with self.assertRaisesRegex(ValueError, "minute-aggregate"):
            probe.validate_minute_object_for_role(
                "us_stocks_sip/day_aggs_v1/2026/07/2026-07-24.csv.gz",
                role="current",
                reference_time=now,
            )
        with self.assertRaisesRegex(ValueError, "from 2016"):
            probe.validate_minute_object_for_role(
                (
                    "us_stocks_sip/minute_aggs_v1/"
                    "2017/01/2017-01-03.csv.gz"
                ),
                role="historical_2016",
                reference_time=now,
            )

    def test_redirect_handler_refuses_signed_redirect(self) -> None:
        request = probe.signed_list_request(
            access_key="ACCESS",
            secret_key="SECRET",
            prefix="fixture/",
            max_keys=1,
            now=dt.datetime(
                2026, 7, 26, 12, 0, tzinfo=dt.timezone.utc
            ),
        )
        redirected = probe.NoRedirectHandler().redirect_request(
            request,
            None,
            307,
            "Temporary Redirect",
            {},
            "https://attacker.example/collect",
        )
        self.assertIsNone(redirected)


if __name__ == "__main__":
    unittest.main()
