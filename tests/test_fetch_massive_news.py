from __future__ import annotations

import gzip
import importlib.util
import json
import sys
import tempfile
import unittest
import urllib.parse
from datetime import date
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "fetch_massive_news.py"
SPEC = importlib.util.spec_from_file_location("fetch_massive_news", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def encoded(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


MASSIVE_PAGE_ONE = encoded(
    {
        "request_id": "request-one",
        "status": "OK",
        "results": [
            {
                "id": "article-one",
                "published_utc": "2024-01-02T12:00:00Z",
                "tickers": ["AMD"],
                "title": "Deterministic first article",
            }
        ],
        "next_url": (
            "https://api.massive.com/v2/reference/news?"
            "cursor=opaque-cursor-two"
        ),
    }
)
MASSIVE_PAGE_TWO = encoded(
    {
        "request_id": "request-two",
        "status": "OK",
        "results": [
            {
                "id": "article-two",
                "published_utc": "2024-01-03T12:00:00Z",
                "tickers": ["AMD"],
                "title": "Deterministic second article",
            }
        ],
    }
)


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


class FetchMassiveNewsTests(unittest.TestCase):
    def test_universe_loader_returns_only_the_30_stocks(self) -> None:
        tickers = MODULE.load_stock_tickers(ROOT / "config" / "price_universe.json")

        self.assertEqual(len(tickers), 30)
        self.assertEqual(len(set(tickers)), 30)
        self.assertIn("AMD", tickers)
        self.assertNotIn("SOXX", tickers)
        self.assertNotIn("SPY", tickers)

    def test_dotenv_parser_strips_quotes_and_trailing_comments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text(
                "\ufeff# comment\n"
                'export MASSIVE_API_KEY = "quoted-secret" # local key\n'
                "PLAIN='value with spaces'\n"
                "UNQUOTED=value # trailing comment\n",
                encoding="utf-8",
            )

            values = MODULE.load_dotenv_values(path)

        self.assertEqual(values["MASSIVE_API_KEY"], "quoted-secret")
        self.assertEqual(values["PLAIN"], "value with spaces")
        self.assertEqual(values["UNQUOTED"], "value")

    def test_cursor_pages_resume_and_never_persist_secret(self) -> None:
        secret = "massive-test-secret"
        requests = []

        def first_urlopen(request, timeout):
            requests.append((request, timeout))
            return FakeResponse(MASSIVE_PAGE_ONE)

        def second_urlopen(request, timeout):
            requests.append((request, timeout))
            return FakeResponse(MASSIVE_PAGE_TWO)

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            client = MODULE.MassiveClient(
                secret,
                calls_per_minute=5,
                sleeper=lambda _: None,
                clock=lambda: 100.0,
            )
            with mock.patch.object(
                MODULE.urllib.request, "urlopen", side_effect=first_urlopen
            ):
                first = MODULE.collect_news(
                    client=client,
                    output_dir=output_dir,
                    tickers=["AMD"],
                    start=date(2024, 1, 1),
                    end=date(2024, 1, 31),
                    max_requests=1,
                )

            self.assertEqual(first["status"], "in_progress")
            state = json.loads(
                (output_dir / "state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(state["next_cursor"], "opaque-cursor-two")
            self.assertEqual(state["next_page_number"], 2)
            self.assertEqual(
                gzip.decompress(
                    (
                        output_dir / "pages" / "AMD" / "page_000001.json.gz"
                    ).read_bytes()
                ),
                MASSIVE_PAGE_ONE,
            )

            with mock.patch.object(
                MODULE.urllib.request, "urlopen", side_effect=second_urlopen
            ):
                second = MODULE.collect_news(
                    client=client,
                    output_dir=output_dir,
                    tickers=["AMD"],
                    start=date(2024, 1, 1),
                    end=date(2024, 1, 31),
                    max_requests=1,
                )

            self.assertEqual(second["status"], "complete")
            manifest_path = output_dir / "manifest.json"
            stable_manifest = manifest_path.read_bytes()
            with mock.patch.object(MODULE.urllib.request, "urlopen") as urlopen:
                repeated = MODULE.collect_news(
                    client=client,
                    output_dir=output_dir,
                    tickers=["AMD"],
                    start=date(2024, 1, 1),
                    end=date(2024, 1, 31),
                    max_requests=10,
                )
            urlopen.assert_not_called()
            self.assertEqual(repeated["status"], "complete")
            self.assertEqual(manifest_path.read_bytes(), stable_manifest)

            manifest = json.loads(stable_manifest)
            self.assertEqual(manifest["totals"]["pages"], 2)
            self.assertEqual(manifest["totals"]["results"], 2)
            self.assertFalse(manifest["credential_values_recorded"])
            for path in output_dir.rglob("*"):
                if not path.is_file():
                    continue
                body = (
                    gzip.decompress(path.read_bytes())
                    if path.suffix == ".gz"
                    else path.read_bytes()
                )
                self.assertNotIn(secret.encode("utf-8"), body)

        self.assertEqual(len(requests), 2)
        first_request = requests[0][0]
        second_request = requests[1][0]
        self.assertEqual(
            first_request.get_header("Authorization"), f"Bearer {secret}"
        )
        self.assertNotIn(secret, first_request.full_url)
        self.assertNotIn(secret, second_request.full_url)
        first_query = urllib.parse.parse_qs(
            urllib.parse.urlsplit(first_request.full_url).query
        )
        second_query = urllib.parse.parse_qs(
            urllib.parse.urlsplit(second_request.full_url).query
        )
        self.assertNotIn("cursor", first_query)
        self.assertEqual(second_query["cursor"], ["opaque-cursor-two"])
        self.assertEqual(first_query["ticker"], ["AMD"])

    def test_next_url_requires_exactly_one_cursor(self) -> None:
        self.assertIsNone(MODULE.extract_cursor(None))
        with self.assertRaises(MODULE.MassiveNewsError):
            MODULE.extract_cursor("https://api.massive.com/v2/reference/news")
        with self.assertRaises(MODULE.MassiveNewsError):
            MODULE.extract_cursor(
                "https://api.massive.com/v2/reference/news?cursor=a&cursor=b"
            )


if __name__ == "__main__":
    unittest.main()
