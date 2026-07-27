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
SCRIPT = ROOT / "scripts" / "fetch_alpha_vantage_news.py"
SPEC = importlib.util.spec_from_file_location("fetch_alpha_vantage_news", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def encoded(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


ALPHA_SMALL_RESPONSE = encoded(
    {
        "items": "1",
        "sentiment_score_definition": "deterministic fixture",
        "feed": [
            {
                "title": "Fixture article",
                "time_published": "20240102T120000",
                "ticker_sentiment": [
                    {"ticker": "AMD", "ticker_sentiment_score": "0.1"}
                ],
            }
        ],
    }
)
ALPHA_EMPTY_RESPONSE = encoded({"items": "0", "feed": []})
ALPHA_CEILING_RESPONSE = encoded(
    {
        "items": "1000",
        "feed": [
            {
                "title": f"Ceiling fixture {index:04d}",
                "time_published": "20240102T120000",
            }
            for index in range(1_000)
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


class FetchAlphaVantageNewsTests(unittest.TestCase):
    def test_dotenv_parser_strips_quotes_without_mutating_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text(
                'ALPHA_VANTAGE_KEY="quoted-key" # comment\n'
                "SINGLE='space preserving value'\n",
                encoding="utf-8",
            )

            values = MODULE.load_dotenv_values(path)

        self.assertEqual(values["ALPHA_VANTAGE_KEY"], "quoted-key")
        self.assertEqual(values["SINGLE"], "space preserving value")

    def test_initial_time_slices_are_inclusive_and_per_ticker(self) -> None:
        slices = MODULE.build_initial_slices(
            ["AMD", "NVDA"],
            start=date(2024, 1, 1),
            end=date(2024, 1, 3),
            slice_days=2,
        )

        self.assertEqual(len(slices), 4)
        self.assertEqual(
            slices[:2],
            [
                {
                    "ticker": "AMD",
                    "time_from": "20240101T0000",
                    "time_to": "20240102T2359",
                    "split_depth": 0,
                },
                {
                    "ticker": "AMD",
                    "time_from": "20240103T0000",
                    "time_to": "20240103T2359",
                    "split_depth": 0,
                },
            ],
        )
        self.assertEqual({task["ticker"] for task in slices}, {"AMD", "NVDA"})

    def test_requests_one_ticker_at_a_time_and_resumes_at_call_cap(self) -> None:
        secret = "alpha-test-secret"
        requests = []
        responses = iter([ALPHA_SMALL_RESPONSE, ALPHA_EMPTY_RESPONSE])

        def urlopen(request, timeout):
            requests.append((request, timeout))
            return FakeResponse(next(responses))

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            client = MODULE.AlphaVantageClient(
                secret,
                calls_per_minute=5,
                sleeper=lambda _: None,
                clock=lambda: 100.0,
            )
            with mock.patch.object(
                MODULE.urllib.request, "urlopen", side_effect=urlopen
            ):
                first = MODULE.collect_news(
                    client=client,
                    output_dir=output_dir,
                    tickers=["AMD", "NVDA"],
                    start=date(2024, 1, 2),
                    end=date(2024, 1, 2),
                    slice_days=1,
                    max_calls=1,
                )
                second = MODULE.collect_news(
                    client=client,
                    output_dir=output_dir,
                    tickers=["AMD", "NVDA"],
                    start=date(2024, 1, 2),
                    end=date(2024, 1, 2),
                    slice_days=1,
                    max_calls=1,
                )

            self.assertEqual(first["status"], "in_progress")
            self.assertEqual(second["status"], "complete")
            state = json.loads(
                (output_dir / "state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                [task["ticker"] for task in state["completed_slices"]],
                ["AMD", "NVDA"],
            )
            manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertFalse(manifest["multi_ticker_batching"])
            self.assertFalse(manifest["credential_values_recorded"])
            self.assertEqual(manifest["totals"]["responses"], 2)
            first_raw_path = (
                output_dir
                / "responses"
                / "AMD"
                / "20240102T0000_20240102T2359.json.gz"
            )
            self.assertEqual(
                gzip.decompress(first_raw_path.read_bytes()),
                ALPHA_SMALL_RESPONSE,
            )
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
        parsed_queries = [
            urllib.parse.parse_qs(
                urllib.parse.urlsplit(request.full_url).query
            )
            for request, _ in requests
        ]
        self.assertEqual(
            [query["tickers"] for query in parsed_queries],
            [["AMD"], ["NVDA"]],
        )
        self.assertTrue(
            all(query["apikey"] == [secret] for query in parsed_queries)
        )
        self.assertTrue(
            all(
                request.get_header("Authorization") is None
                for request, _ in requests
            )
        )

    def test_result_ceiling_bisects_slice_in_resume_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            client = MODULE.AlphaVantageClient(
                "test-key",
                calls_per_minute=5,
                sleeper=lambda _: None,
                clock=lambda: 100.0,
            )
            with mock.patch.object(
                MODULE.urllib.request,
                "urlopen",
                return_value=FakeResponse(ALPHA_CEILING_RESPONSE),
            ):
                result = MODULE.collect_news(
                    client=client,
                    output_dir=output_dir,
                    tickers=["AMD"],
                    start=date(2024, 1, 2),
                    end=date(2024, 1, 2),
                    slice_days=1,
                    max_calls=1,
                )

            state = json.loads(
                (output_dir / "state.json").read_text(encoding="utf-8")
            )

        self.assertEqual(result["status"], "in_progress")
        self.assertEqual(len(state["completed_slices"]), 0)
        self.assertEqual(len(state["pending_slices"]), 2)
        self.assertEqual(state["pending_slices"][0]["time_from"], "20240102T0000")
        self.assertEqual(state["pending_slices"][0]["time_to"], "20240102T1159")
        self.assertEqual(state["pending_slices"][1]["time_from"], "20240102T1200")
        self.assertEqual(state["pending_slices"][1]["time_to"], "20240102T2359")
        self.assertTrue(state["responses"][0]["at_result_ceiling"])

    def test_reported_item_ceiling_is_honored_even_if_feed_is_short(self) -> None:
        self.assertTrue(
            MODULE.response_at_result_ceiling({"items": "1000", "feed": []})
        )
        self.assertFalse(
            MODULE.response_at_result_ceiling({"items": "999", "feed": []})
        )

    def test_information_and_note_payloads_are_detected_without_cache_commit(
        self,
    ) -> None:
        for field in ("Information", "Note"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                output_dir = Path(directory)
                client = MODULE.AlphaVantageClient(
                    "test-key",
                    calls_per_minute=5,
                    sleeper=lambda _: None,
                    clock=lambda: 100.0,
                )
                response = encoded({field: "The API call frequency is capped."})
                with mock.patch.object(
                    MODULE.urllib.request,
                    "urlopen",
                    return_value=FakeResponse(response),
                ):
                    with self.assertRaises(MODULE.AlphaVantageRateLimitError):
                        MODULE.collect_news(
                            client=client,
                            output_dir=output_dir,
                            tickers=["AMD"],
                            start=date(2024, 1, 2),
                            end=date(2024, 1, 2),
                            slice_days=1,
                            max_calls=1,
                        )
                state = json.loads(
                    (output_dir / "state.json").read_text(encoding="utf-8")
                )
                self.assertEqual(state["responses"], [])
                self.assertEqual(len(state["pending_slices"]), 1)
                self.assertEqual(list(output_dir.rglob("*.gz")), [])

    def test_provider_limit_message_echoing_key_is_redacted(self) -> None:
        secret = "alpha-test-secret"
        client = MODULE.AlphaVantageClient(
            secret,
            calls_per_minute=5,
            sleeper=lambda _: None,
            clock=lambda: 100.0,
        )
        response = encoded(
            {
                "Information": (
                    f"The free daily request limit for key {secret} "
                    "has been reached."
                )
            }
        )
        with mock.patch.object(
            MODULE.urllib.request,
            "urlopen",
            return_value=FakeResponse(response),
        ):
            with self.assertRaises(
                MODULE.AlphaVantageRateLimitError
            ) as context:
                client.fetch_slice(
                    ticker="AMD",
                    time_from="20240101T0000",
                    time_to="20240101T2359",
                )
        self.assertNotIn(secret, str(context.exception))
        self.assertIn("[REDACTED]", str(context.exception))

    def test_client_rejects_multi_ticker_query_before_network(self) -> None:
        client = MODULE.AlphaVantageClient(
            "test-key",
            sleeper=lambda _: None,
            clock=lambda: 100.0,
        )
        with mock.patch.object(MODULE.urllib.request, "urlopen") as urlopen:
            with self.assertRaises(MODULE.AlphaVantageNewsError):
                client.fetch_slice(
                    ticker="AMD,NVDA",
                    time_from="20240101T0000",
                    time_to="20240101T2359",
                )
        urlopen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
