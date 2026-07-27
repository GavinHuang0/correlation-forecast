from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "retrieve_article_fulltext.py"
SPEC = importlib.util.spec_from_file_location("retrieve_article_fulltext", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ArticleFulltextTests(unittest.TestCase):
    def test_extracts_json_ld_article_body(self) -> None:
        document = """
        <html><head>
          <script type="application/ld+json">
            {
              "@context": "https://schema.org",
              "@type": "NewsArticle",
              "articleBody": "First paragraph.\\n\\nSecond &amp; final paragraph."
            }
          </script>
        </head><body><main><p>Short teaser.</p></main></body></html>
        """

        result = MODULE.extract_article_text(document)

        self.assertEqual(result.source_method, "json_ld_articleBody")
        self.assertEqual(
            result.text, "First paragraph.\n\nSecond & final paragraph."
        )

    def test_extracts_article_paragraphs_and_removes_boilerplate(self) -> None:
        document = """
        <html><body>
          <nav><p>Navigation paragraph should disappear.</p></nav>
          <article>
            <header><p>Author promo should disappear.</p></header>
            <p>The first <strong>article</strong> paragraph.</p>
            <div class="advertisement"><p>Buy this product.</p></div>
            <p>The second article paragraph with <script>bad()</script>text.</p>
            <aside><p>Related story.</p></aside>
          </article>
          <footer><p>Footer text.</p></footer>
        </body></html>
        """

        result = MODULE.extract_article_text(document)

        self.assertEqual(result.source_method, "article_paragraphs")
        self.assertEqual(
            result.text,
            "The first article paragraph.\n\n"
            "The second article paragraph with text.",
        )

    def test_robots_denial_prevents_article_fetch(self) -> None:
        calls: list[str] = []

        def fetcher(
            url: str, *, timeout: float, max_bytes: int, user_agent: str
        ) -> object:
            calls.append(url)
            if url == "https://news.example/robots.txt":
                return MODULE.HttpResponse(
                    200,
                    {"content-type": "text/plain"},
                    b"User-agent: *\nDisallow: /private/\n",
                    url,
                )
            self.fail("The disallowed article URL must not be requested")

        retriever = MODULE.ArticleRetriever(
            MODULE.RetrievalConfig(
                min_chars=1, domain_interval_seconds=0.0
            ),
            fetcher=fetcher,
            resolver=lambda host, port: ["93.184.216.34"],
        )
        outcome = retriever.retrieve(
            MODULE.ArticleRecord(
                "article-1",
                "https://news.example/private/story",
                "Story",
                "2026-01-02T00:00:00Z",
            )
        )

        self.assertEqual(outcome.status, "rejected")
        self.assertEqual(outcome.reason, "robots_denied")
        self.assertEqual(calls, ["https://news.example/robots.txt"])

    def test_paywall_marker_is_rejected_before_extraction(self) -> None:
        document = """
        <article>
          <p>This teaser is deliberately long enough to resemble content.</p>
          <div class="subscription-wall">Subscribe to continue reading.</div>
        </article>
        """

        self.assertTrue(MODULE.has_paywall_marker(document))

    def test_hashing_is_deterministic_utf8(self) -> None:
        text = "Markets rose.\n\nCaf\u00e9 shares followed."
        expected = hashlib.sha256(text.encode("utf-8")).hexdigest()

        self.assertEqual(MODULE.sha256_text(text), expected)
        self.assertEqual(MODULE.sha256_text(text), MODULE.sha256_text(text))

    def test_materialized_records_are_builder_compatible_and_resumable(self) -> None:
        class FakeRetriever:
            def __init__(self) -> None:
                self.calls: list[str] = []

            def retrieve(self, record: object) -> object:
                self.calls.append(record.article_id)
                text = f"Full text for {record.title}."
                return MODULE.RetrievalOutcome(
                    status="retrieved",
                    reason="ok",
                    http_status=200,
                    response_bytes=100,
                    content_length=len(text.encode("utf-8")),
                    content_sha256=MODULE.sha256_text(text),
                    source_method="article_paragraphs",
                    final_domain="news.example",
                    text=text,
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.jsonl"
            output = root / "fulltext"
            input_path.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "article_id": f"article-{index}",
                            "article_url": f"https://news.example/{index}",
                            "title": f"Story {index}",
                            "published_utc": f"2026-01-0{index}T00:00:00Z",
                        }
                    )
                    for index in (1, 2)
                )
                + "\n",
                encoding="utf-8",
            )
            config = MODULE.RetrievalConfig(min_chars=1)
            first = FakeRetriever()

            summary = MODULE.run(
                [input_path],
                config=config,
                max_documents=1,
                output_root=output,
                retriever=first,
            )

            records = [
                json.loads(line)
                for line in (output / "records.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(summary["stored"], 1)
            self.assertEqual(first.calls, ["article-1"])
            self.assertEqual(
                set(records[0]),
                {
                    "article_id",
                    "article_url",
                    "title",
                    "published",
                    "full_text",
                    "text_file",
                    "content_sha256",
                    "source_method",
                },
            )
            self.assertEqual(records[0]["full_text"], "Full text for Story 1.")

            second = FakeRetriever()
            resumed = MODULE.run(
                [input_path],
                config=config,
                max_documents=2,
                output_root=output,
                retriever=second,
            )
            final_records = [
                json.loads(line)
                for line in (output / "records.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(resumed["stored"], 2)
            self.assertEqual(resumed["resumed"], 1)
            self.assertEqual(second.calls, ["article-2"])
            self.assertEqual(
                [row["article_id"] for row in final_records],
                ["article-1", "article-2"],
            )

    def test_sensitive_query_is_rejected_without_dns(self) -> None:
        resolver_called = False

        def resolver(host: str, port: int) -> list[str]:
            nonlocal resolver_called
            resolver_called = True
            return ["93.184.216.34"]

        reason = MODULE.public_url_rejection_reason(
            "https://news.example/story?access_token=do-not-store",
            resolver=resolver,
        )

        self.assertEqual(reason, "sensitive_query")
        self.assertFalse(resolver_called)

    def test_reads_provider_json_from_gzip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "provider.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                json.dump(
                    {
                        "results": [
                            {
                                "id": "provider-id",
                                "article_url": "https://news.example/story",
                                "title": "Provider story",
                                "published_utc": "2026-04-05T12:00:00Z",
                            }
                        ]
                    },
                    handle,
                )

            records = list(MODULE.iter_articles([path]))

            self.assertEqual(
                records,
                [
                    MODULE.ArticleRecord(
                        "provider-id",
                        "https://news.example/story",
                        "Provider story",
                        "2026-04-05T12:00:00Z",
                    )
                ],
            )

    def test_resume_rejects_article_id_reassigned_to_another_url(self) -> None:
        class FakeRetriever:
            def retrieve(self, record: object) -> object:
                text = "Retrieved fixture text."
                return MODULE.RetrievalOutcome(
                    status="retrieved",
                    reason="ok",
                    http_status=200,
                    response_bytes=100,
                    content_length=len(text),
                    content_sha256=MODULE.sha256_text(text),
                    source_method="article_paragraphs",
                    final_domain="news.example",
                    text=text,
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "fulltext"
            first_input = root / "first.jsonl"
            second_input = root / "second.jsonl"
            base_record = {
                "article_id": "stable-id",
                "title": "Story",
                "published_utc": "2026-04-05T12:00:00Z",
            }
            first_input.write_text(
                json.dumps(
                    {
                        **base_record,
                        "article_url": "https://news.example/original",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            second_input.write_text(
                json.dumps(
                    {
                        **base_record,
                        "article_url": "https://news.example/reassigned",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            config = MODULE.RetrievalConfig(min_chars=1)
            MODULE.run(
                [first_input],
                config=config,
                max_documents=1,
                output_root=output,
                retriever=FakeRetriever(),
            )

            with self.assertRaisesRegex(ValueError, "earlier retrieval run"):
                MODULE.run(
                    [second_input],
                    config=config,
                    max_documents=1,
                    output_root=output,
                    retriever=FakeRetriever(),
                )

    def test_resume_repairs_interrupted_final_audit_line(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            valid = json.dumps(
                {"article_id": "complete", "status": "rejected"}
            )
            path.write_bytes(
                (valid + "\n" + '{"article_id":"partial"').encode("utf-8")
            )

            repaired = MODULE._repair_trailing_audit_fragment(path)
            latest = MODULE._latest_audit_entries(path)

        self.assertTrue(repaired)
        self.assertEqual(path.name, "manifest.jsonl")
        self.assertEqual(set(latest), {"complete"})


if __name__ == "__main__":
    unittest.main()
