from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from pipeline.common.http import FetchError, fetch, retry_delay
from pipeline.papers.collector.download import download_one, source_format, SourceLinks
from pipeline.papers.collector.openalex import (
    abstract_text, collect, normalize_doi, normalize_work, select_unique,
)


def work(identifier="W1", doi="https://doi.org/10.1234/test", title="K-pop fandom and social media"):
    return {"id": "https://openalex.org/" + identifier, "doi": doi, "title": title,
            "publication_year": 2023, "type": "article", "language": "en",
            "open_access": {"is_oa": True, "oa_status": "gold"},
            "authorships": [{"author": {"display_name": "Test Author"}}],
            "abstract_inverted_index": {"Fans": [0], "use": [1], "TikTok": [2]},
            "best_oa_location": {"is_oa": True, "pdf_url": "https://example.org/test.pdf",
                                 "license": "cc-by", "version": "publishedVersion"}}


class PaperCollectorTests(unittest.TestCase):
    def test_doi_and_abstract(self):
        self.assertEqual(normalize_doi("https://dx.doi.org/10.1234/ABC"), "10.1234/abc")
        self.assertIsNone(normalize_doi("not-a-doi"))
        self.assertEqual(abstract_text({"fan": [0, 2], "activity": [1]}), "fan activity fan")
        self.assertIsNone(abstract_text(None))

    def test_relevance_and_exclusions(self):
        self.assertIsNone(normalize_work(work(title="The future of social media marketing"), "K-pop"))
        self.assertIsNone(normalize_work(work(title="BTS clinical trial"), "K-pop"))
        self.assertIsNotNone(normalize_work(work(title="Korean wave and global fan communities"), "K-pop"))
        retracted = work()
        retracted["is_retracted"] = True
        self.assertIsNone(normalize_work(retracted, "K-pop"))
        future = work()
        future["publication_year"] = 9999
        self.assertIsNone(normalize_work(future, "K-pop"))

    def test_preserves_missing_sections_and_sources(self):
        row = normalize_work(work(), "K-pop")
        for field in ("introduction", "body", "conclusion"):
            self.assertIsNone(row[field])
        self.assertEqual(row["source_url"], "https://doi.org/10.1234/test")
        self.assertEqual(row["source_locations"][0]["license"], "cc-by")
        self.assertEqual(row["authors"], ["Test Author"])

    def test_deduplicates_id_doi_and_title_with_query_provenance(self):
        rows = [normalize_work(work(), "a"), normalize_work(work(), "b"),
                normalize_work(work("W2", "10.1234/other", "K POP fandom and social media!"), "c")]
        selected = select_unique(rows, 70)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["matched_queries"], ["a", "b", "c"])

    def test_two_runs_use_cache_and_do_not_append_duplicates(self):
        config = {"queries": ["K-pop"], "target": 70, "minimum": 1,
                  "page_size": 100, "pages_per_query": 1, "open_access_only": True}
        response = {"results": [work(), work()], "meta": {"next_cursor": None}}
        with tempfile.TemporaryDirectory() as directory, \
                patch("pipeline.papers.collector.openalex.get_json", return_value=response) as api, \
                patch("pipeline.papers.collector.openalex.time.sleep"):
            output = Path(directory)
            first = collect(config, output)
            second = collect(config, output)
            self.assertEqual(api.call_count, 1)
            self.assertEqual(first["selected_count"], 1)
            self.assertEqual(second["cached_pages"], 1)
            self.assertEqual(len(json.loads((output / "papers.json").read_text())), 1)
            self.assertEqual(first["training_ready_count"], 0)

    def test_partial_api_failure_is_reported(self):
        config = {"queries": ["first", "second"], "target": 70, "minimum": 1,
                  "page_size": 100, "pages_per_query": 1, "open_access_only": True}
        with tempfile.TemporaryDirectory() as directory, \
                patch("pipeline.papers.collector.openalex.get_json", side_effect=[
                    {"results": [work()]}, FetchError("HTTP 429 from api.openalex.org")]), \
                patch("pipeline.papers.collector.openalex.time.sleep"):
            report = collect(config, Path(directory))
            self.assertEqual(report["selected_count"], 1)
            self.assertEqual(len(report["errors"]), 1)
            self.assertTrue(report["metadata_minimum_met"])

    def test_http_failure_does_not_leak_secret_query(self):
        url = "https://example.org/data?api_key=secret-value"
        failure = HTTPError(url, 401, "unauthorized", {}, BytesIO())
        with patch("pipeline.common.http.urlopen", side_effect=failure):
            with self.assertRaises(FetchError) as caught:
                fetch(url)
        self.assertNotIn("secret-value", str(caught.exception))
        self.assertIn("401", str(caught.exception))

    def test_rate_limit_retries_are_bounded(self):
        failure = HTTPError("https://example.org", 429, "limited", {"Retry-After": "100"}, BytesIO())
        with patch("pipeline.common.http.urlopen", side_effect=failure) as request, \
                patch("pipeline.common.http.time.sleep") as sleep:
            with self.assertRaises(FetchError):
                fetch("https://example.org")
            self.assertEqual(request.call_count, 3)
            self.assertEqual(sleep.call_count, 2)
            self.assertEqual(sleep.call_args.args[0], 30)
        self.assertEqual(retry_delay("invalid", 2), 4)

    def test_download_does_not_treat_challenge_html_as_pdf(self):
        self.assertEqual(source_format(b"%PDF-1.7\nexample", "application/octet-stream"), "pdf")
        with self.assertRaises(FetchError):
            source_format(b"<title>Just a moment...</title>" + b" " * 600, "text/html")
        with self.assertRaises(FetchError):
            source_format(b"not a PDF", "application/pdf")

    def test_download_cache_hash_and_original_source(self):
        from unittest.mock import Mock
        policy = Mock()
        policy.read.return_value = (b"%PDF-1.7\nfixture", "application/pdf", "https://example.org/final.pdf")
        paper = normalize_work(work(), "K-pop")
        with tempfile.TemporaryDirectory() as directory:
            first = download_one(paper, Path(directory), policy)
            second = download_one(paper, Path(directory), policy)
            self.assertEqual(policy.read.call_count, 1)
            self.assertTrue(second["cache_hit"])
            self.assertEqual(first["status"], "downloaded_unverified")
            self.assertEqual(first["source_url"], "https://example.org/test.pdf")
            self.assertEqual(first["license"], "cc-by")
            (Path(directory) / "documents" / "W1" / "source.pdf").write_bytes(b"damaged")
            download_one(paper, Path(directory), policy)
            self.assertEqual(policy.read.call_count, 2)

    def test_download_failure_is_explicit(self):
        from unittest.mock import Mock
        policy = Mock()
        policy.read.side_effect = FetchError("HTTP 403 from example.org")
        with tempfile.TemporaryDirectory() as directory:
            result = download_one(normalize_work(work(), "K-pop"), Path(directory), policy)
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(len(result["attempts"]), 1)

    def test_source_pdf_links_include_ojs_download_and_embedded_document(self):
        parser = SourceLinks()
        parser.feed('<a href="/article/download/7/2">PDF</a><iframe src="/paper.pdf"></iframe><a href="/about">About</a>')
        self.assertEqual(parser.pdfs, ["/article/download/7/2", "/paper.pdf"])


if __name__ == "__main__":
    unittest.main()
