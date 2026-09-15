from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pipeline.common.database import connect, DatabaseConfigurationError
from pipeline.common.loading import prepare_record, utc_timestamp, validate_schema, LoadValidationError, TABLES


class LoadingTests(unittest.TestCase):
    def test_paper_keeps_sections_and_does_not_use_source_id_as_primary_key(self):
        record = {"id": "W1", "title": "K-pop fandom", "authors": ["Kim, A", "Lee B"],
                  "source_url": "https://doi.org/10.1234/test", "doi": "https://doi.org/10.1234/TEST",
                  "introduction": "Introduction", "body": "Actual paper body", "conclusion": "Conclusion"}
        row = prepare_record("papers", record)
        self.assertNotIn("paper_id", row)
        self.assertNotIn("id", row)
        self.assertEqual(json.loads(row["authors"]), ["Kim, A", "Lee B"])
        self.assertEqual(row["doi"], "10.1234/test")
        self.assertEqual(row["body"], "Actual paper body")
        self.assertEqual(len(row["content_hash"]), 64)

    def test_incomplete_korean_translation_cannot_enter_ready_news_table(self):
        row = {"id": "news_1", "original_language": "ko", "title_original": "한국어 원문",
               "content_original": "한국어 기사", "url": "https://example.org/news", "content_status": "fulltext_extracted"}
        with self.assertRaisesRegex(LoadValidationError, "English RAG"):
            prepare_record("news", row)
        row.update(title_en_for_rag="Original title", content_en_for_rag="Translated article",
                   published_at="2026-09-15T13:00:00+09:00", collected_at="2026-09-15T14:00:00+09:00")
        result = prepare_record("news", row)
        self.assertEqual(result["content_original"], "한국어 기사")
        self.assertEqual(result["published_at"], datetime(2026, 9, 15, 4))

    def test_timestamp_does_not_silently_assume_local_timezone(self):
        with self.assertRaises(LoadValidationError):
            utc_timestamp("2026-09-15T12:00:00")

    def test_values_are_not_silently_truncated(self):
        with self.assertRaisesRegex(LoadValidationError, "length"):
            prepare_record("papers", {"title": "x" * 501, "source_url": "https://example.org"})

    def test_bad_extraction_is_quarantined(self):
        with self.assertRaisesRegex(LoadValidationError, "review"):
            prepare_record("papers", {"title": "Example", "source_url": "https://example.org", "quality_flags": ["source_title_mismatch"]})

    def test_schema_change_stops_loader(self):
        fields = {name: {"type": kind, "not_null": False, "default": None} for name, kind in TABLES["papers"]["types"].items()}
        fields["paper_id"] = {"type": "bigint", "not_null": True, "default": "nextval('example')"}
        schema = {"columns": fields, "constraints": [("p", "PRIMARY KEY (paper_id)")], "can_select": True}
        validate_schema("papers", schema)
        fields["title"]["type"] = "character varying(100)"
        with self.assertRaisesRegex(LoadValidationError, "schema mismatch"):
            validate_schema("papers", schema)

    def test_missing_credentials_does_not_connect(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {}, clear=True), patch("pipeline.common.database.psycopg.connect") as db:
            with self.assertRaises(DatabaseConfigurationError):
                connect(Path(directory) / "absent.env")
            db.assert_not_called()


if __name__ == "__main__":
    unittest.main()
