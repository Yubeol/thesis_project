from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from pipeline.news.collector.rss import collect, parse_feed, topics
from pipeline.news.cleaner.process import clean_article_text, process_one
from pipeline.news.deduplicator.records import canonical_url, deduplicate, news_id

NOW = datetime(2026, 9, 15, tzinfo=timezone.utc)
SOURCE = {"name": "Publisher", "language": "ko", "kpop_source": False, "url": "https://example.org/feed"}
RSS = '''<rss xmlns:content="http://purl.org/rss/1.0/modules/content/"><channel><item>
<title>케이팝 글로벌 팬덤과 유튜브</title><link>https://example.org/news/1?utm_source=rss</link>
<pubDate>Mon, 14 Sep 2026 15:00:00 +0900</pubDate><description>기사 요약만 있음</description>
</item></channel></rss>'''.encode("utf-8")


class NewsTests(unittest.TestCase):
    def test_original_fields_and_timezone_preserved(self):
        row = parse_feed(RSS, SOURCE, SOURCE["url"], NOW, 365)[0]
        self.assertEqual(row["title_original"], "케이팝 글로벌 팬덤과 유튜브")
        self.assertEqual(row["published_at"], "2026-09-14T06:00:00+00:00")
        self.assertIsNone(row["content_original"])
        self.assertIsNone(row["content_en_for_rag"])

    def test_old_and_future_items_are_excluded(self):
        self.assertEqual(parse_feed(RSS.replace(b"2026", b"2020"), SOURCE, SOURCE["url"], NOW, 365), [])
        self.assertEqual(parse_feed(RSS.replace(b"2026", b"2027"), SOURCE, SOURCE["url"], NOW, 365), [])

    def test_invalid_feed_items_are_counted(self):
        stats = {}
        self.assertEqual(parse_feed(RSS.replace(b"Mon, 14 Sep 2026 15:00:00 +0900", b"invalid date"), SOURCE, SOURCE["url"], NOW, 365, stats=stats), [])
        self.assertEqual(stats["invalid"], 1)
        self.assertEqual(stats["items"], 1)

    def test_dedup_preserves_alternate_sources(self):
        first = parse_feed(RSS, SOURCE, SOURCE["url"], NOW, 365)[0]
        second = {**first, "id": news_id("https://other.org/news"), "source": "Other", "url": "https://other.org/news"}
        records, duplicates = deduplicate([first, second])
        self.assertEqual(len(records), 1)
        self.assertEqual(len(duplicates), 1)
        self.assertEqual(records[0]["duplicate_sources"][0]["source"], "Other")
        self.assertEqual(canonical_url("https://example.org/news?a=1&utm_source=x#top"), "https://example.org/news?a=1")

    def test_identical_reruns_do_not_add_self_duplicate_sources(self):
        row = parse_feed(RSS, SOURCE, SOURCE["url"], NOW, 365)[0]
        records, _ = deduplicate([row, row, row])
        self.assertEqual(records[0]["duplicate_sources"], [])

    def test_dedup_removes_same_content_with_different_headlines(self):
        first = {**parse_feed(RSS, SOURCE, SOURCE["url"], NOW, 365)[0], "content_original": "The same syndicated K-pop article."}
        second = {**first, "id": news_id("https://other.org/news"), "source": "Other", "url": "https://other.org/news", "title_original": "A different headline"}
        records, duplicates = deduplicate([first, second])
        self.assertEqual(len(records), 1)
        self.assertEqual(len(duplicates), 1)

    def test_unrelated_news_is_excluded(self):
        self.assertEqual(topics("Global oil market and consumption", kpop_source=False), [])
        self.assertIn("global_market", topics("K-pop global fandom", kpop_source=False))

    def test_rss_summary_is_not_misrepresented_as_fulltext(self):
        row = parse_feed(RSS, SOURCE, SOURCE["url"], NOW, 365)[0]
        policy = Mock()
        policy.read.side_effect = ValueError("Source unavailable")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                process_one(row, Path(directory), policy)

    def test_korean_original_is_not_overwritten_with_english(self):
        row = parse_feed(RSS, SOURCE, SOURCE["url"], NOW, 365)[0]
        row["rss_content_html"] = "<p>케이팝 글로벌 팬덤과 유튜브. " + "팬덤은 유튜브를 통해 해외 팬들과 소통하고 문화를 나눈다. " * 20 + "</p>"
        row["rss_content_is_full"] = True
        with tempfile.TemporaryDirectory() as directory:
            result = process_one(row, Path(directory), Mock())
        self.assertIn("팬덤", result["content_original"])
        self.assertIsNone(result["content_en_for_rag"])
        self.assertEqual(result["translation_status"], "pending")

    def test_colon_timezone_offsets_are_preserved(self):
        row = parse_feed(RSS.replace(b"+0900", b"+09:00"), SOURCE, SOURCE["url"], NOW, 365)[0]
        self.assertEqual(row["published_at"], "2026-09-14T06:00:00+00:00")

    def test_unverified_encoded_snippet_requires_publisher_page(self):
        row = parse_feed(RSS, SOURCE, SOURCE["url"], NOW, 365)[0]
        row["rss_content_html"] = "<p>Unverified RSS teaser</p>"
        policy = Mock()
        policy.read.side_effect = ValueError("Source unavailable")
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            process_one(row, Path(directory), policy)
        policy.read.assert_called_once_with(row["url"])

    def test_category_requirement_excludes_drama_fans(self):
        source = {**SOURCE, "kpop_source": True, "require_any_category": ["Music", "kpop"]}
        self.assertEqual(parse_feed(RSS, source, source["url"], NOW, 365), [])

    def test_cleaning_removes_duplicate_title_and_viki_ad(self):
        row = {"title_original": "K-pop fandom", "source": "Soompi"}
        text = "K-pop fandom\nFans attend a global concert.\nWatch the new drama on Viki below:\nOr check out another drama below:"
        self.assertEqual(clean_article_text(text, row), "Fans attend a global concert.")

    def test_repeated_run_is_cached_and_unique(self):
        config = {"feeds": [SOURCE], "minimum": 1, "target": 300, "lookback_days": 365}
        with tempfile.TemporaryDirectory() as directory, \
                patch("pipeline.news.collector.rss.fetch", return_value=(RSS, "application/rss+xml", SOURCE["url"])) as fetch, \
                patch("pipeline.news.collector.rss.time.sleep"):
            collect(config, Path(directory), now=NOW)
            report = collect(config, Path(directory), now=NOW)
            self.assertEqual(fetch.call_count, 1)
            self.assertEqual(report["selected_count"], 1)
            self.assertEqual(report["duplicates"], 0)
            self.assertEqual(len(json.loads((Path(directory) / "news.json").read_text(encoding="utf-8"))), 1)


if __name__ == "__main__":
    unittest.main()
