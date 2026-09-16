import unittest
from unittest.mock import Mock, patch

from pipeline.common.entity_resolver import (
    EntityResolver, protect_official_names, resolve_record,
    restore_official_names,
)
from pipeline.news.translator.translate import translate_record
from rag.graph.sync.entity_sync import merge_entity_mentions
from rag.graph.sync.synchronizer import _normalize_authors, _resolver_or_empty


def work_entry(kind="Work", authority="https://example.org/works/123"):
    return {
        "entity_type": kind, "authority_url": authority,
        "name_ko": "들쥐", "name_en": "Mouse Trap",
        "aliases": ["들쥐", "Mouse Trap"],
        "source_urls": [authority],
    }


class EntityResolutionTests(unittest.TestCase):
    def test_real_official_group_aliases_share_id(self):
        resolver = EntityResolver.from_file()
        korean = resolver.resolve("방탄소년단", "ko", "Group")
        english = resolver.resolve("BTS", "en", "Group")
        self.assertEqual(korean["entity_id"], english["entity_id"])
        self.assertEqual(korean["name_en"], "BTS")
        self.assertEqual(korean["resolution_status"], "resolved")

    def test_verified_work_aliases_share_id(self):
        resolver = EntityResolver([work_entry()])
        self.assertEqual(resolver.resolve("들쥐", "ko", "Work")["entity_id"],
                         resolver.resolve("Mouse Trap", "en", "Work")["entity_id"])

    def test_machine_translation_never_becomes_official_name(self):
        resolver = EntityResolver([])
        value = resolver.resolve("들쥐", "ko", "Work", translated_form="Field Mouse")
        self.assertIsNone(value["entity_id"])
        self.assertIsNone(value["name_en"])
        self.assertEqual(value["name_ko"], "들쥐")
        self.assertEqual(value["resolution_status"], "translated_only")

    def test_unknown_quoted_korean_work_remains_unresolved(self):
        result = EntityResolver([]).resolve_document(
            title="신작 ‘들쥐’ 공개", text="들쥐가 화제다.", language="ko"
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name_ko"], "들쥐")
        self.assertIsNone(result[0]["name_en"])
        self.assertEqual(result[0]["resolution_status"], "unresolved")

    def test_same_text_different_entity_types_is_ambiguous(self):
        resolver = EntityResolver([work_entry(), work_entry("Album", "https://example.org/albums/456")])
        self.assertIsNone(resolver.resolve("들쥐", "ko")["entity_id"])
        self.assertNotEqual(resolver.resolve("들쥐", "ko", "Work")["entity_id"],
                            resolver.resolve("들쥐", "ko", "Album")["entity_id"])
        detected = resolver.resolve_document(title="들쥐", text="", language="ko")
        self.assertTrue(all(item["entity_id"] is None for item in detected))

    def test_resolver_failure_preserves_original_news(self):
        resolver = Mock()
        resolver.resolve_document.side_effect = RuntimeError("failed")
        original = {"title_original": "들쥐", "content_original": "한국어 원문",
                    "original_language": "ko", "entities": [{"surface_form": "들쥐", "entity_type": "Work"}]}
        result = resolve_record(original, "news", resolver)
        self.assertEqual(result["title_original"], original["title_original"])
        self.assertEqual(result["content_original"], original["content_original"])
        self.assertEqual(result["entities"][0]["resolution_status"], "unresolved")
        self.assertEqual(result["entity_resolution_status"], "failed")

    def test_graph_can_keep_base_sync_when_catalogue_is_unavailable(self):
        with patch.object(EntityResolver, "from_file", side_effect=OSError):
            self.assertEqual(_resolver_or_empty().entries, [])

    def test_protected_translation_restores_verified_name_and_original(self):
        entity = EntityResolver([work_entry()]).resolve("들쥐", "ko", "Work")
        title = "들쥐 소식"
        body = "들쥐는 인기 작품이다."
        protected_title, protected_body, markers = protect_official_names(title, body, [entity])
        self.assertNotIn("들쥐", protected_title + protected_body)
        self.assertEqual(restore_official_names(protected_title, protected_body, markers,
                                                protected_title, protected_body),
                         ("Mouse Trap 소식", "Mouse Trap는 인기 작품이다."))
        with self.assertRaises(ValueError):
            restore_official_names("missing", protected_body, markers, protected_title, protected_body)

    def test_translation_interface_uses_protected_markers(self):
        entity = EntityResolver([work_entry()]).resolve("들쥐", "ko", "Work")
        record = {"title_original": "들쥐 소식", "content_original": "들쥐의 새 소식",
                  "original_language": "ko", "entities": [entity]}
        def reply(**kwargs):
            self.assertIn("ENTITYPROTECTTOKEN", kwargs["title"])
            self.assertIn("ENTITYPROTECTTOKEN", kwargs["content"])
            return {"title": kwargs["title"], "content": kwargs["content"]}
        with patch("pipeline.news.translator.translate.default_resolver",
                   return_value=EntityResolver([work_entry()])), patch(
            "pipeline.news.translator.translate.OpenAI"), patch(
            "pipeline.news.translator.translate.request_translation", side_effect=reply
        ):
            result = translate_record(record, model="test", retries=1)
        self.assertEqual(result["title_original"], "들쥐 소식")
        self.assertEqual(result["content_original"], "들쥐의 새 소식")
        self.assertIn("Mouse Trap", result["title_en_for_rag"])
        self.assertIn("Mouse Trap", result["content_en_for_rag"])

    def test_graph_merges_on_id_for_paper_and_news(self):
        entity = EntityResolver([work_entry()]).resolve("들쥐", "ko", "Work")
        driver = Mock()
        with patch("rag.graph.sync.entity_sync.get_neo4j_driver", return_value=driver), patch(
            "rag.graph.sync.entity_sync.get_database_name", return_value="neo4j"
        ):
            merge_entity_mentions("Paper", 1, [entity])
            merge_entity_mentions("News", 2, [entity])
        self.assertEqual(driver.execute_query.call_count, 2)
        first = driver.execute_query.call_args_list[0]
        second = driver.execute_query.call_args_list[1]
        self.assertIn("MERGE (e:Entity {entity_id: $entity_id})", first.args[0])
        self.assertIn("MERGE (e:Entity {entity_id: $entity_id})", second.args[0])
        self.assertEqual(first.kwargs["entity_id"], second.kwargs["entity_id"])

    def test_existing_paper_author_json_is_split_for_graph(self):
        self.assertEqual(_normalize_authors('["First Author", "Second Author"]'),
                         ["First Author", "Second Author"])
        self.assertEqual(_normalize_authors("First Author; Second Author"),
                         ["First Author", "Second Author"])


if __name__ == "__main__":
    unittest.main()
