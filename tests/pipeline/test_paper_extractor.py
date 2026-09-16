import unittest
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from pipeline.papers.cleaner.text import clean_pages
from pipeline.papers.extractor.sections import heading_kind, split_sections
from pipeline.papers.extractor.process import (
    EncryptedPDFError, minimum_exit_code, process_one, process_records, quality_flags,
)


class SectionTests(unittest.TestCase):
    def test_real_sections_exclude_references_and_notes(self):
        text = "Title\nAbstract\nSummary\n1. Introduction\nIntro content\n2. Methods\nMethod content\n3. Results\nResult content\n4. Conclusion\nClosing content\nNotes\nFootnote content\nReferences\nDo not train on bibliography"
        sections = split_sections(text)
        self.assertEqual(sections["introduction"], "Intro content")
        self.assertIn("Method content", sections["body"])
        self.assertIn("Result content", sections["body"])
        self.assertEqual(sections["conclusion"], "Closing content")
        self.assertNotIn("bibliography", sections["body"])
        self.assertNotIn("Footnote", sections["conclusion"])

    def test_missing_sections_are_not_invented(self):
        sections = split_sections("Title\n" + "This is an abstract without full text. " * 200)
        self.assertIsNone(sections["introduction"])
        self.assertIsNone(sections["body"])
        self.assertIsNone(sections["conclusion"])
        self.assertIn("missing_or_short_body", quality_flags({"title": "Title"}, "x" * 6000, sections))

    def test_multilingual_headings_and_sentence_false_positives(self):
        self.assertEqual(heading_kind("Ⅰ. 서론"), "introduction")
        self.assertEqual(heading_kind("V. KESIMPULAN"), "conclusion")
        self.assertEqual(heading_kind("CONCLUSION: FANDOM AS AGENT OF GLOBALIZATION"), "conclusion")
        self.assertIsNone(heading_kind("In conclusion, our findings show fans use social media."))
        self.assertIsNone(heading_kind("Results show that fans engage with social media."))
        self.assertIsNone(heading_kind("2012 marks the worldwide achievement of K-Pop"))
        self.assertIsNone(heading_kind("i Correspondence: email lmcardoso@institution.pt"))
        self.assertIsNone(heading_kind("P-ISSN: 2087-0442, E-ISSN: 2548-8309"))
        self.assertIsNone(heading_kind("ARMY"))

    def test_cleanup_retains_boundaries_and_removes_repeated_footer(self):
        text = clean_pages(["Journal of Fandom\nIntroduction\nSocial me-\ndia matters\n1",
                            "Journal of Fandom\nMethods\nInterviews\n2",
                            "Journal of Fandom\nConclusion\nFans participate\n3"])
        self.assertNotIn("Journal of Fandom", text)
        self.assertIn("Introduction\nSocial media matters", text)
        self.assertIn("Conclusion", text)

    def test_source_mismatch_is_not_eligible(self):
        sections = {"introduction": "a" * 350, "body": "b" * 1200, "conclusion": "c" * 300}
        flags = quality_flags({"title": "K-pop international fandom communities"}, "unrelated " * 700, sections)
        self.assertIn("source_title_mismatch", flags)

    def test_letter_headings_and_footnote_marker(self):
        self.assertEqual(heading_kind("A. INTRODUCTION"), "introduction")
        self.assertEqual(heading_kind("D. CONCLUSION"), "conclusion")
        self.assertEqual(heading_kind("1. Introduction 1"), "introduction")
        self.assertEqual(heading_kind("4. KESIMPULAN DAN SARAN"), "conclusion")
        self.assertEqual(heading_kind("1. PREFACE"), "introduction")

    def test_numbered_prose_and_statistics_do_not_split_sections(self):
        for line in ("1. K-pop fans are actively participating in fan economy.",
                     "58.3 % had experienced a shift in ways of their", "2 = .03.",
                     "25 Oct 2024", "1 shows the details of the selected account for this study.",
                     "method.", "3 porque te gusta un idol, because you like an idol"):
            self.assertIsNone(heading_kind(line), line)

    def test_conclusion_keeps_bullets_until_backmatter(self):
        text = ("1. Introduction\nIntro\n2. Methods\nBody\n3. Conclusion\nClosing\n"
                "1. Consumer Participation\nFirst point\n2. Recommendations\nSecond point\n"
                "References\nBibliography")
        result = split_sections(text)
        self.assertIn("First point", result["conclusion"])
        self.assertIn("Second point", result["conclusion"])
        self.assertNotIn("Bibliography", result["conclusion"])

    def test_introduction_subsections_and_inline_abstract_preserved(self):
        text = ("Abstract: Actual first sentence\nRest of summary\nKeywords: fandom\n"
                "1. Introduction\nStart\n1.1. Study Scope\nScope content\n"
                "2. Methods\nMethods content\n3. Conclusion\nClosing")
        result = split_sections(text)
        self.assertEqual(result["abstract_extracted"], "Actual first sentence\nRest of summary")
        self.assertIn("Scope content", result["introduction"])
        self.assertNotIn("Scope content", result["body"])

    def test_repeated_header_whitespace_is_normalized_consistently(self):
        pages = ["Author Name\nARTICLE TITLE:  K-POP AND FANDOM\n" + content + "\nJournal 2024     " + str(i)
                 for i, content in enumerate(("Introduction\nIntro text", "Methods\nBody text", "Conclusion\nClosing text"), 1)]
        text = clean_pages(pages)
        self.assertNotIn("ARTICLE TITLE", text)
        self.assertNotIn("Journal", text)

    def test_canonical_headings_and_uncertain_heading(self):
        names = {
            "Related Work": "literature_review", "Previous Studies": "literature_review",
            "Background and Related Work": "literature_review", "Experimental Results": "results",
            "Discussion and Implications": "discussion", "Conclusion and Future Work": "conclusion",
            "Supplementary Material": "appendix", "Literature Cited": "references",
            "3. Methods": "methodology", "4. Unknown Topic": "other",
        }
        for heading, expected in names.items():
            self.assertEqual(heading_kind(heading), expected, heading)

    def test_references_and_appendix_are_preserved_separately(self):
        text = ("Abstract\nSummary\n1. Introduction\nIntro\n2. Methods\nMethods text\n"
                "3. Results\nResults text\n4. Conclusion\nClosing\nReferences\nSmith 2024\n"
                "Appendix A\nSupplementary table")
        result = split_sections(text)
        self.assertIn("Smith 2024", result["references_text"])
        self.assertIn("Supplementary table", result["appendix_text"])
        self.assertNotIn("Smith 2024", result["body"])
        self.assertNotIn("Supplementary table", result["conclusion"])

    def test_caption_is_separate_but_discussion_sentence_remains(self):
        text = ("1. Introduction\nIntro\n2. Results\nFigure 1. Fandom participation\n"
                "Figure 3 shows that fan engagement increased (Kim, 2024).\n3. Conclusion\nClosing")
        result = split_sections(text)
        self.assertEqual(len(result["captions"]), 1)
        self.assertIn("Figure 3 shows", result["body"])
        self.assertNotIn("Figure 1. Fandom", result["body"])

    def test_short_caption_and_its_source_are_not_main_body(self):
        result = split_sections("1. Introduction\nIntro\n2. Results\nTable 3.\n"
                                "Source: Survey data\nEngagement increased in the second wave.\n"
                                "3. Conclusion\nClosing")
        self.assertEqual(len(result["captions"]), 2)
        self.assertNotIn("Survey data", result["body"])
        self.assertIn("Engagement increased", result["body"])

    def test_case_study_heading_separate_and_analysis_retained(self):
        result = split_sections("1. Introduction\nIntro\n2. Methods\nMethod text\n"
                                "3. Case Study: BTS\nBTS developed a global fan relationship.\n"
                                "4. Conclusion\nClosing")
        case = next(s for s in result["sections"] if "Case Study" in s["heading"])
        self.assertEqual(case["section_type"], "other")
        self.assertNotIn("Case Study", case["text"])
        self.assertIn("BTS developed", result["body"])

    def test_hyphenation_real_hyphen_and_wrapped_line(self):
        text = clean_pages(["Introduction\nAn inter-\nnational K-pop audience engages with fans\n"
                            "across multiple countries. Cross-cultural exchange remains valuable."])
        self.assertIn("international K-pop", text)
        self.assertIn("fans across", text)
        self.assertIn("Cross-cultural", text)

    def test_duplicate_paragraph_is_removed_without_short_line_loss(self):
        paragraph = "This paragraph describes a distinct finding about digital fandom participation. " * 2
        audit = {}
        result = clean_pages([paragraph, paragraph, "Conclusion\nImportant result."], audit)
        self.assertEqual(result.count(paragraph.strip()), 1)
        self.assertEqual(audit["duplicate_paragraphs_removed"], 1)
        self.assertIn("Important result", result)

    def test_subsection_inherits_known_canonical_type(self):
        result = split_sections("4. Discussion\nDiscussion text\n4.2 Global Fandom Activities\n"
                                "A meaningful analysis sentence.")
        self.assertEqual(result["sections"][1]["section_type"], "discussion")

    def test_encrypted_pdf_uses_html_fallback_without_replacing_original(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root / "documents" / "W1"
            directory.mkdir(parents=True)
            original = directory / "source.pdf"
            original.write_bytes(b"%PDF-test-original")
            fallback = directory / "fallback.html"
            fallback.write_text("<html>alternative</html>", encoding="utf-8")
            source = {"format": "pdf", "filename": "source.pdf",
                      "sha256": hashlib.sha256(original.read_bytes()).hexdigest()}
            (directory / "source.json").write_text(json.dumps(source), encoding="utf-8")
            paper = {"id": "W1", "title": "Fandom participation study", "abstract": "English abstract. " * 30,
                     "language": "en", "source_locations": []}
            article = ("Abstract\n" + "English abstract. " * 30 + "\n1. Introduction\n"
                       + "Fandom participation study. " * 30 + "\n2. Methods\n"
                       + "Participants discussed fandom. " * 90 + "\n3. Results\n"
                       + "Fans engaged globally. " * 90 + "\n4. Conclusion\n"
                       + "Fandom communities persisted. " * 30)
            with patch("pipeline.papers.extractor.process.read_source",
                       side_effect=[EncryptedPDFError("encrypted"), [article]]), \
                 patch("pipeline.papers.extractor.process.alternative_source",
                       return_value=(fallback, {"format": "html", "sha256": "fallbackhash"})):
                record = process_one(paper, root, root / "processed")
            self.assertEqual(original.read_bytes(), b"%PDF-test-original")
            self.assertTrue(record["encrypted_pdf"])
            self.assertTrue(record["html_fallback_used"])
            self.assertEqual(record["fulltext_status"], "text_extracted")

    def test_encrypted_without_fallback_is_one_rejected_record(self):
        papers = [{"id": "W1"}, {"id": "W2"}]
        with patch("pipeline.papers.extractor.process.process_one",
                   side_effect=[EncryptedPDFError("encrypted"),
                                {"id": "W2", "fulltext_status": "text_extracted", "quality_flags": [],
                                 "training_eligible": True, "english_training_eligible": True,
                                 "quality_state": "ready", "quality_score": 1.0}]):
            records, failures = process_records(papers, Path("."), Path("."))
        self.assertEqual(len(failures), 1)
        self.assertEqual(records[0]["failure_reason"], "encrypted_pdf_no_fallback")
        self.assertEqual(records[1]["quality_state"], "ready")

    def test_minimum_is_met_despite_two_document_failures(self):
        records = [{"english_training_eligible": True} for _ in range(98)] + [
            {"english_training_eligible": False} for _ in range(2)]
        self.assertEqual(minimum_exit_code(records, 50), 0)

    def test_minimum_fails_when_only_forty_eligible(self):
        records = [{"english_training_eligible": True} for _ in range(40)] + [
            {"english_training_eligible": False} for _ in range(60)]
        self.assertEqual(minimum_exit_code(records, 50), 1)


if __name__ == "__main__":
    unittest.main()
