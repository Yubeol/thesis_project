import unittest

from pipeline.papers.cleaner.text import clean_pages
from pipeline.papers.extractor.sections import heading_kind, split_sections
from pipeline.papers.extractor.process import quality_flags


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


if __name__ == "__main__":
    unittest.main()
