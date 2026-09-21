from transformer.dataset.build_conclusion_service import (
    clean_candidate,
    conclusion_gate,
    select_evidence,
    teacher_cache_key,
)
from transformer.preprocessing.prompts import encode_input, make_input


def test_noise_filter_removes_reference_fragments():
    reference = {"content": "References https://example.test/doi 10.1111/xyz " * 6}
    assert clean_candidate(reference)[0] is None
    assert clean_candidate({"content": "95-108 Author Name"})[0] is None


def test_evidence_selection_requires_distinct_relevant_papers_and_drops_unrelated_news():
    body = "Online fan communities share translated media and coordinate discussion about cross-border K-pop fandom participation. " * 2
    other = "Transnational fandom participation depends on volunteer translation groups that explain cultural context to distant audiences and organize multilingual conversations among fans. " * 2
    retrieval = {
        "papers": [
            {"paper_id": 11, "title": "Online fandom translation communities", "content": body, "similarity": 0.81},
            {"paper_id": 12, "title": "Transnational fandom participation", "content": other, "similarity": 0.76},
        ],
        "news": [{"title_en": "A new album tops a domestic chart", "content": "A new album topped a domestic chart this week. " * 4, "similarity": 0.48}],
    }
    row = {"title": "Online fandom translation", "topic": "transnational participation", "research_question": "How does online fan translation support transnational participation?"}
    papers, news, report = select_evidence(retrieval, row, "train", {"11": "train", "12": "train"})
    assert len(papers) == 2
    assert news == []
    assert report["distinct_paper_ids"] == 2


def test_teacher_cache_uses_final_visible_input():
    assert teacher_cache_key("Paper Evidence: A", "gpt-4o-mini") != teacher_cache_key("Paper Evidence: B", "gpt-4o-mini")


def test_conclusion_gate_rejects_unsupported_number_and_organization():
    class Tokenizer:
        def __call__(self, value, **_kwargs):
            return {"input_ids": list(range(len(value.split()) + 1))}

        def encode(self, value, **_kwargs):
            return list(range(len(value.split()) + 1))

    row = {
        "title": "Online fandom translation",
        "topic": "transnational fan participation",
        "research_question": "How does fan translation support transnational participation?",
        "input": {"title": "Online fandom translation", "topic": "transnational fan participation", "research_question": "How does fan translation support transnational participation?",
                  "paper_evidence": ["[PAPER 1]\nTitle: Fandom translation\nEvidence: Fans translate media and share it across borders."],
                  "news_evidence": ["[NO_NEWS_EVIDENCE]"], "section": "Conclusion"},
    }
    target = (
        "The evidence indicates that fan translation supports transnational participation by helping fans share media across borders. "
        "Harvard University found that 73% of fans translate media, although the supplied evidence provides no such survey. "
        "Overall, translation can connect participants, but the unsupported survey cannot be used as a conclusion."
    )
    reasons = conclusion_gate(row, target, Tokenizer(), "Paper Evidence: Fans translate media and share it across borders.")
    assert "unsupported_number" in reasons
    assert "unsupported_entity" in reasons
