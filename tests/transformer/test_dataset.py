import json
import random

import pytest

from transformer.dataset import build_dataset
from transformer.dataset.build_dataset import (
    build_samples, export_postgres, prepare_news_rows, select_news_evidence, write_dataset,
)
from transformer.inference import generate_draft
from transformer.inference import generate as inference_module
from transformer.preprocessing.prompts import (
    encode_input, encode_target, evidence_parts, make_input, parse_sections, render_prompt,
)
from transformer.training.data import load_splits


def papers(count=20):
    def paragraph(section, paper_id, sentence_count):
        templates = {
            "abstract": "Researchers summarize volunteer translation and collective participation in international listener communities for case {paper_id}, observation {index}.",
            "introduction": "Case {paper_id} examines how collective participation and volunteer translation connect distant listeners through shared cultural projects, question {index}.",
            "body": "Interview evidence for case {paper_id} describes coordinated translation projects and collective participation among international listeners during activity {index}.",
            "conclusion": "Findings from case {paper_id} suggest that volunteer translation and collective participation can sustain relationships among distant listeners, implication {index}.",
        }
        return " ".join(
            templates[section].format(paper_id=paper_id, index=index)
            for index in range(sentence_count)
        )

    return [
        {
            "paper_id": i,
            "title": f"Fandom research case {i}",
            "language": "en",
            "keywords": ["fandom", "digital participation"],
            "doi": f"10.1000/{i}",
            "content_hash": str(i),
            "abstract": paragraph("abstract", i, 3),
            "introduction": paragraph("introduction", i, 6),
            "body": paragraph("body", i, 12),
            "conclusion": paragraph("conclusion", i, 5),
        }
        for i in range(count)
    ]


def test_reproducible_paper_split_and_no_unrelated_news():
    rows = papers()
    first, _ = build_samples(rows)
    random.Random(9).shuffle(rows)
    second, report = build_samples(rows)
    assert first == second
    assert report["counts"] == {"train": 48, "validation": 6, "test": 6}
    ids = [{r["paper_id"] for r in first[s]} for s in ("train", "validation", "test")]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    for samples in first.values():
        for sample in samples:
            assert sample["input"]["news_evidence"] == ["[NO_NEWS_EVIDENCE]"]
            assert sample["section"] in {"Introduction", "Body", "Conclusion"}
            assert "Introduction:\n" not in sample["target"]
            assert "…" not in sample["target"]


def test_duplicate_aliases_stay_together_transitively():
    rows = papers()
    duplicate = {**rows[0], "paper_id": 100, "title": "An alternate publication title"}
    transitive = {**rows[1], "paper_id": 101, "title": duplicate["title"]}
    samples, report = build_samples(rows + [duplicate, transitive])
    containing = [r for split in samples.values() for r in split if "0" in r["source_paper_ids"]]
    assert len(containing) == 3
    assert all(
        set(sample["source_paper_ids"]) == {"0", "1", "100", "101"}
        for sample in containing
    )
    assert report["duplicates_grouped"] == 3


def test_missing_sections_and_non_english_are_rejected():
    rows = papers(5)
    rows[0]["conclusion"] = None
    rows[1]["language"] = "ko"
    samples, report = build_samples(rows)
    assert sum(map(len, samples.values())) == 9
    assert report["rejected"] == {
        "section_too_short_conclusion": 1,
        "non_english_or_unknown_language": 1,
    }


def test_export_and_loader_validate_snapshot(tmp_path):
    samples, report = build_samples(papers())
    write_dataset(samples, report, tmp_path, 42)
    assert load_splits(tmp_path) == samples
    with pytest.raises(ValueError, match="already exists"):
        write_dataset(samples, report, tmp_path, 42)
    validation = tmp_path / "validation.jsonl"
    validation.write_text(validation.read_text() + json.dumps(samples["train"][0]) + "\n")
    with pytest.raises(ValueError, match="leakage"):
        load_splits(tmp_path)


def test_manifest_rejects_tampering(tmp_path):
    samples, report = build_samples(papers())
    write_dataset(samples, report, tmp_path, 42)
    path = tmp_path / "train.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="manifest"):
        load_splits(tmp_path)


def test_postgres_export_is_read_only_and_schema_checked(monkeypatch):
    import pipeline.common.database as db
    from transformer.dataset.build_dataset import FIELDS
    queries, settings = [], []
    class Cursor:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def execute(self, sql): queries.append(sql)
        def fetchall(self):
            return [(name, "text") for name in FIELDS] if len(queries) == 1 else []
    class Connection:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def cursor(self): return Cursor()
    def connect(path, **kwargs):
        settings.append(kwargs)
        return Connection()
    monkeypatch.setattr(db, "connect", connect)
    export_postgres()
    assert settings == [{"read_only": True}]
    assert all(q.lstrip().startswith("SELECT") for q in queries)


def test_training_inference_share_formatter_and_keep_encoded_payload(monkeypatch):
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained("google/flan-t5-small", local_files_only=True)
    assert build_dataset.make_input is inference_module.make_input
    assert build_dataset.encode_input is inference_module.encode_input
    value = make_input(
        title="BLACKPINK and TikTok dance challenges",
        topic="cross-border dance participation",
        research_question="How do fans share BLACKPINK challenges?",
        paper_evidence=[
            "[PAPER 1]\nTitle: BLACKPINK dance research\nAuthors: Example Author\n"
            "DOI: 10.1/example\nEvidence:\nFans recreate BLACKPINK choreography "
            "and share dance videos across national borders."
        ],
        news_evidence=[
            "[NEWS 1]\nTitle: BLACKPINK dance videos spread\nSource: Example News\n"
            "URL: https://example.test/news\nEvidence:\nViewers share short "
            "BLACKPINK dance videos through social platforms."
        ],
        section="Body",
    )
    prompt = render_prompt(value)
    assert prompt.index("Title:") < prompt.index("Paper Evidence:") < prompt.index("News Evidence:")
    assert prompt.index("News Evidence:") < prompt.index("Instruction:") < prompt.index("Requested Section:")
    assert "Authors:" not in prompt and "URL:" not in prompt and "DOI:" not in prompt
    encoded = encode_input(tokenizer, value, 384)
    decoded = tokenizer.decode(encoded["input_ids"], skip_special_tokens=True)
    assert len(encoded["input_ids"]) <= 384
    assert "Requested Section: Body" in decoded
    assert "Fans recreate BLACKPINK choreography" in decoded
    assert "Viewers share short BLACKPINK dance videos" in decoded
    assert "[PAPER 1]" in decoded and "[NEWS 1]" in decoded

    target = "Fans coordinate dance participation across borders in online communities."
    target_ids = encode_target(tokenizer, target, 384, section="Body")
    assert len(target_ids) <= 384 and parse_sections(target) == {}
    assert evidence_parts(value["paper_evidence"][0], "PAPER")["evidence"]


def test_news_matching_requires_same_named_subject_and_target():
    news = prepare_news_rows([
        {
            "news_id": 1, "title": "BLACKPINK TikTok dance challenge spreads",
            "source": "Example News",
            "content": "BLACKPINK fans share TikTok dance challenges with viewers across borders and communities.",
        },
        {
            "news_id": 2, "title": "ITZY opens a YouTube channel",
            "source": "Example News",
            "content": "ITZY members opened a YouTube channel for their recent music videos and fans.",
        },
    ])
    row = {"title": "BLACKPINK TikTok dance challenges"}
    matched = select_news_evidence(
        row,
        "BLACKPINK fans share TikTok dance challenges across national borders.",
        news,
    )
    assert [item[1] for item in matched] == ["1"]
    assert select_news_evidence(
        row,
        "Independent researchers study unrelated educational institutions.",
        news,
    ) == []


def test_source_paper_split_and_sample_inference_shape(monkeypatch):
    samples, _ = build_samples(papers())
    split_ids = {
        split: {source for row in rows for source in row["source_paper_ids"]}
        for split, rows in samples.items()
    }
    assert not (split_ids["train"] & split_ids["validation"])
    assert not (split_ids["train"] & split_ids["test"])
    assert not (split_ids["validation"] & split_ids["test"])
    sample = samples["train"][0]
    assert sample["source_paper_id"] == sample["paper_id"]
    assert sample["section"] == sample["input"]["section"]
    assert sample["title"] == sample["input"]["title"]
    assert sample["paper_evidence"] == sample["input"]["paper_evidence"]
    assert parse_sections(sample["target"]) == {}
    assert sample["evidence_support_score"] >= 0.16
    from unittest.mock import Mock
    factory = Mock()
    factory.return_value.generate.return_value = "Raw draft"
    monkeypatch.setattr(inference_module, "_generator", factory)
    kwargs = {name: sample["input"][name] for name in (
        "title", "topic", "research_question", "paper_evidence", "news_evidence", "instruction",
    )}
    assert generate_draft(**kwargs) == "Raw draft"
    assert factory.return_value.generate.call_args.kwargs["paper_evidence"] == sample["paper_evidence"]
