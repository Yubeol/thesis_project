import json
import random

import pytest

from transformer.dataset.build_dataset import build_samples, export_postgres, write_dataset
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
