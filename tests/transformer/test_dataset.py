import json
import random

import pytest

from transformer.dataset.build_dataset import build_samples, export_postgres, write_dataset
from transformer.training.data import load_splits


def papers(count=20):
    return [{"paper_id": i, "title": f"Fandom research case {i}", "language": "en", "keywords": ["fandom"],
             "doi": f"10.1000/{i}", "content_hash": str(i), "abstract": "An abstract.",
             "introduction": f"This study investigates how fandom community number {i} coordinates collective activities.",
             "body": f"Participants in community number {i} describe sharing information across digital networks.",
             "conclusion": f"Community number {i} illustrates the role of collective participation in cultural exchange."}
            for i in range(count)]


def test_reproducible_paper_split_and_no_unrelated_news():
    rows = papers()
    first, _ = build_samples(rows)
    random.Random(9).shuffle(rows)
    second, report = build_samples(rows)
    assert first == second
    assert report["counts"] == {"train": 16, "validation": 2, "test": 2}
    ids = [{r["paper_id"] for r in first[s]} for s in ("train", "validation", "test")]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    for samples in first.values():
        for sample in samples:
            assert sample["input"]["news_evidence"] == ["[NO_NEWS_EVIDENCE]"]
            assert "Conclusion:\n" in sample["target"]
            assert "…" not in sample["target"]


def test_duplicate_aliases_stay_together_transitively():
    rows = papers()
    duplicate = {**rows[0], "paper_id": 100, "title": "An alternate publication title"}
    transitive = {**rows[1], "paper_id": 101, "title": duplicate["title"]}
    samples, report = build_samples(rows + [duplicate, transitive])
    containing = [r for split in samples.values() for r in split if "0" in r["source_paper_ids"]]
    assert len(containing) == 1
    assert set(containing[0]["source_paper_ids"]) == {"0", "1", "100", "101"}
    assert report["duplicates_grouped"] == 3


def test_missing_sections_and_non_english_are_rejected():
    rows = papers(5)
    rows[0]["conclusion"] = None
    rows[1]["language"] = "ko"
    samples, report = build_samples(rows)
    assert sum(map(len, samples.values())) == 3
    assert report["rejected"] == {"no_usable_section_sentence": 1, "non_english_or_unknown_language": 1}


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
    assert all(q.startswith("SELECT ") for q in queries)
