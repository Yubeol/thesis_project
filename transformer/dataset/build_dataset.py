from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata

from transformer.preprocessing.prompts import HEADINGS, clean_text, make_input

FIELDS = ("paper_id", "title", "keywords", "abstract", "introduction", "body", "conclusion", "language", "doi", "content_hash")
SPLITS = ("train", "validation", "test")


def export_postgres(env_file=None):
    # Import lazily: offline dataset use and training do not require psycopg.
    from pipeline.common.database import DEFAULT_ENV, connect
    with connect(Path(env_file) if env_file else DEFAULT_ENV, read_only=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_schema='public' AND table_name='papers'")
            schema = dict(cursor.fetchall())
            missing = set(FIELDS) - set(schema)
            if missing:
                raise ValueError("public.papers is missing columns: " + ", ".join(sorted(missing)))
            # Fixed column names verified against the deployed schema; no generated SQL or writes.
            cursor.execute("SELECT paper_id, title, keywords, abstract, introduction, body, conclusion, language, doi, content_hash FROM public.papers ORDER BY paper_id")
            return [dict(zip(FIELDS, row)) for row in cursor.fetchall()], schema


def normalized(value):
    return re.sub(r"[^a-z0-9]+", " ", unicodedata.normalize("NFKC", clean_text(value)).lower()).strip()


def sentences(text):
    text = clean_text(text)
    text = re.sub(r"\[(?:\d+[\s,;–-]*)+\]", "", text)
    text = re.sub(r"\([^)]*\b(?:19|20)\d{2}[a-z]?[^)]*\)", "", text)
    candidates = re.split(r"(?<=[.!?])\s+", text)
    return [clean_text(s) for s in candidates if 8 <= len(s.split()) <= 90
            and len(re.findall(r"[a-zA-Z]", s)) / max(1, len(s)) > 0.55
            and not re.search(r"https?://|doi\s*:|©|all rights reserved", s, re.I)]


def select_sentence(text, keywords):
    candidates = [s for s in sentences(text) if len(s.split()) <= 22 and re.search(r'[.!?]["”’]?$', s)]
    if not candidates:
        raise ValueError("no_usable_section_sentence")
    terms = set(normalized(" ".join(keywords)).split()) - {"the", "and", "of", "in", "to", "a"}
    # Whole, concise sentences only: never train on mid-sentence clipping.
    return max(candidates, key=lambda s: len(terms & set(normalized(s).split())))


def identity_keys(row):
    keys = [("id", str(row["paper_id"]))]
    if row.get("doi"):
        keys.append(("doi", re.sub(r"^(https?://(dx\.)?doi.org/|doi:\s*)", "", row["doi"].lower()).strip()))
    if row.get("content_hash"):
        keys.append(("hash", row["content_hash"]))
    # Titles differing only in punctuation/function words remain in one group.
    title = " ".join(t for t in normalized(row["title"]).split() if t not in {"a", "an", "the", "in", "on", "of", "and", "to"})
    keys.append(("title", title))
    content = "\n".join(normalized(row.get(k)) for k in ("introduction", "body", "conclusion"))
    keys.append(("text", hashlib.sha256(content.encode()).hexdigest()))
    return keys


def build_samples(rows, seed=42):
    rejected = Counter()
    eligible = []
    for row in rows:
        try:
            if not isinstance(row.get("paper_id"), (str, int)) or isinstance(row.get("paper_id"), bool):
                raise ValueError("missing_paper_id")
            if row.get("language") not in ("en", "eng", "english"):
                raise ValueError("non_english_or_unknown_language")
            if not clean_text(row.get("title")):
                raise ValueError("missing_title")
            keywords = row.get("keywords") or []
            if not isinstance(keywords, list) or any(not isinstance(x, str) for x in keywords):
                raise ValueError("invalid_keywords")
            selected = [select_sentence(row.get(k), keywords) for k in ("introduction", "body", "conclusion")]
            if len(set(normalized(s) for s in selected)) != 3:
                raise ValueError("duplicate_sections")
            eligible.append((row, selected))
        except ValueError as exc:
            rejected[str(exc)] += 1
    # Union-find joins duplicate identities transitively BEFORE splitting or emitting samples.
    parent = list(range(len(eligible)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, (row, _) in enumerate(eligible):
        for key in identity_keys(row):
            if key in seen:
                parent[root(i)] = root(seen[key])
            seen[key] = i
    groups = {}
    for i, entry in enumerate(eligible):
        groups.setdefault(root(i), []).append(entry)
    groups = sorted(groups.values(), key=lambda group: min(str(x[0]["paper_id"]) for x in group))
    random.Random(seed).shuffle(groups)
    if len(groups) < 3:
        raise ValueError("At least three eligible, distinct English papers are required for train/validation/test")
    n_val = max(1, round(len(groups) * .1))
    n_test = max(1, round(len(groups) * .1))
    n_train = len(groups) - n_val - n_test
    assignments = ["train"] * n_train + ["validation"] * n_val + ["test"] * n_test
    output = {name: [] for name in SPLITS}
    for group, split in zip(groups, assignments):
        # One representative per duplicate group; all aliases are retained as provenance.
        row, selected = min(group, key=lambda entry: str(entry[0]["paper_id"]))
        keywords = row.get("keywords") or []
        evidence = selected
        value = make_input(row["title"], ", ".join(keywords[:4]) or row["title"],
                           paper_evidence=evidence, news_evidence=[])
        target = "\n\n".join(name + ":\n" + s for name, s in zip(HEADINGS, selected))
        output[split].append({
            "paper_id": row["paper_id"],
            "source_paper_ids": sorted({str(entry[0]["paper_id"]) for entry in group}),
            "input": value, "target": target,
            "target_kind": "extractive_weak_supervision_v1",
        })
    return output, {"source_papers": len(rows), "eligible_papers": len(eligible),
                    "unique_paper_groups": len(groups), "duplicates_grouped": len(eligible) - len(groups),
                    "rejected": dict(rejected), "counts": {k: len(v) for k, v in output.items()}}


def write_dataset(samples, report, output_dir, seed, schema=None):
    output_dir = Path(output_dir)
    targets = [output_dir / (name + ".jsonl") for name in SPLITS] + [output_dir / "manifest.json"]
    if any(p.exists() for p in targets):
        raise ValueError("Dataset output already exists; choose a new --output-dir to preserve the previous snapshot")
    output_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for split in SPLITS:
        content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in samples[split]).encode("utf-8")
        path = output_dir / f"{split}.jsonl"
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(content)
        temporary.replace(path)
        hashes[split] = hashlib.sha256(content).hexdigest()
    manifest = {"schema_version": 1, "seed": seed, "split_ratio": [0.8, 0.1, 0.1],
                "created_at": datetime.now(timezone.utc).isoformat(), "sha256": hashes,
                "source_schema": schema or {}, **report,
                "limitations": "Short extractive weak labels, not human-reviewed abstractive drafts. News not paired. Near-duplicate semantic papers still need review."}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--papers-json", type=Path, help="Optional offline PostgreSQL-shaped list, with real paper_id values")
    parser.add_argument("--output-dir", type=Path, default=Path("data/training"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        if args.papers_json:
            rows, schema = json.loads(args.papers_json.read_text(encoding="utf-8-sig")), None
            if not isinstance(rows, list):
                raise ValueError("--papers-json must contain a JSON list")
        else:
            rows, schema = export_postgres(args.env_file)
        samples, report = build_samples(rows, args.seed)
        write_dataset(samples, report, args.output_dir, args.seed, schema)
        print(json.dumps({"output_dir": str(args.output_dir), **report}, indent=2))
        return 0
    except Exception as exc:
        # DB exceptions can include connection strings; only our validation errors are printed.
        message = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
        parser.exit(1, "Dataset export failed: " + message + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
