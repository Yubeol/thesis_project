"""Resumable Conclusion-only dataset using read-only service RAG evidence."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import time

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIConnectionError, APITimeoutError
from transformers import AutoTokenizer

from agent.evidence import build_evidence_lists
from agent.retrieval import retrieve_hybrid
from transformer.dataset.build_grounded_dataset import (
    atomic_json, content_words, quality_check,
)
from transformer.preprocessing.prompts import (
    DEFAULT_INSTRUCTION, NO_NEWS, encode_input, make_input,
)
from transformer.training.data import dataset_fingerprint, load_splits


VERSION = "conclusion_service_v4"
RETRIEVAL_CACHE_VERSION = "conclusion_service_v3"  # Selection logic did not change.
TEACHER_PROMPT_VERSION = "conclusion_service_teacher_v1"
TEACHER_PROMPT = """Write only the Conclusion section of an academic paper.
Use only the evidence provided in the input. Directly answer the Research Question and synthesize the supported findings.
Summarize rather than introduce arguments, evidence, external knowledge, numbers, organizations, people,
publications, or platforms. Do not invent citations. End with a limited implication supported by the evidence.
Avoid repeated sentences and ideas, headings, bullets, and verbatim copying. Use clear academic English.
Keep it within 384 target tokens. If the visible evidence cannot support a reliable conclusion,
set reject=true and target="". Return JSON keys target (string), reject (boolean), reason (string)."""
GENERIC = set("k pop kpop korean korea music culture cultural paper study research evidence findings impact influence effect global globalization fan fans fandom media digital social platform platforms analysis conclusion".split())
NOISE_PATTERNS = (
    ("bibliography", re.compile(r"(?i)^\s*(references|bibliography|works cited|doi\s*:)\b")),
    ("url", re.compile(r"(?i)https?://|www\.")),
    ("journal_metadata", re.compile(r"(?i)\b(?:issn|isbn|volume\s+\d+|vol\.\s*\d+|journal of|proceedings of)\b")),
)


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def important_words(text: str) -> set[str]:
    return {word for word in content_words(text) if word not in GENERIC and len(word) >= 4}


def clean_candidate(item: dict) -> tuple[dict | None, str | None]:
    body = " ".join(str(item.get("content") or "").split())
    if len(body.split()) < 18:
        return None, "too_short"
    for name, pattern in NOISE_PATTERNS:
        if pattern.search(body[:250]):
            return None, name
    if re.match(r"(?i)^\s*(?:\d+[.)-]?\s*)?(?:[A-Z][a-z]+(?:,?\s+[A-Z][a-z]+){2,}|\d+[-–]\d+)\s*$", body):
        return None, "author_or_page_fragment"
    if len(re.findall(r"\b(?:19|20)\d{2}\b", body)) >= 5:
        return None, "citation_list"
    return {**item, "content": body}, None


def select_evidence(retrieval: dict, row: dict, split: str, paper_splits: dict[str, str]) -> tuple[list[str], list[str], dict]:
    query = " ".join((row["title"], row["topic"], row["research_question"]))
    terms = important_words(query)
    rejected = Counter()
    candidates = []
    for item in retrieval.get("papers", []):
        cleaned, reason = clean_candidate(item)
        if reason:
            rejected[reason] += 1
            continue
        paper_id = str(cleaned.get("paper_id") or "")
        if paper_id in paper_splits and paper_splits[paper_id] != split:
            rejected["cross_split_evidence"] += 1
            continue
        overlap = len(terms & important_words(cleaned["title"] + " " + cleaned["content"]))
        similarity = float(cleaned.get("similarity") or 0)
        if overlap < 1 and similarity < 0.58:
            rejected["unrelated_paper"] += 1
            continue
        candidates.append((cleaned, similarity + min(overlap, 5) * 0.06, important_words(cleaned["content"])))
    selected = []
    seen_ids = set()
    seen_content = set()
    while candidates and len(selected) < 3:
        ranked = []
        for item, score, words in candidates:
            body_key = re.sub(r"\W+", "", item["content"].casefold())[:180]
            if body_key in seen_content:
                continue
            redundant = max((len(words & other) / max(1, len(words | other)) for other in (x[2] for x in selected)), default=0)
            if redundant > 0.72:
                continue
            source = str(item.get("paper_id") or item.get("title"))
            diverse = 0.15 if source not in seen_ids else -0.12
            ranked.append((score + diverse - 0.30 * redundant, item, score, words, body_key, source))
        if not ranked:
            break
        _, item, score, words, body_key, source = max(ranked, key=lambda x: x[0])
        selected.append((item, score, words))
        seen_content.add(body_key)
        seen_ids.add(source)
        candidates = [entry for entry in candidates if entry[0] is not item]
    # A multi-paper target is useful only if at least two distinct retrieved papers support it.
    if len(seen_ids) < 2:
        rejected["insufficient_distinct_papers"] += 1
        selected = []
    news = []
    for item in retrieval.get("news", []):
        cleaned, reason = clean_candidate(item)
        if reason:
            rejected["news_" + reason] += 1
            continue
        title = cleaned.get("title_en") or cleaned.get("title_original") or ""
        overlap = len(terms & important_words(title + " " + cleaned["content"]))
        title_overlap = len(terms & important_words(title))
        if title_overlap < 1 or overlap < 2 or float(cleaned.get("similarity") or 0) < 0.40:
            rejected["unrelated_news"] += 1
            continue
        news.append((cleaned, float(cleaned.get("similarity") or 0) + overlap * 0.05))
    news.sort(key=lambda x: x[1], reverse=True)
    news = [item for item, score in news[:2] if len(news) == 1 or score >= 0.62]
    papers = [item for item, _, _ in selected]
    # The service formatter strips retrieval metadata before token budgeting.
    paper_texts, news_texts = build_evidence_lists(
        {"papers": papers, "news": news}, max_chars_per_item=1200, max_papers=3, max_news=2,
    )
    return paper_texts, news_texts, {
        "retrieved_papers": len(retrieval.get("papers", [])),
        "retrieved_news": len(retrieval.get("news", [])),
        "selected_papers": len(paper_texts), "selected_news": len(news_texts),
        "distinct_paper_ids": len(seen_ids), "rejection_reasons": dict(rejected),
    }


def teacher_cache_key(decoded_prompt: str, model: str) -> str:
    return digest({"prompt_version": TEACHER_PROMPT_VERSION, "model": model, "final_encoded_input": decoded_prompt})


def teacher_call(decoded_prompt: str, model: str, attempts: int = 4) -> dict:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=45, max_retries=0)
    for attempt in range(attempts):
        try:
            answer = client.chat.completions.create(
                model=model, response_format={"type": "json_object"}, temperature=0,
                max_tokens=500, messages=[
                    {"role": "system", "content": TEACHER_PROMPT},
                    {"role": "user", "content": decoded_prompt},
                ],
            )
            value = json.loads(answer.choices[0].message.content or "{}")
            return {"target": str(value.get("target") or "").strip(),
                    "reject": bool(value.get("reject")), "reason": str(value.get("reason") or "")}
        except (RateLimitError, APIConnectionError, APITimeoutError):
            if attempt == attempts - 1:
                raise
            time.sleep(min(20, 2 ** attempt))
    raise RuntimeError("Teacher retry limit reached")


def conclusion_gate(row: dict, target: str, tokenizer, visible: str) -> list[str]:
    reasons = list(quality_check(row, target, tokenizer, visible_prompt=visible)["rejection_reason"])
    sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+", target) if x.strip()]
    if len(sentences) < 2 or len(sentences) > 7:
        reasons.append("conclusion_structure")
    rq_terms = important_words(row["research_question"])
    if len(rq_terms & important_words(target)) < 1:
        reasons.append("research_question_not_answered")
    if re.match(r"(?i)^(this paper|this study|the study)\s+(examines|explores|aims|focuses)", target):
        reasons.append("body_or_introduction_style")
    if sentences and not re.search(r"(?i)\b(?:overall|together|therefore|thus|suggest|indicat|conclud|synthesi|implication|evidence|findings|ultimately|supports|underscores)\w*\b", " ".join(sentences[-2:])):
        reasons.append("missing_synthesis")
    # Do not reject solely on SentencePiece fragmentation. A pilot audit found
    # ordinary words such as "exemplifies" and "reinterpret" split into five
    # pieces; that signal produced false positives rather than reliable errors.
    if "�" in target or re.search(r"(?i)\b[A-Za-z]*([A-Za-z])\1{3,}[A-Za-z]*\b", target):
        reasons.append("corrupt_text")
    return sorted(set(reasons))


def build(source_dir: Path, output_dir: Path, model: str, workers: int = 3, limit: int | None = None, prepare_only: bool = False) -> dict:
    load_dotenv(override=False)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not configured")
    if output_dir.exists() and any(output_dir.glob("*.jsonl")):
        raise FileExistsError("Final dataset already exists; use a new output path")
    splits = load_splits(source_dir)
    jobs = [(split, row) for split in ("train", "validation", "test") for row in splits[split] if row["section"] == "Conclusion"]
    if limit is not None:
        jobs = jobs[:limit]
    paper_splits = {str(r["source_paper_id"]): split for split, rows in splits.items() for r in rows}
    tokenizer = AutoTokenizer.from_pretrained("artifacts/transformer_model_30epoch", local_files_only=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    retrieval_cache = output_dir / "retrieval_cache"
    teacher_cache = output_dir / "teacher_cache"
    retrieval_cache.mkdir(exist_ok=True)
    teacher_cache.mkdir(exist_ok=True)
    counters = Counter()

    def process(job):
        split, row = job
        rpath = retrieval_cache / (digest({"version": RETRIEVAL_CACHE_VERSION, "title": row["title"], "topic": row["topic"], "rq": row["research_question"], "split": split}) + ".json")
        if rpath.exists():
            prepared = json.loads(rpath.read_text(encoding="utf-8"))
            retrieval_source = "cache"
        else:
            try:
                query = row["title"] + " " + row["topic"]
                retrieval = retrieve_hybrid(paper_queries=[query], news_queries=[query], paper_top_k=10, news_top_k=5, use_graph=True, strict_graph=False)
                papers, news, selection = select_evidence(retrieval, row, split, paper_splits)
                prepared = {"paper_evidence": papers, "news_evidence": news, "selection": selection}
                atomic_json(rpath, prepared)
                retrieval_source = "api"
            except Exception as exc:
                return split, row, None, None, "retrieval_" + type(exc).__name__, "failure"
        papers, news = prepared["paper_evidence"], prepared["news_evidence"]
        if len(papers) < 2:
            return split, row, prepared, None, "insufficient_multi_paper_evidence", retrieval_source
        try:
            value = make_input(title=row["title"], topic=row["topic"], research_question=row["research_question"],
                               paper_evidence=papers, news_evidence=news, instruction=DEFAULT_INSTRUCTION, section="Conclusion")
            encoded = encode_input(tokenizer, value, 384)
            visible = tokenizer.decode(encoded["input_ids"], skip_special_tokens=True)
            if not re.search(r"\[PAPER\s+1\].*?Evidence:\s*\S", visible, re.S):
                return split, row, prepared, None, "paper_body_not_visible", retrieval_source
            if news and not re.search(r"\[NEWS\s+1\].*?Evidence:\s*\S", visible, re.S):
                return split, row, prepared, None, "news_body_not_visible", retrieval_source
            key = teacher_cache_key(visible, model)
            tpath = teacher_cache / f"{key}.json"
            if prepare_only:
                response, teacher_source = None, "not_called"
            elif tpath.exists():
                response, teacher_source = json.loads(tpath.read_text(encoding="utf-8")), "cache"
            else:
                response = teacher_call(visible, model)
                atomic_json(tpath, response)
                teacher_source = "api"
            prepared.update({"input": value, "encoded_input_text": visible, "encoded_input_ids_sha256": digest(encoded["input_ids"]),
                             "input_tokens": len(encoded["input_ids"]), "teacher_cache_key": key, "teacher_source": teacher_source})
            return split, row, prepared, response, None, retrieval_source
        except Exception as exc:
            return split, row, prepared, None, "prepare_or_teacher_" + type(exc).__name__, retrieval_source

    results = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(process, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            split, row, prepared, response, error, retrieval_source = future.result()
            results[(split, str(row["source_paper_id"]))] = (prepared, response, error)
            counters["retrieval_" + retrieval_source] += 1
            if prepared:
                counters["teacher_" + prepared.get("teacher_source", "not_called")] += 1
            if error == "insufficient_multi_paper_evidence":
                counters["skipped_insufficient_evidence"] += 1
            elif error:
                counters["errors"] += 1
            if index % 20 == 0 or index == len(jobs):
                print(json.dumps({"completed": index, "total": len(jobs), **counters}), flush=True)

    accepted = {split: [] for split in splits}
    rejected = []
    reasons = Counter()
    selection_reasons = Counter()
    for split, row in jobs:
        prepared, response, error = results[(split, str(row["source_paper_id"]))]
        if prepared:
            selection_reasons.update(prepared["selection"]["rejection_reasons"])
        target = response["target"] if response else ""
        if error:
            why = [error]
        elif prepare_only:
            why = ["prepare_only"]
        else:
            why = conclusion_gate(row if prepared is None else {**row, "input": prepared["input"]}, target, tokenizer, prepared["encoded_input_text"])
            if response["reject"]:
                why.append("teacher_reject")
        why = sorted(set(why))
        reasons.update(why)
        item = {
            **row, "input": prepared["input"] if prepared and "input" in prepared else row["input"],
            "paper_evidence": prepared["paper_evidence"] if prepared else [],
            "news_evidence": prepared["news_evidence"] if prepared and prepared["news_evidence"] else [NO_NEWS],
            "original_target": row["target"], "teacher_target": target,
            "target": target, "teacher_model": model, "dataset_version": VERSION,
            "grounding_pass": not why, "rejection_reason": why,
            "teacher_visible_input": prepared.get("encoded_input_text") if prepared else None,
            "teacher_input_hash": prepared.get("encoded_input_ids_sha256") if prepared else None,
            "input_tokens": prepared.get("input_tokens") if prepared else None,
            "selection": prepared.get("selection") if prepared else None,
        }
        (accepted[split] if not why else rejected).append(item)

    if prepare_only or limit is not None:
        report = {"pilot": True, "candidate": len(jobs), "prepared_multi_paper": sum(bool(results[(s,str(r['source_paper_id']))][0] and len(results[(s,str(r['source_paper_id']))][0]['paper_evidence'])>=2) for s,r in jobs),
                  "accepted": sum(map(len,accepted.values())), "rejected": len(rejected), "reasons": dict(reasons), "selection_reasons": dict(selection_reasons), "calls": dict(counters)}
        atomic_json(output_dir / "pilot_report.json", report)
        return report
    if not all(accepted.values()):
        raise RuntimeError("Service-like Conclusion dataset has empty split; training forbidden")
    for split, items in accepted.items():
        with (output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    with (output_dir / "rejected_samples.jsonl").open("w", encoding="utf-8") as handle:
        for item in rejected:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    atomic_json(output_dir / "manifest.json", {"sha256": dataset_fingerprint(output_dir), "dataset_version": VERSION})
    checked = load_splits(output_dir)
    ids = {split: {str(x["source_paper_id"]) for x in rows} for split, rows in checked.items()}
    leakage = bool(ids["train"] & ids["validation"] or ids["train"] & ids["test"] or ids["validation"] & ids["test"])
    all_rows = [r for rows in accepted.values() for r in rows]
    stats = {
        "candidate": len(jobs), "accepted": len(all_rows), "rejected": len(rejected),
        "split_counts": {name: len(rows) for name, rows in accepted.items()},
        "mean_paper_evidence": round(sum(len(r["paper_evidence"]) for r in all_rows)/len(all_rows), 2),
        "mean_news_evidence": round(sum(sum(x != NO_NEWS for x in r["news_evidence"]) for r in all_rows)/len(all_rows), 2),
        "samples_with_news": sum(any(x != NO_NEWS for x in r["news_evidence"]) for r in all_rows),
        "mean_input_tokens": round(sum(r["input_tokens"] for r in all_rows)/len(all_rows), 2),
        "mean_target_tokens": round(sum(len(tokenizer(r["target"])["input_ids"]) for r in all_rows)/len(all_rows), 2),
        "split_leakage_detected": leakage, "rejection_reasons": dict(reasons),
    }
    atomic_json(output_dir / "dataset_statistics.json", stats)
    atomic_json(output_dir / "teacher_generation_report.json", {"model": model, "calls": dict(counters), "reasons": dict(reasons), "prompt_version": TEACHER_PROMPT_VERSION})
    atomic_json(output_dir / "evidence_selection_report.json", {"selection_reasons": dict(selection_reasons)})
    atomic_json(output_dir / "sample_examples.json", {"examples": all_rows[:3]})
    if leakage:
        raise RuntimeError("Paper split leakage; training forbidden")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path("artifacts/transformer_dataset_grounded_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/transformer_dataset_conclusion_service_v4"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    load_dotenv(override=False)
    report = build(args.source_dir, args.output_dir, args.model or os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini"), args.workers, args.limit, args.prepare_only)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
