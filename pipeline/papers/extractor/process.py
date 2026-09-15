"""Produce source-linked paper text and section quality reports for later DB loading."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from collections import Counter
import hashlib
import json
import logging
from pathlib import Path
import re

from pipeline.papers.cleaner.text import clean_pages
from pipeline.papers.collector.openalex import normalize_title, write_json
from pipeline.papers.extractor.sections import split_sections

LOG = logging.getLogger(__name__)


def read_source(path: Path, kind: str, audit: dict | None = None) -> list[str]:
    if kind == "pdf":
        from pypdf import PdfReader
        reader = PdfReader(path)
        if reader.is_encrypted:
            raise ValueError("Encrypted PDF requires a separately accessible source")
        if len(reader.pages) > 150:
            raise ValueError("Source exceeds 150-page article limit")
        pages, plain_pages, layout_pages = [], [], []
        for page in reader.pages:
            stream = page.get_contents()
            if stream and len(stream.get_data()) > 20_000_000:
                raise ValueError("PDF page stream exceeds memory budget")
            plain = page.extract_text() or ""
            plain_pages.append(plain)
            lines = [line.strip() for line in plain.splitlines() if line.strip()]
            fragmented = len(lines) >= 80 and sum(len(line.split()) == 1 for line in lines) / len(lines) > 0.65
            if fragmented:
                layout = page.extract_text(extraction_mode="layout") or ""
                if len(layout.splitlines()) < len(lines) * 0.7:
                    pages.append(layout)
                    layout_pages.append(len(pages))
                    continue
            pages.append(plain)
        if audit is not None:
            audit.update({"plain_pages": plain_pages, "layout_fallback_pages": layout_pages})
        return pages
    if kind == "html":
        from trafilatura import extract
        content = extract(path.read_bytes(), include_comments=False, include_tables=False,
                          include_links=False, favor_precision=True)
        return [content or ""]
    raise ValueError("Unsupported source format")


def quality_flags(paper: dict, text: str, sections: dict) -> list[str]:
    flags = []
    if len(text) < 5000:
        flags.append("insufficient_fulltext")
    title_words = set(re.findall(r"\w{3,}", paper["title"].casefold()))
    text_words = set(re.findall(r"\w{3,}", text[:12000].casefold()))
    if title_words and len(title_words & text_words) / len(title_words) < 0.6:
        flags.append("source_title_mismatch")
    for name, minimum in (("introduction", 300), ("body", 1000), ("conclusion", 200)):
        if len(sections.get(name) or "") < minimum:
            flags.append("missing_or_short_" + name)
    if text.count("\ufffd") / max(1, len(text)) > 0.005:
        flags.append("text_encoding_damage")
    return flags


def process_one(paper: dict, raw_root: Path, output: Path) -> dict:
    from langdetect import DetectorFactory, detect_langs, LangDetectException
    DetectorFactory.seed = 0
    identifier = paper["id"]
    if not re.fullmatch(r"W\d+", identifier):
        raise ValueError("Invalid paper identifier")
    manifest = raw_root / "documents" / identifier / "source.json"
    row = {**paper, "training_eligible": False}
    if not manifest.exists():
        return {**row, "fulltext_status": "unavailable", "quality_flags": ["source_not_acquired"]}
    source = json.loads(manifest.read_text(encoding="utf-8"))
    kind = source["format"]
    if kind not in {"pdf", "html"} or source["filename"] != "source." + kind:
        raise ValueError("Invalid source manifest")
    path = manifest.parent / source["filename"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
        raise ValueError("Source checksum mismatch; acquire the source again")
    audit = {}
    pages = read_source(path, kind, audit)
    text = clean_pages(pages)
    sections = split_sections(text)
    flags = quality_flags(paper, text, sections)
    language = None
    language_confidence = None
    if len(text) >= 500:
        try:
            # Metadata can label an Indonesian paper English because its abstract is translated.
            sample = "\n".join(sections.get(k) or "" for k in ("introduction", "body", "conclusion"))
            guesses = detect_langs((sample if len(sample) > 500 else text)[-30000:])
            language, language_confidence = guesses[0].lang, guesses[0].prob
        except LangDetectException:
            pass
    if language is None or language_confidence < 0.9:
        flags.append("language_needs_review")
    # Preserve every source abstract and choose one in the detected body language.
    abstract_candidates = [s["text"] for s in sections["abstract_variants"] if s["text"]]
    if paper.get("abstract"):
        abstract_candidates.append(paper["abstract"])
    selected_abstract = sections["abstract_extracted"] or paper.get("abstract")
    abstract_language = None
    for candidate in abstract_candidates:
        try:
            guesses = detect_langs(candidate)
            if guesses[0].lang == language and guesses[0].prob >= 0.9:
                selected_abstract, abstract_language = candidate, guesses[0].lang
                break
        except LangDetectException:
            pass
    record = {
        **row, **sections, "abstract": selected_abstract,
        "abstract_metadata": paper.get("abstract"), "abstract_language": abstract_language,
        "fulltext": text, "fulltext_status": "text_extracted" if text else "no_text",
        "section_status": "complete_heuristic" if not flags else "needs_review",
        "quality_flags": flags, "training_eligible": not flags,
        "english_training_eligible": not flags and language == "en" and abstract_language == "en",
        "extraction_method": ("pypdf_layout_fallback" if audit.get("layout_fallback_pages") else "pypdf") if kind == "pdf" else "trafilatura",
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "source_document": source, "page_count": len(pages) if kind == "pdf" else None,
        "language_metadata": paper.get("language"), "language": language or paper.get("language"),
        "language_detection_confidence": language_confidence,
        "content_sha256": hashlib.sha256(normalize_title(text).encode()).hexdigest(),
    }
    # Preserve raw page text for diagnosing cleanup/heading mistakes without re-downloading.
    write_json(output / "extractions" / (identifier + ".json"), {"id": identifier, "pages": pages, **audit, "record": record})
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/papers"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/papers"))
    parser.add_argument("--minimum-english", type=int, default=0,
                        help="Fail when fewer than this many English papers pass automatic section checks")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.output / "extraction.log", encoding="utf-8")])
    papers = json.loads((args.raw_root / "papers.json").read_text(encoding="utf-8"))
    records, failures, hashes = [], [], set()
    for paper in papers:
        try:
            record = process_one(paper, args.raw_root, args.output)
            digest = record.get("content_sha256")
            if digest and digest in hashes:
                record["quality_flags"].append("duplicate_fulltext")
                record["training_eligible"] = False
                record["english_training_eligible"] = False
            if digest:
                hashes.add(digest)
            records.append(record)
            LOG.info("%s: %s; eligible=%s", paper["id"], record["fulltext_status"], record["training_eligible"])
        except Exception as exc:
            # One corrupt publisher document must not discard other successfully processed papers.
            error = {"id": paper["id"], "error_type": type(exc).__name__, "error": str(exc)}
            failures.append(error)
            records.append({**paper, "training_eligible": False, "quality_flags": ["extraction_failed"]})
            LOG.error("%s extraction failed: %s", paper["id"], type(exc).__name__)
    write_json(args.output / "papers.json", records)
    report = {"finished_at": datetime.now(timezone.utc).isoformat(), "selected_count": len(papers),
              "text_extracted": sum(r.get("fulltext_status") == "text_extracted" for r in records),
              "structurally_eligible": sum(r["training_eligible"] for r in records),
              "english_eligible": sum(bool(r.get("english_training_eligible")) for r in records),
              "eligible_by_language": dict(Counter(r.get("language") for r in records if r["training_eligible"])),
              "quality_flag_counts": dict(Counter(f for r in records for f in r["quality_flags"])),
              "minimum_english": args.minimum_english,
              "english_minimum_met": sum(bool(r.get("english_training_eligible")) for r in records) >= args.minimum_english,
              "human_reviewed": 0, "failures": failures}
    write_json(args.output / "processing_report.json", report)
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0 if report["text_extracted"] and not failures and report["english_minimum_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
