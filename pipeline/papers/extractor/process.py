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


class EncryptedPDFError(ValueError):
    """The source PDF cannot be extracted without a password."""


def read_source(path: Path, kind: str, audit: dict | None = None) -> list[str]:
    if kind == "pdf":
        from pypdf import PdfReader
        reader = PdfReader(path)
        if reader.is_encrypted:
            raise EncryptedPDFError("Encrypted PDF requires a separately accessible source")
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


def alternative_source(paper: dict, raw_root: Path, primary: dict) -> tuple[Path, dict] | None:
    """Find a second accessible OA source without replacing the publisher original."""
    from pipeline.papers.collector.download import AccessPolicy, source_format

    directory = raw_root / "documents" / paper["id"]
    fallback_manifest = directory / "fallback.json"
    if fallback_manifest.exists():
        cached = json.loads(fallback_manifest.read_text(encoding="utf-8"))
        path = directory / cached.get("filename", "")
        if path.name in {"fallback.pdf", "fallback.html"} and path.is_file():
            if hashlib.sha256(path.read_bytes()).hexdigest() == cached.get("sha256"):
                return path, cached
    policy = AccessPolicy()
    excluded = {primary.get("source_url"), primary.get("resolved_url")}
    for location in paper.get("source_locations", [])[:8]:
        url = location.get("url")
        if not url or url in excluded:
            continue
        try:
            payload, content_type, resolved = policy.read(url)
            kind = source_format(payload, content_type)
            # Reject landing pages, access notices and other non-article responses.
            candidate = directory / ("fallback." + kind)
            temporary = directory / (candidate.name + ".tmp")
            temporary.write_bytes(payload)
            try:
                text = "\n".join(read_source(temporary, kind))
                if len(text) < 500:
                    continue
                temporary.replace(candidate)
            finally:
                temporary.unlink(missing_ok=True)
            result = {"id": paper["id"], "status": "downloaded_unverified",
                      "filename": candidate.name, "format": kind, "source_url": url,
                      "resolved_url": resolved, "license": location.get("license"),
                      "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
                      "retrieved_at": datetime.now(timezone.utc).isoformat()}
            write_json(fallback_manifest, result)
            return candidate, result
        except Exception as exc:
            LOG.warning("%s alternative source unavailable: %s", paper["id"], type(exc).__name__)
    return None


def quality_state(flags: list[str], language: str | None, confidence: float | None,
                  abstract_language: str | None) -> tuple[str, float]:
    score = 1.0
    score -= min(0.75, 0.15 * len(flags))
    if language != "en" or confidence is None or confidence < 0.9:
        score -= 0.15
    if abstract_language != "en":
        score -= 0.1
    score = round(max(0.0, score), 2)
    if any(flag in flags for flag in ("source_not_acquired", "extraction_failed", "source_title_mismatch", "no_extracted_text")):
        return "rejected", score
    if not flags and language == "en" and confidence is not None and confidence >= 0.9 and abstract_language == "en":
        return "ready", score
    return "review_required", score


def process_one(paper: dict, raw_root: Path, output: Path) -> dict:
    from langdetect import DetectorFactory, detect_langs, LangDetectException
    DetectorFactory.seed = 0
    identifier = paper["id"]
    if not re.fullmatch(r"W\d+", identifier):
        raise ValueError("Invalid paper identifier")
    manifest = raw_root / "documents" / identifier / "source.json"
    row = {**paper, "training_eligible": False, "english_training_eligible": False}
    if not manifest.exists():
        return {**row, "fulltext_status": "unavailable", "quality_flags": ["source_not_acquired"],
                "quality_state": "rejected", "quality_score": 0.0}
    source = json.loads(manifest.read_text(encoding="utf-8"))
    kind = source["format"]
    if kind not in {"pdf", "html"} or source["filename"] != "source." + kind:
        raise ValueError("Invalid source manifest")
    path = manifest.parent / source["filename"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
        raise ValueError("Source checksum mismatch; acquire the source again")
    audit = {}
    raw_path = raw_root / "extracted" / (identifier + "-" + source["sha256"][:12] + ".json")
    cached_pages = None
    for cache_path in (raw_path, raw_root / "extracted" / (identifier + ".json")):
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            cached_hash = cached.get("source_sha256") or cached.get("record", {}).get("source_document", {}).get("sha256")
            if cached_hash == source["sha256"] and isinstance(cached.get("pages"), list):
                cached_pages = cached["pages"]
                audit.update({k: cached.get(k, []) for k in ("plain_pages", "layout_fallback_pages")})
                break
    encrypted_pdf = False
    html_fallback_used = False
    try:
        pages = cached_pages if cached_pages is not None else read_source(path, kind, audit)
    except EncryptedPDFError:
        encrypted_pdf = True
        alternate = alternative_source(paper, raw_root, source)
        if alternate is None:
            raise
        path, source = alternate
        kind = source["format"]
        html_fallback_used = kind == "html"
        pages = read_source(path, kind, audit)
    text = clean_pages(pages, audit)
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
    elif language != "en":
        flags.append("non_english_fulltext")
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
    if not text:
        flags.append("no_extracted_text")
    state, score = quality_state(flags, language, language_confidence, abstract_language)
    quality = {
        "text_extracted": bool(text), "has_abstract": bool(selected_abstract),
        **{"has_" + kind: any(s["section_type"] == kind and s["text"] for s in sections["sections"])
           for kind in ("introduction", "literature_review", "background", "methodology",
                        "results", "discussion", "conclusion")},
        "reference_removed_from_body": bool(sections["references_text"])
                                       and not bool(sections["references_text"] in (sections["body"] or "")),
        "character_count": len(text), "section_count": len(sections["sections"]),
        "quality_score": score,
    }
    record = {
        **row, **sections, "abstract": selected_abstract,
        "abstract_metadata": paper.get("abstract"), "abstract_language": abstract_language,
        "fulltext": text, "fulltext_status": "text_extracted" if text else "no_text",
        "section_status": "complete_heuristic" if state == "ready" else "needs_review",
        "quality_flags": flags, "quality_state": state, "quality_score": score,
        "quality": quality,
        "training_eligible": state == "ready", "english_training_eligible": state == "ready",
        "extraction_method": ("pypdf_layout_fallback" if audit.get("layout_fallback_pages") else "pypdf") if kind == "pdf" else "trafilatura",
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "source_document": source, "page_count": len(pages) if kind == "pdf" else None,
        "language_metadata": paper.get("language"), "language": language or paper.get("language"),
        "language_detection_confidence": language_confidence,
        "content_sha256": hashlib.sha256(normalize_title(text).encode()).hexdigest(),
        "encrypted_pdf": encrypted_pdf, "html_fallback_used": html_fallback_used,
        "duplicate_paragraphs_removed": audit.get("duplicate_paragraphs_removed", 0),
        "repeated_edge_lines_removed": audit.get("repeated_edge_lines_removed", 0),
    }
    # Raw page text stays with the immutable acquired source, not the processed dataset.
    raw_path = raw_root / "extracted" / (identifier + "-" + source["sha256"][:12] + ".json")
    if not raw_path.exists():
        write_json(raw_path,
                   {"id": identifier, "source_sha256": source["sha256"], "pages": pages,
                    "plain_pages": audit.get("plain_pages", []),
                    "layout_fallback_pages": audit.get("layout_fallback_pages", [])})
    return record


def process_records(papers: list[dict], raw_root: Path, output: Path) -> tuple[list[dict], list[dict]]:
    records, failures, hashes = [], [], set()
    for paper in papers:
        try:
            record = process_one(paper, raw_root, output)
            digest = record.get("content_sha256")
            if digest and digest in hashes:
                record["quality_flags"].append("duplicate_fulltext")
                record["training_eligible"] = False
                record["english_training_eligible"] = False
                record["quality_state"] = "review_required"
                record["quality_score"] = round(max(0.0, record["quality_score"] - 0.15), 2)
                if record.get("quality"):
                    record["quality"]["quality_score"] = record["quality_score"]
            if digest:
                hashes.add(digest)
            records.append(record)
            LOG.info("%s: %s; eligible=%s", paper["id"], record["fulltext_status"], record["training_eligible"])
        except Exception as exc:
            error = {"id": paper["id"], "error_type": type(exc).__name__, "error": str(exc)}
            failures.append(error)
            reason = "encrypted_pdf_no_fallback" if isinstance(exc, EncryptedPDFError) else "extraction_failed"
            records.append({**paper, "training_eligible": False, "english_training_eligible": False,
                            "fulltext_status": "extraction_failed", "quality_state": "rejected",
                            "quality_score": 0.0, "quality_flags": [reason], "failure_reason": reason,
                            "encrypted_pdf": isinstance(exc, EncryptedPDFError)})
            LOG.error("%s extraction failed: %s", paper["id"], type(exc).__name__)
    return records, failures


def minimum_exit_code(records: list[dict], minimum: int) -> int:
    return 0 if sum(bool(r.get("english_training_eligible")) for r in records) >= minimum else 1


def sample_audit(records: list[dict], count: int = 12) -> list[dict]:
    """Compact real-paper audit: structure and residual-noise signals, never full text."""
    extracted = [r for r in records if r.get("fulltext_status") == "text_extracted"]
    if not extracted:
        return []
    indices = sorted({round(i * (len(extracted) - 1) / max(1, count - 1))
                      for i in range(min(count, len(extracted)))})
    return [{"id": (r := extracted[index])["id"], "title": r["title"],
             "quality_state": r["quality_state"], "quality_flags": r["quality_flags"],
             "abstract_length": len(r.get("abstract") or ""),
             "introduction_length": len(r.get("introduction") or ""),
             "methodology_length": sum(len(s["text"]) for s in r.get("sections", [])
                                       if s["section_type"] == "methodology"),
             "results_length": sum(len(s["text"]) for s in r.get("sections", [])
                                   if s["section_type"] == "results"),
             "discussion_length": sum(len(s["text"]) for s in r.get("sections", [])
                                      if s["section_type"] == "discussion"),
             "conclusion_length": len(r.get("conclusion") or ""),
             "body_length": len(r.get("body") or ""),
             "references_separated": bool(r.get("references_text")),
             "appendix_separated": bool(r.get("appendix_text")),
             "repeated_header_footer_removed": r.get("repeated_edge_lines_removed", 0),
             "possible_header_footer_residue": bool(re.search(
                 r"(?im)^(?:downloaded from|copyright\s+\d{4}|https?://(?:dx\.)?doi\.org/)",
                 r.get("fulltext") or ""))} for index in indices]


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
    if args.minimum_english < 0:
        parser.error("minimum-english must not be negative")
    source_formats = Counter()
    for paper in papers:
        manifest = args.raw_root / "documents" / paper["id"] / "source.json"
        if manifest.exists():
            try:
                source_formats[json.loads(manifest.read_text(encoding="utf-8"))["format"]] += 1
            except (ValueError, KeyError):
                pass
    records, failures = process_records(papers, args.raw_root, args.output)
    write_json(args.output / "papers.json", records)
    write_json(args.output / "papers_ready.json", [r for r in records if r.get("quality_state") == "ready"])
    write_json(args.output / "sample_audit.json", sample_audit(records))
    report = {"finished_at": datetime.now(timezone.utc).isoformat(), "metadata_selected": len(papers),
              "selected_count": len(papers), "downloaded_pdf": source_formats["pdf"],
              "downloaded_html": source_formats["html"],
              "text_extracted": sum(r.get("fulltext_status") == "text_extracted" for r in records),
              "extract_failed": sum(r.get("fulltext_status") == "extraction_failed" for r in records),
              "encrypted_pdf": sum(bool(r.get("encrypted_pdf")) for r in records),
              "html_fallback_used": sum(bool(r.get("html_fallback_used")) for r in records),
              "section_parsed": sum(bool(r.get("sections")) for r in records),
              "structurally_eligible": sum(r["training_eligible"] for r in records),
              "english_eligible": sum(bool(r.get("english_training_eligible")) for r in records),
              "ready": sum(r.get("quality_state") == "ready" for r in records),
              "review_required": sum(r.get("quality_state") == "review_required" for r in records),
              "rejected": sum(r.get("quality_state") == "rejected" for r in records),
              "duplicate_removed": sum(r.get("duplicate_paragraphs_removed", 0) for r in records),
              "eligible_by_language": dict(Counter(r.get("language") for r in records if r["training_eligible"])),
              "quality_flag_counts": dict(Counter(f for r in records for f in r["quality_flags"])),
              "minimum_english": args.minimum_english,
              "minimum_required": args.minimum_english,
              "english_minimum_met": sum(bool(r.get("english_training_eligible")) for r in records) >= args.minimum_english,
              "minimum_met": sum(bool(r.get("english_training_eligible")) for r in records) >= args.minimum_english,
              "human_reviewed": 0, "failures": failures}
    write_json(args.output / "processing_report.json", report)
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return minimum_exit_code(records, args.minimum_english)


if __name__ == "__main__":
    raise SystemExit(main())
