"""Conservative text cleanup that retains section boundaries and source text."""

import re
import unicodedata
from collections import Counter


def clean_pages(pages: list[str], audit: dict | None = None) -> str:
    def edge_key(line: str) -> str:
        line = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", line)).strip()
        return re.sub(r"\d+", "#", line)

    edge_lines = Counter()
    for page in pages:
        lines = [line.strip() for line in page.splitlines() if line.strip()]
        edge_lines.update({edge_key(line) for line in lines[:5] + lines[-5:]})
    repeated = {line for line, count in edge_lines.items()
                if len(pages) >= 3 and count >= max(3, len(pages) // 2) and len(line) < 180}
    cleaned = []
    removed_edges = 0
    for page in pages:
        lines = []
        for line in unicodedata.normalize("NFKC", page).splitlines():
            line = re.sub(r"[^\S\n]+", " ", line.replace("\x00", "")).strip()
            if edge_key(line) in repeated or re.fullmatch(r"(?:page\s*)?\d+(?:\s*(?:of|/)\s*\d+)?", line, re.I):
                removed_edges += 1
                continue
            # Remove publisher access watermarks; keep the unmodified extraction separately.
            if re.match(r"^\[[\d.:]+\]\s+Project MUSE", line):
                removed_edges += 1
                continue
            if re.fullmatch(r"(?:[PE]-)?ISSN[-\s\w:.,\d]+", line, re.I):
                removed_edges += 1
                continue
            if re.match(r"^(?:https?://(?:dx\.)?doi\.org/|doi\s*[:：]|\*?\s*corresponding author[,;:]|downloaded from\b|©|copyright\s+\d{4})", line, re.I):
                removed_edges += 1
                continue
            if "@" in line and len(line) < 180 and re.search(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", line):
                removed_edges += 1
                continue
            lines.append(line)
        cleaned.append("\n".join(lines))
    text = "\n\n".join(cleaned)
    text = re.sub(r"(?<=[a-z])-\n(?=[a-z])", "", text)
    # Only reflow obvious wrapped prose. Short lines may be headings, lists or captions.
    text = re.sub(r"(?m)(?<=[a-z,;])\n(?=[a-z])", " ", text)
    paragraphs = re.split(r"\n\s*\n", text)
    seen = set()
    unique = []
    duplicates = 0
    for paragraph in paragraphs:
        key = re.sub(r"\s+", " ", paragraph).casefold().strip()
        if len(key) >= 80 and key in seen:
            duplicates += 1
            continue
        if key:
            seen.add(key)
            unique.append(paragraph.strip())
    if audit is not None:
        audit["repeated_edge_lines_removed"] = removed_edges
        audit["duplicate_paragraphs_removed"] = duplicates
    return "\n\n".join(unique).strip()
