"""Conservative text cleanup that retains section boundaries and source text."""

import re
import unicodedata
from collections import Counter


def clean_pages(pages: list[str]) -> str:
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
    for page in pages:
        lines = []
        for line in unicodedata.normalize("NFKC", page).splitlines():
            line = re.sub(r"[^\S\n]+", " ", line.replace("\x00", "")).strip()
            if edge_key(line) in repeated or re.fullmatch(r"(?:page\s*)?\d+(?:\s*(?:of|/)\s*\d+)?", line, re.I):
                continue
            # Remove publisher access watermarks; keep the unmodified extraction separately.
            if re.match(r"^\[[\d.:]+\]\s+Project MUSE", line):
                continue
            if re.fullmatch(r"(?:[PE]-)?ISSN[-\s\w:.,\d]+", line, re.I):
                continue
            lines.append(line)
        cleaned.append("\n".join(lines))
    text = "\n\n".join(cleaned)
    text = re.sub(r"(?<=[a-z])-\n(?=[a-z])", "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()
