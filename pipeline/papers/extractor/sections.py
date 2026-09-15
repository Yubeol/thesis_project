"""Extract actual headed sections; absent sections stay null, never synthesized."""

import re

PREFIX = re.compile(r"^(?:\d{1,2}(?:\.\d{1,2}){0,3}[.)]?\s+|[IVX]+[.)]\s+|[A-H][.)]\s+|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ][.\s]*)")
NAMES = {
    "abstract": r"abstract|abstrak|초록|요약",
    "introduction": r"introduction|preface|서론|서 론|pendahuluan",
    "conclusion": r"conclusions?(?: and (?:discussion|implications|recommendations?|research limitations))?|discussion and conclusions?|결론(?: 및 제언)?|결 론|(?:kesimpulan|simpulan)(?: dan saran)?|penutup",
    "references": r"references|bibliography|works cited|참고문헌|참고 문헌|daftar pustaka|referensi",
    "backmatter": r"acknowledg(?:e)?ments?|funding|disclosure statement|conflict(?:s)? of interest|notes|endnotes|appendix(?: [A-Z])?",
    "keywords": r"keywords?|key words|kata[- ]kata kunci|kata kunci|주제어|핵심어",
    "body": r"literature review|theoretical (?:framework|background)|research (?:method|design|methodology)s?|methods?|methodology|materials and methods|results?(?: and discussion)?|findings(?: and discussion)?|discussion|background|연구 방법|연구방법|연구 결과|연구결과|이론적 배경|논의|metode(?: penelitian)?|hasil(?: dan pembahasan)?|pembahasan",
}


def heading_kind(line: str) -> str | None:
    line = line.strip().lstrip("#").strip()
    if not line or len(line) > 140 or len(line.split()) > 18:
        return None
    if re.search(r"https?://|\bISSN\b|\bISBN\b|\bDOI\b|@", line, re.I):
        return None
    bare = PREFIX.sub("", line).strip().rstrip(":")
    # A whole lower-case word such as 'method.' in prose is not a heading.
    if bare.endswith(".") and bare.islower():
        return None
    bare = bare.rstrip(".")
    # Superscript footnote numbers are sometimes extracted on the heading baseline.
    bare = re.sub(r"\s+\d{1,2}$", "", bare)
    for kind, expression in NAMES.items():
        if re.fullmatch("(?:" + expression + r")(?:\s*[:：]\s*[^.!?]+)?", bare, re.I):
            return kind
    # Numbered tables, quotations and sentence fragments must not create sections.
    words = re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", bare)
    significant = [w for w in words if w.casefold() not in {
        "a", "an", "and", "as", "at", "by", "for", "from", "in", "of", "on", "or", "the", "to", "with",
    }]
    title_case = significant and sum(w[0].isupper() for w in significant) / len(significant) >= 0.65
    if not re.search(r"[.!?]$|[,;%=]|\d", bare) and len(bare.split()) <= 12:
        if (PREFIX.match(line) and title_case) or (bare.isupper() and len(words) >= 2 and len(bare) >= 10):
            return "body"
    return None


def split_sections(text: str) -> dict:
    lines = text.splitlines(keepends=True)
    headings, offset = [], 0
    for line in lines:
        kind = heading_kind(line)
        if kind:
            content_start = offset + len(line)
            if kind in {"abstract", "keywords"} and re.search(r"[:：]", line):
                # 'Abstract: actual first sentence' retains the actual first sentence.
                content_start = offset + re.search(r"[:：]", line).end()
            headings.append({"heading": line.strip(), "kind": kind,
                             "start": offset, "content_start": content_start})
        offset += len(line)
    sections = []
    for i, item in enumerate(headings):
        end = headings[i + 1]["start"] if i + 1 < len(headings) else len(text)
        sections.append({**item, "end": end, "text": text[item["content_start"]:end].strip()})
    intro = next((s for s in sections if s["kind"] == "introduction"), None)
    conclusion = next((s for s in sections if s["kind"] == "conclusion"
                       and (intro is None or s["start"] > intro["start"])), None)
    abstracts = [s for s in sections if s["kind"] == "abstract"
                 and (intro is None or s["start"] < intro["start"])]
    abstract = abstracts[-1] if abstracts else None
    body_start = None
    if intro:
        for section in sections:
            if section["start"] <= intro["start"]:
                continue
            if conclusion and section["start"] >= conclusion["start"]:
                break
            if section["kind"] in {"references", "backmatter"}:
                break
            # Introductory subsections (1.1, 1.2) remain part of Introduction.
            intro_major = re.match(r"(\d+)\.", intro["heading"])
            subsection = re.match(r"(\d+)\.\d", section["heading"])
            if intro_major and subsection and intro_major[1] == subsection[1]:
                continue
            if section["kind"] == "body" and section["text"]:
                body_start = section["start"]
                break
    endmatter = next((s["start"] for s in sections if s["kind"] in {"references", "backmatter"}
                      and intro and s["start"] > intro["start"]), len(text))
    intro_end = body_start or (conclusion["start"] if conclusion else endmatter)
    body_end = min(conclusion["start"] if conclusion else len(text), endmatter)
    conclusion_end = next((s["start"] for s in sections if s["kind"] in {"references", "backmatter"}
                           and conclusion and s["start"] > conclusion["start"]), len(text))
    return {"abstract_extracted": abstract["text"] if abstract else None,
            "abstract_variants": [{"heading": s["heading"], "text": s["text"]} for s in abstracts],
            "introduction": text[intro["content_start"]:intro_end].strip() if intro else None,
            "body": text[body_start:body_end].strip() if body_start is not None else None,
            "conclusion": text[conclusion["content_start"]:conclusion_end].strip() if conclusion else None,
            "sections": sections}
