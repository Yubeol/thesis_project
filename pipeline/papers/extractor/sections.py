"""Extract actual headed sections; absent sections stay null, never synthesized."""

import re

PREFIX = re.compile(r"^(?:\d{1,2}(?:\.\d{1,2}){0,3}[.)]?\s+|[IVX]+[.)]\s+|[A-H][.)]\s+|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ][.\s]*)")
NAMES = {
    "abstract": r"abstract|abstrak|초록|요약",
    "introduction": r"introduction|preface|서론|서 론|pendahuluan",
    "conclusion": r"conclusions?(?: and (?:discussion|implications|recommendations?|research limitations|future work))?|discussion and conclusions?|결론(?: 및 제언)?|결 론|(?:kesimpulan|simpulan)(?: dan saran)?|penutup",
    "references": r"references|bibliography|works cited|literature cited|참고문헌|참고 문헌|daftar pustaka|referensi",
    "appendix": r"appendix(?: [A-Z])?|appendices|supplementary material|supporting information|부록",
    "backmatter": r"acknowledg(?:e)?ments?|funding|disclosure statement|conflict(?:s)? of interest|notes|endnotes",
    "keywords": r"keywords?|key words|kata[- ]kata kunci|kata kunci|주제어|핵심어",
    "literature_review": r"literature review|review of (?:the )?literature|related work|previous studies|background and related work|선행연구|선행 연구",
    "background": r"(?:theoretical|conceptual) (?:framework|background)|background|이론적 배경",
    "methodology": r"research (?:method|design|methodology)s?|methods?|methodology|materials and methods|연구 방법|연구방법|metode(?: penelitian)?",
    "results": r"results?|experimental results|findings|연구 결과|연구결과|hasil",
    "discussion": r"(?:results?|findings) and discussion|discussion(?: and implications)?|implications|논의|pembahasan|hasil dan pembahasan",
}

CAPTION = re.compile(r"^(?:fig(?:ure)?\.?|table)\s+\d+[A-Za-z]?(?:[.:]\s*.*)?$", re.I)


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
    if re.fullmatch(r"(?:case study|case\s+\d+|example)(?:\s*[:：]\s*[^.!?]+)?", bare, re.I):
        return "other"
    # Numbered tables, quotations and sentence fragments must not create sections.
    words = re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", bare)
    significant = [w for w in words if w.casefold() not in {
        "a", "an", "and", "as", "at", "by", "for", "from", "in", "of", "on", "or", "the", "to", "with",
    }]
    title_case = significant and sum(w[0].isupper() for w in significant) / len(significant) >= 0.65
    if not re.search(r"[.!?]$|[,;%=]|\d", bare) and len(bare.split()) <= 12:
        if (PREFIX.match(line) and title_case) or (bare.isupper() and len(words) >= 2 and len(bare) >= 10):
            return "other"
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
    captions = []
    for i, item in enumerate(headings):
        end = headings[i + 1]["start"] if i + 1 < len(headings) else len(text)
        body_lines = []
        last_was_caption = False
        for line in text[item["content_start"]:end].splitlines():
            is_caption = bool(CAPTION.fullmatch(line.strip()))
            is_caption_note = last_was_caption and bool(re.fullmatch(
                r"(?:source|note)\s*[:：]\s*.{0,200}", line.strip(), re.I))
            if is_caption or is_caption_note:
                captions.append({"section_type": item["kind"], "text": line.strip()})
                last_was_caption = True
            else:
                body_lines.append(line)
                last_was_caption = False
        section_type = item["kind"]
        if section_type == "other":
            number = re.match(r"(\d+)\.\d", item["heading"])
            if number:
                previous = next((s for s in reversed(sections)
                                 if re.match(r"(\d+)[.)]?\s", s["heading"])
                                 and re.match(r"(\d+)[.)]?\s", s["heading"])[1] == number[1]), None)
                if previous and previous["section_type"] in {
                    "introduction", "literature_review", "background", "methodology",
                    "results", "discussion", "conclusion"}:
                    section_type = previous["section_type"]
        sections.append({**item, "section_type": section_type, "end": end,
                         "text": "\n".join(body_lines).strip()})
    intro = next((s for s in sections if s["kind"] == "introduction"), None)
    conclusion = next((s for s in sections if s["kind"] == "conclusion"
                       and (intro is None or s["start"] > intro["start"])), None)
    abstracts = [s for s in sections if s["kind"] == "abstract"
                 and (intro is None or s["start"] < intro["start"])]
    abstract = abstracts[-1] if abstracts else None
    endmatter = next((s["start"] for s in sections if s["kind"] in {"references", "appendix", "backmatter"}
                      and intro and s["start"] > intro["start"]), len(text))
    body_start = None
    if intro:
        intro_major = re.match(r"(\d+)\.", intro["heading"])
        for section in sections:
            if section["start"] <= intro["start"] or section["start"] >= endmatter:
                continue
            if conclusion and section["start"] >= conclusion["start"]:
                break
            subsection = re.match(r"(\d+)\.\d", section["heading"])
            if intro_major and subsection and intro_major[1] == subsection[1]:
                continue
            if section["kind"] in {"literature_review", "background", "methodology", "results", "discussion", "other"}:
                body_start = section["start"]
                break
    intro_end = body_start if body_start is not None else (conclusion["start"] if conclusion else endmatter)
    body_end = min(conclusion["start"] if conclusion else len(text), endmatter)
    conclusion_end = next((s["start"] for s in sections if s["kind"] in {"references", "appendix", "backmatter"}
                           and conclusion and s["start"] > conclusion["start"]), len(text))
    def joined(start, end):
        return "\n\n".join(s["text"] for s in sections if start <= s["start"] < end and s["text"]).strip() or None
    first_references = next((s for s in sections if s["kind"] == "references"), None)
    references_end = next((s["start"] for s in sections if first_references
                           and s["start"] > first_references["start"]
                           and s["kind"] in {"appendix", "backmatter"}), len(text))
    return {"abstract_extracted": abstract["text"] if abstract else None,
            "abstract_variants": [{"heading": s["heading"], "text": s["text"]} for s in abstracts],
            "introduction": joined(intro["start"], intro_end) if intro else None,
            "body": joined(body_start, body_end) if body_start is not None else None,
            "conclusion": joined(conclusion["start"], conclusion_end) if conclusion else None,
            "references_text": joined(first_references["start"], references_end) if first_references else None,
            "appendix_text": "\n\n".join(s["text"] for s in sections if s["kind"] == "appendix" and s["text"]) or None,
            "captions": captions, "sections": sections}
