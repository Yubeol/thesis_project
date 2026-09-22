import re
from io import BytesIO
from functools import lru_cache
import os
from pathlib import Path
from urllib.parse import quote
from xml.sax.saxutils import escape

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from backend.app.schemas.generate import (
    DownloadDocxRequest,
    DownloadPdfRequest,
)


router = APIRouter(
    prefix="/api",
    tags=["paper-download"],
)


def _add_body_text(
    document: Document,
    text: str,
) -> None:
    paragraphs = re.split(
        r"\n\s*\n",
        text.strip(),
    )

    for content in paragraphs:
        content = content.strip()

        if not content:
            continue

        paragraph = document.add_paragraph(
            content
        )

        paragraph.paragraph_format.space_after = Pt(8)


@router.post("/download/docx")
def download_docx(
    request: DownloadDocxRequest,
):
    document = Document()

    normal_style = document.styles["Normal"]
    normal_style.font.size = Pt(10.5)

    # 제목
    title_paragraph = document.add_paragraph()
    title_paragraph.alignment = (
        WD_ALIGN_PARAGRAPH.CENTER
    )

    title_run = title_paragraph.add_run(
        request.draft.title
    )

    title_run.bold = True
    title_run.font.size = Pt(18)

    # 팀 정보
    team_paragraph = document.add_paragraph()

    team_paragraph.alignment = (
        WD_ALIGN_PARAGRAPH.RIGHT
    )

    team_run = team_paragraph.add_run(
        "Team C (류민규, 박수암, 이혜림)"
    )

    team_run.font.size = Pt(10)

    document.add_paragraph()

    # 서론
    document.add_heading(
        "서론",
        level=1,
    )
    _add_body_text(
        document,
        request.draft.introduction,
    )

    # 본론
    document.add_heading(
        "본론",
        level=1,
    )
    _add_body_text(
        document,
        request.draft.body,
    )

    # 결론
    document.add_heading(
        "결론",
        level=1,
    )
    _add_body_text(
        document,
        request.draft.conclusion,
    )

    # 근거 자료
    if request.sources:
        document.add_heading(
            "근거 자료",
            level=1,
        )

        for index, source in enumerate(
            request.sources,
            start=1,
        ):
            source_type = (
                "논문"
                if source.type == "paper"
                else "뉴스"
            )

            paragraph = document.add_paragraph()

            paragraph.add_run(
                f"{index}. [{source_type}] "
            ).bold = True

            paragraph.add_run(
                source.title
            )

            document.add_paragraph(
                source.url
            )

    buffer = BytesIO()
    document.save(buffer)
    buffer.seek(0)

    safe_title = re.sub(
        r'[\\/:*?"<>|]',
        "",
        request.draft.title,
    ).strip()

    if not safe_title:
        safe_title = "generated_paper"

    filename = f"{safe_title}_teamC.docx"
    encoded_filename = quote(filename)

    headers = {
        "Content-Disposition": (
            "attachment; "
            'filename="generated_paper.docx"; '
            f"filename*=UTF-8''{encoded_filename}"
        )
    }

    return StreamingResponse(
        buffer,
        media_type=(
            "application/vnd.openxmlformats-"
            "officedocument.wordprocessingml.document"
        ),
        headers=headers,
    )


@lru_cache(maxsize=1)
def _pdf_korean_font() -> str:
    """Register a Windows TrueType font so Korean glyphs are embedded in the PDF."""
    font_dir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"

    for filename in ("malgun.ttf", "NanumGothic.ttf"):
        font_path = font_dir / filename
        if font_path.is_file():
            pdfmetrics.registerFont(TTFont("PaperKorean", str(font_path)))
            return "PaperKorean"

    raise HTTPException(
        status_code=503,
        detail="PDF 다운로드에 필요한 한글 TrueType 글꼴을 찾을 수 없습니다.",
    )


def _pdf_paragraphs(text: str, style: ParagraphStyle) -> list[Paragraph]:
    paragraphs = []
    for content in re.split(r"\n\s*\n", text.strip()):
        if content.strip():
            safe_text = escape(content.strip()).replace("\n", "<br/>")
            paragraphs.append(Paragraph(safe_text, style))
    return paragraphs


@router.post("/download/pdf")
def download_pdf(request: DownloadPdfRequest):
    font_name = _pdf_korean_font()
    normal = ParagraphStyle(
        "PaperBody", fontName=font_name, fontSize=10.5, leading=17,
        spaceAfter=10, wordWrap="CJK", splitLongWords=True,
    )
    title_style = ParagraphStyle(
        "PaperTitle", parent=normal, fontSize=17, leading=25,
        alignment=TA_CENTER, spaceAfter=15,
    )
    team_style = ParagraphStyle(
        "PaperTeam", parent=normal, fontSize=9, alignment=TA_RIGHT,
        spaceAfter=18,
    )
    heading_style = ParagraphStyle(
        "PaperHeading", parent=normal, fontSize=13, leading=21,
        spaceBefore=17, spaceAfter=8, keepWithNext=True,
    )
    source_style = ParagraphStyle(
        "PaperSource", parent=normal, fontSize=9, leading=14,
        spaceAfter=7,
    )

    story = [
        Paragraph(escape(request.draft.title), title_style),
        Paragraph("Team C (류민규, 박수암, 이혜림)", team_style),
        Spacer(1, 8),
    ]
    for heading, body in (
        ("서론", request.draft.introduction),
        ("본론", request.draft.body),
        ("결론", request.draft.conclusion),
    ):
        story.append(Paragraph(heading, heading_style))
        story.extend(_pdf_paragraphs(body, normal))

    if request.sources:
        story.append(Paragraph("근거 자료", heading_style))
        for index, source in enumerate(request.sources, start=1):
            source_type = "논문" if source.type == "paper" else "뉴스"
            story.extend(_pdf_paragraphs(
                f"{index}. [{source_type}] {source.title}\n{source.url}",
                source_style,
            ))

    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer, pagesize=A4, leftMargin=52, rightMargin=52,
        topMargin=55, bottomMargin=55, title=request.draft.title,
    )
    document.build(story)
    buffer.seek(0)

    safe_title = re.sub(r'[\\/:*?"<>|]', "", request.draft.title).strip(" .")[:80]
    filename = f"{safe_title or 'generated_paper'}_teamC.pdf"
    headers = {
        "Content-Disposition": (
            "attachment; filename=generated_paper.pdf; "
            f"filename*=UTF-8''{quote(filename)}"
        )
    }
    return StreamingResponse(buffer, media_type="application/pdf", headers=headers)
