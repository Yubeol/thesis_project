import re
from io import BytesIO
from urllib.parse import quote

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from backend.app.schemas.generate import (
    DownloadDocxRequest,
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