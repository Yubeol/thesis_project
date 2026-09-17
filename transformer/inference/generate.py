from __future__ import annotations

import argparse
from functools import lru_cache
import json
import os
from pathlib import Path
import re
import threading
import warnings

from transformer.configs.config import (
    TrainingConfig,
)

from transformer.preprocessing.prompts import (
    HEADINGS,
    clean_text,
    encode_input,
    make_input,
    normalize_section,
    parse_sections,
)


ABSTENTION = (
    "Insufficient evidence to generate this section reliably."
)

GENERIC_CAPITALIZED = {
    "A",
    "An",
    "The",
    "This",
    "That",
    "These",
    "Those",
    "It",
    "Its",
    "We",
    "Our",
    "Their",
    "In",
    "On",
    "At",
    "By",
    "From",
    "For",
    "To",
    "As",
    "With",
    "Without",
    "Of",
    "And",
    "Or",
    "However",
    "Therefore",
    "Overall",
    "Furthermore",
    "Moreover",
    "Additionally",
    "Although",
    "While",
    "When",
    "Based",
    "Evidence",
    "Results",
    "Findings",
    "Further",
    "Future",
    "Research",
    "Study",
    "Studies",
    "Introduction",
    "Body",
    "Conclusion",
    "Insufficient",
}


def _validation_reasons(
    text,
    support_text,
):
    """
    완벽한 citation validator가 아니라
    Transformer가 명백한 수치/인용/고유명사를
    임의 생성하는 것을 보수적으로 잡기 위한 검사다.

    최종 사실 검증은 downstream LLM/Grounding 단계가 담당한다.
    """
    support_numbers = set(
        re.findall(
            r"\b\d+(?:[.,]\d+)*\b",
            support_text,
        )
    )

    numbers = set(
        re.findall(
            r"\b\d+(?:[.,]\d+)*\b",
            text,
        )
    )

    reasons = []

    if (
        numbers
        - support_numbers
    ):
        reasons.append(
            "unsupported_number"
        )

    citation = re.search(
        (
            r"\[\s*\d+"
            r"(?:\s*[,;–-]\s*\d+)*"
            r"\s*\]"
            r"|"
            r"\([^)]*\b(?:19|20)\d{2}[^)]*\)"
        ),
        text,
    )

    if citation:
        reasons.append(
            "citation_requires_downstream_verification"
        )

    # 대문자로 시작했다는 이유만으로 일반 단어를 모두 거부하면
    # 정상적인 문장까지 잘리는 문제가 생긴다.
    # 따라서 acronym과 2단어 이상의 proper-name 형태만 검사한다.
    possible_names = set(
        re.findall(
            (
                r"\b[A-Z]{2,}\b"
                r"|"
                r"\b[A-Z][A-Za-z’'-]+"
                r"(?:\s+[A-Z][A-Za-z’'-]+)+"
            ),
            text,
        )
    )

    support_lower = (
        support_text.casefold()
    )

    unsupported_names = {
        name
        for name in possible_names
        if (
            name
            not in GENERIC_CAPITALIZED
            and name.casefold()
            not in support_lower
        )
    }

    if unsupported_names:
        reasons.append(
            "unsupported_named_entity"
        )

    return reasons


def format_section(
    raw,
    section,
    support_text,
):
    section = normalize_section(
        section,
        required=True,
    )

    raw = clean_text(
        raw
    )

    if not raw:
        return (
            ABSTENTION,
            {
                "section": section,
                "empty": True,
                "raw_output_had_headings": False,
                "rejected": True,
                "reasons": [
                    "empty_output"
                ],
            },
        )

    # 모델에게 heading을 출력하지 말라고 가르치지만,
    # 혹시 출력해도 최대한 복구
    parsed = parse_sections(
        raw
    )

    had_headings = bool(
        parsed
    )

    if parsed:
        text = parsed.get(
            section
        )

        if (
            not text
            and len(parsed) == 1
        ):
            text = next(
                iter(
                    parsed.values()
                )
            )

        text = (
            text
            or ""
        )

    else:
        text = raw

    text = clean_text(
        text
    )

    if not text:
        return (
            ABSTENTION,
            {
                "section": section,
                "empty": True,
                "raw_output_had_headings": had_headings,
                "rejected": True,
                "reasons": [
                    "missing_requested_section"
                ],
            },
        )

    reasons = (
        _validation_reasons(
            text,
            support_text,
        )
    )

    rejected = bool(
        reasons
    )

    return (
        (
            ABSTENTION
            if rejected
            else text
        ),
        {
            "section": section,
            "empty": False,
            "raw_output_had_headings": had_headings,
            "rejected": rejected,
            "reasons": reasons,
        },
    )


def format_draft(
    raw,
    evidence,
):
    """
    과거 테스트 코드와의 호환을 위해 유지.
    새 generate()에서는 section별 생성 방식을 사용한다.
    """
    sections = parse_sections(
        raw
    )

    diagnostics = {
        "missing_sections": [],
        "rejected_sections": [],
        "rejection_reasons": {},
        "raw_output_had_headings": bool(
            sections
        ),
    }

    if (
        not sections
        and raw.strip()
    ):
        sections[
            "Body"
        ] = raw.strip()

    output = []

    for name in HEADINGS:
        text = sections.get(
            name,
            "",
        )

        if not text:
            diagnostics[
                "missing_sections"
            ].append(
                name
            )

            text = ABSTENTION

        else:
            reasons = (
                _validation_reasons(
                    text,
                    evidence,
                )
            )

            if reasons:
                diagnostics[
                    "rejected_sections"
                ].append(
                    name
                )

                diagnostics[
                    "rejection_reasons"
                ][name] = reasons

                text = ABSTENTION

        output.append(
            name
            + ":\n"
            + text
        )

    return (
        "\n\n".join(
            output
        ),
        diagnostics,
    )


class DraftGenerator:
    def __init__(
        self,
        model_path,
        device="auto",
    ):
        import torch

        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
        )

        from transformer.training.runtime import (
            check_device,
        )

        path = Path(
            model_path
        )

        if not (
            path
            / "config.json"
        ).is_file():
            raise ValueError(
                "Saved model not found. Train first or set "
                "TRANSFORMER_MODEL_PATH to the extracted model artifact"
            )

        self.device = check_device(
            device
        )[
            "device"
        ]

        metadata_path = (
            path
            / "training_metadata.json"
        )

        metadata = (
            json.loads(
                metadata_path.read_text(
                    encoding="utf-8"
                )
            )
            if metadata_path.exists()
            else {}
        )

        config = metadata.get(
            "config",
            {},
        )

        self.max_input_length = (
            config.get(
                "max_input_length",
                TrainingConfig().max_input_length,
            )
        )

        self.max_target_length = (
            config.get(
                "max_target_length",
                TrainingConfig().max_target_length,
            )
        )

        torch.set_num_threads(
            config.get(
                "cpu_threads",
                8,
            )
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                path,
                local_files_only=True,
            )
        )

        self.model = (
            AutoModelForSeq2SeqLM.from_pretrained(
                path,
                local_files_only=True,
                torch_dtype=torch.float32,
            )
            .to(
                self.device
            )
            .eval()
        )

        self.lock = (
            threading.Lock()
        )

        self.last_diagnostics = {}


    def _generate_section_with_diagnostics(
        self,
        title,
        topic=None,
        research_question=None,
        paper_evidence=None,
        news_evidence=None,
        instruction=None,
        section=None,
    ):
        import torch

        section = normalize_section(
            section,
            required=True,
        )

        value = make_input(
            title=title,
            topic=topic,
            research_question=research_question,
            paper_evidence=paper_evidence,
            news_evidence=news_evidence,
            instruction=instruction,
            section=section,
        )

        encoded = encode_input(
            self.tokenizer,
            value,
            self.max_input_length,
        )

        inputs = {
            key: torch.tensor(
                [
                    val
                ],
                device=self.device,
            )
            for key, val
            in encoded.items()
        }

        with (
            self.lock,
            torch.inference_mode(),
        ):
            tokens = (
                self.model.generate(
                    **inputs,
                    max_new_tokens=(
                        self.max_target_length
                    ),
                    do_sample=False,
                    num_beams=1,
                    no_repeat_ngram_size=3,
                )
            )

        raw = (
            self.tokenizer.decode(
                tokens[0],
                skip_special_tokens=True,
            )
        )

        # Title / Topic / RQ도 entity grounding 근거로 인정
        support_text = " ".join(
            [
                value[
                    "title"
                ],
                value[
                    "topic"
                ],
                value[
                    "research_question"
                ],
                *value[
                    "paper_evidence"
                ],
                *value[
                    "news_evidence"
                ],
            ]
        )

        text, diagnostics = (
            format_section(
                raw=raw,
                section=section,
                support_text=support_text,
            )
        )

        diagnostics[
            "raw_output"
        ] = raw

        return (
            text,
            diagnostics,
        )


    def generate_section(
        self,
        title,
        topic=None,
        research_question=None,
        paper_evidence=None,
        news_evidence=None,
        instruction=None,
        section=None,
    ):
        text, diagnostics = (
            self._generate_section_with_diagnostics(
                title=title,
                topic=topic,
                research_question=research_question,
                paper_evidence=paper_evidence,
                news_evidence=news_evidence,
                instruction=instruction,
                section=section,
            )
        )

        normalized = normalize_section(
            section,
            required=True,
        )

        self.last_diagnostics = {
            "mode": (
                "single_section"
            ),
            "sections": {
                normalized: diagnostics
            },
        }

        if diagnostics[
            "rejected"
        ]:
            warnings.warn(
                "Generated section was unsupported or malformed; "
                "an explicit abstention was inserted. "
                "Inspect generation diagnostics.",
                stacklevel=2,
            )

        return text


    def generate(
        self,
        title,
        topic=None,
        research_question=None,
        paper_evidence=None,
        news_evidence=None,
        instruction=None,
        section=None,
    ):
        # 직접 section을 지정한 경우 한 section만 생성 가능
        if section is not None:
            return self.generate_section(
                title=title,
                topic=topic,
                research_question=research_question,
                paper_evidence=paper_evidence,
                news_evidence=news_evidence,
                instruction=instruction,
                section=section,
            )

        # 기본 공개 동작:
        # 같은 모델을 총 3회 호출
        outputs = []
        section_diagnostics = {}

        for requested_section in HEADINGS:
            text, diagnostics = (
                self._generate_section_with_diagnostics(
                    title=title,
                    topic=topic,
                    research_question=research_question,
                    paper_evidence=paper_evidence,
                    news_evidence=news_evidence,
                    instruction=instruction,
                    section=requested_section,
                )
            )

            outputs.append(
                requested_section
                + ":\n"
                + text
            )

            section_diagnostics[
                requested_section
            ] = diagnostics

        self.last_diagnostics = {
            "mode": (
                "three_section_generation"
            ),

            "sections": (
                section_diagnostics
            ),

            "rejected_sections": [
                name
                for name, diagnostics
                in section_diagnostics.items()
                if diagnostics[
                    "rejected"
                ]
            ],
        }

        if self.last_diagnostics[
            "rejected_sections"
        ]:
            warnings.warn(
                "One or more generated sections were unsupported "
                "or malformed; explicit abstentions were inserted. "
                "Inspect generation diagnostics.",
                stacklevel=2,
            )

        return "\n\n".join(
            outputs
        )


@lru_cache(
    maxsize=1
)
def _generator(
    path,
    device,
):
    return DraftGenerator(
        path,
        device,
    )


def generate_draft(
    title,
    topic=None,
    research_question=None,
    paper_evidence=None,
    news_evidence=None,
    instruction=None,
    *,
    evidence=None,
    model_path=None,
    device="auto",
) -> str:
    """
    팀원이 사용하던 공개 API는 그대로 유지한다.

    내부적으로만:
    Introduction
    Body
    Conclusion

    세 번 생성한다.
    """
    if evidence is not None:
        if paper_evidence is not None:
            raise ValueError(
                "Pass either paper_evidence or legacy evidence, not both"
            )

        paper_evidence = [
            evidence
        ]

    value = make_input(
        title=title,
        topic=topic,
        research_question=research_question,
        paper_evidence=paper_evidence,
        news_evidence=news_evidence,
        instruction=instruction,
    )

    default_path = (
        Path(
            __file__
        ).resolve().parents[2]
        / "artifacts"
        / "transformer_model"
    )

    path = Path(
        model_path
        or os.environ.get(
            "TRANSFORMER_MODEL_PATH"
        )
        or default_path
    ).resolve()

    return _generator(
        str(path),
        device,
    ).generate(
        **value
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a three-section evidence-grounded "
            "first draft from a saved local model"
        )
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(
            "artifacts/transformer_model"
        ),
    )

    parser.add_argument(
        "--device",
        choices=(
            "auto",
            "cpu",
            "cuda",
        ),
        default="auto",
    )

    parser.add_argument(
        "--title",
        required=True,
    )

    parser.add_argument(
        "--topic",
    )

    parser.add_argument(
        "--research-question",
    )

    parser.add_argument(
        "--paper-evidence-file",
        type=Path,
    )

    parser.add_argument(
        "--news-evidence-file",
        type=Path,
    )

    parser.add_argument(
        "--instruction",
    )

    parser.add_argument(
        "--output",
        type=Path,
    )

    args = parser.parse_args()

    def read(
        path,
    ):
        if not path:
            return []

        return json.loads(
            path.read_text(
                encoding="utf-8-sig"
            )
        )

    try:
        result = generate_draft(
            title=args.title,
            topic=args.topic,
            research_question=(
                args.research_question
            ),
            paper_evidence=read(
                args.paper_evidence_file
            ),
            news_evidence=read(
                args.news_evidence_file
            ),
            instruction=args.instruction,
            model_path=args.model_path,
            device=args.device,
        )

        if args.output:
            args.output.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            args.output.write_text(
                result
                + "\n",
                encoding="utf-8",
            )

        print(
            result
        )

    except (
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        parser.exit(
            1,
            str(exc)
            + "\n",
        )


if __name__ == "__main__":
    main()