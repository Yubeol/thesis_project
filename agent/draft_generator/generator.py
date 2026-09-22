import hashlib
import json
import os
from pathlib import Path
import warnings

from transformer.inference import generate_draft

from .grounded_fallback import replace_abstained_draft


_PROMPT_SIGNATURE_KEY = "preprocessing/prompts.py"
_LEGACY_PROMPT_HASHES = {
    # grounded_shared_v1 was trained from a checkout with mixed CRLF/LF.
    # The executable prompt was identical to the repository version.
    "67c5ca378b017d9a81ebce19b356e31bc972a06fcd9efc2c84d8281dd16d343c":
        "4bb5822c0116c1469d5f9206c67d30a8c5e9ae59b18c75183386cf26ef178f64",
}


def _normalized_source_hash(path: Path) -> str:
    source = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(source).hexdigest()


def _model_contract_matches() -> bool:
    root = Path(__file__).resolve().parents[2]
    model_path = Path(
        os.getenv("TRANSFORMER_MODEL_PATH")
        or root / "artifacts" / "transformer_model"
    )
    metadata_path = model_path / "training_metadata.json"
    prompt_path = root / "transformer" / "preprocessing" / "prompts.py"

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        signature = metadata["signature"]
        recorded_raw = signature["code"][_PROMPT_SIGNATURE_KEY].casefold()
        recorded_normalized = signature.get("code_normalized", {}).get(
            _PROMPT_SIGNATURE_KEY
        ) or _LEGACY_PROMPT_HASHES.get(recorded_raw)
        if not recorded_normalized:
            return False
        current = _normalized_source_hash(prompt_path)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False

    return recorded_normalized.casefold() == current.casefold()


def generate_transformer_draft(
    *,
    title: str,
    topic: str | None = None,
    research_question: str | None = None,
    paper_evidence: list[str] | None = None,
    news_evidence: list[str] | None = None,
    instruction: str | None = None,
) -> str:
    """
    Transformer inference 호출을 담당하는 단일 진입점.

    실제 모델 호출은 이 파일에서만 수행한다.
    """

    paper_evidence = paper_evidence or []
    news_evidence = news_evidence or []

    if (
        not paper_evidence
        and not news_evidence
    ):
        raise ValueError(
            "Transformer에 전달할 "
            "paper_evidence 또는 news_evidence가 필요합니다."
        )

    contract_matches = _model_contract_matches()

    if not contract_matches:
        draft, _ = replace_abstained_draft(
            "",
            title=title,
            topic=topic,
            research_question=research_question,
            paper_evidence=paper_evidence,
            news_evidence=news_evidence,
            force=True,
        )
        warnings.warn(
            "Transformer model/code contract does not match; the stale "
            "model was not executed and a grounded extractive fallback "
            "draft was used.",
            stacklevel=2,
        )
        return draft

    draft = generate_draft(
        title=title,
        topic=topic,
        research_question=research_question,
        paper_evidence=paper_evidence,
        news_evidence=news_evidence,
        instruction=instruction,
    )

    if not isinstance(
        draft,
        str,
    ):
        raise TypeError(
            "generate_draft() 반환값은 str이어야 합니다. "
            f"현재 타입: {type(draft).__name__}"
        )

    draft = draft.strip()

    if not draft:
        raise RuntimeError(
            "Transformer가 빈 초안을 반환했습니다."
        )

    draft, fallback_used = replace_abstained_draft(
        draft,
        title=title,
        topic=topic,
        research_question=research_question,
        paper_evidence=paper_evidence,
        news_evidence=news_evidence,
    )

    if fallback_used:
        warnings.warn(
            "Transformer output was malformed or unsupported; a grounded "
            "extractive fallback draft was used.",
            stacklevel=2,
        )

    return draft
