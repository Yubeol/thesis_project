from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from agent.evidence import build_evidence_lists
from agent.retrieval import retrieve_hybrid
from transformer.preprocessing.prompts import (
    HEADINGS,
    encode_input,
    make_input,
    render_prompt,
)
from transformer.training.runtime import check_device


ABSTENTION = "Insufficient evidence to generate this section reliably."
CONTENT_WORD = re.compile(r"[A-Za-z][A-Za-z0-9'-]{2,}")
NUMBER = re.compile(
    r"(?<![\w])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?(?![\w])"
)
POSSIBLE_NAME = re.compile(
    r"\b[A-Z]{2,}\b|\b[A-Z][A-Za-z’'-]+(?:\s+[A-Z][A-Za-z’'-]+)+"
)
GENERIC_NAMES = {
    "A",
    "An",
    "The",
    "This",
    "These",
    "Introduction",
    "Body",
    "Conclusion",
    "Evidence",
    "Research",
    "Study",
    "Studies",
    "Results",
    "Findings",
    "Future",
    "Further",
    "K",
}


CASES = [
    {
        "title": "TikTok and the Global Diffusion of K-pop",
        "topic": "short-video platforms and global cultural diffusion",
        "research_question": "How does TikTok contribute to the global diffusion of K-pop?",
        "query": "TikTok K-pop global diffusion short video platform",
    },
    {
        "title": "Social Media and Transnational K-pop Fandom",
        "topic": "social media and transnational fandom",
        "research_question": "How does social media support transnational participation in K-pop fandom?",
        "query": "social media transnational K-pop fandom participation",
    },
    {
        "title": "K-pop Fandom Activism Online",
        "topic": "digital fandom activism",
        "research_question": "What forms of social activism are associated with online K-pop fandom?",
        "query": "K-pop fandom online activism social movement",
    },
    {
        "title": "Online Fandom and Cultural Participation",
        "topic": "online communities and cultural participation",
        "research_question": "How does online fandom affect cultural participation?",
        "query": "online fandom cultural participation community",
    },
    {
        "title": "Short-Video Platforms in K-pop Marketing",
        "topic": "digital marketing and short-video platforms",
        "research_question": "How are short-video platforms used in K-pop marketing?",
        "query": "K-pop digital marketing short video platforms",
    },
    {
        "title": "Global K-pop Fan Communities",
        "topic": "global fan communities and collective identity",
        "research_question": "How do global K-pop fan communities construct collective identity?",
        "query": "global K-pop fan community collective identity",
    },
    {
        "title": "K-pop and English Language Learning",
        "topic": "K-pop media and informal language learning",
        "research_question": "How does engagement with K-pop media relate to English language learning?",
        "query": "K-pop English language learning social media",
    },
    {
        "title": "The Korean Wave and Youth Cultural Attitudes",
        "topic": "Korean Wave and youth cultural attitudes",
        "research_question": "What does the evidence show about the Korean Wave and youth cultural attitudes?",
        "query": "Korean Wave youth cultural attitudes behavior",
    },
    {
        "title": "Platform Algorithms and K-pop Visibility",
        "topic": "platform algorithms and cultural visibility",
        "research_question": "How do platform algorithms shape the visibility of K-pop content?",
        "query": "platform algorithm K-pop content visibility",
    },
    {
        "title": "Fan Translation and the Circulation of K-pop",
        "topic": "fan translation and global media circulation",
        "research_question": "How does fan translation support the global circulation of K-pop?",
        "query": "K-pop fan translation global circulation",
    },
    {
        "title": "K-pop Dance Challenges and Cultural Diffusion",
        "topic": "dance challenges and cultural diffusion",
        "research_question": "How do online dance challenges participate in K-pop cultural diffusion?",
        "query": "K-pop dance challenge cultural diffusion TikTok",
    },
    {
        "title": "Digital Fandom and Political Participation",
        "topic": "digital fandom and political participation",
        "research_question": "What evidence links digital fandom practices with political participation?",
        "query": "K-pop digital fandom political participation activism",
    },
    {
        "title": "K-pop Fandom and Consumer Engagement",
        "topic": "fandom and consumer engagement",
        "research_question": "How is K-pop fandom associated with consumer engagement?",
        "query": "K-pop fandom consumer engagement social media",
    },
    {
        "title": "Hashtag Activism in K-pop Communities",
        "topic": "hashtags and networked activism",
        "research_question": "How do K-pop communities use hashtags for networked activism?",
        "query": "K-pop hashtag activism Twitter community",
    },
    {
        "title": "YouTube and Instagram in K-pop Promotion",
        "topic": "social platforms and global promotion",
        "research_question": "How are YouTube and Instagram used in the global promotion of K-pop?",
        "query": "YouTube Instagram K-pop global promotion",
    },
]


def normalized(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").casefold()).strip()


def words(text: str) -> list[str]:
    return [match.group(0).casefold() for match in CONTENT_WORD.finditer(text or "")]


def sentences(text: str) -> list[str]:
    return [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+", (text or "").strip())
        if part.strip()
    ]


def numeric_claims(text: str) -> set[str]:
    return {match.group(0).replace(",", "") for match in NUMBER.finditer(text or "")}


def unsupported_names(text: str, support: str) -> list[str]:
    support_lower = support.casefold()
    return sorted(
        {
            name
            for name in POSSIBLE_NAME.findall(text or "")
            if name not in GENERIC_NAMES and name.casefold() not in support_lower
        }
    )


def repetition_score(text: str) -> float:
    tokens = words(text)

    if len(tokens) < 10:
        return 0.0

    ngrams = [tuple(tokens[index:index + 4]) for index in range(len(tokens) - 3)]

    if not ngrams:
        return 0.0

    counts = Counter(ngrams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(ngrams)


def role_mismatch(section: str, text: str) -> tuple[bool, str | None]:
    value = normalized(text)

    if section == "Introduction" and re.match(r"^(in conclusion|to conclude|overall)\b", value):
        return True, "conclusion_marker_in_introduction"

    if section == "Body" and re.match(r"^(in conclusion|to conclude)\b", value):
        return True, "conclusion_marker_in_body"

    if section == "Conclusion":
        markers = {
            "conclusion",
            "conclude",
            "overall",
            "therefore",
            "thus",
            "suggest",
            "indicate",
            "evidence",
            "implication",
            "future",
            "further",
        }

        if not markers.intersection(words(text)):
            return True, "no_concluding_or_synthesis_marker"

    return False, None


def analyze_raw(
    *,
    section: str,
    raw: str,
    title: str,
    support: str,
    evidence_text: str,
    evidence_payload_visible: bool,
    generated_ids: list[int],
    eos_ids: set[int],
) -> dict[str, Any]:
    token_ids = generated_ids[1:] if generated_ids and generated_ids[0] == 0 else generated_ids
    eos_positions = [index for index, token in enumerate(token_ids) if token in eos_ids]
    output_words = words(raw)
    evidence_words = set(words(evidence_text))
    overlap = len(set(output_words) & evidence_words)
    overlap_ratio = overlap / max(1, len(set(output_words)))
    output_sentences = sentences(raw)
    repeated_sentence = len({normalized(item) for item in output_sentences}) < len(output_sentences)
    unsupported_number_values = sorted(numeric_claims(raw) - numeric_claims(support))
    unsupported_name_values = unsupported_names(raw, support)
    bad_role, bad_role_reason = role_mismatch(section, raw)
    stripped = (raw or "").strip()
    broken = bool(
        stripped
        and (
            stripped[-1] not in ".!?\"'’”)]"
            or "�" in stripped
            or any(len(item) > 35 for item in stripped.split())
        )
    )
    title_norm = normalized(title)
    raw_norm = normalized(raw)
    title_repetition = bool(
        raw_norm == title_norm
        or (
            title_norm
            and raw_norm.count(title_norm) >= 2
        )
        or (
            title_norm
            and raw_norm.startswith(title_norm)
            and len(output_words) <= len(words(title)) + 8
        )
    )
    repetition = repetition_score(raw)
    insufficient = ABSTENTION.casefold() in stripped.casefold()

    diagnostics = {
        "empty_output": not bool(stripped),
        "word_count": len(output_words),
        "too_short": len(output_words) < (35 if section == "Body" else 18),
        "title_repetition": title_repetition,
        "broken_or_incomplete_sentence": broken,
        "excessive_repetition": repeated_sentence or repetition >= 0.18,
        "repetition_score": round(repetition, 4),
        "unsupported_number": bool(unsupported_number_values),
        "unsupported_numbers": unsupported_number_values,
        "unsupported_named_entity": bool(unsupported_name_values),
        "unsupported_names": unsupported_name_values,
        "evidence_not_used": not evidence_payload_visible or overlap < 5 or overlap_ratio < 0.10,
        "evidence_payload_visible": evidence_payload_visible,
        "evidence_token_overlap": overlap,
        "evidence_overlap_ratio": round(overlap_ratio, 4),
        "section_role_mismatch": bad_role,
        "section_role_reason": bad_role_reason,
        "insufficient_evidence": insufficient,
        "generated_token_count": len(token_ids),
        "eos_reached": bool(eos_positions),
        "eos_position": eos_positions[0] if eos_positions else None,
        "eos_too_early": bool(eos_positions and eos_positions[0] < 12),
    }

    failure_keys = (
        "empty_output",
        "too_short",
        "title_repetition",
        "broken_or_incomplete_sentence",
        "excessive_repetition",
        "unsupported_number",
        "unsupported_named_entity",
        "evidence_not_used",
        "evidence_payload_visible",
        "section_role_mismatch",
        "insufficient_evidence",
        "eos_too_early",
    )
    diagnostics["normal"] = not any(diagnostics[key] for key in failure_keys)
    return diagnostics


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def dataset_statistics(dataset_dir: Path, tokenizer) -> dict[str, Any]:
    split_rows = {
        split: load_jsonl(dataset_dir / f"{split}.jsonl")
        for split in ("train", "validation", "test")
    }
    all_rows = [row for rows in split_rows.values() for row in rows]
    section_counts = Counter(row["section"] for row in all_rows)
    targets = Counter(normalized(row["target"]) for row in all_rows)
    target_lengths: dict[str, list[tuple[int, int]]] = {name: [] for name in HEADINGS}

    for row in all_rows:
        section = row["section"]
        target_lengths[section].append(
            (
                len(row["target"].split()),
                len(tokenizer.encode(row["target"], add_special_tokens=True)),
            )
        )

    split_paper_ids = {
        split: {str(row["paper_id"]) for row in rows}
        for split, rows in split_rows.items()
    }
    split_source_ids = {
        split: {
            str(source_id)
            for row in rows
            for source_id in row.get("source_paper_ids", [row["paper_id"]])
        }
        for split, rows in split_rows.items()
    }
    leakage = {}

    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        leakage[f"{left}_vs_{right}"] = {
            "paper_ids": sorted(split_paper_ids[left] & split_paper_ids[right]),
            "source_paper_ids": sorted(split_source_ids[left] & split_source_ids[right]),
        }

    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8-sig"))
    real_news = 0
    no_news = 0
    paper_evidence = 0

    for row in all_rows:
        papers = [item for item in row["input"].get("paper_evidence", []) if item.strip()]
        news = row["input"].get("news_evidence", [])
        paper_evidence += int(bool(papers))
        real_news += int(any(item != "[NO_NEWS_EVIDENCE]" for item in news))
        no_news += int(bool(news) and all(item == "[NO_NEWS_EVIDENCE]" for item in news))

    section_lengths = {}

    for section, values in target_lengths.items():
        section_lengths[section] = {
            "samples": len(values),
            "average_words": round(sum(item[0] for item in values) / max(1, len(values)), 2),
            "average_tokens": round(sum(item[1] for item in values) / max(1, len(values)), 2),
            "minimum_words": min((item[0] for item in values), default=0),
            "maximum_words": max((item[0] for item in values), default=0),
        }

    return {
        "manifest": manifest,
        "original_paper_count": manifest.get("source_papers"),
        "eligible_paper_count": manifest.get("eligible_papers"),
        "sample_counts_by_split": {split: len(rows) for split, rows in split_rows.items()},
        "total_samples": len(all_rows),
        "section_counts": dict(section_counts),
        "samples_with_paper_evidence": paper_evidence,
        "samples_with_real_news_evidence": real_news,
        "samples_with_no_news_evidence": no_news,
        "target_lengths_by_section": section_lengths,
        "duplicate_target_groups": sum(1 for count in targets.values() if count > 1),
        "samples_in_duplicate_target_groups": sum(count for count in targets.values() if count > 1),
        "very_short_targets_under_20_words": sum(
            1 for row in all_rows if len(row["target"].split()) < 20
        ),
        "paper_ids_by_split": {split: len(ids) for split, ids in split_paper_ids.items()},
        "split_leakage": leakage,
        "leakage_detected": any(
            details["paper_ids"] or details["source_paper_ids"]
            for details in leakage.values()
        ),
    }


def contract_comparison(
    *,
    dataset_dir: Path,
    model_metadata: dict[str, Any],
    tokenizer,
    sample_inference_evidence: tuple[list[str], list[str]],
) -> dict[str, Any]:
    sample = load_jsonl(dataset_dir / "train.jsonl")[0]
    training_input = sample["input"]
    inference_input = make_input(
        title=CASES[0]["title"],
        topic=CASES[0]["topic"],
        research_question=CASES[0]["research_question"],
        paper_evidence=sample_inference_evidence[0],
        news_evidence=sample_inference_evidence[1],
        instruction="Write only the requested section of a concise academic first draft based on the provided evidence.",
        section="Introduction",
    )
    training_prompt = render_prompt(training_input)
    inference_prompt = render_prompt(inference_input)

    return {
        "training_field_order": list(training_input.keys()),
        "inference_field_order": list(inference_input.keys()),
        "field_order_matches": list(training_input.keys()) == list(inference_input.keys()),
        "training_section": training_input.get("section"),
        "inference_section": inference_input.get("section"),
        "requested_section_present_training": "Requested Section:" in training_prompt,
        "requested_section_present_inference": "Requested Section:" in inference_prompt,
        "training_instruction": training_input.get("instruction"),
        "inference_instruction": inference_input.get("instruction"),
        "instruction_matches": training_input.get("instruction") == inference_input.get("instruction"),
        "training_paper_evidence_items": len(training_input.get("paper_evidence", [])),
        "inference_paper_evidence_items": len(inference_input.get("paper_evidence", [])),
        "training_paper_evidence_has_rag_metadata": any(
            "[PAPER " in item or "Evidence:" in item
            for item in training_input.get("paper_evidence", [])
        ),
        "inference_paper_evidence_has_rag_metadata": any(
            "[PAPER " in item or "Evidence:" in item
            for item in inference_input.get("paper_evidence", [])
        ),
        "training_has_real_news": any(
            item != "[NO_NEWS_EVIDENCE]"
            for item in training_input.get("news_evidence", [])
        ),
        "inference_has_real_news": any(
            item != "[NO_NEWS_EVIDENCE]"
            for item in inference_input.get("news_evidence", [])
        ),
        "tokenizer_special_tokens": tokenizer.special_tokens_map,
        "model_max_input_length": model_metadata["config"]["max_input_length"],
        "model_max_target_length": model_metadata["config"]["max_target_length"],
    }


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    flag_names = (
        "normal",
        "empty_output",
        "too_short",
        "title_repetition",
        "broken_or_incomplete_sentence",
        "excessive_repetition",
        "unsupported_number",
        "unsupported_named_entity",
        "evidence_not_used",
        "section_role_mismatch",
        "insufficient_evidence",
        "eos_too_early",
        "eos_reached",
    )
    output: dict[str, Any] = {"cases": len(results), "sections": {}}

    for section in HEADINGS:
        diagnostics = [row["diagnostics"][section] for row in results]
        counts = {
            name: sum(1 for item in diagnostics if item[name])
            for name in flag_names
        }
        counts["average_generated_tokens"] = round(
            sum(item["generated_token_count"] for item in diagnostics) / max(1, len(diagnostics)),
            2,
        )
        counts["minimum_generated_tokens"] = min(
            (item["generated_token_count"] for item in diagnostics),
            default=0,
        )
        counts["maximum_generated_tokens"] = max(
            (item["generated_token_count"] for item in diagnostics),
            default=0,
        )
        counts["hallucination"] = sum(
            1
            for item in diagnostics
            if item["unsupported_number"] or item["unsupported_named_entity"]
        )
        output["sections"][section] = counts

    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate raw Transformer sections without fallback or LLM Finalizer."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "transformer_quality",
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--use-graph", action="store_true")
    args = parser.parse_args()

    if not 10 <= args.limit <= len(CASES):
        raise ValueError(f"--limit must be between 10 and {len(CASES)}")

    load_dotenv(ROOT / ".env")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_path = args.model_path.resolve()
    dataset_dir = args.dataset_dir.resolve()
    metadata = json.loads(
        (model_path / "training_metadata.json").read_text(encoding="utf-8")
    )
    device_info = check_device(args.device)
    device = device_info["device"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.float32,
    ).to(device).eval()
    max_input_length = int(metadata["config"]["max_input_length"])
    max_target_length = int(metadata["config"]["max_target_length"])
    eos_value = model.generation_config.eos_token_id
    eos_ids = {
        int(item)
        for item in (eos_value if isinstance(eos_value, (list, tuple)) else [eos_value])
        if item is not None
    }
    results: list[dict[str, Any]] = []

    for case_index, case in enumerate(CASES[:args.limit], start=1):
        retrieval = retrieve_hybrid(
            paper_queries=[case["query"]],
            news_queries=[case["query"]],
            paper_top_k=4,
            news_top_k=2,
            use_graph=args.use_graph,
            strict_graph=False,
        )
        paper_evidence, news_evidence = build_evidence_lists(
            retrieval,
            max_chars_per_item=1200,
            max_papers=4,
            max_news=2,
        )

        if not paper_evidence and not news_evidence:
            raise RuntimeError(f"No RAG evidence for case {case_index}: {case['title']}")

        row: dict[str, Any] = {
            "case_id": case_index,
            "title": case["title"],
            "topic": case["topic"],
            "research_question": case["research_question"],
            "paper_evidence": paper_evidence,
            "news_evidence": news_evidence,
            "rag_debug": retrieval.get("debug", {}),
            "diagnostics": {},
            "model_inputs": {},
        }
        raw_by_section: dict[str, str] = {}
        evidence_text = "\n".join([*paper_evidence, *news_evidence])
        support = "\n".join(
            [
                case["title"],
                case["topic"],
                case["research_question"],
                evidence_text,
            ]
        )

        for section in HEADINGS:
            value = make_input(
                title=case["title"],
                topic=case["topic"],
                research_question=case["research_question"],
                paper_evidence=paper_evidence,
                news_evidence=news_evidence,
                instruction=(
                    "Write only the requested section of a concise academic "
                    "first draft based on the provided evidence."
                ),
                section=section,
            )
            encoded = encode_input(tokenizer, value, max_input_length)
            tensors = {
                key: torch.tensor([token_ids], device=device)
                for key, token_ids in encoded.items()
            }

            with torch.inference_mode():
                sequence = model.generate(
                    **tensors,
                    max_new_tokens=max_target_length,
                    do_sample=False,
                    num_beams=1,
                    no_repeat_ngram_size=3,
                )[0]

            generated_ids = sequence.tolist()
            raw = tokenizer.decode(sequence, skip_special_tokens=True).strip()
            raw_by_section[section] = raw
            row[f"{section.casefold()}_raw"] = raw
            decoded_input = tokenizer.decode(
                encoded["input_ids"],
                skip_special_tokens=True,
            )
            payload_visible = (
                decoded_input.count("Evidence:")
                - int("Paper Evidence:" in decoded_input)
                - int("News Evidence:" in decoded_input)
            ) > 0
            row["model_inputs"][section] = {
                "decoded_model_input": decoded_input,
                "input_token_count": len(encoded["input_ids"]),
                "evidence_payload_visible": payload_visible,
            }
            row["diagnostics"][section] = analyze_raw(
                section=section,
                raw=raw,
                title=case["title"],
                support=decoded_input,
                evidence_text=evidence_text,
                evidence_payload_visible=payload_visible,
                generated_ids=generated_ids,
                eos_ids=eos_ids,
            )

        for section in HEADINGS:
            other_sections = [name for name in HEADINGS if name != section]
            similarities = {
                other: SequenceMatcher(
                    None,
                    normalized(raw_by_section[section]),
                    normalized(raw_by_section[other]),
                ).ratio()
                for other in other_sections
            }
            maximum = max(similarities.values(), default=0.0)
            row["diagnostics"][section]["maximum_cross_section_similarity"] = round(maximum, 4)

            if maximum >= 0.90:
                row["diagnostics"][section]["section_role_mismatch"] = True
                row["diagnostics"][section]["section_role_reason"] = "near_duplicate_other_section"
                row["diagnostics"][section]["normal"] = False

        results.append(row)
        print(
            json.dumps(
                {
                    "case": case_index,
                    "title": case["title"],
                    "paper_evidence": len(paper_evidence),
                    "news_evidence": len(news_evidence),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    jsonl_path = args.output_dir / "transformer_raw_results.jsonl"
    jsonl_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results),
        encoding="utf-8",
    )
    summary = aggregate(results)
    summary.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_path": str(model_path),
            "dataset_dir": str(dataset_dir),
            "model_config": metadata.get("config", {}),
            "training_steps": metadata.get("steps"),
            "training_loss": metadata.get("training_loss"),
            "validation_before": metadata.get("validation_before"),
            "validation_after": metadata.get("validation_after"),
            "device": device_info,
            "fallback_used": False,
            "finalizer_used": False,
            "postprocessing_rewrite_used": False,
        }
    )
    write_json(args.output_dir / "quality_summary.json", summary)
    stats = dataset_statistics(dataset_dir, tokenizer)
    write_json(args.output_dir / "dataset_statistics.json", stats)
    first = results[0]
    contract = contract_comparison(
        dataset_dir=dataset_dir,
        model_metadata=metadata,
        tokenizer=tokenizer,
        sample_inference_evidence=(first["paper_evidence"], first["news_evidence"]),
    )
    write_json(args.output_dir / "contract_comparison.json", contract)
    print(
        json.dumps(
            {
                "status": "completed",
                "results": str(jsonl_path),
                "summary": str(args.output_dir / "quality_summary.json"),
                "dataset_statistics": str(args.output_dir / "dataset_statistics.json"),
                "contract": str(args.output_dir / "contract_comparison.json"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
