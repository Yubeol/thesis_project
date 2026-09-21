"""Offline raw Conclusion comparison; never invokes service fallback or Finalizer."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import torch

from transformer.dataset.build_grounded_dataset import content_words, quality_check
from transformer.inference.generate import SectionAdapterDraftGenerator
from transformer.preprocessing.prompts import DEFAULT_INSTRUCTION, encode_input, make_input


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def compare(previous_dir: Path, model_path: Path, output_dir: Path) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Comparison output already exists")
    models = {name: load_rows(previous_dir / f"{name}_raw.jsonl") for name in ("baseline", "candidate", "section_specific")}
    if any(len(rows) != 15 for rows in models.values()):
        raise ValueError("Expected 15 stored service RAG cases per model")
    for index in range(15):
        cases = [rows[index] for rows in models.values()]
        if len({case["case_id"] for case in cases}) != 1 or any(
            case[field] != cases[0][field]
            for case in cases[1:]
            for field in ("title", "topic", "research_question", "paper_evidence", "news_evidence")
        ):
            raise ValueError(f"Stored comparison inputs differ at case {index + 1}")
    generator = SectionAdapterDraftGenerator(model_path, "cuda" if torch.cuda.is_available() else "cpu")
    generator.model.set_adapter("Conclusion")
    tokenizer, model = generator.tokenizer, generator.model
    records = []
    for index, case in enumerate(models["section_specific"], 1):
        value = make_input(
            title=case["title"], topic=case["topic"], research_question=case["research_question"],
            paper_evidence=case["paper_evidence"], news_evidence=case["news_evidence"],
            instruction=DEFAULT_INSTRUCTION, section="Conclusion",
        )
        encoded = encode_input(tokenizer, value, 384)
        visible = tokenizer.decode(encoded["input_ids"], skip_special_tokens=True)
        tensors = {key: torch.tensor([ids], device=generator.device) for key, ids in encoded.items()}
        with torch.inference_mode():
            ids = model.generate(**tensors, max_new_tokens=384, do_sample=False, num_beams=1, no_repeat_ngram_size=3)[0].tolist()
        raw = tokenizer.decode(ids, skip_special_tokens=True)
        diagnostic = quality_check({"input": value}, raw, tokenizer, visible_prompt=visible)
        rq_terms = set(content_words(case["research_question"]))
        diagnostic.update({
            "generated_tokens": len(ids) - int(ids[0] == model.config.decoder_start_token_id),
            "eos_reached": tokenizer.eos_token_id in ids,
            "rq_keyword_overlap": len(rq_terms & set(content_words(raw))),
            "evidence_used": diagnostic["support_score"] >= 0.24,
        })
        records.append({
            "case_id": case["case_id"], "title": case["title"], "research_question": case["research_question"],
            "paper_evidence": case["paper_evidence"], "news_evidence": case["news_evidence"],
            "baseline_raw": models["baseline"][index - 1]["sections"]["Conclusion"]["raw"],
            "shared_raw": models["candidate"][index - 1]["sections"]["Conclusion"]["raw"],
            "previous_section_raw": case["sections"]["Conclusion"]["raw"],
            "new_conclusion_raw": raw, "new_diagnostics": diagnostic,
        })
        if index % 5 == 0:
            print(json.dumps({"completed": index, "total": 15}), flush=True)
    output_dir.mkdir(parents=True)
    with (output_dir / "raw_comparison.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    diagnostics = [r["new_diagnostics"] for r in records]
    report = {
        "cases": 15,
        "conditions": {"input_tokens": 384, "max_new_tokens": 384, "do_sample": False, "num_beams": 1, "no_repeat_ngram_size": 3},
        "new_automatic": {
            "normal": sum(d["grounding_pass"] for d in diagnostics),
            "reasons": dict(Counter(reason for d in diagnostics for reason in d["rejection_reason"])),
            "evidence_used": sum(d["evidence_used"] for d in diagnostics),
            "unsupported_number": sum(bool(d["unsupported_numbers"]) for d in diagnostics),
            "unsupported_entity": sum(bool(d["unsupported_entities"]) for d in diagnostics),
            "rq_keyword_overlap": sum(d["rq_keyword_overlap"] > 0 for d in diagnostics),
            "eos_reached": sum(d["eos_reached"] for d in diagnostics),
            "mean_generated_tokens": round(sum(d["generated_tokens"] for d in diagnostics) / len(diagnostics), 1),
        },
    }
    (output_dir / "automatic_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous-dir", type=Path, default=Path("artifacts/transformer_grounded_comparison_sections_v1"))
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/transformer_model_grounded_conclusion_service_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/transformer_conclusion_service_comparison_v1"))
    args = parser.parse_args()
    print(json.dumps(compare(args.previous_dir, args.model_path, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
