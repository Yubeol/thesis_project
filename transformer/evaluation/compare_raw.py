"""Compare two local models using the same saved RAG evidence and raw decoding."""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from transformer.dataset.build_grounded_dataset import content_words, quality_check
from transformer.inference.generate import SectionAdapterDraftGenerator
from transformer.preprocessing.prompts import DEFAULT_INSTRUCTION, HEADINGS, encode_input, make_input


def evaluate_model(path: Path, cases: list[dict], device: str) -> list[dict]:
    metadata = json.loads((path / "training_metadata.json").read_text(encoding="utf-8"))
    section_model = metadata.get("architecture") == "section_specific_lora"
    if section_model:
        generator = SectionAdapterDraftGenerator(path, device)
        tokenizer, model = generator.tokenizer, generator.model
    else:
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(path, local_files_only=True).to(device).eval()
    max_input = int(metadata["config"]["max_input_length"])
    max_output = int(metadata["config"]["max_target_length"])
    outputs = []
    for index, case in enumerate(cases, 1):
        item = {
            "case_id": case.get("case_id", index),
            "title": case["title"], "topic": case["topic"],
            "research_question": case["research_question"],
            "paper_evidence": case["paper_evidence"], "news_evidence": case["news_evidence"],
            "sections": {},
        }
        for section in HEADINGS:
            if section_model:
                model.set_adapter(section)
            value = make_input(
                title=case["title"], topic=case["topic"],
                research_question=case["research_question"],
                paper_evidence=case["paper_evidence"], news_evidence=case["news_evidence"],
                instruction=DEFAULT_INSTRUCTION, section=section,
            )
            encoded = encode_input(tokenizer, value, max_input)
            visible = tokenizer.decode(encoded["input_ids"], skip_special_tokens=True)
            inputs = {key: torch.tensor([ids], device=device) for key, ids in encoded.items()}
            with torch.inference_mode():
                ids = model.generate(**inputs, max_new_tokens=max_output, do_sample=False, num_beams=1, no_repeat_ngram_size=3)[0].tolist()
            raw = tokenizer.decode(ids, skip_special_tokens=True)
            diagnostic = quality_check({"input": value}, raw, tokenizer, visible_prompt=visible)
            topic_words = set(content_words(case["topic"]))
            output_words = set(content_words(raw))
            diagnostic["topic_keyword_coverage"] = round(len(topic_words & output_words) / max(1, len(topic_words)), 4)
            diagnostic["evidence_used"] = diagnostic["support_score"] >= 0.24
            diagnostic["generated_tokens"] = len(ids) - int(ids[0] == model.config.decoder_start_token_id)
            diagnostic["eos_reached"] = tokenizer.eos_token_id in ids
            diagnostic["insufficient_evidence"] = "insufficient evidence" in raw.lower()
            diagnostic["input_tokens"] = len(encoded["input_ids"])
            item["sections"][section] = {"raw": raw, "diagnostics": diagnostic}
        outputs.append(item)
        if index % 5 == 0 or index == len(cases):
            print(json.dumps({"model": str(path), "completed_cases": index, "total_cases": len(cases)}), flush=True)
    del model
    if section_model:
        del generator
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return outputs


def summarize(rows: list[dict]) -> dict:
    output = {"cases": len(rows), "sections": {}, "same_section_outputs": {}}
    for section in HEADINGS:
        diagnostics = [r["sections"][section]["diagnostics"] for r in rows]
        reasons = Counter(reason for d in diagnostics for reason in d["rejection_reason"])
        output["sections"][section] = {
            "normal": sum(d["grounding_pass"] for d in diagnostics),
            "rejection_reasons": dict(reasons),
            "unsupported_number": sum(bool(d["unsupported_numbers"]) for d in diagnostics),
            "unsupported_named_entity": sum(bool(d["unsupported_entities"]) for d in diagnostics),
            "evidence_used": sum(d["evidence_used"] for d in diagnostics),
            "mean_topic_keyword_coverage": round(sum(d["topic_keyword_coverage"] for d in diagnostics) / max(1, len(diagnostics)), 3),
            "insufficient_evidence": sum(d["insufficient_evidence"] for d in diagnostics),
            "mean_generated_tokens": round(sum(d["generated_tokens"] for d in diagnostics) / max(1, len(diagnostics)), 1),
            "eos_reached": sum(d["eos_reached"] for d in diagnostics),
        }
    for first, second in (("Introduction", "Body"), ("Introduction", "Conclusion"), ("Body", "Conclusion")):
        output["same_section_outputs"][f"{first}=={second}"] = sum(
            r["sections"][first]["raw"].strip() == r["sections"][second]["raw"].strip()
            for r in rows
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=Path("artifacts/transformer_quality_30epoch/transformer_raw_results.jsonl"))
    parser.add_argument("--baseline", type=Path, default=Path("artifacts/transformer_model_30epoch"))
    parser.add_argument("--candidate", type=Path, default=Path("artifacts/transformer_model_grounded_shared_v1"))
    parser.add_argument("--section-model", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/transformer_grounded_comparison_v1"))
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Comparison output already exists; use a new path")
    cases = [json.loads(line) for line in args.cases.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(cases) != 15:
        raise ValueError("Expected exactly 15 saved RAG cases")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    baseline = evaluate_model(args.baseline, cases, device)
    candidate = evaluate_model(args.candidate, cases, device)
    section_rows = evaluate_model(args.section_model, cases, device) if args.section_model else None
    args.output_dir.mkdir(parents=True)
    models = [("baseline", baseline), ("candidate", candidate)]
    if section_rows is not None:
        models.append(("section_specific", section_rows))
    for label, rows in models:
        with (args.output_dir / f"{label}_raw.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    base_metadata = json.loads((args.baseline / "training_metadata.json").read_text(encoding="utf-8"))
    cand_metadata = json.loads((args.candidate / "training_metadata.json").read_text(encoding="utf-8"))
    report = {
        "device": device, "cases": len(cases), "decoding": {"max_new_tokens": 384, "do_sample": False, "num_beams": 1, "no_repeat_ngram_size": 3},
        "baseline": {**summarize(baseline), "validation_loss": base_metadata["validation_after"]["eval_loss"]},
        "candidate": {**summarize(candidate), "validation_loss": cand_metadata["validation_after"]["eval_loss"]},
    }
    if section_rows is not None:
        section_metadata = json.loads((args.section_model / "training_metadata.json").read_text(encoding="utf-8"))
        report["section_specific"] = {
            **summarize(section_rows),
            "validation_loss_by_section": {name: section_metadata["sections"][name]["validation_after"] for name in HEADINGS},
        }
    (args.output_dir / "comparison_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    examples = [
        {"case_id": baseline[i]["case_id"], "title": baseline[i]["title"],
         "baseline": baseline[i]["sections"], "candidate": candidate[i]["sections"]}
        for i in range(3)
    ]
    (args.output_dir / "side_by_side_examples.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
