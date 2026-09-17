from __future__ import annotations

import argparse
from functools import lru_cache
import json
import os
from pathlib import Path
import re
import threading
import warnings

from transformer.configs.config import TrainingConfig
from transformer.preprocessing.prompts import HEADINGS, encode_input, make_input, parse_sections

ABSTENTION = "Insufficient evidence to generate this section reliably."
GENERIC_CAPITALIZED = {
    "A", "An", "The", "This", "That", "These", "Those", "It", "Its", "We", "Our", "Their",
    "In", "On", "At", "By", "From", "For", "To", "As", "With", "Without", "Of", "And", "Or",
    "However", "Therefore", "Overall", "Furthermore", "Moreover", "Although", "While", "When",
    "Based", "Evidence", "Results", "Findings", "Further", "Future", "Research", "Study", "Studies",
    "Introduction", "Body", "Conclusion", "Insufficient",
}


def format_draft(raw, evidence):
    sections = parse_sections(raw)
    diagnostics = {"missing_sections": [], "rejected_sections": [], "rejection_reasons": {}, "raw_output_had_headings": bool(sections)}
    if not sections and raw.strip():
        sections["Body"] = raw.strip()
    evidence_numbers = set(re.findall(r"\b\d+(?:[.,]\d+)*\b", evidence))
    evidence_words = set(re.findall(r"[\w’'-]+", evidence.casefold()))
    output = []
    for name in HEADINGS:
        text = sections.get(name, "")
        if not text:
            diagnostics["missing_sections"].append(name)
            text = ABSTENTION
        else:
            numbers = set(re.findall(r"\b\d+(?:[.,]\d+)*\b", text))
            # Citation assembly belongs to the downstream evidence-aware finalizer.
            citation = re.search(r"\[\s*\d+(?:\s*[,;–-]\s*\d+)*\s*\]|\([^)]*\b(?:19|20)\d{2}[^)]*\)", text)
            # Conservative name heuristic: reject unseen capitalized tokens (including
            # acronyms) rather than pass an invented author/institution through silently.
            capitalized = set(re.findall(r"\b[A-Z][A-Za-z’'-]*\b", text)) - GENERIC_CAPITALIZED
            unseen_names = {word for word in capitalized if word.casefold() not in evidence_words}
            reasons = []
            if numbers - evidence_numbers:
                reasons.append("unsupported_number")
            if citation:
                reasons.append("citation_requires_downstream_verification")
            if unseen_names:
                reasons.append("unsupported_capitalized_token")
            if reasons:
                diagnostics["rejected_sections"].append(name)
                diagnostics["rejection_reasons"][name] = reasons
                text = ABSTENTION
        output.append(name + ":\n" + text)
    return "\n\n".join(output), diagnostics


class DraftGenerator:
    def __init__(self, model_path, device="auto"):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from transformer.training.runtime import check_device
        path = Path(model_path)
        if not (path / "config.json").is_file():
            raise ValueError("Saved model not found. Train first or set TRANSFORMER_MODEL_PATH to the extracted model artifact")
        self.device = check_device(device)["device"]
        metadata_path = path / "training_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        config = metadata.get("config", {})
        self.max_input_length = config.get("max_input_length", TrainingConfig().max_input_length)
        self.max_target_length = config.get("max_target_length", TrainingConfig().max_target_length)
        torch.set_num_threads(config.get("cpu_threads", 8))
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path, local_files_only=True, torch_dtype=torch.float32).to(self.device).eval()
        self.lock = threading.Lock()
        self.last_diagnostics = {}

    def generate(self, title, topic=None, research_question=None, paper_evidence=None,
                 news_evidence=None, instruction=None):
        import torch
        value = make_input(title, topic, research_question, paper_evidence, news_evidence, instruction)
        encoded = encode_input(self.tokenizer, value, self.max_input_length)
        inputs = {key: torch.tensor([val], device=self.device) for key, val in encoded.items()}
        with self.lock, torch.inference_mode():
            tokens = self.model.generate(**inputs, max_new_tokens=self.max_target_length,
                                         do_sample=False, num_beams=1, no_repeat_ngram_size=3)
            raw = self.tokenizer.decode(tokens[0], skip_special_tokens=True)
            evidence = " ".join(value["paper_evidence"] + value["news_evidence"])
            result, self.last_diagnostics = format_draft(raw, evidence)
            if self.last_diagnostics["missing_sections"] or self.last_diagnostics["rejected_sections"]:
                warnings.warn("Generated draft has missing/unsupported sections; explicit abstentions were inserted. Inspect generation diagnostics before use.", stacklevel=2)
        return result


@lru_cache(maxsize=1)
def _generator(path, device):
    return DraftGenerator(path, device)


def generate_draft(title, topic=None, research_question=None, paper_evidence=None,
                   news_evidence=None, instruction=None, *, evidence=None,
                   model_path=None, device="auto") -> str:
    # Existing Agent passes a combined evidence string. Keep it working unchanged.
    if evidence is not None:
        if paper_evidence is not None:
            raise ValueError("Pass either paper_evidence or legacy evidence, not both")
        paper_evidence = [evidence]
    value = make_input(title, topic, research_question, paper_evidence, news_evidence, instruction)
    default_path = Path(__file__).resolve().parents[2] / "artifacts/transformer_model"
    path = Path(model_path or os.environ.get("TRANSFORMER_MODEL_PATH") or default_path).resolve()
    return _generator(str(path), device).generate(**value)


def main():
    parser = argparse.ArgumentParser(description="Generate an evidence-grounded first draft from a saved local model")
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/transformer_model"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--title", required=True)
    parser.add_argument("--topic")
    parser.add_argument("--research-question")
    parser.add_argument("--paper-evidence-file", type=Path)
    parser.add_argument("--news-evidence-file", type=Path)
    parser.add_argument("--instruction")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    def read(path):
        return json.loads(path.read_text(encoding="utf-8-sig")) if path else []
    try:
        result = generate_draft(args.title, args.topic, args.research_question,
                                read(args.paper_evidence_file), read(args.news_evidence_file),
                                args.instruction, model_path=args.model_path, device=args.device)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(result + "\n", encoding="utf-8")
        print(result)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, str(exc) + "\n")


if __name__ == "__main__":
    main()
