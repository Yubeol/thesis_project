"""Fail-closed gate before training the service-like Conclusion adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from transformers import AutoTokenizer

from transformer.dataset.build_conclusion_service import conclusion_gate, digest
from transformer.preprocessing.prompts import NO_NEWS, encode_input, make_input
from transformer.training.data import load_splits


def validate(directory: Path, tokenizer_path: Path) -> dict:
    splits = load_splits(directory)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    errors = []
    seen = {}
    count = 0
    for split, rows in splits.items():
        for row in rows:
            count += 1
            paper_id = str(row["source_paper_id"])
            if paper_id in seen and seen[paper_id] != split:
                errors.append(f"paper_leakage:{paper_id}")
            seen[paper_id] = split
            if row["section"] != "Conclusion" or row["input"]["section"] != "Conclusion":
                errors.append(f"wrong_section:{paper_id}")
            if not row.get("grounding_pass") or row.get("rejection_reason"):
                errors.append(f"rejected_target_in_training:{paper_id}")
            papers = row["input"]["paper_evidence"]
            news = [x for x in row["input"]["news_evidence"] if x != NO_NEWS]
            if not 2 <= len(papers) <= 4 or len(news) > 2:
                errors.append(f"evidence_count:{paper_id}")
            encoded = encode_input(tokenizer, make_input(**row["input"]), 384)
            decoded = tokenizer.decode(encoded["input_ids"], skip_special_tokens=True)
            if len(encoded["input_ids"]) > 384 or len(tokenizer(row["target"])["input_ids"]) > 384:
                errors.append(f"token_limit:{paper_id}")
            if decoded != row.get("teacher_visible_input") or digest(encoded["input_ids"]) != row.get("teacher_input_hash"):
                errors.append(f"teacher_transformer_input_mismatch:{paper_id}")
            if "Requested Section: Conclusion" not in decoded:
                errors.append(f"section_not_encoded:{paper_id}")
            for kind, entries in (("PAPER", papers), ("NEWS", news)):
                for index in range(1, len(entries) + 1):
                    if not re.search(rf"\[{kind}\s+{index}\][^|]*?Evidence:\s*[^|\s]", decoded, re.S):
                        errors.append(f"{kind.lower()}_body_not_encoded:{paper_id}:{index}")
            reasons = conclusion_gate(row, row["target"], tokenizer, decoded)
            if reasons:
                errors.append(f"target_quality:{paper_id}:{','.join(reasons)}")
    return {"passed": not errors, "samples": count, "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("artifacts/transformer_dataset_conclusion_service_v4"))
    parser.add_argument("--tokenizer-path", type=Path, default=Path("artifacts/transformer_model_30epoch"))
    args = parser.parse_args()
    report = validate(args.dataset_dir, args.tokenizer_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
