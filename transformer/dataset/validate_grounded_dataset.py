"""Pre-training gate for the grounded dataset. Exits nonzero on any violation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from transformers import AutoTokenizer

from transformer.dataset.build_grounded_dataset import quality_check
from transformer.preprocessing.prompts import NO_NEWS, encode_input, make_input
from transformer.training.data import load_splits


def validate(directory: Path, tokenizer_path: Path) -> dict:
    splits = load_splits(directory)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    errors = []
    ids = {}
    total = 0
    for split, rows in splits.items():
        for row in rows:
            total += 1
            paper = str(row["source_paper_id"])
            if paper in ids and ids[paper] != split:
                errors.append(f"paper_leakage:{paper}")
            ids[paper] = split
            if not row.get("grounding_pass") or row.get("rejection_reason"):
                errors.append(f"rejected_target_in_training:{paper}:{row['section']}")
            target = row["target"]
            if re.search(r"(?i)^\s*(?:introduction|body|conclusion)\s*:", target):
                errors.append(f"heading:{paper}:{row['section']}")
            if len(tokenizer(target, add_special_tokens=True)["input_ids"]) > 384:
                errors.append(f"target_over_384:{paper}:{row['section']}")
            value = make_input(**row["input"])
            encoded = encode_input(tokenizer, value, 384)
            decoded = tokenizer.decode(encoded["input_ids"], skip_special_tokens=True)
            if len(encoded["input_ids"]) > 384:
                errors.append(f"input_over_384:{paper}:{row['section']}")
            if "Requested Section: " + row["section"] not in decoded:
                errors.append(f"section_missing:{paper}:{row['section']}")
            if not re.search(r"\[PAPER\s+1\].*?Evidence:\s*\S", decoded, re.S):
                errors.append(f"paper_body_missing:{paper}:{row['section']}")
            if row["news_evidence"] != [NO_NEWS] and not re.search(r"\[NEWS\s+1\].*?Evidence:\s*\S", decoded, re.S):
                errors.append(f"news_body_missing:{paper}:{row['section']}")
            check = quality_check(row, target, tokenizer, visible_prompt=decoded)
            if not check["grounding_pass"]:
                errors.append(f"visible_evidence_quality:{paper}:{row['section']}:{','.join(check['rejection_reason'])}")
    return {"passed": not errors, "samples": total, "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("artifacts/transformer_dataset_grounded_v1"))
    parser.add_argument("--tokenizer-path", type=Path, default=Path("artifacts/transformer_model_30epoch"))
    args = parser.parse_args()
    report = validate(args.dataset_dir, args.tokenizer_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
