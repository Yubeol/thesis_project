import hashlib
import json
from pathlib import Path

from transformer.preprocessing.prompts import encode_input, encode_target, make_input, parse_sections


def dataset_fingerprint(directory):
    return {name: hashlib.sha256((Path(directory) / f"{name}.jsonl").read_bytes()).hexdigest()
            for name in ("train", "validation", "test")}


def load_splits(directory):
    result, seen_ids, seen_samples = {}, {}, {}
    for split in ("train", "validation", "test"):
        path = Path(directory) / f"{split}.jsonl"
        rows = []
        with path.open(encoding="utf-8-sig") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if row.get("paper_id") is None:
                        raise ValueError("paper_id is required")
                    row["input"] = make_input(**row["input"])
                    parse_sections(row["target"], strict=True)
                    ids = {str(row["paper_id"]), *map(str, row.get("source_paper_ids", []))}
                    for paper_id in ids:
                        if paper_id in seen_ids and seen_ids[paper_id] != split:
                            raise ValueError("Paper leakage across splits: " + paper_id)
                        seen_ids[paper_id] = split
                    digest = hashlib.sha256(json.dumps([row["input"], row["target"]], sort_keys=True).encode()).hexdigest()
                    if digest in seen_samples and seen_samples[digest] != split:
                        raise ValueError("Duplicate sample across splits")
                    seen_samples[digest] = split
                    rows.append(row)
                except (TypeError, KeyError, ValueError) as exc:
                    raise ValueError(f"{path.name}:{number}: {exc}") from None
        if not rows:
            raise ValueError(f"{split} dataset is empty")
        result[split] = rows
    manifest_path = Path(directory) / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("sha256") != dataset_fingerprint(directory):
            raise ValueError("Dataset no longer matches its manifest; export a fresh snapshot")
    return result


class TokenizedDataset:
    def __init__(self, rows, tokenizer, config):
        self.rows = []
        for row in rows:
            item = encode_input(tokenizer, row["input"], config.max_input_length)
            item["labels"] = encode_target(tokenizer, row["target"], config.max_target_length)
            self.rows.append(item)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]
