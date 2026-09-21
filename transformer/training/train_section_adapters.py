"""Experimental section LoRAs: one frozen backbone, three adapter-only checkpoints."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import gc
import hashlib
import json
from pathlib import Path

from peft import PeftModel
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

from transformer.configs.config import TrainingConfig
from transformer.preprocessing.prompts import HEADINGS
from transformer.training.data import TokenizedDataset, dataset_fingerprint, load_splits
from transformer.training.runtime import check_device


def train_adapters(dataset_dir: Path, shared_path: Path, output_dir: Path, epochs: float = 5.0) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Section adapter output already exists; it will not be overwritten")
    if not (shared_path / "adapter" / "adapter_config.json").is_file():
        raise ValueError("Shared LoRA adapter is required as the section warm start")
    rows = load_splits(dataset_dir)
    config = replace(TrainingConfig(), epochs=epochs)
    config.validate()
    check_device("cuda")
    set_seed(config.seed)
    torch.set_num_threads(config.cpu_threads)
    tokenizer = AutoTokenizer.from_pretrained(shared_path, local_files_only=True)
    output_dir.mkdir(parents=True)
    results = {}
    for section in HEADINGS:
        section_rows = {split: [r for r in rows[split] if r["section"] == section] for split in rows}
        if not all(section_rows.values()):
            raise ValueError(f"Section {section} has an empty split")
        base = AutoModelForSeq2SeqLM.from_pretrained(config.model_name, revision=config.model_revision)
        base.config.use_cache = False
        if section == HEADINGS[0]:
            base.save_pretrained(output_dir / "backbone", safe_serialization=True)
            tokenizer.save_pretrained(output_dir / "backbone")
        model = PeftModel.from_pretrained(base, shared_path / "adapter", is_trainable=True)
        model.enable_input_require_grads()
        train_data = TokenizedDataset(section_rows["train"], tokenizer, config)
        val_data = TokenizedDataset(section_rows["validation"], tokenizer, config)
        section_dir = output_dir / section.lower()
        args = Seq2SeqTrainingArguments(
            output_dir=str(section_dir / "checkpoints"),
            num_train_epochs=config.epochs,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            seed=config.seed, data_seed=config.seed,
            fp16=False, bf16=False, use_cpu=False,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            optim="adamw_torch", eval_strategy="epoch", save_strategy="epoch",
            save_total_limit=2, logging_strategy="epoch", report_to="none",
            dataloader_num_workers=0, dataloader_pin_memory=False,
            predict_with_generate=False, load_best_model_at_end=True,
            metric_for_best_model="eval_loss", greater_is_better=False,
            label_names=["labels"], disable_tqdm=True,
        )
        trainer = Seq2SeqTrainer(
            model=model, args=args, train_dataset=train_data, eval_dataset=val_data,
            processing_class=tokenizer,
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100),
        )
        before = trainer.evaluate()["eval_loss"]
        outcome = trainer.train()
        after = trainer.evaluate()["eval_loss"]
        adapter_path = section_dir / "adapter"
        trainer.model.save_pretrained(adapter_path, safe_serialization=True)
        results[section] = {
            "train_samples": len(train_data), "validation_samples": len(val_data),
            "test_samples": len(section_rows["test"]),
            "validation_before": before, "validation_after": after,
            "training_loss": outcome.training_loss,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "steps": trainer.state.global_step,
        }
        print(json.dumps({"section": section, **results[section]}), flush=True)
        del trainer, model, base, train_data, val_data
        gc.collect()
        torch.cuda.empty_cache()
    metadata = {
        "status": "trained", "architecture": "section_specific_lora",
        "shared_warm_start": str(shared_path),
        "backbone": "backbone", "adapters": {name: name.lower() + "/adapter" for name in HEADINGS},
        "config": asdict(config), "dataset_sha256": dataset_fingerprint(dataset_dir),
        "signature": {"code": {"preprocessing/prompts.py": hashlib.sha256(
            (Path(__file__).resolve().parents[1] / "preprocessing" / "prompts.py").read_bytes()
        ).hexdigest()}},
        "sections": results,
    }
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("artifacts/transformer_dataset_grounded_v2"))
    parser.add_argument("--shared-path", type=Path, default=Path("artifacts/transformer_model_grounded_shared_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/transformer_model_grounded_sections_v1"))
    parser.add_argument("--epochs", type=float, default=5.0)
    args = parser.parse_args()
    result = train_adapters(args.dataset_dir, args.shared_path, args.output_dir, args.epochs)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
