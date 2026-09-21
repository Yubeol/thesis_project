"""Fine-tune only the Conclusion adapter on final encoded service-like inputs."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import shutil

from peft import PeftModel
import torch
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
    Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed,
)

from transformer.configs.config import TrainingConfig
from transformer.dataset.validate_conclusion_service import validate
from transformer.training.data import TokenizedDataset, dataset_fingerprint, load_splits
from transformer.training.runtime import check_device


def train_conclusion(dataset_dir: Path, previous_path: Path, output_dir: Path, epochs: float = 5.0) -> dict:
    if output_dir.exists():
        raise FileExistsError("Output path already exists; prior models will not be overwritten")
    gate = validate(dataset_dir, previous_path / "backbone")
    if not gate["passed"]:
        raise ValueError("Dataset gate failed: " + ", ".join(gate["errors"][:5]))
    rows = load_splits(dataset_dir)
    if any(row["section"] != "Conclusion" for split in rows.values() for row in split):
        raise ValueError("Only Conclusion rows are allowed")
    config = replace(TrainingConfig(), epochs=epochs)
    config.validate()
    check_device("cuda")
    set_seed(config.seed)
    torch.set_num_threads(config.cpu_threads)
    tokenizer = AutoTokenizer.from_pretrained(previous_path / "backbone", local_files_only=True)
    base = AutoModelForSeq2SeqLM.from_pretrained(previous_path / "backbone", local_files_only=True)
    base.config.use_cache = False
    model = PeftModel.from_pretrained(base, previous_path / "conclusion" / "adapter", is_trainable=True)
    model.enable_input_require_grads()
    train_data = TokenizedDataset(rows["train"], tokenizer, config)
    validation_data = TokenizedDataset(rows["validation"], tokenizer, config)
    # Only a fresh destination is allowed. The backbone and two untouched
    # adapters are copied once into the standalone experimental artifact.
    output_dir.mkdir(parents=True)
    shutil.copytree(previous_path / "backbone", output_dir / "backbone")
    for section in ("introduction", "body"):
        shutil.copytree(previous_path / section / "adapter", output_dir / section / "adapter")
    arguments = Seq2SeqTrainingArguments(
        output_dir=str(output_dir / "conclusion" / "checkpoints"),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
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
        model=model, args=arguments, train_dataset=train_data, eval_dataset=validation_data,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100),
    )
    before = trainer.evaluate()["eval_loss"]
    outcome = trainer.train()
    after = trainer.evaluate()["eval_loss"]
    trainer.model.save_pretrained(output_dir / "conclusion" / "adapter", safe_serialization=True)
    previous = json.loads((previous_path / "training_metadata.json").read_text(encoding="utf-8"))
    best_step = int(trainer.state.best_model_checkpoint.rsplit("checkpoint-", 1)[-1]) if trainer.state.best_model_checkpoint else None
    best_epoch = next((entry.get("epoch") for entry in trainer.state.log_history
                       if entry.get("step") == best_step and "eval_loss" in entry), None)
    conclusion = {
        "train_samples": len(train_data), "validation_samples": len(validation_data),
        "test_samples": len(rows["test"]), "validation_before": before,
        "validation_after": after, "training_loss": outcome.training_loss,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_epoch": best_epoch,
        "steps": trainer.state.global_step,
    }
    metadata = {
        "status": "trained", "architecture": "section_specific_lora",
        "inherited_from": str(previous_path), "backbone": "backbone",
        "adapters": {name: name.lower() + "/adapter" for name in ("Introduction", "Body", "Conclusion")},
        "config": asdict(config), "dataset_sha256": dataset_fingerprint(dataset_dir),
        "signature": {"code": {"preprocessing/prompts.py": hashlib.sha256(
            (Path(__file__).resolve().parents[1] / "preprocessing" / "prompts.py").read_bytes()
        ).hexdigest()}},
        "sections": {**previous["sections"], "Conclusion": conclusion},
        "inherited_sections": ["Introduction", "Body"],
    }
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("artifacts/transformer_dataset_conclusion_service_v4"))
    parser.add_argument("--previous-path", type=Path, default=Path("artifacts/transformer_model_grounded_sections_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/transformer_model_grounded_conclusion_service_v1"))
    parser.add_argument("--epochs", type=float, default=5.0)
    args = parser.parse_args()
    result = train_conclusion(args.dataset_dir, args.previous_path, args.output_dir, args.epochs)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
