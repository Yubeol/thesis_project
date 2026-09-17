from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import logging
from pathlib import Path
import time

from transformer.configs.config import (
    TrainingConfig,
)

from transformer.training.data import (
    TokenizedDataset,
    dataset_fingerprint,
    load_splits,
)

from transformer.training.runtime import (
    check_device,
    package_versions,
)


def validate_receipt(
    path,
    signature,
):
    if not path.is_file():
        raise ValueError(
            "Full training requires a successful matching dry-run. "
            "Run --dry-run first and pass --dry-run-receipt "
            "<dry-output>/dry_run_success.json"
        )

    receipt = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if (
        receipt.get(
            "status"
        )
        != "passed"
        or receipt.get(
            "signature"
        )
        != signature
    ):
        raise ValueError(
            "Dry-run receipt does not match this "
            "dataset/model/device/configuration. "
            "Repeat --dry-run with the full-run settings."
        )


def code_fingerprint():
    root = (
        Path(
            __file__
        ).resolve().parents[1]
    )

    return {
        str(
            path.relative_to(
                root
            )
        ).replace(
            "\\",
            "/",
        ): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()

        for path in sorted(
            root.rglob(
                "*.py"
            )
        )
    }


def train(
    args,
    config,
):
    import torch

    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
    )

    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        TrainerCallback,
        set_seed,
    )

    config.validate()

    set_seed(
        config.seed
    )

    torch.set_num_threads(
        config.cpu_threads
    )

    device = check_device(
        args.device,
        config.fp16,
    )

    splits = load_splits(
        args.dataset_dir
    )

    hashes = dataset_fingerprint(
        args.dataset_dir
    )

    if (
        args.output_dir.exists()
        and any(
            args.output_dir.iterdir()
        )
    ):
        raise ValueError(
            "Output directory is not empty; use a new --output-dir "
            "(previous model will not be overwritten)"
        )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            config.model_name,
            revision=config.model_revision,
        )
    )

    model = (
        AutoModelForSeq2SeqLM.from_pretrained(
            config.model_name,
            revision=config.model_revision,
            torch_dtype=torch.float32,
        )
    )

    if not (
        model.config.is_encoder_decoder
    ):
        raise ValueError(
            "This trainer requires an encoder-decoder model"
        )

    model_commit = getattr(
        model.config,
        "_commit_hash",
        None,
    )

    signature = {
        "config": (
            config.fingerprint_fields()
        ),
        "dataset": hashes,
        "model_commit": (
            model_commit
        ),
        "device": device,
        "versions": (
            package_versions()
        ),
        "code": (
            code_fingerprint()
        ),
    }

    if not args.dry_run:
        validate_receipt(
            args.dry_run_receipt,
            signature,
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path = (
        args.output_dir
        / "training.log"
    )

    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(
                log_path,
                encoding="utf-8",
            ),
            logging.StreamHandler(),
        ],
        force=True,
    )

    started = (
        time.monotonic()
    )

    train_rows = (
        splits[
            "train"
        ]
    )

    validation_rows = (
        splits[
            "validation"
        ]
    )

    if args.dry_run:
        train_rows = train_rows[
            : max(
                8,
                (
                    2
                    * config.batch_size
                    * config.gradient_accumulation_steps
                ),
            )
        ]

        validation_rows = (
            validation_rows[:2]
        )

    train_data = TokenizedDataset(
        train_rows,
        tokenizer,
        config,
    )

    validation_data = (
        TokenizedDataset(
            validation_rows,
            tokenizer,
            config,
        )
    )

    model.config.use_cache = False

    model = get_peft_model(
        model,
        LoraConfig(
            task_type=(
                TaskType.SEQ_2_SEQ_LM
            ),
            r=config.lora_rank,
            lora_alpha=(
                config.lora_alpha
            ),
            lora_dropout=0.05,
            target_modules=[
                "q",
                "v",
            ],
            bias="none",
        ),
    )

    model.enable_input_require_grads()

    trainable, total = (
        model.get_nb_trainable_parameters()
    )

    print(
        (
            f"TRAINABLE_PARAMETERS={trainable} "
            f"TOTAL_PARAMETERS={total}"
        ),
        flush=True,
    )


    class LogCallback(
        TrainerCallback
    ):
        def on_log(
            self,
            arguments,
            state,
            control,
            logs=None,
            **kwargs,
        ):
            record = {
                "step": (
                    state.global_step
                ),
                "epoch": (
                    state.epoch
                ),
                **(
                    logs
                    or {}
                ),
            }

            logging.info(
                json.dumps(
                    record
                )
            )


    training_args = (
        Seq2SeqTrainingArguments(
            output_dir=str(
                args.output_dir
                / "checkpoints"
            ),

            num_train_epochs=(
                config.epochs
            ),

            max_steps=(
                2
                if args.dry_run
                else -1
            ),

            per_device_train_batch_size=(
                config.batch_size
            ),

            per_device_eval_batch_size=1,

            gradient_accumulation_steps=(
                config.gradient_accumulation_steps
            ),

            learning_rate=(
                config.learning_rate
            ),

            seed=config.seed,

            data_seed=config.seed,

            use_cpu=(
                device[
                    "device"
                ]
                == "cpu"
            ),

            fp16=config.fp16,

            bf16=False,

            gradient_checkpointing=True,

            gradient_checkpointing_kwargs={
                "use_reentrant": False
            },

            optim="adamw_torch",

            eval_strategy=(
                "steps"
                if args.dry_run
                else "epoch"
            ),

            eval_steps=(
                1
                if args.dry_run
                else None
            ),

            save_strategy=(
                "steps"
                if args.dry_run
                else "epoch"
            ),

            save_steps=1,

            save_total_limit=2,

            logging_strategy="steps",

            logging_steps=(
                1
                if args.dry_run
                else 10
            ),

            logging_first_step=True,

            report_to="none",

            dataloader_num_workers=0,

            dataloader_pin_memory=False,

            predict_with_generate=False,

            load_best_model_at_end=True,

            metric_for_best_model=(
                "eval_loss"
            ),

            greater_is_better=False,

            label_names=[
                "labels"
            ],

            disable_tqdm=True,
        )
    )

    trainer = Seq2SeqTrainer(
        model=model,

        args=training_args,

        train_dataset=(
            train_data
        ),

        eval_dataset=(
            validation_data
        ),

        processing_class=(
            tokenizer
        ),

        data_collator=(
            DataCollatorForSeq2Seq(
                tokenizer,
                model=model,
                label_pad_token_id=-100,
            )
        ),

        callbacks=[
            LogCallback()
        ],
    )

    before = (
        trainer.evaluate()
    )

    result = (
        trainer.train()
    )

    after = (
        trainer.evaluate()
    )

    checkpoints = list(
        (
            args.output_dir
            / "checkpoints"
        ).glob(
            "checkpoint-*/trainer_state.json"
        )
    )

    if (
        trainer.state.global_step
        < 1
        or not checkpoints
    ):
        raise RuntimeError(
            "Training did not save a real optimizer checkpoint"
        )

    import math

    losses = (
        result.training_loss,
        before[
            "eval_loss"
        ],
        after[
            "eval_loss"
        ],
    )

    if not all(
        math.isfinite(
            value
        )
        for value
        in losses
    ):
        raise RuntimeError(
            "Non-finite loss: dry-run/full training "
            "cannot be marked successful"
        )

    trainer.save_state()

    # LoRA adapter 별도 저장
    model.save_pretrained(
        args.output_dir
        / "adapter",
        safe_serialization=True,
    )

    # GTX 1050 2GB에서 merge 순간 VRAM 중복 사용을 피하기 위해 CPU로 이동
    model.to(
        "cpu"
    )

    merged = (
        model.merge_and_unload()
    )

    merged.config.use_cache = True

    # 병합된 전체 모델 저장
    merged.save_pretrained(
        args.output_dir,
        safe_serialization=True,
    )

    tokenizer.save_pretrained(
        args.output_dir
    )

    metadata = {
        "status": "trained",

        "mode": (
            "dry-run"
            if args.dry_run
            else "full"
        ),

        "config": (
            asdict(
                config
            )
        ),

        "signature": (
            signature
        ),

        "created_at": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "samples": {
            "train": (
                len(
                    train_rows
                )
            ),

            "validation": (
                len(
                    validation_rows
                )
            ),

            "test_held_out": (
                len(
                    splits[
                        "test"
                    ]
                )
            ),
        },

        "steps": (
            trainer.state.global_step
        ),

        "trainable_parameters": (
            trainable
        ),

        "validation_before": (
            before
        ),

        "validation_after": (
            after
        ),

        "training_loss": (
            result.training_loss
        ),

        "elapsed_seconds": (
            round(
                time.monotonic()
                - started,
                2,
            )
        ),

        "peak_gpu_allocated_mib": (
            round(
                torch.cuda.max_memory_allocated()
                / 1024**2
            )
            if (
                device[
                    "device"
                ]
                == "cuda"
            )
            else None
        ),

        "quality_note": (
            "Dry-run validates execution only. "
            "Section-level weak supervision and validation loss "
            "do not establish final draft quality."
        ),
    }

    metadata_path = (
        args.output_dir
        / "training_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Trainer/model GPU memory 정리
    del trainer
    del merged
    del model

    gc.collect()

    if (
        device[
            "device"
        ]
        == "cuda"
    ):
        torch.cuda.empty_cache()

    # 저장된 모델을 다시 실제 public inference 경로로 로딩하여 검증
    from transformer.inference.generate import (
        DraftGenerator,
    )

    generator = DraftGenerator(
        args.output_dir,
        device=device[
            "device"
        ],
    )

    # validation sample에는 section 필드가 들어있다.
    # 그러나 여기서는 모델이 3-section 초안을 실제로
    # 생성 가능한지 검사해야 하므로 section만 제거한다.
    sample_input = dict(
        validation_rows[
            0
        ][
            "input"
        ]
    )

    sample_input.pop(
        "section",
        None,
    )

    draft = generator.generate(
        **sample_input
    )

    (
        args.output_dir
        / "sample_draft.txt"
    ).write_text(
        draft
        + "\n",
        encoding="utf-8",
    )

    (
        args.output_dir
        / "generation_diagnostics.json"
    ).write_text(
        json.dumps(
            generator.last_diagnostics,
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.dry_run:
        receipt = {
            "status": (
                "passed"
            ),

            "signature": (
                signature
            ),

            "steps": (
                metadata[
                    "steps"
                ]
            ),

            "model_reload": True,

            "created_at": (
                metadata[
                    "created_at"
                ]
            ),
        }

        (
            args.output_dir
            / "dry_run_success.json"
        ).write_text(
            json.dumps(
                receipt,
                indent=2,
            ),
            encoding="utf-8",
        )

    print(
        json.dumps(
            {
                "status": "passed",

                "mode": (
                    metadata[
                        "mode"
                    ]
                ),

                "output_dir": (
                    str(
                        args.output_dir
                    )
                ),

                "validation_loss": (
                    after[
                        "eval_loss"
                    ]
                ),

                "steps": (
                    metadata[
                        "steps"
                    ]
                ),
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "LoRA section-level first-draft training "
            "on CPU or Pascal CUDA; JSONL input only"
        )
    )

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(
            "data/training"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    parser.add_argument(
        "--dry-run-receipt",
        type=Path,
        default=Path(
            "artifacts/transformer_dry_run/"
            "dry_run_success.json"
        ),
    )

    parser.add_argument(
        "--device",
        choices=(
            "auto",
            "cpu",
            "cuda",
        ),
        default="auto",
    )

    defaults = (
        TrainingConfig()
    )

    for name, value in (
        asdict(
            defaults
        ).items()
    ):
        parser.add_argument(
            "--"
            + name.replace(
                "_",
                "-",
            ),

            default=value,

            **(
                {
                    "action": (
                        "store_true"
                    )
                }
                if isinstance(
                    value,
                    bool,
                )
                else {
                    "type": (
                        type(
                            value
                        )
                    )
                }
            ),
        )

    args = (
        parser.parse_args()
    )

    if (
        args.output_dir
        is None
    ):
        args.output_dir = Path(
            (
                "artifacts/transformer_dry_run"
            )
            if args.dry_run
            else (
                "artifacts/transformer_model"
            )
        )

    config = TrainingConfig(
        **{
            key: getattr(
                args,
                key,
            )
            for key
            in asdict(
                defaults
            )
        }
    )

    may_write_failure = (
        not args.output_dir.exists()
        or not any(
            args.output_dir.iterdir()
        )
    )

    try:
        train(
            args,
            config,
        )

    except (
        RuntimeError,
        ValueError,
        OSError,
    ) as exc:
        message = str(
            exc
        )

        if (
            "out of memory"
            in message.lower()
        ):
            message = (
                "CUDA/RAM out of memory. "
                "Close GPU-heavy applications, keep batch-size 1, "
                "and reduce input/target lengths "
                "(for example 320/256), then repeat dry-run. "
                "LoRA and checkpointing are already enabled. "
                "Local CPU fallback is explicit: --device cpu. "
                "GPU workflow never silently falls back."
            )

        if (
            may_write_failure
            and args.output_dir.exists()
        ):
            (
                args.output_dir
                / "failure.json"
            ).write_text(
                json.dumps(
                    {
                        "error_type": (
                            type(
                                exc
                            ).__name__
                        ),

                        "message": (
                            message
                        ),
                    }
                ),
                encoding="utf-8",
            )

        parser.exit(
            1,
            message
            + "\n",
        )


if __name__ == "__main__":
    main()