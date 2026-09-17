from dataclasses import (
    asdict,
    dataclass,
)

import math


@dataclass(
    frozen=True
)
class TrainingConfig:
    model_name: str = (
        "google/flan-t5-small"
    )

    model_revision: str = (
        "main"
    )

    epochs: float = 3.0

    batch_size: int = 1

    learning_rate: float = (
        3e-4
    )

    max_input_length: int = (
        384
    )

    # 기존 128 → 384
    # 전체 논문이 아니라 section 하나씩 생성하므로
    # GTX 1050 dry-run을 먼저 확인한다.
    max_target_length: int = (
        384
    )

    gradient_accumulation_steps: int = (
        8
    )

    seed: int = 42

    lora_rank: int = 4

    lora_alpha: int = 8

    fp16: bool = False

    cpu_threads: int = 8


    def validate(
        self,
    ):
        for key in (
            "epochs",
            "batch_size",
            "learning_rate",
            "max_input_length",
            "max_target_length",
            "gradient_accumulation_steps",
            "lora_rank",
            "lora_alpha",
            "cpu_threads",
        ):
            value = getattr(
                self,
                key,
            )

            if (
                not math.isfinite(
                    value
                )
                or value <= 0
            ):
                raise ValueError(
                    f"{key} must be positive"
                )

        if (
            self.max_input_length
            < 192
            or self.max_target_length
            < 96
        ):
            raise ValueError(
                "Use input length >=192 and target length >=96 "
                "for section-level training"
            )

        if self.seed < 0:
            raise ValueError(
                "seed must be nonnegative"
            )


    def fingerprint_fields(
        self,
    ):
        values = asdict(
            self
        )

        # dry-run과 full-run은 epoch 수가 달라도
        # 나머지 환경이 같으면 같은 학습 설정으로 본다.
        values.pop(
            "epochs"
        )

        return values