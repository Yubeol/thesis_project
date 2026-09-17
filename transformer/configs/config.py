from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class TrainingConfig:
    model_name: str = "google/flan-t5-small"
    model_revision: str = "main"
    epochs: float = 3.0
    batch_size: int = 1
    learning_rate: float = 3e-4
    max_input_length: int = 384
    max_target_length: int = 128
    gradient_accumulation_steps: int = 8
    seed: int = 42
    lora_rank: int = 4
    lora_alpha: int = 8
    fp16: bool = False
    cpu_threads: int = 8

    def validate(self):
        for key in ("epochs", "batch_size", "learning_rate", "max_input_length",
                    "max_target_length", "gradient_accumulation_steps", "lora_rank",
                    "lora_alpha", "cpu_threads"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if self.max_input_length < 192 or self.max_target_length < 48:
            raise ValueError("Use input length >=192 and target length >=48 to preserve all fields/sections")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")

    def fingerprint_fields(self):
        values = asdict(self)
        # A two-step rehearsal and the full run necessarily use different epoch counts.
        values.pop("epochs")
        return values
