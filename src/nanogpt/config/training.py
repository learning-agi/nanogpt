from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Configuration for training settings."""

    batch_size: int = 16
    desired_batch_size: int = 524288  # 2**19, ~0.5M, in number of tokens
    eval_interval: int = 500
    learning_rate: float = 6e-4
    min_learning_rate: float = 6e-5
    weight_decay: float = 1e-1
    beta2: float = 0.98
    grad_clip: float = 1.0
    adam_betas: tuple[float, float] = (0.9, 0.95)
    adam_eps: float = 1e-8
    adam_weight_decay: float = 0.1
    grad_clip_value: float = 1.0
    warmup_steps: int = 10
