from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Configuration for training settings."""

    batch_size: int = 4
    max_iters: int = 5000
    eval_interval: int = 500
    learning_rate: float = 3e-4
    weight_decay: float = 1e-1
    beta2: float = 0.98
    grad_clip: float = 1.0
