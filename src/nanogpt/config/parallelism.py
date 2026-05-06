from dataclasses import dataclass


@dataclass
class ParallelismConfig:
    """Configuration for parallelism settings."""

    world_size: int = 1
    rank: int = 0
    base_seed: int = 0
    is_deterministic: bool = True
