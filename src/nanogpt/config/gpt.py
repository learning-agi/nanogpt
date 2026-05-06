from dataclasses import dataclass


@dataclass
class GPTConfig:
    """GPT Base Config, provide overrides for specific versions"""

    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
