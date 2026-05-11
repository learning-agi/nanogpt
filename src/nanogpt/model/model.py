import math

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from nanogpt.config.gpt import GPTConfig


class MLP(nn.Module):
    """
    Feedforward network for each transformer block,
    consisting of two linear layers with a non-linearity in between.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        # Flag to indicate scaling residual stream acc. to number of layers, as per GPT-2 paper
        self.c_proj.NANOGPT_SCALE_INIT = 1  # type: ignore[assignment]

    def forward(self, x: Tensor) -> Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class CausalSelfAttention(nn.Module):
    """
    Causal self attention layer for each transformer block,
    consisting of multi-head self attention with a causal mask
    to prevent attending to future tokens.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        assert config.n_embd % config.n_head == 0, (
            "Embedding dimension must be divisible by number of heads"
        )
        self.c_atten = nn.Linear(
            config.n_embd, 3 * config.n_embd
        )  # for query, key, value
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)  # output projection
        # Flag to indicate scaling residual stream acc. to number of layers, as per GPT-2 paper
        self.c_proj.NANOGPT_SCALE_INIT = 1  # type: ignore[assignment]
        # Mask
        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(config.block_size, config.block_size).view(
                    1, 1, config.block_size, config.block_size
                )
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.size()
        H = self.n_head
        E = D // H
        qkv: Tensor = self.c_atten(x)  # (B, T, 3 * D)
        q, k, v = qkv.split(self.n_embd, dim=-1)
        q = q.view(B, T, H, E).transpose(1, 2)  # (B, H, T, E)
        k = k.view(B, T, H, E).transpose(1, 2)  # (B, H, T, E)
        v = v.view(B, T, H, E).transpose(1, 2)  # (B, H, T, E)

        # att: Tensor = q @ k.transpose(-2, -1) * (E**-0.5)  # att shape: (B, H, T, T)
        # att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))  # type: ignore[index]
        # att = F.softmax(att, dim=-1)
        # y = att @ v  # (B, H, T, E)
        # The above is the standard attention implementation, but it can be optimized using flash attention, which avoids materializing the full attention matrix in memory.
        # Flash attention implementation:
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B, H, T, E)

        y = y.transpose(1, 2).contiguous().view(B, T, D)  # (B, T, D)
        y = self.c_proj(y)  # (B, T, D)
        return y


class Block(nn.Module):
    """
    Transformer block consisting of a causal self attention layer followed
    by a feedforward network, with layer normalization and residual connections.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """
    GPT language model consisting of a token embedding layer,
    position embedding layer, a stack of transformer blocks,
    and a final layer normalization and output projection to vocabulary size.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)  # token embedding
        self.wpe = nn.Embedding(config.block_size, config.n_embd)  # position embedding
        self.h = nn.ModuleList(
            [Block(config) for _ in range(config.n_layer)]
        )  # transformer blocks
        self.ln_f = nn.LayerNorm(config.n_embd)  # final layer norm
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight sharing between token embedding and output projection - grads add from both directions
        self.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Custom weight initialization as per GPT-2 paper."""
        # TODO: Why do we only assign NANOGPT_SCALE_INIT to the
        # TODO: output projection linear layers of each block?
        # TODO: Should we assign it to other linear layers in
        # TODO: the block as well? (e.g. c_fc in MLP, c_atten in CausalSelfAttention)
        std = 0.02  # Approx. 1/sqrt(n_embd), n_embd = 768
        if isinstance(module, nn.Linear):
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                # Scale std by sqrt(2 * n_layer) for residual stream projections, as per GPT-2 paper
                # 2 because each block has 2 projections (attn and mlp), n_layer because of depth scaling
                std *= math.sqrt(2 * self.config.n_layer)
            # For linear layers, use normal initialization with mean 0 and std
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # For embedding layers, use normal initialization with mean 0 and std 0.02
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(
        self, idx: Tensor, targets: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        B, T = idx.size()
        assert T <= self.config.block_size, (
            f"Sequence length {T} exceeds block size {self.config.block_size}"
        )
        token_embd = self.wte(idx)  # (B, T, D)
        pos_embd = self.wpe(torch.arange(T, device=idx.device))  # (T, D)
        x = (
            token_embd + pos_embd
        )  # (B, T, D), done through broadcasting since position embedding is same for all examples in the batch
        for block in self.h:
            x = block(x)  # (B, T, D)
        x = self.ln_f(x)  # (B, T, D)
        logits = self.lm_head(x)  # (B, T, V)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )  # (B * T, V) and (B * T,) since targets are token indices
        return logits, loss
