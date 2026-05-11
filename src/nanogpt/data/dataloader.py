import tiktoken
import torch
from torch import Tensor


class DataLoader:
    """
    DataLoader for loading and batching text data for training the GPT model.
    """
    def __init__(self, data: str, block_size: int, batch_size: int) -> None:
        self.data = data
        self.block_size = block_size
        self.batch_size = batch_size
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.tokenizer.n_vocab
        self.encoded_data = self.tokenizer.encode(self.data)
        self.idx = 0 # Pointer to current position in the encoded data
        self.window_size = block_size * batch_size # Total number of tokens in one window of data for a batch

    def __len__(self) -> int:
        # Total number of tokens in the encoded data
        return len(self.encoded_data) // self.window_size

    def __iter__(self):
        return self

    def __next__(self) -> tuple[Tensor, Tensor]:
        if self.idx >= len(self.encoded_data) // self.window_size:
            self.idx = 0 # Reset index for next epoch
            raise StopIteration
        # Get the next window of data
        buffer = torch.tensor(
            self.encoded_data[
                self.idx * self.window_size
                : (self.idx + 1) * self.window_size + 1
            ]
        )
        x = buffer[:-1].view(self.batch_size, self.block_size) # Input tokens
        y = buffer[1:].view(self.batch_size, self.block_size) # Target tokens
        self.idx += 1
        return x, y

    def next_batch(self) -> tuple[Tensor, Tensor]:
        """Get the next batch of data for training."""
        try:
            return next(self)
        except StopIteration:
            self.idx = 0 # Reset index for next epoch
            return next(self)
