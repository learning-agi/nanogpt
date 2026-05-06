import numpy as np
from rich import print
import torch
import tqdm

from nanogpt.config.gpt import GPTConfig
from nanogpt.config.parallelism import ParallelismConfig
from nanogpt.config.training import TrainingConfig
from nanogpt.data.dataloader import DataLoader
from nanogpt.model.model import GPT
from nanogpt.settings import DATA_ROOT


class TrainingRun:
    def __init__(
        self,
        gpt_config: GPTConfig,
        training_config: TrainingConfig,
        parallelism_config: ParallelismConfig | None = None,
    ) -> None:
        self.gpt_config = gpt_config
        self.training_config = training_config
        self.parallelism_config = parallelism_config
        self.setup_device()
        self.stats_before_training()
        self.init_data_loader()
        self.init_model()
        self.init_optimizer()

    def setup_device(self) -> None:
        """Setup CUDA device for training."""
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        print(f"Using device: {self.device}")

    def stats_before_training(self) -> None:
        """Print some stats before starting training."""
        print("Calculating expected loss at init")
        rough_pred_per_token_at_init = 1 / self.gpt_config.vocab_size
        expected_loss_at_init = -np.log(rough_pred_per_token_at_init)
        print(f"Expected loss at init: {expected_loss_at_init:.4f}")

    def init_data_loader(self) -> None:
        """Initialize the data loader for training."""
        print("Loading data from file...")
        with open(DATA_ROOT / "input.txt", "r") as f:
            data = f.read()
        print("Initializing data loader...")
        self.data_loader = DataLoader(
            data=data,
            block_size=self.gpt_config.block_size,
            batch_size=self.training_config.batch_size,
        )

    def init_model(self) -> None:
        """Initialize the GPT model for training."""
        print("Initializing model...")
        self.model = GPT(self.gpt_config)
        self.model.to(self.device)
        print(
            f"[bold green]Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters[/bold green]"
        )

    def init_optimizer(self) -> None:
        """Initialize the optimizer for training."""
        print("Initializing optimizer...")
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.training_config.learning_rate
        )

    def train(self) -> None:
        """Main training loop."""
        print("Starting training loop...")
        losses = []
        with tqdm.tqdm(total=self.training_config.max_iters, desc="Training") as pbar:
            for step, (x, y) in enumerate(self.data_loader):
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                logits, loss = self.model(x, targets=y)
                loss.backward()
                losses.append(loss.item())
                self.optimizer.step()
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())
        print(f"[bold orange]Final loss after training: {losses[-1]:.4f}[/bold orange]")


if __name__ == "__main__":
    gpt_config = GPTConfig()
    training_config = TrainingConfig()
    parallelism_config = ParallelismConfig(is_deterministic=True)
    training_run = TrainingRun(gpt_config, training_config, parallelism_config)
    training_run.train()
