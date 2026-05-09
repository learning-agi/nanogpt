import numpy as np
from rich import print
import time
import torch
import tqdm

from nanogpt.config.gpt import GPTConfig
from nanogpt.config.parallelism import ParallelismConfig
from nanogpt.config.training import TrainingConfig
from nanogpt.data.dataloader import DataLoader
from nanogpt.model.model import GPT
from nanogpt.settings import DATA_ROOT

torch.set_float32_matmul_precision("high") # Use TF32 for faster matrix multiplication on supported GPUs (Ampere series and later)


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
        self.model = torch.compile(self.model) # Use torch.compile to optimize the model for faster training
        print(
            f"[bold blue]Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters[/bold blue]"
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
        print(f"Training for {len(self.data_loader)} steps...")
        start = time.time()
        tps_list = []
        with tqdm.tqdm(total=len(self.data_loader), desc="Training") as pbar:
            for step, (x, y) in enumerate(self.data_loader):
                t0 = time.time()
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                with torch.amp.autocast(device_type=self.device, dtype=torch.bfloat16): # Use mixed precision for faster training and reduced memory usage
                    logits, loss = self.model(x, targets=y)                
                loss.backward()
                losses.append(loss.item())
                self.optimizer.step()
                torch.cuda.synchronize() # Ensure all GPU operations are complete before measuring time
                t1 = time.time()
                dt = (t1 - t0) * 1000 # Convert to milliseconds
                tokens_per_second = self.training_config.batch_size * self.gpt_config.block_size / (t1 - t0)
                tps_list.append(tokens_per_second)
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())
                pbar.set_description(f"Step {step}, Time {dt:.2f} ms, TPS {tokens_per_second:.2f}")
        end = time.time()
        print(f"[bold green]Final loss after training: {losses[-1]:.4f}[/bold green]")
        print(f"[bold green]Training time: {end - start:.2f} seconds[/bold green]")
        print(f"[bold green]Average time per step: {(end - start) / len(self.data_loader) * 1000:.2f} ms[/bold green]")
        print(f"[bold green]Total tokens processed: {len(self.data_loader) * self.training_config.batch_size * self.gpt_config.block_size}[/bold green]")
        print(f"[bold green]Average Throughput (Tokens per second): {np.mean(tps_list):.2f}[/bold green]")


if __name__ == "__main__":
    gpt_config = GPTConfig()
    training_config = TrainingConfig()
    parallelism_config = ParallelismConfig(is_deterministic=True)
    training_run = TrainingRun(gpt_config, training_config, parallelism_config)
    training_run.train()



# Andrej Karpathy video notes

# Optimizing for speed and memory:

# 1. Use mixed precision for faster computation and reduced memory usage. Read torch "Automated Mixed Precision" documentation.

# a. TF32
# accelerating matrix multiplication through tensor cores. torch.set_float32_matmul_precision("high") to enable TF32 on supported GPUs.
# Tensor core is just an instruction in GPU arch. e.g. 4x4 matrix multiplication. Few switches to dictate the precision of multiplication and accumulator.
# How is TF32 faster? Read NVIDIA A100 Tensor Core GPU architecture whitepaper.
# It uses 10-bit precision for multiplication and 32-bit precision for accumulation.
# This allows it to perform matrix multiplications much faster than traditional FP32 precision while still maintaining a good level of accuracy for training deep learning models.
# 19-bit multiplication and 32-bit accumulation strikes a good balance between speed and accuracy for training large language models like GPT.
# Generally, 8x faster than FP32 for matrix multiplication.
# Most computation is in the matrix multiplication, so using TF32 can speed up training significantly without much loss in model quality.
# Normally, CPU requests GPU for some piece of work and continues running, so GPU is doing work in the background. CPU can do other work while GPU is busy.
# But sometimes, we need to wait for GPU to finish before CPU can continue, which is called synchronization. e.g. measuring time taken for a training step. We need to synchronize to get accurate timing.
# If training doesn't fit in memory, we can start with smaller batch size and gradually increase it as we optimize the code and reduce memory usage. This is called "batch size scaling".
# By default the idea is to maximize batch size to fully utilize GPU memory and achieve faster training. Also recommended to use numbers with lots of power of 2 for batch size for better performance on GPU.
# Next steps for fitting in memory are gradient checkpointing, which saves memory by not storing intermediate activations during the forward pass and recomputing them during the backward pass.
# This can reduce memory usage significantly at the cost of increased computation time.
# TF32 promises 8x speedup for matrix multiplication, but actual speedup is likely to be around 3x-4x.
# This is because most operations tend to be memory bound rather than compute bound, so the overall speedup is limited by memory bandwidth rather than just the speed of matrix multiplication.
# We are still moving 32bit activations and gradients around, so the memory bandwidth is still a bottleneck.

# b. BF16
# Now we reduce the space to 16 bits, so we make computation even faster and reduce memory usage even more. But we need to be careful about the precision loss.
# BF16 uses 8 bits for exponent and 7 bits for mantissa, which allows it to represent a wide range of values while still maintaining a reasonable level of precision for training deep learning models.
# It is particularly effective for training large language models like GPT, where the activations and gradients can have a wide range of values.
# BF16 can provide even faster training than TF32, especially on newer GPUs that have native support for BF16 precision.
# However, it may require more careful tuning of hyperparameters and may not be suitable for all models or training scenarios due to potential precision loss.
# Using BF16 through torch.amp.autocast makes the activations and gradients use BF16 precision but the model parameters remain in FP32 precision, which helps to mitigate some of the precision loss while still providing significant speedup and memory reduction.
# So we do not reap full benefits of BF16 (2x) if the model parameters are still in FP32, but it can still provide significant speedup and memory reduction for the activations and gradients, which are often the largest contributors to memory usage during training.

# c. FP16
# FP16 uses 5 bits for exponent and 10 bits for mantissa, which provides more precision but less range compared to BF16. It can be more sensitive to precision loss, especially for large language models like GPT.
# It can still provide significant speedup and memory reduction, but may require more careful tuning and may not be suitable for all models or training scenarios.
# This might require techniques like gradient scaling to prevent underflow during training, which can add complexity to the training process.


# 2. torch.compile - Read torch docs, NVIDIA GPU SM, HBM, GPU SRAM docs
# PyTorch 2.0 introduced torch.compile, which can optimize the training loop for faster execution. It uses a just-in-time (JIT) compiler to optimize the code and can provide significant speedup for training deep learning models.
# Use by - model = torch.compile(model) after initializing the model. It can optimize the entire training loop, including the forward and backward passes, as well as the optimizer step.
# Speedup mainly comes from reducing Python overheads and GPU read/writes.
# Compiler analyses the entire code and identifies what to run. It optimizes by compiling the code into a single object eliminating the Python interpreter.
# Next, optimizations are performed on the compiled code, such as operator fusion, which combines multiple operations into a single kernel to reduce GPU read/writes and improve performance.
# HBM <---> GPU <---> CPU <---> RAM
# For each complex operation, we constantly need to launch kernels, i.e., move data from HBM to GPU (slow), do computation in GPU chips (fast), then store results back to HBM (slow). This is a bottleneck because of the latency of GPU read/writes.
# To solve this, one idea is to do operator fusion, which combines multiple operations into a single kernel, so we can do more computation in GPU chips without needing to read/write to HBM as often. This can significantly reduce the latency and improve performance.