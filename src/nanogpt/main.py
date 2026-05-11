import math
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
        self.grad_accumulation_setup()

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
        self.max_steps = len(self.data_loader)

    def init_model(self) -> None:
        """Initialize the GPT model for training."""
        print("Initializing model...")
        self.model = GPT(self.gpt_config)
        self.model.to(self.device)
        self.model = torch.compile(self.model) # Use torch.compile to optimize the model for faster training
        print(
            f"[bold blue]Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters[/bold blue]"
        )

    def configure_optimizers(self) -> None:
        param_dict = {name: param for name, param in self.model.named_parameters() if param.requires_grad}
        # Create optimizer groups, any parameter that is 2D will be weight decayed
        # TODO: understand why we do not weight decay 1D parameters like biases and layer norms. Is it because they are less likely to overfit? Or is it just a convention that has been found to work well in practice?
        decay_params = [param for name, param in param_dict.items() if param.dim() >= 2]
        no_decay_params = [param for name, param in param_dict.items() if param.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": self.training_config.adam_weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_no_decay_params = sum(p.numel() for p in no_decay_params)
        print(f"Optimizer groups: {len(decay_params)} decay params tensors, with {num_decay_params} parameters | {len(no_decay_params)} no decay params tensors, with {num_no_decay_params} parameters")
        fused_available = "cuda" in self.device
        print(f"Using fused Adam optimizer: {fused_available}")
        self.optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.training_config.learning_rate,
            betas=self.training_config.adam_betas,
            eps=self.training_config.adam_eps,
            fused=fused_available,  # Use fused Adam for faster optimization on supported GPUs
        )

    def init_optimizer(self) -> None:
        """Initialize the optimizer for training."""
        print("Initializing optimizer...")
        self.configure_optimizers()

    def lr_scheduler(self, step: int) -> float:
        """Calculate learning rate with linear warmup and cosine decay."""
        if step < self.training_config.warmup_steps:
            return self.training_config.learning_rate * (step + 1) / self.training_config.warmup_steps
        
        cosine_decay_ratio = (step - self.training_config.warmup_steps) / (self.max_steps - self.training_config.warmup_steps)
        assert 0 <= cosine_decay_ratio <= 1, f"Cosine decay ratio should be between 0 and 1, got {cosine_decay_ratio:.4f}"
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * cosine_decay_ratio))  # TODO: Understand this formula better
        decayed_lr = (self.training_config.learning_rate - self.training_config.min_learning_rate) * cosine_decay + self.training_config.min_learning_rate
        return decayed_lr

    def grad_accumulation_setup(self) -> None:
        B, T = self.training_config.batch_size, self.gpt_config.block_size
        assert self.training_config.desired_batch_size % (B * T) == 0, "Desired batch size must be a multiple of the actual batch size for gradient accumulation to work properly."
        self.grad_accumulation_steps = self.training_config.desired_batch_size // (B * T)
        print(f"Using gradient accumulation with {self.grad_accumulation_steps} steps to achieve effective batch size of {self.training_config.desired_batch_size} tokens")

    def train(self) -> None:
        """Main training loop."""
        print("Starting training loop...")
        losses = []
        print(f"Training for {self.max_steps} steps...")
        start = time.time()
        tps_list = []
        with tqdm.tqdm(total=self.max_steps, desc="Training") as pbar:
            for step in range(self.max_steps):
                t0 = time.time()
                self.optimizer.zero_grad()
                # TODO: Why does grad norm increase with gradient accumulation?
                loss_accum = 0.0
                for micro_step in range(self.grad_accumulation_steps):
                    x, y = self.data_loader.next_batch() # Get the next batch of data for training
                    x, y = x.to(self.device), y.to(self.device)
                    with torch.amp.autocast(device_type=self.device, dtype=torch.bfloat16): # Use mixed precision for faster training and reduced memory usage
                        logits, loss = self.model(x, targets=y)    
                    loss = loss / self.grad_accumulation_steps # Scale the loss by the number of gradient accumulation steps to get the correct gradient magnitude (equivalent to doing mean in MSE loss)            
                    loss_accum += loss.detach()  # Detach single micro batch loss tensor from the computation graph and accumulate it for logging purposes, so we can log the average loss over the gradient accumulation steps without affecting the gradients
                    # TODO: Doesnt detaching the loss here cause any issues with backpropagation? We still want to backpropagate through the loss to update the model parameters, but we also want to accumulate the loss for logging purposes.
                    # By detaching the loss, we can accumulate it without affecting the gradients, which should be fine as long as we are still backpropagating through the original loss tensor that is not detached.
                    loss.backward()
                # Do gradient norm clipping to prevent exploding gradients, which can be more likely with mixed precision training or just bad data batch
                # grad_norm = sum(p.grad.data.item() ** 2 for p in self.model.parameters()) ** (1. / 2) - TODO: check if this is correct way to compute grad norm
                # Make sure this grad_norm is not too large, otherwise it can cause instability in training. We can clip the gradients to a maximum norm value to prevent this.
                # TODO: How does clipping grad norm affect grads themselves?
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.grad_clip)
                losses.append(loss.item())
                # Set learning rate with linear warmup and cosine decay
                lr = self.lr_scheduler(step)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                
                self.optimizer.step()
                torch.cuda.synchronize() # Ensure all GPU operations are complete before measuring time
                t1 = time.time()
                dt = (t1 - t0) * 1000 # Convert to milliseconds
                tokens_per_second = self.training_config.batch_size * self.gpt_config.block_size * self.grad_accumulation_steps / (t1 - t0)
                tps_list.append(tokens_per_second)
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())
                pbar.set_description(f"Step {step} | Time {dt:.2f} ms | TPS {tokens_per_second:.2f} | Loss: {loss_accum.item():.4f} | Grad Norm: {norm:.4f} | LR: {lr:.2e}")
        end = time.time()
        print(f"[bold green]Final loss after training: {losses[-1]:.4f}[/bold green]")
        print(f"[bold green]Training time: {end - start:.2f} seconds[/bold green]")
        print(f"[bold green]Average time per step: {(end - start) / len(self.data_loader) * 1000:.2f} ms[/bold green]")
        print(f"[bold green]Total tokens processed: {self.max_steps * self.training_config.batch_size * self.gpt_config.block_size * self.grad_accumulation_steps}[/bold green]")
        print(f"[bold green]Average Throughput (Tokens per second): {np.mean(tps_list):.2f}[/bold green]")


if __name__ == "__main__":
    gpt_config = GPTConfig(vocab_size=50_304)
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


# 3. Flash Attention - Read flash attention paper and NVIDIA GPU architecture whitepaper
# There are some optimizations that torch.compile cannot do, such as optimizing the attention mechanism in GPT (requires a rewrite of attention algo).
# # Flash Attention is a technique that can optimize the attention mechanism for faster training and reduced memory usage.
# Flash Attention is a kernel fusion operation that combines multiple attention operations into a single kernel, reducing the number of memory accesses and improving performance.
# PyTorch: MatMul -> Mask -> Softmax -> Dropout -> MatMul
# Flash Attention: Fused Kernel that does all of the above in one go, reducing memory accesses and improving performance.
# Flash Attention does more FLOPs than the standard attention implementation, but it is much faster because it reduces the number of memory accesses, which are the bottleneck for performance in attention mechanisms.
# The trick is to never materialize the full attention matrix in memory (NxN), which can be huge for large sequence lengths.
# Instead, it computes the attention in a way that only requires storing a small portion of the attention matrix at a time, which significantly reduces memory usage and allows for much faster computation.
# It does so by using an online softmax trick. Update softmax using the max and sum of the current block of attention scores, which allows it to compute the softmax without needing to store the entire attention matrix in memory.


# 4. Ugly numbers removal
# Using numbers with lots of powers of 2 can improve performance on GPUs due to better memory alignment and more efficient use of GPU resources. 
# This is because GPUs are optimized for processing data in blocks that are powers of 2, so using batch sizes and sequence lengths that are powers of 2 can lead to better performance.
# E.g. increase vocab to 50,304 (instead of 50,207) to make it a multiple of 256, which is a common block size for GPU processing. This can help to improve the efficiency of the model and speed up training.
# Even though we are doing more (& redundant) computation & waste additional memory with a larger vocab size, the overall training time can still be reduced due to better GPU utilization and faster processing of the data.
# This is a common trade-off in deep learning where we may do more computation but achieve faster training times due to better hardware utilization.