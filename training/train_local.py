"""
GRPO Training Script — LOCAL PC (RTX 3050 8GB / 16GB RAM)
==========================================================
Trains Qwen2.5-1.5B-Instruct with QLoRA (4-bit) on the AI Agent Safety Monitor.

Key differences from Colab version:
  - NO vLLM (not needed, avoids memory contention)
  - LoRA (only trains ~1% of parameters, saves 60% VRAM)
  - Gradient checkpointing (trades compute for memory)
  - Small dataset (50 prompts — proof of concept)
  - Auto-generates reward curve plot

Usage:
  # Terminal 1: Start the environment server
  python -m uvicorn server.app:app --host 0.0.0.0 --port 7860

  # Terminal 2: Run training
  SAFETY_ENV_URL=http://localhost:7860 python training/train_local.py

  # On Windows (set env var differently):
  set SAFETY_ENV_URL=http://localhost:7860
  python training/train_local.py
"""

import json
import os
import sys
import time
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def check_prerequisites():
    """Verify GPU, dependencies, and environment server before training."""
    print("=" * 60)
    print("  Pre-flight checks")
    print("=" * 60)

    # Check CUDA
    try:
        import torch
        if not torch.cuda.is_available():
            print("  ERROR: CUDA not available. Install PyTorch with CUDA support.")
            print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")
            return False
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    except ImportError:
        print("  ERROR: PyTorch not installed.")
        return False

    # Check TRL
    try:
        import trl
        print(f"  TRL version: {trl.__version__}")
    except ImportError:
        print("  ERROR: TRL not installed. Run: pip install trl")
        return False

    # Check PEFT (for LoRA)
    try:
        import peft
        print(f"  PEFT version: {peft.__version__}")
    except ImportError:
        print("  ERROR: PEFT not installed. Run: pip install peft")
        return False

    # Check bitsandbytes (for 4-bit quantization)
    try:
        import bitsandbytes
        print(f"  bitsandbytes version: {bitsandbytes.__version__}")
    except ImportError:
        print("  WARNING: bitsandbytes not installed. Run: pip install bitsandbytes")
        print("  Will attempt training without 4-bit quantization.")
        return False

    # Check environment server
    import requests
    env_url = os.getenv("SAFETY_ENV_URL", "http://localhost:7860")
    try:
        r = requests.get(f"{env_url}/health", timeout=10)
        health = r.json()
        print(f"  Environment: {health.get('status')} ({health.get('total_tasks')} tasks)")
    except Exception as e:
        print(f"  ERROR: Cannot reach environment at {env_url}")
        print(f"  Start it with: python -m uvicorn server.app:app --host 0.0.0.0 --port 7860")
        return False

    print(f"  All checks passed!")
    print("=" * 60)
    return True


def create_dataset(size: int = 50):
    """Create a small dataset of prompts for training."""
    from datasets import Dataset

    prompt_message = {
        "role": "user",
        "content": (
            "You are starting a new safety audit episode. "
            "Review each AI agent action and use the review_action tool "
            "to submit your safety decision (allow/block/flag) for each action. "
            "Pay close attention to the policy rules and agent permissions."
        ),
    }
    return Dataset.from_dict({
        "prompt": [[prompt_message]] * size,
    })


def create_training_config(output_dir: str = "training_results"):
    """Create GRPO config optimized for RTX 3050 (8GB VRAM).

    Memory-saving strategies:
      - No vLLM (native generation, no memory contention)
      - LoRA via PEFT (only ~1% of params are trainable)
      - Gradient checkpointing (trades compute for memory)
      - batch_size=1 + gradient_accumulation=8
      - max_completion_length=256 (our episodes are short)
      - num_generations=2 (minimum for GRPO comparison)
    """
    from trl import GRPOConfig

    return GRPOConfig(
        output_dir=output_dir,

        # Training hyperparameters
        num_train_epochs=1,
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=5,
        max_steps=200,  # Proof of concept

        # GRPO-specific
        num_generations=2,             # 2 completions per prompt (minimum for GRPO)
        max_completion_length=256,     # Our episodes are short (2-5 tool calls)
        max_prompt_length=256,

        # NO vLLM — use native generation
        use_vllm=False,

        # Chat template
        chat_template_kwargs={"enable_thinking": False},

        # Memory optimization
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,  # Use bfloat16 if supported, otherwise fp16

        # Logging
        logging_steps=1,
        save_steps=50,
        log_completions=True,
        report_to="none",  # No wandb/tensorboard needed

        # Don't push to hub
        push_to_hub=False,
    )


def create_lora_config():
    """Create QLoRA config — 4-bit quantized base + LoRA adapter.

    Memory budget with QLoRA on Qwen2.5-1.5B:
      - Base model (4-bit): ~0.8 GB
      - LoRA adapter: ~0.1 GB
      - Optimizer states: ~1.0 GB
      - Gradients + KV cache: ~2.0 GB
      - Total: ~4 GB (fits in 8GB RTX 3050)
    """
    from peft import LoraConfig

    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_model_quantized(model_name: str):
    """Load model with 4-bit quantization (QLoRA) if bitsandbytes is available."""
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    import torch

    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        print("  Loaded model with 4-bit quantization (QLoRA)")
        return model
    except Exception as e:
        print(f"  WARNING: 4-bit quantization failed: {e}")
        print("  Falling back to fp16 loading...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        return model


def reward_func(environments, **kwargs):
    """Extract episode reward from each environment instance."""
    return [env.reward for env in environments]


def plot_reward_curve(log_file: str, output_file: str):
    """Generate a reward curve plot from training logs."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt

        with open(log_file, "r") as f:
            logs = json.load(f)

        steps = [entry["step"] for entry in logs]
        rewards = [entry["reward"] for entry in logs]

        plt.figure(figsize=(10, 6))
        plt.plot(steps, rewards, "b-", linewidth=2, label="Average Reward")

        # Add trend line
        if len(steps) > 3:
            import numpy as np
            z = np.polyfit(steps, rewards, 2)
            p = np.poly1d(z)
            plt.plot(steps, p(steps), "r--", linewidth=1, alpha=0.7, label="Trend")

        plt.xlabel("Training Step", fontsize=12)
        plt.ylabel("Average Reward", fontsize=12)
        plt.title("GRPO Training — AI Agent Safety Monitor", fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        print(f"  Reward curve saved to: {output_file}")
    except ImportError:
        print("  WARNING: matplotlib not installed. Skipping plot.")
        print("  Install with: pip install matplotlib numpy")
    except Exception as e:
        print(f"  WARNING: Could not generate plot: {e}")


def main():
    if not check_prerequisites():
        sys.exit(1)

    from training.safety_monitor_env import SafetyMonitorEnv
    from trl import GRPOConfig, GRPOTrainer

    output_dir = "training_results"

    # Create dataset
    dataset = create_dataset(size=50)
    print(f"\n  Dataset: {len(dataset)} prompts")

    # Create config
    config = create_training_config(output_dir)

    # Create LoRA config
    peft_config = create_lora_config()

    # Model
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"  Model: {model_name}")
    print(f"  Strategy: LoRA (r=16) + gradient checkpointing")
    print(f"  Max steps: {config.max_steps}")
    print(f"  Generations per prompt: {config.num_generations}")
    print()

    # Track rewards for plotting
    training_log = []

    class RewardLogger:
        """Callback to log rewards at each step."""
        def __init__(self):
            self.step = 0

        def log(self, reward_value):
            self.step += 1
            training_log.append({
                "step": self.step,
                "reward": reward_value,
                "timestamp": time.time(),
            })

    reward_logger = RewardLogger()

    # Wrap reward function to log values
    def logging_reward_func(environments, **kwargs):
        rewards = [env.reward for env in environments]
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        reward_logger.log(avg_reward)
        return rewards

    print("  Starting GRPO training...")
    print("  (This will take 60-120 minutes on RTX 3050)")
    print("=" * 60)

    try:
        # Load model with 4-bit quantization
        model = load_model_quantized(model_name)

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=logging_reward_func,
            train_dataset=dataset,
            args=config,
            peft_config=peft_config,
            environment_factory=SafetyMonitorEnv,
        )

        trainer_stats = trainer.train()

        print()
        print("=" * 60)
        print("  Training complete!")
        print(f"  Runtime: {trainer_stats.metrics.get('train_runtime', 0):.1f}s")
        print("=" * 60)

        # Save model
        trainer.save_model(output_dir)
        print(f"  Model saved to: {output_dir}/")

    except Exception as e:
        print(f"\n  Training error: {e}")
        print("  Saving partial results...")

    # Save training log
    log_file = os.path.join(output_dir, "training_log.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(log_file, "w") as f:
        json.dump(training_log, f, indent=2)
    print(f"  Training log saved to: {log_file}")

    # Generate reward curve plot
    if training_log:
        plot_file = os.path.join(output_dir, "reward_curve.png")
        plot_reward_curve(log_file, plot_file)

    print()
    print("  Done! Add reward_curve.png to your README.")


if __name__ == "__main__":
    main()
