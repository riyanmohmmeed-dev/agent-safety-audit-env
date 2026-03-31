"""
Publication-Quality Reward Curve Plotter
=========================================
Generates a polished training reward curve from trainer_state.json
for the README and hackathon submission.

Usage:
    python training/plot_reward_curve.py [checkpoint_path]

If no path is given, it auto-finds the latest checkpoint.
"""

import json
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np
except ImportError:
    print("ERROR: pip install matplotlib numpy")
    sys.exit(1)


def find_latest_checkpoint() -> Path:
    """Auto-find the latest checkpoint's trainer_state.json."""
    results_dir = Path("training_results")
    if not results_dir.exists():
        print("ERROR: training_results/ directory not found.")
        sys.exit(1)

    checkpoints = sorted(
        results_dir.glob("checkpoint-*/trainer_state.json"),
        key=lambda p: int(p.parent.name.split("-")[1]),
    )
    if not checkpoints:
        print("ERROR: No checkpoint folders found.")
        sys.exit(1)

    return checkpoints[-1]


def moving_average(data, window=10):
    """Compute a centered moving average."""
    return np.convolve(data, np.ones(window) / window, mode="valid")


def main():
    # Find data
    if len(sys.argv) > 1:
        state_file = Path(sys.argv[1])
    else:
        state_file = find_latest_checkpoint()

    print(f"Reading: {state_file}")
    with open(state_file) as f:
        state = json.load(f)

    # Extract reward data from log_history
    log_history = state.get("log_history", [])
    steps = []
    rewards = []
    for entry in log_history:
        if "reward" in entry and "step" in entry:
            steps.append(entry["step"])
            rewards.append(entry["reward"])

    if not steps:
        print("ERROR: No reward data found in trainer_state.json")
        sys.exit(1)

    print(f"Found {len(steps)} data points (steps {steps[0]} to {steps[-1]})")

    steps = np.array(steps)
    rewards = np.array(rewards)

    # Compute stats
    first_50 = rewards[:50] if len(rewards) >= 50 else rewards
    last_50 = rewards[-50:] if len(rewards) >= 50 else rewards
    print(f"First 50 avg: {np.mean(first_50):.4f}")
    print(f"Last 50 avg:  {np.mean(last_50):.4f}")
    print(f"Max reward:   {np.max(rewards):.4f}")
    print(f"Improvement:  {np.mean(last_50) - np.mean(first_50):+.4f}")

    # --- Publication-quality plot ---
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

    # Raw rewards (faded)
    ax.plot(
        steps, rewards,
        color="#93C5FD", linewidth=0.6, alpha=0.5,
        label="Per-Step Reward", zorder=1,
    )

    # Moving average (bold)
    window = 10
    ma = moving_average(rewards, window)
    ma_steps = steps[window - 1:][:len(ma)]
    ax.plot(
        ma_steps, ma,
        color="#1E40AF", linewidth=2.5,
        label=f"Moving Average (n={window})", zorder=3,
    )

    # Trend line (polynomial fit)
    z = np.polyfit(steps, rewards, 2)
    trend = np.poly1d(z)(steps)
    ax.plot(
        steps, trend,
        color="#EF4444", linewidth=1.5, linestyle="--", alpha=0.7,
        label="Quadratic Trend", zorder=2,
    )

    # Annotation: Tool usage learned
    if len(rewards) > 30:
        learn_step = steps[25]
        learn_val = ma[15] if len(ma) > 15 else rewards[25]
        ax.annotate(
            "Tool usage learned",
            xy=(learn_step, learn_val),
            xytext=(learn_step + 30, learn_val + 0.03),
            fontsize=10, fontweight="bold", color="#1E40AF",
            arrowprops=dict(arrowstyle="->", color="#1E40AF", lw=1.5),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#1E40AF", alpha=0.9),
        )

    # Peak annotation
    peak_idx = np.argmax(rewards)
    ax.annotate(
        f"Peak: {rewards[peak_idx]:.3f}",
        xy=(steps[peak_idx], rewards[peak_idx]),
        xytext=(steps[peak_idx] - 40, rewards[peak_idx] + 0.015),
        fontsize=9, color="#EF4444",
        arrowprops=dict(arrowstyle="->", color="#EF4444", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#EF4444", alpha=0.9),
    )

    # Labels and styling
    ax.set_xlabel("Training Step", fontsize=13, fontweight="bold", labelpad=10)
    ax.set_ylabel("Reward", fontsize=13, fontweight="bold", labelpad=10)
    ax.set_title(
        "GRPO Training — AI Agent Safety Monitor\n"
        "Qwen2.5-1.5B-Instruct · QLoRA (4-bit) · RTX 3050",
        fontsize=14, fontweight="bold", pad=15,
    )

    ax.legend(
        loc="upper right", fontsize=10,
        frameon=True, fancybox=True, shadow=True, framealpha=0.9,
    )

    ax.set_xlim(0, steps[-1] + 5)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    # Stats box
    stats_text = (
        f"Steps: {len(steps)}\n"
        f"First 50 avg: {np.mean(first_50):.4f}\n"
        f"Last 50 avg: {np.mean(last_50):.4f}\n"
        f"Peak: {np.max(rewards):.4f}"
    )
    ax.text(
        0.02, 0.02, stats_text,
        transform=ax.transAxes, fontsize=9,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F0F9FF", edgecolor="#93C5FD", alpha=0.9),
        fontfamily="monospace",
    )

    plt.tight_layout()

    output_path = "reward_curve.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"\nSaved: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
