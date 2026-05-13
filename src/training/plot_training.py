"""
plot_training.py

Generates training progress graphs from a training_history.csv file.
Saves a PNG next to the CSV.

Usage:
    # Plot v1 (default):
    python plot_training.py

    # Plot a specific model directory:
    python plot_training.py --csv data/models/v2/training_history.csv --title "V2 Phase1"
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import csv


def load_history(csv_path: Path) -> dict:
    history = {"epoch": [], "train_loss": [], "val_loss": [],
               "train_acc": [], "val_acc": [], "lr": []}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            history["epoch"].append(int(row["epoch"]))
            history["train_loss"].append(float(row["train_loss"]))
            history["val_loss"].append(float(row["val_loss"]))
            history["train_acc"].append(float(row["train_acc"]) * 100)
            history["val_acc"].append(float(row["val_acc"]) * 100)
            history["lr"].append(float(row["lr"]))
    return history


def plot(csv_path: Path, title: str, out_path: Path):
    h = load_history(csv_path)
    epochs = h["epoch"]

    best_val_idx = h["val_acc"].index(max(h["val_acc"]))
    best_epoch   = epochs[best_val_idx]
    best_acc     = h["val_acc"][best_val_idx]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # --- Accuracy ---
    ax = axes[0]
    ax.plot(epochs, h["train_acc"], label="Train", color="#2196F3", linewidth=1.5)
    ax.plot(epochs, h["val_acc"],   label="Val",   color="#4CAF50", linewidth=1.5)
    ax.axvline(best_epoch, color="#F44336", linestyle="--", linewidth=1,
               label=f"Best val epoch {best_epoch} ({best_acc:.2f}%)")
    ax.set_title("Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Loss ---
    ax = axes[1]
    ax.plot(epochs, h["train_loss"], label="Train", color="#2196F3", linewidth=1.5)
    ax.plot(epochs, h["val_loss"],   label="Val",   color="#4CAF50", linewidth=1.5)
    ax.axvline(best_epoch, color="#F44336", linestyle="--", linewidth=1,
               label=f"Best val epoch {best_epoch}")
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Learning Rate ---
    ax = axes[2]
    ax.semilogy(epochs, h["lr"], color="#9C27B0", linewidth=1.5)
    ax.set_title("Learning Rate")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR (log scale)")
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Graph saved to: {out_path}")


def main():
    ROOT = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description="Plot training history")
    parser.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "data" / "models" / "training_history.csv",
        help="Path to training_history.csv (default: data/models/training_history.csv)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Plot title (default: derived from CSV parent directory name)",
    )
    args = parser.parse_args()

    csv_path = args.csv
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        return

    title    = args.title or f"Training History — {csv_path.parent.name}"
    out_path = csv_path.parent / (csv_path.stem + "_graph.png")

    plot(csv_path, title, out_path)


if __name__ == "__main__":
    main()
