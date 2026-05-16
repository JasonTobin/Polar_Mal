"""
compare_greyscale_models.py

Generates a full set of comparison images for all greyscale models:
  1. Individual training-history plots (accuracy, loss, LR) — one per model
  2. Overlay plot of all val_acc curves on a single chart
  3. Bar-chart comparison of best-val-acc vs test-acc / test-loss

Usage:
    python3 src/training/compare_greyscale_models.py

All outputs are saved under data/models/greyscale/.
"""

import csv
import io
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT      = Path(__file__).resolve().parents[2]
DATA_DIR  = ROOT / "data" / "images_greyscale"
MODEL_DIR = ROOT / "data" / "models" / "greyscale"

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODELS = [
    # (short_label, display_label, subdir, arch)
    ("cnn", "CNN (v1)",  MODEL_DIR / "cnn", "cnn"),
    ("v2",  "V2",        MODEL_DIR / "v2",  "cnn"),
    ("v3",  "V3",        MODEL_DIR / "v3",  "cnn"),
    ("v4",  "V4 (ResNet)", MODEL_DIR / "v4", "cnnv4"),
]

NUM_WORKERS = min(8, (os.cpu_count() or 4))

# ---------------------------------------------------------------------------
# Eval transform
# ---------------------------------------------------------------------------
eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# ---------------------------------------------------------------------------
# RAM-cached dataset
# ---------------------------------------------------------------------------
class CachedImageFolder(datasets.ImageFolder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(f"  Caching {len(self.samples)} test images into RAM ...", flush=True)
        self._cache: list[bytes] = []
        for path, _ in tqdm(self.samples, desc="  Loading", unit="img", leave=True):
            with open(path, "rb") as f:
                self._cache.append(f.read())
        print(f"  Cached {len(self._cache)} images.\n", flush=True)

    def __getitem__(self, index: int):
        img = Image.open(io.BytesIO(self._cache[index])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.targets[index]

# ---------------------------------------------------------------------------
# Model architectures
# ---------------------------------------------------------------------------
class MalwareCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(self.relu(self.main(x) + self.skip(x)))


class MalwareCNNv4(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.block1 = ResBlock(3,   32)
        self.block2 = ResBlock(32,  64)
        self.block3 = ResBlock(64,  128)
        self.block4 = ResBlock(128, 256)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.classifier(x)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_history(csv_path: Path) -> dict:
    h = {"epoch": [], "train_loss": [], "val_loss": [],
         "train_acc": [], "val_acc": [], "lr": []}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            h["epoch"].append(int(row["epoch"]))
            h["train_loss"].append(float(row["train_loss"]))
            h["val_loss"].append(float(row["val_loss"]))
            h["train_acc"].append(float(row["train_acc"]) * 100)
            h["val_acc"].append(float(row["val_acc"]) * 100)
            h["lr"].append(float(row["lr"]))
    return h


def load_model(arch: str, ckpt: dict, device: torch.device) -> nn.Module:
    num_classes = len(ckpt["classes"])
    if arch == "cnn":
        model = MalwareCNN(num_classes)
    elif arch == "cnnv4":
        model = MalwareCNNv4(num_classes)
    else:
        raise ValueError(f"Unknown arch: {arch}")
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    criterion = nn.CrossEntropyLoss()
    correct = total = 0
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Test", unit="batch", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            correct    += (outputs.argmax(dim=1) == labels).sum().item()
            total      += images.size(0)
    return total_loss / total, correct / total

# ---------------------------------------------------------------------------
# Plot 1: Individual training history (per model)
# ---------------------------------------------------------------------------
def plot_individual(h: dict, display_label: str, out_path: Path):
    epochs = h["epoch"]
    best_idx   = h["val_acc"].index(max(h["val_acc"]))
    best_epoch = epochs[best_idx]
    best_acc   = h["val_acc"][best_idx]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Greyscale {display_label} — Training History", fontsize=14, fontweight="bold")

    # Accuracy
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

    # Loss
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

    # Learning rate
    ax = axes[2]
    ax.semilogy(epochs, h["lr"], color="#9C27B0", linewidth=1.5)
    ax.set_title("Learning Rate")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR (log scale)")
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.relative_to(ROOT)}")

# ---------------------------------------------------------------------------
# Plot 2: Overlay of all val_acc curves
# ---------------------------------------------------------------------------
COLOURS = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

def plot_overlay(histories: list[tuple[str, dict]], out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Greyscale Models — Val Accuracy & Val Loss Overlay", fontsize=14, fontweight="bold")

    for (label, h), col in zip(histories, COLOURS):
        best_acc = max(h["val_acc"])
        axes[0].plot(h["epoch"], h["val_acc"], label=f"{label} (best {best_acc:.2f}%)",
                     color=col, linewidth=1.5)
        axes[1].plot(h["epoch"], h["val_loss"], label=label, color=col, linewidth=1.5)

    axes[0].set_title("Validation Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Val Accuracy (%)")
    axes[0].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val Loss")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.relative_to(ROOT)}")

# ---------------------------------------------------------------------------
# Plot 3: Bar chart — val acc vs test acc + test loss
# ---------------------------------------------------------------------------
def plot_bar_comparison(results: list[dict], out_path: Path):
    labels   = [r["label"] for r in results]
    val_accs = [r["val_acc"] * 100  for r in results]
    tst_accs = [r["test_acc"] * 100 for r in results]
    tst_loss = [r["test_loss"]       for r in results]

    x     = list(range(len(labels)))
    bar_w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Greyscale Models — Test Set Comparison", fontsize=14, fontweight="bold")

    ax = axes[0]
    b1 = ax.bar([i - bar_w / 2 for i in x], tst_accs, width=bar_w,
                color="#2196F3", label="Test Acc", zorder=3)
    ax.bar([i + bar_w / 2 for i in x], val_accs, width=bar_w,
           color="#4CAF50", label="Best Val Acc", zorder=3, alpha=0.85)
    ax.set_title("Accuracy")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    for bar, v in zip(b1, tst_accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{v:.2f}%", ha="center", va="bottom", fontsize=8)

    ax = axes[1]
    b2 = ax.bar(x, tst_loss, width=0.5, color="#F44336", zorder=3)
    ax.set_title("Test Loss")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    for bar, v in zip(b2, tst_loss):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.relative_to(ROOT)}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ---- Individual training plots ----------------------------------------
    print("Generating individual training history plots ...")
    histories = []
    for short, display, model_dir, _ in MODELS:
        csv_path = model_dir / "training_history.csv"
        if not csv_path.exists():
            print(f"  [SKIP] {display}: {csv_path} not found")
            continue
        h = load_history(csv_path)
        out = model_dir / "training_history_graph.png"
        plot_individual(h, display, out)
        histories.append((display, h))

    # ---- Overlay plot -------------------------------------------------------
    if histories:
        print("\nGenerating overlay comparison plot ...")
        plot_overlay(histories, MODEL_DIR / "greyscale_overlay.png")

    # ---- Test-set evaluation + bar chart ------------------------------------
    print("\nLoading greyscale test dataset ...")
    test_ds = CachedImageFolder(str(DATA_DIR / "test"), transform=eval_transform)
    test_loader = DataLoader(
        test_ds, batch_size=512, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
        persistent_workers=True, prefetch_factor=2,
    )

    results = []
    header = f"{'Model':<18}  {'Val Acc':>8}  {'Test Acc':>9}  {'Test Loss':>10}"
    sep    = "-" * len(header)
    print(f"\n{header}\n{sep}")

    for short, display, model_dir, arch in MODELS:
        ckpt_path = model_dir / "best_model.pth"
        if not ckpt_path.exists():
            print(f"  [SKIP] {display}: {ckpt_path} not found")
            continue

        ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
        model = load_model(arch, ckpt, device)
        val_acc = float(ckpt.get("val_acc", float("nan")))

        test_loss, test_acc = evaluate(model, test_loader, device)
        print(f"  {display:<16}  {val_acc:>7.2%}  {test_acc:>8.2%}  {test_loss:>10.4f}")
        results.append({"label": display, "val_acc": val_acc,
                        "test_acc": test_acc, "test_loss": test_loss})

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(sep)
    if results:
        best = max(results, key=lambda r: r["test_acc"])
        print(f"\nBest greyscale model: {best['label']}  "
              f"test_acc={best['test_acc']:.2%}  test_loss={best['test_loss']:.4f}")
        print("\nGenerating bar-chart comparison ...")
        plot_bar_comparison(results, MODEL_DIR / "greyscale_comparison.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
