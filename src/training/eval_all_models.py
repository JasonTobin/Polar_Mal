"""
eval_all_models.py

Evaluates every saved checkpoint against the held-out test set and
produces a side-by-side bar chart of test accuracy and loss.

Usage:
    python3 src/training/eval_all_models.py
"""

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
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parents[2]
DATA_DIR   = ROOT / "data" / "images"
MODEL_DIR  = ROOT / "data" / "models"
OUT_PATH   = MODEL_DIR / "model_comparison.png"
NUM_WORKERS = min(8, (os.cpu_count() or 4))

# ---------------------------------------------------------------------------
# Eval transform (identical across all model versions)
# ---------------------------------------------------------------------------
eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# ---------------------------------------------------------------------------
# RAM-cached test dataset (load once, reuse across all models)
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
    """Sequential CNN used in v1, v2, v3."""
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
    """Residual CNN used in v4."""
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
# Checkpoint registry
# ---------------------------------------------------------------------------
CHECKPOINTS = [
    # (label, path, arch)  arch: "cnn" | "cnnv4" | "swa_v4"
    ("v1",        MODEL_DIR / "best_model.pth",        "cnn"),
    ("v3",        MODEL_DIR / "v3" / "best_model.pth", "cnn"),
    ("v4 (best)", MODEL_DIR / "v4" / "best_model.pth", "cnnv4"),
    ("v4 (SWA)",  MODEL_DIR / "v4" / "swa_model.pth",  "swa_v4"),
]


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    correct = total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Test ", unit="batch", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            correct    += (outputs.argmax(dim=1) == labels).sum().item()
            total      += images.size(0)
    return total_loss / total, correct / total


def load_model(arch: str, ckpt: dict, device: torch.device) -> nn.Module:
    classes    = ckpt["classes"]
    num_classes = len(classes)
    state      = ckpt["model_state"]

    if arch == "cnn":
        model = MalwareCNN(num_classes)
        model.load_state_dict(state)
    elif arch == "cnnv4":
        model = MalwareCNNv4(num_classes)
        model.load_state_dict(state)
    elif arch == "swa_v4":
        # SWA model was saved as AveragedModel.state_dict() (keys prefixed "module.")
        base = MalwareCNNv4(num_classes)
        swa  = AveragedModel(base)
        swa.load_state_dict(state)
        model = swa
    else:
        raise ValueError(f"Unknown arch: {arch}")

    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_comparison(results: list[dict], out_path: Path):
    labels    = [r["label"] for r in results]
    accs      = [r["test_acc"] * 100 for r in results]
    losses    = [r["test_loss"] for r in results]
    val_accs  = [r["val_acc"] * 100 for r in results]

    x = range(len(labels))
    bar_w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Model Comparison — Test Set", fontsize=14, fontweight="bold")

    # --- Accuracy bar chart ---
    ax = axes[0]
    bars = ax.bar([i - bar_w/2 for i in x], accs, width=bar_w,
                  color="#2196F3", label="Test Acc", zorder=3)
    ax.bar([i + bar_w/2 for i in x], val_accs, width=bar_w,
           color="#4CAF50", label="Best Val Acc (checkpoint)", zorder=3, alpha=0.85)
    ax.set_title("Accuracy")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    # Annotate bars with value
    for bar, v in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{v:.2f}%", ha="center", va="bottom", fontsize=7)

    # --- Loss bar chart ---
    ax = axes[1]
    bars2 = ax.bar(list(x), losses, width=0.5, color="#F44336", zorder=3)
    ax.set_title("Test Loss")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    for bar, v in zip(bars2, losses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{v:.4f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nComparison chart saved to: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device}\n")

    print("Loading test dataset into RAM ...")
    test_ds = CachedImageFolder(str(DATA_DIR / "test"), transform=eval_transform)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))

    results = []
    header = f"{'Model':<18}  {'Val Acc':>8}  {'Test Acc':>9}  {'Test Loss':>10}"
    sep    = "-" * len(header)
    print(f"\n{header}\n{sep}")

    for label, ckpt_path, arch in CHECKPOINTS:
        if not ckpt_path.exists():
            print(f"  [SKIP] {label}: {ckpt_path} not found")
            continue

        ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
        model = load_model(arch, ckpt, device)
        val_acc = ckpt.get("val_acc", float("nan"))

        print(f"  Evaluating {label} ...", flush=True)
        test_loss, test_acc = evaluate(model, test_loader, device)

        print(f"  {label:<16}  {val_acc:>7.2%}  {test_acc:>8.2%}  {test_loss:>10.4f}")
        results.append({"label": label, "val_acc": val_acc,
                        "test_acc": test_acc, "test_loss": test_loss})

        # Free GPU memory between models
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(sep)
    best = max(results, key=lambda r: r["test_acc"])
    print(f"\nBest model: {best['label']}  test_acc={best['test_acc']:.2%}  "
          f"test_loss={best['test_loss']:.4f}")

    plot_comparison(results, OUT_PATH)


if __name__ == "__main__":
    main()
