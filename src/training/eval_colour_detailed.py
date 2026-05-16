"""
eval_colour_detailed.py

Per-model detailed evaluation for all colour (non-greyscale) models.

For each model produces:
  1. Confusion matrix heatmap
  2. Per-class metrics bar chart (Precision, Recall, F1, FNR)

Plus cross-model charts:
  3. Per-class F1 heatmap (all models)
  4. Summary metrics comparison (Accuracy, Macro-F1, Weighted-F1, MCC)

Outputs saved under data/models/detailed/.

Usage:
    python3 src/training/eval_colour_detailed.py
"""

import io
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
    f1_score,
)
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT      = Path(__file__).resolve().parents[2]
DATA_DIR  = ROOT / "data" / "images"          # colour images
MODEL_DIR = ROOT / "data" / "models"          # colour model root
OUT_DIR   = MODEL_DIR / "detailed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_WORKERS = min(8, (os.cpu_count() or 4))

MODELS = [
    # (short_label, display_label, ckpt_path, arch)
    ("cnn",              "CNN (v1)",              MODEL_DIR / "best_model.pth",                        "cnn"),
    ("v2",               "V2",                    MODEL_DIR / "v2" / "best_model.pth",                 "cnn"),
    ("v3",               "V3",                    MODEL_DIR / "v3" / "best_model.pth",                 "cnn"),
    ("v4",               "V4 (ResNet)",           MODEL_DIR / "v4" / "best_model.pth",                 "cnnv4"),
    ("v4_swa",           "V4 (SWA)",              MODEL_DIR / "v4" / "swa_model.pth",                  "swa_v4"),
    # Imbalance-aware variants (created after class-balance work)
    ("v4_wce_acc",       "V4 WCE (best acc)",     MODEL_DIR / "v4_wce"      / "best_model_acc.pth",    "cnnv4"),
    ("v4_wce_f1",        "V4 WCE (best F1)",      MODEL_DIR / "v4_wce"      / "best_model_f1.pth",     "cnnv4"),
    ("v4_sampler_acc",   "V4 Sampler (best acc)", MODEL_DIR / "v4_sampler"  / "best_model_acc.pth",    "cnnv4"),
    ("v4_sampler_f1",    "V4 Sampler (best F1)",  MODEL_DIR / "v4_sampler"  / "best_model_f1.pth",     "cnnv4"),
    ("v4_balanced_acc",  "V4 Balanced (best acc)",MODEL_DIR / "v4_balanced" / "best_model_acc.pth",    "cnnv4"),
    ("v4_balanced_f1",   "V4 Balanced (best F1)", MODEL_DIR / "v4_balanced" / "best_model_f1.pth",     "cnnv4"),
]

MODEL_COLOURS = [
    "#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0",  # baseline
    "#00BCD4", "#0097A7", "#8BC34A", "#558B2F", "#FF5722", "#BF360C",  # balanced
]

# ---------------------------------------------------------------------------
# Transforms / dataset
# ---------------------------------------------------------------------------
eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


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
        x = self.block1(x); x = self.block2(x)
        x = self.block3(x); x = self.block4(x)
        return self.classifier(x)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_model(arch: str, ckpt: dict, device: torch.device) -> nn.Module:
    num_classes = len(ckpt["classes"])
    if arch == "cnn":
        model = MalwareCNN(num_classes)
        model.load_state_dict(ckpt["model_state"])
    elif arch == "cnnv4":
        model = MalwareCNNv4(num_classes)
        model.load_state_dict(ckpt["model_state"])
    elif arch == "swa_v4":
        base = MalwareCNNv4(num_classes)
        swa  = AveragedModel(base)
        swa.load_state_dict(ckpt["model_state"])
        model = swa
    else:
        raise ValueError(f"Unknown arch: {arch}")
    return model.to(device).eval()


def run_inference(model: nn.Module, loader: DataLoader,
                  device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Inference", unit="batch", leave=False):
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                preds = model(images).argmax(dim=1).cpu().numpy()
            all_labels.append(labels.numpy())
            all_preds.append(preds)
    return np.concatenate(all_labels), np.concatenate(all_preds)

# ---------------------------------------------------------------------------
# Plot 1: Confusion matrix heatmap
# ---------------------------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, class_names, display_label, out_path):
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    n = len(class_names)
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(f"Colour {display_label} — Confusion Matrix", fontsize=14, fontweight="bold")

    for ax, data, title, fmt in [
        (axes[0], cm,      "Raw counts",             "d"),
        (axes[1], cm_norm, "Row-normalised (recall)", ".2f"),
    ]:
        im = ax.imshow(data, interpolation="nearest",
                       cmap="Blues" if fmt == "d" else "RdYlGn",
                       vmin=0, vmax=(None if fmt == "d" else 1))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True", fontsize=10)
        ax.set_title(title, fontsize=11)
        thresh = data.max() / 2.0 if fmt == "d" else 0.5
        for i in range(n):
            for j in range(n):
                val = data[i, j]
                ax.text(j, i, f"{val:{fmt}}" if fmt == "d" else f"{val:.2f}",
                        ha="center", va="center", fontsize=6,
                        color="white" if val > thresh else "black")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.relative_to(ROOT)}")

# ---------------------------------------------------------------------------
# Plot 2: Per-class Precision / Recall / F1 + FNR
# ---------------------------------------------------------------------------
def plot_per_class_metrics(y_true, y_pred, class_names, display_label, out_path):
    report    = classification_report(y_true, y_pred, target_names=class_names,
                                      output_dict=True, zero_division=0)
    precision = [report[c]["precision"] * 100 for c in class_names]
    recall    = [report[c]["recall"]    * 100 for c in class_names]
    f1        = [report[c]["f1-score"]  * 100 for c in class_names]
    support   = [report[c]["support"]         for c in class_names]

    x     = np.arange(len(class_names))
    bar_w = 0.28

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(f"Colour {display_label} — Per-Class Metrics", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.bar(x - bar_w, precision, width=bar_w, color="#2196F3", label="Precision", zorder=3)
    ax.bar(x,         recall,    width=bar_w, color="#4CAF50", label="Recall",    zorder=3)
    ax.bar(x + bar_w, f1,        width=bar_w, color="#FF9800", label="F1",        zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Precision / Recall / F1 per Class")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    for xi, v in zip(x + bar_w, f1):
        ax.text(xi, v + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=6.5)

    fnr     = [100 - r for r in recall]
    colours = ["#F44336" if v > 10 else "#FF9800" if v > 5 else "#4CAF50" for v in fnr]
    ax2 = axes[1]
    bars = ax2.bar(x, fnr, color=colours, zorder=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax2.set_ylabel("False Negative Rate (%)")
    ax2.set_title("False Negative Rate per Class  (lower = better)  "
                  "  \u2502  green <5%  \u2502  orange 5-10%  \u2502  red >10%")
    ax2.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    for bar, v, n_ in zip(bars, fnr, support):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 f"{v:.1f}%\n(n={n_})", ha="center", va="bottom", fontsize=6.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.relative_to(ROOT)}")

# ---------------------------------------------------------------------------
# Plot 3: Cross-model per-class F1 heatmap
# ---------------------------------------------------------------------------
def plot_cross_model_f1(all_results, class_names, out_path):
    model_labels = [r["display"] for r in all_results]
    data = np.array([
        [r["report"][c]["f1-score"] * 100 for c in class_names]
        for r in all_results
    ])

    fig, ax = plt.subplots(figsize=(16, len(model_labels) + 2))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="F1 (%)")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(model_labels)))
    ax.set_yticklabels(model_labels, fontsize=10)
    ax.set_title("Per-Class F1 Score (%) by Model  —  Colour", fontsize=13, fontweight="bold")
    for i, row in enumerate(data):
        for j, val in enumerate(row):
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=8, color="black" if 20 < val < 85 else "white")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.relative_to(ROOT)}")

# ---------------------------------------------------------------------------
# Plot 4: Summary metrics comparison
# ---------------------------------------------------------------------------
def plot_summary_metrics(all_results, out_path):
    labels      = [r["display"]     for r in all_results]
    accuracy    = [r["accuracy"]    * 100 for r in all_results]
    macro_f1    = [r["macro_f1"]    * 100 for r in all_results]
    weighted_f1 = [r["weighted_f1"] * 100 for r in all_results]
    mcc         = [r["mcc"]               for r in all_results]

    x     = np.arange(len(labels))
    bar_w = 0.22

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Colour Models — Summary Metrics", fontsize=14, fontweight="bold")

    ax = axes[0]
    b1 = ax.bar(x - bar_w,     accuracy,    width=bar_w, color="#2196F3", label="Accuracy",    zorder=3)
    b2 = ax.bar(x,             macro_f1,    width=bar_w, color="#4CAF50", label="Macro-F1",    zorder=3)
    b3 = ax.bar(x + bar_w,     weighted_f1, width=bar_w, color="#FF9800", label="Weighted-F1", zorder=3)
    ax.set_title("Accuracy & F1 Scores")
    ax.set_ylabel("Score (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    for bars, vals in [(b1, accuracy), (b2, macro_f1), (b3, weighted_f1)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    ax = axes[1]
    bars = ax.bar(x, mcc, width=0.5,
                  color=[MODEL_COLOURS[i % len(MODEL_COLOURS)] for i in range(len(labels))],
                  zorder=3)
    ax.set_title("Matthews Correlation Coefficient")
    ax.set_ylabel("MCC  (1.0 = perfect)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    for bar, v in zip(bars, mcc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
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

    print("Loading colour test dataset ...")
    test_ds = CachedImageFolder(str(DATA_DIR / "test"), transform=eval_transform)
    test_loader = DataLoader(
        test_ds, batch_size=512, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
        persistent_workers=True, prefetch_factor=2,
    )
    class_names = test_ds.classes

    all_results = []

    for short, display, ckpt_path, arch in MODELS:
        if not ckpt_path.exists():
            print(f"\n[SKIP] {display}: {ckpt_path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"  Evaluating: {display}")
        print(f"{'='*60}")

        ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
        model = load_model(arch, ckpt, device)

        y_true, y_pred = run_inference(model, test_loader, device)

        report      = classification_report(y_true, y_pred, target_names=class_names,
                                            output_dict=True, zero_division=0)
        accuracy    = report["accuracy"]
        macro_f1    = f1_score(y_true, y_pred, average="macro",    zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        mcc         = matthews_corrcoef(y_true, y_pred)

        print(f"  Accuracy:    {accuracy:.4f}  ({accuracy*100:.2f}%)")
        print(f"  Macro-F1:    {macro_f1:.4f}")
        print(f"  Weighted-F1: {weighted_f1:.4f}")
        print(f"  MCC:         {mcc:.4f}")

        prefix = OUT_DIR / f"colour_{short}"
        plot_confusion_matrix(y_true, y_pred, class_names, display,
                              Path(str(prefix) + "_confusion_matrix.png"))
        plot_per_class_metrics(y_true, y_pred, class_names, display,
                               Path(str(prefix) + "_per_class_metrics.png"))

        all_results.append({
            "short": short, "display": display,
            "y_true": y_true, "y_pred": y_pred,
            "report": report, "accuracy": accuracy,
            "macro_f1": macro_f1, "weighted_f1": weighted_f1, "mcc": mcc,
        })

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if all_results:
        print(f"\n{'='*60}")
        print("  Generating cross-model charts ...")
        print(f"{'='*60}")
        plot_cross_model_f1(all_results, class_names,
                            OUT_DIR / "colour_cross_model_f1_heatmap.png")
        plot_summary_metrics(all_results,
                             OUT_DIR / "colour_summary_metrics.png")

    print("\nDone.  All outputs in:", OUT_DIR.relative_to(ROOT))


if __name__ == "__main__":
    main()
