"""
train_cnn.py

CNN training script for BODMAS malware spiral-image classification.
Uses PyTorch ImageFolder — expects the dataset layout produced by
generate_dataset.py:

    data/images/
        train/<category>/<sha256>.png
        val/<category>/<sha256>.png
        test/<category>/<sha256>.png

Images are 128x128 RGB. Training auto-detects CUDA GPU; falls back to CPU if unavailable.
Best model (by val accuracy) is saved to data/models/best_model.pth.
"""

import csv
import io
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).resolve().parents[2]
DATA_DIR     = ROOT / "data" / "images"
MODEL_DIR    = ROOT / "data" / "models"
MODEL_PATH   = MODEL_DIR / "best_model.pth"
HISTORY_PATH = MODEL_DIR / "training_history.csv"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
IMAGE_SIZE    = 128
BATCH_SIZE    = 256
NUM_EPOCHS    = 100
LEARNING_RATE = 1e-3
NUM_WORKERS   = min(8, (os.cpu_count() or 4) - 1)  # capped to avoid spawn overhead on Windows
EARLY_STOP_PATIENCE = 10              # stop if val_acc doesn't improve for this many epochs

# ---------------------------------------------------------------------------
# Data transforms
# ---------------------------------------------------------------------------
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# ---------------------------------------------------------------------------
# RAM-cached dataset (eliminates HDD I/O after first load)
# ---------------------------------------------------------------------------
class CachedImageFolder(datasets.ImageFolder):
    """
    Loads every image as compressed bytes into RAM on init.
    __getitem__ decodes from BytesIO — zero disk I/O per batch.
    Requires ~2-3 GB RAM for this dataset; safe with 64 GB available.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(f"  Caching {len(self.samples)} images into RAM ...", flush=True)
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
# CNN Architecture
# ---------------------------------------------------------------------------
class MalwareCNN(nn.Module):
    """
    4-block CNN for 128x128 RGB spiral images.
    Each block: Conv2d -> BatchNorm -> ReLU -> MaxPool2d (halves spatial dims).
    Final FC layers produce class logits.

    Spatial progression: 128 -> 64 -> 32 -> 16 -> 8
    Feature maps:         3  -> 32 -> 64 -> 128 -> 256
    Flattened: 256 * 8 * 8 = 16384
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 128 -> 64
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 64 -> 32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 32 -> 16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4: 16 -> 8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# Train / eval helpers
# ---------------------------------------------------------------------------
def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    """
    Run one full pass over loader.
    Pass optimizer to train; omit (or pass None) to evaluate.
    Pass scaler (GradScaler) to enable AMP during training.
    Returns (avg_loss, accuracy).
    """
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    correct    = 0
    total      = 0

    phase = "Train" if training else "Val  "
    with torch.set_grad_enabled(training):
        pbar = tqdm(loader, desc=phase, unit="batch", leave=False)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs = model(images)
                loss    = criterion(outputs, labels)

            if training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct    += (outputs.argmax(dim=1) == labels).sum().item()
            total      += images.size(0)
            pbar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total:.2%}")

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True   # auto-tune fastest conv kernels for fixed input size
        print(f"Training on: cuda  ({torch.cuda.get_device_name(0)})")
    else:
        print(f"Training on: cpu")
    print(f"Workers for data loading: 0 (RAM-cache mode)\n")

    # Datasets — load all images into RAM to eliminate HDD bottleneck
    print("Loading datasets into RAM (one-time cost):")
    train_ds = CachedImageFolder(DATA_DIR / "train", transform=train_transform)
    val_ds   = CachedImageFolder(DATA_DIR / "val",   transform=eval_transform)
    test_ds  = CachedImageFolder(DATA_DIR / "test",  transform=eval_transform)

    num_classes = len(train_ds.classes)
    print(f"Classes ({num_classes}): {train_ds.classes}")
    print(f"Samples  — train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}\n")

    # num_workers=0: data already in RAM so main-process loading is fastest.
    # Spawning workers on Windows would pickle the entire cache into each worker.
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=pin,
    )

    # Model, loss, optimizer, LR scheduler
    model     = MalwareCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    scaler    = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    best_val_acc      = 0.0
    epochs_no_improve = 0
    start_epoch       = 1
    history           = []

    # Resume from last checkpoint if available
    ckpt_path = MODEL_DIR / "last_checkpoint.pth"
    if ckpt_path.exists():
        print(f"Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        if "scaler_state" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt["epoch"] + 1
        # Restore best val acc from best model checkpoint
        if MODEL_PATH.exists():
            best_ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
            best_val_acc = best_ckpt.get("val_acc", 0.0)
        # Reload history from CSV so it stays continuous
        if HISTORY_PATH.exists():
            with open(HISTORY_PATH, newline="") as f:
                for row in csv.DictReader(f):
                    history.append({
                        "epoch":      int(row["epoch"]),
                        "train_loss": float(row["train_loss"]),
                        "train_acc":  float(row["train_acc"]),
                        "val_loss":   float(row["val_loss"]),
                        "val_acc":    float(row["val_acc"]),
                        "lr":         float(row["lr"]),
                    })
        print(f"Resuming from epoch {start_epoch}  (best val acc so far: {best_val_acc:.2%})\n")
    else:
        print(f"No checkpoint found — starting from scratch.\n")

    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>8}  {'Val Acc':>7}  {'LR':>8}  {'Time':>6}")
    print("-" * 68)

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, device)

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        elapsed = time.time() - t0
        print(f"{epoch:>5}  {train_loss:>10.4f}  {train_acc:>8.2%}  "
              f"{val_loss:>8.4f}  {val_acc:>7.2%}  {current_lr:>8.2e}  {elapsed:>5.0f}s")

        # Record history
        history.append({
            "epoch":      epoch,
            "train_loss": train_loss,
            "train_acc":  train_acc,
            "val_loss":   val_loss,
            "val_acc":    val_acc,
            "lr":         current_lr,
        })

        if val_acc > best_val_acc:
            best_val_acc      = val_acc
            epochs_no_improve = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_acc":     val_acc,
                "classes":     train_ds.classes,
            }, MODEL_PATH)
            print(f"         -> New best saved (val_acc={val_acc:.2%})")
        else:
            epochs_no_improve += 1
            print(f"         -> No improvement ({epochs_no_improve}/{EARLY_STOP_PATIENCE})")
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping triggered at epoch {epoch}.")
                break

        # Per-epoch checkpoint (overwrites each epoch — keeps disk usage low)
        torch.save({
            "epoch":          epoch,
            "model_state":    model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state":   scaler.state_dict(),
            "val_acc":        val_acc,
            "classes":        train_ds.classes,
        }, MODEL_DIR / "last_checkpoint.pth")

    # Save training history to CSV for graphing
    with open(HISTORY_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc",
                                               "val_loss", "val_acc", "lr"])
        writer.writeheader()
        writer.writerows(history)
    print(f"Training history saved to {HISTORY_PATH}")

    # Final test evaluation using best checkpoint
    print(f"\nLoading best model from {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])

    test_loss, test_acc = run_epoch(model, test_loader, criterion, device)
    print(f"Test loss: {test_loss:.4f}  Test accuracy: {test_acc:.2%}")
    print(f"Best val accuracy achieved at epoch {checkpoint['epoch']}: {checkpoint['val_acc']:.2%}")


if __name__ == "__main__":
    main()
