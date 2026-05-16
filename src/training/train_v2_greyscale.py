"""
train_v2_greyscale.py

Greyscale variant of train_v2_phase1.py.
The only differences are that DATA_DIR points to data/images_greyscale/
and outputs are saved to data/models/greyscale/v2/.
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
# Paths  (greyscale v2 lives in its own subdirectory)
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).resolve().parents[2]
DATA_DIR     = ROOT / "data" / "images_greyscale"
MODEL_DIR    = ROOT / "data" / "models" / "greyscale" / "v2"
MODEL_PATH   = MODEL_DIR / "best_model.pth"
CKPT_PATH    = MODEL_DIR / "last_checkpoint.pth"
HISTORY_PATH = MODEL_DIR / "training_history.csv"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
IMAGE_SIZE          = 128
BATCH_SIZE          = 512
NUM_EPOCHS          = 100
LEARNING_RATE       = 1e-3
EARLY_STOP_PATIENCE = 10
NUM_WORKERS         = min(8, (os.cpu_count() or 4))

# ---------------------------------------------------------------------------
# Data transforms
# ---------------------------------------------------------------------------
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
])

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
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

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
        torch.backends.cudnn.benchmark = True
        print(f"Training on: cuda  ({torch.cuda.get_device_name(0)})")
    else:
        print("Training on: cpu")
    print(f"Workers for data loading: {NUM_WORKERS} (RAM-cache + fork)\n")

    print("Loading datasets into RAM (one-time cost):")
    train_ds = CachedImageFolder(DATA_DIR / "train", transform=train_transform)
    val_ds   = CachedImageFolder(DATA_DIR / "val",   transform=eval_transform)
    test_ds  = CachedImageFolder(DATA_DIR / "test",  transform=eval_transform)

    num_classes = len(train_ds.classes)
    print(f"Classes ({num_classes}): {train_ds.classes}")
    print(f"Samples  — train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}\n")

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin,
                              persistent_workers=True, prefetch_factor=4)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS // 2, pin_memory=pin,
                              persistent_workers=True, prefetch_factor=2)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS // 2, pin_memory=pin,
                              persistent_workers=True, prefetch_factor=2)

    model     = MalwareCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    best_val_acc      = 0.0
    epochs_no_improve = 0
    start_epoch       = 1
    history           = []

    if CKPT_PATH.exists():
        print(f"Resuming from checkpoint: {CKPT_PATH}")
        ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        if "scaler_state" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt["epoch"] + 1
        if MODEL_PATH.exists():
            best_ckpt    = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
            best_val_acc = best_ckpt.get("val_acc", 0.0)
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
        print("No checkpoint found — starting from scratch.\n")

    print(f"Early stopping patience: {EARLY_STOP_PATIENCE} epochs\n")
    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>8}  {'Val Acc':>7}  {'LR':>10}  {'Time':>6}")
    print("-" * 72)

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, device)

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        elapsed = time.time() - t0
        print(f"{epoch:>5}  {train_loss:>10.4f}  {train_acc:>8.2%}  "
              f"{val_loss:>8.4f}  {val_acc:>7.2%}  {current_lr:>10.2e}  {elapsed:>5.0f}s")

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "lr": current_lr,
        })

        if val_acc > best_val_acc:
            best_val_acc      = val_acc
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "val_acc": val_acc, "classes": train_ds.classes,
            }, MODEL_PATH)
            print(f"         -> New best saved (val_acc={val_acc:.2%})")
        else:
            epochs_no_improve += 1
            print(f"         -> No improvement ({epochs_no_improve}/{EARLY_STOP_PATIENCE})")
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping triggered at epoch {epoch}.")
                break

        torch.save({
            "epoch": epoch, "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "val_acc": val_acc, "classes": train_ds.classes,
        }, CKPT_PATH)

        with open(HISTORY_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc",
                                                   "val_loss", "val_acc", "lr"])
            writer.writeheader()
            writer.writerows(history)

    print(f"\nTraining history saved to {HISTORY_PATH}")

    print(f"\nLoading best model from {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])

    test_loss, test_acc = run_epoch(model, test_loader, criterion, device)
    print(f"Test loss: {test_loss:.4f}  Test accuracy: {test_acc:.2%}")
    print(f"Best val accuracy at epoch {checkpoint['epoch']}: {checkpoint['val_acc']:.2%}")


if __name__ == "__main__":
    main()