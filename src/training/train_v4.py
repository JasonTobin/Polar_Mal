"""
train_v4.py

V4 improvements over V3:
  - Residual connections: each conv block becomes a ResBlock with a 2-conv
    main path and a 1x1 skip projection, letting gradients flow directly
    to earlier layers.
  - Stochastic Weight Averaging (SWA): epochs 1-(SWA_START-1) train normally
    with CosineAnnealingLR; from SWA_START onwards the averaged model is
    updated every epoch with a constant SWALR.  After all epochs, BN stats
    are re-computed on the training set and the SWA model is evaluated.
  - Longer training: NUM_EPOCHS=150, SWA_START=100 (51 SWA epochs)
  - Increased early stopping patience: 15 (phase-1 only; SWA phase never
    stops early)
  - All V3 improvements retained: AdamW, CosineAnnealingLR, label_smoothing
    =0.05, RandomErasing, AMP, RAM cache

Saves to data/models/v4/
"""

import csv
import io
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).resolve().parents[2]
DATA_DIR     = ROOT / "data" / "images"
MODEL_DIR    = ROOT / "data" / "models" / "v4"
MODEL_PATH   = MODEL_DIR / "best_model.pth"
CKPT_PATH    = MODEL_DIR / "last_checkpoint.pth"
HISTORY_PATH = MODEL_DIR / "training_history.csv"
SWA_PATH     = MODEL_DIR / "swa_model.pth"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE          = 256
NUM_EPOCHS          = 150
LEARNING_RATE       = 1e-3
EARLY_STOP_PATIENCE = 15   # phase-1 only; SWA phase always runs to completion
SWA_START           = 100  # epoch at which SWA averaging begins
SWA_LR              = 1e-5 # constant LR during SWA phase

# ---------------------------------------------------------------------------
# Data transforms (identical to V3)
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
# RAM-cached dataset (identical to V3)
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
# Architecture: CNN with residual connections
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    """
    Residual block: a 2-conv main path with BN+ReLU, a 1x1 skip projection
    to match output channels, element-wise addition, ReLU, then MaxPool(2).

    Spatial reduction: handled by MaxPool after the residual add.
    Channels always increase (3->32->64->128->256), so the skip always uses
    a learned 1x1 projection rather than an identity shortcut.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        # 1x1 projection so skip has the same channel depth as main output
        self.skip = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.relu(self.main(x) + self.skip(x)))


class MalwareCNNv4(nn.Module):
    """
    4-block residual CNN for 128x128 RGB spiral images.
    Spatial: 128 -> 64 -> 32 -> 16 -> 8
    Channels:   3 -> 32 -> 64 -> 128 -> 256
    Flattened: 256 * 8 * 8 = 16384
    """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.classifier(x)


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
    print("Workers for data loading: 0 (RAM-cache mode)\n")

    print("Loading datasets into RAM (one-time cost):")
    train_ds = CachedImageFolder(DATA_DIR / "train", transform=train_transform)
    val_ds   = CachedImageFolder(DATA_DIR / "val",   transform=eval_transform)
    test_ds  = CachedImageFolder(DATA_DIR / "test",  transform=eval_transform)

    num_classes = len(train_ds.classes)
    print(f"Classes ({num_classes}): {train_ds.classes}")
    print(f"Samples  — train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}\n")

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=pin)

    model     = MalwareCNNv4(num_classes=num_classes).to(device)
    swa_model = AveragedModel(model)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    # Phase 1: cosine decay 1e-3 -> 1e-6 over NUM_EPOCHS
    scheduler     = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )
    # Phase 2: constant SWA_LR (activated at SWA_START)
    swa_scheduler = SWALR(optimizer, swa_lr=SWA_LR)
    scaler        = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    best_val_acc      = 0.0
    epochs_no_improve = 0
    start_epoch       = 1
    in_swa_phase      = False
    history           = []

    # --- Resume from checkpoint ---
    if CKPT_PATH.exists():
        print(f"Resuming from checkpoint: {CKPT_PATH}")
        ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        if "scaler_state" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state"])
        if "swa_model_state" in ckpt:
            swa_model.load_state_dict(ckpt["swa_model_state"])
        if "swa_scheduler_state" in ckpt:
            swa_scheduler.load_state_dict(ckpt["swa_scheduler_state"])
        in_swa_phase = ckpt.get("in_swa_phase", False)
        start_epoch  = ckpt["epoch"] + 1
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
        print(f"Resuming from epoch {start_epoch}  "
              f"(best val acc so far: {best_val_acc:.2%}, SWA phase: {in_swa_phase})\n")
    else:
        print("No checkpoint found — starting from scratch.\n")

    print(f"Early stopping patience : {EARLY_STOP_PATIENCE} epochs (phase 1 only)")
    print(f"SWA phase               : epochs {SWA_START} - {NUM_EPOCHS} "
          f"({NUM_EPOCHS - SWA_START + 1} averaging steps)\n")
    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>8}  {'Val Acc':>7}  {'LR':>10}  {'Time':>6}  {'Phase':>5}")
    print("-" * 82)

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        t0 = time.time()

        # Transition to SWA phase
        if epoch >= SWA_START and not in_swa_phase:
            in_swa_phase = True

        # LR actually used for this epoch's training step
        current_lr = optimizer.param_groups[0]["lr"]

        # --- Train ---
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler
        )

        # --- Update SWA model / step scheduler ---
        if in_swa_phase:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        # --- Validate (regular model) ---
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)

        elapsed   = time.time() - t0
        phase_str = "SWA" if in_swa_phase else "base"
        print(f"{epoch:>5}  {train_loss:>10.4f}  {train_acc:>8.2%}  "
              f"{val_loss:>8.4f}  {val_acc:>7.2%}  {current_lr:>10.2e}  "
              f"{elapsed:>5.0f}s  {phase_str:>5}")

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "lr": current_lr,
        })

        # --- Early stopping (phase 1 only) ---
        if not in_swa_phase:
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
                    print(f"\nEarly stopping triggered at epoch {epoch} "
                          f"(before SWA phase — SWA skipped).")
                    break
        else:
            swa_step = epoch - SWA_START + 1
            print(f"         -> SWA update #{swa_step}")

        # --- Checkpoint ---
        torch.save({
            "epoch":               epoch,
            "model_state":         model.state_dict(),
            "swa_model_state":     swa_model.state_dict(),
            "optimizer_state":     optimizer.state_dict(),
            "scheduler_state":     scheduler.state_dict(),
            "swa_scheduler_state": swa_scheduler.state_dict(),
            "scaler_state":        scaler.state_dict(),
            "val_acc":             val_acc,
            "in_swa_phase":        in_swa_phase,
            "classes":             train_ds.classes,
        }, CKPT_PATH)

        # --- Per-epoch CSV (enables mid-run graph generation) ---
        with open(HISTORY_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc",
                                                   "val_loss", "val_acc", "lr"])
            writer.writeheader()
            writer.writerows(history)

    print(f"\nTraining history saved to {HISTORY_PATH}")

    # --- SWA final evaluation ---
    if in_swa_phase:
        print("\nUpdating SWA model batch norm statistics (one pass over training set)...")
        update_bn(train_loader, swa_model, device=device)

        print("Evaluating SWA model on validation set...")
        swa_val_loss, swa_val_acc = run_epoch(swa_model, val_loader, criterion, device)
        print(f"SWA model — val  loss: {swa_val_loss:.4f}  val  acc: {swa_val_acc:.2%}")

        print("Evaluating SWA model on test set...")
        swa_test_loss, swa_test_acc = run_epoch(swa_model, test_loader, criterion, device)
        print(f"SWA model — test loss: {swa_test_loss:.4f}  test acc: {swa_test_acc:.2%}")

        # Save SWA model (loadable via AveragedModel(MalwareCNNv4(...)).load_state_dict(...))
        torch.save({
            "epoch": NUM_EPOCHS, "model_state": swa_model.state_dict(),
            "val_acc": swa_val_acc, "classes": train_ds.classes,
        }, SWA_PATH)
        print(f"SWA model saved to {SWA_PATH}")

        if swa_val_acc > best_val_acc:
            print(f"\nSWA model ({swa_val_acc:.2%}) beats best regular model "
                  f"({best_val_acc:.2%}).")
        else:
            print(f"\nBest regular model ({best_val_acc:.2%}) retained over "
                  f"SWA ({swa_val_acc:.2%}).")

    # --- Final test evaluation with best regular model ---
    print(f"\nLoading best regular model from {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_acc = run_epoch(model, test_loader, criterion, device)
    print(f"Test loss: {test_loss:.4f}  Test accuracy: {test_acc:.2%}")
    print(f"Best val accuracy at epoch {checkpoint['epoch']}: {checkpoint['val_acc']:.2%}")


if __name__ == "__main__":
    main()
