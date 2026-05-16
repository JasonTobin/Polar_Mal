"""
train_v4_greyscale_wce.py

Imbalance strategy: Weighted CrossEntropyLoss only.
  - Class weights w_i = total / (num_classes * count_i) passed to
    nn.CrossEntropyLoss(weight=...).  Rare classes (cryptominer=14,
    other=26, pua=20) receive >1000× higher per-sample loss weight
    than benign (53 999), forcing the model to pay attention to them.
  - WeightedRandomSampler is NOT used; batches are still drawn uniformly
    at random.  This isolates the loss-weighting effect for ablation.

Other changes vs train_v4.py:
  - run_epoch(return_preds=False) — set True to accumulate y_true/y_pred
  - val_macro_f1 computed every epoch; used for early stopping and the
    second (f1) best-model checkpoint.
  - Two best-model checkpoints:
      best_model_acc.pth  — highest val_acc
      best_model_f1.pth   — highest val_macro_f1
  - training_history.csv gains a val_macro_f1 column.
  - All other hyperparameters (SWA, AdamW, cosine LR, AMP, etc.) are
    identical to train_v4.py.

Saves to data/models/greyscale/v4_wce/
"""

import csv
import io
import os
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import f1_score
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT          = Path(__file__).resolve().parents[2]
DATA_DIR      = ROOT / "data" / "images_greyscale"
MODEL_DIR     = ROOT / "data" / "models" / "greyscale" / "v4_wce"
MODEL_ACC_PATH = MODEL_DIR / "best_model_acc.pth"
MODEL_F1_PATH  = MODEL_DIR / "best_model_f1.pth"
CKPT_PATH     = MODEL_DIR / "last_checkpoint.pth"
HISTORY_PATH  = MODEL_DIR / "training_history.csv"
SWA_PATH      = MODEL_DIR / "swa_model.pth"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE          = 512
NUM_WORKERS         = min(8, (os.cpu_count() or 4))
NUM_EPOCHS          = 150
LEARNING_RATE       = 1e-3
EARLY_STOP_PATIENCE = 15   # phase-1 only; SWA phase always runs to completion
SWA_START           = 100
SWA_LR              = 1e-5

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
# Architecture: CNN with residual connections (identical to v4)
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Train / eval helper
# ---------------------------------------------------------------------------
def run_epoch(model, loader, criterion, device,
              optimizer=None, scaler=None, return_preds: bool = False):
    """
    Run one epoch.  Returns (loss, acc) or (loss, acc, y_true, y_pred)
    when return_preds=True.  Use return_preds=True for validation so that
    macro-F1 can be computed without a second pass over the data.
    """
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    correct    = 0
    total      = 0
    all_true: list[int] = []
    all_pred: list[int] = []

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

            preds = outputs.argmax(dim=1)
            total_loss += loss.item() * images.size(0)
            correct    += (preds == labels).sum().item()
            total      += images.size(0)
            pbar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total:.2%}")

            if return_preds:
                all_true.extend(labels.cpu().tolist())
                all_pred.extend(preds.cpu().tolist())

    loss_avg = total_loss / total
    acc      = correct / total
    if return_preds:
        return loss_avg, acc, all_true, all_pred
    return loss_avg, acc


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
    print(f"Imbalance strategy: Weighted CrossEntropyLoss only\n")

    print("Loading datasets into RAM (one-time cost):")
    train_ds = CachedImageFolder(DATA_DIR / "train", transform=train_transform)
    val_ds   = CachedImageFolder(DATA_DIR / "val",   transform=eval_transform)
    test_ds  = CachedImageFolder(DATA_DIR / "test",  transform=eval_transform)

    num_classes = len(train_ds.classes)
    print(f"Classes ({num_classes}): {train_ds.classes}")
    print(f"Samples  — train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}\n")

    # --- Compute per-class weights for CE loss (sqrt of inverse frequency) ---
    # Linear inverse-freq weights (e.g. 560x for cryptominer) cause gradient collapse
    # when most batches have zero minority samples.  Sqrt scaling keeps the ratio at
    # ~62:1 max while still giving meaningful emphasis to rare classes.
    class_counts = Counter(train_ds.targets)
    total_samples = len(train_ds.targets)
    class_weights = torch.tensor(
        [(total_samples / (num_classes * class_counts[i])) ** 0.5 for i in range(num_classes)],
        dtype=torch.float,
    )
    print("Class sqrt-weights for WCE loss (sqrt of inverse frequency):")
    for i, cls in enumerate(train_ds.classes):
        print(f"  [{i:2d}] {cls:<22} count={class_counts[i]:6d}  sqrt_weight={class_weights[i]:.4f}")
    print()

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

    model     = MalwareCNNv4(num_classes=num_classes).to(device)
    swa_model = AveragedModel(model)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device), label_smoothing=0.05
    )
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler     = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )
    swa_scheduler = SWALR(optimizer, swa_lr=SWA_LR)
    scaler        = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    best_val_acc      = 0.0
    best_val_f1       = 0.0
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
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        best_val_f1  = ckpt.get("best_val_f1", 0.0)
        if HISTORY_PATH.exists():
            with open(HISTORY_PATH, newline="") as f:
                for row in csv.DictReader(f):
                    history.append({
                        "epoch":        int(row["epoch"]),
                        "train_loss":   float(row["train_loss"]),
                        "train_acc":    float(row["train_acc"]),
                        "val_loss":     float(row["val_loss"]),
                        "val_acc":      float(row["val_acc"]),
                        "val_macro_f1": float(row["val_macro_f1"]),
                        "lr":           float(row["lr"]),
                    })
        print(f"Resuming from epoch {start_epoch}  "
              f"(best val_acc={best_val_acc:.2%}, best val_f1={best_val_f1:.4f}, "
              f"SWA phase: {in_swa_phase})\n")
    else:
        print("No checkpoint found — starting from scratch.\n")

    print(f"Early stopping patience : {EARLY_STOP_PATIENCE} epochs (phase 1 only, "
          f"based on val_macro_f1)")
    print(f"SWA phase               : epochs {SWA_START} - {NUM_EPOCHS} "
          f"({NUM_EPOCHS - SWA_START + 1} averaging steps)\n")
    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>8}  {'Val Acc':>7}  {'Val F1':>7}  {'LR':>10}  {'Time':>6}  {'Phase':>5}")
    print("-" * 91)

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        t0 = time.time()

        if epoch >= SWA_START and not in_swa_phase:
            in_swa_phase = True

        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler
        )

        if in_swa_phase:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        val_loss, val_acc, y_true, y_pred = run_epoch(
            model, val_loader, criterion, device, return_preds=True
        )
        val_macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        elapsed   = time.time() - t0
        phase_str = "SWA" if in_swa_phase else "base"
        print(f"{epoch:>5}  {train_loss:>10.4f}  {train_acc:>8.2%}  "
              f"{val_loss:>8.4f}  {val_acc:>7.2%}  {val_macro_f1:>7.4f}  "
              f"{current_lr:>10.2e}  {elapsed:>5.0f}s  {phase_str:>5}")

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
            "val_macro_f1": val_macro_f1, "lr": current_lr,
        })

        if not in_swa_phase:
            improved = False

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    "epoch": epoch, "model_state": model.state_dict(),
                    "val_acc": val_acc, "val_macro_f1": val_macro_f1,
                    "classes": train_ds.classes,
                }, MODEL_ACC_PATH)
                print(f"         -> New best val_acc={val_acc:.2%} saved to best_model_acc.pth")
                improved = True

            if val_macro_f1 > best_val_f1:
                best_val_f1 = val_macro_f1
                epochs_no_improve = 0
                torch.save({
                    "epoch": epoch, "model_state": model.state_dict(),
                    "val_acc": val_acc, "val_macro_f1": val_macro_f1,
                    "classes": train_ds.classes,
                }, MODEL_F1_PATH)
                print(f"         -> New best val_macro_f1={val_macro_f1:.4f} saved to best_model_f1.pth")
                improved = True

            if not improved:
                epochs_no_improve += 1
                print(f"         -> No improvement in macro-F1 ({epochs_no_improve}/{EARLY_STOP_PATIENCE})")
                if epochs_no_improve >= EARLY_STOP_PATIENCE:
                    print(f"\nEarly stopping triggered at epoch {epoch} "
                          f"(before SWA phase — SWA skipped).")
                    break
        else:
            swa_step = epoch - SWA_START + 1
            print(f"         -> SWA update #{swa_step}")

        torch.save({
            "epoch":               epoch,
            "model_state":         model.state_dict(),
            "swa_model_state":     swa_model.state_dict(),
            "optimizer_state":     optimizer.state_dict(),
            "scheduler_state":     scheduler.state_dict(),
            "swa_scheduler_state": swa_scheduler.state_dict(),
            "scaler_state":        scaler.state_dict(),
            "val_acc":             val_acc,
            "best_val_acc":        best_val_acc,
            "best_val_f1":         best_val_f1,
            "in_swa_phase":        in_swa_phase,
            "classes":             train_ds.classes,
        }, CKPT_PATH)

        with open(HISTORY_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc",
                                                   "val_loss", "val_acc", "val_macro_f1", "lr"])
            writer.writeheader()
            writer.writerows(history)

    print(f"\nTraining history saved to {HISTORY_PATH}")

    # --- SWA final evaluation ---
    if in_swa_phase:
        print("\nUpdating SWA model batch norm statistics ...")
        update_bn(train_loader, swa_model, device=device)

        print("Evaluating SWA model on validation set ...")
        swa_val_loss, swa_val_acc, swa_y_true, swa_y_pred = run_epoch(
            swa_model, val_loader, criterion, device, return_preds=True
        )
        swa_val_f1 = f1_score(swa_y_true, swa_y_pred, average="macro", zero_division=0)
        print(f"SWA model — val  loss: {swa_val_loss:.4f}  val  acc: {swa_val_acc:.2%}  "
              f"val  macro-F1: {swa_val_f1:.4f}")

        print("Evaluating SWA model on test set ...")
        swa_test_loss, swa_test_acc = run_epoch(swa_model, test_loader, criterion, device)
        print(f"SWA model — test loss: {swa_test_loss:.4f}  test acc: {swa_test_acc:.2%}")

        torch.save({
            "epoch": NUM_EPOCHS, "model_state": swa_model.state_dict(),
            "val_acc": swa_val_acc, "val_macro_f1": swa_val_f1,
            "classes": train_ds.classes,
        }, SWA_PATH)
        print(f"SWA model saved to {SWA_PATH}")

    # --- Final test evaluation with best F1 model ---
    if MODEL_F1_PATH.exists():
        print(f"\nLoading best macro-F1 model from {MODEL_F1_PATH}")
        ckpt = torch.load(MODEL_F1_PATH, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        test_loss, test_acc = run_epoch(model, test_loader, criterion, device)
        print(f"[best_model_f1] Test loss: {test_loss:.4f}  Test accuracy: {test_acc:.2%}")
        print(f"  Saved at epoch {ckpt['epoch']}  val_acc={ckpt['val_acc']:.2%}  "
              f"val_macro_f1={ckpt['val_macro_f1']:.4f}")

    if MODEL_ACC_PATH.exists():
        print(f"\nLoading best val_acc model from {MODEL_ACC_PATH}")
        ckpt = torch.load(MODEL_ACC_PATH, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        test_loss, test_acc = run_epoch(model, test_loader, criterion, device)
        print(f"[best_model_acc] Test loss: {test_loss:.4f}  Test accuracy: {test_acc:.2%}")
        print(f"  Saved at epoch {ckpt['epoch']}  val_acc={ckpt['val_acc']:.2%}  "
              f"val_macro_f1={ckpt['val_macro_f1']:.4f}")


if __name__ == "__main__":
    main()
