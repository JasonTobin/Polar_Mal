"""
generate_dataset.py

Converts all BODMAS feature vectors into spiral images organized for
PyTorch ImageFolder:

    data/images/<split>/<category>/<sha256>.png

Splits: train (70%) / val (15%) / test (15%), stratified by category.
Global per-feature normalization bounds are computed once and cached in
data/spiral_bounds.npz so every image uses a consistent color scale.
"""

import csv
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.colors as mcolors
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parents[2]
DATA_DIR    = ROOT / "data"
NPZ_PATH    = DATA_DIR / "models" / "bodmas.npz"
CAT_PATH    = DATA_DIR / "models" / "bodmas_malware_category.csv"
META_PATH   = DATA_DIR / "models" / "bodmas_metadata.csv"
BOUNDS_PATH = DATA_DIR / "spiral_bounds.npz"
OUT_DIR     = DATA_DIR / "images"

# ---------------------------------------------------------------------------
# Color LUT: byte 0 → yellow (60°), byte 255 → purple (270°)
# ---------------------------------------------------------------------------
def build_color_lut() -> np.ndarray:
    lut = np.zeros((256, 3), dtype=np.uint8)
    for val in range(256):
        hue = (60.0 - (val / 255.0) * 150.0) / 360.0 % 1.0
        r, g, b = mcolors.hsv_to_rgb([hue, 1.0, 1.0])
        lut[val] = (int(r * 255), int(g * 255), int(b * 255))
    return lut


COLOR_LUT = build_color_lut()

# ---------------------------------------------------------------------------
# Global normalization
# ---------------------------------------------------------------------------
def compute_global_bounds(X: np.ndarray, percentiles=(2, 98)):
    """Per-feature p2/p98 bounds on the log-transformed matrix."""
    print("Computing global per-feature bounds (one-time, may take ~30s)...")
    transformed = np.sign(X) * np.log1p(np.abs(X))
    p_low  = np.percentile(transformed, percentiles[0], axis=0)
    p_high = np.percentile(transformed, percentiles[1], axis=0)
    return p_low, p_high


def load_or_compute_bounds(X: np.ndarray):
    if BOUNDS_PATH.exists():
        print("Loading precomputed bounds...")
        bounds = np.load(BOUNDS_PATH)
        return bounds["p_low"], bounds["p_high"]
    p_low, p_high = compute_global_bounds(X)
    np.savez(BOUNDS_PATH, p_low=p_low, p_high=p_high)
    print(f"Bounds saved to {BOUNDS_PATH}")
    return p_low, p_high

# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------
def spiral_image_from_vector(
    vector: np.ndarray,
    p_low: np.ndarray,
    p_high: np.ndarray,
    size: int = 128,
    turns: int = 32,
) -> Image.Image:
    vector = np.asarray(vector, dtype=np.float32)

    transformed = np.sign(vector) * np.log1p(np.abs(vector))

    clipped = np.clip(transformed, p_low, p_high)
    denom = p_high - p_low
    byte_vals = np.where(
        denom > 1e-8,
        ((clipped - p_low) / denom * 255),
        0,
    ).astype(np.uint8)

    n = len(byte_vals)
    center = size // 2
    max_radius = center - 2
    step = max_radius / n

    i = np.arange(n)
    theta = 2 * np.pi * turns * i / n
    radius = step * i

    x = np.clip((center + radius * np.cos(theta)).astype(int), 0, size - 1)
    y = np.clip((center + radius * np.sin(theta)).astype(int), 0, size - 1)

    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[y, x] = COLOR_LUT[byte_vals]           # vectorized, last-write-wins

    img_pil = Image.fromarray(img, mode="RGB")
    img_pil = ImageEnhance.Brightness(img_pil).enhance(1.4)
    img_pil = ImageEnhance.Contrast(img_pil).enhance(1.6)
    img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=1))
    return img_pil

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # 1. Load features and labels
    print("Loading feature matrix...")
    npz = np.load(NPZ_PATH)
    X = npz["X"]                                # (134435, 2381)

    # metadata.csv is row-aligned with the npz — gives us sha256 for each sample
    print("Loading metadata...")
    with open(META_PATH, newline="", encoding="utf-8-sig") as f:
        meta_rows = list(csv.DictReader(f))
    assert len(meta_rows) == len(X), (
        f"Metadata row count {len(meta_rows)} != X samples {len(X)}"
    )
    sha256s = [r["sha"] for r in meta_rows]

    # category CSV only covers malware samples; everything else is benign
    print("Loading category labels...")
    with open(CAT_PATH, newline="", encoding="utf-8-sig") as f:
        cat_lookup = {r["sha256"]: r["category"] for r in csv.DictReader(f)}

    categories = [cat_lookup.get(sha, "benign") for sha in sha256s]
    indices    = list(range(len(X)))

    cat_counts = {}
    for c in categories:
        cat_counts[c] = cat_counts.get(c, 0) + 1
    print("Category distribution:", cat_counts)

    # Classes need enough samples for the smallest 15% slice to have >=1 sample.
    # ceil(1/0.15)=7, but the two-stage split compounds rounding; use 20 for safety.
    MIN_SAMPLES = 20
    categories = [c if cat_counts[c] >= MIN_SAMPLES else "other" for c in categories]
    cat_counts2 = {}
    for c in categories:
        cat_counts2[c] = cat_counts2.get(c, 0) + 1
    print("After merging rare classes:", cat_counts2)

    # 2. Stratified 70 / 15 / 15 split
    print("Splitting dataset (70/15/15 stratified)...")
    idx_train, idx_tmp, _, cat_tmp = train_test_split(
        indices, categories, test_size=0.30, random_state=42, stratify=categories
    )
    idx_val, idx_test, _, _ = train_test_split(
        idx_tmp, cat_tmp, test_size=0.50, random_state=42, stratify=cat_tmp
    )

    split_map = (
        [("train", i) for i in idx_train]
        + [("val",   i) for i in idx_val]
        + [("test",  i) for i in idx_test]
    )

    # 3. Normalization bounds
    p_low, p_high = load_or_compute_bounds(X)

    # 4. Create output directories
    splits = ("train", "val", "test")
    unique_cats = sorted(set(categories))
    for split in splits:
        for cat in unique_cats:
            (OUT_DIR / split / cat).mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating {len(split_map)} images into {OUT_DIR} ...")
    skipped = 0

    for split, idx in tqdm(split_map, unit="img"):
        cat   = categories[idx]
        sha   = sha256s[idx]
        dest  = OUT_DIR / split / cat / f"{sha}.png"

        if dest.exists():
            skipped += 1
            continue

        img = spiral_image_from_vector(X[idx], p_low, p_high)
        img.save(dest)

    print(f"\nDone. Skipped {skipped} already-existing images.")

    # 5. Summary
    print("\nSplit / category counts:")
    for split in splits:
        counts = {cat: len(list((OUT_DIR / split / cat).glob("*.png")))
                  for cat in unique_cats}
        total = sum(counts.values())
        print(f"  {split:5s}  total={total:6d}  " +
              "  ".join(f"{c}={n}" for c, n in counts.items()))


if __name__ == "__main__":
    main()
