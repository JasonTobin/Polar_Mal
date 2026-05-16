"""
generate_dataset_greyscale.py

Identical to generate_dataset.py except the spiral images use a greyscale
palette instead of the HSV colour LUT.  Byte value 0 → black, 255 → white.

Output layout (kept separate from the colour dataset):

    data/images_greyscale/<split>/<category>/<sha256>.png

Normalization bounds are shared with the colour dataset (spiral_bounds.npz)
so both datasets are directly comparable — only the colouring differs.
"""

import csv
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
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
BOUNDS_PATH = DATA_DIR / "spiral_bounds.npz"   # shared with colour dataset
OUT_DIR     = DATA_DIR / "images_greyscale"    # separate output tree

# ---------------------------------------------------------------------------
# Greyscale LUT: byte 0 → black (0,0,0), byte 255 → white (255,255,255)
# Stored as RGB so the image pipeline (ImageFolder, transforms) is identical
# to the colour dataset — no code changes needed in training scripts.
# ---------------------------------------------------------------------------
GREY_LUT = np.stack([np.arange(256, dtype=np.uint8)] * 3, axis=1)  # (256, 3)

# ---------------------------------------------------------------------------
# Global normalization (shared bounds)
# ---------------------------------------------------------------------------
def compute_global_bounds(X: np.ndarray, percentiles=(2, 98)):
    print("Computing global per-feature bounds (one-time, may take ~30s)...")
    transformed = np.sign(X) * np.log1p(np.abs(X))
    p_low  = np.percentile(transformed, percentiles[0], axis=0)
    p_high = np.percentile(transformed, percentiles[1], axis=0)
    return p_low, p_high


def load_or_compute_bounds(X: np.ndarray):
    if BOUNDS_PATH.exists():
        print("Loading precomputed bounds (shared with colour dataset)...")
        bounds = np.load(BOUNDS_PATH)
        return bounds["p_low"], bounds["p_high"]
    p_low, p_high = compute_global_bounds(X)
    np.savez(BOUNDS_PATH, p_low=p_low, p_high=p_high)
    print(f"Bounds saved to {BOUNDS_PATH}")
    return p_low, p_high

# ---------------------------------------------------------------------------
# Image generation (greyscale)
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

    # Greyscale: R == G == B == byte value, background stays black
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[y, x] = GREY_LUT[byte_vals]

    img_pil = Image.fromarray(img, mode="RGB")
    img_pil = ImageEnhance.Brightness(img_pil).enhance(1.4)
    img_pil = ImageEnhance.Contrast(img_pil).enhance(1.6)
    img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=1))
    return img_pil

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading feature matrix...")
    npz = np.load(NPZ_PATH)
    X = npz["X"]

    print("Loading metadata...")
    with open(META_PATH, newline="", encoding="utf-8-sig") as f:
        meta_rows = list(csv.DictReader(f))
    assert len(meta_rows) == len(X), (
        f"Metadata row count {len(meta_rows)} != X samples {len(X)}"
    )
    sha256s = [r["sha"] for r in meta_rows]

    print("Loading category labels...")
    with open(CAT_PATH, newline="", encoding="utf-8-sig") as f:
        cat_lookup = {r["sha256"]: r["category"] for r in csv.DictReader(f)}

    categories = [cat_lookup.get(sha, "benign") for sha in sha256s]
    indices    = list(range(len(X)))

    cat_counts = {}
    for c in categories:
        cat_counts[c] = cat_counts.get(c, 0) + 1
    print("Category distribution:", cat_counts)

    MIN_SAMPLES = 20
    categories = [c if cat_counts[c] >= MIN_SAMPLES else "other" for c in categories]
    cat_counts2 = {}
    for c in categories:
        cat_counts2[c] = cat_counts2.get(c, 0) + 1
    print("After merging rare classes:", cat_counts2)

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

    p_low, p_high = load_or_compute_bounds(X)

    splits = ("train", "val", "test")
    unique_cats = sorted(set(categories))
    for split in splits:
        for cat in unique_cats:
            (OUT_DIR / split / cat).mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating {len(split_map)} greyscale images into {OUT_DIR} ...")
    skipped = 0

    for split, idx in tqdm(split_map, unit="img"):
        cat  = categories[idx]
        sha  = sha256s[idx]
        dest = OUT_DIR / split / cat / f"{sha}.png"

        if dest.exists():
            skipped += 1
            continue

        img = spiral_image_from_vector(X[idx], p_low, p_high)
        img.save(dest)

    print(f"\nDone. Skipped {skipped} already-existing images.")

    print("\nSplit / category counts:")
    for split in splits:
        counts = {cat: len(list((OUT_DIR / split / cat).glob("*.png")))
                  for cat in unique_cats}
        total = sum(counts.values())
        print(f"  {split:5s}  total={total:6d}  " +
              "  ".join(f"{c}={counts[c]}" for c in unique_cats))


if __name__ == "__main__":
    main()
