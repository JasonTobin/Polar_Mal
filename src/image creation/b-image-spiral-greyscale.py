import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance


def build_greyscale_lut():
    """
    256-entry greyscale LUT: byte 0 → black (0), byte 255 → white (255).
    Returns an (256, 3) RGB array where R == G == B (pure grey shades).
    """
    lut = np.zeros((256, 3), dtype=np.uint8)
    for val in range(256):
        lut[val] = (val, val, val)
    return lut


GREY_LUT = build_greyscale_lut()


def compute_global_bounds(X, percentiles=(2, 98)):
    transformed = np.sign(X) * np.log1p(np.abs(X))
    p_low  = np.percentile(transformed, percentiles[0], axis=0)
    p_high = np.percentile(transformed, percentiles[1], axis=0)
    return p_low, p_high


def spiral_image_from_vector(vector, p_low, p_high, size=128, turns=32):
    vector = np.asarray(vector, dtype=np.float32)

    # Signed log transform
    transformed = np.sign(vector) * np.log1p(np.abs(vector))

    # Apply precomputed global per-feature bounds
    clipped = np.clip(transformed, p_low, p_high)
    denom = p_high - p_low
    byte_vals = np.where(
        denom > 1e-8,
        ((clipped - p_low) / denom * 255),
        0
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

    # Greyscale RGB image: R == G == B mapped from byte value
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[y, x] = GREY_LUT[byte_vals]

    img_pil = Image.fromarray(img, mode='RGB')
    img_pil = ImageEnhance.Brightness(img_pil).enhance(1.4)
    img_pil = ImageEnhance.Contrast(img_pil).enhance(1.6)
    return img_pil


# ===== USAGE =====
if __name__ == "__main__":
    data_path   = Path(__file__).resolve().parents[2] / "data" / "bodmas.npz"
    bounds_path = Path(__file__).resolve().parents[2] / "data" / "spiral_bounds.npz"
    data = np.load(data_path)

    print("Keys:", data.files)
    X = data["X"]

    if bounds_path.exists():
        print("Loading precomputed bounds...")
        bounds = np.load(bounds_path)
        p_low, p_high = bounds["p_low"], bounds["p_high"]
    else:
        print("Computing global per-feature bounds (one-time, may take a moment)...")
        p_low, p_high = compute_global_bounds(X)
        np.savez(bounds_path, p_low=p_low, p_high=p_high)
        print(f"Bounds saved to {bounds_path}")

    idx = np.random.randint(len(X))
    vector = X[idx]
    print(f"Using sample index: {idx}")

    img = spiral_image_from_vector(vector, p_low, p_high, size=128, turns=32)
    img.save("spiral_feature_greyscale.png")

    print("Saved greyscale spiral image.")
