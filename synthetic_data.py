"""
Generates synthetic wafer maps that mimic the structure and class imbalance
of the real WM-811K dataset. Use this to build/test the pipeline locally.
Swap in real WM-811K.pkl later (see data_loader.py) with zero code changes
downstream, since the output shape/format is identical.

Real dataset: https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map
(download manually, place as data/WM811K.pkl)
"""
import numpy as np

WAFER_SIZE = 52  # WM-811K maps are variable size; we standardize to 52x52
CLASSES = ["None", "Center", "Donut", "Edge-Loc", "Edge-Ring",
           "Local", "Random", "Scratch", "Near-full"]

# Real WM-811K class frequencies are wildly imbalanced (~85% "None").
# We mirror that ratio here so the imbalance-handling code is actually tested.
CLASS_WEIGHTS = {
    "None": 0.60, "Edge-Ring": 0.12, "Edge-Loc": 0.08, "Center": 0.06,
    "Scratch": 0.05, "Random": 0.04, "Local": 0.03, "Donut": 0.015,
    "Near-full": 0.005,
}


def _circular_mask(size):
    yy, xx = np.mgrid[0:size, 0:size]
    center = size / 2
    r = size / 2 - 1
    return (xx - center) ** 2 + (yy - center) ** 2 <= r ** 2


def _make_wafer(defect_class, size=WAFER_SIZE, multi_label_prob=0.12, rng=None):
    """0=no die/outside wafer, 1=pass die, 2=fail die."""
    rng = rng or np.random.default_rng()
    mask = _circular_mask(size)
    wafer = np.zeros((size, size), dtype=np.uint8)
    wafer[mask] = 1  # pass by default

    yy, xx = np.mgrid[0:size, 0:size]
    center = size / 2

    def apply_pattern(cls):
        if cls == "Center":
            r = rng.uniform(6, 10)
            fail = ((xx - center) ** 2 + (yy - center) ** 2) <= r ** 2
        elif cls == "Donut":
            r_in, r_out = rng.uniform(8, 10), rng.uniform(14, 18)
            d2 = (xx - center) ** 2 + (yy - center) ** 2
            fail = (d2 >= r_in ** 2) & (d2 <= r_out ** 2)
        elif cls == "Edge-Ring":
            r_out = size / 2 - 1
            d = np.sqrt((xx - center) ** 2 + (yy - center) ** 2)
            fail = d >= (r_out - rng.uniform(2, 4))
        elif cls == "Edge-Loc":
            r_out = size / 2 - 1
            d = np.sqrt((xx - center) ** 2 + (yy - center) ** 2)
            angle0 = rng.uniform(0, 2 * np.pi)
            angle = np.arctan2(yy - center, xx - center)
            arc = np.abs(((angle - angle0 + np.pi) % (2 * np.pi)) - np.pi) < 0.6
            fail = (d >= (r_out - 6)) & arc
        elif cls == "Scratch":
            fail = np.zeros_like(mask)
            x0, y0 = rng.uniform(5, size - 5, 2)
            angle = rng.uniform(0, np.pi)
            length = rng.uniform(15, 30)
            for t in np.linspace(0, length, 60):
                x = int(x0 + t * np.cos(angle))
                y = int(y0 + t * np.sin(angle))
                if 0 <= x < size and 0 <= y < size:
                    fail[max(0, y - 1):y + 2, max(0, x - 1):x + 2] = True
        elif cls == "Local":
            fail = np.zeros_like(mask)
            for _ in range(rng.integers(1, 3)):
                cx, cy = rng.uniform(8, size - 8, 2)
                r = rng.uniform(3, 6)
                fail |= ((xx - cx) ** 2 + (yy - cy) ** 2) <= r ** 2
        elif cls == "Random":
            fail = rng.random((size, size)) < 0.08
        elif cls == "Near-full":
            fail = rng.random((size, size)) < 0.85
        else:  # None
            fail = np.zeros_like(mask)
        return fail

    fail_mask = apply_pattern(defect_class)

    # multi-label overlap: sometimes stack a second pattern (real fabs see this)
    second_label = None
    if defect_class != "None" and rng.random() < multi_label_prob:
        candidates = [c for c in CLASSES if c not in ("None", defect_class)]
        second_label = rng.choice(candidates)
        fail_mask = fail_mask | apply_pattern(second_label)

    wafer[mask & fail_mask] = 2
    labels = [defect_class] if defect_class != "None" else []
    if second_label:
        labels.append(second_label)
    return wafer, labels


def generate_dataset(n_samples=6000, seed=42):
    rng = np.random.default_rng(seed)
    classes = list(CLASS_WEIGHTS.keys())
    weights = np.array(list(CLASS_WEIGHTS.values()))
    weights = weights / weights.sum()

    sampled = rng.choice(classes, size=n_samples, p=weights)
    wafers, label_lists = [], []
    for cls in sampled:
        w, labels = _make_wafer(cls, rng=rng)
        wafers.append(w)
        label_lists.append(labels)

    X = np.stack(wafers)
    return X, label_lists


if __name__ == "__main__":
    X, y = generate_dataset(2000)
    print("Shape:", X.shape)
    from collections import Counter
    flat = [l for labels in y for l in labels] or ["None"]
    print(Counter(flat))
