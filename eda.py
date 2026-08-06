
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from data_loader import load_wafer_data, CLASSES

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def plot_class_distribution(y, save_path=None):
    save_path = save_path or os.path.join(FIG_DIR, "class_distribution.png")
    flat = [l for labels in y for l in labels]
    flat = flat if flat else ["None"]
    counts = Counter(flat)
    # include "None" explicitly (wafers with empty label list)
    n_none = sum(1 for labels in y if len(labels) == 0)
    counts["None"] = n_none

    classes = [c for c in CLASSES if c in counts]
    values = [counts[c] for c in classes]

    plt.figure(figsize=(9, 5))
    bars = plt.bar(classes, values, color="#4C72B0")
    plt.yscale("log")
    plt.ylabel("Count (log scale)")
    plt.title("Wafer Defect Class Distribution (note log scale + imbalance)")
    plt.xticks(rotation=45, ha="right")
    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, v, str(v), ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    print("Class counts:", dict(counts))


def plot_sample_wafers(X, y, save_path=None):
    """Show one example wafer map per class."""
    save_path = save_path or os.path.join(FIG_DIR, "sample_wafers.png")
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    shown = set()
    idx = 0
    for i, labels in enumerate(y):
        key = labels[0] if labels else "None"
        if key in shown:
            continue
        shown.add(key)
        axes[idx].imshow(X[i], cmap="viridis", vmin=0, vmax=2)
        axes[idx].set_title(key)
        axes[idx].axis("off")
        idx += 1
        if idx >= len(axes):
            break
    for j in range(idx, len(axes)):
        axes[j].axis("off")
    plt.suptitle("Sample Wafer Maps by Defect Class (0=no die, 1=pass, 2=fail)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    X, y, source = load_wafer_data(n_samples=40000)
    print(f"\nData source: {source} | X shape: {X.shape}\n")
    plot_class_distribution(y)
    plot_sample_wafers(X, y)
