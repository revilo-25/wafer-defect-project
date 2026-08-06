
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt

from data_loader import load_wafer_data, CLASSES

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


class WaferDataset(Dataset):
    def __init__(self, X, y_idx):
        self.X = X.astype(np.float32) / 2.0  # normalize 0/1/2 -> 0/0.5/1
        self.y = y_idx

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return torch.tensor(self.X[i]).unsqueeze(0), torch.tensor(self.y[i])


class BaselineCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 4 * 4, 128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.head(self.net(x))


def prepare_single_label(X, y):
    """Collapse multi-label to primary label for the baseline (Phase 2 handles multi-label)."""
    primary = [labels[0] if labels else "None" for labels in y]
    class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    y_idx = np.array([class_to_idx[c] for c in primary])
    return y_idx, class_to_idx


def plot_results(test_true, test_preds, present_labels, target_names, source="real"):
    # --- Confusion matrix ---
    cm = confusion_matrix(test_true, test_preds, labels=present_labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(target_names)))
    ax.set_yticks(range(len(target_names)))
    ax.set_xticklabels(target_names, rotation=45, ha="right")
    ax.set_yticklabels(target_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix - Baseline CNN (row-normalized)")
    for i in range(len(target_names)):
        for j in range(len(target_names)):
            val = cm_norm[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(im, ax=ax, label="Fraction of true class")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

    # --- Per-class F1 bar chart ---
    f1s = f1_score(test_true, test_preds, labels=present_labels, average=None, zero_division=0)
    order = np.argsort(f1s)[::-1]
    sorted_names = [target_names[i] for i in order]
    sorted_f1s = [f1s[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#4C72B0" if f >= 0.5 else "#C44E52" for f in sorted_f1s]
    bars = ax.bar(sorted_names, sorted_f1s, color=colors)
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1)
    ax.set_title(f"Per-Class F1 - Baseline CNN ({'Real WM-811K' if source == 'real' else 'Synthetic'} Data)")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    plt.xticks(rotation=45, ha="right")
    for bar, f in zip(bars, sorted_f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, f + 0.02, f"{f:.2f}", ha="center", fontsize=8)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "per_class_f1.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def train_baseline(epochs=8, batch_size=64, lr=1e-3):
    X, y, source = load_wafer_data(n_samples=40000)
    y_idx, class_to_idx = prepare_single_label(X, y)
    print(f"Data source: {source}")

   
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_idx, test_size=0.2, stratify=y_idx, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
    )

    train_ds = WaferDataset(X_train, y_train)
    val_ds = WaferDataset(X_val, y_val)
    test_ds = WaferDataset(X_test, y_test)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)
    test_dl = DataLoader(test_ds, batch_size=batch_size)

    model = BaselineCNN(n_classes=len(CLASSES)).to(DEVICE)
    # class-weighted loss since "None" dominates ~60% of samples
    class_counts = np.bincount(y_train, minlength=len(CLASSES))
    weights = torch.tensor(1.0 / np.maximum(class_counts, 1), dtype=torch.float32)
    weights = (weights / weights.sum() * len(CLASSES)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                out = model(xb.to(DEVICE))
                val_preds.extend(out.argmax(1).cpu().numpy())
                val_true.extend(yb.numpy())
        val_f1 = f1_score(val_true, val_preds, average="macro", zero_division=0)
        print(f"Epoch {epoch+1}/{epochs} | train_loss={total_loss/len(train_ds):.4f} | val_macro_f1={val_f1:.4f}")

    # final test evaluation
    model.eval()
    test_preds, test_true = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            out = model(xb.to(DEVICE))
            test_preds.extend(out.argmax(1).cpu().numpy())
            test_true.extend(yb.numpy())

    present_labels = sorted(set(test_true) | set(test_preds))
    target_names = [CLASSES[i] for i in present_labels]
    print("\n=== Test set report ===")
    print(classification_report(test_true, test_preds, labels=present_labels,
                                 target_names=target_names, zero_division=0))
    macro_f1 = f1_score(test_true, test_preds, average="macro", zero_division=0)
    print(f"Headline metric - Test macro F1: {macro_f1:.4f}")

    plot_results(test_true, test_preds, present_labels, target_names, source=source)

    torch.save(model.state_dict(), "baseline_cnn.pt")
    return model, macro_f1


if __name__ == "__main__":
    train_baseline()
