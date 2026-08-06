"""
Phase 2: Multi-label defect classification.

Real wafers can show overlapping defect signatures (e.g. edge-ring AND
scratch on the same wafer). Treating this as single-label throws away
information and can't represent compound failure modes. We reframe as
multi-label (sigmoid output per class + BCE-based loss) and use focal
loss to keep rare classes from being swamped by "None".
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from data_loader import load_wafer_data
from synthetic_data import CLASSES  # defect classes excluding "None" for multi-label targets

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFECT_CLASSES = [c for c in CLASSES if c != "None"]  # 8 classes; "None" = all-zero vector


class MultiLabelWaferDataset(Dataset):
    def __init__(self, X, y_multi_hot):
        self.X = X.astype(np.float32) / 2.0
        self.y = y_multi_hot.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return torch.tensor(self.X[i]).unsqueeze(0), torch.tensor(self.y[i])


class MultiLabelCNN(nn.Module):
    def __init__(self, n_labels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 4 * 4, 128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, n_labels),  # raw logits, sigmoid applied in loss/eval
        )

    def forward(self, x):
        return self.head(self.net(x))


class FocalLoss(nn.Module):
    """Focal loss for multi-label: down-weights easy (well-classified) examples
    so rare defect classes contribute more to the gradient than the dominant
    'no defect' signal. gamma controls how aggressively easy examples are down-weighted."""
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)  # p_t = model's estimated prob of the TRUE class
        focal_term = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * focal_term * bce
        return loss.mean()


def labels_to_multihot(y):
    idx = {c: i for i, c in enumerate(DEFECT_CLASSES)}
    multi_hot = np.zeros((len(y), len(DEFECT_CLASSES)), dtype=np.float32)
    for i, labels in enumerate(y):
        for l in labels:
            if l in idx:
                multi_hot[i, idx[l]] = 1.0
    return multi_hot


def train_multilabel(epochs=10, batch_size=64, lr=1e-3, threshold=0.5):
    X, y, source = load_wafer_data(n_samples=40000)
    print(f"Data source: {source}")
    Y = labels_to_multihot(y)

    # stratify on "has any defect" as a proxy since sklearn can't stratify multi-label directly
    has_defect = (Y.sum(axis=1) > 0).astype(int)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, stratify=has_defect, random_state=42
    )
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train, Y_train, test_size=0.15,
        stratify=(Y_train.sum(axis=1) > 0).astype(int), random_state=42
    )

    train_dl = DataLoader(MultiLabelWaferDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(MultiLabelWaferDataset(X_val, Y_val), batch_size=batch_size)
    test_dl = DataLoader(MultiLabelWaferDataset(X_test, Y_test), batch_size=batch_size)

    model = MultiLabelCNN(n_labels=len(DEFECT_CLASSES)).to(DEVICE)
    criterion = FocalLoss(gamma=2.0, alpha=0.25)
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
                out = torch.sigmoid(model(xb.to(DEVICE)))
                val_preds.append((out.cpu().numpy() > threshold).astype(int))
                val_true.append(yb.numpy())
        val_preds = np.concatenate(val_preds)
        val_true = np.concatenate(val_true)
        val_f1 = f1_score(val_true, val_preds, average="macro", zero_division=0)
        print(f"Epoch {epoch+1}/{epochs} | train_loss={total_loss/len(train_dl.dataset):.4f} | val_macro_f1={val_f1:.4f}")

    # test eval + per-class breakdown
    model.eval()
    test_preds, test_true = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            out = torch.sigmoid(model(xb.to(DEVICE)))
            test_preds.append((out.cpu().numpy() > threshold).astype(int))
            test_true.append(yb.numpy())
    test_preds = np.concatenate(test_preds)
    test_true = np.concatenate(test_true)

    print("\n=== Per-class multi-label F1 (test set) ===")
    for i, cls in enumerate(DEFECT_CLASSES):
        f1 = f1_score(test_true[:, i], test_preds[:, i], zero_division=0)
        support = int(test_true[:, i].sum())
        print(f"  {cls:12s} F1={f1:.3f}  (support={support})")

    macro_f1 = f1_score(test_true, test_preds, average="macro", zero_division=0)
    print(f"\nHeadline metric - Test macro F1 (multi-label): {macro_f1:.4f}")

    # sanity check: how many test wafers actually had >1 label (validates multi-label framing matters)
    n_multi = (test_true.sum(axis=1) > 1).sum()
    print(f"Wafers with 2+ overlapping defect labels in test set: {n_multi}/{len(test_true)}")

    torch.save(model.state_dict(), "multilabel_cnn.pt")
    return model, macro_f1


if __name__ == "__main__":
    train_multilabel()
