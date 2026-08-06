
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve

from data_loader import load_wafer_data

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ReconDataset(Dataset):
    def __init__(self, X):
        self.X = X.astype(np.float32) / 2.0

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = torch.tensor(self.X[i]).unsqueeze(0)
        return x, x  # autoencoder: input == target


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),   # 52->26
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),  # 26->13
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),  # 13->7
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=0), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        # crop/pad to match input size exactly (transpose convs can be off by a pixel)
        if out.shape[-2:] != x.shape[-2:]:
            out = nn.functional.interpolate(out, size=x.shape[-2:], mode="nearest")
        return out


def reconstruction_error(model, dl):
    model.eval()
    errors = []
    with torch.no_grad():
        for xb, _ in dl:
            xb = xb.to(DEVICE)
            recon = model(xb)
            err = ((recon - xb) ** 2).mean(dim=[1, 2, 3])
            errors.extend(err.cpu().numpy())
    return np.array(errors)


def train_autoencoder(epochs=15, batch_size=64, lr=1e-3):
    X, y, source = load_wafer_data(n_samples=40000)
    print(f"Data source: {source}")

    is_none = np.array([len(labels) == 0 for labels in y])
    X_normal = X[is_none]
    X_defect = X[~is_none]

    # train ONLY on normal wafers - the model should never see defects during training
    X_train, X_val = train_test_split(X_normal, test_size=0.15, random_state=42)

    train_dl = DataLoader(ReconDataset(X_train), batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(ReconDataset(X_val), batch_size=batch_size)

    model = ConvAutoencoder().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, target in train_dl:
            xb, target = xb.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            recon = model(xb)
            loss = criterion(recon, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        print(f"Epoch {epoch+1}/{epochs} | train_recon_loss={total_loss/len(train_dl.dataset):.5f}")

    # evaluate: does reconstruction error separate normal (val) from defective (held-out) wafers?
    normal_errors = reconstruction_error(model, val_dl)
    defect_dl = DataLoader(ReconDataset(X_defect), batch_size=batch_size)
    defect_errors = reconstruction_error(model, defect_dl)

    labels = np.concatenate([np.zeros(len(normal_errors)), np.ones(len(defect_errors))])
    scores = np.concatenate([normal_errors, defect_errors])
    auc = roc_auc_score(labels, scores)

    print(f"\nNormal wafer recon error: mean={normal_errors.mean():.5f}, std={normal_errors.std():.5f}")
    print(f"Defect wafer recon error: mean={defect_errors.mean():.5f}, std={defect_errors.std():.5f}")
    print(f"Headline metric - Novelty detection ROC-AUC: {auc:.4f}")

    # pick an operating threshold using precision-recall tradeoff on this held-out mix
    precisions, recalls, thresholds = precision_recall_curve(labels, scores)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1s[:-1])
    print(f"Suggested threshold (max F1 on val mix): {thresholds[best_idx]:.5f} "
          f"(precision={precisions[best_idx]:.3f}, recall={recalls[best_idx]:.3f})")

    torch.save(model.state_dict(), "autoencoder.pt")
    return model, auc


if __name__ == "__main__":
    train_autoencoder()
