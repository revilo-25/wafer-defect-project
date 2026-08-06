"""
Unified loader. If data/WM811K.pkl exists (real Kaggle dataset), loads it.
Otherwise falls back to the synthetic generator so the pipeline is always runnable.

To use the REAL dataset:
1. Download from https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map
2. Place the .pkl file at data/WM811K.pkl
3. Re-run - this loader auto-detects it, no other code changes needed.
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from synthetic_data import generate_dataset, CLASSES, WAFER_SIZE

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "WM811K.pkl")


def _patch_legacy_pandas_modules():
    """The WM-811K pkl was created with a very old pandas version, whose
    internal module layout (e.g. pandas.indexes -> pandas.core.indexes) has
    since changed. This aliases the old paths to the CURRENT submodule that
    actually contains each class (RangeIndex, Index, etc.), so pickle.load
    can resolve them without needing to install an ancient pandas version.
    Each alias is attempted independently since exact submodule layout
    varies across pandas versions (some don't have e.g. indexes.numeric)."""
    alias_map = {
        "pandas.indexes.base": "pandas.core.indexes.base",
        "pandas.indexes.range": "pandas.core.indexes.range",
        "pandas.indexes.numeric": "pandas.core.indexes.base",  # fallback target
        "pandas.indexes": "pandas.core.indexes.base",
        "pandas.core.index": "pandas.core.indexes.base",
    }
    for old_name, current_name in alias_map.items():
        try:
            module = __import__(current_name, fromlist=["_"])
            sys.modules[old_name] = module
        except ImportError:
            pass  # this submodule doesn't exist in the installed pandas version - skip

    # If a NEW class lookup still fails (different AttributeError, different
    # class name), the fix is the same pattern: find which current pandas
    # submodule defines that class, then add an entry to alias_map above:
    #   "pandas.indexes.<name>": "pandas.core.indexes.<wherever it lives now>"
    # Send Claude the exact error and it'll add the right line.


def _resize_to_standard(wafer, size=WAFER_SIZE):
    from scipy.ndimage import zoom
    factors = (size / wafer.shape[0], size / wafer.shape[1])
    resized = zoom(wafer, factors, order=0)  # nearest-neighbor, preserves 0/1/2 labels
    return np.clip(resized, 0, 2).astype(np.uint8)


def load_wafer_data(n_samples=6000, seed=42, use_real_if_available=True):
    """
    Returns:
        X: (N, H, W) uint8 array, values {0=no die, 1=pass, 2=fail}
        y: list of label-lists (multi-label), e.g. [['Edge-Ring'], [], ['Scratch','Local']]
        source: 'real' or 'synthetic'
    """
    if use_real_if_available and os.path.exists(DATA_PATH):
        print(f"Loading REAL WM-811K data from {DATA_PATH}")
        _patch_legacy_pandas_modules()
        try:
            with open(DATA_PATH, "rb") as f:
                df = pickle.load(f, encoding="latin1")
        except (ModuleNotFoundError, AttributeError) as e:
            msg = str(e)
            raise type(e)(
                f"\n\nHit another legacy-pandas compatibility issue: {msg}\n"
                f"This is expected with old WM-811K pickle exports - pandas' internal "
                f"module layout has changed since it was created, and sometimes more "
                f"than one class needs aliasing.\n"
                f"Fix: open data_loader.py, find _patch_legacy_pandas_modules(), and add "
                f"an alias for whichever class/module is named in the error above.\n"
                f"Send Claude the exact error text and it can add the right line."
            ) from e
        # WM-811K pkl is typically a pandas DataFrame with 'waferMap' and 'failureType' columns
        # real WM-811K pickles often use "Loc" instead of "Local" - normalize
        LABEL_ALIASES = {"Loc": "Local", "loc": "Local", "none": None, "": None}

        def _extract_label(ft):
            if isinstance(ft, np.ndarray):
                ft = ft.item() if ft.size == 1 else None
            return LABEL_ALIASES.get(ft, ft)

        n_total = len(df)
        if n_samples is not None and n_samples < n_total:
            # Naive random sampling is a bad idea here: defects are only ~3%
            # of the full dataset, so a small random sample barely contains
            # any rare classes (e.g. 5 Donut out of 6000). Instead: keep ALL
            # labeled-defect wafers, and subsample only the "None" (normal)
            # wafers down to fill the rest of the budget. This gives every
            # phase enough rare-class examples to actually learn from.
            df["_label"] = df["failureType"].apply(_extract_label)
            defect_df = df[df["_label"].notna()]
            none_df = df[df["_label"].isna()]

            n_none_target = max(n_samples - len(defect_df), 0)
            if n_none_target < len(none_df):
                none_df = none_df.sample(n=n_none_target, random_state=seed)

            df = pd.concat([defect_df, none_df]).sample(frac=1, random_state=seed)  # shuffle
            print(f"Stratified sample: kept all {len(defect_df)} labeled-defect wafers + "
                  f"{len(none_df)} 'None' wafers = {len(df)} total "
                  f"(pass n_samples=None to load_wafer_data to use all {n_total}).")

        wafers, labels = [], []
        for _, row in df.iterrows():
            wm = np.array(row["waferMap"])
            if wm.shape[0] < 5 or wm.shape[1] < 5:
                continue
            wafers.append(_resize_to_standard(wm))
            ft = _extract_label(row.get("failureType", None))
            labels.append([ft] if ft else [])
        X = np.stack(wafers)
        return X, labels, "real"
    else:
        print("Real dataset not found at data/WM811K.pkl - using SYNTHETIC data.")
        print("(This is fine for building/testing the pipeline. Swap in the real")
        print(" file later for your actual portfolio results.)")
        X, y = generate_dataset(n_samples=n_samples, seed=seed)
        return X, y, "synthetic"


if __name__ == "__main__":
    X, y, source = load_wafer_data()
    print(f"Source: {source}, X shape: {X.shape}, n_labels: {len(y)}")
