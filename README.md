# Semiconductor Wafer Defect Analysis: Classification, Novelty Detection & Causal Root-Cause Analysis

A 4-phase applied data science project targeting real problems in semiconductor
manufacturing: defect pattern classification, detecting *unknown* defect types,
and correctly separating correlation from causation in fab root-cause analysis.

Built to demonstrate skills directly relevant to yield engineering / manufacturing
data science roles (Intel, TSMC, Micron, Applied Materials, KLA, Lam Research, TI).

## Why this project

Most "wafer map classification" portfolios stop at training a CNN and reporting
accuracy. That undersells the actual job. Three things fab data scientists deal
with daily that a plain classifier ignores:

1. **Defects overlap.** A wafer can show an edge-ring pattern *and* a scratch. Single-label classification can't represent that.
2. **Labeled taxonomies are never complete.** New tools and recipes create new failure signatures nobody has labeled yet.
3. **Correlation misleads constantly.** Maintenance schedules and shift changes confound sensor readings with yield outcomes, and naive analysis blames the wrong variable.

This project addresses all three, each as an independent, runnable phase.

## Dataset

Built and tested against a **synthetic wafer map generator** (`src/synthetic_data.py`)
that mirrors the real **WM-811K** dataset's structure, defect classes, and class
imbalance (~811K wafer maps, 9 defect classes, real dataset on
[Kaggle](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)).

**To run on real data:** download `WM-811K.pkl` (often named `LSWMD.pkl`) from
Kaggle, place it at `data/WM811K.pkl`. `src/data_loader.py` auto-detects it and
handles two common legacy-pickle issues out of the box (old pandas internal
module paths, and Python 2 pickle encoding). Results below are from the real
dataset, confirmed working end-to-end.

## Project structure

```
wafer-defect-project/
├── data/                          # place WM811K.pkl here for real data
├── figures/                       # generated EDA plots
├── src/
│   ├── synthetic_data.py          # synthetic wafer map generator (WM-811K format)
│   ├── data_loader.py             # unified loader: real data if present, else synthetic
│   ├── eda.py                     # Phase 1a: EDA - class distribution, sample wafers
│   ├── phase1_baseline_cnn.py     # Phase 1b: baseline single-label CNN classifier
│   ├── phase2_multilabel.py       # Phase 2: multi-label CNN + focal loss
│   ├── phase3_novelty_detection.py# Phase 3: autoencoder for unlabeled defect detection
│   └── phase4_causal_confounding.py # Phase 4: correlation vs causation demo
└── README.md
```

## Figures

![Class distribution](figures/class_distribution.png)
![Confusion matrix](figures/confusion_matrix.png)
![Per-class F1](figures/per_class_f1.png)
![Sample wafer maps](figures/sample_wafers.png)

All four figures below are from the real WM-811K dataset.

## Results (real WM-811K dataset, 40K stratified sample)

Ran on the actual WM-811K dataset (not the synthetic placeholder) using a
stratified sample that keeps all 25,519 labeled-defect wafers plus 14,481
"None" wafers, for a representative but computationally tractable subset of
the full 811K wafers.

| Phase | Headline metric |
|---|---|
| 1. Baseline CNN (single-label) | macro F1 = **0.74** |
| 2. Multi-label CNN + focal loss | macro F1 = **0.68** |
| 3. Autoencoder novelty detection | ROC-AUC = **0.78** |
| 4. Causal confounding analysis | Correctly isolates true root cause from a confounded decoy variable |

### Phase 1 — per-class breakdown
Edge-Ring (F1=0.96), Center (0.88), and Donut (0.78) — classes with
distinctive, large-scale geometric shapes — are classified reliably. Scratch
(F1=0.27) is the hardest class: thin, small, low-contrast lines are difficult
for a plain CNN to pick up at 52x52 resolution, and this is consistent across
every experiment in this project.

### Phase 2 — a genuine finding, not just a metric
Multi-label Scratch F1 dropped to **exactly 0.000** (vs. 0.27 in the
single-label baseline) — the model's sigmoid output for Scratch never crossed
the 0.5 decision threshold on any test wafer. This is a specific, diagnosable
failure mode: with focal loss's `alpha=0.25` weighting, a class that's already
visually ambiguous can get pushed to always-negative predictions rather than
partially-correct ones. Two follow-ups worth trying:
- Inspect raw sigmoid outputs for Scratch wafers directly — likely clustered just under 0.5, not confidently near 0
- Tune `alpha` upward for hard/rare classes, or use per-class thresholds instead of a global 0.5 cutoff

### Phase 3 — a real tradeoff worth naming
Class-weighted loss (used to boost rare-class recall in Phase 1) has a visible
cost: "None" wafer *recall* only reaches 0.79-0.94 depending on the run, even
though "None" precision stays high (~0.88-0.98). The model is trading some
false alarms on clean wafers for better detection of rare defects — a
legitimate, tunable production tradeoff depending on whether false alarms or
missed defects cost more in a given fab.

## Previous results (synthetic data, for pipeline validation only)

Before running on the real dataset, all four phases were validated end-to-end
on the synthetic generator to confirm the pipeline itself (data loading,
training loops, metrics, and the causal demo) worked correctly:
- Phase 1: macro F1 = 0.78
- Phase 2: macro F1 = 0.76, ROC-AUC unaffected (Phase 3 unsupervised)
- Phase 3: ROC-AUC = 0.999 (separable-by-design synthetic patterns)
- Phase 4: unaffected by data source (self-contained synthetic causal demo)

The gap between synthetic and real-data results (e.g. Phase 3 AUC 0.999 vs.
0.78) is itself informative: real sensor/defect data is messier and less
cleanly separable than hand-designed synthetic patterns, which is exactly why
validating on synthetic data first (to catch pipeline bugs) and then
re-running on real data (to get honest numbers) is good practice.

## Phase 4 — Correlation vs. causation in root-cause analysis

The differentiator most portfolios skip entirely. Builds a synthetic fab dataset
with a **known ground-truth causal structure**: tool maintenance status
confounds a decoy sensor (chamber temperature) and yield loss, while the *true*
cause (etch time deviation) is independent of maintenance.

- **Naive correlation** flags chamber_temp as the driver (r=0.60, p<0.001) — wrong.
- **Stratifying by maintenance status** collapses that correlation to ~0, while the true cause's relationship holds.
- **Multivariate regression including the confounder** recovers the correct attribution (chamber_temp coefficient becomes insignificant; etch_time-deviation stays significant).

This demonstrates the reasoning fab engineers actually need: not just "what
correlates with yield loss" but "what would we still see if we intervened."

## Running it

```bash
cd src
pip install torch scikit-learn matplotlib scipy statsmodels
python eda.py                          # generates figures/
python phase1_baseline_cnn.py
python phase2_multilabel.py
python phase3_novelty_detection.py
python phase4_causal_confounding.py
```

## What I'd extend next

- Real lot-ID metadata for proper GroupShuffleSplit
- Higher-resolution wafer maps + attention-based architecture for thin defects (Scratch)
- Formal causal graph (DAG) with `dowhy` for the Phase 4 analysis at scale
- Virtual metrology: predict a continuous process output (e.g. film thickness) from sensor streams instead of only classifying discrete defect types
