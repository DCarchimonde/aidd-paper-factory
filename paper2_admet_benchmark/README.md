# Paper 2: Reliability-Oriented ADMET Prediction Benchmark

Working title:

**Beyond Accuracy in ADMET Prediction: Applicability-Domain Diagnostics and Conformal Calibration under Chemical Distribution Shift**

## Study purpose

This project studies **when ADMET prediction models are reliable**, not merely which model gives the highest benchmark score.

The central question is:

> When should ADMET model predictions be trusted under chemical distribution shift?

The study evaluates public ADMET and molecular property endpoints using lightweight, reproducible molecular machine-learning baselines. Reliability is assessed through calibration metrics, conformal empirical coverage, applicability-domain diagnostics, and selective prediction.

## Relationship to Paper 1

Paper 1 audited how train/test splitting protocols affect molecular property benchmark interpretation.

Paper 2 addresses a distinct practical question: whether ADMET predictions remain calibrated, reliable, and actionable under chemical distribution shift.

This study does **not** claim state-of-the-art ADMET prediction. It provides a reproducible reliability audit for AIDD decision support.

## Planned workflow

```text
Public ADMET endpoints
        ↓
SMILES cleaning, canonicalization, duplicate/conflict handling
        ↓
ECFP4 / Morgan fingerprint, radius 2, 2048 bits
        ↓
Random and scaffold train/calibration/test splits
        ↓
Baseline models:
LR/Ridge, Random Forest, XGBoost, MLP
        ↓
Performance:
ROC-AUC, PR-AUC, RMSE, MAE
        ↓
Calibration:
Brier score, ECE, reliability diagrams
        ↓
Conformal prediction:
empirical coverage, set size, interval width
        ↓
Applicability domain:
Tanimoto similarity, top-k similarity, unseen scaffold
        ↓
Domain-conditioned reliability:
high / medium / low chemical-domain bins
        ↓
Selective prediction:
risk-coverage curves and abstention analysis
```

## Repository layout

```text
paper2_admet_benchmark/
├── configs/        Experiment configuration files
├── data/           Raw, processed, and manifest files
├── scripts/        Reproducibility scripts
├── results/        Tables, metrics, calibration, conformal outputs, logs
├── figures/        Manuscript figures
├── manuscript/     Draft manuscript, cover letter, references
└── notes/          Story, journal notes, experiment logs, reviewer checklist
```

Raw datasets, processed datasets, trained models, and logs are intentionally kept local and ignored by Git.

## Protocol status

Protocol version: **v1.0 frozen for MVP execution**

MVP requirements:

- at least 6 ADMET or molecular property endpoints;
- random and scaffold validation settings;
- train/calibration/test separation;
- at least 5 random seeds where applicable;
- baseline models: LR/Ridge, Random Forest, XGBoost, MLP on ECFP;
- performance, calibration, conformal, applicability-domain, and selective-prediction analyses.
