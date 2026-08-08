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

## RACER extension status

- **RACER-C v1.0:** prospectively frozen and executed on 60 primary cells. The
  run completed, but its predicted-class policy gate was infeasible in all 60
  cells. Its code, tag, and raw results remain immutable.
- **RACER-C2:** a separate development-only method line under
  `scripts/racer_c2/`. It replaces predicted-class routing with
  candidate-label exponential reliability tilting, retains an exact
  stacking-Mondrian fallback, and certifies final conformal set states directly.
  The first learned counterfactual-score draft was rejected after honest
  development selected zero weight. Its additive retrospective runner reuses
  the completed 60-cell v1 probability artifacts and adds the fixed C2 method,
  its exact fallback, and three ablations without retraining any base model or
  rerunning the 540 old method results. RACER-C2 is not frozen and has no
  authorized prospective test panel yet.

Run the additive CPU-only experiment from the RACER-C2 branch on Windows:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  paper2_admet_benchmark\scripts\racer_c2\run_racer_c2_increment.ps1
```

The command reproduces the development-only selection and then writes the new
results under `results/racer_c2_retrospective_extension_v0/`. The v1 source
directory is hash-verified before and after evaluation and remains read-only.

The known v1 Tox21 outcomes may be used only for v2 architecture development and
failure analysis. A new confirmatory claim requires untouched endpoints and a new
user-approved protocol tag.
