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

## Post-v1 RACER-C3 development

The isolated branch `paper2-racer-c3-development-2026` contains a new
development-only algorithm candidate under `scripts/racer_c3/`. It has **not
passed its architecture freeze gate**. RACER-C3 uses
candidate-label-specific experts at a verified chemical frontier, an unlabeled
permutation-invariant batch route, class-conditional calibration, and an exact
fallback to the completed v1 no-gate RACER score.

The four-endpoint v1 panel was already known when this architecture was chosen.
Accordingly, `results/racer_c3_development/` is retrospective development
evidence and cannot support a superiority claim. The algorithm specification,
prior-art boundary, and prospective firewall are in:

- `docs/racer_c3_algorithm_specification_v0.1.md`;
- `docs/racer_c3_prior_art_boundary.md`; and
- `protocols/racer_c3_prospective_protocol_draft.md`.

Run the isolated numerical/contract tests with:

```bash
python -m unittest discover -s paper2_admet_benchmark/tests/racer_c3 -v
```

## RACER-C4 independent validation candidate

RACER-C3 did not pass its freeze gate and remains retrospective. RACER-C4/TAME
is a separate safety-first candidate built around two label-free transport
views, explicit effective-sample-size/balance audits, a baseline-containing
protected-label consensus envelope, and a fail-closed final-label firewall.

The public Tox21 leaderboard batch is development-only. The independent EPA
batch uses fresh seeds 211--215; its labels cannot be parsed until every final
prediction is hashed into a promotion record. Exact method, source, gate, and
non-claim boundaries are in:

- `configs/racer_c4/prospective_lock_v1.yaml`;
- `docs/racer_c4_algorithm_specification_v1.md`;
- `docs/racer_c4_prior_art_boundary.md`; and
- `protocols/racer_c4_independent_epa_validation_protocol.md`.

Windows one-command reproduction after `git pull`:

```powershell
powershell -ExecutionPolicy Bypass -File paper2_admet_benchmark\scripts\racer_c4\run_racer_c4_overnight.ps1
```
