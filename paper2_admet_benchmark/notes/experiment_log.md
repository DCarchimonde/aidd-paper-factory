# Experiment Log - Paper 2

## 2026-07-08

Goal:

- Initialize `paper2_admet_benchmark` inside `aidd-paper-factory`.
- Freeze protocol v1.0 for the MVP execution.
- Define Paper 2 as a reliability-oriented ADMET prediction study.
- Create endpoint, split, model, calibration, conformal, and reviewer-risk configuration files.

Completed:

- Created Paper 2 README.
- Added Paper 2 story note.
- Added MVP endpoint configuration.
- Added split configuration with train/calibration/test separation.
- Added model configuration with LR/Ridge, Random Forest, XGBoost, and MLP-on-ECFP baselines.
- Added calibration and conformal configuration.
- Added reviewer risk checklist.

Issues:

- Exact endpoint loading code still needs to be implemented.
- TDC endpoint names and column formats need to be verified in code.
- MoleculeNet/DeepChem endpoint URLs and source notes need to be recorded in endpoint registry.

Next:

- Implement `00_test_environment.py`.
- Implement `01_prepare_endpoints.py`.
- Generate `data/manifests/endpoint_registry.csv` after first data-preparation run.
