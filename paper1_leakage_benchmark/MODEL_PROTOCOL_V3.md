# Paper 1 model rerun protocol v3

## Status

This protocol is frozen before any production model rerun. It must be used with the frozen v3 manifests generated under `paper1_leakage_benchmark/results/frozen_v3`.

## Statistical unit

- `partition_seed` defines data-split stochasticity.
- `model_seed` defines stochastic model fitting only.
- Partition and model seeds are never reused as one shared seed variable.
- The inferential unit is the unique partition hash.
- For stochastic models, replicate model seeds are averaged within each unique partition before any partition-level inference.
- Candidate-pool members are optimization diagnostics and are never treated as inferential replicates.

## Fingerprints

All models use RDKit Morgan fingerprints with radius 2 and 2048 bits. Fingerprints are generated from the `clean_v2` canonical isomeric SMILES and cached locally only. No feature selection or split-specific fingerprint tuning is permitted.

## Models

### Classification

1. Logistic regression (`LR`)
   - `C=1.0`
   - `class_weight='balanced'`
   - `solver='liblinear'`
   - `max_iter=5000`
   - fixed fitting seed `0`
   - one fit per partition because this model is treated as the deterministic baseline

2. Random forest (`RF`)
   - `n_estimators=500`
   - `max_depth=None`
   - `min_samples_leaf=1`
   - `class_weight='balanced_subsample'`
   - `n_jobs=-1`
   - stochastic model seeds: `17, 29, 43`

3. XGBoost (`XGB`)
   - `n_estimators=500`
   - `max_depth=6`
   - `learning_rate=0.05`
   - `subsample=0.8`
   - `colsample_bytree=0.8`
   - `objective='binary:logistic'`
   - `eval_metric='logloss'`
   - `tree_method='hist'`
   - `n_jobs=-1`
   - training-split `scale_pos_weight = n_negative / n_positive`
   - stochastic model seeds: `17, 29, 43`

### Regression

1. Ridge regression (`Ridge`)
   - `alpha=1.0`
   - `solver='lsqr'`
   - one fit per partition

2. Random forest (`RF`)
   - `n_estimators=500`
   - `max_depth=None`
   - `min_samples_leaf=1`
   - `n_jobs=-1`
   - stochastic model seeds: `17, 29, 43`

3. XGBoost (`XGB`)
   - `n_estimators=500`
   - `max_depth=6`
   - `learning_rate=0.05`
   - `subsample=0.8`
   - `colsample_bytree=0.8`
   - `objective='reg:squarederror'`
   - `tree_method='hist'`
   - `n_jobs=-1`
   - stochastic model seeds: `17, 29, 43`

No hyperparameter tuning on the frozen test partitions is permitted.

## Metrics

### Classification

Primary metric: ROC-AUC.

Supporting metrics: average precision, F1 at threshold 0.5, accuracy, balanced accuracy, and Brier score. Train/test positive and negative counts must be saved for every job. A production classification partition with only one class in train or test fails the model-readiness gate rather than being silently dropped.

### Regression

Primary metric: RMSE.

Supporting metrics: MAE and R-squared.

## Protocols to fit

### Main analyses

For `main_classification` and `main_regression`, fit:

- `legacy_scaffold` as a deterministic/sensitivity reference only;
- `random_observation` as an observation-level reference;
- `size_matched_scaffold` as the target-blind scaffold baseline;
- `target_balanced_scaffold` as the paired target-aware scaffold design.

Only the 20 unique `size_matched_scaffold` versus `target_balanced_scaffold` partition pairs define the primary split-design comparison. Legacy scaffold has one partition and is not assigned 20-fold inferential weight.

### Acyclic sensitivity

For `acyclic_singleton_sensitivity`, fit only:

- `size_matched_scaffold`;
- `target_balanced_scaffold`.

Random-observation results are not duplicated because acyclic scaffold identity does not affect observation-level random splitting.

## Primary paired effect

For each dataset-model pair, after averaging stochastic model seeds within partition:

- classification improvement = `ROC_AUC(target_balanced) - ROC_AUC(size_matched)`;
- regression improvement = `RMSE(size_matched) - RMSE(target_balanced)`.

Positive values therefore always indicate better performance under target-balanced scaffold splitting.

Random split is a separate descriptive reference and is not algebraically subtracted from both scaffold protocols.

## Inference after rerun

Inference is performed only after all production jobs pass completeness checks. Planned outputs are:

- partition-level paired effect estimates;
- paired bootstrap 95% confidence intervals over unique partitions;
- two-sided Wilcoxon signed-rank tests at the partition level;
- Holm correction across the 18 main dataset-model cells;
- supporting-metric sensitivity summaries.

The legacy `9 smaller / 6 larger / 3 inconclusive` result remains superseded until the complete v3 rerun is finished.