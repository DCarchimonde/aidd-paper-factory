# Paper 2 confirmatory analysis plan

## Status

This document freezes the transition from exploratory development to confirmatory analysis.

- **Development/pilot phase:** all results generated from `split_random_seed0` and the current deterministic class-balance-aware `split_scaffold`.
- **Confirmatory phase:** only results generated after this protocol was frozen, using previously unseen random, label-blind scaffold, and similarity-cluster splits.
- Development results may be used for debugging, method selection, and hypothesis generation, but not as the sole evidence for confirmatory claims.

## Frozen research focus

The primary contribution is a unified audit of molecular prediction reliability under chemical distribution shift, with emphasis on:

1. chemical applicability-domain distance;
2. probability calibration;
3. marginal versus class-conditional conformal prediction;
4. shift-aware conformal prediction;
5. selective prediction and abstention behavior;
6. classification and regression endpoints.

The manuscript must not frame minority-class conformal failure alone as the main novelty.

## Confirmatory research questions

### RQ1
How do discrimination, regression error, and calibration change across random, label-blind scaffold, and similarity-cluster shifts?

### RQ2
Does distance from the training chemical domain identify subgroups with elevated error and conformal undercoverage?

### RQ3
Do marginal conformal summaries conceal class-conditional or domain-conditional failures, especially for imbalanced toxicity endpoints?

### RQ4
Can Mondrian and shift-aware conformal methods improve subgroup coverage, and what efficiency cost do they incur in prediction-set size or interval width?

## Frozen split policy

### Development splits

- `split_random_seed0`
- `split_scaffold` (deterministic, class-balance-aware)

### Confirmatory random splits

- Seeds: `101, 102, 103, 104, 105, 106, 107, 108, 109, 110`
- Classification random splits may be stratified.
- Regression random splits are unstratified.

### Confirmatory label-blind scaffold splits

- Seeds: `101, 102, 103, 104, 105, 106, 107, 108, 109, 110`
- Scaffold groups are assigned using only scaffold identity and group size.
- Target labels or target values must not be used during scaffold assignment.
- Every scaffold must appear in exactly one role.

### Confirmatory similarity-cluster splits

- Seeds: `101, 102, 103, 104, 105`
- Clusters are generated without target labels.
- Every similarity cluster must appear in exactly one role.
- The clustering algorithm and similarity threshold must be reported exactly.
- Very large endpoints may be excluded from exact pairwise clustering if the pre-specified computational cap is exceeded; such exclusions must be documented before result inspection.

## Frozen data roles

For every split:

- 60% train
- 20% calibration
- 20% test

Model fitting uses train only. Hyperparameter selection, probability calibration, conformal threshold estimation, and any learned uncertainty transformation may use train and/or calibration only. Test labels must never affect fitting, threshold selection, model selection, split creation, or method selection.

## Frozen model comparison policy

Two imbalance regimes must be reported separately for classification:

1. **Unweighted regime:** no class weighting or resampling for any model.
2. **Balanced regime:** class weighting or sample weighting is applied consistently across all supported model families.

Architecture comparisons must not mix weighted and unweighted models in the same ranking table.

Core model families:

- Logistic regression / ridge
- Random forest
- XGBoost
- MLP on ECFP

## Frozen uncertainty and reliability methods

### Classification

- Raw model probability
- Marginal split conformal (LAC)
- Mondrian/class-conditional split conformal
- One pre-specified shift-aware conformal method

### Regression

- Symmetric split conformal with absolute residuals
- One adaptive or normalized conformal baseline
- One pre-specified shift-aware conformal method where computationally feasible

## Frozen applicability-domain diagnostics

Primary continuous diagnostics:

- maximum Tanimoto similarity to the training set;
- mean top-5 Tanimoto similarity to the training set;
- unseen-scaffold indicator.

Primary fixed bins:

- high domain: max Tanimoto >= 0.70
- medium domain: 0.40 <= max Tanimoto < 0.70
- low domain: max Tanimoto < 0.40

Sensitivity analyses:

- thresholds 0.30/0.60, 0.40/0.70, and 0.50/0.80;
- quantile-based similarity bins;
- continuous similarity-risk association.

## Frozen primary metrics

### Classification performance

- ROC-AUC
- PR-AUC
- balanced accuracy
- Brier score
- negative log-likelihood

### Regression performance

- RMSE
- MAE
- R-squared

### Conformal reliability

- marginal empirical coverage
- class-conditional coverage
- domain-conditional coverage
- prediction-set size or interval width
- empty, singleton, and ambiguous set rates for classification

### Selective prediction

- risk-coverage curve
- area under the risk-coverage curve (AURC)
- class-balanced or cost-sensitive risk for imbalanced classification
- positive and negative retention rates
- random-rejection baseline with repeated sampling

## Frozen reporting rules

- Report means and standard deviations across repeated splits.
- Report 95% confidence intervals where appropriate.
- Use paired split-level comparisons for method differences.
- Flag subgroup rows with small sample size or fewer than 10 samples in either class.
- Do not interpret single-class bins using ROC-AUC or PR-AUC.
- Do not claim theoretical conformal guarantees under distribution shift unless the method assumptions are explicitly satisfied.
- Treat development results as exploratory evidence only.

## Pre-specified hypotheses

1. Low chemical-domain similarity is associated with higher prediction risk for at least some endpoints and model families.
2. Marginal conformal coverage can conceal substantial class-conditional or domain-conditional undercoverage.
3. Mondrian conformal improves class-conditional coverage on imbalanced classification endpoints but may increase prediction-set size or ambiguity.
4. Chemical-domain filtering and model-confidence filtering can produce different class-retention behavior.
5. No single trust indicator is universally reliable across endpoints, model families, and shift types.

## Change control

Any change after confirmatory results are inspected must be documented as exploratory and reported separately from the frozen confirmatory analysis.