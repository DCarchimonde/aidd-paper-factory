# Final confirmatory method lock

## Status

This addendum was frozen before any confirmatory results from seeds 101-110 were generated or inspected. Seed 99 is technical validation only.

## Locked endpoints and splits

Core endpoints: BBBP, ClinTox, ESOL, and Lipophilicity.

- Random confirmatory splits: seeds 101-110.
- Label-blind scaffold confirmatory splits: seeds 101-110.
- Similarity-cluster confirmatory splits: seeds 101-105.
- Data roles: 60% train, 20% calibration, 20% test.

## Locked model and imbalance regimes

Model families:

- logistic regression / ridge;
- random forest;
- XGBoost;
- ECFP MLP.

Classification regimes are reported separately:

1. unweighted training on the original training distribution;
2. balanced random oversampling of the minority class within the training role only.

No architecture ranking may mix these regimes.

## Locked calibration diagnostics

Classification reports both:

- positive-probability ECE/MCE: mean predicted P(y=1) versus observed positive rate;
- confidence ECE/MCE: maximum class probability versus classification accuracy.

Brier score and negative log-likelihood remain primary proper scoring rules.

## Locked conformal methods

### Marginal split conformal

- Classification: LAC score, 1 minus predicted probability of the true class.
- Regression: absolute residual score.
- Finite-sample quantile: ceil((n+1)(1-alpha)).

### Mondrian conformal

- Separate LAC calibration thresholds for class 0 and class 1.
- Report positive/negative coverage and prediction-set efficiency.

### Shift-weighted conformal

A transductive covariate-shift-weighted baseline is used for classification and regression.

- Covariates: fixed ECFP4 features.
- Domain labels: calibration covariates = 0; unlabeled test covariates = 1.
- Density-ratio estimator: five-fold cross-fitted L2 logistic regression, C=1.0, liblinear, no class weighting.
- Ratio correction: odds multiplied by n_calibration / n_test.
- Fixed clipping: [0.05, 20.0].
- Weighted conformal quantile includes the test-point mass at infinity.
- Report domain-classifier AUC, calibration-weight effective sample size, infinite-threshold rate, coverage, and efficiency.

Because density ratios are estimated, results are described as empirical shift-aware diagnostics, not exact guarantees under arbitrary chemical shift.

### Adaptive regression conformal

A split-calibration normalized residual baseline is used.

- Calibration role is divided 50/50 using a deterministic RNG derived from the split seed.
- First half: random-forest error-scale model fitted to log(1 + absolute residual), using 300 trees, min_samples_leaf=20, max_features=sqrt.
- Second half: normalized residual conformal quantile.
- Scale floor: max(0.1 times the median scale-fit absolute residual, 0.001).
- Test interval half-width: normalized quantile multiplied by predicted test error scale.

## Locked applicability-domain diagnostics

- ECFP4 maximum Tanimoto to training;
- mean top-5 Tanimoto to training;
- unseen-scaffold indicator;
- fixed bins: low <0.40, medium 0.40-0.70, high >=0.70.

Threshold sensitivity is restricted to 0.30/0.60, 0.40/0.70, and 0.50/0.80.

## Locked selective-prediction analysis

Primary rankings:

- classification: one minus maximum class probability; one minus maximum Tanimoto;
- regression: one minus maximum Tanimoto.

Coverage grid: 0.10 to 1.00 in increments of 0.05.

Metrics:

- classification error rate;
- balanced error rate when both classes remain;
- false-negative and false-positive rates;
- positive and negative retention rates;
- regression RMSE and MAE;
- AURC.

At every matched coverage, the ranking is compared with 500 deterministic repeated random-rejection samples. Report random mean and 95% empirical interval.

## Locked reporting and stopping rules

- Confirmatory results are summarized across split seeds, not selected by best seed.
- Report mean, standard deviation, and 95% confidence interval.
- Comparisons are paired at the endpoint/model/regime/split-seed level.
- No method, threshold, endpoint, model, or seed may be removed because of an unfavorable result.
- Failures caused by small or single-class subgroups are reported with warnings rather than silently omitted.
- Any code or method change after confirmatory output inspection is exploratory and must be labeled separately.
