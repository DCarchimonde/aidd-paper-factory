# Paper 1: Interpretation of the 20-Seed Robustness and Null-Audit Results

Date: 2026-07-22

## Overall conclusion

The 20-seed analysis strengthens the paper, but it does not support a universal claim that target-balanced scaffold splitting improves model performance. Instead, it supports a more rigorous conclusion:

> The target-balanced scaffold procedure successfully controls target-distribution shift in most datasets, but the resulting change in apparent generalization is strongly dataset- and model-dependent.

This distinction is central to the revised chemometric framing. The split procedure is effective at its stated optimization objective, but controlling one confounder does not create a universally easier or more valid benchmark.

## Null-audit result

The null audit compares the target-balanced scaffold split with 5,000 random scaffold-subset assignments under the same approximate test-size constraints.

Balanced improvement percentiles:

- BACE: mean 0.9716, minimum 0.9206
- BBBP: mean 0.9577, minimum 0.9150
- ClinTox: mean 0.9170, minimum 0.8734
- ESOL: mean 0.9680, minimum 0.9208
- FreeSolv: mean/minimum 0.9410
- HIV: mean 0.6729, minimum 0.2170

Interpretation:

- For BACE, BBBP, ClinTox, ESOL, and FreeSolv, the balanced split usually achieved a smaller target gap than approximately 92% to 97% of random scaffold assignments.
- HIV is different. Its ordinary scaffold target gap was already very small, and the balanced search was only modestly better than the random-scaffold null distribution.
- FreeSolv remains structurally constrained: its balanced target gap is still large and its mean test-size deviation is 18.75%.

## 20-seed paired robustness result

Positive gap reduction means that the balanced scaffold split has a smaller random-versus-scaffold performance gap than the ordinary scaffold split. Negative gap reduction means that balancing increases that gap.

### Rows with statistically supported positive gap reduction after Holm correction

- BBBP Logistic Regression: +0.0841 ROC-AUC gap reduction
- BBBP Random Forest: +0.0336
- ClinTox Logistic Regression: +0.0547
- ClinTox Random Forest: +0.0741
- ESOL Random Forest: +0.5027 RMSE gap reduction
- ESOL XGBoost: +0.6029
- FreeSolv Random Forest: +1.0529
- FreeSolv Ridge: +0.0905
- FreeSolv XGBoost: +0.9482

### Rows with statistically supported negative gap reduction after Holm correction

- BACE XGBoost: -0.0197
- ClinTox XGBoost: -0.0204
- ESOL Ridge: -0.7484
- HIV Logistic Regression: -0.0690
- HIV Random Forest: -0.0558
- HIV XGBoost: -0.0774

### Rows without Holm-corrected evidence

- BACE Logistic Regression
- BACE Random Forest
- BBBP XGBoost

Fifteen of the eighteen dataset-model comparisons remain significant after Holm correction. Nine show smaller gaps after balancing, six show larger gaps, and three remain inconclusive.

## Dataset-level interpretation

### BACE

The direction depends on the model. Logistic Regression shows a small positive mean reduction, Random Forest a small negative mean reduction, and XGBoost a significant negative reduction. BACE therefore does not support a dataset-wide claim that balancing improves the benchmark.

### BBBP

Balancing clearly reduces the gap for Logistic Regression and Random Forest. XGBoost shows only weak evidence. BBBP provides one of the strongest examples where ordinary scaffold performance was partly confounded by target shift.

### ClinTox

Logistic Regression and Random Forest improve after balancing, whereas XGBoost becomes worse. This model-dependent reversal is important evidence that split effects interact with model inductive bias.

### ESOL

Random Forest and XGBoost show large positive reductions, while Ridge shows a large negative reduction. This is the clearest example of a split changing model comparison rather than uniformly changing dataset difficulty.

### FreeSolv

All three models show smaller gaps after balancing. However, FreeSolv remains a structural edge case because the dataset is small, scaffold-concentrated, and difficult to divide near the requested test proportion while also controlling the target distribution.

### HIV

All three models show larger gaps after balancing. The ordinary scaffold target gap was already very small, and the balanced split was not strongly separated from the random-scaffold null distribution. HIV therefore demonstrates that optimizing target balance is unnecessary when target shift is already negligible and may move the test set into a different chemical difficulty regime.

## Revised claim boundary

Supported claim:

> The proposed target-balanced scaffold search reliably reduces target-distribution mismatch relative to random scaffold assignments in most tested datasets, but its effect on predictive performance is heterogeneous and model-dependent.

Unsupported claims:

- target-balanced scaffold splitting is universally better;
- reducing target shift necessarily reduces the generalization gap;
- a lower random-versus-scaffold gap automatically indicates a more realistic split;
- all performance differences are caused by target shift.

## Manuscript implications

The revised paper should emphasize three distinct layers:

1. **Optimization validity:** the method usually succeeds at reducing target shift relative to random scaffold assignments.
2. **Evaluation consequence:** controlling target shift changes benchmark results, often substantially.
3. **Non-universality:** the direction of the change depends on the dataset and model, so no split should be interpreted from one score alone.

This is a stronger and more defensible contribution than claiming that the balanced split simply improves performance.
