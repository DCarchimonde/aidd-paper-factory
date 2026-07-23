# Paper 2 manuscript scope, claim, and submission lock

## Status

This document is frozen after completion of the confirmatory analyses. It governs manuscript writing, figure design, journal selection, and interpretation. No claim may be strengthened by selecting individual seeds, models, thresholds, endpoints, or split types after viewing the results.

## Working title

**Beyond Accuracy in ADMET Prediction: Applicability-Domain Diagnostics and Conformal Calibration under Chemical Distribution Shift**

The title may be shortened during journal formatting, but it must retain the following three elements:

1. reliability beyond point accuracy;
2. applicability-domain or distribution-shift diagnostics;
3. conformal calibration or prediction-set/interval reliability.

## Target-journal lock

### Primary target: Journal of Chemometrics

The manuscript will be framed as an applied chemometrics and cheminformatics study of reliability diagnostics, supervised modelling, calibration, pattern recognition, data mining, and benchmark design. It must not read as a routine ADMET model-comparison paper.

### Backup target: Chemometrics and Intelligent Laboratory Systems

The backup framing will emphasize a reproducible chemical-data reliability evaluation framework and the non-equivalence of predictive accuracy, marginal validity, class-conditional validity, applicability-domain risk, and selective-prediction utility. Because routine applications of established techniques are not sufficient for this journal, the contribution must be the integrated diagnostic framework, paired confirmatory design, and cross-task evidence rather than the use of any single established method.

### Scope lesson from Paper 1

Paper 1 was desk rejected on 21 July 2026 because it did not fit the selected journal's scope; no methodological or reviewer criticism was provided. For Paper 2, journal fit must be explicit in the title, abstract, introduction, contribution statement, keywords, figures, and cover letter before submission.

## Frozen study design

- Endpoints: BBBP, ClinTox, ESOL, and Lipophilicity.
- Tasks: binary classification and regression.
- Splits: random, label-blind scaffold, and similarity-cluster.
- Confirmatory seeds: random/scaffold 101-110; cluster 101-105.
- Classification models: linear, random forest, XGBoost, and MLP, each under unweighted and balanced-oversample regimes.
- Regression models: four frozen regressors.
- Formal conclusions use only confirmatory outputs. Seed 99 is a technical smoke test and must never be cited as scientific evidence.
- Cross-model headline means are descriptive summaries. Models are not treated as independent inferential replicates.
- Inferential stability is evaluated within model across split seeds, using model-specific confidence intervals and paired treatment-minus-control analyses.

## Research questions

### RQ1

How do predictive performance and calibration change across random, scaffold, and similarity-cluster splits?

### RQ2

Does chemical-domain distance identify high-error, high-miscoverage, or selectively rejectable compounds, and is this relationship robust to endpoint, split, and threshold choice?

### RQ3

Can aggregate accuracy, aggregate calibration, marginal conformal coverage, or ordinary selective risk conceal minority-class or subgroup failure?

### RQ4

Do Mondrian, covariate-shift-weighted, or locally adaptive conformal methods improve practical reliability, and what efficiency or informativeness costs accompany any coverage repair?

## Central contribution

The paper does not claim a universally superior ADMET model or uncertainty method. Its contribution is a confirmatory, label-blind, repeated-split diagnostic framework showing that reliability is multidimensional and endpoint-dependent.

The principal scientific message is:

> Aggregate accuracy and marginal coverage can conceal chemically and clinically relevant failure modes. Applicability-domain scores, conformal variants, and selective-prediction rules provide complementary but non-interchangeable evidence, and apparent coverage repair may not translate into informative or practically useful predictions.

## Claims supported by the frozen evidence

### Claim 1: chemical shift degrades predictive performance and calibration, but the magnitude is endpoint dependent

- Random splits generally produce the strongest classification and regression performance and the best classification calibration.
- ESOL performance deteriorates sharply under scaffold splitting.
- Mean Lipophilicity R2 is negative under all three splitting schemes, demonstrating that nominal coverage validity does not imply useful point prediction.

### Claim 2: applicability-domain utility is not universal

- Tanimoto similarity is a stable risk and miscoverage indicator for BBBP and especially Lipophilicity.
- The association is weak for ClinTox.
- The association reverses or becomes non-monotonic for ESOL under scaffold splitting.
- Threshold sensitivity and continuous analyses therefore support endpoint- and shift-specific AD interpretation, not a universal cutoff.

### Claim 3: marginal coverage can hide severe minority-class failure

- BBBP marginal and shift-weighted methods over-cover the positive class and under-cover the negative class.
- ClinTox marginal and shift-weighted methods achieve overall coverage near 0.90 while positive-class coverage remains extremely low.
- Ordinary aggregate coverage must therefore be accompanied by class-conditional coverage and class counts.

### Claim 4: Mondrian repair is statistically stable but may be practically uninformative

- Mondrian conformal strongly improves ClinTox positive-class coverage and reduces class-conditional coverage gaps across all frozen model/regime combinations.
- The repair is accompanied by large prediction sets and high ambiguous-set rates, often approaching a near-vacuous two-label output.
- The correct conclusion is coverage-informativeness trade-off, not that Mondrian is universally best.

### Claim 5: shift weighting and adaptive regression are not consistently superior

- Density-ratio weighting occasionally changes coverage or width, but improvements are endpoint- and split-dependent and are not consistently supported across model families.
- Adaptive regression intervals show modest width variation and weak width-error association, with no stable coverage-efficiency advantage over marginal conformal.

### Claim 6: selective prediction can create an illusion of safety

- Model confidence improves ordinary selective risk for ClinTox while worsening balanced-risk summaries and disproportionately removing positive examples.
- Similarity-based rejection retains more ClinTox positives but is not uniformly effective across splits.
- Confidence and chemical-domain distance are complementary trust indicators and cannot be substituted for each other.

## Prohibited or downgraded claims

The manuscript must not state or imply any of the following:

1. one model family is universally best;
2. random-split performance represents prospective chemical generalization;
3. approximately 90% marginal coverage means every class or domain is reliable;
4. a lower ClinTox Brier score than BBBP proves ClinTox is better calibrated across endpoints;
5. Mondrian conformal solves practical reliability without qualification;
6. shift weighting corrects class imbalance;
7. adaptive intervals successfully identify difficult compounds without reporting weak width-error correlations;
8. low Tanimoto similarity universally predicts high error;
9. an AD threshold of 0.40 or 0.70 has universal chemical meaning;
10. ordinary selective-risk improvement is beneficial when minority-class retention deteriorates;
11. negative or null results are implementation failures to be hidden;
12. models are independent replicates for statistical inference;
13. pAURC over coverage 0.10-1.00 is full AURC;
14. conformal validity implies high predictive accuracy or useful decisions;
15. annotation, calibration, applicability-domain analysis, or prediction-set coverage constitutes biological or experimental validation.

## Required reporting safeguards

Every main result must satisfy the following:

- Report endpoint, task, split type, method, alpha, seed support, and aggregation level.
- Distinguish descriptive averages across models from model-specific cross-seed inference.
- Report class-conditional coverage for classification.
- Report prediction-set size, ambiguous-set rate, or interval width together with coverage.
- Report minority-class retention together with selective-risk improvement.
- State that pAURC is integrated over retained coverage 0.10-1.00.
- Flag sparse subgroup support, especially ClinTox high-domain cluster and scaffold groups.
- Interpret negative R2 values directly and without euphemism.
- Keep threshold analyses as sensitivity evidence; continuous associations are the primary AD evidence.
- Use paired same-seed, same-model comparisons for method contrasts.

## Planned manuscript structure

### 1. Introduction

1. ADMET models are commonly compared by aggregate predictive performance.
2. Random splitting, chemical shift, class imbalance, and heterogeneous error undermine direct interpretation.
3. Calibration, conformal prediction, applicability domains, and selective prediction address different reliability questions.
4. Existing studies often evaluate these tools separately or rely on marginal summaries.
5. This study jointly evaluates four endpoints, two task types, three label-blind split regimes, repeated confirmatory seeds, class/domain diagnostics, and paired conformal trade-offs.
6. State RQ1-RQ4 explicitly.

### 2. Methods

1. Data sources and endpoint-specific composition.
2. Molecular representation and frozen model families.
3. Random, label-blind scaffold, and similarity-cluster splitting.
4. Calibration design and train/calibration/test separation.
5. Marginal, Mondrian, density-ratio-weighted, and adaptive conformal methods.
6. Applicability-domain definitions, continuous diagnostics, threshold sensitivity, and quartiles.
7. Selective prediction, matched random rejection, pAURC range, and class retention.
8. Confirmatory seeds, paired comparisons, confidence intervals, multiplicity/interpretation boundaries, and reproducibility.

### 3. Results

#### 3.1 Distribution shift changes predictive performance and calibration

RQ1 classification performance, regression performance, and classification calibration.

#### 3.2 Chemical similarity is an endpoint-dependent applicability-domain indicator

RQ2 continuous association first; threshold and quartile analyses second.

#### 3.3 Marginal validity conceals class-conditional failure

RQ3 BBBP and ClinTox marginal and shift-weighted coverage.

#### 3.4 Mondrian coverage repair trades validity for informativeness

RQ4 paired coverage-gap reduction, prediction-set expansion, and ambiguity.

#### 3.5 Regression conformal variants offer no consistent universal improvement

Coverage, width, interval heterogeneity, and width-error association.

#### 3.6 Selective prediction can reduce ordinary risk by erasing the minority class

Confidence versus chemical similarity, balanced pAURC, and ClinTox positive retention.

### 4. Discussion

1. Reliability is multidimensional.
2. Chemical-domain distance is not a universal uncertainty proxy.
3. Marginal validity and practical informativeness are distinct.
4. Class imbalance can invalidate apparently reassuring aggregate metrics.
5. Negative results for shift weighting and adaptive intervals are informative method-selection evidence.
6. Implications for prospective ADMET deployment and benchmark reporting.
7. Limitations: four endpoints, ECFP-centred representation, fixed model families, no prospective wet-lab validation, finite seed counts, threshold dependence, and sparse subgroups.

### 5. Conclusion

A concise statement that trustworthy ADMET evaluation requires joint reporting of point performance, calibration, class-conditional coverage, efficiency, applicability-domain diagnostics, and retention-aware selective risk.

## Main figure plan

### Figure 1: study and confirmatory evaluation design

Datasets, task types, label-blind splits, train/calibration/test roles, frozen methods, repeated seeds, and RQ mapping.

### Figure 2: predictive performance and calibration under chemical shift

Classification ROC-AUC/balanced accuracy, regression RMSE/R2, and classification calibration across split types.

### Figure 3: class-conditional conformal validity and informativeness

Positive/negative coverage, class-conditional gap, mean set size, and ambiguous-set rate for marginal, shift-weighted, and Mondrian methods.

### Figure 4: endpoint-dependent applicability-domain diagnostics

Continuous similarity-risk/miscoverage associations, OOD AUC, and selected threshold-sensitivity contrasts.

### Figure 5: selective prediction and minority retention

Matched pAURC improvement, balanced pAURC improvement, risk improvement at 50% retention, and ClinTox positive retention.

### Figure 6: regression conformal coverage-efficiency-adaptivity comparison

Coverage, width, width variation, and width-error association for marginal, shift-weighted, and adaptive intervals.

## Table plan

- Main Table 1: datasets, task types, sample counts, class balance, and split/calibration design.
- Main Table 2: compact RQ1 performance and calibration summary.
- Main Table 3: classification conformal coverage-efficiency trade-offs.
- Main Table 4: AD and selective-prediction headline findings.
- Supplementary tables: complete model-specific cross-seed intervals, threshold sensitivity, quartiles, alpha=0.20 analyses, sparse subgroup support, and integrity manifest.

## Pre-submission acceptance-risk checks

Before submission, all items must pass:

- [ ] Journal scope language appears in title, abstract, introduction, keywords, and cover letter.
- [ ] Dataset composition and train/calibration/test counts are explicit.
- [ ] Split generation is label blind and reproducible.
- [ ] All repeated-seed results and confidence intervals are reported or supplied.
- [ ] No model-selection or method-selection decision was made from confirmatory outcomes.
- [ ] All negative and null results are retained.
- [ ] Claims use paired evidence where available.
- [ ] Marginal and class-conditional coverage are not conflated.
- [ ] Coverage and efficiency/informativeness appear together.
- [ ] Selective risk and minority retention appear together.
- [ ] Sparse groups and missing metrics are explained rather than silently dropped.
- [ ] Figures remain legible at journal column width, with no tiny annotations.
- [ ] Main and supplementary tables have stable row counts and SHA256 manifests.
- [ ] Code, seeds, environment, and output provenance are documented.
- [ ] The cover letter explains why the work is chemometrics rather than a routine ADMET benchmark.
