# Paper 1 Manuscript Outline

Working title:

**A Diagnostic Audit of Split-Induced Generalization Gaps in Molecular Property Prediction for AI-Driven Drug Discovery**

## Core message

This paper argues that molecular property prediction benchmarks should diagnose the train/test split itself, not only report model scores. Random splitting, ordinary scaffold splitting, and target-balanced scaffold splitting produce different performance estimates, target distribution shifts, scaffold statistics, chemical similarity patterns, and model rankings.

## Proposed abstract structure

1. Background: Molecular property prediction is central to AI-driven drug discovery, but benchmark conclusions depend strongly on how molecules are split into train and test sets.
2. Problem: Random splits can be overly optimistic, while ordinary scaffold splits may confound chemical scaffold generalization with target distribution shift and scaffold concentration.
3. Method: We audit six public molecular property datasets under random, ordinary scaffold, and target-balanced scaffold splitting. We evaluate classical ML baselines using Morgan fingerprints across five seeds and report performance gaps, target-shift diagnostics, scaffold statistics, test-to-train Tanimoto similarity, outlier cases, and model ranking changes.
4. Results: Random splits often produce higher test-to-train similarity and more optimistic scores. Ordinary scaffold splits remove scaffold overlap but can introduce target shifts and scaffold concentration. Target-balanced scaffold splits reduce target shift in several datasets, but residual performance gaps and model-ranking changes remain dataset- and model-dependent.
5. Conclusion: Split diagnostics should be reported alongside model performance in AIDD benchmarks.

## 1. Introduction

### Paragraph 1: Importance of molecular property prediction

- Molecular property prediction supports early-stage AIDD tasks such as ADMET prediction, toxicity screening, and lead prioritization.
- Public datasets and benchmark protocols have accelerated method development.

### Paragraph 2: Why splitting matters

- Benchmark scores depend on whether test molecules resemble training molecules.
- Random splits can place chemically similar molecules in train and test sets.
- Scaffold splits are commonly used to evaluate scaffold generalization.

### Paragraph 3: Why scaffold split is not enough

- Ordinary scaffold splits remove scaffold overlap, but may create target distribution shift.
- A split can be chemically non-overlapping yet statistically biased.
- Therefore, model performance should be interpreted together with split diagnostics.

### Paragraph 4: Our contribution

Contributions:

1. A reproducible split-diagnostic audit pipeline for molecular property prediction.
2. Comparison of random, ordinary scaffold, and target-balanced scaffold protocols.
3. Joint reporting of model performance gaps, target distribution shift, scaffold statistics, and test-to-train Tanimoto similarity.
4. Analysis of model-ranking instability and outlier cases across datasets and split protocols.

## 2. Related Work

Use `docs/paper1_related_work_table.md` as the base.

Key positioning:

- MoleculeNet: public benchmark datasets and model evaluation.
- TDC: therapeutics ML task platform and robust generalization emphasis.
- Scaffold split critique: scaffold splits may still overestimate performance due to chemical similarity.
- Imbalanced AIDD benchmarks: class imbalance affects metrics and model interpretation.

Safe wording:

> Unlike standard benchmark reports that primarily compare model performance under predefined splits, this study treats the split itself as an object of diagnosis.

Avoid:

- Claiming the first random-vs-scaffold split study.
- Claiming target-balanced scaffold split is universally better.
- Claiming a new SOTA molecular predictor.

## 3. Materials and Methods

### 3.1 Datasets

Use manuscript Table 1.

Datasets:

- Classification: BBBP, BACE, ClinTox, HIV.
- Regression: ESOL, FreeSolv.

Report:

- raw rows
- rows after missing-value removal
- valid SMILES
- deduplicated canonical SMILES
- duplicates removed
- target mean/min/max

### 3.2 Molecular representation

- Canonicalize SMILES using RDKit.
- Generate Morgan fingerprints with radius 2 and 2048 bits.
- Use the same features across all models and splits.

### 3.3 Splitting protocols

Define three splits:

1. Random split: stratified for classification when applicable, repeated across five seeds.
2. Ordinary scaffold split: Bemis-Murcko scaffold grouping, no train/test scaffold overlap.
3. Target-balanced scaffold split: scaffold-level split selected to reduce train/test target mean difference while maintaining scaffold separation.

Important limitation wording:

> Target-balanced scaffold splitting is used as a diagnostic counterfactual, not as a universally superior benchmark split.

### 3.4 Models

Classification:

- Logistic Regression
- Random Forest
- XGBoost

Regression:

- Ridge Regression
- Random Forest
- XGBoost

### 3.5 Evaluation metrics

Classification:

- ROC-AUC as primary metric
- accuracy, F1, average precision as supporting metrics

Regression:

- RMSE as primary metric
- MAE and R2 as supporting metrics

Generalization gap:

- Classification gap = random ROC-AUC minus scaffold ROC-AUC.
- Regression gap = scaffold RMSE minus random RMSE.

### 3.6 Split diagnostics

Report:

- target mean gap
- total number of scaffolds
- largest scaffold fraction
- singleton scaffold fraction
- train/test shared scaffold count
- test-to-train maximum Tanimoto similarity
- model ranking changes
- outlier cases

## 4. Results

### 4.1 Dataset summary

Use Table 1.

Main point:

- The benchmark covers four classification and two regression datasets.
- HIV and ClinTox are highly imbalanced classification datasets.
- ESOL and FreeSolv are regression datasets with different target ranges.

### 4.2 Ordinary scaffold splitting removes scaffold overlap but can induce target shift

Use Table 2 and Figure 2.

Main observations:

- Ordinary scaffold splits have zero shared scaffolds between train and test.
- Random splits have substantial shared scaffold fractions.
- Ordinary scaffold splits can introduce large target mean gaps, especially ESOL and FreeSolv.
- Balanced scaffold splitting reduces target gaps in several datasets, but FreeSolv remains difficult because of strong scaffold concentration.

### 4.3 Split choice affects model performance estimates

Use Table 3 and Figure 1.

Main observations:

- Random splits often yield more optimistic performance.
- Balanced scaffold gaps are often smaller than ordinary scaffold gaps, showing that some ordinary scaffold gap is explained by target shift.
- Residual gaps remain after balancing in several dataset-model combinations.
- FreeSolv and ESOL Ridge show that effects are dataset- and model-dependent.

### 4.4 Random splits have higher test-to-train chemical similarity

Use Table 4 and Figure 3.

Main observations:

- Random split test molecules generally have higher maximum Tanimoto similarity to training molecules.
- Ordinary scaffold splitting reduces similarity most strongly.
- Balanced scaffold splitting often lies between random and ordinary scaffold.

### 4.5 Split choice can change model rankings

Use appendix model-ranking table.

Main observations:

- Model ranking changes across random, scaffold, and balanced scaffold splits.
- Evaluation split affects not only absolute scores but also which model appears best.

### 4.6 Outlier and limitation analysis

Use appendix outlier table.

Discuss:

- ClinTox is class-imbalance sensitive.
- ESOL Ridge behaves differently from tree-based models.
- FreeSolv balanced scaffold remains difficult due to scaffold concentration and target shift.

## 5. Discussion

### Main interpretation

The results support a split-diagnostic view of molecular property prediction benchmarks. Random splitting can create chemically similar train/test sets and optimistic performance. Ordinary scaffold splitting removes scaffold overlap but may create target distribution shift and scaffold concentration. Target-balanced scaffold splitting helps separate target-shift effects from scaffold-generalization effects, but does not eliminate all performance gaps.

### Practical recommendation

Benchmark reports should include:

1. Model scores.
2. Split type.
3. Target distribution diagnostics.
4. Scaffold statistics.
5. Test-to-train chemical similarity.
6. Model ranking stability.

### Limitations

- The study uses classical fingerprint-based models rather than deep molecular encoders.
- The target-balanced scaffold split is heuristic and diagnostic, not a new universal standard.
- The benchmark covers six public datasets, not all AIDD scenarios.
- Prospective/time-based validation is not included.
- Some small datasets, especially FreeSolv, are constrained by scaffold concentration.

## 6. Conclusion

AIDD benchmark evaluation should treat data splitting as an object of diagnosis. Reporting performance without target-shift, scaffold, and similarity diagnostics can lead to overconfident conclusions about molecular generalization. A split-diagnostic audit pipeline provides a more transparent way to interpret molecular property prediction results.

## Figures

- Figure 1: Generalization gap under ordinary vs target-balanced scaffold splitting.
- Figure 2: Target distribution shift by split strategy.
- Figure 3: Test-to-train Tanimoto similarity by split strategy.

## Tables

- Table 1: Dataset summary.
- Table 2: Split diagnostics and scaffold statistics.
- Table 3: Generalization gap summary.
- Table 4: Similarity audit summary.
- Appendix Table A1: Outlier cases.
- Appendix Table A2: Model rankings by split.

## Current writing status

Ready for manuscript drafting. The experimental section can be frozen unless a later journal target requires additional datasets or deep-learning baselines.
