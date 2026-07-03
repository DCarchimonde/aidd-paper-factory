# Paper 1 Research Audit

Date: 2026-07-03

Working title:

**A Diagnostic Audit of Split-Induced Generalization Gaps in Molecular Property Prediction for AI-Driven Drug Discovery**

## 1. Current status

This project is **not yet a finished SCI manuscript**. It is currently a reproducible experimental prototype for Paper 1.

Completed so far:

- Public molecular property datasets prepared and cleaned.
- RDKit canonical SMILES and Morgan fingerprints generated.
- Random split, ordinary scaffold split, and target-balanced scaffold split evaluated.
- Multiple classical ML baselines run across 5 seeds.
- Split-induced target distribution shift diagnosed.
- Main result tables and initial figures generated.

Not completed yet:

- Full literature review.
- Journal-specific framing.
- Additional robustness checks.
- Final manuscript writing.
- External validation against additional datasets or split strategies.

## 2. Core research question

Do common train/test splitting protocols change the estimated generalization performance of molecular property prediction models in AIDD, and does ordinary scaffold splitting confound chemical scaffold shift with target distribution shift?

## 3. Proposed contribution

The contribution should be framed narrowly and honestly:

> We present a reproducible split-diagnostic audit pipeline that quantifies both model performance gaps and split-induced target distribution shifts across random, ordinary scaffold, and target-balanced scaffold evaluation protocols.

This is a benchmark audit contribution, **not** a new molecular prediction algorithm.

## 4. What is already known

The following topics are already established in the literature:

1. Molecular property prediction benchmarks such as MoleculeNet provide public datasets, metrics, and baseline evaluation protocols.
2. Random splitting can produce optimistic performance because structurally similar molecules may appear in both train and test sets.
3. Scaffold splitting is widely used to test chemical scaffold generalization.
4. Scaffold splitting is not perfect; recent work argues that scaffold splits can still overestimate virtual screening performance because different scaffolds may remain chemically similar.
5. Therapeutics Data Commons and related benchmark platforms emphasize reproducibility, meaningful splits, distribution shifts, and robust generalization.
6. Imbalanced datasets are a known issue in AIDD and can distort metrics such as accuracy and F1.

## 5. What appears potentially distinctive in this project

Based on the preliminary literature scan, the potentially distinctive angle is **not** simply random vs scaffold splitting.

The potentially distinctive angle is the combined audit:

1. Quantify random-vs-scaffold performance gaps.
2. Measure target distribution shift induced by ordinary scaffold splitting.
3. Introduce a target-balanced scaffold split as a diagnostic counterfactual.
4. Compare whether performance gaps shrink, remain, or change direction after target balancing.
5. Report both performance gaps and target-shift diagnostics as mandatory benchmark outputs.

This combination may be a publishable diagnostic benchmark angle, but it still requires a fuller related-work check before strong novelty claims.

## 6. Claims we can make now

Safe claims:

- Random split often gives more optimistic performance estimates than scaffold-based evaluation in the current benchmark setup.
- Ordinary scaffold split can introduce substantial target distribution shift.
- Target-balanced scaffold splitting reduces target distribution shift in the tested datasets.
- After target balancing, residual generalization gaps remain for several dataset-model combinations.
- Split choice should be reported together with split diagnostics, not only model performance.

## 7. Claims we must not make

Unsafe claims:

- We are the first to study random vs scaffold split.
- Scaffold split is always worse than random split.
- Target-balanced scaffold split is a universally better benchmark.
- We have invented a new AIDD model.
- The current results alone are enough for SCI acceptance.
- The current four datasets cover all AIDD molecular property prediction scenarios.

## 8. Current evidence

Current generated tables:

- `paper1_main_gap_table.csv`
- `paper1_main_gap_table_rounded.csv`
- `paper1_split_shift_table.csv`
- `paper1_split_shift_table_rounded.csv`
- `paper1_combined_split_performance.csv`
- `paper1_combined_gap_summary.csv`
- `paper1_combined_split_diagnostics.csv`

Current generated figures:

- `figure1_gap_ordinary_vs_balanced.png`
- `figure2_split_target_shift.png`

## 9. Interpretation of current evidence

The results support a nuanced conclusion:

1. Ordinary scaffold split often produces performance drops relative to random split.
2. However, ordinary scaffold split also creates target distribution shift.
3. Target-balanced scaffold split greatly reduces this target shift.
4. After balancing, performance gaps shrink in several cases but do not disappear completely.
5. Therefore, ordinary scaffold split performance should be interpreted as a mixture of chemical scaffold shift and target distribution shift.

This is stronger and more honest than saying simply: "scaffold split is harder."

## 10. Remaining experiments for Paper 1

Recommended before manuscript submission:

1. Add at least 2 additional datasets, preferably one classification and one regression task.
2. Add a scaffold statistics table: number of scaffolds, largest scaffold fraction, singleton scaffold fraction, train/test shared scaffold count.
3. Add a simple similarity audit, such as maximum Tanimoto similarity from each test molecule to the training set under each split.
4. Add confidence intervals or paired seed-level comparison for key gaps.
5. Inspect outlier cases such as ESOL Ridge and ClinTox XGBoost.
6. Clean and standardize all figure captions.
7. Write a related-work table comparing MoleculeNet, TDC, scaffold split critique papers, imbalance benchmarks, and this project.

## 11. Paper 1 positioning

Best positioning:

- Benchmark audit
- Evaluation protocol diagnosis
- Reproducibility and split quality
- AIDD molecular property prediction reliability

Avoid positioning as:

- New model paper
- State-of-the-art ADMET predictor
- Pure algorithm paper
- Full drug discovery pipeline

## 12. Relationship to the broader factory

The AIDD paper factory should be staged:

- Paper 1: split-diagnostic benchmark audit.
- Paper 2: ADMET task extension using the same diagnostics pipeline.
- Paper 3: target-specific interpretable virtual screening case study.

Paper 1 is the foundation. Paper 2 and Paper 3 should reuse the shared utilities, split diagnostics, and reporting templates developed here.

## 13. Current go/no-go assessment

Current status: **Go, but not ready for submission.**

The project has a credible research question and useful preliminary results. It should proceed to experimental strengthening and manuscript drafting, but must not be represented as an already completed SCI paper.
