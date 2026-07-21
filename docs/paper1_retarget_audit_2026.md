# Paper 1 Retargeting Audit After JMGM Desk Rejection

Date: 2026-07-22

## Rejection interpretation

The JMGM decision was a scope-based desk rejection. No reviewer comments, methodological criticisms, novelty criticisms, or language criticisms were provided. The manuscript should therefore be retargeted rather than abandoned.

## Recommended journal order

### 1. Journal of Chemometrics

Best current fit after revision.

Why it fits:

- publishes fundamental and applied chemometrics and cheminformatics;
- explicitly welcomes modeling, simulation, pattern recognition, data mining, software, and high-quality benchmark datasets;
- has recently published work focused on model validation and random splitting;
- offers a subscription route, so open access is optional.

Main desk-rejection risk:

- the paper may be considered too narrow for the general chemometrics community;
- the target-balanced scaffold split may be considered a minor heuristic unless its objective and statistical behavior are formalized;
- five seeds may be considered insufficient for strong uncertainty claims.

Required positioning:

- present the contribution as a chemometric evaluation framework;
- emphasize model-validation and sample-partitioning methodology;
- retain molecular property prediction as the application domain, not the sole contribution.

### 2. Chemometrics and Intelligent Laboratory Systems

Strong backup but higher methodological threshold.

Why it fits:

- publishes new statistical, mathematical, and computational techniques in chemistry;
- publishes benchmark datasets and systematic algorithm-evaluation frameworks;
- subscription publication has no author publication fee.

Main risk:

- explicitly rejects routine applications of established methods;
- current manuscript requires a stronger formal method and robustness analysis before submission.

### 3. Molecular Informatics

Not recommended as the next submission.

Reason:

- recent research articles are dominated by new molecular-generation, graph-learning, chemical-space, and drug-design methods;
- acceptance rate is low;
- the current manuscript does not propose a new molecular representation or predictor.

## Revision strategy

### A. Reframe the contribution

Proposed title:

`A Split-Diagnostic Framework for Molecular Property Prediction Benchmarks: Disentangling Target Shift, Scaffold Separation, and Chemical Similarity`

Alternative title with stronger chemometrics positioning:

`Chemometric Diagnosis of Data-Splitting Effects in Molecular Property Prediction Benchmarks`

The paper should be described as a model-validation and sample-partitioning framework for chemical data.

### B. Formalize the target-balanced scaffold objective

Current objective implemented by the code:

`J(S) = |n_S - n_target| / n_target + |mean(y_S) - mean(y)| / sd(y)`

where `S` is the selected test-scaffold set.

Constraints:

- complete scaffold groups are assigned together;
- no scaffold appears in both train and test sets;
- the selected test size remains near the requested proportion.

The manuscript should state that this is a stochastic greedy search over scaffold groups, repeated for 300 trials per seed.

### C. Add statistical robustness

Required minimum:

1. increase the evaluation from 5 to 20 seeds;
2. calculate paired ordinary-versus-balanced gap differences using the same seed;
3. report bootstrap 95% confidence intervals for gap reduction;
4. report Wilcoxon signed-rank tests with Holm correction;
5. report target-gap reduction and test-size deviation;
6. compare target-balanced splits with random scaffold-subset null assignments.

### D. Preserve claim boundaries

Do not claim:

- a universally superior split;
- a new state-of-the-art predictor;
- prospective drug-discovery validation;
- that target balance alone guarantees an unbiased benchmark.

Claim instead:

- a reproducible diagnostic framework;
- a formalized split objective;
- evidence that split-induced confounding changes performance estimates and rankings;
- recommendations for reporting chemical benchmark validity.

## Immediate execution plan

1. Add a 20-seed robustness experiment script.
2. Add a random scaffold-assignment null audit.
3. Regenerate tables and figures.
4. Rewrite title, abstract, introduction, methods, results, and discussion for Journal of Chemometrics.
5. Add recent chemometric validation references.
6. Prepare a Wiley-format cover letter and submission package.
