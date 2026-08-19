# Frozen molecular null-permutation protocol

**Status:** pre-outcome frozen protocol  
**Version:** 1.0.0  
**Frozen:** 20 August 2026  
**Base submission commit:** `28a5dff363f4b2db1ef31b065116488ab45bec71`  
**Protocol SHA-256:** `eb05576aaed9ce2035d66067929dd8201692767b73d8fe931e446ecaa050cc7e`

## Scientific question

Does response-aware selection of an exactly size-matched scaffold-disjoint test set alter apparent QSAR benchmark difficulty under a molecular null in which the real molecules, scaffold geometry, group-size distribution, and endpoint marginal distribution are preserved, but the molecule–endpoint association is destroyed?

## Frozen design

- Datasets: BACE, BBBP, ClinTox, HIV, ESOL, and FreeSolv.
- Structural semantics:
  - classification: single-group treatment of acyclic molecules;
  - regression: single-group and singleton acyclic semantics.
- Twenty predeclared partition seeds are reused from the empirical audit.
- Two hundred independent endpoint permutations are used per dataset.
- The same endpoint permutation is reused across partition seeds and, for regression, across acyclic semantics.
- Candidate pools are target blind and nested by requested draw index.
- The baseline minimizes realized test-size deviation.
- The response-aware counterpart minimizes train–test target-mean mismatch only among candidates with exactly the same realized test size.
- Candidate budgets are frozen in `simulation_protocol_v1.json`.
- Constant predictors use the training-set mean (regression) or training-set prevalence (classification).

## Inferential unit

Within each endpoint permutation, effects are first averaged across the 20 partition seeds. The 200 endpoint permutations—not the 4,000 seed–permutation rows—are the simulation replicates. Empirical 2.5th and 97.5th percentiles are reported across permutations.

## Primary mechanistic checks

1. RMSE/MSE coupling and its decomposition into test-variance and squared mean-gap components.
2. Candidate-budget dependence of the coupling effect.
3. Classification contrast across ROC–AUC, average precision, Brier score, and log loss.
4. Sensitivity of regression coupling to single-group versus singleton acyclic semantics.
5. Dataset-to-dataset variation induced by real scaffold group-size distributions.

## Fail-closed quality gates

- exact test cardinality within every pair;
- response-aware target gap no worse than its paired baseline;
- maximum MSE decomposition error no greater than `1e-10`;
- complete dataset × semantics × seed × permutation × budget cells;
- protocol and input hashes recorded in every result;
- resumable checkpoint blocks must pass provenance and row-count validation before reuse.

No primary result was inspected before this protocol and its hash were committed.
