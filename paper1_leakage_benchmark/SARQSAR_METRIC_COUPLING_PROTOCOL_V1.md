# SAR/QSAR Metric-Coupling Null Simulation Protocol v1.0

**Status:** frozen before any null-simulation result is generated  
**Target manuscript:** *Split-Objective–Metric Coupling in QSAR Benchmarks: Null Simulations and Exact-Size Paired Audits*  
**Target journal:** *SAR and QSAR in Environmental Research*  
**Protocol version:** 1.0  
**Master simulation seed:** 20260820

## 1. Scientific question

This study tests whether selecting scaffold-disjoint test partitions with endpoint information can couple the split-selection objective to the downstream evaluation metric and alter apparent QSAR benchmark difficulty before molecular features contribute predictive information.

The null simulation preserves each real cleaned molecular universe, its Bemis–Murcko scaffold geometry, scaffold-group sizes, endpoint marginal distribution, exact-size pairing rule, partition seeds, and target-blind candidate-generation algorithm. It destroys only the structure–endpoint association by independently permuting endpoint values across molecules.

## 2. Frozen datasets and task types

Classification:
- BACE
- BBBP
- ClinTox
- HIV

Regression:
- ESOL
- FreeSolv

The authoritative inputs are the existing `clean_v2` files used by the frozen Paper 1 audit. Their SHA-256 values are recorded at runtime.

## 3. Scaffold semantics

Primary null simulation:
- `single_group`: all acyclic molecules share `__ACYCLIC__`.

Regression sensitivity:
- `singleton`: each acyclic molecule receives a molecule-specific scaffold identity.

Classification is simulated under `single_group` only. ESOL and FreeSolv are simulated under both modes using identical endpoint permutations, permitting paired interpretation across scaffold semantics.

## 4. Candidate generation and nested search budgets

For each dataset, scaffold mode, and one of the 20 frozen partition seeds, random scaffold-prefix candidates are generated without reading endpoint values.

The maximum requested draw budget is generated once. Each unique candidate records the requested draw on which it first appeared. A budget prefix contains every unique candidate first observed by that requested draw. Duplicate candidate draws consume budget and are not silently replaced.

Frozen requested-draw budgets:

Classification:
- 10
- 30
- 100
- 300

Regression:
- 100
- 300
- 1,000
- 3,000
- 5,000
- 10,000
- 20,000

For each budget:
1. the size-matched baseline minimizes test-size deviation without endpoint information;
2. ties are resolved deterministically from the partition seed and candidate hash;
3. the response-aware candidate is selected only among candidates with exactly the same realized test size as the baseline;
4. it minimizes the absolute train–test endpoint-mean gap under the permuted endpoint;
5. ties are resolved deterministically.

The candidate pool, scaffold rule, requested draw budget, partition seed, and realized test size are therefore paired. Molecular and scaffold composition are not assumed to be controlled.

## 5. Endpoint permutations

For each dataset, 200 independent endpoint permutations are generated from master seed 20260820.

- Molecular rows and scaffold identities remain fixed.
- The endpoint marginal distribution is preserved exactly.
- Structure–endpoint association is destroyed.
- The same permutation is reused across partition seeds.
- For ESOL and FreeSolv, the same permutation is reused across `single_group` and `singleton` modes.

The inferential simulation replicate is the endpoint permutation. The 20 partition seeds are averaged within each permutation and are not treated as 20 independent simulation replicates.

## 6. Null predictors and metrics

### Regression

Every test molecule receives the training-set endpoint mean.

Reported metrics:
- RMSE
- MSE
- MAE
- R-squared

The identity

`MSE = Var(y_test) + (mean(y_test) - mean(y_train))^2`

is evaluated for every partition. Positive effects favor the response-aware split:
- error effects = size-matched minus response-aware;
- R-squared effect = response-aware minus size-matched.

The MSE effect is decomposed into:
- test-variance effect;
- squared-mean-gap effect.

### Classification

Every test molecule receives the training-set positive-class prevalence as a constant probability.

Reported metrics:
- ROC-AUC
- average precision
- Brier score
- log loss

Positive effects favor the response-aware split:
- AUC/AP effects = response-aware minus size-matched;
- Brier/log-loss effects = size-matched minus response-aware.

ROC-AUC is recorded as undefined when a test partition contains only one class; it is never imputed. The undefined rate is reported.

## 7. Frozen estimands

Primary simulation estimands:
1. mean null RMSE effect by regression dataset, scaffold mode, and budget;
2. mean squared-mean-gap and test-variance contributions to the null MSE effect;
3. mean null Brier and log-loss effects by classification dataset and budget;
4. mean null ROC-AUC effect where defined;
5. budget dependence of all effects;
6. differences between single-group and singleton acyclic semantics;
7. frequency with which response-aware and size-matched selections are the same partition.

## 8. Aggregation and uncertainty

For every dataset, mode, budget, permutation, and metric:
1. effects are averaged across the 20 partition seeds;
2. the 200 permutation-level means form the simulation distribution;
3. the mean, standard deviation, median, 2.5th percentile, 97.5th percentile, valid count, and fraction above zero are reported.

No p-value family is added to the original empirical hypothesis family. Null simulations are mechanistic evidence and are reported separately.

## 9. Quality gates

The run fails if any of the following occurs:
- missing or malformed clean-v2 data;
- non-binary classification target;
- incomplete nested candidate prefixes;
- failure of exact test-size pairing;
- response-aware target gap exceeding its paired baseline;
- MSE decomposition residual above 1e-9;
- nonzero constant-score AUC effect beyond numerical tolerance where both AUCs are defined;
- missing dataset/mode/seed/budget/permutation cells;
- protocol/config hash mismatch with an existing result directory.

Seed-level checkpoints are immutable under the same protocol and input hashes. Interrupted runs resume from validated checkpoints.

## 10. Reporting checklist

The revised manuscript will include a formal minimum-reporting checklist covering:
- molecular identity and duplicate/conflict policy;
- disconnected-component policy;
- scaffold definition and acyclic semantics;
- whether endpoint information enters split generation or selection;
- requested and realized test size;
- candidate-generation algorithm and search budget;
- trivial response-only controls;
- collateral target and scaffold diagnostics;
- number of unique partitions;
- inferential unit and multiplicity policy;
- stochastic-model sensitivity;
- machine-readable manifests, hashes, software versions, and code release.

## 11. Change control

Any substantive change after results are generated requires:
- a new protocol version;
- a new output directory;
- explicit disclosure of the change and its timing.

The v1.0 results will not be deleted or overwritten to improve the narrative.
