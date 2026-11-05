# KMM-CP and Selective KMM-CP implementation

This module implements the complete post-prediction procedures from Laghuvarapu,
Deb and Sun, **KMM-CP: Practical Conformal Prediction under Covariate Shift via
Selective Kernel Mean Matching**, UAI 2026, PMLR 337:3237–3261.

- Final publication: <https://proceedings.mlr.press/v337/laghuvarapu26a.html>
- Final paper and supplements: <https://raw.githubusercontent.com/mlresearch/v337/main/assets/laghuvarapu26a/laghuvarapu26a.pdf>
- PDF SHA-256: `dbcc53ccd8a594b076f698f1fb2516cae701af803be8979eef3af788bbdae5ec`
- Earlier version: <https://arxiv.org/html/2603.26415v1>

The final PDF, including its supplements, was read for this implementation. The
algorithm equations and Appendix D/E implementation statements examined here
agree with arXiv v1. The referenced author repository returned HTTP 404 when
queried. Accordingly this is an **independent paper-based implementation**, not
author-code verification or a claim to have reproduced their published numbers.
All running code is included; no author repository is required to execute it.

## Implemented algorithm

The runner supplies held-out calibration probabilities, source labels and
penultimate feature vectors, together with target probabilities and features.
No target outcomes enter the fitting API. For a given target cohort the code:

1. Calculates the cross-domain median Euclidean distance. It considers RBF
   bandwidth multipliers `0.01, 0.1, 0.5, 1, 2` and chooses the largest
   permutation-standardized two-sample MMD statistic (Appendix E.5, p. 3257).
2. Solves ordinary bounded KMM using equations (5)–(6), pp. 3239–3240.
3. Solves the joint source-weight/target-selection quadratic program using
   equations (7)–(8), p. 3241. Source weights lie in `[0,30]`; target selection
   weights lie in `[0,1]`; their average masses agree within epsilon, and target
   selection mass is at least `tau=0.5`.
4. Retains targets with selection weight at least `0.2`, then **refits ordinary
   KMM against the retained targets**, as required by Algorithm 1, p. 3242.
   The joint source weights are not reused as final calibration weights.
5. Constructs empirical weighted LAC quantiles, separately for each requested
   alpha, with global and both explicit Mondrian interpretations below. This
   follows Algorithm 1's empirical quantile; it has no query atom at infinity
   and no `(n+1)` rank correction.

The KMM objective used by the solver is `0.5*w.T*Kss*w-kappa.T*w`, with
`kappa=(n/m)*Kst.sum(axis=1)`. For joint selection the quadratic matrix is
`[[Kss, -(n/m)*Kst], [-(n/m)*Kst.T, (n/m)**2*Ktt]]`, with zero linear term.
These are constant/scalar-equivalent versions of the paper objectives; no
ridge is inserted. OSQP's numerical internal regularization is a solver step,
not an added scientific penalty. The final solver status and constraint
residuals are checked. Failure raises an error; there is no substitute baseline
or hidden subsampling.

## The paper's Mondrian inconsistency is exposed

Appendix D describes calibration by true class. Algorithm 2, lines 17–18
(p. 3255), nevertheless applies the **predicted class's threshold to all
candidate labels**. These are different algorithms. Both are implemented and
named separately; neither may silently replace the other:

| Suffix | Calibration | Query threshold |
|---|---|---|
| `global` | All true-label scores | One global threshold |
| `mondrian_candidate` | True class subsets | Candidate label `y` uses `q_y` |
| `mondrian_paper_predicted` | True class subsets | Every candidate uses `q_argmax(p)` |

`mondrian_candidate` is the proper candidate-label interpretation for the
study's class-conditional comparison. `mondrian_paper_predicted` is a literal
pseudocode fidelity control; no class-conditional validity claim is made for
it. A missing or zero-weight calibration class receives an infinite threshold
and is explicitly recorded. Exact probability ties select class 0.

## Fixed choices where the paper is underspecified

The final paper does **not** specify a numerical epsilon, the permutation
count/seed, the MMD estimator's bias convention, numerical solver tolerances,
or behavior for zero pairwise distances. The runnable defaults below are
explicit implementation choices, not purported author settings:

| Item | Locked implementation |
|---|---|
| Mass tolerance epsilon | 0: exact mean-mass constraints |
| RBF definition | `exp(-distance**2/(2*sigma**2))` |
| Feature scaling | None for the paper's penultimate representations; optional source-only standardization is separately recorded |
| MMD statistic | Unbiased squared two-sample MMD |
| Permutations | 100, preserving original domain sizes; seeded and shared across candidates |
| Standardization | `(observed MMD² - permutation mean) / permutation SD`, SD uses `ddof=1` |
| Bandwidth tie | First candidate in the reported grid |
| Zero median distance | Use positive-distance median; if all distances are zero, base bandwidth 1 |
| Zero null variance | Exclude that candidate from z-score comparison; if all are degenerate, choose the multiplier nearest 1 and record it |
| Refit bandwidth | Keep the selected kernel, consistent with Algorithm 1's single supplied kernel `k` |
| Solver | OSQP 1.0.4 in place of the reported CVXOPT solver; actual version and lock match recorded |
| Solver tolerances | Absolute/relative `1e-7`; maximum 100,000 iterations; constraints validated within `5e-6` |
| Input cap | 10,000 total rows, configurable and reported; exceeding it errors without subsampling |

The mean selection-mass constraint `mean(a)>=0.5` is **not** an assertion that
exactly 50% of target rows will pass the `a>=0.2` threshold. The realized
retention fraction and per-class retention must be reported. There is no
automatic top-k replacement if thresholding retains nothing.

## Backbone and experimental interpretation

Appendix E.3–E.4, pp. 3256–3257, uses AttentiveFP with source-training labels:
512-dimensional graph embeddings, a 64-dimensional classifier hidden layer,
three GNN steps, two outputs, Adam with learning rate `0.001`, StepLR gamma
`0.8` each epoch, at most 50 epochs, batch size 128, training-loss patience 5,
and inverse-frequency cross-entropy weights. The paper does not fully specify
every architecture detail despite reporting a total parameter count; an
independent implementation must record its actual dimensions/count rather
than asserting the number matches. The orchestration records the actual
backbone and supplies its penultimate embeddings to this module. Applying this
module to the common classical stack is an explicitly matched-backbone
comparison, not an AttentiveFP reproduction.

`backbone_config()` provides the exact trainer keys for this run: graph size
512, classifier hidden sizes `[64]`, three message-passing layers, and two
readout iterations. The last value is a disclosed framework default because
the paper does not state the number of readout iterations. The returned
64-dimensional penultimate classifier embedding is used for KMM; it is kept
distinct from the 512-dimensional graph embedding. The shared trainer runs
without energy regularization or auxiliary molecules and skips the unnecessary
KDE fit when `compute_conformal=False`.

## Output and selected-population reporting

`fit_predict(cal_x, cal_y, cal_p, target_x, target_p, alphas, seed, config)` returns:

- `predictions[alpha][method]`: all target rows. Rejected SKMM rows are `{0,1}`
  as a study-level defer policy.
- `raw_predictions[alpha][method]`: raw quantile predictions. For rejected rows
  these are extrapolations for traceability, not accepted predictions.
- `method_selected[method]`, `selected`, and `selection_weights`: explicit
  selection decisions; all KMM rows are selected.
- `source_weights`: final ordinary KMM, final selective-refit KMM, and initial
  joint-selection weights, kept distinct.
- `diagnostics`: bandwidth candidates/null statistics, solver results, realized
  retention, thresholds, class support, configuration and source identity.

Report selected-only coverage with its selected denominator, plus full-cohort
deferral, singleton yield, false-clearance exposure, and per-class retention.
Do not compare selected-only SKMM coverage to another method's full-cohort
coverage as though the populations were equal. `evaluate_selection` returns
both populations explicitly. An external label mask must be applied uniformly
before evaluating either method; selection itself remains label-free.

## Verification and execution

Install `osqp==1.0.4` with the revision environment, then run from repository
root:

```bash
python -m unittest paper2_admet_benchmark.revision_2026.full_baselines.test_kmm_cp -v
```

The synthetic checks exercise the unbiased MMD calculation against explicit
pair sums; deterministic permutation selection; QP identity solutions;
joint-selection constraints and low-overlap filtering; threshold refitting;
both Mondrian rules; empirical-quantile semantics; class absence; target-free
selection invariance to calibration-label permutation; failure handling; and
separate retained/full-population denominators. These checks establish
implementation behavior, not real-data performance or a new coverage theorem.
