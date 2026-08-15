# RACER-C3 algorithm specification v0.1

Status: **retrospective architecture candidate that has not passed its freeze
gate; no confirmatory test prediction is authorized**.

RACER-C3 is a new algorithm version in a separate namespace. It does not alter
the completed RACER-C v1 results, and it does not relabel the weak RACER-C2
reliability tilt as a positive result. The v1 predicted-class gate remains a
completed negative confirmatory result. RACER-C2 remains a documented failed or
weak prototype whose incremental retrospective effect was too small and mixed
to justify primary status.

## 1. Problem and design target

For binary conformal classification, a correct singleton for class 0 requires
including candidate 0 while excluding candidate 1; the reverse holds for class
1. A single probability or a symmetric risk transform therefore forces two
different decisions to share one ranking. The v1 panel showed that this is most
restrictive under scaffold extrapolation, where minority support and structural
reliability fail differently.

RACER-C3 targets full-population **Macro Correct-Singleton Yield (MacroCSY)**
subject to class-conditional coverage. It does not improve actionability by
deleting difficult rows. Every molecule receives one of four conformal set
states: `{0}`, `{1}`, `{0,1}`, or the empty set.

## 2. New mechanism: candidate-specific experts

Let the five honest probability views be ECFP, D-MPNN, MoLFormer, the restricted
stacker, and the unrestricted stacker. Let `r(x)` be the fixed development-
reference Brier-risk percentile.

### Candidate 0 expert

The candidate-0 expert first forms a robust probability for class 1:

`p_R(x) = sigmoid(mean(logit(p_ECFP), logit(p_DMPNN), logit(p_MoLFormer)))`.

Its support for candidate 0 is `c_0(x)=1-p_R(x)`. This expert avoids allowing a
single high-capacity view or the unrestricted reliability feature to dominate
the non-critical candidate ranking at a new scaffold frontier.

### Candidate 1 expert

A shared candidate-correctness model is trained on two development rows per
molecule: one row asks whether candidate 0 is correct and the other asks whether
candidate 1 is correct. The target is `1[Y=y]`. Features include candidate-wise
logits from all five views, robust summaries of the three base views, BRI,
disagreement, ECFP distance, local honest-OOF Brier loss, risk percentile,
candidate nonconformity interactions, and the candidate-label indicator.

The estimator is L2 logistic regression with equal total weight in each of the
four `(candidate label, correctness)` cells. It is cross-fitted by development
meta-fold. External correctness is the mean of the fold-model predictions. The
candidate-1 support is the resulting `c_1(x)`; no model is fitted to policy,
conformal, or deployment labels.

### Risk tempering and nonconformity

For each candidate `y`, support is tempered continuously:

`c_y^T(x) = sigmoid(logit(c_y(x)) / (1 + (T_y-1) r(x)))`,

with `T_0=T_1=1.5`. The frontier nonconformities are

`s_y^F(x)=1-c_y^T(x)`.

The two candidate scores need not sum to one. This is deliberate: conformal
prediction requires a fixed score for each candidate label, not a normalized
probability vector. No step branches on `p>=0.5`.

## 3. Batch-symmetric chemical-frontier route

RACER-C3 uses the frontier experts only when an unlabeled structural audit shows
true scaffold extrapolation. Let `D` be development and let `U` be the union of
the independent conformal covariates and the deployment-batch covariates.

The route uses two permutation-invariant summaries of `U`:

1. the molecule-weighted fraction of valid `U` scaffolds also present in `D`;
2. the median ECFP distance to the allowed development reference.

The frontier route is active only when the overlap fraction is at most `0.05`,
the median distance is at least `0.57`, and at least 100 valid union rows are
available. The thresholds were chosen from unlabeled separation of the three v1
track constructions, not from target outcomes. Missing or insufficient inputs
fail closed to the fallback.

The conformal and deployment covariates enter only through their unordered
union. This symmetry is essential: a test-only drift gate would generally break
the rank symmetry used by conformal calibration.

## 4. Exact fallback

When the frontier route is inactive, RACER-C3 exactly reproduces the v1 RACER
score without its failed gate:

`p_B(x)=sigmoid(logit(p_stack(x))/(1+0.5 r(x)))`,

`s_0^B(x)=p_B(x)` and `s_1^B(x)=1-p_B(x)`.

The route chooses either the complete frontier score matrix or the complete
fallback matrix. It never rejects rows. On ordinary or insufficiently proven
frontiers, RACER-C3 therefore cannot change the parent no-gate prediction sets
apart from numerical equality tolerance.

## 5. Mondrian calibration

After the candidate-correctness experts and the unlabeled route are fixed, the
independent conformal role calculates

`q_y = Quantile_{ceil((n_y+1)(1-alpha))} {s_y(X_i):Y_i=y}`

The exact fallback uses `alpha_0=alpha_1=0.10`. At an active frontier,
RACER-C3 uses `alpha_0=0.10` and the more conservative
`alpha_1=0.095` for the predeclared critical class. This correction was added
after the first retrospective candidate exposed a critical-class coverage
shortfall; it protects rather than spends critical-class miscoverage. The
prediction set is

`Gamma(x)={y:s_y(x)<=q_y}`.

Under class-conditional exchangeability and fixed development-trained experts,
ordinary Mondrian coverage follows for a fixed route. For the batch-dependent
route, the intended validity argument additionally requires that the route be a
permutation-invariant function of the conformal-plus-query covariate union. A
formal proof and independent statistical review are mandatory before a theorem
claim. RACER-C3 does not claim distribution-free coverage under arbitrary
chemical shift.

## 6. What is and is not the contribution

RACER-C3 does not claim to invent Mondrian conformal prediction, learned
nonconformity, score aggregation, alpha allocation, covariate-weighted
conformal prediction, selective risk control, or batch conformal prediction.

The proposed contribution is the joint mechanism:

1. candidate labels are routed to different reliability experts rather than
   receiving a symmetric transform of one probability;
2. the positive-candidate score estimates candidate correctness directly while
   the negative candidate uses a robust multi-view structural score;
3. a label-free, batch-symmetric chemical-frontier audit activates the
   asymmetric score only for genuine structural extrapolation;
4. a mathematically exact parent-score fallback preserves ordinary-domain
   behavior; and
5. the development objective and final certificate are expressed in actual
   class-specific singleton/set states over the full population.

This combination is a defensible research hypothesis, not a guarantee of legal
or scholarly novelty. A full pre-submission search of papers, code, patents,
and priority dates remains required.

## 7. Retrospective signal and its limitation

The complete four-endpoint v1 panel was already open when RACER-C3 was designed.
It is therefore architecture-development evidence only. With frontier experts
active on the strict-scaffold track and the exact parent fallback elsewhere,
the observed mean MacroCSY difference versus `RACER_score_no_gate` was:

| Track | Cells | Mean delta (percentage points) | Route |
|---|---:|---:|---|
| strict scaffold | 20 | +0.2097 | frontier |
| similarity cluster | 20 | 0.0000 | exact fallback |
| grouped random | 20 | 0.0000 | exact fallback |
| all tracks | 60 | +0.0699 | routed |

On strict scaffold, the coverage-protected frontier candidate won 10 and lost
10 cells. Mean class-0 coverage increased by 0.4272 points and mean class-1
coverage decreased by 0.5736 points; class-0 CSY decreased by 0.7037 points and
critical-class CSY increased by 1.1232 points. An endpoint-cluster bootstrap
interval for MacroCSY crossed zero even before correcting for architecture
search. The architecture was selected after the outer v1 outcomes were known;
it has therefore **not passed the freeze gate** and cannot authorize a
superiority claim.

## 8. Required prospective test

The next test must use genuinely untouched endpoints, but it should not begin
until the weak and mixed retrospective effect is judged worth the cost or a
different mechanism passes a new label-firewalled development gate. Before any
test target is opened, it must freeze endpoint provenance, licenses, exact label
polarity, chemical grouping, role counts, precision, seeds, comparators,
failure policy, the score implementation, and the route thresholds.

Required comparators include marginal and Mondrian LAC, weighted/CoDrug-style
covariate-shift CP, RC3P, SOCOP, SCRC, SCoRE, confidence/similarity/disagreement
rejection, the v1 no-gate score, all RACER-C3 component ablations, and a score-
aggregation method such as COLA when access and estimands are compatible.

The strict-scaffold frontier is primary; similarity cluster is secondary; the
grouped-random track is a negative control. All failed endpoints, seeds, and
certificates remain in the denominator.

## 9. Mandatory implementation tests

1. Fallback scores exactly reproduce v1 no-gate attenuation.
2. Route choice is invariant to row order in the conformal-plus-batch union.
3. Route construction has no target-label argument.
4. Missing/short frontier audits choose the fallback.
5. Candidate-correctness predictions are complete honest OOF predictions.
6. Candidate 0 and candidate 1 scores can change independently.
7. No score branches on a hard predicted class.
8. Test-target permutation changes no model, route, score, or conformal
   threshold.
9. The finite-sample quantile, including `+infinity`, matches hand calculations.
10. Every query receives a set state; no confidence or domain gate deletes it.
11. Known v1 outputs are read-only and C3 writes only to a new namespace.
12. Prospective execution is impossible without a new user-approved freeze tag.
