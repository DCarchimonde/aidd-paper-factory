# RACER-C2 algorithm specification v0.2

Status: **development only; not frozen; no prospective test prediction is
authorized**.

RACER-C2 is a new method version, not a repair or relabelling of RACER-C v1.
The v1 60-cell/540-method run remains immutable and retains its negative
confirmatory conclusion: the predicted-class policy gate was infeasible in
every primary cell.

## 1. Design problem exposed by v1

V1 first assigned a hard class at stacked probability 0.5 and then routed a row
to a risk threshold indexed by that predicted class. For a rare critical class,
a false negative had therefore already entered the non-critical branch. The
gate could reject that decision, but could not add critical-class support or
turn the row into an explicit two-label set. Requiring that same gate to retain
at least half of the critical class while certifying at most 10% base-classifier
error was incompatible with the observed stacker.

RACER-C2 removes predicted-class routing. It constructs and calibrates one score
for candidate label 0 and a second score for candidate label 1. Reliability can
change the two candidate scores in different directions before the conformal
set is formed.

## 2. Honest inputs and firewall

The development implementation reuses the honest v1 prediction lineage:

- calibrated ECFP, D-MPNN, MoLFormer, stacked, and unrestricted probabilities;
- heterogeneous disagreement;
- ECFP distance to the allowed development reference;
- local honest-OOF Brier loss; and
- the v1 Brier risk index and its fixed development percentile.

Only development-role labels may be materialized by the v2 selector. V1 policy,
conformal, and test labels are not inputs. Because the v1 test outcomes are
already known to the investigators, all four v1 endpoints and seeds 101--105
are retrospective development evidence. The known v1 panel can never become its confirmatory panel.

## 3. Candidate-label reliability tilting

Let `p_stack(x)` be the honest stacked probability for label 1 and `r(x)` the
fixed development-reference reliability-risk percentile in `[0,1]`. First apply
the v1 continuous attenuation for candidate `T_max`:

`p_T(x) = sigmoid(logit(p_stack(x)) / (1 + (T_max - 1) r(x)))`.

The un-tilted candidate-label nonconformities are

`a_0(x) = p_T(x)` and `a_1(x) = 1 - p_T(x)`.

RACER-C2 then introduces **candidate-label exponential reliability tilting**:

`s_y(x) = a_y(x) exp(gamma_y r(x)),  y in {0,1}`.

A positive `gamma_y` penalizes inclusion of candidate `y` as reliability risk
rises; a negative value protects its inclusion. The exponential form is
strictly positive, continuous, monotone in risk for fixed `a_y`, and has the
exact identity element `gamma_y=0`.

For the toxicity/inhibition setting, label 1 is predeclared as the critical
class. The finite development family therefore uses `gamma_0 >= 0` and
`gamma_1 <= 0`. This can convert a high-risk non-critical singleton into an
ambiguous set or a critical singleton without using the hard rule `p >= 0.5`.
It does not force either outcome: membership is still determined by the two
class-conditional conformal thresholds.

## 4. Safe finite family and development selection

The finite family is the Cartesian product of:

- `T_max in {1.0, 1.5, 2.0}`;
- `gamma_0 in {0.0, 0.1, 0.25}`; and
- `gamma_1 in {-0.1, 0.0}`.

It contains two explicit safe fallbacks:

- for any `T_max`, `(gamma_0,gamma_1)=(0,0)` exactly reproduces the corresponding
  v1 RACER score without its failed gate; and
- `(T_max,gamma_0,gamma_1)=(1,0,0)` exactly reproduces stacking-Mondrian.

One global configuration is selected from honest development OOF sets by
endpoint/cell-equal mean MacroCSY. For each class, the cell-equal mean coverage
may be at most 1 percentage point below stacking-Mondrian and no individual
development cell may have empirical class coverage below 85%. The mean
non-inferiority rule avoids treating a one-cell finite-sample fluctuation as a
global veto; the absolute floor remains fail-closed. Ties prefer zero or smaller
total absolute tilt and then smaller `T_max`. Endpoint-specific or test-aware
choices are prohibited.

The primary selection tracks are strict scaffold and similarity cluster because
the method's scope is chemical distribution shift. Grouped random splitting is
a declared negative-control track, not another source of votes in the selection
objective. This scope decision is based on retrospective v1 development and
must be validated on untouched endpoints.

## 5. Rejected counterfactual-model ablation

An earlier v2 draft fitted two monotone gradient-boosted candidate-error indices
to `1[Y != y]`, cross-fitted by development meta-fold, and blended their fixed
reference percentiles into `a_y`. Honest development selected blend weight zero.
The learned module therefore did not earn entry into the primary algorithm.

Its pure numerical implementation remains available for a transparent rejected
ablation, but the v0 development lock fixes `counterfactual_blend=0`. It may not
be silently revived after prospective outcomes are seen.

## 6. Class-conditional conformal set

After the global score configuration is fixed, an independent conformal role
calculates, for each true label `y`:

`q_y = Quantile_{ceil((n_y+1)(1-alpha))} {s_y(X_i): Y_i=y}`.

The prediction set is

`Gamma(x) = {y: s_y(x) <= q_y}`.

The four observable states are:

- `{0}` -> `Accept(0)`;
- `{1}` -> `Accept(1)`;
- `{0,1}` -> `Ambiguous`; and
- the empty set -> `Defer-empty`.

There is no predicted-class risk gate between score construction and these
states. Under class-conditional exchangeability, this is ordinary Mondrian
split-conformal validity applied to the fixed learned score. Scaffold and
similarity-cluster tracks are empirical shift tests; RACER-C2 does not claim
distribution-free coverage under arbitrary chemical shift.

## 7. Direct action certificate

V1 certified a pre-conformal proxy: base hard-classification error inside a
searched gate. RACER-C2 instead evaluates the actual final set states on an
independent policy role after the score and conformal thresholds are fixed.

Candidate constraints are:

- lower confidence bounds for class-specific truth-containing coverage;
- upper confidence bounds for class-specific wrong-singleton exposure;
- upper confidence bounds for class-specific empty-set exposure; and
- a lower confidence bound for critical-class correct-singleton yield when the
  pre-freeze precision audit supports a numerical floor.

Only enabled constraints enter the simultaneous correction. The policy stage
tests one fixed configuration and no gate-threshold grid, so
`selection_grid_test_count=0`. Failure is retained and reported; constraints
cannot be relaxed after policy outcomes.

The coverage, empty-set, and critical-CSY numerical floors remain pending a new
count/precision audit. They must not be inferred from the known v1 results.

## 8. Methodological contribution and boundaries

RACER-C2 does not claim to invent conformal prediction, Mondrian calibration,
one-sided selective classification, Learn-Then-Test, conformal risk control, or
selective conformal risk control. Its proposed methodological contribution is:

1. a continuous, label-asymmetric reliability allocation inside candidate-label
   nonconformity scores;
2. an exact identity fallback to a strong stacking-Mondrian baseline;
3. removal of hard predicted-class routing before set construction; and
4. exact certification of final class-specific set states rather than a
   pre-conformal hard-classification proxy.

This is a candidate algorithmic contribution until untouched prospective data
show improved actionability without loss of class-specific reliability. The
current evidence can support method development, not a superiority claim or a
claim of being the first reliability-weighted conformal method.

### Retrospective additive evaluation

`scripts/racer_c2/run_retrospective_extension.py` applies the one globally
selected configuration to the completed v1 60-cell artifacts. It recalibrates
only the fixed C2 candidate scores on each existing conformal role and evaluates
the final sets using the already unsealed v1 test labels. It does not retrain a
base learner or rerun any of the 540 parent-method results. The fixed comparison
family is stacking-Mondrian, the v1 no-gate reliability score, both one-label
tilt ablations, and full RACER-C2. These outputs are post-hoc development
evidence regardless of their numerical result.

## 9. Required prospective evaluation

The known Tox21 v1 panel may be used to reject designs and estimate compute only.
Primary v2 validation requires endpoints whose outcomes have not been opened,
with frozen provenance, license, label semantics, grouping, precision,
comparators, and seeds before test prediction. The large Veith CYP datasets are
only candidates: their distributed inhibitor polarity is supported, but claims
must remain source-defined unless the original binary transformation is fully
recovered.

Comparators should include stacking-Mondrian, v1 RACER score without gate,
unrestricted reliability stacking, one-sided prediction, SCRC, SCoRE, and a
class-wise conformal method such as RC3P where access regimes and estimands
match. The IID/random track remains a negative control for a shift-targeted
method.

## 10. Mandatory tests

1. `(gamma_0,gamma_1)=(0,0)` reproduces the un-tilted score exactly.
2. Positive and negative tilts change only their declared candidate label and
   direction as reliability risk rises.
3. Candidate scores never branch on a hard predicted class.
4. Every fitted development component is out-of-fold.
5. External percentiles are invariant to the external batch.
6. Test-label permutation leaves every fitted object, score, threshold, and hash
   unchanged.
7. Conformal quantile edge cases, including `+infinity`, match hand calculations.
8. Action-certificate denominators use true-class members and final set states.
9. Disabled numerical constraints are not silently counted in multiplicity.
10. Selection is deterministic under row/configuration permutation.
11. Only predeclared shift tracks vote in configuration selection.
12. V1 result paths are read-only and v2 writes use a separate namespace.

## 11. Primary references for method boundaries

- Gangrade, Kag, and Saligrama. *Selective Classification via One-Sided
  Prediction*. AISTATS 2021. https://proceedings.mlr.press/v130/gangrade21a.html
- Angelopoulos et al. *Learn then Test: Calibrating Predictive Algorithms to
  Achieve Risk Control*. https://arxiv.org/abs/2110.01052
- Xu et al. *Selective Conformal Risk Control*.
  https://arxiv.org/abs/2512.12844
- Bai and Jin. *Conformal Selective Prediction with General Risk Control*.
  https://arxiv.org/abs/2603.24704
- Shi et al. *Conformal Prediction for Class-wise Coverage via Augmented Label
  Rank Calibration*. https://arxiv.org/abs/2406.06818
- Laghuvarapu, Lin, and Sun. *Conformal Drug Property Prediction with Density
  Estimation under Covariate Shift*. https://arxiv.org/abs/2310.12033
