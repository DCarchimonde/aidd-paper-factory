# RACER-C algorithm specification v0.1

Status: **pre-freeze draft**. Normative words (`MUST`, `MUST NOT`, `SHOULD`) apply
to the future implementation. The scientific protocol takes precedence if a
conflict is found.

## Inputs

- standardized molecular structure and immutable `structure_id`;
- binary endpoint label for development/policy/conformal roles only;
- group identifiers for identical structure, Murcko scaffold, and similarity
  cluster as required by the evaluation track;
- a role manifest created before modelling;
- three predictor blocks: ECFP ensemble, Chemprop D-MPNN, frozen MoLFormer head.

No function used to fit or choose a component may read `D_test.y`.

## Numerical conventions

- All probability arrays MUST be finite float64 at calibration/stacking time.
- Clip probabilities to `eps=1e-6` before `logit`; record counts by model/role.
- Ties in empirical CDFs use mid-ranks and division by development-reference size.
- Tanimoto ties sort by `structure_id`; nearest-neighbour results MUST be stable.
- Quantiles use `k=ceil((n+1)(1-alpha))`, 1-indexed; `k>n` returns `+inf`.

## Development lineage

Create three grouped meta-folds. For held-out fold `f`:

1. assign the remaining two folds as the inner development pool;
2. generate inner OOF raw predictions for every base learner;
3. fit one Platt calibrator per learner on inner OOF predictions;
4. form calibrated learner probabilities;
5. form ECFP block probability by uniform mean of the four ECFP probabilities;
6. fit the L2 stacker from the three block logits using data that exclude `f`;
7. obtain honest stacked probabilities and Brier losses for BRI references;
8. build disagreement, ECFP distance, and local-loss features without `f`;
9. fit the constrained BRI without `f` and predict `f`;
10. save predictions plus fit lineage for `f`.

The implementation MUST assert that a row's `structure_group_id` is absent from
every fit set in its lineage. A second assertion MUST show that policy, conf, and
test identifiers never occur in a development fit lineage.

## Base blocks

For calibrated learner probabilities `p_j`:

`p_ECFP = mean(p_LR, p_RF, p_XGB, p_MLP)`

`l_b = logit(clip(p_b, eps, 1-eps))`, where
`b in {ECFP, DMPNN, MoLFormer}`.

The stacker is L2 logistic regression:

`m(x) = beta_0 + sum_b beta_b l_b(x)`

and `p_stack=sigmoid(m)`. The sign of `m` defines the base class. Reliability
features MUST NOT directly flip that sign.

## Reliability vector

`D_hetero = variance([l_ECFP,l_DMPNN,l_MoLFormer], ddof=0)`.

`d_ECFP = 1 - max_j Tanimoto(fp_x, fp_j)` over the allowed development reference
pool after excluding the query's standardized-structure group.

`L_local = mean_j (y_j-p_stack_j)^2` over the `k=min(20,n_ref)` closest reference
structures. Reference probabilities and losses MUST be honest OOF values.

Latent distances, entropy, and aleatoric/epistemic labels are not primary inputs.

## Balanced Brier Risk Index

Target: `b_i=(y_i-p_stack_i)^2`.

Features: `z=[|m|,D_hetero,d_ECFP,L_local]`. Continuous feature transforms and
knots, if any, MUST be fitted from development only and serialized.

Class weights are `w_y=n/(2*n_y)`. Monotonic directions are
`[-,+,+,+]`. The frozen implementation MUST be low capacity and deterministic;
its exact solver, penalties, basis dimension, clipping, and convergence rule are
written to the environment lock before execution.

The output `u` is the Balanced Brier Risk Index. It MUST NOT be labelled a
probability of error. Honest held-out `u_i` values define predicted-class-specific
development CDFs. For an external query with predicted class `c`:

`r(x)=F_hat_c(u(x))`.

## Attenuation and scores

For globally frozen `T_max`:

`T(x)=1+(T_max-1)r(x)`

`eta(x)=sigmoid(m(x)/T(x))`

`s_1(x)=1-eta(x)` and `s_0(x)=eta(x)`.

This is attenuation-only and preserves the base class sign. Variable attenuation
changes score ordering and therefore can change conformal efficiency; subsequent
calibration, not the transformation itself, supplies conformal validity under the
relevant exchangeability condition.

## Gate

For predicted class `c`, select if `r(x)<=t_c`. Candidate thresholds are fixed at
`0.50,0.60,0.70,0.80,0.90,1.00`. Policy selection uses the confidence-bound and
tie-break procedure in the protocol. Store every candidate's counts, estimates,
bounds, feasibility flags, and rejection reasons.

## Conformal calibration

For each true class `y`:

- full threshold `q_all_y` is calibrated on all conformal-role scores with `Y=y`;
- selected threshold `q_sel_y` is calibrated only where `Y=y` and the frozen gate
  selects the observation.

Prediction sets include label `y` when its candidate-label score is no greater
than the corresponding threshold. Empty and two-label sets are valid possible
outputs. Infinite thresholds include that label for all queries.

## Required artefacts

- endpoint and role manifests with hashes;
- per-row fold and fit lineage;
- raw, calibrated, block, stacked, BRI, percentile, gate, and conformal outputs;
- calibrator/stacker/BRI coefficients and transforms;
- nearest-neighbour IDs and similarities for audit samples;
- policy grid with simultaneous bounds;
- conformal cell counts and order statistics;
- software/model/checkpoint hashes;
- failed-run and deviation logs.

## Mandatory unit tests

1. standardized duplicates never cross roles or folds;
2. changing one held-out label cannot change an artefact allegedly trained without it;
3. test-label permutation leaves every prediction, threshold, gate, and model hash unchanged;
4. self-neighbours and grouped duplicates are excluded as specified;
5. external percentiles do not change when the external batch changes;
6. probability clipping and logit transforms are finite;
7. conformal quantile edge cases match hand calculations, including `+inf`;
8. no attenuation setting changes the base class sign;
9. policy selection is deterministic under row permutation;
10. all method-access contracts are enforced.

## Non-claims

The implementation does not by itself guarantee conditional coverage, singleton
accuracy, safe decisions, minority retention, fairness, or validity under arbitrary
chemical shift. `Accept(y)` means a gate-selected singleton output only.
