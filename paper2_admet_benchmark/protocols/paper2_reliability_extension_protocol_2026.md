# Paper 2 prospectively frozen reliability extension protocol

Version: **draft 0.1 (not frozen)**  
Date opened: 2026-08-01  
Base commit: `a1d5f71790cfc7f668fee5455b2be8f88267c5ea`  
Branch: `paper2-reliability-extension-2026`

## 1. Status and firewall

This document governs only the new reliability extension. It does not retroactively
preregister the original four-endpoint results and does not replace or overwrite
their frozen assets. No extension test prediction may be generated before:

1. endpoint Freeze 0 and eligibility Freeze 1 are committed;
2. cleaning, role allocation, precision, and method-access audits pass;
3. exact software/model revisions and random seeds are committed;
4. leakage tests pass; and
5. the user approves the formal protocol tag.

All work before that tag is development. Deviations after the tag are recorded in
`protocol_deviations.md` before affected results are inspected whenever possible.

## 2. Scientific objective

The extension asks whether heterogeneous model disagreement, ECFP chemical-domain
support, and local honest OOF loss can identify false reliability and improve the
transparent trade-off among class-conditional validity, informative singleton
outputs, and retention of a scientifically critical class under chemical shift.

The primary method is a model-agnostic triage layer, provisionally called RACER-C.
It is not presented as a new base molecular predictor, a universal uncertainty
estimator, or a safety guarantee.

## 3. Endpoints and roles

Candidate classification endpoints are recorded in
`endpoint_candidate_manifest.csv`. Eligibility uses only provenance, label
semantics, standardized structures, class counts, grouped role feasibility,
conflicts, and precision calculations. Model performance cannot affect eligibility.

Statuses:

- `primary`: eligible for confirmatory superiority testing;
- `secondary`: scientifically valid but not powered for all primary claims;
- `calibration-limited`: retained to expose finite-sample limitations, without
  non-inferiority or superiority claims;
- `excluded-with-reason`: fails a deterministic eligibility or licensing rule.

The original ESOL and Lipophilicity results remain in the baseline paper. Caco2,
PPBR, LD50, and hepatocyte clearance may be audited as secondary regression
extensions; RACER-R is not co-primary.

## 4. Chemical standardization

Before Freeze 1, the implementation must lock and log:

- RDKit version and standardization code commit;
- parsing and sanitization failures;
- fragment, salt/solvent, charge, isotope, stereochemistry, and tautomer policies;
- canonical isomeric SMILES policy;
- duplicate grouping after standardization;
- classification conflict resolution and regression aggregation;
- units, transformations, thresholds, species, and assay definitions;
- Murcko scaffold construction and Morgan radius 2 / 2048-bit fingerprints;
- raw/download/cleaned SHA256 values and deterministic rerun hashes.

Identical standardized structures are indivisible groups in every outer role and
cross-fitting fold. No majority vote may silently erase conflicting labels: ties
and conflicts follow the frozen endpoint-specific rule and are counted.

## 5. Outer roles

The count-only precision audit selected `50/20/15/15` for
`dev/policy/conf/test`. Under the initial `50/10/20/20` candidate, the smallest
primary policy critical-class cell could not certify a 10% error ceiling under
the predeclared 108-test Bonferroni contract even with zero observed errors.
Increasing policy to 20% removes that structural impossibility while retaining
at least 70 observations per class in every primary conformal and test cell. The
decision used no predictions or model outputs. The rejected allocations remain
in the audit outputs rather than being deleted.

The allocation is selected by a deterministic rule using grouped class counts and
precision, never model outputs. Each role is group-exclusive.

- `D_dev`: base fitting, honest nested cross-fitting, calibration, stacking,
  reliability-feature construction, BRI fitting, development-only attenuation
  selection, and the development reference CDF.
- `D_policy`: gate-threshold feasibility and selection only.
- `D_conf`: full-population and selected-domain conformal quantiles only.
- `D_test`: one final confirmatory evaluation after all components are frozen.

## 6. Evaluation tracks

- **A Random/IID:** grouped stratified random allocation where feasible. Under
  exchangeability, class-conditional conformal validity may be discussed.
- **B Strict scaffold OOD:** base, policy, conf, and test scaffolds are disjoint.
  Coverage is empirical; no arbitrary-shift guarantee is claimed.
- **C Similarity-cluster shift:** label-blind group construction with a frozen
  similarity algorithm and complexity cap. Coverage is empirical.
- **D Target-calibrated scaffold:** optional and separately labelled because
  target-domain labels are available for calibration. Its access regime cannot be
  mixed with Track B.
- **E External/temporal/source-held-out:** included only after endpoint, unit,
  threshold, standardization, overlap, and license audits pass.

Main seeds are 101--105 for eligible primary endpoints. Anchors may add seeds
106--110 for Tracks A and B. A single frozen training seed is used in the main
analysis; three training seeds are an anchor sensitivity.

## 7. Primary predictor blocks

Primary training is unweighted so predicted probabilities target the observed
development prevalence. Class-weighted training is an anchor sensitivity and is
recalibrated separately.

1. **ECFP block:** logistic regression, random forest, XGBoost, and ECFP MLP on
   Morgan radius-2 2048-bit fingerprints. Each model is Platt-calibrated honestly;
   the block probability is their uniform mean.
2. **Graph block:** Chemprop v2 D-MPNN with one frozen architecture, optimizer,
   stopping rule, epoch cap, and deterministic seed policy.
3. **Pretrained block:** frozen IBM MoLFormer embeddings plus an L2 logistic head.
   The immutable model/tokenizer revisions, pooling, max length, input SMILES, and
   embedding dtype are frozen before use.

Probabilities are clipped to `[1e-6, 1-1e-6]` before logits. Clipping counts are
reported. Isotonic calibration is sensitivity-only when prespecified class-count
criteria pass.

## 8. Honest nested cross-fitting

`D_dev` uses three deterministic, grouped meta-folds. For each held-out meta-fold:

1. the other two folds are used to generate inner OOF raw probabilities;
2. base models used to predict a row never train on that row or its structure group;
3. Platt calibrators are fitted only on inner OOF predictions;
4. calibrated block logits feed an L2 logistic stacker fitted without the held-out
   meta-fold;
5. Brier targets, local losses, and BRI training references exclude the held-out
   meta-fold;
6. the fitted chain predicts the held-out meta-fold once.

The three fitted chains form the deployment ensemble for policy/conf/test.
Development reference distributions use only honest held-out outputs. A lineage
manifest records, for every prediction row, all fit/calibration/meta/BRI fold IDs.

## 9. Reliability features and BRI

Let calibrated block logits be `l_ECFP`, `l_DMPNN`, and `l_MoLFormer`, and let
the L2 stacker margin be `m(x)`. The base class is `1[m(x) >= 0]`.

The primary reliability vector is:

1. variance of the three clipped calibrated logits, called heterogeneous
   representation--model disagreement;
2. `1 - max Tanimoto` to the allowed development reference pool, excluding the
   query structure group;
3. unweighted mean honest OOF Brier loss among the `min(20, n_ref)` nearest ECFP
   neighbours in the allowed development reference pool.

The Balanced Brier Risk Index fits squared stacked-probability loss from
`|m|` and the three features using class-balanced sample weights. The primary
implementation is a low-capacity constrained additive/linear model with monotonic
directions fixed in advance: larger margin cannot increase fitted loss; larger
disagreement, distance, or local loss cannot decrease it. Predictions are clipped
to `[0,1]` for numerical stability but are still called an index, not a calibrated
error probability.

BRI predictions for the development percentile reference are themselves honest
cross-fitted predictions. For a new point, `r(x)` is its mid-rank empirical CDF
within the base-predicted class using the development reference only. Policy,
conf, or test batches are never used to rerank scores.

## 10. Attenuation

The candidate transformation is

`T(x) = 1 + (T_max - 1) r(x)` and
`eta_RACER(x) = sigmoid(m(x) / T(x))`.

`T_max` is selected once globally from `{1.0, 1.5, 2.0}` using only nested
development predictions, endpoint-equal weighting, and the frozen objective:
maximize development MacroCSY subject to no class having a development coverage
shortfall worse than the precision-audit margin relative to stacking+Mondrian.
Ties choose the smaller `T_max`. It is never selected per endpoint. The selected
value is committed before policy/conf/test evaluation. `1.0` is the no-rectification
candidate and prevents a forced positive result.

## 11. Risk gate

`S(x)=1[r(x) <= t_yhat]` with `t_yhat` in
`{0.50,0.60,0.70,0.80,0.90,1.00}`.

On `D_policy`, search the 36 pairs using a deterministic lexicographic rule:

1. retain pairs whose simultaneous one-sided confidence bounds satisfy the frozen
   class-retention and critical-class accepted-error constraints;
2. maximize total retention;
3. maximize the smaller class retention;
4. choose the smaller sum of thresholds;
5. choose the smaller `t_critical`, then the smaller other threshold.

The precision audit freezes the bounds, familywise confidence correction, margins,
and minimum counts. No feasible pair yields `policy-infeasible`; constraints and
grid are not changed.

The Phase-2 numerical contract is a familywise alpha of 0.05 with conservative
Bonferroni correction across 36 pairs times three simultaneous constraints. Both
true classes require an exact one-sided retention lower bound of at least 0.50.
Among selected members of the scientifically critical true class, the exact
one-sided upper bound for base-classification error must not exceed 0.10. Each
policy true-class denominator and the selected critical-class denominator must
contain at least 25 observations. This policy error is deliberately measured
before `D_conf` conformal calibration; selected-domain prediction-set validity is
evaluated separately on `D_conf` and `D_test`.

## 12. Dual-track class-conditional conformal prediction

At nominal `alpha=0.10`, use finite-sample order statistic
`k=ceil((n_y+1)(1-alpha))`.

- Full-population Mondrian thresholds use every `D_conf` score with true class `y`.
- Selected-domain thresholds first apply the frozen gate, then use cells
  `(Y=y, S=1)`.
- If `k > n_y`, set the threshold to `+infinity` and mark the cell
  `calibration-limited`; do not resample, duplicate, lower coverage, or change roles.

Under exchangeability, full-population class-conditional coverage and selected
coverage conditional on a fixed covariate-based gate may be discussed. This does
not guarantee singleton correctness. Scaffold and cluster results are empirical.

## 13. Outputs and estimands

For each test row:

- `Defer-risk/domain` if `S=0`;
- `Accept(y)` if selected and the selected-domain set is the singleton `{y}`;
- `Ambiguous` if selected and the set is `{0,1}`;
- `Defer-empty` if selected and the set is empty.

Class-specific correct singleton yield is
`P(S=1, Gamma={y} | Y=y)`. MacroCSY averages both classes; WorstCSY is their
minimum. Overall CSY is secondary.

All safety metrics have explicit denominators:

- wrong-singleton exposure: wrong accepted singletons / all class members;
- accepted-singleton error: wrong accepted singletons / accepted singletons;
- gate retention: selected / all class members;
- singleton formation: singleton / selected within class;
- singleton correctness: correct / singleton within class;
- accepted FNR/FPR: errors among accepted singleton decisions using conventional
  true-class denominators, with deferred/ambiguous counts reported separately;
- empty and ambiguity rates: state count / all class members and / selected.

## 14. Comparators

Core, all primary endpoints:

1. ECFP marginal LAC;
2. ECFP Mondrian LAC;
3. heterogeneous block-average Mondrian;
4. stacking-only Mondrian;
5. unrestricted reliability-feature stacker plus Mondrian;
6. RACER score without gate;
7. stacking score plus RACER gate and selected Mondrian;
8. full RACER-C;
9. estimand-matched RCP plus class-conditional calibration.

Anchor-only after access/estimand audit: InfoSP, InfoSCOP, SCRC-I, SCRC-T, and
SCoRE. Transductive or batch methods receive separate tables and cannot be ranked
as if they were inductive per-molecule methods.

## 15. Primary hierarchy and statistics

The precision audit freezes numerical margins. The hierarchy is:

1. class-conditional validity/non-inferiority;
2. critical-class CSY preservation;
3. wrong-singleton, empty-set, and accepted-error safety;
4. only then, MacroCSY improvement over stacking+Mondrian.

Comparisons are paired on the same test molecules. Random splits use paired
molecule bootstrap; scaffold/cluster splits use group bootstrap. Coverage uses
exact binomial intervals. Effects are first estimated within endpoint/split/seed,
then summarized with endpoint-equal weighting. Seeds are repeated algorithmic
splits, not independent biological samples. Multiplicity is handled by the frozen
hierarchy and FDR for declared secondary families; effect sizes and intervals are
always reported.

## 16. Stop rules

- No primary-eligible endpoints: stop the superiority study and publish only an
  eligibility/feasibility audit if scientifically useful.
- Policy infeasible: report full-population CP and the failed policy; do not relax.
- Too few selected calibration observations: use `+infinity` and report limitation.
- Model or seed failure: retain and diagnose; do not silently drop.
- RACER fails superiority: reframe as an empirical reliability triage framework.
- Major estimand, endpoint, or algorithm change after freeze: pause and obtain user
  approval before new confirmatory results.

## 17. Pre-freeze blockers

- populate raw and cleaned hashes and label semantics;
- execute grouped role-feasibility and precision simulations;
- lock exact package/model revisions and hyperparameters;
- implement and pass lineage/leakage tests;
- benchmark one anchor on the available GPU and replace planning estimates;
- freeze critical classes and margins;
- obtain user approval for the protocol tag and full GPU run.
