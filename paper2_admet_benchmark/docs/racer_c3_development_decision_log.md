# RACER-C3 development decision log

Status: **known-outcome retrospective architecture development**

Date: 2026-08-09

## Immutable parent and stopped C2 line

RACER-C v1 remains immutable. Its predicted-class policy gate was infeasible in
all 60 primary cells. RACER-C2 was retained as a transparent weak prototype:
its selected reliability tilt improved the no-tilt score by only 0.2424
percentage points over the 40 chemical-shift cells, with 21 wins and 19 losses,
and did not produce a positive certificate. C2 is not the primary algorithm.

## Module screens that were rejected

All screens used honest development roles for fitting or selection. The already
known v1 outer panel was read only after method selection unless explicitly
labelled architecture search.

Rejected families included:

- ordinary meta-logistic, histogram-boosted, and extra-trees probability
  stacking;
- global and structural-frontier Mondrian bins;
- multi-view alpha allocation/intersection;
- SOCOP size-penalty variants;
- class-balanced singleton-utility lower-envelope scores;
- group-equal calibration weights;
- unconditional and soft-label class-conditional transport weights; and
- per-cell label-view routing, which overfit severely.

The class-balanced utility screen selected `kappa=0`, exactly its SOCOP/no-tilt
fallback. The soft-label transport screen produced approximately +0.02
percentage points on the chemical-shift panel and slightly harmed random
splitting. Neither mechanism earned primary status.

## Surviving candidate-expert signal

A global candidate-label score matrix showed a reproducible structural
asymmetry:

- candidate 0: equal-weight logit mean of ECFP, D-MPNN, and MoLFormer;
- candidate 1: shared cross-fitted candidate-correctness logistic model; and
- both supports: v1 continuous risk tempering with `T_max=1.5`.

Applied everywhere, this score improved the two chemical tracks by 0.4518
percentage points on average versus the v1 no-gate score but reduced grouped-
random MacroCSY by 0.1862 points. The improvement was concentrated in strict
scaffold (+0.8433 points); similarity-cluster change was +0.0604 points and
grouped-random change was -0.1862 points.

## Label-free route decision

Unlabeled scaffold overlap with the development reference separated the v1
tracks without using outcomes: strict-scaffold overlap was exactly zero in all
20 cells, whereas the minimum molecule-weighted overlap was above 0.41 in the
other 40 cells. Median ECFP distance also separated strict scaffold (minimum
above 0.59) from the other tracks (maximum below 0.55).

The first RACER-C3 draft therefore activated the asymmetric experts only when overlap was at
most 0.05 and median distance is at least 0.57. The rule is computed on the
unordered union of conformal and deployment covariates. All other cases use the
exact v1 no-gate fallback. On the known panel, routing yields the strict-
scaffold +0.8433-point signal while making similarity and random outputs
identical to the parent score.

## Coverage correction and freeze-gate result

The +0.8433-point strict-scaffold effect used equal `alpha=0.10` but reduced
critical-class empirical coverage by 1.2603 percentage points. A coverage
sensitivity audit showed that this apparent efficiency was partly a
coverage--ambiguity trade. The development candidate was therefore corrected
to `alpha_0=0.10` and the more conservative `alpha_1=0.095`; relaxing class 0
above 0.10 was rejected because it would weaken the declared 90% class-wise
nominal target.

After this protection, the strict-scaffold MacroCSY difference was +0.2097
percentage points, with 10 wins and 10 losses. Mean class-0/class-1 coverage
differences were +0.4272/-0.5736 points. Critical-class CSY increased by 1.1232
points while class-0 CSY decreased by 0.7037 points. The endpoint-cluster
bootstrap interval crossed zero. Routed similarity and grouped-random outputs
remain exact parent fallbacks, so the all-60-cell mean difference is +0.0699
points.

## Interpretation

This is sufficient to define and implement a new algorithm candidate, but it is
not sufficient to freeze it as the primary Paper 2 algorithm. The effect is
small, mixed, and chosen after the v1 outer outcomes were known. RACER-C3 is
therefore retained as a transparent mechanism/ablation while the prospective
test remains unauthorized. If it is later promoted after an independent
label-firewalled rationale, failure and fallback activation rates must be
reported without redesign.
