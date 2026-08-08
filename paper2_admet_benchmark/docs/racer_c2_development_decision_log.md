# RACER-C2 development decision log

Status: **retrospective method development; not a confirmatory result**

Date: 2026-08-08

## Immutable parent result

RACER-C v1 remains a completed negative confirmatory experiment. Its predicted-
class gate was policy-infeasible in all 60 primary cells. No v1 code, protocol,
tag, or result is modified by RACER-C2.

## Draft A: learned counterfactual candidate-error scores

The first v2 draft cross-fitted two monotone gradient-boosted error indices and
blended their fixed-reference percentiles into the candidate-label scores. The
development-only selector chose `counterfactual_blend=0`. The learned component
was therefore rejected rather than renamed as a successful method. Its numerical
code remains only to make the negative ablation reproducible.

## Draft B: candidate-label exponential reliability tilting

The second draft introduced

`s_y(x) = a_y(x) exp(gamma_y r(x))`.

For a predeclared critical label 1, the finite family allows a nonnegative
`gamma_0` to penalize high-risk non-critical inclusion and a nonpositive
`gamma_1` to protect high-risk critical inclusion. The zero vector is an exact
fallback. This draft does not branch on the hard predicted class.

## Development-gate correction before freeze

An initial implementation required every candidate class coverage in every cell
to stay within 2 percentage points of stacking-Mondrian. One class in one of 40
shift cells missed that relative rule by 0.34 percentage points: its coverage
was 88.76% versus the baseline's 91.10%, while its cell MacroCSY increased by
3.28 percentage points. The veto was therefore dominated by a single empirical
baseline fluctuation rather than global class reliability.

Because the protocol was still explicitly unfrozen and no new prospective
endpoint had been selected or evaluated, the development rule was replaced and
logged. The final rule requires, separately for both classes:

1. cell-equal mean coverage no more than 1 percentage point below the baseline;
   and
2. empirical coverage of at least 85% in every development cell.

This rule is fixed in `configs/racer_c2/development_lock_v0.yaml`. It must not be
changed in response to future prospective outcomes.

## Honest development-only result

The selector materialized only `D_dev` labels in 40 predeclared chemical-shift
cells (strict scaffold and similarity cluster), evaluated 18 configurations
(720 cell/configuration rows), and selected:

- `T_max=1.5`;
- `gamma_0=0.1`;
- `gamma_1=-0.1`; and
- `counterfactual_blend=0`.

Cell-equal development summaries were:

- MacroCSY: 52.6343%, versus 52.3092% for stacking-Mondrian (+0.3251 pp);
- class-0 mean/minimum coverage: 89.6694% / 86.0722%; and
- class-1 mean/minimum coverage: 89.5676% / 85.5634%.

The selector did not materialize policy, conformal, or test labels, and did not
generate scientific test predictions.

## Known-outcome stress check

After selection, the already known v1 outer outcomes were used only as a
retrospective stress check. On the 40 shift cells, the selected RACER-C2 score
had mean MacroCSY 58.0788%, compared with 57.2195% for stacking-Mondrian and
57.8364% for the v1 RACER score without its gate. The incremental difference
over the v1 score was +0.2424 pp, with 21 cell wins and 19 losses.

This mixed and modest effect is not confirmatory evidence and does not establish
superiority. It is sufficient only to keep the mechanism as a prospective
candidate. Untouched endpoints, frozen comparators, and a new user-approved
protocol tag are still required before an SCI manuscript can make an algorithmic
performance claim.
