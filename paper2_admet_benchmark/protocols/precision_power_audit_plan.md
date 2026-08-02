# Precision and power audit plan

Status: **count-only Phase 2 audit completed; model-dependent precision remains
pre-freeze**.

## Required inputs

For every endpoint, track, allocation, and split seed record true-class counts in
dev, policy, conformal, and test; group-size distributions; and the minimum planned
gate retention. No model predictions are used in the first-stage audit.

## Conformal resolution

For `alpha=0.10`, calculate `k=ceil((n+1)*0.90)` for every full and selected
class cell. Mark `k>n` as an infinite-threshold cell. Simulate selected counts
under retention values 0.50--1.00 and class-composition perturbations consistent
with the grouped allocation.

## Interval precision

Use Clopper--Pearson intervals for coverage and exact one-sided bounds for zero or
sparse wrong-singleton counts. Report expected interval width at coverage values
0.80, 0.90, and 0.95 for each realized `n`. Do not treat the algebraic minimum of
nine calibration observations as adequate precision.

## Paired effect precision

Before predictions, simulate paired binary outcomes for baseline versus RACER over
a grid of discordant-pair rates. For each endpoint/track, estimate attainable
interval widths for:

- class-specific coverage difference;
- critical-class CSY difference;
- MacroCSY difference;
- accepted-singleton error difference;
- class-specific gate retention.

For scaffold/cluster tracks, repeat with cluster bootstrap using the observed
group-size distribution. Endpoint-level simulations must not pool molecules across
endpoints as if independent.

## Policy feasibility

For all 36 gate pairs, simulate selection and accepted errors under null and
boundary cases. Compare Bonferroni simultaneous one-sided exact bounds with a
predeclared less-conservative alternative only if its familywise behavior is
demonstrated. Freeze:

- class-retention lower bounds;
- critical-class wrong-singleton/accepted-error upper bound;
- familywise confidence level and correction;
- minimum policy count;
- coverage and CSY non-inferiority margins;
- wrong-singleton and empty-set limits.

If data cannot support a numerical non-inferiority margin with useful precision,
the endpoint is descriptive or calibration-limited. A conventional value such as
0.02 must not be inserted without this audit.

## Deliverables

- `precision_power_inputs.csv` with hashes;
- `role_count_feasibility.csv`;
- `conformal_resolution.csv`;
- `exact_interval_precision.csv`;
- `paired_effect_simulation.csv`;
- `policy_grid_error_control.csv`;
- a signed decision file containing frozen margins and endpoint statuses.

## Phase 2 execution decision (2026-08-03)

The count-only audit was executed across four allocation candidates, three tracks,
and seeds 101--105 without fitting a model or inspecting an extension prediction.
Outputs are committed under `results/racer_c_phase2_preflight/`.

The original `50/10/20/20` candidate is structurally incapable of certifying the
predeclared 10% critical-class base-error ceiling in its smallest policy cells
under a familywise alpha of 0.05, 36 gate pairs, and three simultaneous
constraints per pair (108 Bonferroni tests), even when zero errors are observed.
The count-only selection therefore chose `50/20/15/15`. This choice used no model
outputs and is recorded in `protocol_deviations.md`.

After correcting the residual label-dependent group-order key, four Tox21
endpoints pass all 15 track-seed count cells: `Tox21_NR_AhR`, `Tox21_NR_ER`,
`Tox21_SR_ARE`, and `Tox21_SR_MMP`. Their smallest policy critical-class cell is
104. Exact zero-error bounds are mathematically capable of meeting the 10% ceiling
at sufficiently high retention; this is not evidence that the trained method will
do so. The policy selector must return
`policy-infeasible` if no one of the 36 pairs satisfies the frozen simultaneous
bounds.

The signed machine-readable decision is
`results/racer_c_phase2_preflight/precision_policy_decision.json`. Model-dependent
paired effects, empirical gate feasibility, and measured compute remain unexecuted.
