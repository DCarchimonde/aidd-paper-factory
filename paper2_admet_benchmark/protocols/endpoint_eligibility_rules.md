# Endpoint eligibility rules

Status: draft for pre-model precision audit.

## Deterministic order

1. Verify original source, data identifier, label definition, units/threshold,
   species/assay, and license.
2. Standardize structures using the frozen cleaning implementation.
3. Resolve duplicates/conflicts using the frozen endpoint rule.
4. Count classes and groups; create candidate outer-role allocations without
   fitting a model.
5. Run the precision/feasibility simulation across all confirmatory seeds.
6. Assign one status. Model metrics are not available to this process.

## Hard exclusions

Use `excluded-with-reason` if any condition holds:

- redistribution/analysis rights cannot be established;
- label meaning, unit, threshold, species, or assay cannot be recovered;
- fewer than 300 clean unique structures;
- either class has fewer than 30 clean unique structures;
- unresolved conflicting-label structures exceed 10% of clean structures;
- deterministic role construction fails structural-group exclusivity;
- a source-provided test set would be contaminated by reconstructing roles.

## Primary classification status

All conditions must hold for every main split seed:

- both classes have at least 350 clean unique structures before allocation;
- policy role has at least 25 observations per true class;
- conformal role has at least 70 observations per true class;
- at minimum gate retention 0.50, the planned selected conformal count is at
  least 35 per true class;
- test role has at least 70 observations per true class;
- at least 100 distinct Murcko scaffolds overall and no single structure/scaffold
  group makes the requested allocation impossible;
- the exact-interval precision audit passes its frozen margins.

These are minimum gates, not proof of statistical power. Grouped allocation and
simulation can still downgrade a numerically large endpoint.

## Secondary and calibration-limited

- `secondary`: scientifically valid and structurally feasible, with adequate
  point-prediction analysis, but failing at least one primary precision condition.
- `calibration-limited`: a conformal method can be executed, but one or more
  class/selection cells are expected to have an infinite quantile or intervals too
  wide for the primary hierarchy. These endpoints remain visible as failure modes.

## Critical class

Critical class is endpoint-specific and frozen from scientific semantics, not
prevalence. Toxicity, mutagenicity, DILI, hERG blocking, P-gp inhibition, and CYP
inhibition usually make the adverse-liability class critical, subject to source
verification. Poor HIA or poor bioavailability may make label 0 critical. BBB
penetration is context-dependent; unless a deployment context is prespecified,
both-class and worst-class preservation replace a single critical class.
