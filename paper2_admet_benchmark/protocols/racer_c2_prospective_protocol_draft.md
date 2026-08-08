# RACER-C2 prospective protocol draft

Version: **0.2 development draft -- not frozen**

Opened: 2026-08-08

Branch: `paper2-racer-c2-development-2026`

Parent evidence: RACER-C v1 completed 60/60 cells and 540/540 method results with
zero execution failures, but its gate was policy-infeasible in 60/60 cells. V1
outputs and protocol tag remain immutable.

## Firewall

No RACER-C2 result may be described as prospective or confirmatory until a new
user-approved protocol tag exists. The completed v1 Tox21 panel, including seeds
101--105, is now known evidence and is restricted to retrospective development,
ablation, failure analysis, and compute planning.

The current development command may materialize only `D_dev` labels from v1 raw
artifacts. It cannot evaluate test rows. All v2 outputs use
`results/racer_c2_development/` or a later versioned prospective namespace and
must not overwrite `results/racer_c_confirmatory_v1/`.

## Research questions

1. Does candidate-label exponential reliability tilting improve MacroCSY over
   stacking-Mondrian while preserving both class coverages?
2. Does removing the predicted-class gate avoid v1's structural loss of the
   critical class?
3. Can the final wrong-singleton, empty-set, coverage, and critical-CSY states be
   certified with useful retention on independent policy data?
4. Are effects reproducible under strict scaffold and similarity-cluster shift,
   with grouped random splitting retained as a negative control?

## Development stage

The v1 panel is used only to choose or reject algorithmic architecture. The
finite score family, safe fallback, fitting lineage, and deterministic selection
rule are in `configs/racer_c2/development_lock_v0.yaml` and
`docs/racer_c2_algorithm_specification_v0.1.md`.

Development outputs must record `test_labels_used=false`,
`scientific_test_predictions_generated=false`, and `freeze_authorized=false`.
The first learned counterfactual-error module selected
`counterfactual_blend=0` and is retained only as a rejected ablation. Selecting
`gamma_0=gamma_1=0` is an allowed negative result for the primary reliability-
tilting mechanism.

The retrospective development selector is restricted to the predeclared
chemical-shift tracks (`strict_scaffold` and `similarity_cluster`). The
`random_grouped` track does not vote in algorithm selection and may be reported
only as a negative-control analysis.

The development feasibility rule requires each class's cell-equal mean coverage
to remain within 1 percentage point of stacking-Mondrian and every cell's
empirical class coverage to remain at least 85%. The pre-freeze rule revision and
its rationale are recorded in `docs/racer_c2_development_decision_log.md`.

The resulting development-only configuration is `T_max=1.5`, `gamma_0=0.1`,
`gamma_1=-0.1`, and `counterfactual_blend=0`. It is a candidate for prospective
review, not a frozen or validated final method.

## Pre-freeze work still required

1. Audit and select genuinely untouched prospective endpoints using provenance,
   license, semantics, counts, and chemical groups only.
2. Narrow the scientific scope if the only adequately powered endpoints are the
   three source-defined Veith CYP inhibition datasets.
3. Re-run the exact count and precision audit for the new endpoint panel and role
   allocation.
4. Freeze numerical coverage, wrong-singleton, empty-set, and critical-CSY
   constraints; no placeholder may be converted into a number after outcomes.
5. Implement estimand-matched one-sided, SCRC, SCoRE, RC3P, and standard
   conformal comparators or document access incompatibility.
6. Add test-label permutation, transitive lineage, batch-invariance, and
   interruption/resume tests to the future production runner.
7. Complete a prediction-free formal review and obtain explicit user approval
   for a new tag.

## Prospective role order

The intended order is:

1. `D_dev`: base fitting, nested cross-fitting, reliability-index construction,
   and global candidate-label tilt selection;
2. `D_conf`: class-conditional quantiles for the single selected score;
3. `D_policy`: one direct certificate for the fixed final set states; and
4. `D_test`: one final evaluation after all previous artefacts are hashed.

This order is different from v1 by design. Policy does not search a threshold
grid. If the policy certificate fails, the test run follows the predeclared stop
rule; the bounds or endpoint list are not changed.

## Claim hierarchy

1. class-specific coverage and final-state certificate;
2. critical-class CSY non-inferiority;
3. wrong-singleton and empty-set safety;
4. MacroCSY improvement over stacking-Mondrian; and
5. shift-track consistency.

Algorithmic superiority may be claimed only if the frozen hierarchy passes on
the untouched prospective panel. Retrospective Tox21 effects are labelled
development results regardless of their magnitude.
