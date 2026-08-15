# RACER-C3 prospective protocol draft

Version: **0.1 development draft -- not freeze-ready**

Opened: 2026-08-09

Branch: `paper2-racer-c3-development-2026`

## Firewall

The four v1 Tox21 endpoints and seeds 101--105 are fully known and may only be
used for architecture development, ablation, compute planning, and failure
analysis. No result on that panel is prospective or confirmatory.

No RACER-C3 deployment target may be evaluated unless the candidate first
passes a prediction-free promotion review and then receives a new protocol tag
that is
approved after a prediction-free formal review. V1 and C2 namespaces are read-
only. All development outputs use `results/racer_c3_development/`; a future
prospective run must use a separately versioned namespace.

## Frozen-candidate research questions

1. Does the frontier candidate-expert score improve full-population MacroCSY on
   untouched strict-scaffold endpoints while retaining both classes?
2. Does the unlabeled symmetric frontier audit activate only under genuine
   structural extrapolation?
3. Does exact fallback make ordinary-domain outputs identical to the v1
   no-gate score?
4. Can coverage, wrong-singleton exposure, empty exposure, and critical-class
   CSY be certified on final set states before the test is opened?

## Work required before freeze

1. Identify untouched endpoints and verify provenance, license, exact target
   polarity, deduplication, and chemical-group fields.
2. Recompute role/class/group counts and finite-sample conformal resolution.
3. Obtain independent review of the batch-symmetric route validity argument.
4. Implement or audit estimand-matched comparators: marginal/Mondrian LAC,
   weighted/CoDrug, RC3P, SOCOP, SCRC, SCoRE, rejection baselines, COLA or an
   equivalent score aggregator, v1 no-gate, and all C3 ablations.
5. Freeze the endpoint-equal primary estimand, confidence intervals,
   multiplicity, non-inferiority margins, and stop rules.
6. Add lineage, interruption/resume, target-permutation, route-symmetry, and
   exact-fallback tests.
7. Record hashes in a prediction-free review and obtain explicit approval for
   the protocol tag.

Current blocker: after protecting the critical class with
`alpha_0=0.10, alpha_1=0.095`, the known-panel strict-scaffold effect is only
+0.2097 percentage points with 10 wins and 10 losses. This does not pass the
current architecture freeze gate.

## Intended role order

1. `D_dev`: base fitting, honest meta-fold predictions, candidate-correctness
   cross-fitting, and fixed development reference construction;
2. unlabeled `D_conf X` plus deployment `X`: one symmetric route audit;
3. `D_conf`: one class-conditional calibration of the selected route;
4. `D_policy`: one simultaneous certificate of the fixed final set states;
5. `D_test`: one evaluation only after all earlier artifacts are hashed.

The policy role cannot search experts, route thresholds, temperatures, alpha,
or a rejection threshold. A failed certificate is retained and follows the
predeclared stop rule.

## Claim hierarchy draft

1. implementation identity and route audit;
2. class-specific coverage and final-state safety certificate;
3. strict-scaffold critical-class CSY non-inferiority;
4. strict-scaffold MacroCSY superiority over v1 no-gate;
5. similarity-track consistency; and
6. grouped-random exact-fallback identity.

Numerical margins remain deliberately unset until the untouched-panel count and
precision audit. They may not be inferred after outcomes are opened.
