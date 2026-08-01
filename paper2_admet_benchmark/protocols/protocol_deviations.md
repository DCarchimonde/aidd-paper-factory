# Paper 2 reliability extension protocol deviations

No frozen protocol exists yet. The entries below are pre-freeze design changes,
not post-outcome deviations.

| Date | Stage | Change | Reason | Outcome access |
|---|---|---|---|---|
| 2026-08-01 | Phase 0 | Prefer 50/10/20/20 role allocation pending deterministic feasibility audit | selected class-conditional calibration needs more observations than the original 15% candidate | no extension predictions generated |
| 2026-08-01 | Phase 0 | Define three-fold honest nested meta-learning and cross-fit ensemble deployment | prevent second-stage calibration/stacking/BRI leakage while bounding compute | no extension predictions generated |
| 2026-08-01 | Phase 0 | Make unweighted base training primary; class weighting anchor-only | preserve natural-prevalence probability interpretation and prevent model-grid explosion | no extension predictions generated |
| 2026-08-01 | Phase 0 | Select one global `T_max` from a development-only grid including 1.0 | fixed 2.0 lacked justification; including no rectification prevents forced benefit | no extension predictions generated |
| 2026-08-01 | Phase 0 | Move InfoSP/InfoSCOP, SCRC, and SCoRE to access-matched anchor analyses | their estimands/access regimes are not interchangeable with inductive Mondrian CP | no extension predictions generated |
| 2026-08-01 | Phase 0 | Treat ClinTox as a likely calibration-limited failure-mode anchor | approximately 93 positives cannot support the proposed four-role selected CP hierarchy | only old frozen baseline outcomes known |

Post-freeze entries MUST include affected files, commits, seeds/endpoints, whether
labels or results had been inspected, scientific impact, and user approval where
required.
