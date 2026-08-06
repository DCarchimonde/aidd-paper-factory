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
| 2026-08-02 | Phase 1 | Treat `Dataset License: Not Specified` as a Freeze-1 blocker | TDC code licensing and individual dataset licensing are distinct; availability is not permission | no extension predictions generated |
| 2026-08-02 | Phase 1 | Use RDKit 2026.03.4 classification standardization candidate with explicit conflict exclusion | create deterministic hashes and prevent majority vote from erasing label conflicts | no extension predictions generated |
| 2026-08-02 | Phase 1 | Replace label-aware scaffold/cluster allocation with size-only label-blind allocation | the prototype violated the chemical-shift access contract | no extension predictions generated; earlier feasibility output discarded |
| 2026-08-02 | Phase 1 | Use one deterministic 0.60-similarity leader partition per endpoint | separate group construction from role-allocation seeds and include the 12k--13k CYP endpoints under a frozen 15k cap | no extension predictions generated |
| 2026-08-02 | Phase 1 | Add a study-level source/mechanism breadth gate | three endpoints from one Veith qHTS campaign cannot support an ADMET-wide pooled claim | no extension predictions generated |
| 2026-08-03 | Phase 2 | Add `50/20/15/15` as a pre-freeze outer-role candidate | under `50/10/20/20`, the smallest primary policy critical-class cell cannot certify a 10% error ceiling under the frozen 108-test Bonferroni contract even with zero observed errors | count-only precision evidence; no model outputs generated |
| 2026-08-03 | Phase 2 | Run a development-only seed-99 ECFP/logistic nested-OOF CPU smoke | exercise actual group allocation, feature generation, nested calibration, and transitive lineage before the production trainer exists | 2,928 ephemeral development OOF probabilities checked only for finiteness and lineage; no performance metric or policy/conformal/test prediction generated |
| 2026-08-03 | Phase 2 correction | Remove the remaining label-dependent scaffold/cluster group-order key and strengthen the permutation test | the size-only role objective was label-blind, but its ordering still used the largest within-group class count; a global 0/1 flip test could not expose this | all role and precision outputs regenerated before GPU or confirmatory runs; NR-AhR changed from 14/15 to 15/15 and became a fourth count-only primary candidate; no model output inspected |
| 2026-08-06 | Phase 3 pre-freeze benchmark correction | Exclude structures above the pinned MoLFormer 202-token pretraining domain before role allocation and every component fit; prohibit truncation and positional-domain extension | the first real seed-99 run found three of 5,855 NR-ER structures above 202 tokens (227, 239, 242); the pinned model card states that overlength molecules were dropped during pretraining | seed-99 stopped before any component fit or metric; the exact tokenizer was verified against the offline contract on 5,855/5,855 structures; the eligible NR-ER cohort is 5,852 and the strict-scaffold development benchmark is 2,926; confirmatory seeds remain blocked |

Post-freeze entries MUST include affected files, commits, seeds/endpoints, whether
labels or results had been inspected, scientific impact, and user approval where
required.
