# RACER extension project status

Last updated: 2026-08-01

## Current phase

Phase 1 protocol drafting, after completion of Phase 0 repository/scientific audit.
The protocol is explicitly **not frozen**.

## Completed work

- Verified GitHub default branch, permissions, HEAD, branch list, and Paper 2 layout.
- Preserved the original four-endpoint frozen assets and manuscript boundary.
- Audited current code syntax, bibliography wiring, frozen assets, and local LaTeX build.
- Audited adjacent 2025--2026 RCP, InfoSP/InfoSCOP, SCRC, SCoRE, Chemprop,
  MoLFormer, TDC, and the 2026 ADMET reliability benchmark.
- Classified pre-freeze design issues as P0/P1/P2/DELETE.
- Created the isolated extension branch and initial protocol, algorithm spec,
  endpoint/eligibility/access manifests, precision plan, deviations log, and config.
- Corrected bibliography audit scope to include both bibliography files loaded by
  `main.tex`.

## Commits

- `a620ad6` Fix Paper 2 bibliography audit inputs.
- `5cdced5` Add Paper 2 RACER pre-freeze protocol audit.
- `2a8d49a` Record Paper 2 extension Phase 0 status.

## Artifacts

- `docs/phase0_repository_scientific_audit_2026.md`
- `protocols/paper2_reliability_extension_protocol_2026.md`
- `docs/racer_c_algorithm_specification_v0.1.md`
- `protocols/endpoint_candidate_manifest.csv`
- `protocols/endpoint_eligibility_rules.md`
- `protocols/precision_power_audit_plan.md`
- `protocols/method_access_manifest.csv`
- `protocols/protocol_deviations.md`
- `configs/racer_c/study_design.yaml`

## Quality gates

- Existing Python scripts compile: PASS (non-fatal warnings documented).
- Existing bibliography audit: PASS (42 cited keys, no missing or duplicate keys,
  34 cited references from 2021--2026).
- Existing frozen asset integrity: PASS (CRLF-aware for historical table hashes).
- New protocol-contract tests: PASS.
- Formal endpoint freeze: BLOCKED.
- Full GPU run: BLOCKED pending measured benchmark and user approval.

## Known risks

- Many small/imbalanced endpoints will be secondary or calibration-limited after
  four-way grouped allocation.
- Honest nested D-MPNN cross-fitting may require 150--400 RTX-4090 GPU-hours.
- Recent selected-risk methods use different estimands and access regimes.
- RACER-C novelty is empirical/framework-level, not a new coverage theorem.
- The clean checkout lacks ignored raw/processed data and cannot yet execute data
  eligibility or full manuscript rebuild from local row-level sources.
- This container's pdfTeX font setup fails before manuscript layout inspection;
  the source itself has not been implicated.

## Blocked items

- raw dataset download and original-source license verification;
- exact class/scaffold counts and role-allocation simulation;
- immutable Chemprop/MoLFormer/environment lock;
- GPU smoke benchmark;
- user approval for formal protocol tag and full compute.

## Next automatic action

Begin deterministic endpoint acquisition/cleaning preparation: lock source
identifiers and original license records, add hash/rejection logging, and implement
pre-model role-feasibility inputs without generating extension model predictions.

## Protocol deviations

See `protocols/protocol_deviations.md`. All current entries occurred before any
extension prediction was generated.
