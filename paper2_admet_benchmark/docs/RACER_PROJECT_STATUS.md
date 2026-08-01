# RACER extension project status

Last updated: 2026-08-02

## Current phase

Phase 1 data provenance, cleaning, and grouped role-feasibility checkpoint.
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
- Audited 17 candidate endpoints against official source/license records.
- Locked file IDs and raw SHA256 values for three CC-BY-4.0 Veith CYP endpoints.
- Implemented fail-closed RDKit classification cleaning with rejection accounting.
- Built deterministic label-blind scaffold and similarity-cluster group inputs.
- Completed 135 grouped count audits and 540 conformal-resolution cells without
  generating a model prediction.

## Commits

- `a620ad6` Fix Paper 2 bibliography audit inputs.
- `5cdced5` Add Paper 2 RACER pre-freeze protocol audit.
- `2a8d49a` Record Paper 2 extension Phase 0 status.
- Phase 1 checkpoint: this follow-up commit.

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
- `docs/phase1_data_and_role_feasibility_audit_2026.md`
- `protocols/data_provenance_license_manifest.csv`
- `protocols/chemical_standardization_contract.yaml`
- `protocols/data_acquisition_and_cleaning_runbook.md`
- `data/manifests/racer_c/*_acquisition.json`
- `data/manifests/racer_c/*_cleaning.json`
- `data/manifests/racer_c/*_similarity_clusters.json`
- `results/racer_c_preflight/*.csv`

## Quality gates

- Existing Python scripts compile: PASS (non-fatal warnings documented).
- Existing bibliography audit: PASS (42 cited keys, no missing or duplicate keys,
  34 cited references from 2021--2026).
- Existing frozen asset integrity: PASS (CRLF-aware for historical table hashes).
- New protocol-contract tests: PASS.
- Phase 1 provenance/role tests: PASS (15 tests total after Phase 1).
- Three licensed CYP raw-file hashes: PASS.
- Source-row cleaning reconciliation: PASS for 37,550 rows across three endpoints.
- Grouped role count gate: PASS (135/135 cells).
- Conformal resolution/minimum selected count: PASS (540/540 cells; minimum 152).
- Formal endpoint freeze: BLOCKED.
- Full GPU run: BLOCKED pending measured benchmark and user approval.

## Known risks

- Many small/imbalanced endpoints will be secondary or calibration-limited after
  four-way grouped allocation; ClinTox remains the explicit rare-class anchor.
- The only current count-eligible endpoints are three CYP assays from one source
  family, insufficient for an ADMET-wide pooled claim.
- Honest nested D-MPNN cross-fitting may require 150--400 RTX-4090 GPU-hours.
- Recent selected-risk methods use different estimands and access regimes.
- RACER-C novelty is empirical/framework-level, not a new coverage theorem.
- Raw/clean row-level data remain intentionally ignored; committed acquisition,
  cleaning, clustering, and result hashes provide the rebuild contract.
- This container's pdfTeX font setup fails before manuscript layout inspection;
  the source itself has not been implicated.

## Blocked items

- exact Veith-to-TDC binary label transformation;
- original-source terms or licensed replacements for non-CYP candidates;
- a scientifically diverse primary endpoint panel;
- immutable Chemprop/MoLFormer/environment lock;
- policy-grid and paired-effect precision margins;
- lineage/leakage implementation tests;
- GPU smoke benchmark;
- user approval for formal protocol tag and full compute.

## Next automatic action

Audit explicitly licensed non-CYP classification candidates and reconstruct the
Veith/TDC binary label mapping. Do not fit extension models. If a diverse,
semantically recoverable panel cannot be established, narrow the study before the
formal protocol tag.

## Protocol deviations

See `protocols/protocol_deviations.md`. All current entries occurred before any
extension prediction was generated.
