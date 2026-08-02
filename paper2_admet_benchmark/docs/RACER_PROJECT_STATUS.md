# RACER extension project status

Last updated: 2026-08-03

## Current phase

Phase 2 endpoint, policy-precision, and lineage-contract checkpoint.
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
- Locked the official NCATS Tox21 Challenge archive/member hashes and processed
  all 12 assay properties with complete source-row reconciliation.
- Completed 300 allocation/track/seed count cells and 1,200 conformal-resolution
  cells across four prospective allocations.
- Selected `50/20/15/15` using count-only exact precision evidence after proving
  the previous policy fraction could be structurally incapable of certification.
- Assigned three Tox21 primary candidates, two track-limited secondary endpoints,
  and seven calibration-limited endpoints without model outcomes.
- Implemented a deterministic fail-closed 36-pair policy selector and a transitive
  training/prediction lineage validator.
- Passed a development-only seed-99 CPU nested-OOF integration smoke on 2,928
  NR-ER rows without computing a performance metric or predicting any outer role.

## Commits

- `a620ad6` Fix Paper 2 bibliography audit inputs.
- `5cdced5` Add Paper 2 RACER pre-freeze protocol audit.
- `2a8d49a` Record Paper 2 extension Phase 0 status.
- Phase 1 remote checkpoint: `a70ca3bed3ddb77470e6507b9659edb750ba6fcd`.
- Phase 2 checkpoint: this follow-up commit.

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
- `docs/phase2_endpoint_policy_lineage_audit_2026.md`
- `docs/veith_tdc_label_semantics_audit_2026.md`
- `scripts/racer_c/prepare_tox21_challenge.py`
- `scripts/racer_c/precision_policy_audit.py`
- `scripts/racer_c/policy_selection.py`
- `scripts/racer_c/lineage_contract.py`
- `results/racer_c_phase2_preflight/*`
- `scripts/racer_c/run_seed99_cpu_lineage_smoke.py`

## Quality gates

- Existing Python scripts compile: PASS (non-fatal warnings documented).
- Existing bibliography audit: PASS (42 cited keys, no missing or duplicate keys,
  34 cited references from 2021--2026).
- Existing frozen asset integrity: PASS (CRLF-aware for historical table hashes).
- Protocol, provenance, role, policy, lineage, smoke-output, and integrity tests:
  PASS (28).
- Three licensed CYP raw-file hashes: PASS.
- Source-row cleaning reconciliation: PASS for 37,550 rows across three endpoints.
- Grouped role count gate: PASS (135/135 cells).
- Conformal resolution/minimum selected count: PASS (540/540 cells; minimum 152).
- Tox21 cleaning reconciliation: PASS for all 12 endpoints and all 11,764 source
  records per endpoint.
- Phase 2 primary count gate: 3 primary, 2 secondary, 7 calibration-limited.
- Count-only policy precision audit: PASS; selected allocation `50/20/15/15`.
- Formal endpoint/protocol freeze: BLOCKED.
- Full GPU run: BLOCKED pending measured benchmark and user approval.

## Known risks

- Many small/imbalanced endpoints will be secondary or calibration-limited after
  four-way grouped allocation; ClinTox remains the explicit rare-class anchor.
- The three current primary candidates are distinct Tox21 mechanisms but one NCATS
  Challenge source family, insufficient for an ADMET-wide pooled claim.
- Honest nested D-MPNN cross-fitting may require 150--400 RTX-4090 GPU-hours.
- Recent selected-risk methods use different estimands and access regimes.
- RACER-C novelty is empirical/framework-level, not a new coverage theorem.
- Raw/clean row-level data remain intentionally ignored; committed acquisition,
  cleaning, clustering, and result hashes provide the rebuild contract.
- This container's pdfTeX font setup fails before manuscript layout inspection;
  the source itself has not been implicated.

## Blocked items

- exact Veith-to-TDC binary transformation (polarity is supported, transformation
  provenance remains unresolved);
- a primary panel spanning an independent source family, or a prospectively
  narrowed Tox21-family claim;
- immutable Chemprop/MoLFormer/environment lock;
- production Chemprop/MoLFormer lineage instrumentation;
- GPU smoke benchmark;
- user approval for formal protocol tag and full compute.

## Next automatic action

Lock the model/container environment, connect the production trainer to lineage
records, and measure the seed-99 GPU smoke on one Tox21 primary candidate. Do not
inspect an extension test prediction, run seeds 101--110, or create a formal
protocol tag. If no independent source family is prospectively admitted, narrow
the claim to the Tox21 assay family before freeze.

## Protocol deviations

See `protocols/protocol_deviations.md`. All current entries occurred before any
extension prediction was generated.
