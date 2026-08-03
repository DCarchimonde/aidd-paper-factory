# Paper 2 RACER-C Phase 3 GPU benchmark readiness

Date: 2026-08-04
Status: **implementation ready for target-GPU component timing; GPU not run**

## Phase-2 correction before GPU work

Production wiring exposed a residual label access in the scaffold/cluster role
allocator. The role objective was size-only, but the group-order key still used
the largest within-group class count. The previous test flipped every label, which
only swapped the two class counts and therefore could not expose the dependency.

The corrected order uses total group size, a seeded hash of the group ID, and the
group ID only. A non-complement label permutation is now tested. All 300 role
cells, 1,200 conformal-resolution cells, precision simulations, endpoint
decisions, and the seed-99 CPU smoke were regenerated before any GPU or
confirmatory prediction. NR-AhR changed from 14/15 to 15/15 count cells, producing
four count-only primary candidates: NR-AhR, NR-ER, SR-ARE, and SR-MMP. The selected
`50/20/15/15` allocation and its smallest primary policy critical-class count of
104 did not change.

In addition to the unit test, all five clustered Tox21 endpoints were subjected to
an arbitrary label shuffle for both scaffold and similarity-cluster grouping at
seed 99 and seeds 101--105. Assignments were identical in 60/60 real-data checks.

## Candidate production environment

The target candidate is Python 3.11.13, PyTorch 2.13.0 with CUDA 13.0, Chemprop
2.3.0, Transformers 5.12.1, RDKit 2026.3.4, scikit-learn 1.9.0, and XGBoost 3.3.0.
The IBM Research model is fixed to
`ibm-research/MoLFormer-XL-both-10pct` revision
`361063d0ad524ef77cf39b08469f6be770dc550f`. Frozen embeddings use
`pooler_output` in float32. Token sequences above 202 tokens fail before any fit;
truncation is prohibited.

This is not yet an immutable environment lock. The active runtime auditor requires
the user's Windows NVIDIA GeForce RTX 4060 Laptop GPU, at least 7 GiB visible VRAM,
exact package versions, the expected CUDA build, `nvidia-smi`, a complete `pip
freeze`, and the fixed model revision. Any mismatch yields `fail_closed`.

## Benchmark contract

The committed seed-99 plan uses only the NR-ER strict-scaffold development role:

- development rows: 2,928 (class 0: 2,653; class 1: 275);
- three outer meta-folds of 976 rows each;
- six inner D-MPNN fits plus three outer-final fits per full nested cell;
- four primary endpoints, three tracks, and five main seeds, or 60 primary cells;
- no policy/conformal/test prediction and no performance metric.

The first target-GPU command measures MoLFormer extraction and one representative
outer-final Chemprop fit. Its runtime is converted transparently to six
outer-final-fit equivalents per endpoint/track/seed cell and reported both with
and without a 20% rerun reserve. The projection is an engineering estimate, not a
scientific result.

## Active Windows RTX-4060 Laptop execution target

The user confirmed that the available target is a Windows laptop with an NVIDIA
GeForce RTX 4060 Laptop GPU and that comparable workloads run quickly on it. This
is now the active pre-freeze target in the default environment lock and runbook;
the former RTX-4090 assumption is superseded. The platform adaptation fixes
MoLFormer inference batch size at 8 but does not change scientific inputs, model
revision, Chemprop settings, token policy, endpoint, roles, or seed. The runtime
audit records platform and visible VRAM, while the component benchmark records
memory peaks and fails on input-hash drift.

## Current boundary

The present container has no NVIDIA device, PyTorch, Chemprop, or Transformers, so
the GPU command was not run here. Plan generation, command construction,
development-only lineage, exact version mismatch, token policy, and label-blind
allocation are locally testable. Formal protocol freeze, the complete RACER-C
production chain, and seeds 101--110 remain blocked until the Windows RTX-4060
Laptop component benchmark passes and its measured budget is approved.
