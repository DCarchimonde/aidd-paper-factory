# Paper 1 P0 Rebuild Status

## Status

The previously reported ordinary-scaffold, target-balanced-scaffold, null-audit, similarity, ranking, and paired-significance results are frozen as historical outputs. They must not be used for submission until the audited v2 split manifests and downstream reruns are complete.

## Confirmed reasons for rebuild

1. The legacy largest-groups-first scaffold split can overshoot the nominal 20% test size substantially.
2. Empty Bemis--Murcko scaffold strings can become NaN after a CSV round-trip and be silently excluded by pandas groupby.
3. The previous `gap_reduction` contrast algebraically reduces to the direct balanced-versus-ordinary performance difference; it will be renamed.
4. Partition seeds and model seeds were coupled, and unique partition counts were not reported.
5. Similarity and scaffold-statistics scripts used duplicated split implementations and only five seeds.
6. The original null audit did not match the 300-trial search budget.

## V2 infrastructure

- `shared_utils/scaffold_identity.py`
  - explicit `__ACYCLIC__` sentinel;
  - scaffold recomputation from canonical SMILES;
  - NaN/empty/invalid checks;
  - split assertions and partition hashing.
- `shared_utils/split_search_v2.py`
  - legacy split retained only for sensitivity;
  - common search engine for size-only and target-balanced scaffold splits;
  - identical size constraints and search budgets.
- `shared_utils/split_manifest_v2.py`
  - molecule-level manifests;
  - target-distribution diagnostics;
  - test-set scaffold concentration metrics;
  - acyclic train/test counts.
- `paper1_leakage_benchmark/scripts/00_test_splitting_v2.py`
  - deterministic smoke tests.
- `paper1_leakage_benchmark/scripts/00_audit_split_rebuild_v2.py`
  - pretraining audit over all datasets and partition seeds.

## Required gates before model retraining

1. Smoke tests pass.
2. Every row is assigned exactly once.
3. Scaffold-based train/test sets share no scaffold identifiers.
4. No scaffold value is NaN or empty.
5. Test-size deviations and structural infeasibility are reported explicitly.
6. Requested and unique partition counts are reported for every dataset/protocol.
7. Split hashes and molecule-level manifests are saved.
8. Acyclic single-group and singleton sensitivity audits are compared for ESOL and FreeSolv.
9. Raw duplicate/conflicting-target handling is audited separately from the already processed tables.

## Planned final protocols

1. Random observation split.
2. Legacy deterministic scaffold split, sensitivity analysis only.
3. Size-matched target-blind scaffold split.
4. Target-balanced size-matched scaffold split.

All predictive, similarity, scaffold, ranking, and null analyses will read the same pre-generated manifests. Partition seed and model seed will be separated in the final training design.
