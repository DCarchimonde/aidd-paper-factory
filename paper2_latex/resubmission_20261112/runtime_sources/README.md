# Executed baseline sources

The source collection `TAME_Runtime_Sources_20261112.zip` has SHA-256 `ab087359b7b1a29eb0cbf12536b9dc27a449356f5f277da4ea424e96fbd67a8a`. Its files are retained here with their original bytes. `COLLECTION_REPORT.json` records the collected paths, hashes, and differences.

Both files in `actual_sources/` match the hashes recorded in the formal experiment's `../inputs/provenance.json`. They are also copied byte-for-byte to the active repository paths under `paper2_admet_benchmark/revision_2026/full_baselines/`.

| Executed source | SHA-256 |
|---|---|
| `codrug.py` | `dd0e16ff8c9d76ac797afc7637d23e2320e1d424229f430e3976bf5e49a76f44` |
| `kmm_cp.py` | `e57042e1121be80b49d3d26d63714af2b12cb6a720d69feb0d46c058f1fb7637` |

The reference files were checked against GitHub commit `ae9c3927f0e34f86f58005c4ccea9d50ff5a7376`. The supplied patches identify three implementation changes:

1. CoDrug retries Murcko scaffold extraction after removing stereochemistry when the first extraction raises `RuntimeError`. This addresses the recorded stereochemistry exception during auxiliary-pool preparation.
2. CoDrug moves each saved CUDA random-state tensor to CPU before passing the states to `torch.cuda.set_rng_state_all` during checkpoint restoration.
3. KMM divides both the quadratic matrix and the linear objective term by the largest positive eigenvalue of the quadratic matrix before calling OSQP. Constraints and bounds keep their original scale; the returned objective is converted to the original units. This positive scaling preserves the mathematical minimizer. Solver tolerances and residuals apply to the scaled objective.

## Earlier completed CoDrug cells

The formal record identifies 120 cells completed before the scaffold retry was added. `pre_scaffold_fix/codrug.py` was reconstructed from the supplied reference source and the observed CUDA state-restoration change, preserving the recorded CRLF bytes. Its SHA-256 is `4f902c30f23548021f7c3ded5ba8c802bbf1164f6b68c26f9b1944d23eca7760`, exactly matching `../inputs/provenance.pre_scaffold_fix.json`. It is a reconstructed, hash-matched snapshot, not an additional file supplied by the execution machine.

## Verification

`../RUNTIME_SOURCE_VERIFICATION.json` records the source matches and the 21 passing checks; `verification_tests.txt` contains the test output. Checks cover the existing KMM and weighted-set behavior, auxiliary-pool chemistry, a known constrained quadratic optimum and objective units, the scaffold retry, and CPU conversion in the saved CUDA-state expression. The CUDA expression was exercised with mock states; this was not a GPU checkpoint-resume run.

From a full repository checkout, with its baseline dependencies installed, run:

```text
python paper2_latex/resubmission_20261112/verify_runtime_sources.py
```

The full repository test modules are required for this command. The numerical supplement includes the source snapshots for inspection and the manuscript's fixed-input rebuild separately. No model was retrained and no frozen prediction was recomputed during this reconciliation. `../FROZEN_CONTENT_VERIFICATION.json` records the unchanged numerical inputs, tables, and figures. The supporting information documents the objective scaling; later authorship and editorial changes are recorded in `../EDITORIAL_REVIEW.md`.
