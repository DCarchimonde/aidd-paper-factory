# Paper 1 — Q1 final build

Run from the repository root:

```powershell
python paper1_leakage_benchmark\scripts\17_build_paper1_visual_submission_v3.py
```

The legacy script-17 entry point now delegates to the Q1 final pipeline in script 22.

## What the build does

1. Completes the predeclared RF/XGB stochastic-model sensitivity for model seeds 29 and 43 on partition seeds 42, 123, 2024, 2026, and 3407. Existing completed jobs are reused.
2. Regenerates the authoritative primary/supporting metric summaries from production jobs only.
3. Computes the mean-only regression control from frozen manifests and verifies the exact MSE decomposition identity.
4. Computes paired collateral target/scaffold diagnostics from the frozen partitions.
5. Builds closed raw-to-clean accounting tables.
6. Captures the exact software environment used by the final build.
7. Regenerates publication-size main and supplementary figures.
8. Enforces source-level submission gates for abstract length, keyword count, running-title length, figure/table counts, and overstrong legacy claim wording.
9. Compiles the main manuscript and Supporting Information.
10. Runs a post-build gate for missing PDFs, undefined citations/references, and the target-journal double-spaced page limit when a local PDF page-count reader is available.
11. Packages PDFs, key result tables, figure PDFs, and a Git commit/build manifest.

## Outputs

```text
paper1_submission_q1_final_v3/
  main.pdf
  supplementary.pdf
  BUILD_MANIFEST.json
  figures/
  tables/
```

The final immutable Git tag/release should be created only after the compiled PDFs have passed visual inspection.
