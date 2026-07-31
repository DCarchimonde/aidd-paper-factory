# Paper 2 Submission Package v1.0.0

## Title

**Beyond Accuracy in ADMET Prediction: Applicability-Domain Diagnostics and Conformal Calibration under Chemical Distribution Shift**

## Purpose of this release

This release freezes the reproducibility package accompanying the Paper 2 journal submission. It is intended to provide a stable, citable snapshot of the code, confirmatory result tables, figure-generation workflow, manuscript source, and Supporting Information source used for the submitted study.

## Frozen confirmatory scope

- Endpoints: BBBP, ClinTox, ESOL, and Lipophilicity.
- Split designs: random, label-blind scaffold, and label-blind similarity-cluster splits.
- Random and scaffold confirmatory seeds: 101--110.
- Similarity-cluster confirmatory seeds: 101--105.
- Development seed 99 is excluded from scientific conclusions.
- Model families are not treated as independent inferential replicates.
- Method comparisons are paired within matched endpoint, split, seed, model, and classification imbalance regime cells.

## Included submission artifacts

- Main manuscript LaTeX source under `paper2_latex/`.
- Standalone Supporting Information source and automated table builder.
- Frozen manuscript-ready result tables under `paper2_admet_benchmark/results/manuscript_assets/tables/`.
- Figure-generation scripts and publication-ready figure assets.
- Result-integrity manifest with expected row counts and SHA-256 hashes.
- Reproducibility documentation and build commands.

## Rebuild commands

From the repository root:

```powershell
conda activate aidd_paper
python paper2_admet_benchmark/scripts/34_build_main_figures.py
python paper2_admet_benchmark/scripts/36_build_clean_supporting_information.py
```

Build the manuscript and Supporting Information from `paper2_latex/`:

```powershell
latexmk -C
latexmk -pdf -bibtex -interaction=nonstopmode -file-line-error -halt-on-error main.tex
latexmk -C supplementary.tex
latexmk -pdf -interaction=nonstopmode -file-line-error -halt-on-error supplementary.tex
```

## Evidence boundary

This release does not refit models, rebuild data splits, re-estimate conformal thresholds, or select results after inspection. The submission-facing tables and figures are generated from the frozen confirmatory evidence package. Sparse and adverse results are retained.

## Integrity

The authoritative integrity manifest is:

`paper2_admet_benchmark/results/manuscript_assets/final_results_integrity_manifest.csv`

It records the expected row count, validation status, and full SHA-256 hash for each manuscript-ready result table.

## Archival DOI

A version-specific DOI will be assigned by Zenodo after this GitHub release is archived. The DOI should be inserted into the manuscript Data and Code Availability statement before final journal submission.
