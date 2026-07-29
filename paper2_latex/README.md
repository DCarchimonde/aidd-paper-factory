# Paper 2 LaTeX manuscript

## Main file

Open `paper2_latex/main.tex` in VSCode and build with LaTeX Workshop, or use the command-line workflow below.

## Recommended final refresh

From the repository root:

```powershell
cd E:\AIDD_Paper_Factory
git pull origin main
conda activate aidd_paper
python paper2_admet_benchmark/scripts/34_build_main_figures.py
```

The figure refresh does not refit models or change frozen numerical results. It rebuilds the publication assets from frozen manuscript tables and reuses frozen figures when the large row-level selective-curve source is unavailable.

## Recommended manuscript build

```powershell
cd E:\AIDD_Paper_Factory\paper2_latex
latexmk -C
latexmk -pdf -bibtex -interaction=nonstopmode -file-line-error -halt-on-error main.tex
```

Clean generated files when needed:

```powershell
latexmk -C
```

## Figure source

The manuscript does not duplicate figure binaries. `main.tex` reads the PDF figures directly from:

```text
../paper2_admet_benchmark/results/manuscript_assets/figures/
```

## Evidence boundary

Scientific conclusions are restricted to frozen confirmatory outputs. Development checks and seed 99 are excluded from manuscript conclusions. Random and scaffold analyses use confirmatory seeds 101--110; similarity-cluster analyses use seeds 101--105. Model families are not treated as independent inferential replicates, and method contrasts are paired within endpoint, split, model, regime, and seed.

## Current status

- Confirmatory experiments: frozen
- Main numerical result package: integrity checks passed
- Abstract, Introduction, Methods, Results, Discussion, and Conclusion: complete and under final language polishing
- Figures 1--6: publication assets linked and integrity-manifested
- Supporting Information: available in `paper2_latex/supplementary.tex`
- Bibliography: 41 cited entries; duplicate, missing-key, metadata, and unused-entry audits passed
- Remaining pre-submission tasks: clean recompilation, visual PDF inspection, repository archival release/persistent identifier, and journal submission packaging
