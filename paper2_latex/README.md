# Paper 2 LaTeX manuscript

## Main file

Open `paper2_latex/main.tex` in VSCode and build with LaTeX Workshop.

## Recommended command-line build

From the repository root:

```powershell
cd E:\AIDD_Paper_Factory\paper2_latex
latexmk -pdf -interaction=nonstopmode -file-line-error main.tex
```

Clean generated files:

```powershell
latexmk -C
```

## Figure source

The manuscript does not duplicate figure binaries. `main.tex` reads the frozen PDF figures directly from:

```text
../paper2_admet_benchmark/results/manuscript_assets/figures/
```

Run the figure builder before compiling the manuscript when the frozen figures are updated:

```powershell
cd E:\AIDD_Paper_Factory
python paper2_admet_benchmark/scripts/34_build_main_figures.py
```

## Current status

- Abstract: first frozen-evidence draft
- Introduction: structural draft; verified literature citations pending
- Methods: grounded scaffold; exact dataset counts and configuration details pending extraction from frozen manifests
- Results: full draft from frozen confirmatory evidence
- Discussion: first complete draft
- Figures 1--6: linked from frozen manuscript assets
- Bibliography: awaiting verified reference audit
