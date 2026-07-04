# Paper 1 Clean State After JMGM Figure Revision

Date: 2026-07-05

## Current manuscript entry point

The active LaTeX entry file is:

- `paper1_latex/main.tex`

It currently loads:

- `sections/abstract`
- `sections/introduction`
- `sections/related_work`
- `sections/methods`
- `sections/results_jmgm`
- `sections/discussion`
- `statements`
- `appendix_jmgm`

## Active figure-generation script

The active figure-generation script is:

- `paper1_leakage_benchmark/scripts/make_paper_figures.py`

It generates:

- `figure1_split_diagnostic_workflow.png`
- `figure2_generalization_gap.png`
- `figure3_target_shift.png`
- `figure4_tanimoto_similarity.png`
- `figureS1_model_ranking_stability.png`
- `figureS2_outlier_sensitivity_cases.png`

## Clean-up performed

Unused legacy files removed:

- `paper1_latex/appendix.tex`
- `paper1_leakage_benchmark/figures/figure1_gap_ordinary_vs_balanced.png`
- `paper1_leakage_benchmark/figures/figure2_split_target_shift.png`
- `paper1_leakage_benchmark/figures/figure3_tanimoto_similarity.png`

The old `sections/results.tex` file is not present in the current repository state.

## Notes

Figure 1 panel A was revised from a single-row workflow into a two-row workflow to avoid text overlap in the compiled manuscript.

Before compiling after a fresh pull, regenerate figures with:

```powershell
python paper1_leakage_benchmark\scripts\outlier_inspection.py
python paper1_leakage_benchmark\scripts\make_paper_figures.py
```

Then compile:

```powershell
cd paper1_latex
latexmk -C main.tex
latexmk -pdf main.tex
```
