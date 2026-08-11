# Paper 2 LaTeX manuscript

## Pull the manuscript branch

~~~powershell
cd E:\AIDD_Paper_Factory
git switch paper2-racer-c4-development-2026
git pull --ff-only origin paper2-racer-c4-development-2026
~~~

## Compile directly in VS Code

Open `paper2_latex/main.tex` and run **LaTeX Workshop: Build LaTeX project**. The manuscript uses `latexmk` and BibTeX. All six publication figure PDFs are versioned in the repository, so compiling the article does **not** require Python, model fitting, or figure regeneration.

If a clean command-line build is useful:

~~~powershell
cd E:\AIDD_Paper_Factory\paper2_latex
latexmk -C main.tex
latexmk -pdf -bibtex -interaction=nonstopmode -file-line-error -halt-on-error main.tex
~~~

The main figure path is:

~~~text
../paper2_admet_benchmark/results/manuscript_assets/figures/
~~~

## Optional: rebuild the six figures

The reporting script reads only frozen Stage I tables and frozen RACER-C4/TAME summaries. It does not fit a model, regenerate a prediction, open a sealed label, or change an inferential quantity.

~~~powershell
cd E:\AIDD_Paper_Factory
conda activate aidd_paper
python paper2_admet_benchmark/scripts/34_build_main_figures.py
~~~

The entrypoint creates both vector PDF and high-resolution PNG assets and refreshes `main_figure_integrity_manifest.csv`.

## Supporting Information

The audit tables are generated from the versioned manuscript CSVs before `supplementary.tex` is compiled:

~~~powershell
cd E:\AIDD_Paper_Factory
conda activate aidd_paper
python paper2_admet_benchmark/scripts/36_build_clean_supporting_information.py

cd paper2_latex
latexmk -C supplementary.tex
latexmk -pdf -interaction=nonstopmode -file-line-error -halt-on-error supplementary.tex
~~~

The builder writes `paper2_latex/generated_supplementary_tables.tex`. This generated file is intentionally ignored by Git; the frozen CSVs and builder are authoritative.

## Evidence boundary

- Stage I: four public endpoints; confirmatory seeds 101--110 for random/scaffold and 101--105 for cluster splitting; seed 99 excluded.
- Stage II development: public Tox21 leaderboard cohort, six primary endpoints, seeds 101--105.
- Stage II independent evaluation: final EPA cohort, the same six primary endpoints, fresh seeds 211--215.
- Final predictions and the label-blind transport audit were hashed before final labels were opened.
- Publication inference uses the deterministic repaired interval; no model, prediction, label, point estimate, or efficiency estimate changed during that repair.
- TAME does not claim exact coverage under arbitrary shift, conditional coverage, clinical safety, or universal superiority.

## Manuscript package

- `main.tex`: submission manuscript
- `supplementary.tex`: Supporting Information
- `sections/`: abstract, introduction, methods, results, and discussion
- `references.bib` and `references_2026.bib`: bibliography
- `submission_cils/`: cover letter, highlights, and checklist for *Chemometrics and Intelligent Laboratory Systems*
