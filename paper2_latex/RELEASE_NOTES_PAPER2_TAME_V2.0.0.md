# Paper 2 integrated TAME package v2.0.0

## Title

**Beyond Aggregate Reliability in ADMET Prediction: Auditing Chemical Shift and Protecting Weakest-Class Coverage with RACER-C4/TAME**

## Purpose

This development release integrates the frozen four-endpoint reliability audit with the frozen RACER-C4/TAME development and independent EPA result. It contains the manuscript source, Supporting Information source, vector figures, reporting-only figure builder, executable lock, cryptographic audit records, frozen summaries, and build instructions.

## Evidence stages

### Stage I: confirmatory diagnostic audit

- Endpoints: BBBP, ClinTox, ESOL, and Lipophilicity.
- Split designs: random, label-blind scaffold, and label-blind similarity-cluster.
- Random/scaffold confirmatory seeds: 101--110.
- Similarity-cluster confirmatory seeds: 101--105.
- Development seed 99 excluded from scientific conclusions.
- Paired comparisons within endpoint, split, seed, model, and regime.

### Stage II: RACER-C4/TAME

- Public Tox21 leaderboard batch: architecture development only, seeds 101--105.
- Final EPA batch: one label-firewalled evaluation, fresh seeds 211--215.
- Primary endpoints: NR-AhR, NR-ER, SR-ARE, SR-ATAD5, SR-MMP, and SR-p53.
- Primary minimum-class-coverage change: +1.3649 percentage points.
- Deterministic hierarchical-bootstrap 95% interval: +0.5827 to +2.0051 points.
- MacroCSY change: -1.6067 points, inside the frozen -5-point bound.
- Approved physicochemical and score views: active in 60/60 final cells.
- Diagnostic ECFP view: failed its frozen certificate in 60/60 final cells.

## Pull and build

~~~powershell
cd E:\AIDD_Paper_Factory
git switch paper2-racer-c4-development-2026
git pull --ff-only origin paper2-racer-c4-development-2026

cd paper2_latex
latexmk -C main.tex
latexmk -pdf -bibtex -interaction=nonstopmode -file-line-error -halt-on-error main.tex
~~~

All six figure PDFs are committed, so VS Code/LaTeX Workshop can compile the manuscript without running Python. Optional reporting-only regeneration uses:

~~~powershell
conda activate aidd_paper
python paper2_admet_benchmark/scripts/34_build_main_figures.py
~~~

## Claim boundary

This package does not authorize claims of exact coverage under arbitrary shift, conditional coverage, clinical safety, or universal superiority. The final result is a coverage gain with efficiency non-inferiority in the locked Tox21 evaluation. Null and adverse endpoint results remain included.

## Archival status

No archival DOI is claimed in the manuscript. A DOI should be added only after an actual repository release has been archived.
