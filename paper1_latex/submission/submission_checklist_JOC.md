# Journal of Chemometrics Submission Checklist

## Article identity

- Journal: Journal of Chemometrics
- Article type: Original Research Article
- Title: Chemometric Diagnosis of Data-Splitting Effects in Molecular Property Prediction Benchmarks
- Short title: Split Diagnostics for Molecular Property Benchmarks
- Corresponding author: Siyuan Tong
- Co-author: Yuechen Wang

## Current manuscript limits

- Double-spaced manuscript: 22 pages before the two added Journal of Chemometrics references; recompile and confirm final page count remains at or below 25.
- Figures: 5, below the limit of 7.
- Tables: 3, below the limit of 4.
- Abstract: 229 words, below the 250-word limit.
- Keywords: 5.

## Required declarations already included

- Data and Code Availability
- Competing Interests
- Author Contributions
- Declaration of AI-Assisted Technologies
- Funding
- Acknowledgements

## LaTeX upload designations in Research Exchange

1. Upload `main.tex` as `Main Document – LaTeX .tex File`.
2. Upload the newly compiled `main.pdf` as `Main Document – LaTeX PDF`.
3. Upload all other `.tex` section files, `.bib` files, and figure files as `LaTeX Supplementary File`.
4. Upload the cover letter separately where requested.

## LaTeX source files

- `main.tex`
- `statements.tex`
- `appendix_chemometrics.tex`
- `sections/abstract_chemometrics.tex`
- `sections/introduction_chemometrics.tex`
- `sections/related_work_chemometrics.tex`
- `sections/methods_chemometrics.tex`
- `sections/results_chemometrics.tex`
- `sections/results_chemometrics_target.tex`
- `sections/results_chemometrics_performance.tex`
- `sections/results_chemometrics_similarity.tex`
- `sections/discussion_chemometrics.tex`
- `references.bib`
- `references_extra.bib`

## Figure files referenced by the manuscript

- `figure1_split_diagnostic_workflow.png`
- `figure_robustness20_split_effects.pdf` or the corresponding high-resolution PNG
- `figure4_tanimoto_similarity.png`
- `figureS1_model_ranking_stability.png`
- `figureS2_outlier_sensitivity_cases.png`

Upload figures as separate files even though they are embedded in the review PDF.

## Supporting submission materials

- `submission/cover_letter_JOC.txt`
- `submission/highlights_JOC.txt` if the portal requests highlights
- Public repository: `https://github.com/DCarchimonde/split-diagnostic-molecular-benchmark`

## Final local checks before submission

```powershell
cd E:\AIDD_Paper_Factory
git pull --rebase origin main
cd paper1_latex
latexmk -pdf -interaction=nonstopmode main.tex
Select-String -Path main.log -Pattern "LaTeX Error|Undefined control sequence|Citation.*undefined|Reference.*undefined|Overfull"
```

Confirm that:

- no citations or references are undefined;
- the final PDF is no more than 25 pages;
- all five figures are readable at normal zoom;
- title, author order, affiliations, corresponding email, and ORCID are correct;
- the repository is public and displays the same title and current reproducibility files;
- the manuscript is not under consideration elsewhere;
- both authors approve the submitted version.

## Submission choices

- Special issue: No, unless intentionally submitting to a currently open themed issue.
- Open access: choose the subscription/non-open-access route unless funding or an institutional agreement is confirmed.
- Preprint/Under Review option: decline unless both authors deliberately choose public preprint posting.
- Transparent peer review: review the portal wording; participation may be optional.
