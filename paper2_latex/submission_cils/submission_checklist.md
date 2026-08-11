# Paper 2 submission checklist

Target journal: Chemometrics and Intelligent Laboratory Systems
Article type: Original Research Article
Publishing route: Subscription (no publication fee charged to authors)

## Files

- Main manuscript PDF compiled from `paper2_latex/main.tex`
- Editable LaTeX source files under `paper2_latex/`
- Bibliography files: `references.bib` and `references_2026.bib`
- Six vector manuscript figure PDFs from the frozen audit/TAME asset package
- Supporting Information PDF compiled from `paper2_latex/supplementary.tex`
- Supporting Information LaTeX source and generated tables
- `submission_cils/highlights.txt`
- `submission_cils/cover_letter.md`

## Metadata to confirm before entering the submission system

- Full given and family names for both authors
- Author order and affiliations
- Corresponding-author email and postal address
- ORCID identifiers, when available
- CRediT contribution roles
- Funding and competing-interest declarations
- Three to five suitable reviewer suggestions, avoiding conflicts of interest

## Submission choices

- Select the subscription publishing agreement rather than paid open access
- Upload editable `.tex` sources as well as the review PDF
- Upload highlights as a separate editable file
- Upload Supporting Information as a separate file
- Enter the public GitHub repository and RACER-C4/TAME branch in the data/code availability field
- Do not claim a Zenodo DOI or archival identifier unless one is later created

## Final local build

```powershell
cd E:\AIDD_Paper_Factory
git switch paper2-racer-c4-development-2026
git pull --ff-only origin paper2-racer-c4-development-2026
conda activate aidd_paper
cd paper2_latex
latexmk -C main.tex
latexmk -pdf -bibtex -interaction=nonstopmode -file-line-error -halt-on-error main.tex
```

The final PDF and Supporting Information should be checked for figure legibility, float placement, reference completeness, and consistency with the frozen publication report.
