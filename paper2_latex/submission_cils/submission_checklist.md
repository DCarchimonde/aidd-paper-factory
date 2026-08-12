# Paper 2 submission checklist

Target journal: Chemometrics and Intelligent Laboratory Systems
Article type: Original Research Article
Publishing route: Subscription (no publication fee charged to authors)

## Files

- Final review manuscript PDF: `CILS_Paper2_TAME_Manuscript.pdf`
- Self-contained editable LaTeX archive: `CILS_Paper2_TAME_LaTeX_Source.zip`
- Bibliography files: `references.bib` and `references_2026.bib`
- Six vector manuscript figure PDFs from the frozen audit/TAME asset package
- Final Supporting Information PDF: `CILS_Paper2_TAME_Supplementary.pdf`
- Supporting Information LaTeX source and generated tables
- `submission_cils/highlights.txt`
- `submission_cils/cover_letter.md`

## Metadata to confirm before entering the submission system

- Full author names confirmed: Siyuan Tong and Yuechen Wang
- Author order and affiliations
- Corresponding-author email and postal address
- ORCID identifiers, when available
- CRediT contribution roles
- Funding and competing-interest declarations
- Three to five suitable reviewer suggestions, avoiding conflicts of interest

## Submission choices

- Select the subscription publishing agreement rather than paid open access
- Upload `CILS_Paper2_TAME_LaTeX_Source.zip` as **LaTeX source files**
- Upload highlights as a separate editable file
- Upload `CILS_Paper2_TAME_Supplementary.pdf` as **Supplementary material**
- Keep `CILS_Paper2_TAME_Manuscript.pdf` as the final review copy; upload it only if the system requests a manuscript PDF in addition to the LaTeX source archive
- Enter the public GitHub repository and `paper2-racer-c4-development-2026` branch in the data/code availability field
- Do not claim a Zenodo DOI or archival identifier unless one is later created

## Final local sync

```powershell
cd E:\AIDD_Paper_Factory
git switch paper2-racer-c4-development-2026
git pull --ff-only origin paper2-racer-c4-development-2026
```

The versioned ZIP and final PDFs are ready after this pull; no local rebuild is required for submission.
