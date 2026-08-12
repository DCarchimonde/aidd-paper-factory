# Journal of Chemometrics submission checklist — Paper 1

Final pre-submission target: **Original Research Article**.

## Manuscript-format gates

- Double line spacing.
- 3 cm margins on all edges.
- Main manuscript <= 25 double-spaced pages.
- <= 7 main illustrations and <= 4 main tables.
- Short title <= 70 characters: `Chemometric Audit of Molecular Benchmark Construction`.
- Summary/abstract <= 250 words.
- 3–5 keywords.
- Main tables use Roman numerals.
- In-text references use superior numbers.
- Reference list follows Journal of Chemometrics examples, with ACS/CASSI-style abbreviated journal titles and `et al.` after the first three authors when there are more than six authors.

## Artwork gates

- All final graph artwork is generated with an Arial-family sans-serif font.
- Source artwork is generated at intended journal reproduction size and automatically checked against the 140 mm × 200 mm maximum envelope.
- Vector PDF is retained for LaTeX/reproduction use.
- A 600-dpi LZW-compressed TIFF companion is generated for every main and supplementary figure.
- Figure 1 card text is programmatically fitted inside its card; the final build fails if a card label cannot fit.

## Reference audit

The 24 references in the submission list were rechecked against publisher/PubMed/Zenodo metadata before the final formatting pass. Two metadata issues in the earlier working bibliography were corrected in the submission list:

1. SIMPD: Gregory A. Landrum and Maximilian Beckers are the correct first two author names.
2. Netzeva et al. (ECVAM Workshop 52): the submission list uses the publisher/PubMed author metadata and the full workshop-report title.

The RDKit website-only entry was replaced for the submission list by the exact software release used in the final environment: **RDKit Release 2026.03.4**, Zenodo DOI **10.5281/zenodo.21291217**.

## Files to upload

For a LaTeX submission through Wiley Research Exchange, use the final build bundle:

- `main.pdf` — Main Document - LaTeX PDF (peer-review PDF)
- `latex_source/main.tex` plus all files in `latex_source/` — Main Document - LaTeX source/supporting files
- `supplementary.pdf` — Supporting Information
- `figures/*.tiff` — separate production-resolution artwork
- `figures/*.pdf` — vector artwork retained with the LaTeX supporting files
- `BUILD_MANIFEST.json` — exact Git commit and build audit

Do not submit an older manuscript PDF or artwork from an earlier visual-build directory.
