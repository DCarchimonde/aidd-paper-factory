# Paper 1 — SAR/QSAR metric-coupling enhancement

This branch converts the original benchmark-construction audit into a stronger QSAR-validation study centered on **split-objective–metric coupling**.

## What the one-command pipeline does

1. validates the pre-outcome-frozen simulation protocol and six frozen clean data sets;
2. runs a molecular null-permutation experiment that preserves real molecules, scaffold geometry, group-size distributions, and endpoint marginals while destroying molecule–endpoint association;
3. uses nested target-blind candidate pools and exact-size paired response-aware selection;
4. aggregates effects at the endpoint-permutation level, not the seed row level;
5. audits MSE decomposition, exact-size pairing, provenance, completeness, and resume checkpoints;
6. builds publication figures;
7. drafts a double-blind manuscript, title page, Supporting Information, cover letter, related-manuscript disclosure, and benchmark reporting checklist;
8. creates an anonymized reviewer code package and a target-journal upload folder.

No molecular model is fitted by the new null experiment, no GPU is required, and completed 10-permutation blocks are reused after interruption.

## Frozen protocol

- 200 endpoint permutations per data set
- 20 partition seeds
- BACE, BBBP, ClinTox, HIV, ESOL, and FreeSolv
- regression single-group and singleton acyclic semantics
- nested candidate budgets declared in `protocol/simulation_protocol_v1.json`
- protocol SHA-256 verified before every outcome-producing step

## Run

From the repository root:

```powershell
python paper1_sarqsar\scripts\99_run_all_sarqsar.py --workers 4
```

The process prevents Windows system sleep while active. It is resumable: rerun the same command after interruption.

## Key outputs

```text
paper1_sarqsar/results/
paper1_sarqsar/build/sarqsar_manuscript/
paper1_sarqsar_submission_package/
```

The final package is double-blind. The main manuscript excludes names, affiliations, ORCID identifiers, and public-account identifiers. The title page and related-manuscript disclosure are separate editor-facing files.

## Scientific separation from Paper 2

Paper 1 studies validation-partition construction, metric coupling, scaffold semantics, and molecular-record policy. Paper 2 develops the TAME post-prediction conformal intervention for weakest-class coverage under distribution shift. The two manuscripts share some public data sets but have different hypotheses, methods, primary analyses, figures, tables, and conclusions.
