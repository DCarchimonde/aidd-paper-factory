# Paper 1 SAR/QSAR submission-final source

This directory contains the frozen editorial source used by
`paper1_leakage_benchmark/scripts/44_build_paper1_sarqsar_submission_v1.py`.

The finalizer does not refit predictive models or regenerate the empirical partitions. It:

1. audits the completed 200-permutation molecular-null experiment;
2. audits the frozen empirical paired-partition results;
3. builds the explicitly exploratory empirical-null bridge at matched search budgets;
4. generates seven publication figures and three main tables;
5. compiles manuscripts with and without author details, an anonymous Supporting Information file, a title page, and a cover letter;
6. checks references, LaTeX warnings, artwork resolution, and double-anonymous leakage;
7. writes an upload map, machine-readable manifest, audit report, and submission ZIP.

## One-command build

From Windows PowerShell:

```powershell
cmd /c "cd /d E:\AIDD_Paper1_Rebuild && git fetch origin && git switch paper1-sarqsar-metric-coupling-2026 && git pull --ff-only origin paper1-sarqsar-metric-coupling-2026 && python -u paper1_leakage_benchmark\scripts\44_build_paper1_sarqsar_submission_v1.py"
```

Expected final status:

```text
NULL SCIENCE AUDIT: PASS
EMPIRICAL SCIENCE AUDIT: PASS
EMPIRICAL-NULL BRIDGE: PASS
REFERENCE GATE: PASS
DOUBLE-ANONYMOUS SOURCE GATE: PASS
DOUBLE-ANONYMOUS PDF GATE: PASS
PAPER 1 SAR/QSAR SUBMISSION-FINAL BUILD: PASS
```

The final upload folder and ZIP are written at repository root as:

- `paper1_sarqsar_submission_v1/`
- `paper1_sarqsar_submission_v1.zip`
