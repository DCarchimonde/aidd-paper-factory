# Paper 1 SAR/QSAR metric-coupling enhancement

Branch: `paper1-sarqsar-metric-coupling-2026`

This branch converts the original paired molecular benchmark audit into a stronger QSAR-validation study centered on split-objective–metric coupling.

## Frozen additions

- `SARQSAR_METRIC_COUPLING_PROTOCOL_V1.md`
- `SARQSAR_METRIC_COUPLING_CONFIG_V1.json`
- molecular endpoint-permutation null simulation preserving real scaffold geometry
- nested requested-draw candidate budgets
- regression MSE decomposition
- classification Brier/log-loss/AP/AUC coupling analysis
- formal QSAR benchmark minimum-reporting checklist
- double-blind scientific working draft and separate title page
- checkpointed one-command overnight runner

## One-command full run

From Windows PowerShell:

```powershell
cmd /c "cd /d E:\AIDD_Paper1_Rebuild && git fetch origin && git switch paper1-sarqsar-metric-coupling-2026 && git pull --ff-only origin paper1-sarqsar-metric-coupling-2026 && python -u paper1_leakage_benchmark\scripts\40_run_paper1_sarqsar_overnight_v1.py"
```

The default run uses 200 endpoint permutations and all 20 frozen partition seeds. It does not refit predictive models. Seed-level checkpoints allow an interrupted run to resume.

## Main outputs

- `paper1_leakage_benchmark/results/sarqsar_metric_coupling_v1/reports/SARQSAR_NULL_SIMULATION_REPORT.md`
- `paper1_leakage_benchmark/results/sarqsar_metric_coupling_v1/tables/null_metric_effect_summary.csv`
- `paper1_leakage_benchmark/results/sarqsar_metric_coupling_v1/figures/`
- `paper1_sarqsar_latex/build/main.pdf`
- `paper1_sarqsar_latex/build/title_page.pdf`
- `paper1_sarqsar_latex/build/supplementary.pdf`
- `paper1_sarqsar_overnight_bundle_v1.zip`
- `paper1_sarqsar_overnight_run_v1.log`

## Resume behavior

A rerun with the same protocol, configuration, and clean-data hashes skips validated seed checkpoints. A protocol/config mismatch is blocked rather than silently overwriting v1 results.

## Smoke test

A non-authoritative smoke run can be made with:

```powershell
python -u paper1_leakage_benchmark\scripts\40_run_paper1_sarqsar_overnight_v1.py --permutations 10
```

Do not mix smoke-test and full-run outputs in the same v1 result directory. The full overnight command above is the authoritative run.
