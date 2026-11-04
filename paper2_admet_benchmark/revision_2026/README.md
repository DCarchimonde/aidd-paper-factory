# CHEMOLAB revision experiment code

**二作请先阅读 [START_HERE_CN.md](START_HERE_CN.md)：完整中文实验计划、命令、交回清单与投稿前要求。**

This directory adds experiments in response to CHEMOLAB-D-26-01118. It leaves
the submitted implementation, data contracts and historical results intact.
The existing EPA cohort is **retrospective**: changing seeds or writing new
hashes does not make it a new blind test. The supplied code is an experiment
delivery, not evidence of improved molecular performance.

## Install and numerical verification

Use a new Python 3.12 environment. These are CPU experiments; a GPU is not
required. The commands below run from the repository root on Windows,
Linux or macOS (replace `python` with the desired interpreter if necessary).
The inherited chemical-standardization contract requires **RDKit 2026.03.4**
(`rdkit==2026.3.4` in pip). `prepare` and `run` check this before downloading;
a newer RDKit must not silently redefine molecular identities.

```sh
python -m pip install -r paper2_admet_benchmark/revision_2026/requirements.txt
python -m unittest discover -s paper2_admet_benchmark/revision_2026 -p "test_*.py" -v
python -m unittest discover -s paper2_admet_benchmark/revision_2026/tests -p "test_*.py" -v
python -m paper2_admet_benchmark.revision_2026.runner smoke --jobs 2
```

`smoke` uses synthetic numbers only. The tests include a mocked-source
end-to-end run with actual estimator fitting, decisions, file writes,
summary-schema validation and verified resume. Neither reports assay results.

## Acquire data and run

```sh
python -m paper2_admet_benchmark.revision_2026.runner prepare
python -m paper2_admet_benchmark.revision_2026.runner run --phase development --endpoints Tox21_NR_AhR --seeds 101 --models logistic --alphas 0.1 --jobs 2 --output .local/revision_pilot
python -m paper2_admet_benchmark.revision_2026.runner run --phase development --jobs 4 --sensitivities --output .local/revision_development
python -m paper2_admet_benchmark.revision_2026.runner run --phase retrospective_epa --jobs 4 --sensitivities --output .local/revision_epa
```

The full default matrix uses all 12 official endpoints, 5 specified seeds,
three predictors (`logistic`, the original learned `stack`, and `equal_logit`),
and alpha 0.05/0.10/0.20. The four component estimators are fitted once per
endpoint/seed and shared by `stack` and `equal_logit`. All methods within a
predictor share fit/calibration roles and probabilities. The logistic score
view is its one logit; ensemble score views contain four component logits
and the aggregate logit. This dimensionality difference is explicit.

`--sensitivities` adds single-factor perturbations of every legacy audit
threshold. It does not select a best threshold. A failed gate returns the
same empty-repaired baseline and remains an observed result. Fixed-budget
confidence, similarity and random rejectors are always computed. Matched
comparators expand exactly the same numbers of baseline `{0}` and `{1}`
singletons as the reference, using the entire eligible unlabeled target
cohort. Counts on the observed-label subset can differ when labels are
missing; `target_eligible_*` fields record the actual operational budget.
The additional `matched_*_omitted_label_probability` comparator ranks the
probability of the missing label (`p1` for `{0}`, `1-p1` for `{1}`). It is a
stronger direct error-ranking control when a Mondrian singleton disagrees with
the probability argmax, and should be included in the primary comparison.

Optionally add `--density-components kde kmm` to a **new output directory**
for fixed-parameter physicochemical KDE and KMM component comparisons. KDE
uses estimated test weight as an infinity atom. KMM uses a weighted empirical
quantile without that atom and has no finite-sample guarantee. These are
diagnostic components, not complete reproductions of CoDrug or KMM-CP.
KMM uses a dense QP and may be slow; a size cap or solver failure stops the
run rather than silently switching methods. The official full baselines
still require the external-adapter workflow and provenance specified in the
handoff before superiority can be claimed.

The default cache is `.local/chemolab_revision_20261104/cache`. `--cache`
can point to another directory. Downloads must match the historical SHA-256
values. Changed hashes are errors, not permission to accept different data.
The processed cache records the standardization code and RDKit version.
If a cache is incomplete or incompatible, use a new cache directory rather
than deleting or repurposing the submitted data. Raw/processed data and
row-level outputs stay local; they are not added to Git.

## Resume and output

Repeat the identical run command with `--resume`. Completed cell files are
verified by content hash. A change in protocol, code, data, versions or
requested matrix requires a new output directory. An interrupted cell
without a completion receipt is recomputed within the same owned run.

Each `cells/PHASE__ENDPOINT__SEED__MODEL/` contains:

- `predictions.csv`: every target ID, standardized structure, eligibility,
  outcome (blank if missing), probability and prediction set per method/alpha.
- `target_manifest.csv`: the same target universe without outcomes, for
  external reference predictions.
- `calibration.csv` and `roles.csv`: probabilities/labels and exact source
  roles needed to compare calibration procedures on common predictions.
- `metrics.csv`: full denominators, class-specific coverage, false clearance,
  false discard, accepted-error rates and deferral/clearance yields.
- `audits.json`, `component_audits.json`, `complete.json`: diagnostics and
  the completed-file checksums.

The run root contains `provenance.json`, `protocol_snapshot.json`,
`all_metrics.csv` and `run_complete.json`. Capture console output separately
if a full training log is desired. No missing true class is replaced with a
one-class minimum: two-class aggregate metrics are undefined in that case.
Invalid structures and source overlaps remain visible as excluded full sets.

`label1_active` is the primary new protection policy. Here `1` means an assay
active call. A `{0}` prediction is treated as an illustrative clearance
decision; it is not proof of chemical safety. Historical label-0 and symmetric
policies are retained as prespecified comparisons. Results must decide whether
transport weighting locates errors better than budget-matched simple rules.

Use `summarize.py --help` for paired, shared-structure bootstrap comparisons.
The fixed endpoint panel and repeated fitted seeds are not independent
replications. Intervals on the already inspected EPA cohort are exploratory.
