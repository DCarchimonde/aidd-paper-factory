# Paper 1 Runbook

## Goal

Run the AIDD benchmark pipeline for random split vs scaffold split.

The current paper direction is a leakage-aware benchmark audit, not a new drug discovery model claim.

## Current scripts

```text
paper1_leakage_benchmark/scripts/01_prepare.py
paper1_leakage_benchmark/scripts/02_featurize_and_split.py
paper1_leakage_benchmark/scripts/03_rf_baseline.py
paper1_leakage_benchmark/scripts/train_many.py
paper1_leakage_benchmark/scripts/diagnose_splits.py
```

## Run commands

From the repository root:

```bat
conda activate aidd_paper
E:
cd E:\AIDD_Paper_Factory

git pull
python paper1_leakage_benchmark\scripts\01_prepare.py
python paper1_leakage_benchmark\scripts\02_featurize_and_split.py
python paper1_leakage_benchmark\scripts\03_rf_baseline.py
python paper1_leakage_benchmark\scripts\diagnose_splits.py
python paper1_leakage_benchmark\scripts\train_many.py
```

## Main outputs

```text
paper1_leakage_benchmark/results/tables/dataset_summary.csv
paper1_leakage_benchmark/results/tables/split_summary.csv
paper1_leakage_benchmark/results/tables/paper1_rf_baseline_metrics.csv
paper1_leakage_benchmark/results/tables/paper1_split_diagnostics.csv
paper1_leakage_benchmark/results/tables/paper1_top_scaffolds.csv
paper1_leakage_benchmark/results/tables/paper1_many_raw.csv
paper1_leakage_benchmark/results/tables/paper1_many_summary.csv
paper1_leakage_benchmark/results/tables/paper1_gap_raw.csv
paper1_leakage_benchmark/results/tables/paper1_gap_summary.csv
```

## Push only small result tables

```bat
git status
git add paper1_leakage_benchmark\results\tables\*.csv
git commit -m "Add paper1 multimodel audit results"
git push
```

Raw data, processed features, and model files should stay local.
