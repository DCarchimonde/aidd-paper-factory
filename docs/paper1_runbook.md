# Paper 1 Runbook

## Goal

Run the first AIDD benchmark pipeline for random split vs scaffold split.

## Current scripts

```text
paper1_leakage_benchmark/scripts/01_prepare.py
paper1_leakage_benchmark/scripts/02_featurize_and_split.py
paper1_leakage_benchmark/scripts/03_rf_baseline.py
```

## Required raw CSV files

Put these files into:

```text
paper1_leakage_benchmark/data/raw/
```

Expected filenames:

```text
bbbp_raw.csv
bace_raw.csv
clintox_raw.csv
esol_raw.csv
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
```

## Main output

```text
paper1_leakage_benchmark/results/tables/paper1_rf_baseline_metrics.csv
```

This table is the first result table for the SCI paper.
