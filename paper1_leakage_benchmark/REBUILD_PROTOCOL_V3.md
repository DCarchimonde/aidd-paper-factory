# Paper 1 rebuild protocol v3

## Status

This protocol supersedes all earlier Paper 1 split, null-audit, and prediction summaries until the v3 manifests and model reruns are complete.

## Clean data policy

- Canonicalize valid SMILES with RDKit using isomeric canonical SMILES.
- Exclude invalid or missing SMILES and nonnumeric targets.
- Classification duplicates with one consistent label are collapsed to one molecule.
- Classification duplicates with conflicting labels are excluded as an unresolved group.
- Regression duplicates are aggregated by the arithmetic mean; source count, range, standard deviation, and row lineage are retained.
- Clean outputs are written under `data/processed_v2` and never overwrite legacy processed files.

## Scaffold identity

- Main analysis: all acyclic molecules share the explicit sentinel scaffold `__ACYCLIC__` (`single_group`).
- Sensitivity analysis: each acyclic molecule receives a molecule-specific scaffold identity (`singleton`) for ESOL and FreeSolv.
- Main and sensitivity results must remain separately labelled.

## Candidate-pool split design

For each partition seed:

1. Generate a target-blind random scaffold candidate pool with a fixed requested budget.
2. Select the size-matched baseline using test-size deviation only.
3. Restrict target-balanced selection to candidates with exactly the same `n_test` as the selected size baseline.
4. Select the candidate with the minimum train-test target-mean gap.
5. Retain the full matched-size candidate-gap distribution as a matched-budget diagnostic, not as independent inferential replicates.

The size-matched and target-balanced partitions must have identical test-set sizes, disjoint train/test scaffolds, complete molecule coverage, and recorded SHA256 partition hashes.

## Candidate-budget audit and stopping rule

Candidate budgets are audited before model training. A single adjacent-budget plateau is not sufficient because a larger nested candidate pool can improve again after a temporary plateau. A lower budget is considered stable only when both of the following hold relative to every larger audited budget:

- mean target-balanced gap changes by no more than 5% relative; and
- absolute mean target-balanced gap changes by no more than 0.02 target units.

Classification datasets were operationally stable from 300 candidates onward and had at least 100 same-size candidates per audit seed. The frozen production budget for BACE, BBBP, ClinTox, and HIV is therefore 300 candidates per partition seed.

For ESOL and FreeSolv under `single_group`, budgets `3000, 5000, 10000, 20000` did not establish stability before the predeclared cap. The frozen production budget is therefore 20000 candidates per partition seed. The manuscript must report the budget-sensitivity analysis and must not claim numerical convergence.

For the ESOL/FreeSolv `singleton` sensitivity, the apparent 500-to-1000 plateau did not remain stable at 3000 and 5000 candidates. The frozen sensitivity budget is therefore the predeclared cap of 5000 candidates per partition seed, again without a convergence claim.

No further candidate-budget escalation is permitted after this freeze.

## Frozen production runs

- Main classification analysis: BACE, BBBP, ClinTox, and HIV; `single_group`; 20 partition seeds; 300 candidates per seed.
- Main regression analysis: ESOL and FreeSolv; `single_group`; 20 partition seeds; 20000 candidates per seed.
- Acyclic-definition sensitivity: ESOL and FreeSolv; `singleton`; 20 partition seeds; 5000 candidates per seed.

All three runs must be produced from `clean_v2`, recorded in separate files, and frozen by SHA256 before model training.

## Statistical unit

- Partition hashes, not requested seeds, define unique partitions.
- Duplicate partitions are not treated as independent observations.
- Candidate-pool members are optimization diagnostics, not independent samples.
- Model stochasticity is separated from partition stochasticity through distinct `partition_seed` and `model_seed` fields.
- Prediction inference is performed at the unique-partition level after averaging repeated model seeds within each partition-model combination.

## Frozen claims pending rerun

The legacy `9 smaller / 6 larger / 3 inconclusive` result, legacy Wilcoxon/Holm tables, legacy similarity summaries, legacy ranking results, and all old manuscript figures are superseded pending the v3 rerun.
