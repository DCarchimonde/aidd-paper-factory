# Parent-fragment classification sensitivity protocol v3

This protocol is frozen before fitting any parent-fragment sensitivity models. It does not replace the source-faithful Paper 1 main analysis.

## Motivation

The source-faithful clean_v2 audit identified disconnected multi-fragment records in BBBP, ClinTox, and HIV. A deterministic largest-fragment diagnostic showed that fragment selection can alter Morgan fingerprints and, for a minority of records, Bemis--Murcko scaffold identity. Parent-fragment collapsing also creates duplicate parent identities, including conflicting classification labels. Therefore parent-fragment selection is not treated as a lossless cleaning operation.

## Scope

Only BBBP, ClinTox, and HIV are included. BACE contains no multi-fragment clean_v2 records and is not rerun. ESOL and FreeSolv contain no multi-fragment clean_v2 records and are not rerun. The previously frozen source-faithful main analysis remains primary.

## Deterministic parent perturbation

For every clean_v2 molecular record, RDKit disconnected fragments are ranked deterministically by: (1) number of heavy atoms, (2) number of carbon atoms, and (3) canonical isomeric SMILES. The top-ranked fragment is called the algorithmic parent fragment for this sensitivity analysis. This rule is an algorithmic perturbation only; it is not asserted to identify the biologically correct parent for salts, co-crystals, mixtures, or metal complexes.

After parent mapping, parent identities with one unique binary target are collapsed to one record. Parent identities with conflicting 0/1 targets are excluded as ambiguous. No majority vote or keep-first rule is used.

## Split protocol

The transformed dataset is split with acyclic_mode=single_group. For each of the 20 frozen partition seeds used in the main analysis, 300 target-blind random scaffold candidates are generated. The size-matched scaffold baseline is selected without target information. The target-balanced scaffold partition is selected only from candidates having exactly the same test-set size as the paired size-matched baseline, minimizing the train--test target-mean gap. The two paired partitions must have identical test-set size and disjoint train/test scaffold sets.

The candidate budget remains 300 because this sensitivity concerns the three classification datasets for which the main-analysis budget was frozen at 300 before model outcomes were examined.

## Modeling

Only the paired size_matched_scaffold and target_balanced_scaffold protocols are fitted. Models are LR, RF, and XGB using the same Morgan radius-2, 2048-bit representation and the same frozen model hyperparameters as the main analysis. LR uses deterministic seed 0; stochastic RF/XGB use model seed 17. XGB scale_pos_weight is computed from each training partition.

## Inference

The primary sensitivity metric is ROC-AUC. For each dataset-model cell, the paired effect is ROC-AUC(target-balanced) minus ROC-AUC(size-matched), so positive values favor target balancing. The inference unit is the unique partition pair (n=20), not model seeds or candidate partitions. Each of the 9 cells receives a 10,000-replicate paired bootstrap 95% CI and a two-sided Wilcoxon signed-rank p value. Holm correction is applied across these 9 sensitivity cells as a separate family. The sensitivity analysis is interpreted as robustness evidence and does not replace the 18-cell primary family.

## Stopping rule

This is the only parent-fragment standardization sensitivity analysis planned for Paper 1. No alternative fragment ranking, salt-removal library, charge normalization, or additional preprocessing variant will be selected after observing model outcomes. If this deterministic perturbation changes conclusions, the disagreement will be reported as preprocessing sensitivity rather than resolved by choosing a preferred post hoc representation.
