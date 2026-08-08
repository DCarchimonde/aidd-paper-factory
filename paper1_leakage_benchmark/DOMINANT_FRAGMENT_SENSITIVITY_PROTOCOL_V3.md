# Dominant-Fragment Representation Sensitivity Protocol V3

## Status

This protocol is frozen after the primary source-faithful v3 analysis was completed and after the post hoc multi-fragment audit identified disconnected-component sensitivity in the classification datasets. It is a **sensitivity analysis only** and does not replace, re-label, or re-fit the primary source-faithful analysis.

## Motivation

The source-faithful clean_v2 datasets retain disconnected components present in the benchmark records. A post hoc audit found multi-fragment records in BBBP, ClinTox, and HIV, and showed that selecting a single dominant fragment can alter Morgan fingerprints, occasionally alter Bemis--Murcko scaffold identity, and collapse distinct source records onto the same single-fragment representation. Some of those collapses have conflicting classification labels.

Therefore, single-fragment representation is not treated as an innocuous cleaning step. It is evaluated separately as a robustness sensitivity.

## Operational dominant-fragment definition

For every clean_v2 classification record:

1. Parse the full canonical SMILES with RDKit.
2. Enumerate disconnected molecular fragments.
3. Rank fragments deterministically by:
   - number of heavy atoms, descending;
   - number of carbon atoms, descending;
   - canonical isomeric SMILES, descending as the final deterministic tie-break.
4. Select the top-ranked fragment as the **dominant fragment**.
5. Canonicalize the selected fragment with RDKit.

This operational definition is deliberately called `dominant fragment`, not `true parent molecule`. For salts and counterions it often corresponds to the expected parent-like organic component, but no claim is made that it chemically resolves every mixture, complex, co-crystal, or multi-component record.

## Duplicate and label-conflict policy after dominant-fragment mapping

The mapping may cause multiple source-faithful records to collapse onto the same dominant-fragment SMILES.

For classification datasets:

- if all mapped rows have the same binary label, collapse them to one dominant-fragment record;
- if mapped rows contain conflicting labels, exclude the entire dominant-fragment group from this sensitivity dataset;
- record complete lineage, source-row counts, and conflict decisions.

No target value is changed or imputed.

## Scope

Sensitivity datasets:

- BACE
- BBBP
- ClinTox
- HIV

BACE is retained as a negative-control dataset even though the audit found no multi-fragment rows.

Regression datasets are not rerun under this sensitivity because ESOL and FreeSolv contained no multi-fragment records in the audited clean_v2 datasets.

## Split protocol

After the dominant-fragment sensitivity datasets pass their cleaning gate, splits will be regenerated from scratch. Primary frozen manifests are never reused because molecule identities and scaffold assignments may change.

The split settings are fixed to match the primary classification analysis:

- acyclic scaffold mode: `single_group`;
- partition seeds: the same 20 pre-specified v3 partition seeds;
- candidate budget: 300 target-blind candidate scaffold partitions per partition seed;
- protocols: legacy scaffold reference, random observation reference, size-matched scaffold, and paired target-balanced scaffold;
- size-matched and target-balanced scaffold partitions must have exactly equal test-set size within each seed;
- partition hashes and manifests are frozen before model fitting.

## Model protocol

The sensitivity uses the same frozen model definitions as the primary classification analysis:

- Morgan radius 2, 2048 bits;
- LR, RF, XGB;
- the same fixed production model seeds and class-imbalance handling;
- ROC-AUC as the primary classification metric, with the same supporting metrics.

The statistical unit remains the unique partition. Model randomness is not treated as replication.

## Inference and reporting

The dominant-fragment sensitivity is analyzed separately from the primary 18-cell family. It does not alter the primary Holm family or primary conclusions.

For each affected dataset/model cell, report the paired size-matched versus target-balanced effect across the 20 unique sensitivity partitions, its paired bootstrap interval, Wilcoxon result, and sensitivity-specific multiplicity correction.

The main robustness question is whether the qualitative primary classification conclusion changes under the operational single-fragment representation. Because the sensitivity is post hoc and representation-dependent, it will be described explicitly as such in the Supplementary Information and Discussion.