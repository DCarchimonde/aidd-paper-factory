# Endpoint-family extension: AMES, standard hERG, DILI, CYP2D6 inhibition

This extension addresses the reviewer's concern that a small improvement on an
already inspected Tox21/EPA protocol does not establish portability. The four
additional biological endpoint families and both outer split regimes are fixed
before their predictive comparison is run. They are **new endpoint-family
external benchmarks under a locked analysis**, not a newly blinded prospective
study. Their labels are public and are read during annotation cleaning and
source-role stratification. Random and scaffold holdouts are constructed within
each endpoint dataset; these are not separately collected external cohorts.

All four tasks remain in the report. Do not replace a task, retry an outer
seed, alter a class label, loosen a transport gate, or select a reporting subset
because a result is unfavorable. This extension tests the frozen procedure; it
does not establish a universal class-conditional coverage theorem or validate a
clinical decision rule.

The initial three-task data audit showed that standard hERG and DILI necessarily
have fewer than 100 calibration observations under the unchanged source-role
split. CYP2D6_Veith was therefore specified **before its predictive comparison**
solely for its documented large sample size, to provide a second non-Tox21
endpoint family with adequate calibration support alongside AMES. hERG and DILI
remain as boundary tests. This addition is not based on predictive results and
does not use results from any other project.

## Primary provenance and endpoint meaning

The official [TDC metadata](https://tdc.readthedocs.io/en/main/_modules/tdc/metadata.html)
maps `ames`, `herg`, `dili`, and `cyp2d6_veith` to Harvard Dataverse file IDs
4259564, 4259588, 4259585, and 4259580. The official [TDC downloader source](https://tdc.readthedocs.io/en/main/_modules/tdc/utils/load.html)
constructs `https://dataverse.harvard.edu/api/access/datafile/{id}` and reads the
returned tab-separated tables. We downloaded those exact files on 2026-11-05,
checked their schema `Drug_ID`, `Drug`, `Y`, and froze their byte digests in
`independent_sources.json`.

The official [TDC toxicity task documentation](https://tdcommons.ai/single_pred_tasks/tox/)
and [TDC ADME task documentation](https://tdcommons.ai/single_pred_tasks/adme/)
identify the endpoint meanings. No label is inverted to make TAME's historical
protected label appear chemically favorable.

| Endpoint | Biological family | Meaning of supplied `Y=1` | Primary source | Frozen raw rows | TDC page count |
|---|---|---|---|---:|---:|
| AMES | Bacterial reverse mutagenicity | Mutagenic | [Xu et al., 2012](https://doi.org/10.1021/ci300400a) | 7,278 | 7,255 |
| hERG | Cardiac ion-channel blockade | hERG blocker | [Wang et al., 2016](https://doi.org/10.1021/acs.molpharmaceut.6b00471) | 655 | 648 |
| DILI | Drug-induced liver injury | Liver-injury positive in the source benchmark | [Xu et al., 2015](https://doi.org/10.1021/acs.jcim.5b00238) | 475 | 475 |
| CYP2D6_Veith | Cytochrome P450 metabolic inhibition | CYP2D6 inhibition-positive in the source benchmark | [Veith et al., 2009](https://doi.org/10.1038/nbt.1581) | 13,130 | 13,130 |

The hERG source is the **standard small TDC `hERG` dataset**, not `hERG_Karim`
(13,445 compounds) or `hERG_Central`. Do not import Karim's threshold definition
into this dataset. The official web page's compound count and the downloaded
table's row count are distinct provenance facts; neither should be silently
substituted for the chemically deduplicated sample size. DILI's binary benchmark
annotation is not a calibrated clinical risk or proof of patient safety.
The [original CYP study](https://pmc.ncbi.nlm.nih.gov/articles/PMC2783980/)
notes that the bioluminescent assay can register inhibition from inhibitors or
competing substrates. Its positive class is an assay endpoint, not a direct
clinical toxicity label or a uniquely established biochemical mechanism.

The toxicity task pages state “Dataset License: Not Specified” and also display a CC BY
4.0 link. We do not infer unrestricted redistribution rights from that ambiguous
combination. Raw and cleaned molecule tables stay in the local execution cache;
the repository contains the source manifest, downloader, cleaning logic and
aggregate audit counts only. The CYP2D6_Veith ADME page explicitly lists CC BY
4.0; we use the same local-download policy for its molecule tables.

## Frozen chemical identity and exclusions

`independent_data.py` imports the existing
`scripts/racer_c/prepare_classification_endpoint.py::standardize_smiles` and
requires RDKit **2026.03.4**. It retains the existing cleanup, organic parent,
uncharging, isotope/stereochemistry, no-tautomer-enumeration, canonical isomeric
SMILES, and achiral Murcko scaffold policies. Acyclic molecules retain the
existing individual `ACYCLIC:<structure_id>` groups; this convention should be
disclosed when interpreting scaffold generalization.

The structure ID is the existing SHA256 of standardization ID plus standardized
SMILES. It is independent of endpoint, record name, row position, outer regime,
and training seed. `sample_id` prefixes that structure ID with the endpoint for
unambiguous joins; use **structure_id**, not sample_id, for molecular clustering
across endpoints or seeds.

Missing/nonbinary annotations and invalid structures are logged and excluded.
After standardization, repeated structures with the same label retain the
lexicographically first source record, as in the existing contract. A group with
both labels is excluded in full, regardless of majority. Labels are never
changed. Each removed raw row has an exclusion reason and, for a same-label
duplicate, the retained source row. Cleaning precedes outer splitting, so no
identical cleaned structure can cross source and target roles within a case.

Actual preparation under the required RDKit version produced:

| Endpoint | Raw rows | Same-label duplicates removed | Conflict groups / rows removed | Other exclusions | Clean structures (`0` / `1`) |
|---|---:|---:|---:|---:|---:|
| AMES | 7,278 | 23 | 0 / 0 | 0 | 7,255 (3,304 / 3,951) |
| hERG | 655 | 30 | 9 / 18 | 0 | 607 (193 / 414) |
| DILI | 475 | 1 | 0 / 0 | 0 | 474 (238 / 236) |
| CYP2D6_Veith | 13,130 | 34 | 6 / 13 | 0 | 13,083 (10,583 / 2,500) |

Removing conflicting annotations defines a consistent-label benchmark
population, rather than proving those compounds are unimportant. Report the
exclusions with the results. Different endpoint families can contain the same
molecule; family extension is not a claim of globally unseen chemical space.

## Fixed source and target roles

Both regimes target 80% source / 20% target. Random selection ranks structure IDs
by `SHA256("43021|<structure_id>")` and takes the first `ceil(0.2*n)` structures
as target. It does not stratify the target by class.

Scaffold selection orders entire scaffold groups by
`SHA256("43022|<scaffold_id>")`. Starting with an empty target, it adds a group
only when this strictly reduces the absolute difference from `ceil(0.2*n)`.
This is a deterministic, label-blind approximate 80/20 split: no scaffold group
is split to force a percentage. The frozen four datasets happen to attain the
requested integer target size. An infeasible empty role raises a recorded
failure; the implementation does not search seeds until it gets a usable
result. This is our specified structural holdout algorithm, not a claim to
reproduce TDC's default scaffold split.

Only structure IDs and scaffold IDs reach `outer_split`. Class counts below are
computed after membership is fixed, for description and uncertainty reporting.

| Endpoint | Regime | Source | Target | Target `0` / `1` | Source–target scaffold overlap |
|---|---|---:|---:|---:|---:|
| AMES | Random | 5,804 | 1,451 | 662 / 789 | Allowed; random reference |
| AMES | Scaffold | 5,804 | 1,451 | 618 / 833 | 0 |
| hERG | Random | 485 | 122 | 42 / 80 | Allowed; random reference |
| hERG | Scaffold | 485 | 122 | 38 / 84 | 0 |
| DILI | Random | 379 | 95 | 47 / 48 | Allowed; random reference |
| DILI | Scaffold | 379 | 95 | 36 / 59 | 0 |
| CYP2D6_Veith | Random | 10,466 | 2,617 | 2,139 / 478 | Allowed; random reference |
| CYP2D6_Veith | Scaffold | 10,466 | 2,617 | 2,129 / 488 | 0 |

All eight cases have zero source–target structure overlap. Within each case,
training seeds **311, 312, 313, 314, 315** call the old
`split_internal_roles(source_y, seed)`: stratified 60% fit, 20% router and 20%
conformal calibration, subject to integer rounding. All 40 actual role splits
were checked and contain both labels in each role. **The outer target is fixed
across those five seeds.** They measure training/role sensitivity, not five
independent external cohorts. Random and scaffold results also share molecules
and should not be counted as independent replications.

Transfer the Tox21-selected rule, protected-label convention, model roster,
nonconformity scores, target-aware comparator settings and six transport gates
without tuning on these benchmark outcomes. In particular, hERG has only 97
calibration observations, and DILI has 76 calibration observations and 95 target
structures. Both datasets necessarily fail the historical requirement of at
least 100 observations in both calibration and target sets. AMES has 1,161 and
CYP2D6_Veith has 2,094 calibration observations, so their sample counts permit
the frozen gates to be evaluated; this is not a prediction that the remaining
gates will pass. The small-dataset fallback is an intended portability boundary
and must stay visible. A tiny
calibration class can cause other gates to fail as well; do not bypass those
gates or discard a task. Report per-gate failures, fallback frequency, coverage,
correct singleton yield, class-specific errors, review burden, and matched-budget
comparisons even when the proposed method has no benefit.

## Runner contract and replay

`case_specs()` returns the complete 40-job base matrix, with keys `phase`,
`endpoint`, `split`, `seed`. Phases are `independent_random` and
`independent_scaffold`; endpoint names are `AMES`, `hERG`, `DILI`, `CYP2D6_Veith`.

```python
case = load_case(cache, endpoint="hERG", split="scaffold", seed=311)
# case["source_rows"], case["source_y"] -> existing source training routine
# case["target_rows"] -> structures only, no target-label column
# case["labels"][sample_id][endpoint] -> evaluation labels, separate from features
# case["eligible"] -> all cleaned target structures; exclusions are already audited
# case["input_hashes"], case["split_metadata"], case["preparation"] -> audit trail
```

The files under `<cache>/independent/<endpoint>/` are `raw.tab`, `cleaned.csv`,
`exclusions.csv`, `split_random.json`, `split_scaffold.json`, and
`preparation.json`. The preparation manifest records source-byte, output-file,
standardizer, standardization-contract, preparation-code, and source-manifest
digests. Existing data or code mismatches fail rather than silently rebuilding
or overwriting a completed cache. Use a fresh cache after an intentional
protocol/code change; do not relabel that change as the old locked analysis.

From the repository root, the isolated data preparation check is:

```bash
python -m paper2_admet_benchmark.revision_2026.independent_data --cache .local/independent_check
INDEPENDENT_DATA_CACHE=.local/independent_check python -m unittest paper2_admet_benchmark.revision_2026.tests.test_independent_data -v
```

The integrated operator entry point handles this acquisition as part of the
full revision workflow. The two commands above are a data diagnostic, not a
request for the second author to design splits or analyze results manually.

Validation completed on the frozen bytes and RDKit 2026.03.4: all eight tests
passed, including real-data loading of all 40 cases, both-class internal roles,
fixed target membership across seeds, disjoint scaffold holdouts, preservation
of label orientation, full removal of conflicting structures, and rejection of
tampered cached bytes. This is **data and implementation validation only**;
these checks do not supply the new predictive performance results.
