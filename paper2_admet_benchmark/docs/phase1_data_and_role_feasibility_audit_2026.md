# Paper 2 RACER-C Phase 1 data and role-feasibility audit

Status: **completed checkpoint; protocol remains pre-freeze**
Audit date: 2026-08-02
Branch: `paper2-reliability-extension-2026`

## Decision

Do not create the formal protocol tag yet. Three Veith CYP inhibition endpoints
now have file-level provenance, explicit CC BY 4.0 evidence, reproducible raw and
clean hashes, complete rejection accounting, deterministic chemical groups, and
adequate grouped role counts. They are **count-eligible**, not yet scientifically
eligible: the exact transformation from the original qHTS activity calls to TDC's
binary `Y` label is not documented by the TDC endpoint page or distributed file.

The remaining panel is also not ready for an ADMET-wide claim. Twelve candidate
endpoints have unresolved original dataset terms, and the three count-eligible
endpoints all derive from the same Veith qHTS campaign. Endpoint-equal averaging
would not create mechanistic or source diversity.

## Provenance and license audit

TDC's official repository states that its code is MIT licensed but directs users
to each dataset's own license. The official TDC ADME page explicitly lists CC BY
4.0 for the Veith CYP datasets. In contrast, the pages for HIA, P-gp,
bioavailability, hERG, AMES, DILI, Caco-2, PPBR, LD50, and hepatocyte clearance
state `Dataset License: Not Specified`. Public availability and a paper citation
were not treated as permission.

Authoritative records:

- TDC code and license boundary: https://github.com/mims-harvard/TDC
- TDC ADME dataset page: https://tdcommons.ai/single_pred_tasks/adme/
- TDC toxicity dataset page: https://tdcommons.ai/single_pred_tasks/tox/
- TDC data-server PID: https://doi.org/10.7910/DVN/21LKWG
- TDC source commit audited: `c310c35f27e3f506411018ac43d97b8ba23ca652`
- Veith et al. source: https://doi.org/10.1038/nbt.1581

The machine-readable audit contains 17 rows. Before cleaning, three extension
classification endpoints were license/hash ready, twelve had original-terms
blockers, and ESOL/Lipophilicity were retained only as legacy secondary endpoints.

## File-level acquisition results

| Endpoint | Dataverse file ID | Raw rows | Raw class 0 | Raw class 1 | Raw SHA256 |
|---|---:|---:|---:|---:|---|
| CYP2C9_Veith | 4259577 | 12,092 | 8,047 | 4,045 | `272bfd9de9899dd978993e23ccfb2a56680912ef6c8d89c0a9eac7f36c1c55c9` |
| CYP2D6_Veith | 4259580 | 13,130 | 10,616 | 2,514 | `1b50a2f02a6d9c02d29fd9e3684d661f64bfe8afb1fea09e5ffb3d54174c411b` |
| CYP3A4_Veith | 4259582 | 12,328 | 7,218 | 5,110 | `4c85a4cea9ffadd36fdd188f96b866d1d8db5d333a6d423172cf4363a585565a` |

Raw files remain untracked. Acquisition JSON manifests record source commit, file
ID, byte count, hash, license evidence, and retrieval time.

## Classification cleaning contract

The development candidate is RDKit 2026.03.4 with cleanup, organic fragment
parent selection, canonical uncharging, retained isotopes/stereochemistry, no
tautomer enumeration, canonical isomeric SMILES, and structure IDs derived from
the frozen standardized representation. Classification conflicts exclude the
entire standardized-structure group; same-label duplicates retain one
deterministic source record and log every other record.

Achiral Bemis--Murcko scaffolds are used. Ringless molecules receive a unique
`ACYCLIC:<structure_id>` group instead of collapsing every acyclic molecule into
one giant empty-scaffold group.

During development, RDKit 2026.03.4 raised a `bad bond stereo` precondition error
when directly canonicalizing the achiral scaffold for TDC Drug_ID `44601848.0`.
The correction explicitly removes stereochemistry from the scaffold molecule and
re-sanitizes it before canonical SMILES generation. The shared edge structure now
passes in all three endpoints; no affected row was silently deleted.

| Endpoint | Clean unique structures | Class 0 | Class 1 | Duplicate rows logged | Conflict groups excluded | Scaffold groups |
|---|---:|---:|---:|---:|---:|---:|
| CYP2C9_Veith | 12,043 | 8,007 | 4,036 | 31 | 7 | 7,365 |
| CYP2D6_Veith | 13,083 | 10,583 | 2,500 | 34 | 6 | 7,998 |
| CYP3A4_Veith | 12,295 | 7,189 | 5,106 | 31 | 1 | 7,656 |

Every source row reconciles to one retained clean structure or one rejection-log
entry. Clean data and rejection logs remain local; their hashes are committed in
the cleaning manifests.

## Label-blind chemical groups

Track C uses one deterministic endpoint-level leader partition built from Morgan
radius-2, 2048-bit, non-chiral fingerprints; Tanimoto similarity at least 0.60
joins the earliest eligible leader in a SHA256-seeded structure order. Labels do
not enter clustering or scaffold/cluster role assignment.

| Endpoint | Similarity clusters | Singleton clusters | Largest cluster |
|---|---:|---:|---:|
| CYP2C9_Veith | 8,047 | 6,463 | 59 |
| CYP2D6_Veith | 8,892 | 7,131 | 43 |
| CYP3A4_Veith | 8,496 | 6,862 | 62 |

An earlier prototype used class counts when assigning scaffold/cluster groups.
That violated the label-blind shift contract and was discarded before any model
prediction. The corrected allocator uses labels only for random-grouped
stratification and post-allocation eligibility counts. A label-permutation test
now proves scaffold assignment is unchanged.

## Role and conformal-resolution audit

The audit covers three endpoints, three tracks, three candidate role allocations,
and five main seeds: 135 endpoint/track/allocation/seed cells. All 135 pass the
declared count gates. All 540 full/selected class-conditional conformal cells have
a finite 90% order statistic and at least 35 observations at the planned 50%
minimum retention. The minimum planned selected-conformal class count is 152.

For the preferred `50/10/20/20` allocation, worst counts over every track and seed
are:

| Endpoint | Policy class 1 | Conformal class 1 | Test class 1 | Selected-conformal class 1 floor |
|---|---:|---:|---:|---:|
| CYP2C9_Veith | 218 | 659 | 660 | 329 |
| CYP2D6_Veith | 170 | 469 | 484 | 234 |
| CYP3A4_Veith | 327 | 1,021 | 995 | 510 |

The three allocations remain candidates until endpoint breadth and label semantics
are resolved. These count results do not authorize model fitting or gate tuning.

## Remaining P0 blockers

1. Reconstruct and document the exact TDC binary-label transformation from Veith
   qHTS activity calls, or exclude the three endpoints under the existing
   semantic-recovery hard rule.
2. Resolve original dataset terms for non-CYP candidates or replace them, before
   results, with explicitly licensed and scientifically diverse endpoints.
3. Satisfy the study-level breadth gate; otherwise narrow every claim to the
   Veith CYP inhibition panel.
4. Complete the policy-grid and paired-effect precision audit and freeze numerical
   non-inferiority/safety margins.
5. Lock the full Chemprop, MoLFormer, solver, CUDA, and package environment.
6. Implement lineage/leakage tests and run a measured single-endpoint GPU smoke
   benchmark.
7. Obtain user approval before the formal protocol tag or confirmatory GPU run.

## Recommended next action

Preserve the broader ADMET objective rather than silently publishing a CYP-only
study. First audit explicitly licensed ToxCast classification assays and any other
license-clear non-CYP candidates using only provenance, scientific semantics,
counts, and group feasibility. In parallel, trace the Veith/TDC label construction.
If breadth cannot be established without ambiguous terms or post-outcome endpoint
shopping, narrow the study prospectively before freeze.
