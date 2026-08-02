# Paper 2 RACER-C Phase 2 endpoint, policy, and lineage audit

Status: **completed pre-freeze checkpoint; no extension scientific outcome inspected**

Audit date: 2026-08-03

Branch: `paper2-reliability-extension-2026`

## Decision

Do not create the formal protocol tag or launch the confirmatory GPU run yet.
Phase 2 replaces the previous single-family CYP-only candidate set with a
license-clear, source-versioned Tox21 Challenge panel and completes the count-only
policy precision and abstract lineage contracts. Three endpoints are now primary
*candidates* for a toxicology reliability study. This does not establish an
ADMET-wide panel, empirical policy feasibility, or a safe/leak-free trainer.

## Authoritative Tox21 source and redistribution boundary

The official NCATS Tox21 Challenge complete training archive defines `1` as
active, `0` as inactive, and missing assay properties as unavailable. The exact
downloaded ZIP and its SDF member are locked as follows:

- challenge page: https://tripod.nih.gov/tox21/challenge/data.jsp
- challenge description: https://tripod.nih.gov/tox21/challenge/about.jsp
- ZIP SHA256: `024a3ae2690bcd4a593e6e0b10b455470b9bcb1d8f299dd36f220a250181517b`
- SDF member SHA256: `d66e1f9ec945ee528b1bea6e49af9c10d0bad546c2b304eb96004c8228824206`
- archive records: 11,764; unique DSSTox CIDs: 8,043.

The official challenge permits analysis participation but does not state a raw
redistribution grant on the download page. The repository therefore commits only
hashes, counts, manifests, and derived aggregate audits; raw and cleaned molecule
rows remain ignored. EPA's current invitroDB release is separately CC0, but it was
not silently substituted for the locked 2014 Challenge source.

## Deterministic cleaning and endpoint statuses

Every assay property was parsed from the same locked SDF. Structures were
standardized once under `racer_c_rdkit_2026_03_4_v1`; missing assay labels,
same-label duplicates, structure failures, and post-standardization conflicts are
fully accounted. For every endpoint, clean rows plus endpoint-specific rejection
rows reconcile to all 11,764 source records.

| Endpoint | Clean | Class 1 | Duplicate rows | Conflict groups | Count-only status |
|---|---:|---:|---:|---:|---|
| NR-AR | 6,972 | 250 | 2,235 | 58 | calibration-limited |
| NR-AhR | 6,289 | 700 | 1,732 | 61 | primary candidate (15/15) |
| NR-AR-LBD | 6,532 | 212 | 2,007 | 22 | calibration-limited |
| NR-ER | 5,855 | 625 | 1,400 | 164 | primary candidate (15/15) |
| NR-ER-LBD | 6,687 | 281 | 1,890 | 63 | calibration-limited |
| NR-Aromatase | 5,596 | 254 | 1,537 | 32 | calibration-limited |
| NR-PPAR-gamma | 6,243 | 162 | 1,905 | 15 | calibration-limited |
| SR-ARE | 5,598 | 845 | 1,365 | 72 | primary candidate (15/15) |
| SR-ATAD5 | 6,825 | 233 | 2,188 | 28 | calibration-limited |
| SR-HSE | 6,224 | 307 | 1,776 | 54 | calibration-limited |
| SR-MMP | 5,588 | 851 | 1,607 | 50 | primary candidate (15/15) |
| SR-p53 | 6,549 | 390 | 2,019 | 27 | track-limited secondary (0/15) |

The seven endpoints below 350 observations in one class were stopped before
similarity clustering. This is a prospective compute-saving gate, not
result-guided endpoint selection. NR-AhR, NR-ER, SR-ARE, SR-MMP, and SR-p53 were
clustered label-blindly and audited over all requested tracks.

## Role and policy precision decision

The final count audit contains 300 endpoint/track/allocation/seed cells and 1,200
full/selected conformal-resolution cells. Role construction is label-blind for
scaffold and similarity-cluster tracks; labels are used only for post-allocation
eligibility counts.

The first Phase-2 checkpoint still sorted covariate groups by the largest
within-group class count before applying its size-only role objective. A global
class-label flip left that key unchanged, so the original permutation test did
not detect the residual access. The corrected implementation orders scaffold and
similarity clusters only by total group size plus seeded group-ID ties. An
arbitrary, non-complement label permutation now leaves assignments unchanged.
All 300 role cells and downstream precision tables were regenerated. NR-AhR
changed from 14/15 to 15/15 and therefore became the fourth count-only primary
candidate; no model output was generated or inspected.

The Phase 0 preference `50/10/20/20` left a smallest prospective primary policy
critical-class cell of 41. With 108 Bonferroni tests, even zero observed errors
cannot always yield a one-sided upper bound at or below 10%. Before any model
output, Phase 2 therefore selected `50/20/15/15`, whose primary policy
critical-class minimum is 104. The change is recorded as a pre-freeze deviation.

The frozen selector evaluates exactly 36 class-specific gate pairs. Each pair
must simultaneously satisfy exact one-sided bounds for both true-class retention
and the selected critical true-class *base* error. It uses alpha `0.05/108`,
requires retention lower bounds at least 0.50, critical selected base-error upper
bound at most 0.10, and minimum class counts of 25. Selection is deterministic and
lexicographic. If no pair passes, the only permitted output is
`policy-infeasible`; thresholds or confidence corrections may not be relaxed.

## Leakage contract

`lineage_contract.py` resolves the transitive training rows behind every fit and
prediction node. It rejects unknown rows, dependency cycles, self/OOF reuse, and
any model fit that reaches policy, conformal, or test rows. Tests cover honest
two-level OOF lineage and each failure class. This closes the abstract contract,
not the implementation risk: the future Chemprop/MoLFormer trainer must emit and
pass the same lineage records in its production implementation.

A seed-99 CPU integration smoke was run on the NR-ER strict-scaffold development
role using ECFP/logistic base models, two-fold inner OOF Platt inputs, and
three-fold outer OOF predictions. All 2,928 development rows received finite OOF
probabilities and all 2,928 transitive lineages excluded their target row and
every non-development role. Probabilities were discarded after the contract
check: no performance metric was computed and no policy, conformal, or test
prediction was generated. This is not the planned Chemprop/MoLFormer trainer and
not a GPU benchmark.

## Veith/TDC label decision

TDC identifies the three CYP endpoints as binary inhibition datasets and publishes
the downloaded `Y` values. Independent published summaries reproduce the exact
raw active counts (4,045, 2,514, 5,110), supporting the polarity `Y=1` = active /
inhibitor. However, neither the TDC page nor current loader code provides a fully
reconstructable mapping from the original qHTS curve classes and assay flags to
that binary value. The CYP endpoints remain outside the primary freeze panel until
that transformation is documented or the protocol explicitly accepts the TDC
binary labels as source-defined observations with a narrowed claim.

Relevant records:

- TDC ADME page: https://tdcommons.ai/single_pred_tasks/adme/
- Veith et al.: https://pmc.ncbi.nlm.nih.gov/articles/PMC2783980/
- PubChem CYP2C9 assay AID 883: https://pubchem.ncbi.nlm.nih.gov/bioassay/883

## Quality gates

- 28 protocol, provenance, role, policy, lineage, smoke-output, and integrity tests: PASS.
- Python compilation and `git diff --check`: PASS.
- Tox21 acquisition/member hashes and 12 endpoint reconciliation identities: PASS.
- 3/12 primary, 2/12 secondary, 7/12 calibration-limited: deterministic count-only decision.
- Formal protocol freeze: BLOCKED.
- Development-only seed-99 CPU lineage smoke: PASS (2,928/2,928 OOF lineages).
- GPU smoke: NOT RUN; this container has no CUDA device or PyTorch environment.

## Remaining blockers

1. Narrow the confirmatory claim to the Tox21 assay family or prospectively add an
   independently sourced, license-clear mechanism family without using results.
2. Lock exact Chemprop, MoLFormer, PyTorch, CUDA, solver, and container revisions.
3. Instrument the production Chemprop/MoLFormer trainer with the transitive
   lineage records; keep seed 99 as the only technical seed before freeze.
4. Measure one primary endpoint on the target RTX 4090-class GPU and replace the
   current 150--400 GPU-hour estimate.
5. Obtain user approval before the formal protocol tag and confirmatory run.
