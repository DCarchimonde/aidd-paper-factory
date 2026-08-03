# Paper 2 reliability extension: Phase 0 audit

Status: **completed, pre-freeze**  
Audit date: 2026-08-01  
Audited base: `main` at `a1d5f71790cfc7f668fee5455b2be8f88267c5ea`  
Working branch: `paper2-reliability-extension-2026`

## Executive decision

The existing four-endpoint Paper 2 is a coherent, frozen baseline study. Its
versioned result tables and figures must remain immutable. The proposed RACER-C
extension is scientifically worth developing, but the supplied design is not
ready for a formal protocol tag. It contains four P0 issues: honest multi-stage
cross-fitting is underspecified; rare-class sample sizes cannot support selected
class-conditional conformal claims; the gate-selection rule can overfit the policy
partition; and several named strong baselines target different estimands or data
access regimes.

The extension is therefore authorized only as a prospectively frozen extension
on this separate branch. No result-generating model may be run until the P0 gates
below pass and the user approves the formal freeze tag.

## Repository audit

| Item | Finding | Status |
|---|---|---|
| Default branch | `main` | verified |
| Base HEAD | `a1d5f717` (`Add no-APC CILS submission checklist for Paper 2`) | verified |
| Remote branches before this work | `main` only | verified |
| Existing Paper 2 code | scripts `00`--`37`, configs, manifests, frozen tables and figures | present |
| Existing manuscript | `paper2_latex/main.tex`, six section files, SI source, two bibliography files | present |
| Raw/processed data | intentionally ignored and absent from the clean checkout | external prerequisite |
| Frozen assets | eleven headline CSV tables and twelve PDF/PNG figure assets | present |
| Frozen hashes | all figure hashes match; all table hashes reproduce after restoring the CRLF bytes used when the Windows manifest was created | PASS with cross-platform verifier |
| Python syntax | all existing scripts compile; script 35 emits non-fatal invalid-escape warnings | PASS with P2 warning |
| Bibliography audit | false failure because it read only `references.bib` while `main.tex` loads two files | fixed on extension branch |
| LaTeX build in this container | stopped at pdfTeX font expansion for non-scalable fonts | environment-limited, not evidence of a source error |

The table manifest is byte-sensitive and was produced from CRLF CSV files. Git's
clean Linux checkout uses LF, so a naive raw-byte check fails even though all 11
historical hashes reproduce after deterministic LF-to-CRLF normalization. The new
contract test handles both representations without changing any frozen table or
manifest value.

The manuscript source matches the handover on the important scientific boundaries:
four endpoints, ECFP-centred models, random/scaffold/cluster confirmatory seeds,
paired comparisons, explicit empirical-only scaffold-shift language, and no
requirement for a Zenodo DOI before submission.

## Latest-method and novelty audit

Primary or author-controlled sources checked on 2026-08-01:

- Zhao et al., *Journal of Cheminformatics* (2026), DOI
  `10.1186/s13321-026-01217-2`, already benchmarks modern ADMET models under
  scarcity, OOD, imbalance, activity cliffs, and beyond-Rule-of-5 space.
- Plassier et al., Rectified Conformal Prediction, ICML 2025, official code at
  `https://github.com/stat-ml/rcp` (MIT). RCP learns a transformation of a
  conformity score to improve approximate conditional coverage while preserving
  marginal validity under the paper's assumptions.
- Gazin et al., InfoSP/InfoSCOP, *JRSS B* (2025), DOI
  `10.1093/jrsssb/qkae120`. These are batch-selection methods controlling false
  coverage rate for informative selected prediction sets.
- Xu et al., Selective Conformal Risk Control, arXiv `2512.12844v2` (2026),
  with SCRC-T (transductive exact construction) and SCRC-I (inductive PAC-style
  construction); author code is linked from the paper.
- Bai and Jin, SCoRE, arXiv `2603.24704v1` (2026). SCoRE uses e-values for
  finite-sample control of a user-defined bounded risk among selected cases and
  includes a drug-discovery application. It is a recent preprint, not a routine
  drop-in Mondrian baseline.
- Chemprop official repository and documentation: v2.3.0 was the current release
  observed at audit time; code is MIT licensed. The exact package version and
  source commit must be locked at freeze.
- IBM MoLFormer official model card: frozen embeddings are an intended use. The
  model identifier, immutable revision, tokenizer revision, pooling, canonical
  SMILES input, and Transformers compatibility revision must be frozen.
- TDC official ADMET group contains 22 endpoints and uses scaffold benchmark
  splits. TDC code is MIT, but each dataset's source terms must be audited
  separately; a generic TDC page statement is not a substitute for original
  dataset licensing.

### Novelty conclusion

RACER-C cannot safely be sold as the first conformal deferral, rectification, or
selected-risk method. In abstract form it is close to heterogeneous stacking plus
a learned reliability score, a class-aware gate, and selected Mondrian
calibration. Its defensible contribution is narrower:

1. a prospectively frozen ADMET reliability triage construction combining
   representation--model disagreement, chemical support, and local honest OOF loss;
2. an endpoint-qualification and data-access audit;
3. critical-class retention and correct-singleton-yield diagnostics evaluated
   alongside validity and efficiency; and
4. an empirical comparison under label-blind chemical shift.

If it does not beat stacking-only or estimand-matched recent methods, the final
paper must be framed as a reliability audit and triage framework, not an
algorithmic breakthrough.

## P0 findings: resolve before protocol freeze

### P0-1: honest multi-stage cross-fitting

The original text requires OOF base probabilities, calibrated probabilities,
stacking margins, Brier losses, local losses, and BRI values, but does not define
how second-stage learners avoid seeing a held-out molecule indirectly. Reusing a
single OOF matrix to fit a calibrator or stacker and then calling its fitted
outputs OOF is not honest cross-fitting.

Resolution adopted in the draft specification: three outer meta-folds. For each
held-out meta-fold, inner OOF base predictions are generated using only the other
two folds; calibrators, stacker, and the constrained BRI are fitted without the
held-out fold; all reference-neighbour losses for that fold also exclude it.
Deployment uses the resulting cross-fit ensemble. Tests must inject unique labels
and confirm that no artefact for a row was trained from that row or its grouped
duplicate.

### P0-2: rare-class precision and conformal resolution

At nominal 90% coverage, a class-specific split-conformal threshold is already
infinite when the selected calibration cell has fewer than nine observations.
That is only an algebraic minimum, not adequate precision. With the proposed
four-way allocation, ClinTox has about 93 positives before grouping; a 20%
conformal role and 50% gate retention would yield roughly nine selected positives,
and scaffold grouping may yield fewer. This cannot support a strong selected-domain
coverage or wrong-singleton claim.

Resolution adopted: endpoint eligibility is determined before modelling. Primary
status requires each class to have at least 35 selected conformal observations
under the minimum planned retention in every confirmatory allocation audit, plus
adequate policy and test support. ClinTox remains a required failure-mode anchor
but is expected to be `calibration-limited`, not a primary superiority endpoint.

### P0-3: policy-grid overfitting

Selecting the highest-retention pair from 36 threshold combinations using small
policy cells can satisfy constraints by noise. Point estimates are insufficient.

Resolution adopted: a fixed grid may be searched only with simultaneous one-sided
confidence bounds and a deterministic lexicographic rule. The policy data cannot
choose reliability features, temperature family, margins, or endpoints. If no
combination is feasible, the status is `policy-infeasible`; constraints are not
relaxed.

### P0-4: non-equivalent strong baselines

RCP, InfoSP/InfoSCOP, SCRC, and SCoRE do not all estimate the same object or use
the same access pattern. InfoSP/InfoSCOP are batch FCR procedures; SCRC-T is
transductive; SCRC-I is PAC-style; SCoRE controls a bounded selected risk.

Resolution adopted: stacking plus Mondrian, score rectification without a gate,
gate without rectification, full RACER, and an estimand-matched RCP construction
form the core comparison. InfoSP/InfoSCOP, SCRC-I, and SCoRE are anchor analyses
only after a method-access and estimand mapping passes. SCRC-T/InfoSCOP are not
ranked against inductive methods in the primary table.

## P1 findings

- Define the ECFP block as the uniform mean of four individually Platt-calibrated
  probabilities; otherwise “three-block disagreement” is ambiguous.
- Clip probabilities to `[1e-6, 1-1e-6]` before logits and record clipping counts.
- Use unweighted base training as the primary probability regime. Class-weighted
  training is an anchor sensitivity after recalibration, not a second full grid.
- Replace an unspecified monotone GAM with a reproducible low-capacity constrained
  additive/linear Brier-loss model; BRI remains an index, not an error probability.
- Select one global attenuation cap from `{1.0, 1.5, 2.0}` using development-only
  nested evidence and a predeclared objective. Never select it per endpoint or
  from policy/conf/test results.
- Define all denominators for wrong singletons, accepted error, FNR/FPR, and CSY.
- Treat BBBP criticality as context-dependent; report both classes and worst class
  rather than labelling BBB penetration intrinsically hazardous.
- Freeze dataset-specific label semantics, thresholds, units, species, assay type,
  and original-source license before setting critical classes.
- Predeclare stochastic training seeds and retain all failures.

## P2 / sensitivity items

- Latent-space distances, isotonic calibration, bidirectional rectification,
  unconstrained BRI, and class-weighted base training.
- Similarity-cluster Track C for every endpoint may be reduced to eligible endpoints
  if deterministic complexity caps are exceeded; exclusions must precede results.
- Regression expansion remains secondary and must not delay the classification
  protocol.
- Remove Python invalid-escape warnings in script 35 during a later manuscript
  maintenance pass.

## DELETE from the primary design

- Claims of guaranteed coverage under scaffold/cluster shift.
- Any claim that BRI is a calibrated error probability.
- Any universal “critical class = label 1” rule.
- Per-endpoint or test-guided temperature selection.
- SMOTE or duplicate-label augmentation for conformal calibration.
- A single leaderboard mixing inductive, transductive, batch-FCR, PAC, and
  selected-risk methods.
- Full regression RACER-R as a co-primary innovation.

## Initial compute envelope

Before eligibility, let `E` be the number of primary classification endpoints.
The main design has `15E` endpoint/track/split cells (three tracks, five split
seeds) plus 40 extra random/scaffold cells for four anchors at seeds 106--110.
With three-fold honest nesting, each stochastic base family requires nine fits
per cell. If `E=6`, this is 130 cells and approximately:

- 1,170 D-MPNN fits for the main nested design;
- 4,680 ECFP-model fits across four families;
- 1,170 frozen-embedding logistic-head fits;
- approximately 120 extra D-MPNN fits for the three-training-seed anchor
  sensitivity if only deployment models are repeated.

The original planning range assumed an RTX 4090 and was 150--400 GPU-hours,
including a 20% rerun allowance, plus 300--900 CPU-hours and 50--150 GB of working
storage. That hardware assumption is now superseded by the user's Windows RTX-4060
Laptop target. These are historical engineering bounds, not a spending
authorization; the measured 4060 smoke benchmark must replace them before the full
run, and the user must approve the full GPU run.

## Phase 0 decision

Creating `paper2-reliability-extension-2026` is safe because the checkout was clean
apart from an audit-generated SI file, `main` is preserved, and no remote branch of
that name existed. Protocol drafting may proceed. Formal tagging, endpoint freeze,
and full GPU execution remain blocked pending the precision audit and user approval.
