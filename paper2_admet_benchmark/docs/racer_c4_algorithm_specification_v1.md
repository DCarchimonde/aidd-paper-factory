# RACER-C4/TAME algorithm specification

Status: **candidate frozen before the independent EPA labels are opened**

Version: `4.0.0-rc1`

## Purpose

RACER-C4/TAME is a safety-first conformal layer for binary molecular assays
under observable covariate shift. TAME means **transport-audited multi-view
envelope**. It does not claim that estimated density ratios restore exact
finite-sample validity under arbitrary chemical shift.

The earlier C3 candidate tried to improve correct-singleton yield with a local
candidate expert. On the independent public development batch, the strongest
version gained about 0.92 percentage points of mean MacroCSY but lost about
3.04 percentage points of class-0 coverage. Coverage-matched variants lost
MacroCSY. That family is rejected from the C4 primary method and retained only
as negative development evidence.

## Fixed base score

For each endpoint and seed, the official 10K training set is split 60/20/20
into fit, router, and conformal roles by a stratified seed-fixed allocation.
Four ECFP4 classifiers are fitted on the fit role:

1. logistic regression;
2. random forest;
3. extra trees; and
4. Bernoulli naive Bayes.

An L2 logistic router is fitted on the four component logits in the router
role. For routed probability `p`, candidate-label LAC scores are

`s_0(x)=p(x)` and `s_1(x)=1-p(x)`.

Ordinary class-conditional Mondrian thresholds at `alpha_0=alpha_1=0.10` are
the baseline. Finite-sample ranks are
`ceil((n_y+1)(1-alpha_y))`; an unavailable rank returns positive infinity.

## Label-free transport views

The internal conformal covariates are the source domain and the external batch
covariates are the target domain. No external label enters either view.

- Physicochemical view: molecular weight, logP, TPSA, donor/acceptor counts,
  rotatable bonds, rings, fraction sp3, and heavy-atom count.
- Score view: four component logits plus the routed logit.

A 2,048-bit ECFP domain classifier is retained as a required diagnostic
comparator, but is not an envelope view: on the public development panel it
failed its ESS/clipping certificate in all 30 primary cells. Replacing the
certificate with looser thresholds is prohibited.

Each view uses five-fold cross-fitted L2 logistic domain classification for the
domain-AUC audit, followed by the same regularized estimator refitted on all
unlabeled source/target covariates for the actual ratio. Posterior odds are
prior-corrected by `n_source/n_target` and clipped to `[0.05,20]`. The resulting
source and query weights define candidate-specific weighted Mondrian thresholds
with the query weight placed at infinity.

## Transport audit

A view is active only when all frozen checks pass:

- at least 100 source and 100 target structures;
- at least 20 conformal examples per true class;
- total and per-class effective sample size at least 25% of the corresponding
  raw count;
- no more than 30% of estimated weights at either clipping boundary;
- cross-fitted domain AUC no greater than 0.99; and
- the weighted standardized mean gap no more than 1.05 times the unweighted
  gap.

If fewer than both views pass, TAME disables every transport augmentation and
returns the ordinary Mondrian set, except that an ordinary empty set becomes
`{0,1}`. This is an exact no-transport fallback with a non-actionable empty-set
repair; it is an operational failure certificate, not evidence that the
transport assumption holds.

## Frozen envelope

The public development batch showed a large class-0 coverage deficit and no
class-1 mean coverage deficit. The final candidate therefore protects label 0
only. For each target structure:

1. start with the ordinary Mondrian set;
2. add label 0 only when both active weighted views include label 0;
3. preserve every baseline label; and
4. convert every baseline-empty set to `{0,1}`.

These operations give three deterministic invariants for any target labels:

1. the TAME set contains the ordinary baseline set;
2. TAME emits no empty set; and
3. every TAME singleton was already a baseline singleton.

Consequently, empirical class coverage cannot decrease and wrong-singleton
exposure cannot increase relative to the ordinary baseline on the same rows.
Correct-singleton yield also cannot increase; the primary scientific question
is whether the coverage/safety improvement generalizes while its loss remains
inside the frozen 5-percentage-point MacroCSY non-inferiority margin.

On the 30-cell public development panel, the frozen candidate increased mean
class-0 coverage by 5.4394 percentage points, left mean class-1 coverage
unchanged, and changed MacroCSY by -4.9169 percentage points. Both approved
views were active in 24/30 cells; the exact ordinary fallback handled the other
six. All eight promotion checks passed. These are architecture-development
results, not independent evidence.

## Independent validation firewall

- Public leaderboard batch (~296 structures): architecture development only;
  seeds 101--105.
- Independent final EPA batch (~647 structures): one final evaluation only;
  fresh seeds 211--215.
- Standardized structures overlapping the 10K training cohort are excluded
  from domain fitting and evaluation, never silently counted.
- The final-label file is not acquired or parsed until the development gate
  passes, all final predictions are written, and their SHA256 is bound into a
  promotion record.
- A negative or mixed final result is retained without threshold, endpoint,
  seed, representation, or margin changes.

## Non-claims

TAME does not invent Mondrian conformal prediction, density-ratio weighting,
domain-classifier ratio estimation, multi-distribution max-p methods, or
utility-aware conformal sets. It supplies no conditional-coverage theorem and
no exact guarantee with estimated weights. Its defensible contribution is the
joint, auditable molecular workflow: two label-free transport views, explicit
transport failure certificates, a baseline-containing labelwise consensus
envelope, and a cryptographically ordered independent-label firewall.
