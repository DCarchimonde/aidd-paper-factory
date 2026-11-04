# Literature and comparator implementation record

Checked 2026-11-04. Scope: CHEMOLAB-D-26-01118, not the later CYP/SAR project.

## What the review establishes

The distinction between marginal and class-conditional coverage is established prior work. The revised study must not describe its ClinTox illustration as a new statistical principle. The defensible research question is whether a specific intervention improves endpoint-relevant decisions at comparable information/deferral cost, across chemically shifted populations. This is a proposed empirical question; no new result is asserted here.

TAME's add-only construction cannot reduce any class's empirical coverage on a fixed evaluation sample. That set inclusion observation alone supplies neither a nominal class-conditional coverage guarantee under shift nor evidence that TAME selects useful additions. Compare it against simple additions at exactly the same budget, and report which molecules were rescued and how many additional ambiguous decisions were created.

## Sources and implementation boundaries

| Comparator | Primary source | Role and status in this revision |
|---|---|---|
| Marginal and class-conditional split CP | [Vovk, 2012, PMLR 25:475–490](https://proceedings.mlr.press/v25/vovk12.html) | Core reference methods; class thresholds must be indexed by each candidate label. |
| Likelihood-ratio weighted CP | [Tibshirani, Barber, Candès and Ramdas, NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html); [author PDF](https://www.stat.berkeley.edu/~ryantibs/papers/weightedcp.pdf) | Core estimated-ratio comparator; exact validity of the oracle construction must not be attributed to estimated/clipped ratios. |
| APS; class-conditional APS adaptation | [Romano, Sesia and Candès, NeurIPS 2020](https://proceedings.neurips.cc/paper_files/paper/2020/hash/244edd7e85dc81602b7615cd705545f5-Abstract.html) | Efficiency/adaptivity reference. APS is not, by itself, a distribution-shift solution. State the implemented score, tie convention and whether it is randomized. |
| RAPS | [Angelopoulos, Bates, Malik and Jordan, ICLR 2021](https://openreview.net/pdf?id=eNdiU_DbM9) | Literature context; no complete RAPS implementation claimed here. With only two labels, assess whether regularization offers a distinct useful comparator before expanding the matrix. |
| CoDrug | [Laghuvarapu, Lin and Sun, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/7691484a7a35d5e2742279c1d926b778-Abstract-Conference.html); [full paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/7691484a7a35d5e2742279c1d926b778-Paper-Conference.pdf) | **Full reproduction pending and required before resubmission.** A KDE density-ratio component is not full CoDrug. |
| KMM-CP, including selective KMM | [Laghuvarapu, Deb and Sun, UAI 2026, PMLR 337:3237–3261](https://proceedings.mlr.press/v337/laghuvarapu26a.html); [readable arXiv v1](https://arxiv.org/html/2603.26415v1) | **Full reproduction pending and required before resubmission.** A constrained KMM component is not the full published pipeline, especially not its selection mechanism. |
| Confidence rejection and AD rejection | [El-Yaniv and Wiener, JMLR 11:1605–1641, 2010](https://jmlr.org/papers/v11/el-yaniv10a.html); [Sahigara et al., 2013](https://pmc.ncbi.nlm.nih.gov/articles/PMC3679843/) | Practical non-CP controls. A simple nearest-neighbor similarity rule is a transparent baseline, not a reproduction of every AD algorithm in those references. |

The bibliography identifies method provenance, not evidence that all those methods have been run. Actual run completion must come from the output manifest and verification report.

## Exact finite-sample conventions

Use a score where a larger value means less conformity, e.g. `s(x,y)=1-p_y(x)`. For class `c`, let `n_c` be its calibration count. Ordinary class-conditional split CP uses rank `ceil((n_c+1)*(1-alpha))`; if that rank exceeds `n_c`, its threshold is infinity. If the class is absent from calibration, include that candidate label rather than inventing a quantile. For prediction, compare `s(x,c)` to `q_c` separately for each candidate `c`; applying the predicted class's threshold to both candidates is a different procedure.

For oracle covariate-shift weighting, the weighted distribution for test covariates `x` is

\[
 F_x=\frac{\sum_{i=1}^{n} w(X_i)\delta_{s_i}+w(x)\delta_{+\infty}}
 {\sum_{i=1}^{n}w(X_i)+w(x)}.
\]

Its `(1-alpha)` quantile defines the set. The test-point mass must not be silently dropped. Under covariate shift, `w(x)=p_target(x)/p_source(x)`, with unchanged `P(Y|X)` and source support covering target support. With class calibration, restrict the sum to that candidate class. The class-specific density ratio differs by a constant that cancels in the normalized construction. With estimated ratios, clipping, or finite-sample dependent ratio fitting, report empirical performance rather than an unconditional exact guarantee.

For a domain classifier with target-domain probability `eta(x)`, the ratio estimate includes the domain sampling-prior correction `(pi_source/pi_target)*eta/(1-eta)`. Report clipping, zero/near-zero denominator handling and held-out domain discrimination. A gate based on AUC or ESS cannot establish the unobserved `P(Y|X)` invariance assumption.

## CoDrug: what still has to be reproduced

The published pipeline combines an energy-regularized molecular classifier with KDE on its outputs/representations and a test-mass weighted conformal quantile. The paper has separate NoEnergy and density-estimator ablations. Its unlabeled auxiliary data and representation training are part of the method. Thus a KDE applied to the existing TAME features should be named `kde_ratio_component` or similarly explicit, not `CoDrug`.

The paper names [github.com/siddharthal/CoDrug](https://github.com/siddharthal/CoDrug) as its code location. On 2026-11-04 this environment received HTTP 404 from repository and contents requests. Search did not establish a replacement official location. This means the code was not obtained here; it does not establish that the authors deleted it or that another environment cannot access it.

The second author must obtain the official source or implement the published algorithm independently, record a pinned source revision/license/dependency environment, and document every departure from the paper. First reproduce a reported configuration as a sanity check, then run the agreed protocol. An independent reimplementation can be legitimate, but must be labelled as such and reviewed for equation-to-code fidelity before a manuscript claim of comparison is made.

## KMM-CP: what still has to be reproduced

The accessible v1 describes bounded kernel moment matching, followed by a normalized weighted empirical score quantile. Its selective variant jointly obtains target selection weights, selects a subset, and refits KMM there. Its guarantee is asymptotic under additional assumptions; its empirical quantile does not include a query infinity atom. Do not invent a query weight of one and call it an exact implementation. A local QP-only component should be named `kmm_empirical_component`.

The primary proceedings page confirms the UAI 2026 publication. The detailed formula check here used arXiv v1 because the proceedings PDF was unavailable through the current retrieval path; reconcile with the final version before claiming exact reproduction. In particular, v1 Appendix D's predicted-class threshold notation differs from candidate-label Mondrian calibration. The adapter must record the actual convention; do not silently call different conventions identical.

The paper names [github.com/siddharthal/KMM-CP](https://github.com/siddharthal/KMM-CP). Repository/contents requests returned 404 in this environment. The same source-retrieval, version-pinning and independent-reimplementation requirements apply as for CoDrug. A selected-subpopulation result must never be presented as whole-population coverage.

## Fair comparison and chemical meaning

These are design requirements for this revision, not reported findings:

1. Keep score, backbone, source roles, target molecules and data access identical for the component comparison. Separately report full-method comparisons when the published method trains its own backbone; do not credit differences in representation learning to the conformal wrapper.
2. Compare TAME additions with random additions, probability-based additions, similarity-based additions and unconditional protection at the same eligible-molecule/addition budget. Record exact realized counts, because ties and an insufficient eligible pool can change the nominal budget.
3. Show class coverage, worst-class coverage, mean set size, singleton yield, empty-set rate, doubleton rate and endpoint-relevant error/deferral metrics. Report all prespecified endpoints and seeds, including failures, with molecular grouping accounted for in uncertainty estimates. Repeated seeds do not create independent new compounds.
4. For selective KMM or rejection baselines, report selection rate, retained-population class coverage and the full-population operational result. In full-population decision accounting, unselected molecules are deferred, not removed from the denominator. Preserve empty sets separately: a selected empty prediction and an unselected molecule are different events.
5. Select the protected label from the endpoint's scientific meaning before evaluating the target. Development undercoverage is a statistical diagnostic, not evidence of chemical cost. [The original Tox21 challenge publication](https://www.frontiersin.org/journals/environmental-science/articles/10.3389/fenvs.2015.00085/full) describes nuclear-receptor and stress-response assays; a positive assay label is not equivalent to demonstrated clinical toxicity, and a negative prediction is not proof of safety.
6. If protection is of assay-positive label 1, report `P(predict {0} | Y=1)` and `P(Y=1 | predict {0})` separately, along with negative-singleton yield and deferral. Name these missed-activity/false-inactive quantities for Tox21, not clinical false-safe rates.
7. The already-opened EPA labels cannot become a newly sealed test by changing seeds or hashes. Existing EPA comparisons are retrospective. Any new independent-confirmation claim requires a genuinely unexamined source/cohort and frozen processing before labels are used.

## External prediction import

`external_adapter.py` accepts the runner's label-free, single-cell `target_manifest.csv`, a predictions CSV, and a provenance JSON. It checks exact sample membership at every expected alpha, rejects duplicate/missing/extra IDs and label columns, verifies protocol/manifest hashes, and preserves the reference model separately from the external model. These checks protect joins; declarations do not prove what the upstream code accessed.

Prediction columns are `sample_id,alpha,include_0,include_1`, optionally `selected,probability_1`. Include flags and selection must be literal `0` or `1`. Supply every target molecule, including unselected ones. The output preserves `external_original_include_0/1` and `external_selected`. For full-population metrics, an unselected molecule receives both labels to represent deferral; for retained-population metrics use the original flags restricted to `external_selected==1`. This does not give a guarantee for either population.

Required metadata fields:

```json
{
  "method": "codrug_paper_reimplementation",
  "evaluation_model": "external_codrug",
  "source_url": "https://proceedings.neurips.cc/paper_files/paper/2023/hash/7691484a7a35d5e2742279c1d926b778-Abstract-Conference.html",
  "source_revision": "REPLACE_WITH_EXACT_UPSTREAM_COMMIT_OR_PAPER_VERSION_AND_LOCAL_CODE_HASH",
  "adapter_version": "1",
  "implementation_kind": "paper_reimplementation",
  "protocol_sha256": "REPLACE_WITH_ACTUAL_64_CHARACTER_HASH",
  "target_manifest_sha256": "REPLACE_WITH_ACTUAL_64_CHARACTER_HASH",
  "target_labels_used": false,
  "target_features_used": true,
  "fitted_label_roles": ["fit", "router", "conformal"],
  "dependencies": {"python": "REPLACE_WITH_ACTUAL_VERSION"}
}
```

The example is a schema, not completed provenance. `official_code`, `paper_reimplementation` and `component_only` are distinct implementation kinds. A full method still must obey the prescribed role separation: supervised training on `fit`, tuning on `router`, and conformal thresholds on `conformal`; listing all roles does not authorize training a classifier on calibration labels.

```bash
python -m paper2_admet_benchmark.revision_2026.external_adapter \
  --target-manifest /path/to/cell/target_manifest.csv \
  --predictions /path/to/external_predictions.csv \
  --metadata /path/to/external_provenance.json \
  --protocol-sha256 ACTUAL_PROTOCOL_SHA256 \
  --alphas 0.05 0.1 0.2 \
  --output-dir /path/to/new_external_import
```

Use exactly the alphas in the frozen protocol. Outputs are `predictions.csv` and `import_report.json`; the successful status is deliberately `IMPORTED_NOT_REPRODUCTION_CERTIFIED`. Formal resubmission remains blocked until both required full comparators and the result-based paper revision are complete.

### Score frozen external predictions

After the import is frozen, use the matching runner cell's evaluated predictions for an explicit label join:

```bash
python -m paper2_admet_benchmark.revision_2026.score_external \
  --import-dir /path/to/new_external_import \
  --reference-predictions /path/to/cell/predictions.csv \
  --output-dir /path/to/new_external_scored
```

This command verifies the imported prediction hash **before reading labels**. It requires the same phase, endpoint, seed, reference model, exact sample universe at each alpha, structure IDs and eligibility. Reference methods must agree on labels and cohort facts. It does not refit thresholds or change the submitted sets after seeing outcomes.

The scored `predictions.csv` includes the joined `y` and `eligible`, so it can be combined with the corresponding internal predictions and passed to `summarize.py` using the external model name. `metrics_full.csv` counts all common eligible labeled molecules, including external unselected molecules as deferred doubletons. `metrics_selected.csv` reports only selected eligible labeled molecules with the original set flags. Both tables report selected counts and class retention; absent denominators remain undefined. `scoring_report.json` records input/output hashes. Neither successful import nor successful scoring certifies an upstream method reproduction.

When assembling an inference directory, place **one** copy of each internal prediction file and **only the scored** external files there. Do not include the label-free import files alongside their scored copies: those are earlier artifacts, not additional replicates. Pair the external `model` against `reference_model`; this is a full-method comparison and can include differences in the predictive model.
