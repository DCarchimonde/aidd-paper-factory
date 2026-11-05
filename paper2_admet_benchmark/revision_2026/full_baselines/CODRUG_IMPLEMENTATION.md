# CoDrug: complete, independent implementation from the paper

`codrug.py` implements the complete train → energy regularization → density
estimation → weighted prediction-set procedure. It does not use a precomputed
logistic classifier as a substitute for the learned molecular model. The
implementation is independent of the authors' original code; it must be cited
as a **paper-based independent reimplementation**, not an official or numerically
identical reproduction. The paper's linked repository,
<https://github.com/siddharthal/CoDrug>, was unavailable when this implementation
was prepared. No manual retrieval, programming, or baseline reconstruction is
left for the experiment operator.

## Sources and provenance

- S. Laghuvarapu, Z. Lin, and J. Sun, “CoDrug: Conformal Drug Property Prediction
  with Density Estimation under Covariate Shift,” NeurIPS 2023:
  [paper and supplementary appendix](https://proceedings.neurips.cc/paper_files/paper/2023/file/7691484a7a35d5e2742279c1d926b778-Paper-Conference.pdf).
  The consulted PDF SHA256 is
  `b1c36816e37b39bc84cb0ce415eaee378fef1043ddac0ae8025ba1c657f4a6a2`.
- The paper specifies DGL-LifeSci AttentiveFP. Our tensor operations independently
  implement its attention, GRU, and recurrent graph-readout construction,
  using these Apache-2.0 source files at DGL-LifeSci commit
  `be8bc71d29ecf34a9dab7c7bd47c08f3383d9be0`:
  [message passing](https://github.com/awslabs/dgl-lifesci/blob/be8bc71d29ecf34a9dab7c7bd47c08f3383d9be0/python/dgllife/model/gnn/attentivefp.py),
  [readout](https://github.com/awslabs/dgl-lifesci/blob/be8bc71d29ecf34a9dab7c7bd47c08f3383d9be0/python/dgllife/model/readout/attentivefp_readout.py),
  and [featurizers](https://github.com/awslabs/dgl-lifesci/blob/be8bc71d29ecf34a9dab7c7bd47c08f3383d9be0/python/dgllife/utils/featurizers.py).
  Their SHA256 hashes are respectively
  `239fb0ada0c521275f9767d06dc88b12b51afeb2cfcb35b0cfc8a916871e06a0`,
  `454d71ef66665c2fbe02d84c53872802b3c8bc76e81c6a1d3edca3e9e2dbfcd3`, and
  `52311b84715242fc4c42d869a6354f4ca20115d0646bc82158fb9de535ded889`.
  This source inspection establishes the implemented architecture; it does not
  establish that these were the authors' original dependency versions or
  initialization choices.
- The automatically acquired auxiliary pool is the public
  [chemical_vae ZINC250k snapshot](https://github.com/the-matter-lab/chemical_vae/blob/main/models/zinc/250k_rndm_zinc_drugs_clean_3.csv).
  The file SHA256 is
  `35e3f1a52b1badc0697e373d73a18ad773f415936ff992f4c6baa2e067b3e6ae`.
  A download or cached file with any other content is rejected. Only `smiles`
  is read; calculated ZINC properties are not training targets. The snapshot
  identity is pinned for this study, rather than claimed to be an author-provided
  manifest from 2023. Auxiliary compounds whose RDKit scaffold transformation
  fails are excluded and counted under `scaffold_failure`; such failures do
  not permit a molecule to bypass OOD filtering. Source-role failures still stop
  the run.

## Equations and code

Let `z=f(x)` be the two logits and let `p1` denote the class-1 softmax probability.

| Paper component | Implementation | Calculation |
|---|---|---|
| Appendix B.1 molecular network | `molecular_graph`, `build_attentivefp` | Bidirectional bonds, 39 atom and 10 bond features; AttentiveFP message passing and recurrent graph readout; 512 → 256 → 8 → 2 classifier. |
| Eq. 14 class probability | `_train`, `fit_predict` | `softmax(z/T)`; supervised CE uses source fit labels only. |
| Eq. 15 energy | `energy_penalty` | `E(x) = -T logsumexp(z/T)`. |
| Eqs. 16–17 objective | `energy_penalty`, `_train` | CE + λ times the mean squared positive part of `E(in)-m_in` and `m_out-E(out)`. OOD batches are sampled from the eligible unlabeled auxiliary pool. |
| Appendix B.3.2 OOD selection | `filter_auxiliary_smiles` | Remove auxiliary source identities and source scaffolds; an explicit fingerprint alternative implements the paper's strict minimum-similarity rule. |
| Eqs. 18, 21 and Appendix B.4 KDE | `fit_kde_ratios` | Separate Gaussian KDEs fit to calibration and target representations; each domain independently selects bandwidth by CV log likelihood. Ratios are target density divided by calibration density. |
| Eqs. 22–23 set construction | `weighted_sets_from_log_ratios` | LAC score `1-p(y|x)`, calibration score atoms weighted by density ratio, and a candidate-query weight at positive infinity. |
| Label-specific extension for this study | `weighted_sets_from_log_ratios(..., mondrian=True)` | The candidate label's calibration subset supplies its weighted score distribution; an absent calibration class yields an infinite threshold. |

The global weighted set is the direct equation-level CoDrug comparator and is
named `codrug_global`. The explicitly labeled class-specific extension is
`codrug_mondrian`; it is not presented as a distinct theorem from the paper.
Both remain in the main output so that the comparison does not silently benefit
from an unreported change in the calibration procedure.

The query weight is essential. If `A` is the sum of calibration weights, the
normalized calibration CDF is queried at
`(1-alpha) * (1 + w(query)/A)`. A required value above one gives an infinite
threshold, including both labels. Logarithmic ratio arithmetic avoids numerical
clipping without removing the query's infinity atom. The code neither replaces
this with an ordinary empirical weighted percentile nor assigns a unit query
weight without stating it.

## Reported settings and explicit choices

| Setting | Default | Evidence or study decision |
|---|---|---|
| Backbone and hidden sizes | AttentiveFP 512; FC 256, 8; two logits | Appendix B.1. |
| Optimizer / learning rate / batch size | Adam / 0.001 / 64 | Appendix B.1. |
| Energy loss coefficient | 0.01 | Appendix B.1. |
| Energy margins | `m_in=-5`, `m_out=-35` | Literal order printed in Appendix B.1; see ambiguity below. |
| Density representation | Two logits | CoDrug Energy variant; optional `density_representation="embedding"` uses the final eight-dimensional hidden representation and must be labeled separately. |
| KDE | Gaussian; separate domain fits | Section 4.4. |
| Bandwidth search | 25 linearly spaced values from `10**-1.3` through 10 | Explicit interpretation of Appendix B.4's uniformly spaced range. |
| KDE selection | Ten shuffled folds, seed fixed; fewer folds only when a domain has fewer than ten rows | Ten folds are reported; shuffle/seed and the small-sample rule are implementation choices. |
| Message-passing layers / readout iterations | 2 / 2 | Not specified by the CoDrug paper; locked here before formal runs. |
| FC activation / dropout / temperature / weight decay | ReLU / 0 / 1 / 0 | Not specified; locked here. |
| Epoch cap / checkpoint / patience | 100 / lowest router CE / 15 epochs | Not specified; source-router selection is a study adaptation. Calibration and target labels cannot select a checkpoint. |
| Class weights / scheduler | None / none | No such choices reported for CoDrug; this method's defaults are separate from KMM's training settings. |
| Auxiliary mini-batch size | Equal to the current fit mini-batch, sampled with replacement | Batch mixing was not specified; locked here. |
| OOD pool | All eligible pinned ZINC250k compounds | No default size cap. `aux_max_molecules` is for an explicitly reported pilot or sensitivity, never an unrecorded production shortcut. |
| OOD filtering | Scaffold exclusion using all source-role structures | Paper discusses training and calibration removal; router structures are also excluded in our three-role study. Empty acyclic scaffold is treated as a scaffold. |
| Fingerprint alternative | Morgan radius 2, 2048 bits | Paper states ECFP but does not specify these details for this filter; our choices are explicit and fixed. |

**Published margin ambiguity.** Equations 15–17 seek low in-domain energy and
high OOD energy. The printed margins `(-5,-35)` admit an overlapping zero-penalty
range; they do not require a separating gap. We preserve that printed order for
the primary reimplementation instead of guessing which value the unavailable
code used. The predeclared sensitivity with `margin_in=-35, margin_out=-5` must
be reported under a separate name. Its performance cannot be used to replace
the primary result after inspecting target outcomes.

**Study split adaptation.** The paper reports a 70/15/15 training/calibration/test
split. This repository uses the same frozen fit/router/conformal source roles
and target population as the other revision methods. This is necessary for an
apples-to-apples study, but is not reproduction of the paper's original splits
or numerical benchmark tables. Source-role identity overlap is rejected. Target
identity overlap is counted and handled by the orchestrator's common eligibility
mask rather than used to train the model.

Appendix B.3.2's opening description includes calibration structures in auxiliary
exclusion, although its subsequent scaffold bullet mentions only training. Our
choice follows the broader opening description. Consequently calibration **X**
affects the eligible auxiliary pool and hence training; calibration **y** does
not. This is a label-free dependence, not full independence of the trained model
from the calibration covariates. Standard split-conformal finite-sample validity
cannot simply be asserted for this training procedure. This caveat is additional
to the use of estimated representation-space density ratios.

**Fingerprint filtering can legitimately fail.** It retains an auxiliary
compound only when every source similarity is below the minimum source-pair
similarity. A zero minimum can make the eligible pool empty. The implementation
raises an explicit error in that case; it does not relax the paper's rule and
still call the result an exact implementation of that rule. The complete
scaffold-based primary workflow does not depend on this optional alternative.

## API, resumption, and independent KMM backbone

The unified revision executor calls:

```python
result = fit_predict(
    train_smiles, train_y, tuning_smiles, tuning_y,
    cal_smiles, cal_y, target_smiles,
    alphas=alphas, seed=seed, device=device, jobs=jobs,
    model_dir=model_dir, config=config,
)
```

There is deliberately no `target_y` parameter. Outputs include target
probabilities, calibration and target logits, hidden and graph embeddings,
both set families for every requested alpha, and complete configuration,
training, auxiliary-filter and KDE diagnostics.

The atomic epoch checkpoint contains network and optimizer states, selected
weights, optional scheduler, epoch history, and NumPy/PyTorch RNG states.
Its signature binds the fit/router labeled identities, calibration structures,
eligible auxiliary pool, training settings, seed, and implementation-file hash.
A directory holding an incompatible checkpoint is rejected. Calibration
outcomes and target outcomes cannot change training; target features are used
only after selecting the network. Use separate model directories for different
method settings and sensitivity runs. A code change intentionally invalidates
an old checkpoint rather than silently treating it as a compatible model.

Repeated auxiliary processing uses bounded, process-local LRU caches:
canonical SMILES and scaffold strings each have at most 300,000 entries, and
the last eight filtered pools are retained as immutable tuples plus JSON
metadata. No RDKit molecule objects enter these caches. Pool keys contain the
ordered auxiliary input, source structures, filtering settings, size limit,
and seed; changing energy margins does not rerun identical chemistry. Returned
lists and metadata dictionaries are private copies. This preserves the original
row-level exclusion counts, retained-pool hash and deterministic subsampling.
KDE cross-validation uses joblib threads, avoiding child-process imports and
copies of an already loaded neural network; fixed folds, scores and selected
bandwidths are unchanged.

For KMM-CP, the executor passes its own paper's backbone configuration with
`energy_weight=0` and `compute_conformal=False`. This trains a separate vanilla
AttentiveFP, needs no ZINC pool, returns the requested embeddings and probabilities,
and does not reuse CoDrug's energy-trained embeddings. The trainer supports
KMM's inverse-frequency loss, learning-rate scheduler, and training-loss stopping
rule; these are recorded in that method's configuration. Actual parameter counts
are measured and reported; no unverified parameter-count claim is copied from
either paper.

## Validation and interpretation

Run with the full requirements installed:

```bash
python -m unittest paper2_admet_benchmark.revision_2026.full_baselines.test_codrug -v
```

The eleven tests cover weighted quantiles including query infinity mass,
invariance to a common log-weight offset, missing calibration classes, independent
KDE fits, auxiliary hash checks, OOD exclusion and scaffold-error accounting, graph node permutation and
single-versus-batch equivalence including isolated atoms, the exact energy-loss
value and finite gradients, cold/warm/bypassed-cache equivalence and energy-margin
cache reuse, one-versus-two-thread KDE equivalence, complete molecular-graph training and checkpoint
reuse, and the independent vanilla KMM mode. They passed with PyTorch 2.5.1 CPU
and RDKit 2026.03.4. The small graph-training fixtures test executable behavior;
they are not scientific evidence of benchmark coverage, superiority, or paper
numerical replication.

An additional comparison using the first 5,000 rows of the pinned ZINC file
confirmed identical retained molecules and every exclusion count between the
original uncached implementation, a cold cache, and a warm cache. In that check
the timings were 2.22 s, 2.29 s, and 0.00038 s respectively; this is a local
preprocessing check, not a prediction-quality or full-workflow runtime claim.

Estimated KDE ratios in learned representation space do not supply an exact
finite-sample coverage guarantee. Representation compression, density estimation
error, target self-inclusion, finite-sample bandwidth selection, and departures
from covariate shift remain relevant. Full-run coverage, class coverage, set
size, empty sets, and eligibility must be computed on the common target
population by the repository's reporting stage, without selecting configurations
on those outcomes.
