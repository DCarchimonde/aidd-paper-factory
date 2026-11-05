# Literature and comparator implementation record

Updated 2026-11-05. Scope: CHEMOLAB-D-26-01118, separate from the later CYP/SAR project.

## Current implementation status

The repository now includes executable independent implementations of the full CoDrug and KMM-CP comparison procedures, including Selective KMM-CP's joint selection and refitting mechanism. Training, prediction, calibration and provenance interfaces are integrated into the revision workflow. The second author runs the supplied commands and returns their outputs; implementing these algorithms or obtaining unavailable author repositories is not a remaining assignment.

Numerical/unit tests and small synthetic runs verify software behavior. They are not evidence of molecular prediction performance. Final comparison tables and conclusions must be generated from completed real-data runs. **Independent paper-based implementation**, **author-code reproduction**, and **replication of the original reported numbers** are distinct claims. Only the first describes the implementations supplied here.

## Scientific question and evidence standard

The difference between marginal and class-conditional conformal coverage is established knowledge. The ClinTox illustration cannot be promoted into a new statistical principle. TAME's add-only construction also makes empirical coverage nondecreasing on a fixed cohort by construction. Its scientific value therefore depends on whether its additions prevent meaningful errors more effectively than simpler rules at the same information/deferral cost.

The revision measures that question directly, alongside strong published methods. It reports both class coverage and operational consequences, including missed activity, singleton yield, abstention and selection composition. A gate based on effective sample size, weight clipping or domain AUC does not prove that the conditional outcome distribution is unchanged across domains.

## Sources and implemented comparisons

| Method | Primary source | Implementation and interpretation |
|---|---|---|
| Marginal / Mondrian split CP | [Vovk, 2012](https://proceedings.mlr.press/v25/vovk12.html) | `core.py`: finite-sample rank correction; threshold indexed by candidate label for Mondrian. |
| Likelihood-ratio weighted CP | [Tibshirani et al., NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html) | Estimated domain-classifier ratios with query mass at infinity; estimated/clipped ratios receive no oracle guarantee. |
| APS | [Romano, Sesia and Candès, NeurIPS 2020](https://proceedings.neurips.cc/paper_files/paper/2020/hash/244edd7e85dc81602b7615cd705545f5-Abstract.html) | Deterministic cumulative candidate score and label-0-first tie rule; marginal and candidate-class versions. No randomization is claimed. |
| CoDrug | [Laghuvarapu, Lin and Sun, NeurIPS 2023, full paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/7691484a7a35d5e2742279c1d926b778-Paper-Conference.pdf) | `full_baselines/codrug.py`: molecular graph classifier, energy regularization with an auxiliary unlabeled pool, KDE bandwidth selection and weighted prediction sets. |
| KMM-CP / Selective KMM-CP | [Laghuvarapu, Deb and Sun, UAI 2026](https://proceedings.mlr.press/v337/laghuvarapu26a.html) | `full_baselines/kmm_cp.py`: permutation-based bandwidth choice, standard and joint-selection QPs, thresholding, standard-KMM refit and empirical weighted calibration. |
| Confidence rejection | [El-Yaniv and Wiener, JMLR 2010](https://jmlr.org/papers/v11/el-yaniv10a.html) | Argmax classification with a fixed abstention count; a practical control, not a claim to reproduce every selective-classification method. |
| Similarity/AD rejection | [Sahigara et al., 2013](https://pmc.ncbi.nlm.nih.gov/articles/PMC3679843/) | Maximum training-set Tanimoto similarity ranking; no claim to reproduce every published AD algorithm. |
| Budget-matched additions | Study control | Random, uncertainty and similarity rankings match how many `{0}` and `{1}` baseline singletons TAME expands. Outcomes do not define rankings or budgets. |

The older `density_baselines.py` KDE/KMM procedures remain explicitly labelled components. They diagnose representation/reweighting effects but do not replace full reference procedures. RAPS is contextual literature, not an implemented comparison claimed in the matrix.

## CoDrug: included full procedure and explicit choices

The full paper was checked; its PDF SHA-256 is `b1c36816e37b39bc84cb0ce415eaee378fef1043ddac0ae8025ba1c657f4a6a2`. The implementation trains an AttentiveFP graph classifier with the published energy-margin objective and an auxiliary unlabeled molecular pool after structural exclusion. It chooses calibration/target KDE bandwidths from covariates, then forms weighted conformal sets including each query's mass at infinity. Stable log-ratio arithmetic avoids silently clipping large density ratios. The code returns global CoDrug and an explicitly identified candidate-class Mondrian adaptation.

The reported graph/classifier dimensions are `512 → 256 → 8 → 2`, with Adam learning rate `0.001`, batch size 64 and energy penalty weight `0.01`. Default KDE uses the two output logits for the energy-trained model. The 8-dimensional classifier embedding is retained separately; selecting it is a recorded representation variant.

The code preserves the appendix's printed margin order, `m_in=-5`, `m_out=-35`. This differs from the usual energy-separation order; it is not silently changed. A swapped-margin analysis, if included in the frozen matrix, is explicitly a sensitivity analysis. The paper does not fully specify message-passing/readout counts, FC nonlinearities, maximum epochs or checkpoint selection. The implementation records two message-passing layers, two readout iterations, ReLU FC layers, dropout 0, temperature 1, weight decay 0, maximum 100 epochs, and router-cross-entropy checkpoint selection with patience 15. These are disclosed fixed choices rather than reconstructed author settings.

The auxiliary ZINC snapshot is acquired automatically and checked against a pinned SHA-256. Structural filtering and any explicit auxiliary-size limit are recorded. Supervised fitting uses the fit role; checkpoint selection uses the router role; calibration labels form conformal scores only. No target outcome labels enter training or KDE fitting.

The independent PyTorch graph implementation follows [DGL-LifeSci AttentiveFP](https://github.com/awslabs/dgl-lifesci/blob/be8bc71d29ecf34a9dab7c7bd47c08f3383d9be0/python/dgllife/model/gnn/attentivefp.py) and its [readout](https://github.com/awslabs/dgl-lifesci/blob/be8bc71d29ecf34a9dab7c7bd47c08f3383d9be0/python/dgllife/model/readout/attentivefp_readout.py). It does not require DGL binary packages. The actual architecture, parameter count, checkpoint, dependencies, auxiliary identities and KDE choices are saved. Equations and configuration details are in `full_baselines/CODRUG_IMPLEMENTATION.md`.

The paper-named [CoDrug author repository](https://github.com/siddharthal/CoDrug) returned HTTP 404 to this environment. This explains the independent implementation; it is not a claim that the authors removed their code. No request to the second author to recreate the method remains.

## KMM-CP: final publication checked and complete procedure included

The final 25-page [UAI paper and supplements](https://raw.githubusercontent.com/mlresearch/v337/main/assets/laghuvarapu26a/laghuvarapu26a.pdf) were downloaded and read. Their SHA-256 is `dbcc53ccd8a594b076f698f1fb2516cae701af803be8979eef3af788bbdae5ec`. The relevant algorithm statements agree with [arXiv v1](https://arxiv.org/html/2603.26415v1).

Ordinary KMM minimizes kernel mean discrepancy under bounded source weights. Selective KMM jointly solves for source weights and target selection weights, retains targets with selection weight at least `0.2`, then **refits ordinary KMM against the retained cohort**. Reported defaults `B=30` and `tau=0.5` are preserved. Realized retention is measured: `mean(selection_weight)>=0.5` does not imply exactly 50% of rows pass the threshold.

Bandwidths are selected from `0.01, 0.1, 0.5, 1, 2` times the cross-domain median distance by maximizing permutation-standardized MMD, as in Appendix E.5. The paper leaves several numerical choices unstated. This implementation fixes epsilon to 0, uses 100 seeded permutations and unbiased MMD squared, retains the selected kernel for refitting, and documents degenerate distance/variance handling. OSQP 1.0.4 replaces CVXOPT; the scientific objective has no added ridge. Solver status and constraint residuals must pass. Failed fits are not converted into ordinary-CP results.

The paper uses a normalized weighted **empirical** quantile with no query atom at infinity and no ordinary `(n+1)` correction. This differs from Tibshirani's oracle-weighted construction and has an asymptotic, assumption-dependent interpretation.

The final paper contains a Mondrian inconsistency: Appendix D calibrates by true class, but Algorithm 2 applies the predicted class's threshold to every candidate. Both interpretations are reported separately:

| Method suffix | Meaning |
|---|---|
| `global` | One empirical weighted threshold |
| `mondrian_candidate` | Candidate `y` uses threshold `q_y` |
| `mondrian_paper_predicted` | Every candidate uses the predicted class's threshold, as printed in Algorithm 2 |

All suffixes exist for `kmm_cp_` and `skmm_cp_`. The literal-pseudocode control is not called a valid class-conditional method. Empty or zero-weight calibration classes receive explicit infinite thresholds and are recorded.

The full KMM reference uses a separately trained vanilla AttentiveFP: graph size 512, classifier hidden size 64, three message-passing layers, inverse-frequency cross entropy, no energy penalty, Adam `0.001`, StepLR `gamma=0.8` per epoch, batch 128, at most 50 epochs and training-loss patience 5. Two readout iterations are a disclosed framework default because the paper does not specify them. The 64-dimensional penultimate classifier embedding is distinct from the 512-dimensional graph embedding. Actual parameter counts are recorded rather than assumed equal to the paper's rounded figure. Details and tests are in `full_baselines/KMM_IMPLEMENTATION.md`.

The [KMM-CP author repository](https://github.com/siddharthal/KMM-CP) also returned 404 here. The complete independent implementation executes without it.

## Fair comparison, denominators and chemical meaning

1. **Components versus full methods.** Shared-backbone comparisons keep probabilities, source roles, target molecules and scores fixed. Full CoDrug/KMM comparisons include their trained representations and are labelled accordingly. Representation gains cannot be attributed solely to a conformal wrapper.
2. **Matched additions.** Counts are matched separately within original `{0}` and `{1}` baseline singleton groups. Empty-to-full repair is common to expansion controls and is not counted as transport gain.
3. **Selection.** Every SKMM run returns raw sets and a selected mask. Selected-only coverage uses selected denominators. Full-cohort accounting treats unselected molecules as deferred full sets and retains them in the denominator. Report retention overall and by true class. Selected empty sets remain distinct from unselected molecules.
4. **Inference.** Report every prespecified endpoint/seed, class sample size and failure. Repeated seeds do not create independent molecules. Resampling preserves molecular grouping and paired predictions.
5. **Assay meaning.** [Tox21](https://www.frontiersin.org/journals/environmental-science/articles/10.3389/fenvs.2015.00085/full) labels denote nuclear-receptor/stress-response assay activity. Positivity does not establish clinical toxicity; an inactive prediction does not establish safety. Protecting label 1 has a missed-activity interpretation, different from protecting label 0.
6. **Distinct risks.** Report `P(set={0}|Y=1)` separately from `P(Y=1|set={0})`, with inactive-singleton yield and deferral. Neither is a demonstrated clinical safety risk.
7. **Data history.** Previously opened EPA labels remain retrospective. New seeds or hashes cannot make them newly blinded. Independent confirmation has its own prespecified source, disjoint identities and outcome-access record.

## Optional external predictions

`external_adapter.py` and `score_external.py` remain available for additional future methods or independently supplied predictions. They are **optional extension interfaces**, not a dependency for executing the included CoDrug/KMM-CP implementations and not a second-author coding assignment.

The adapter checks the exact target ID/alpha universe, protocol/manifest hashes, binary include/selection flags and source metadata. It preserves raw sets and selected flags, while making unselected full-cohort rows explicitly deferred. The scorer verifies prediction hashes before joining outcomes and checks common eligibility. Successful import/scoring certifies those joins, not upstream method validity. Imported implementation kinds distinguish official code, independent paper implementations and partial components.

Actual method completion, source hashes and evaluated outputs come from execution manifests and completion files. Publication claims require the resulting real-data evidence; this literature record is not itself a completed experiment report.
