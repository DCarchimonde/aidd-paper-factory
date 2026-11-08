# Finite-batch dual-class triage: theory and positioning

Version: 2026-11-08. This document specifies the implemented certification target and the proposed methodological increment. The count confidence sequence is an existing statistical construction; the technical focus here is its exact robust counterpart for two unknown class denominators, joint three-action allocation, and acquisition directed at certified assay demand.

## 1. Target and probability space

Condition on a fixed batch of `N` compounds, all endpoint labels, frozen predictions, and a partition into `H` nonempty strata determined without viewing the audit outcomes. Labels are binary endpoint outcomes, not generic clinical safety judgments. Randomness for the certificate comes from uniform sampling without replacement within each stratum. The next stratum may be selected adaptively using all previously queried labels.

At a stopping time, each unqueried stratum remainder receives one action: automatic label 0, automatic label 1, or assay. Previously queried compounds are handled according to the observed reference label and are charged as assays. The target is simultaneous control of the realized errors in the **current entire batch**:

\[
R_{\rm FN}=\frac{\#\{\text{automatic 0 and true 1}\}}{\#\{\text{true 1 in the batch}\}},\qquad
R_{\rm FP}=\frac{\#\{\text{automatic 1 and true 0}\}}{\#\{\text{true 0 in the batch}\}}.
\]

These are class-conditioned error exposures for the complete assay-assisted labeling system. They are not errors conditional on automatic release, predictive values, false discovery/omission rates, or expected deployment risks. A zero true class count is assigned zero error computationally, with an explicit absence-possible flag; it supplies no empirical evidence about performance on that absent class.

The desired statement is

\[
\Pr\{\text{both }R_{\rm FN}\leq\epsilon_1\text{ and }R_{\rm FP}\leq\epsilon_0
\text{ at every certified stopping state}\}\geq 1-\delta.
\]

It applies to one frozen batch and partition. It does not extend to unseen future compounds. No source-to-target exchangeability, correct probability calibration, or estimated density ratio is needed for this design-based statement. Uniform within-stratum sampling, fixed membership, and accurate audit bookkeeping are essential. Labels represent the chosen assay reference, not an error-free account of biology.

## 2. Existing finite-population count confidence sequence

For stratum `h`, write `N_h` for its original size, `t` for the number queried, `S_t` for queried positives, and `K_h` for its unknown original positive total. Under uniform sampling without replacement,

\[
f_t(s;K)=\Pr_K(S_t=s)=
\frac{\binom K s\binom{N_h-K}{t-s}}{\binom{N_h}t}.
\]

For a prior over candidate totals fixed before querying, let `Q_t(s)` be its mixture likelihood for the observed count. The mixture-to-point likelihood ratio is

\[
E_t(K)=\frac{Q_t(S_t)}{f_t(S_t;K)}.
\]

For the true total, this is a nonnegative supermartingale under the sequential without-replacement experiment. Boundary totals can create support differences, giving the supermartingale formulation. Ville's inequality yields a time-uniform confidence set by retaining totals with

\[
f_t(S_t;K)\geq\delta_h Q_t(S_t)
\]

and intersecting the sets over all observed prefixes. The primary implementation fixes the following prior on the original positive total before any audit:

\[
\pi(K)=0.4\mathbf1\{K=0\}+0.4\mathbf1\{K=N_h\}+0.2\operatorname{BetaBinomial}(K;N_h,1/2,1/2).
\]

Its predictive count likelihood is

\[
Q_t(s)=0.4\mathbf1\{s=0\}+0.4\mathbf1\{s=t\}+0.2\operatorname{BetaBinomial}(s;t,1/2,1/2).
\]

At `t=0` this is one. The boundary components place substantial fixed mass on nearly pure strata; the positive Beta-Binomial component gives support to every total. A uniform-total prior instead has `Q_t(s)=1/(t+1)`. Both choices use the existing mixture construction. Prior choice changes efficiency, not the nature of the guarantee. Choosing whichever prior gives the tightest bound after seeing audit outcomes would require a further valid combination and is not allowed here.

Inclusive boundary retention and the implementation's conservative numerical guard can enlarge the set. At zero queries the interval is `[0,N_h]`; a fully queried stratum has its exact observed total. This is the prior/posterior likelihood-ratio specialization of Waudby-Smith and Ramdas (2020), not a new confidence-sequence theorem.

Allocate `delta_h` before viewing audit labels, with `sum_h delta_h <= delta`; the default is `delta/H`. A union bound supplies the event

\[
\mathcal E=\{K_h\text{ belongs to its running interval for every }h\text{ and every local query count}\},
\quad \Pr(\mathcal E)\geq 1-\delta.
\]

One can imagine a random permutation of each stratum fixed before querying. Adaptive cross-stratum choices reveal prefixes of these permutations. The event already covers every prefix, so adaptive querying and stopping need no additional per-look failure spending. Independent strata are not required for the union bound. This certificate is not automatically a simultaneous `1-delta` guarantee across separately chosen partitions or unrelated experimental runs.

## 3. Remaining counts and coupled denominators

After the current audit, let `n_h` and `s_h` be the number queried and the positives observed in stratum `h`. Set

\[
M_h=N_h-n_h,\quad a=\sum_hs_h,\quad b=\sum_h(n_h-s_h).
\]

If the original-total interval is `[ell_h,u_h]`, remaining positive totals `k_h` satisfy

\[
L_h=\max(0,\ell_h-s_h)\leq k_h\leq
U_h=\min(M_h,u_h-s_h).
\]

On `E`, the full batch class totals are simultaneously

\[
N_1=a+\sum_hk_h,\qquad N_0=b+\sum_h(M_h-k_h).
\]

Let `c_h=1` mean automatic label 0 and `d_h=1` automatic label 1; require `c_h+d_h<=1`. The two risks are

\[
R_{\rm FN}(k)=\frac{\sum_hc_hk_h}{a+\sum_hk_h},\qquad
R_{\rm FP}(k)=\frac{\sum_hd_h(M_h-k_h)}{b+\sum_h(M_h-k_h)}.
\]

The same unknown count appears in the numerator and denominator. Treating their bounds independently generally loses this coupling.

## 4. Exact robust counterpart over the count box

For `C={h:c_h=1}`, the false-negative risk is nondecreasing in counts inside `C` and nonincreasing in counts outside `C`. Therefore its maximum over the integer box is

\[
\overline R_{\rm FN}(C)=
\frac{\sum_{h\in C}U_h}{a+\sum_{h\in C}U_h+\sum_{h\notin C}L_h}.
\]

Writing `D={h:d_h=1}`, the false-positive maximum is

\[
\overline R_{\rm FP}(D)=
\frac{\sum_{h\in D}(M_h-L_h)}
{b+\sum_{h\in D}(M_h-L_h)+\sum_{h\notin D}(M_h-U_h)}.
\]

Zero denominators use the stated zero-error convention. Equivalently, the inequalities can be checked without division, preserving exact integer arithmetic at the feasibility boundary. For `0<epsilon_1,epsilon_0<1`, robust feasibility is exactly

\[
\sum_h[(1-\epsilon_1)U_h+\epsilon_1L_h]c_h
\leq \epsilon_1(a+\sum_hL_h),
\tag{FN}
\]

\[
\sum_h[(1-\epsilon_0)(M_h-L_h)+\epsilon_0(M_h-U_h)]d_h
\leq\epsilon_0(b+\sum_h(M_h-U_h)).
\tag{FP}
\]

A direct proof subtracts `epsilon` times the denominator from the numerator. For (FN), the coefficient of `k_h` is `1-epsilon_1` in `C` and `-epsilon_1` elsewhere; maximize each at its appropriate box endpoint. The same calculation applied to negative counts proves (FP). The two risk maxima need not occur at the same count vector: requiring both separate maxima to satisfy their limits is exactly requiring both inequalities at every vector in the shared box.

“Exact” refers to the worst case over this rectangular confidence set. It does not mean that the rectangle itself is the smallest possible joint confidence region or that its coverage is exactly `1-delta`. In particular, a uniform-total prior can be extremely conservative when the allowed number of missed rare positives is only one or two; even a fixed boundary mixture cannot remove this information cost. A safe procedure may require nearly a census.

The denominator constraints can also be written as signed-loss sums, for example `sum_i Y_i(1{automatic 0}-epsilon_1)<=0`. This relates the class-denominator constraints to standard risk-control formulations; the computational step here is the exact finite-box counterpart and its joint action allocation.

## 5. Three-action optimization and certificate

With equal unit assay costs, solve

\[
\max_{c,d}\quad L=\sum_hM_h(c_h+d_h)
\]

subject to (FN), (FP), `c_h,d_h in {0,1}`, and `c_h+d_h<=1`. Unselected remainders are assayed. Current total assay demand is

\[
C^*=\underbrace{a+b}_{\text{already queried}}+
\sum_hM_h(1-c_h-d_h)=N-L.
\]

The core supports fixed stratum cost weights, but the primary experiment uses unit costs; no currency or real laboratory price is inferred. A final capacity `B` can be satisfied by the current certificate exactly when `C*<=B`. Deferring everything is always statistically safe; it may exceed a requested operational capacity. These are different notions of feasibility.

On `E`, every robust-feasible assignment satisfies both actual risk limits. Therefore the optimizer may use the audit results to select, expand, shrink, or change both automatic actions without another correction for the `3^H` possible assignments. This follows because the shared count event already covers every assignment. The implementation independently rechecks the returned actions against the exact worst-case fractions. Empty confidence intersections, nonoptimal solver results, or failed verification cannot authorize automatic decisions; the caller must retain manual handling and report the failure.

The optimization is exact only within the frozen stratum-union action class at the current state. It is not globally optimal over arbitrary subsets of compounds or over all future querying policies. A policy cannot pick a score-dependent subset inside a certified stratum while reusing the whole-stratum count bound. Such subsets require their own predeclared count structure.

## 6. Acquiring labels for reduced certified assay demand

For each candidate stratum, the acquisition rule predicts outcomes of a query block of at most four compounds, updates a hypothetical count state, and resolves its allocation problem. Its objective is the predicted increase of `L`, equivalently the predicted decrease in total assay demand `C*`. Queried compounds are removed from the automatically decidable remainder in each forecast, so their cost is included.

The implemented prediction uses a smoothed rate based on the frozen score and revealed labels, with Binomial outcome weights. For a hypothetical block, it intersects the existing interval with the future endpoint interval, without inventing an ordering of unobserved intermediate labels. Actual observations retain every prefix. An empty hypothetical intersection is assigned zero automatic savings for that forecast. This is a model-based short-horizon heuristic. The actual finite-population outcome distribution need not equal that forecast; the rule has no global optimality or regret claim. Hypothetical outcome calculations never query the actual oracle. Once a stratum is chosen, actual compounds are sampled uniformly without replacement and the count confidence sequence is updated after each real label.

The count certificate supplies validity even with a poor model. The query heuristic is evaluated by its reduction in **total** assay demand against uniform, entropy, and Neyman-style acquisition using the same certificate. Nested-threshold, independent-ratio, score-only, and uniform-total-CS controls examine the remaining components. The score-by-support partition has at most eight nonempty strata, while the score-only partition has eight in the present pools. Their actual `H` values need not match. Each run records `H`, stratum sizes and `delta/H`, so the partition comparison can account for differences in both resolution and confidence allocation. Improved forecasting can change efficiency without changing the certificate.

## 7. Core implementation contract

| API in `core.py` | Meaning |
|---|---|
| `CountCS(N, delta, prior=...)` | One frozen stratum's original-total running confidence interval |
| `observe(y)` / `observe_many(labels)` | Consume the next actual uniform-WOR label(s), retaining every prefix |
| `remaining_bounds()` / `snapshot()` | Remaining positive count interval and auditable state |
| `update_count_interval(...)` | Stateless update given an already verified prefix history |
| `certify_triage(...)` | Exact worst-case two-risk check for whole-stratum automatic actions |
| `optimize_triage(...)` | Joint MILP plus independent verification of its solution |

The surrounding data layer must keep unqueried labels out of the public feature/score view. All real observations go through a query ledger. Tests must cover count-CS enumeration on small populations, exact-corner agreement with exhaustive count assignments, optimized decisions against exhaustive small action spaces, arbitrary audited prefixes, zero-class/zero-query/full-query cases, solver failure, and absence of double counting in assay savings. Numerical tests support implementation correctness; they do not replace the probability argument or demonstrate molecular efficiency.

## 8. Closest work and the boundary of the contribution

| Prior work | Already established | Focus of this study |
|---|---|---|
| Candès, Ilyas and Zrnic, **Probably Approximately Correct Labels**, ICML 2026 | Fixed-pool random-audit labeling; adaptive post-audit uncertainty thresholds; uncertainty multicalibration and cost-sensitive routing | Two true-class denominators, joint nonnested 0/1/assay allocation, and certified-demand acquisition. A threshold comparator is essential; “post-audit adaptation” alone is not novel. |
| Zhang, Huang and Jin, **CARED**, 2026 preprint | Randomized review under workflow/distribution shift and class-specific long-run error control | A high-probability certificate for the realized errors of one finite batch at adaptive stopping times. Long-run and single-batch guarantees must not be conflated. |
| Xu, Karampatziakis and Mineiro, **Active, anytime-valid risk controlling prediction sets**, NeurIPS 2024 | Active labeling, anytime validity, prediction assistance and optimal-query analysis for its risk-control setting | Adaptive without-replacement acquisition from a fixed finite batch with joint class-denominator constraints and a total assay objective. Active querying itself is not a contribution. |
| Bai and Jin, **SCoRE**, 2026 preprint | General bounded deployment risks, risk/reward selection, distribution-shift extensions, and drug-discovery applications | Realized batch FNR/FPR have different denominators and a different probability target. Chemical costs or risk control alone do not distinguish this work. |
| Hu et al., **Exact Finite-Population Sequential Auditing for Selective Release from Model-Proposed Candidate Sets**, Mathematics 2026 | Frozen-candidate random audit, a current-batch missed-adverse share, sequential release/fallback, and complete review-cost accounting; Lemma 2 already preserves the coupled denominator | Multiple strata and both unknown class totals with jointly selected automatic labels, rather than one fixed candidate release region. Coupled-ratio reasoning or net-workload accounting alone is not new. |
| Anthony and Salehzadeh Nobari, **Finite-Sample Coverage Audits for High-Recall Candidate Generation**, 2026 preprint | Excluded-pool and two-pool recall certification, finite-corpus lower bounds, and selection among predeclared nested prefixes | Two simultaneous class risks and joint nonnested actions with adaptive allocation. Adding a rare-class label or stratification alone is insufficient. |
| Waudby-Smith and Ramdas, **Confidence sequences for sampling without replacement**, NeurIPS 2020 | Time-uniform finite-population/hypergeometric inference | Used directly as the count-certification component; no new CS theorem is claimed. |

The narrow proposed increment is a **shared finite-population count region, exact coupled robust constraints for two unknown class totals, a jointly optimized three-action policy, and acquisition aimed at its net assay demand**. Even the single-region coupling between a missed-positive numerator and its class denominator already appears in finite-population selective-release work; our proposed computational extension is joint multi-stratum allocation for both classes. The molecular experiments test whether these components reduce total assay demand.

The implemented threshold control orders the nonempty score-by-support strata by their mean frozen score, and restricts automatic 0 to a prefix and automatic 1 to a nonoverlapping suffix. It is a study-specific class-risk adaptation under the shared certifier. It is an adaptation to the present class-risk target, rather than a reproduction of PAC Labels or a test of its original global-error target.

## 9. What would support a paper

The question is whether the coupled certificate and resulting decisions save a meaningful number of assays at the same two error limits and confidence level. Report the eight methods separately, actual and certified errors, queried and ultimately deferred counts, computation, and small-batch failures.

The model, strata and assay labels may be arbitrary fixed features of the batch for certification. This robustness comes from paying for randomized target labels; it is not evidence that unlabeled transport diagnostics detect conditional shift. Public molecular holdouts provide retrospective replay, not prospective laboratory validation. The fixed 20 audit seeds quantify randomization variability on those particular pools and cannot, by themselves, empirically establish a 5% failure bound.

## Primary sources

1. Candès, E., Ilyas, A., Zrnic, T. (2026). *Probably Approximately Correct Labels*. ICML, PMLR 306:10984–11001. [Final proceedings](https://proceedings.mlr.press/v306/candes26a.html); [public full-text version](https://arxiv.org/html/2506.10908v3). See Section 2, Algorithm 1, Section 2.1 and Section 3.
2. Zhang, X., Huang, Z., Jin, Y. (2026). *Conformal Annotation with Randomized Expert Review: Application to Online Cell-Type Labeling*, preprint (CARED). [arXiv record](https://arxiv.org/abs/2610.40252); [PDF](https://arxiv.org/pdf/2610.40252). See Section 3.3, Theorem 3.3, Corollary 3.4 and the finite-sample discussion.
3. Xu, Z., Karampatziakis, N., Mineiro, P. (2024). *Active, anytime-valid risk controlling prediction sets*. [Primary full text](https://arxiv.org/html/2406.10490v2). See Sections 2.1–3.
4. Bai, T., Jin, Y. (2026). *Conformal Selective Prediction with General Risk Control*. [Primary full text](https://arxiv.org/html/2603.24704v1). See risk definitions, Section 7.1 and Appendices A.4–A.5.
5. Waudby-Smith, I., Ramdas, A. (2020). *Confidence sequences for sampling without replacement*. NeurIPS 33. [Proceedings](https://papers.nips.cc/paper_files/paper/2020/hash/e96c7de8f6390b1e6c71556e4e0a4959-Abstract.html); [paper](https://proceedings.neurips.cc/paper_files/paper/2020/file/e96c7de8f6390b1e6c71556e4e0a4959-Paper.pdf).

6. Hu, Y., Zhou, Z., Zhu, Y., Liang, K. (2026). *Exact Finite-Population Sequential Auditing for Selective Release from Model-Proposed Candidate Sets*. Mathematics 14(18):3427. [Publisher](https://www.mdpi.com/2227-7390/14/18/3427); [DOI](https://doi.org/10.3390/math14183427). See Section 3, especially Lemma 2 and the sequential-policy evaluation.
7. Anthony, M., Salehzadeh Nobari, K. (2026). *Finite-Sample Coverage Audits for High-Recall Candidate Generation: Certification and Learning-Theoretic Design*, preprint. [Primary full text](https://arxiv.org/html/2607.21480v1). See Sections 4.2–4.8 and 9.
