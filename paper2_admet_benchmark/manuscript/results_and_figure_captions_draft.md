# Paper 2 Results and Figure Captions — Frozen-Evidence Draft

## Writing status and evidence boundary

This draft is written exclusively from the frozen confirmatory outputs and manuscript-assets tables. The numerical values reported below are descriptive means across the frozen model/regime combinations unless a paired model-specific confidence-interval count is stated explicitly. Model families are not treated as independent inferential replicates. Inferential stability is based on within-model, cross-seed confidence intervals and paired comparisons under the same endpoint, split, seed, and model. Seed 99 is excluded from every scientific conclusion.

---

# 3. Results

## 3.1 Chemical distribution shift altered predictive performance and probability calibration

Predictive performance was consistently strongest under random splitting, although the magnitude and form of degradation under label-blind chemical shift depended on the endpoint and task (Figure 2; Table 2). For BBBP, the descriptive mean ROC-AUC decreased from 0.886 under random splitting to 0.864 under similarity-cluster splitting and 0.851 under scaffold splitting. Balanced accuracy showed a parallel decline from 0.789 to 0.748 and 0.736, respectively, whereas PR-AUC remained comparatively high across the three designs (0.940–0.949). Thus, BBBP retained useful discrimination under chemical shift, but its class-balanced performance was measurably lower outside the random-split setting.

ClinTox was substantially more difficult. Mean ROC-AUC ranged from 0.675 to 0.703, PR-AUC from 0.243 to 0.283, and balanced accuracy from 0.537 to 0.575. Random splitting again yielded the strongest values, whereas scaffold splitting produced the lowest PR-AUC (0.243) and balanced accuracy (0.537). These results indicate that apparently acceptable aggregate probability scores for ClinTox should not be interpreted as strong minority-class discrimination.

Regression endpoints showed larger endpoint-specific effects. ESOL performance deteriorated sharply under scaffold splitting: RMSE increased from 1.441 under random splitting to 2.035, MAE increased from 1.091 to 1.613, and mean R² decreased from 0.488 to 0.089. The cluster split was intermediate (RMSE 1.508, MAE 1.158, R² 0.451). In contrast, Lipophilicity exhibited weak point-prediction utility under every split, with negative mean R² values under random (−0.175), scaffold (−0.503), and cluster (−0.617) splitting. The persistence of nominal conformal coverage for this endpoint therefore cannot be interpreted as evidence of accurate point prediction.

Probability calibration also favored random splitting within each classification endpoint. For BBBP, the Brier score, negative log-likelihood, and probability ECE were 0.102, 0.439, and 0.075 under random splitting, compared with 0.120, 0.525, and 0.096 under scaffold splitting. ClinTox showed the same direction, with random-split values of 0.060, 0.323, and 0.051 versus scaffold-split values of 0.068, 0.379, and 0.057. Because Brier score and negative log-likelihood are affected by endpoint prevalence, the lower raw values observed for ClinTox were interpreted only within endpoint and not as evidence that ClinTox was globally better calibrated than BBBP.

Together, these results answer the first research question by showing that chemical shift generally weakened predictive performance and calibration, but the severity and even the practical meaning of the degradation depended on the endpoint. Random-split estimates were consistently optimistic relative to label-blind scaffold or cluster evaluation.

## 3.2 Chemical similarity was an endpoint- and split-dependent applicability-domain indicator

Continuous applicability-domain analyses showed that similarity to the training set was informative for some endpoints but not universally so (Figure 4; Table 3). For BBBP, similarity was negatively associated with both predictive risk and conformal miscoverage across all three split types. Mean Spearman correlations between similarity and predictive risk ranged from −0.131 to −0.159, while correlations with miscoverage ranged from −0.120 to −0.127. The corresponding out-of-domain scores discriminated higher-risk samples with mean AUC values of approximately 0.61–0.63. All eight frozen classification model/regime combinations supported the negative association across seeds.

Lipophilicity showed the strongest and most consistent applicability-domain signal. Similarity–risk correlations ranged from approximately −0.22 to −0.33, and similarity–miscoverage correlations ranged from approximately −0.19 to −0.24. Low-similarity compounds also showed consistently higher risk and miscoverage across thresholds from 0.40 to 0.70. Across these thresholds, the mean low-domain minus high-domain risk difference was approximately 0.37–0.54 and the miscoverage difference approximately 0.09–0.15, with all four frozen regression models supporting the direction of effect.

ClinTox provided a contrasting result. Similarity–risk correlations were weak (approximately −0.02 to −0.05), and similarity–miscoverage correlations were similarly small (approximately −0.04 to −0.06). Threshold-based contrasts were unstable and could change sign at higher thresholds. This weakness is consistent with the dominance of the negative class in ordinary error summaries and with the minority-retention results reported below.

ESOL demonstrated that applicability-domain behavior could reverse under a particular shift mechanism. Similarity was informative under random and cluster splitting, with mean risk correlations of approximately −0.16 and −0.15, respectively. Under scaffold splitting, however, the association reversed: the mean similarity–risk correlation was approximately +0.04, the similarity–miscoverage correlation approximately +0.08, and the miscoverage out-of-domain AUC fell below 0.5. At a Tanimoto threshold of 0.40, the low-similarity scaffold group even showed lower average risk than the high-similarity group. Scaffold novelty was therefore not a monotonic proxy for ESOL prediction difficulty.

Threshold sensitivity reinforced the continuous findings but also showed why no universal similarity cutoff should be proposed. At high thresholds, the low-domain fraction could encompass most of the test set, leaving a small and potentially unstable high-domain reference group. We therefore treated continuous associations as the primary applicability-domain evidence and threshold partitions as sensitivity analyses. Overall, chemical similarity was a robust trust indicator for BBBP and especially Lipophilicity, weak for ClinTox, and non-monotonic for ESOL under scaffold shift.

## 3.3 Marginal conformal coverage concealed severe class-conditional failure

At nominal 90% coverage, overall classification coverage was close to target for all three conformal variants, but these marginal summaries concealed large and endpoint-specific class imbalances (Figure 3; Table 4). For BBBP, marginal conformal prediction achieved overall coverage of 0.904–0.907 across splits. However, positive-class coverage was 0.971–0.978 while negative-class coverage was only 0.664–0.694, producing class-conditional coverage gaps of approximately 0.28–0.31. Density-ratio weighting changed these values only modestly: positive coverage remained 0.972–0.977 and negative coverage 0.674–0.699.

The failure was more severe for ClinTox. Marginal overall coverage remained 0.897–0.902, yet positive-class coverage was only 0.068 under scaffold splitting, 0.069 under cluster splitting, and 0.099 under random splitting. In contrast, negative-class coverage was 0.954–0.968. Density-ratio weighting also preserved apparently acceptable overall coverage (0.890–0.912) while positive-class coverage remained only 0.073–0.158. Thus, neither marginal conformal prediction nor covariate-shift weighting prevented near-complete undercoverage of the ClinTox minority class.

The efficiency summaries further showed that apparently reassuring marginal coverage was not achieved through broad, conservative prediction sets. ClinTox marginal prediction sets had mean sizes below one (0.950–0.971) because empty sets occurred for approximately 3.5–5.0% of samples, while ambiguous two-label sets were nearly absent. The principal problem was therefore not excessive ambiguity but asymmetric failure: the method often returned a precise-looking set that did not contain the positive label.

These findings answer the third research question by showing that marginal coverage can be statistically close to its nominal target while being practically misleading. Classification conformal evaluation must report class-conditional coverage, empty-set frequency, prediction-set size, and class support together with overall coverage.

## 3.4 Mondrian calibration repaired class coverage but substantially reduced informativeness

Mondrian conformal prediction greatly reduced class-conditional coverage gaps, but the repair was accompanied by substantially larger and more ambiguous prediction sets (Figure 3; paired results in Table 5). For ClinTox, Mondrian increased positive-class coverage by a descriptive mean of 0.841 under random splitting, 0.857 under cluster splitting, and 0.872 under scaffold splitting relative to marginal conformal prediction. All eight frozen model/regime combinations had model-specific 95% cross-seed confidence intervals entirely above zero for this positive-coverage effect.

The class-conditional coverage gap decreased by 0.770–0.828 across the three ClinTox split types, again with all eight model/regime combinations showing confidence intervals entirely below zero. The resulting positive-class coverage was approximately 0.926–0.940, while negative-class coverage remained approximately 0.895–0.900. This represented a statistically stable correction of the minority-class failure observed under marginal and shift-weighted calibration.

The practical cost was large. Mean prediction-set size increased by approximately 0.746–0.769, reaching 1.71–1.72 labels on average. Ambiguous two-label sets occurred for approximately 70.9–71.9% of test samples. In other words, Mondrian calibration restored class-conditional validity largely by returning both possible labels for most compounds.

BBBP showed a related but less extreme trade-off. Mondrian reduced the class-conditional coverage gap by approximately 0.241–0.250 across splits, with all eight model/regime combinations supporting the reduction. Because marginal BBBP calibration strongly over-covered the positive class, Mondrian positive coverage decreased by approximately 0.064–0.075 while negative coverage increased toward the nominal target. Mean set size increased by approximately 0.208–0.232, and ambiguous-set rate increased by approximately 0.207–0.232.

Density-ratio weighting did not provide a comparable repair. Across endpoints and splits, changes in minority coverage and class-conditional gap were small and rarely supported consistently across model families. For ClinTox, the descriptive positive-coverage increase ranged from 0.005 to 0.090, but no model-specific positive-coverage confidence interval was entirely above zero. Coverage repair was therefore not equivalent to practical reliability repair: Mondrian achieved the former robustly, but at a substantial loss of informativeness, whereas shift weighting produced no stable class-level correction.

## 3.5 Regression conformal variants provided no consistent universal advantage

All three regression conformal methods produced marginal empirical coverage near the nominal 0.90 level, but neither shift weighting nor adaptive normalization consistently improved the coverage–efficiency trade-off relative to standard marginal absolute-residual conformal prediction (Figure 6; Tables 6–7).

For ESOL, marginal coverage was 0.901 under random splitting, 0.891 under scaffold splitting, and 0.911 under cluster splitting. Shift-weighted coverage ranged from 0.893 to 0.907, while adaptive coverage ranged from 0.893 to 0.901. Paired coverage differences relative to marginal conformal prediction were small (−0.010 to +0.006), and no comparison was consistently supported across all four frozen regression models. Under scaffold splitting, shift weighting reduced the descriptive absolute coverage gap by 0.026, but only one of four models had a confidence interval entirely below zero.

For Lipophilicity, marginal coverage was 0.892–0.900, shift-weighted coverage 0.898–0.906, and adaptive coverage 0.891–0.900. Shift weighting increased coverage by 0.011 under cluster splitting and 0.006 under scaffold splitting, but these gains were accompanied by wider intervals: mean width increased by 0.234 and 0.174, respectively. Only one of four models consistently supported either width increase.

Adaptive normalization generated variable-width intervals, but the degree of adaptivity was modest. Interval-width coefficients of variation were only 0.007–0.013 for ESOL and approximately 0.032–0.033 for Lipophilicity. Width–error associations were also weak: mean Spearman correlations ranged from approximately 0.05 to 0.20, with the strongest value observed for ESOL under scaffold splitting. Adaptive intervals were typically slightly narrower than marginal intervals, but no endpoint–split combination showed a stable cross-model coverage–efficiency advantage.

These results support a restrained conclusion. Standard marginal conformal prediction was already highly competitive for the two regression endpoints. Covariate-shift weighting occasionally altered coverage or interval width, and adaptive normalization introduced some local heterogeneity, but neither method produced a general and reproducible reliability improvement across endpoints, split types, and model families.

## 3.6 Selective prediction could reduce ordinary risk by disproportionately removing the minority class

Selective prediction revealed that model confidence and chemical-domain distance ranked test samples in different ways (Figure 5; Table 8). For BBBP, confidence-based rejection improved ordinary-risk pAURC by 0.058–0.070 across splits and balanced-risk pAURC by 0.055–0.060. Similarity-based rejection produced smaller ordinary-risk gains (0.028–0.036) but somewhat larger balanced-risk gains (0.066–0.075). Confidence therefore ranked aggregate error more effectively, whereas chemical similarity better preserved class-balanced risk.

ClinTox showed why ordinary selective risk alone can be misleading. Confidence-based rejection improved ordinary-risk pAURC by 0.021–0.028, but balanced-risk pAURC worsened by 0.021–0.029. At 50% retained coverage, confidence preserved only approximately 26.9–30.6% of the positive-class examples. At 10% retained coverage, positive retention fell to approximately 5.4–7.8%. The apparent reduction in ordinary error was therefore achieved partly by removing the minority class at a faster rate than the overall sample population.

Similarity-based rejection retained substantially more ClinTox positives. At 50% retained coverage, positive retention was approximately 44.0–46.1%, and at 10% retained coverage it was approximately 10.6–17.4%. Nevertheless, balanced-risk performance remained split dependent: the balanced pAURC gain was −0.004 under cluster splitting, +0.043 under random splitting, and +0.012 under scaffold splitting. Chemical-domain rejection was therefore less destructive to the minority class than confidence-based rejection, but not universally effective.

Regression selective prediction also depended strongly on endpoint and split. For Lipophilicity, similarity-based rejection improved primary pAURC by 0.200–0.250 across all three splits and reduced risk at 50% retained coverage by approximately 0.245–0.321. For ESOL, the same score improved pAURC under random and cluster splitting (0.109 and 0.100, respectively) but worsened it under scaffold splitting (−0.047). At 50% retained coverage, ESOL scaffold risk was worse than matched random rejection by approximately 0.101.

These results demonstrate that selective prediction does not have a single endpoint-independent interpretation. Confidence and chemical similarity are complementary ranking signals. A low ordinary selective risk may reflect genuine error ranking, class-composition changes, or both. Selective-prediction studies in imbalanced ADMET tasks should therefore report balanced risk and class-specific retention over the full retained-coverage range, rather than relying on aggregate risk alone.

## 3.7 Integrated result

Across the four endpoints, no single metric or reliability method was sufficient. Random splitting generally yielded optimistic estimates; chemical similarity was a strong risk indicator for some endpoints but failed or reversed for others; marginal conformal coverage could conceal catastrophic minority-class undercoverage; Mondrian calibration repaired class validity while producing largely ambiguous sets; shift-weighted and adaptive regression methods offered no universal improvement; and confidence-based rejection could appear safer by preferentially removing positive cases. These findings jointly support a multidimensional definition of ADMET reliability that requires point performance, probability calibration, class-conditional validity, efficiency, applicability-domain diagnostics, and retention-aware selective risk to be assessed together.

---

# Figure captions

## Figure 1

**Confirmatory reliability-evaluation workflow.** Four ADMET endpoints spanning binary classification (BBP permeability [BBBP] and clinical toxicity [ClinTox]) and regression (aqueous solubility [ESOL] and Lipophilicity) were evaluated using frozen ECFP-based model families. Random, scaffold, and similarity-cluster partitions were generated without using outcome labels and were divided into training, calibration, and test sets. Confirmatory analyses used ten seeds for random and scaffold splitting and five seeds for cluster splitting. Method comparisons were paired within the same endpoint, split, model, and seed. The reliability modules addressed predictive performance and probability calibration (RQ1), applicability-domain robustness (RQ2), class-conditional and subgroup failure (RQ3), and coverage–efficiency–informativeness trade-offs among conformal variants (RQ4). Smoke-test seed 99 was excluded from all scientific conclusions.

## Figure 2

**Predictive performance and probability calibration under chemical distribution shift.** Descriptive means across the frozen model/regime combinations are shown for random, scaffold, and similarity-cluster splits. (A) ROC-AUC for BBBP and ClinTox. (B) PR-AUC. (C) Balanced accuracy; the dashed line marks 0.5. (D) Probability expected calibration error. (E) RMSE for ESOL and Lipophilicity. (F) MAE. (G) R²; the dashed line marks zero, and negative values indicate performance below the corresponding mean-prediction reference. (H) Negative log-likelihood for classification. Numeric labels report descriptive means. Model-specific cross-seed uncertainty and confidence intervals are reported in the supplementary tables; the bars must not be interpreted as independent model replicates.

## Figure 3

**Mondrian conformal prediction repairs class-conditional coverage at the cost of less informative prediction sets.** Positive- and negative-class empirical coverage at nominal 90% coverage is shown for marginal, density-ratio-weighted, and Mondrian label-conditional conformal prediction. (A–C) BBBP under random, scaffold, and cluster splitting. (D–F) ClinTox under the same split types. Dashed horizontal lines mark nominal coverage. (G) Mean prediction-set size and (H) ambiguous two-label set rate for all endpoint–split–method combinations. R, S, and C denote random, scaffold, and cluster splits. Values are descriptive means across frozen model/regime combinations. Paired model-specific cross-seed analyses showed stable Mondrian reductions in class-conditional coverage gap, but also substantial increases in prediction-set size and ambiguity.

## Figure 4

**Applicability-domain diagnostics are endpoint- and split-dependent.** (A) Spearman correlation between maximum Tanimoto similarity to the training set and predictive risk. (B) Spearman correlation between similarity and marginal conformal miscoverage. Negative values indicate higher risk or miscoverage at lower similarity; positive values indicate reversal of this relationship. (C) Sensitivity of the low-domain minus high-domain risk difference to the Tanimoto threshold for BBBP (solid lines) and Lipophilicity (dashed lines), with color indicating split type. (D) BBBP low-domain risk difference across thresholds. (E) ClinTox low-domain risk difference. (F) Lipophilicity low-domain minus high-domain miscoverage across thresholds. Continuous similarity analyses are the primary applicability-domain evidence; threshold partitions are presented as sensitivity analyses because high cutoffs can leave small high-domain reference groups.

## Figure 5

**Selective prediction can reduce ordinary risk while disproportionately removing the minority class.** (A) Matched partial-area-under-the-risk–coverage-curve (pAURC) improvement of confidence-based rejection over matched random rejection for ordinary error and balanced error. R, S, and C denote random, scaffold, and cluster splits. (B) ClinTox positive-class retention across retained coverage from 0.10 to 1.00. Colors indicate split type, solid lines indicate confidence-based rejection, dashed lines indicate chemical-similarity-based rejection, and the dotted diagonal represents proportional class retention. (C) Similarity-based risk improvement at 50% retained coverage for ESOL and Lipophilicity. (D) Primary pAURC improvement for similarity-based rejection across the two regression endpoints. pAURC was integrated over retained coverage 0.10–1.00 using matched finite coverage points for the selective and random curves. Positive improvement denotes lower risk than matched random rejection.

## Figure 6

**Regression conformal methods show no universal coverage–efficiency advantage.** Marginal absolute-residual, density-ratio-weighted, and split-calibration adaptive normalized conformal intervals were compared at nominal 90% coverage. (A) Empirical coverage; the dashed line marks 0.90. (B) Mean interval width. (C) Interval-width coefficient of variation, showing the degree of test-specific width heterogeneity. (D) Spearman association between interval width and realized absolute error for the variable-width methods. Marginal conformal intervals are constant-width within each run, so the width–error correlation is undefined and is marked as not applicable. Bars are descriptive means across frozen regression models; paired model-specific cross-seed analyses showed no universal coverage–efficiency advantage for shift weighting or adaptive normalization.

---

# Internal revision checklist before LaTeX integration

- Verify that every numeric value matches the final manuscript-assets CSV after any future rebuild.
- Replace provisional table numbers with final cross-references after the LaTeX structure is created.
- Keep “descriptive mean across frozen model/regime combinations” in the Results or caption wherever bars aggregate models.
- Do not add significance language to model-average bars.
- Keep ClinTox Brier/NLL comparisons within endpoint.
- Preserve negative Lipophilicity R² values and the ESOL scaffold applicability-domain reversal.
- Keep “pAURC over retained coverage 0.10–1.00,” not “full AURC.”
- Report sparse domain-conditioned ClinTox groups only in the supplementary material with explicit support counts.
