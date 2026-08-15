# RACER-C3 prior-art boundary (development audit)

Status: literature-positioning aid, **not a novelty opinion**.

| Adjacent method | Prior module | RACER-C3 use or distinction | Prohibited claim |
|---|---|---|---|
| Mondrian CP | class-conditional quantiles | retained as the final coverage wrapper | invented class-wise coverage |
| CoDrug / weighted CP | density or likelihood-ratio weighting under covariate shift | screened and rejected as the primary mechanism | invented shift-aware CP |
| RC3P | augmented label-rank calibration for class-wise efficiency | required comparator; C3 instead routes candidate labels to distinct experts | first efficient class-wise CP |
| Kandinsky CP | overlapping `(X,Y)` groups and distribution-shift coverage | adjacent to candidate-label-dependent calibration; C3 makes no arbitrary-shift guarantee | first label-and-covariate conditional CP |
| COLA | score/set aggregation by alpha allocation, including individualized choices | required comparator; C3 chooses a whole candidate-expert matrix by a batch-symmetric frontier audit | first adaptive score aggregation |
| SOCOP / utility-directed CP | decision- or size-aware conformal set construction | screened; C3 uses MacroCSY as an evaluation/selection target | first utility-aware CP |
| SCRC | selective conformal risk control | final-state certificate comparator; C3 does not delete difficult rows | first selective risk control |
| SCoRE | selective prediction with general risk control | comparator for general risk/selection claims | first general conformal risk control |
| Powerful batch CP | combines p-values for joint label-vector coverage | C3 uses an unlabeled batch only to choose one permutation-invariant score route | first batch conformal classifier |
| learned/local nonconformity | learns score or conditional thresholds from features | candidate-correctness expert is one instance | first learned conformal score |

Primary sources checked during development:

- Shi et al., *Conformal Prediction for Class-wise Coverage via Augmented Label
  Rank Calibration*: https://arxiv.org/abs/2406.06818
- Laghuvarapu et al., *Conformal Drug Property Prediction with Density
  Estimation under Covariate Shift*: https://arxiv.org/abs/2310.12033
- Bairaktari et al., *Kandinsky Conformal Prediction*: https://proceedings.mlr.press/v267/bairaktari25a.html
- Xu et al., *Aggregating Conformal Prediction Sets via alpha-Allocation*:
  https://arxiv.org/abs/2511.12065
- Xu et al., *Selective Conformal Risk Control*:
  https://arxiv.org/abs/2512.12844
- Bai and Jin, *Conformal Selective Prediction with General Risk Control*:
  https://arxiv.org/abs/2603.24704
- Gazin et al., *Powerful Batch Conformal Prediction for Classification*:
  https://proceedings.mlr.press/v258/gazin25a.html
- Fisch et al., *Conformal Prediction Sets with Limited False Positives*:
  https://proceedings.mlr.press/v162/fisch22a.html

The defensible novelty question is narrow: whether a full-population conformal
classifier has previously combined (i) candidate-label-specific heterogeneous
experts, (ii) a permutation-invariant unlabeled chemical-frontier route, (iii)
an exact strong-score fallback, and (iv) a class-balanced correct-singleton
objective/certificate. Exact-formula, patent, code, and priority-date searches
are still required before submission.
