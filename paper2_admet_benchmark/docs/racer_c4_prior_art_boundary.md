# RACER-C4/TAME prior-art boundary

Status: literature positioning for development; **not a legal novelty or
patent opinion**.

| Adjacent work | Prior contribution | C4 boundary |
|---|---|---|
| Weighted conformal under covariate shift | Importance-weighted calibration with a query mass at infinity | C4 reuses this construction and explicitly treats estimated weights as empirical |
| CoDrug | Molecular density-ratio weighting for drug-property conformal prediction | C4 does not claim the first shift-aware molecular conformal predictor |
| KMM-CP | Kernel mean matching and diagnostics for conformal prediction under covariate shift | C4 uses scalable cross-fitted domain ratios, not a claim to have invented KMM or transport weighting |
| Multi-distribution robust conformal prediction | Max-p/union constructions robust across candidate distributions | C4's baseline-containing envelope is a narrower operational molecular construction |
| SOCOP | Singleton/utility-optimized conformal prediction | C4 does not claim the first correct-singleton objective; its local utility candidate was rejected |
| SCoRE and post-hoc conformal selection | General risk control and valid selection after conformal calibration | C4 does not claim general selective-risk or post-selection validity |
| Batch conformal prediction | Joint use of unlabeled or label-vector batch structure | C4 uses the unlabeled target batch only for domain-ratio estimation and audits |

Primary sources checked for the C4 boundary:

- Tibshirani et al., *Conformal Prediction Under Covariate Shift*:
  https://arxiv.org/abs/1904.06019
- Laghuvarapu et al., *Conformal Drug Property Prediction with Density
  Estimation under Covariate Shift*: https://arxiv.org/abs/2310.12033
- *Kernel Mean Matching Conformal Prediction under Covariate Shift*:
  https://arxiv.org/abs/2603.26415
- *Multi-Distribution Conformal Prediction*:
  https://arxiv.org/abs/2601.02998
- *Singleton-Optimized Conformal Prediction*:
  https://arxiv.org/abs/2509.24095
- Bai and Jin, *Conformal Selective Prediction with General Risk Control*:
  https://arxiv.org/abs/2603.24704
- *Post-hoc Conformal Selection*: https://arxiv.org/abs/2604.11305
- Gazin et al., *Powerful Batch Conformal Prediction for Classification*:
  https://proceedings.mlr.press/v258/gazin25a.html

The narrow research question is whether the complete TAME mechanism—two
audited molecular transport views, a protected-label consensus augmentation,
no-new-singleton/full-set failure semantics, and a sealed external-label
promotion record—has useful independent coverage/efficiency behavior. Component
novelty is not asserted.
