# Paper 2 Story

## Working title

**Beyond Accuracy in ADMET Prediction: Applicability-Domain Diagnostics and Conformal Calibration under Chemical Distribution Shift**

## Core question

When should ADMET model predictions be trusted under chemical distribution shift?

## One-paragraph story

ADMET prediction models are increasingly used to prioritize candidate molecules in AI-driven drug discovery. However, conventional benchmark reports often focus on point-prediction accuracy, such as ROC-AUC or RMSE, without asking whether individual predictions are calibrated, reliable, and actionable. This study evaluates public ADMET and molecular property endpoints under random and scaffold validation settings using train/calibration/test separation. We quantify model reliability through probability calibration, conformal empirical coverage, applicability-domain diagnostics, and selective prediction. The goal is not to claim state-of-the-art ADMET prediction, but to provide a reproducible reliability audit for AIDD decision support.

## Boundary from Paper 1

Paper 1:

- split-diagnostic evaluation;
- molecular property benchmark interpretation;
- focus on split effects, target shift, scaffold statistics, Tanimoto similarity, generalization gap, and model-ranking stability.

Paper 2:

- reliability-oriented ADMET evaluation;
- focus on calibration, conformal empirical coverage, applicability domain, and selective prediction;
- split is used as a stress-test setting, not as the main contribution.

## Claims we can make

- Benchmark accuracy alone is insufficient for ADMET decision support.
- Reliability metrics provide additional information beyond ROC-AUC, PR-AUC, RMSE, and MAE.
- Calibration and conformal diagnostics can reveal when models become overconfident or uncertain.
- Applicability-domain measures help identify chemically distant or unseen-scaffold molecules with elevated prediction risk.
- Selective prediction can reduce error by abstaining from low-confidence predictions.

## Claims we must avoid

- Do not claim state-of-the-art ADMET prediction.
- Do not claim conformal prediction has guaranteed validity under scaffold or chemical distribution shift.
- Do not claim wet-lab validation.
- Do not imply that uncertainty diagnostics replace experimental validation.
- Do not present Paper 2 as a simple extension of Paper 1.

## Manuscript positioning

This paper should be written as a reliability and decision-support study for computational drug discovery, not as a raw benchmark leaderboard.
