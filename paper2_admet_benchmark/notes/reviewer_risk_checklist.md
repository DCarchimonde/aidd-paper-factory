# Reviewer Risk Checklist - Paper 2

This checklist is used to prevent avoidable reviewer criticism before manuscript drafting.

## 1. Repetition risk with Paper 1

- [ ] The manuscript states that Paper 2 is about prediction reliability, not split effects alone.
- [ ] Split is presented as a stress-test setting, not the main contribution.
- [ ] Figures emphasize calibration, conformal empirical coverage, applicability domain, and selective prediction.
- [ ] The paper does not reuse Paper 1 wording as a simple template.

## 2. Data leakage risk

- [ ] Canonical SMILES are created before deduplication and splitting.
- [ ] Duplicate molecules are resolved before splitting.
- [ ] Train, calibration, and test sets are strictly separated.
- [ ] Test data are never used for model fitting, calibration, conformal threshold selection, threshold tuning, or AD-bin tuning.
- [ ] All split files are saved with endpoint, split setting, seed, and role columns.

## 3. Calibration risk

- [ ] Calibration methods are fitted only on the calibration split.
- [ ] Raw, sigmoid, and optionally isotonic calibration are clearly separated.
- [ ] ECE bin count is fixed in the main analysis and checked in sensitivity analysis.
- [ ] Reliability diagrams are not overinterpreted for tiny endpoints.

## 4. Conformal prediction risk

- [ ] The manuscript uses the phrase "empirical coverage" under shift.
- [ ] It does not claim theoretical conformal validity under scaffold or chemical distribution shift.
- [ ] The calibration split is used to estimate nonconformity thresholds.
- [ ] Classification prediction-set size and regression interval width are reported alongside coverage.

## 5. Endpoint risk

- [ ] Endpoints with too few clean samples are excluded or moved to supplement.
- [ ] Class imbalance is reported for every classification endpoint.
- [ ] Conflicting duplicate labels are handled and reported.
- [ ] Endpoint source, citation, and license/source notes are recorded.

## 6. Model risk

- [ ] The paper does not claim SOTA.
- [ ] Lightweight baselines are justified by reproducibility and reliability-audit goals.
- [ ] MLP on ECFP is included as a lightweight neural baseline.
- [ ] Fixed hyperparameters are stated clearly for the MVP study.

## 7. Applicability-domain risk

- [ ] At least max train Tanimoto similarity and unseen Murcko scaffold are reported.
- [ ] Domain-conditioned error and calibration are reported.
- [ ] Similarity bin thresholds are justified and checked in sensitivity analysis.

## 8. Statistical robustness risk

- [ ] At least five seeds are used where feasible.
- [ ] Mean and standard deviation are reported for primary metrics.
- [ ] Bootstrap confidence intervals are used for risk-coverage or error-reduction summaries where feasible.

## 9. Manuscript claim control

Allowed:

- Benchmark accuracy alone is insufficient for ADMET decision support.
- Reliability diagnostics reveal calibration and applicability-domain limitations.
- Selective prediction can reduce empirical error by abstaining from low-confidence predictions.

Forbidden:

- This model is state-of-the-art.
- Conformal prediction is guaranteed under chemical distribution shift.
- The predictions are experimentally validated.
- The method is ready for clinical or wet-lab deployment.
