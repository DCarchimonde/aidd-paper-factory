# Paper 1 Claim and Reference Audit

Date: 2026-07-04

This document records the scientific story, claim boundaries, evidence sources, and reference requirements for Paper 1.

## 1. Core story

The paper should tell one clean story:

> Molecular property prediction benchmarks should not report model scores alone. The train/test split itself can change test difficulty, target distribution, chemical similarity, and model ranking. Therefore, split diagnostics should accompany model performance in AIDD benchmark reports.

## 2. Main contribution statement

Safe contribution wording:

> We present a reproducible split-diagnostic audit pipeline that jointly evaluates performance gaps, target distribution shift, scaffold statistics, test-to-train Tanimoto similarity, outlier cases, and model-ranking changes under random, ordinary scaffold, and target-balanced scaffold splitting protocols.

Do not claim:

- A new molecular predictor.
- A new state-of-the-art model.
- The first study of random versus scaffold splitting.
- A universal replacement for scaffold splitting.
- Complete coverage of all AIDD benchmark settings.

## 3. Claim-to-evidence map

| Claim | Evidence in this project | Required reference support | Status |
|---|---|---|---|
| MoleculeNet is a standard molecular ML benchmark | Dataset choice and framing | MoleculeNet | Supported |
| TDC emphasizes therapeutics ML datasets, meaningful splits, and robust generalization | Related work framing | TDC | Supported |
| Scaffold splitting is used to evaluate scaffold generalization | split implementation and discussion | Bemis-Murcko, MoleculeNet | Supported |
| Scaffold split is not always sufficient and can still overestimate performance | Similarity audit motivation | Scaffold split critique paper | Supported |
| Morgan fingerprints are standard molecular representations | Feature extraction pipeline | Rogers and Hahn ECFP paper | Supported |
| RDKit is used for SMILES handling and fingerprints | preprocessing scripts | RDKit software citation | Supported |
| scikit-learn and XGBoost are used for baselines | training scripts | scikit-learn, XGBoost citations | Supported |
| Random splits produce higher test-to-train similarity in this benchmark | Table 4 and Figure 3 | Project evidence, not external claim | Supported |
| Ordinary scaffold split can induce target shift | Table 2 and Figure 2 | Project evidence; literature context from split/distribution-shift references | Supported |
| Target-balanced scaffold reduces target shift in several datasets | Table 2 | Project evidence | Supported with limitation |
| Target-balanced scaffold is universally better | Not supported | None | Must not claim |
| Model ranking can change by split | Appendix ranking table | Project evidence | Supported |

## 4. Required citations currently in `references.bib`

- MoleculeNet: molecular ML benchmark.
- Therapeutics Data Commons: therapeutics ML tasks and evaluation.
- Bemis and Murcko: molecular frameworks/scaffolds.
- Rogers and Hahn: extended-connectivity fingerprints.
- Scaffold split critique paper: scaffold splits can overestimate virtual screening performance.
- ImDrug: imbalanced AIDD benchmark context.
- scikit-learn: classical ML implementation.
- XGBoost: gradient boosting implementation.
- RDKit: cheminformatics toolkit.

## 5. Claims that require careful wording

### Target-balanced scaffold split

Use:

> diagnostic counterfactual

Avoid:

> new benchmark standard
> better split
> fairer split
> universally superior split

### Generalization gap

Use:

> split-induced performance difference
> observed generalization gap under the tested protocol

Avoid:

> true real-world generalization gap
> prospective validation performance

### Dataset scope

Use:

> six public molecular property datasets

Avoid:

> all AIDD datasets
> comprehensive coverage of drug discovery

### Model scope

Use:

> lightweight fingerprint-based classical baselines

Avoid:

> state-of-the-art molecular AI models

## 6. Reviewer-risk checklist

Potential reviewer criticism and response strategy:

1. **Only classical ML baselines are used.**
   - Response: The study is an evaluation audit, not a SOTA model paper. Lightweight baselines isolate split effects and improve reproducibility.

2. **Target-balanced scaffold split is heuristic.**
   - Response: It is explicitly framed as a diagnostic counterfactual, not as a universal replacement for scaffold splitting.

3. **FreeSolv remains target-shifted after balancing.**
   - Response: This demonstrates an important small-dataset limitation caused by scaffold concentration.

4. **Only six datasets are used.**
   - Response: This is a focused audit across representative classification and regression datasets. Larger ADMET expansion can be reserved for Paper 2.

5. **No prospective/time split.**
   - Response: Prospective validation is acknowledged as future work; this paper focuses on diagnostic differences among common public benchmark split protocols.

## 7. Current go/no-go status

Status: **Go for manuscript polishing.**

The experimental package is sufficiently complete for a first manuscript draft. Further experiments should be added only if a target journal or reviewer requires them.
