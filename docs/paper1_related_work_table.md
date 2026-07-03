# Paper 1 Related-Work Positioning Table

This table is a working related-work map for Paper 1. It should be expanded during formal manuscript writing.

| Work / Line | Main focus | What is already covered by prior work | Relevance to Paper 1 | What Paper 1 should not claim | Potential distinction of Paper 1 |
|---|---|---|---|---|---|
| MoleculeNet | Molecular ML benchmark | Public molecular datasets, common metrics, baseline comparisons, benchmark framing | Establishes that molecular property prediction benchmark papers already exist | Do not claim to create the first molecular property benchmark | Paper 1 focuses on split diagnostics and target-shift audit rather than benchmark collection alone |
| Therapeutics Data Commons | Therapeutics ML task platform | Standardized therapeutic datasets, meaningful splits, systematic evaluation, robust generalization challenges | Shows that TDC already frames distribution shift and meaningful evaluation as central problems | Do not claim that split-aware evaluation is entirely new | Paper 1 is narrower: molecular property split diagnostics across random, scaffold, and target-balanced scaffold protocols |
| Random vs scaffold split literature | Chemical generalization evaluation | Random split can be optimistic; scaffold split is commonly used for chemical scaffold generalization | Confirms that ordinary random-vs-scaffold comparison alone is not novel | Do not claim first random-vs-scaffold comparison | Paper 1 adds target distribution shift diagnosis and balanced scaffold counterfactual |
| Scaffold split critique papers | Realism of scaffold splits | Scaffold split can still overestimate performance because different scaffolds may remain chemically similar | Motivates the need for Tanimoto similarity audit and more careful split interpretation | Do not claim scaffold split is a perfect proxy for real prospective validation | Paper 1 explicitly audits both target shift and chemical similarity patterns |
| Imbalanced AIDD benchmarks | Data imbalance in drug discovery | Imbalanced drug datasets affect fairness, robustness, and generalization; accuracy can be misleading | Motivates careful interpretation of ClinTox and use of ROC-AUC/AP/F1 rather than accuracy only | Do not claim class imbalance issue is new | Paper 1 connects imbalance with split-induced target distribution shift |
| This project | Split-diagnostic audit | Not a new model and not a new dataset platform | Quantifies performance gap, target shift, and similarity shift under multiple split protocols | Do not overclaim universality or SOTA | A reproducible audit workflow reporting performance, target-shift, scaffold-statistics, and similarity diagnostics together |

## Safe novelty wording

A cautious novelty sentence:

> Unlike standard benchmark reports that primarily compare model performance under predefined splits, this study treats the split itself as an object of diagnosis. We jointly quantify performance gaps, target distribution shifts, scaffold statistics, and test-to-train chemical similarity under random, ordinary scaffold, and target-balanced scaffold protocols.

## Unsafe novelty wording

Avoid:

- "first study of random and scaffold splits"
- "first leakage-aware benchmark"
- "new state-of-the-art molecular predictor"
- "target-balanced scaffold split is universally superior"
- "ordinary scaffold split is invalid"

## Literature-search checklist before submission

Before submitting, search and document:

- random scaffold split molecular property prediction
- scaffold split distribution shift molecular machine learning
- target distribution shift scaffold split
- balanced scaffold split molecular property prediction
- Tanimoto similarity train test split molecular benchmark
- MoleculeNet split benchmark
- TDC molecule split benchmark
- ImDrug imbalanced AI drug discovery
