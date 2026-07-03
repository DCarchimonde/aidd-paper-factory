# Paper 1 Submission Readiness Checklist

This checklist is used to keep the manuscript scientifically clean before journal submission.

## 1. Story check

The manuscript should tell one story:

> Molecular property prediction benchmarks should report split diagnostics together with model scores, because train/test splitting can change test difficulty, target distribution, chemical similarity, and model ranking.

The manuscript should not drift into a model-performance or state-of-the-art claim.

## 2. Claim discipline

Before submission, verify that the manuscript does not claim:

- a new molecular predictor;
- a new state-of-the-art result;
- the first study of random versus scaffold splitting;
- a universal replacement for scaffold splitting;
- complete coverage of all AIDD datasets;
- prospective or real-world deployment performance.

## 3. Required manuscript sections

The LaTeX paper should contain:

- Title
- Abstract
- Keywords
- Introduction
- Related Work
- Materials and Methods
- Results
- Discussion
- Limitations
- Conclusion
- Data and Code Availability
- Competing Interests
- Author Contributions
- Funding
- Acknowledgements
- Appendix
- References

## 4. Reference audit

Every external factual claim needs a real citation.

Current required citation families:

- MoleculeNet for molecular ML benchmarking.
- TDC for therapeutics ML tasks and benchmark culture.
- Bemis--Murcko for molecular frameworks/scaffolds.
- Rogers and Hahn for ECFP/Morgan fingerprints.
- Scaffold split critique for why scaffold split alone is insufficient.
- ImDrug for imbalance issues in AIDD datasets.
- RDKit for cheminformatics toolkit.
- scikit-learn for classical ML baselines.
- XGBoost for gradient boosting baseline.

## 5. Evidence audit

Results in the manuscript must be traceable to generated tables:

- Dataset description: `manuscript_table1_dataset_summary.csv`
- Split diagnostics: `manuscript_table2_split_diagnostics.csv`
- Generalization gap: `manuscript_table3_generalization_gap.csv`
- Similarity audit: `manuscript_table4_similarity_audit.csv`
- Outliers: `appendix_table_outlier_cases.csv`
- Ranking stability: `appendix_table_model_rankings.csv`

## 6. Figure audit

Required figures:

- Figure 1: ordinary versus balanced scaffold gap.
- Figure 2: target shift by split strategy, optional for main text or appendix.
- Figure 3: test-to-train Tanimoto similarity.

Check before submission:

- Captions explain what positive/negative values mean.
- Captions do not overclaim.
- Axes are readable.
- Figures compile correctly from relative paths.

## 7. Known limitations to keep

Do not remove these limitations:

- Classical fingerprint-based models only.
- No GNN or transformer molecular encoders.
- Target-balanced scaffold split is heuristic and diagnostic.
- Six public datasets only.
- No prospective or time-based validation.
- FreeSolv is constrained by small size and scaffold concentration.

## 8. Before choosing a journal

Decide:

- target journal scope;
- whether APC is acceptable;
- manuscript type: original research, benchmark paper, short communication, or methods/resource article;
- required word limit;
- reference style;
- figure/table limits;
- data/code availability policy.

## 9. Current status

Status: manuscript draft is ready for language polishing and journal targeting.

Do not add more experiments unless a journal requirement or reviewer-risk audit specifically demands them.
