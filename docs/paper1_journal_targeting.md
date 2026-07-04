# Paper 1 Journal Targeting Decision

Date: 2026-07-04

## 1. Current manuscript identity

Paper 1 is not a new-model paper. It is an evaluation / benchmark audit paper.

Core story:

> Molecular property prediction benchmarks should report split diagnostics together with model scores, because train/test splitting can change test difficulty, target distribution, chemical similarity, and model ranking.

Current paper type:

- original research article;
- benchmark / evaluation audit;
- cheminformatics + AIDD + molecular machine learning;
- no new deep model;
- no docking / wet-lab validation;
- classical fingerprint-based baselines used deliberately for reproducibility.

## 2. Constraints

Preferred constraints:

1. Strong scope match.
2. No mandatory APC if possible.
3. Reasonable acceptance probability.
4. Does not require SOTA deep learning or wet-lab validation.
5. Can accept a methods/evaluation/benchmark-style paper.

## 3. Recommended primary target

### Primary target: Computational Biology and Chemistry

Publisher: Elsevier

Why it fits:

- Broad computational life sciences scope.
- Scope includes bioinformatics, computational pharmacology, chemical-biology-specific modeling theory/practice, and innovative AI methodologies for biological data and machine-learning models.
- Full-length original articles are allowed.
- The paper can be positioned as a computational pharmacology / AIDD benchmark evaluation audit.
- More realistic than high-bar cheminformatics journals for a first submission.

Required positioning for this journal:

- Emphasize computational life sciences and computational pharmacology.
- Avoid making it look like a narrow dataset-cleaning report.
- Make the contribution methodological/evaluation-oriented: split-diagnostic audit for molecular property prediction benchmarks.
- Keep the manuscript as a full-length original research article.

Manuscript adjustments needed:

- Abstract should stay within about 250 words.
- References should be expanded toward 30--50 references.
- Add Highlights if required by Elsevier submission system.
- Add declaration of competing interests and funding statement.
- Add data/code availability statement.
- Add declaration of generative AI use if required by the submission system.

## 4. Stretch target

### Stretch target: Journal of Cheminformatics

Why it fits:

- Strong cheminformatics scope.
- The topic is directly about molecular ML benchmark interpretation and chemical similarity diagnostics.

Why it is risky:

- Fully open access with APC.
- Higher bar for novelty and cheminformatics depth.
- The current paper may need stronger chemical interpretation, more references, and possibly deeper external comparison.

Use this only if:

- APC/funding is acceptable or waived;
- we expand related work;
- we make the chemical-similarity audit and scaffold analysis more central.

## 5. Backup targets to verify later

Potential backups:

1. Molecular Informatics
   - Strong topical fit for cheminformatics and QSAR.
   - Need to verify current author charges and article type requirements before choosing.

2. SAR and QSAR in Environmental Research
   - Possible fit if framed around QSAR, toxicity, and environmental/chemical property prediction.
   - Less ideal because this paper is more general AIDD benchmark audit than environmental QSAR.

3. Journal of Computational Biology
   - Indexed and computational, but less chemically focused.
   - Backup only if computational benchmark angle is strengthened.

## 6. Decision

Current decision:

> Target Computational Biology and Chemistry first.

Rationale:

- Best balance of scope fit, cost risk, acceptance probability, and current manuscript strength.
- It allows a computational life-science / machine-learning evaluation story.
- It is less APC-risky than fully open-access journals.
- It does not force the paper to pretend to be a new SOTA molecular predictor.

## 7. Next manuscript actions

Before submission to Computational Biology and Chemistry:

1. Prepare Elsevier-style highlights.
2. Keep abstract under 250 words.
3. Expand references to around 30--50.
4. Add more recent split/generalization/bias references.
5. Add code availability and data availability statements.
6. Add declaration of competing interests.
7. Add funding statement.
8. Prepare cover letter emphasizing split-diagnostic audit, not SOTA modeling.
9. Consider adding Figure 2 target-shift plot into the main manuscript or supplementary material.

## 8. Cover letter angle

Suggested cover letter angle:

> This manuscript presents a reproducible diagnostic audit of split-induced generalization gaps in molecular property prediction benchmarks. Rather than proposing a new molecular predictor, it addresses a practical evaluation problem in computational pharmacology and AIDD: benchmark scores can be strongly affected by train/test split design. The study shows that random, ordinary scaffold, and target-balanced scaffold splits produce different target distributions, chemical similarity structures, model performance gaps, and model rankings. The work provides a transparent diagnostic framework that can improve the interpretation of molecular machine-learning benchmarks.
