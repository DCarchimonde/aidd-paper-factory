# Paper 1 Target Journal Strategy

Date: 2026-07-04

## Roadmap constraint

The long-term roadmap remains:

- build an AIDD/AI publication track before PhD enrollment;
- prioritize SCI/SCIE-indexed journals;
- avoid mandatory APC whenever possible;
- use hybrid/subscription routes when available;
- prioritize acceptance probability and publication momentum over prestige;
- keep Paper 1 aligned with AIDD/AI, but the journal itself does not have to be an AIDD-only journal.

## Paper 1 identity

Paper 1 is not a new predictor or SOTA model paper.

Paper 1 is:

> a split-diagnostic benchmark/audit paper for molecular property prediction.

Core story:

> Molecular property prediction benchmarks should report split diagnostics together with model scores, because train/test splitting can change test difficulty, target distribution, chemical similarity, and model ranking.

## Updated journal conclusion

The first-pass target should not be locked to a single AIDD journal. Paper 1 should use a **three-journal main target pool**:

1. Journal of Molecular Graphics and Modelling
2. Chemometrics and Intelligent Laboratory Systems
3. Journal of Computer-Aided Molecular Design

The reason is that Paper 1 is an evaluation/benchmark-diagnostic paper, not a drug-design validation paper. Journals that welcome computational molecular modelling, structure-property relationships, database mining, chemometrics, and reproducible computational insight may be more suitable than journals that strongly prefer experimental validation of predictive drug-discovery models.

## Candidate 1: Journal of Molecular Graphics and Modelling

Decision: **New first-choice candidate / strongest balance of fit, speed, and no mandatory APC route**

Why it fits:

- Elsevier journal with subscription publication option and no publication fee for subscription publication.
- Scope includes computer use in theoretical investigations of molecular structure, function, interaction, and design.
- Scope includes molecular modelling, computational chemistry, drug design, structure-activity and structure-property relationships, database mining, and compound library design.
- Fast journal insights are compatible with the high-output roadmap.
- The journal emphasizes conclusions and implications, which matches our split-diagnostic story.

Main risk:

- Routine applications of standard modelling approaches are not considered.
- Our paper must be framed as producing new benchmark interpretation and split-diagnostic insight, not as a routine QSAR/model comparison paper.

Required framing:

- Title should emphasize “split-diagnostic audit” and “molecular property prediction benchmarks”.
- Cover letter should say the contribution is benchmark reliability and molecular generalization interpretation.
- Avoid presenting RF/XGBoost/Ridge as the contribution.

## Candidate 2: Chemometrics and Intelligent Laboratory Systems

Decision: **Co-first or very strong backup**

Why it fits:

- Elsevier journal with subscription publication option and no publication fee for subscription publication.
- Scope includes development of statistical, mathematical, and computer techniques in chemistry and related disciplines.
- Scope includes well-characterized datasets to test performance of new methods and software.
- The paper can be framed as a chemometric/computational evaluation methodology for molecular ML benchmarks.

Main risk:

- Routine applications of established chemometric methods are not considered.
- Our paper must emphasize diagnostic methodology, not just application of existing ML models.

Required framing:

- Use “diagnostic evaluation methodology for chemical machine-learning benchmarks”.
- Highlight target shift, scaffold concentration, Tanimoto similarity, and ranking stability as audit dimensions.
- Avoid making the paper look like an application-only QSAR benchmark.

## Candidate 3: Journal of Computer-Aided Molecular Design

Decision: **Still suitable, but no longer automatically first**

Why it fits:

- SCIE-indexed according to the journal page.
- Hybrid journal.
- Scope includes chemoinformatics, bioinformatics, computational drug design, machine learning methods, structure-property relationships, and chemical database development/usage.
- Recent articles include computational drug design and machine-learning/QSAR-style topics.

Main risk:

- The journal states that predictive methods using traditional approaches should generally have compelling experimental evidence.
- Paper 1 has retrospective benchmark/audit evidence, not experimental validation.
- It may desk-reject if the editor interprets the paper as a traditional predictive method paper rather than a benchmark audit.

Required framing:

- Emphasize “benchmark audit”, “split diagnostics”, and “retrospective validation”.
- Avoid “new predictor”, “SOTA”, or direct drug-discovery efficacy claims.

## Candidate 4: Computational Biology and Chemistry

Decision: **Practical backup**

Why it fits:

- Elsevier journal with subscription publication option and no publication fee for subscription publication.
- Scope includes computational pharmacology, chemical-biology-specific modelling, bioinformatics, and innovative AI methodologies for biological data and machine-learning models.
- The journal insights show fast first decision and acceptance timelines compatible with the high-output route.

Main risk:

- It may expect more biological insight or validation.
- A purely molecular benchmark audit may feel less central unless framed as computational pharmacology / molecular ML benchmark reliability.

Required framing:

- Emphasize computational pharmacology and AI benchmark reliability.
- Avoid presenting it as a chemistry-only QSAR split comparison.

## Candidate 5: Molecular Informatics

Decision: **Topic-fit candidate, but needs manual verification before being placed first**

Why it fits:

- Strong topical fit: cheminformatics, QSAR, molecular informatics.
- The journal identity matches molecular ML and structure-property prediction well.

Main uncertainty:

- Current official author-fee and speed information needs to be manually verified from the Wiley page before using it as the first target.
- If it has a no-mandatory-APC route and reasonable first-decision speed, it may move into the top three.

## Broad AI backup: Applied Intelligence

Decision: **Do not use as first target for Paper 1**

Why it may fit:

- Broad AI scope includes intelligent systems, learning methodologies, classification/clustering, pattern recognition, data mining, bioinformatics, and real-life complex problems.
- Hybrid/subscription route may avoid mandatory APC.

Main risk:

- The paper is stronger as a domain-specific molecular benchmark audit than as a general AI method paper.
- Applied Intelligence may expect stronger algorithmic novelty.

Use only if:

- The manuscript is reframed toward general AI benchmark reliability under distribution shift.

## Journals to avoid for Paper 1 right now

- Fully open-access journals with mandatory APC, unless waiver or institutional coverage is guaranteed.
- Very high-bar AI journals requiring major algorithmic novelty.
- Pure biomedical informatics journals requiring direct clinical translation or healthcare professional involvement.
- AIDD journals that require experimental validation when the manuscript is only retrospective benchmark/audit.

## Updated final target sequence

Current target sequence:

1. Journal of Molecular Graphics and Modelling
2. Chemometrics and Intelligent Laboratory Systems
3. Journal of Computer-Aided Molecular Design
4. Computational Biology and Chemistry
5. Molecular Informatics, pending manual fee/speed verification
6. Applied Intelligence

## Immediate action

The next manuscript version should be formatted first for **Journal of Molecular Graphics and Modelling**.

If JMGM desk-rejects as not sufficiently molecular-modelling focused, immediately retarget to **Chemometrics and Intelligent Laboratory Systems** with stronger chemometric/evaluation-methodology framing.

If CILS desk-rejects as too molecular-ML specific or not sufficiently chemometric, retarget to **Journal of Computer-Aided Molecular Design** with stronger chemoinformatics/computer-aided molecular design framing.
