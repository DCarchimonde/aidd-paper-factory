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

## First-choice journal

### Journal of Computer-Aided Molecular Design

Decision: **First submission target**

Why it fits:

- SCIE-indexed according to the journal page.
- Hybrid journal with subscription route available.
- Subscription route has no APC.
- Scope includes chemoinformatics, bioinformatics, computational drug design, machine learning methods, structure-property relationships, and chemical database development/usage.
- Median submission-to-first-decision is currently short enough to support the high-output roadmap.
- Recent articles include computational drug design and machine-learning/QSAR-style topics.

Main risk:

- The journal states that predictive methods using traditional approaches should generally have compelling experimental evidence.
- Our paper must therefore be framed as a benchmark/evaluation methodology audit, not as a routine predictive model study.
- The manuscript should emphasize retrospective validation, reproducibility, split diagnostics, and benchmark interpretation.

Required framing before submission:

- Put “split-diagnostic audit” in title, abstract, and cover letter.
- Avoid “new predictor”, “SOTA”, or “drug discovery validation” claims.
- Emphasize chemoinformatics benchmark reliability and molecular generalization.

## Strong backup journal

### Chemometrics and Intelligent Laboratory Systems

Decision: **Backup 1 / possibly co-first if JCAMD fit becomes questionable**

Why it fits:

- Elsevier journal with subscription option and no publication fee for subscription publication.
- Scope includes development of statistical, mathematical, and computer techniques in chemistry and related disciplines.
- Scope also includes well-characterized datasets to test performance of new methods and software.
- The paper's split-diagnostic audit can be framed as a chemometric/computational evaluation methodology for molecular property prediction.

Main risk:

- The journal does not consider routine applications of established chemometric methods.
- The paper must be framed as an evaluation methodology/benchmark diagnostic contribution, not just applying RF/XGBoost/Ridge to public datasets.

Required framing before submission:

- Emphasize “diagnostic evaluation methodology for chemical ML benchmarks”.
- Highlight target shift, scaffold concentration, and Tanimoto similarity as audit dimensions.
- Avoid making the paper look like an application-only QSAR benchmark.

## Broad AI backup journal

### Applied Intelligence

Decision: **Backup 2, not first choice**

Why it may fit:

- SCIE-indexed according to the journal page.
- Hybrid journal with subscription route available.
- Broad AI scope includes intelligent systems, learning methodologies, classification/clustering, pattern recognition, data mining, bioinformatics, and real-life complex problems.

Main risk:

- The journal emphasizes new and innovative intelligent systems and technological developments rather than applying existing technologies to new datasets.
- Our current paper is stronger as a domain-specific benchmark audit than as a general AI method paper.

Use only if:

- We broaden the framing from AIDD-specific molecular benchmark to general AI benchmark evaluation under distribution shift.
- We strengthen the methodological novelty language without overclaiming.

## Journals to avoid for Paper 1 right now

- Fully open-access journals with mandatory APC, unless waiver or institutional coverage is guaranteed.
- Very high-bar AI journals requiring major algorithmic novelty.
- Pure biomedical informatics journals requiring direct clinical translation or healthcare professional involvement.
- AIDD journals that require experimental validation when the manuscript is only retrospective benchmark/audit.

## Final decision

Current target sequence:

1. Journal of Computer-Aided Molecular Design
2. Chemometrics and Intelligent Laboratory Systems
3. Applied Intelligence

The manuscript should now be formatted and cover-lettered for Journal of Computer-Aided Molecular Design first.

If JCAMD desk-rejects for lack of experimental validation or insufficient drug-design application, immediately retarget to Chemometrics and Intelligent Laboratory Systems with a revised title and cover letter emphasizing chemical data-analysis methodology.
