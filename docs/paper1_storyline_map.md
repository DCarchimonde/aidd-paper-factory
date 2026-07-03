# Paper 1 Storyline Map

This document defines the narrative arc for Paper 1. It is stricter than a normal outline: every section must serve the same story.

## One-sentence story

Molecular property prediction benchmarks are not only tests of models; they are also tests of the train/test split, and split diagnostics are necessary to interpret whether reported performance reflects chemical generalization, target distribution shift, chemical similarity leakage, or model-ranking instability.

## Story in five moves

### Move 1: The field relies on benchmark scores

AIDD researchers commonly compare molecular property prediction models using public benchmark datasets. These scores influence which models appear strong and which modeling choices seem promising.

**Reader expectation:** benchmark scores should be comparable and meaningful.

### Move 2: But the split can change what the score means

If the test set is randomly sampled, chemically similar molecules may appear in both train and test. If the test set is scaffold-separated, the model may face new scaffolds, but the split can also change the target distribution.

**Problem:** a model score alone cannot tell us which of these effects is happening.

### Move 3: Ordinary scaffold split solves one problem but can create another

Ordinary scaffold splitting removes scaffold overlap, but our diagnostics show that it can produce severe target shifts and scaffold concentration. This means a performance drop under scaffold splitting is not automatically pure evidence of failed chemical generalization.

**Key nuance:** scaffold split is useful, but not self-explanatory.

### Move 4: A split-diagnostic audit makes the score interpretable

We compare random, ordinary scaffold, and target-balanced scaffold splits and report multiple diagnostics: performance gap, target shift, scaffold statistics, test-to-train Tanimoto similarity, outlier cases, and ranking stability.

**Contribution:** not a new predictor; a reproducible audit framework for interpreting benchmark results.

### Move 5: The conclusion is a reporting recommendation

Benchmark papers should report split diagnostics alongside model performance. This prevents overclaiming from random splits, prevents misreading scaffold gaps as purely chemical generalization failures, and helps identify when rankings depend on split artifacts.

**Take-home message:** score + split diagnostics > score alone.

## Section-by-section story role

| Section | Story role | Must do | Must avoid |
|---|---|---|---|
| Abstract | Compress the full story | Problem -> audit -> findings -> recommendation | Sound like a method/SOTA paper |
| Introduction paragraph 1 | Establish benchmark importance | Explain why molecular property prediction benchmarks matter | Overstate drug discovery impact |
| Introduction paragraph 2 | Introduce split problem | Explain random and scaffold split risks | Claim scaffold split is bad |
| Introduction paragraph 3 | Define the gap | Explain why split needs diagnosis | Claim target-balanced scaffold is a new standard |
| Contributions paragraph | State what we actually contribute | Audit framework and evidence dimensions | Claim novelty too broadly |
| Related Work | Position without overclaiming | Cite MoleculeNet, TDC, scaffold critique, imbalance | Pretend no one studied splits before |
| Methods | Make reproduction easy | Datasets, features, splits, models, metrics | Add unnecessary model hype |
| Results 1 | Show scaffold split is clean but shifted | Scaffold overlap = 0; target gaps can be large | Say scaffold split is invalid |
| Results 2 | Show performance depends on split | Gap table and gap figure | Say every model behaves the same |
| Results 3 | Show chemical similarity explanation | Tanimoto table and figure | Treat similarity as the only explanation |
| Results 4 | Synthesize diagnostics | Explain why score alone is insufficient | Overinterpret FreeSolv/HIV outliers |
| Discussion | Turn results into recommendation | Report split diagnostics | Claim real-world prospective validation |
| Limitations | Protect rigor | Classical models, six datasets, no time split | Hide weaknesses |
| Conclusion | One clean final message | Split diagnostics should accompany model scores | Introduce new claims |

## Current manuscript diagnosis

The current LaTeX draft already has the correct skeleton and scientific caution. It has a story, but the story is still more like a technically correct draft than a polished journal narrative.

What is already good:

- It clearly avoids claiming a new molecular predictor.
- It defines target-balanced scaffold as a diagnostic counterfactual.
- It connects performance gaps, target shift, scaffold statistics, and similarity.
- It keeps FreeSolv, HIV, ESOL Ridge, and ClinTox as limitations/outliers instead of hiding them.

What still needs polishing:

- The Abstract needs a sharper final sentence.
- The Introduction needs stronger paragraph transitions.
- Results should contain fewer isolated numbers and more interpretation.
- Discussion should more explicitly state the reporting recommendation.
- Related Work should be expanded after selecting a target journal.

## Polished storyline paragraph for Introduction

A stronger version of the central narrative is:

> In molecular property prediction, a benchmark score is often interpreted as evidence of model generalization. However, the score is inseparable from the train/test split that produced it. Random splitting can make the test set chemically close to the training set, while scaffold splitting can remove scaffold overlap but simultaneously alter the target distribution. Therefore, a performance gap between random and scaffold splits is not self-explanatory: it may reflect chemical scaffold generalization, target distribution shift, residual chemical similarity, or their interaction. This study addresses that interpretability problem by treating the split itself as an object of diagnosis.

## Polished take-home paragraph for Discussion

A stronger version of the main discussion message is:

> The central implication is that molecular property prediction benchmarks should move from score-only reporting to score-plus-diagnostics reporting. Random split performance should be accompanied by chemical similarity diagnostics, scaffold split performance should be accompanied by target-shift and scaffold-concentration diagnostics, and model comparisons should report whether model rankings remain stable across split protocols. Such reporting does not eliminate the difficulty of evaluating molecular generalization, but it makes the assumptions and failure modes of a benchmark visible.

## Go-forward writing rule

Every future edit should pass this test:

> Does this sentence help the reader understand why split diagnostics are needed to interpret molecular property prediction benchmarks?

If not, remove or move it to the appendix.
