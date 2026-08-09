# RACER-C4 independent EPA validation protocol

Version: **1.0 candidate frozen before final EPA label access**

## Data roles and hashes

The executable lock is
`configs/racer_c4/prospective_lock_v1.yaml`. It contains official NCATS URLs,
archive/member hashes, endpoint order, seeds, models, audits, gates, and final
failure behavior.

| Cohort | Size | Labels | Role |
|---|---:|---|---|
| Tox21 10K training | endpoint-specific after cleaning | open | fit, router, and internal conformal roles |
| Public leaderboard batch | 296 structures | open | architecture development only |
| Final EPA batch | 647 structures | sealed | single independent evaluation |

The six primary endpoints were selected by a label-count rule—at least 20
positive labels in the public development batch—not by C4 performance:
NR-AhR, NR-ER, SR-ARE, SR-ATAD5, SR-MMP, and SR-p53. All six remaining official
endpoints are retained as secondary small-sample stress tests.

Standardized-structure overlap with the 10K training cohort is recorded and
excluded from both external-domain fitting and evaluation. No endpoint, seed,
or structure may be removed because of an unfavorable model result.
An external structure that cannot pass the frozen RDKit standardizer retains
its public sample identity, receives `{0,1}` for every method, and is excluded
from domain fitting and endpoint metrics with an explicit reason code.

## Development stage

Run all six primary endpoints at seeds 101--105. The promotion denominator is
30 endpoint-seed cells. Comparators are ordinary Mondrian global stacking,
ECFP-weighted Mondrian, score-view-weighted Mondrian, and ordinary Mondrian
equal-logit pooling.

The two C4 transport views are a nine-descriptor physicochemical view and a
five-logit score view. High-dimensional ECFP weighting remains a diagnostic
comparator after failing its transport certificate; it is not allowed into the
envelope. If both approved views are not active, transport augmentation is
disabled and the ordinary baseline is returned, with baseline empty sets
converted to non-actionable full sets.

Promotion requires all of the following:

1. 30/30 primary C4 cells;
2. zero violations of ordinary-set inclusion;
3. zero C4 empty sets;
4. zero newly created singletons;
5. nonnegative mean coverage delta for both classes;
6. nonpositive mean wrong-singleton-exposure delta for both classes;
7. mean MacroCSY delta at least -0.05; and
8. both transport views active in at least 80% of primary cells.

Any failure stops before final-label download or parsing. The failed gate and
all unfavorable cells remain in the output.

## Prediction-to-label firewall

After a passed development gate:

1. train fresh prospective seeds 211--215;
2. read final structures only;
3. build every comparator and C4 prediction;
4. write `sealed_final_predictions.csv` and the transport-audit table;
5. hash both files and the executable lock;
6. write `promotion_record.json` with `final_labels_opened=false`;
7. only then acquire the locked final-label bytes;
8. verify their frozen SHA256;
9. parse and identity-join labels; and
10. evaluate once.

The parser has no callable path around the promotion record. Tests verify the
source-code ordering and exercise missing/invalid promotion records.

## Estimands and inference

Primary estimand: endpoint-seed-equal mean change in minimum class coverage for
C4 versus ordinary Mondrian on the six primary endpoints.

The 95% interval uses a frozen hierarchical bootstrap: resample endpoints,
then resample labeled final compounds within each endpoint and true class
(preserving endpoint class counts), retain all five prospective seeds, and run
2,000 draws with seed 44021.

Secondary descriptive estimands are MacroCSY delta, class-specific coverage,
wrong-singleton exposure, ambiguity, and empty-set rate. The hierarchy is
primary first, then descriptive secondary results; no multiplicity-adjusted
superiority claim is authorized automatically.

Estimated density ratios do not justify an exact arbitrary-shift coverage
claim. A positive interpretation requires a positive primary point estimate
and MacroCSY non-inferiority within five percentage points. Otherwise the result
is reported as negative or mixed without retuning.

## Reproduction

After pulling the published branch, the Windows entrypoint performs tests,
locked acquisition, deterministic cleaning, development gating, final
prediction sealing, final evaluation, hashing, logging, safe retries, and
process-scoped keep-awake behavior:

```powershell
powershell -ExecutionPolicy Bypass -File paper2_admet_benchmark\scripts\racer_c4\run_racer_c4_overnight.ps1
```

The entrypoint first requires the frozen RDKit runtime `2026.03.4`. A different
or missing RDKit build is repaired with the exact `rdkit==2026.3.4` binary wheel
using `--no-deps`, followed by a fresh runtime-version check before any cleaning
or model execution. This preprocessing is part of the RACER-C4 from-source
reproduction and does not rerun or overwrite the historical RACER-C v1 or
seed-99 experiments.
