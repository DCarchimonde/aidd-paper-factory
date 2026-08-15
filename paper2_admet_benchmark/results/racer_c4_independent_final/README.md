# RACER-C4 independent final EPA result

Status: **complete independent evaluation and deterministic inference repair; no automatic superiority claim**

The executable candidate was frozen at Git commit
`4b117fefa648d082ee0d2fada4188a7d90642115`. All final predictions were written
and hashed before the locked EPA label file was acquired or parsed.

Across the six primary endpoints and five fresh seeds (211--215), RACER-C4/TAME
improved the endpoint-seed-equal minimum-class-coverage estimand by 0.0136487.
The frozen stratified hierarchical bootstrap 95% interval was
`[0.0058271, 0.0196959]`. Mean MacroCSY changed by -0.0160669, inside the frozen
-0.05 non-inferiority margin. The predeclared interpretation is therefore
`coverage_gain_with_efficiency_noninferiority`.

This result does not authorize a claim of exact coverage under arbitrary shift,
conditional coverage, or universal algorithmic superiority. Estimated density
ratios remain empirical. Two of 647 external structures failed the frozen RDKit
standardizer and were retained as explicit full-set exclusions; endpoint-specific
standardized overlaps with the training cohort were also excluded as locked.

The 39 MB row-level prediction file is intentionally not committed. Its SHA256
is recorded in `promotion_record.json` and `integrity_manifest.json`; the
one-command runner rebuilds it deterministically.

## Deterministic publication-facing inference

Post-evaluation audit note (2026-08-10): the original bootstrap loop used an
unordered Python set when assigning its fixed random stream to unique resampled
endpoints. The original interval above is retained as the sealed-run audit
value. Repair commit
`693c505fb55d287514530f2f6e92a50b96c8fa6a` changes only the endpoint traversal
order.

The hash-verified inference-only repair completed without model refitting,
prediction regeneration, or any label/prediction change. The publication-facing
95% interval is `[0.0058269, 0.0200509]`, or **+0.5827 to +2.0051 percentage
points**. The point estimate (1.3649 points), MacroCSY delta (-1.6067 points),
and predeclared interpretation are unchanged.

Use `publication_final_report.json` for manuscript values. The exact Windows
repair output is retained in
`deterministic_inference_repair_windows_20260810.json`, and
`deterministic_inference_publication_manifest.json` records the promotion.
The original `final_report.json` and `integrity_manifest.json` remain
unchanged to preserve the sealed audit chain.
