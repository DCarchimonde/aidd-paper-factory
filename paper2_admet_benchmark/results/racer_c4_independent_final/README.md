# RACER-C4 independent final EPA result

Status: **complete independent evaluation; no automatic superiority claim**

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

Post-evaluation audit note (2026-08-10): the original bootstrap loop used an
unordered Python set when assigning its fixed random stream to unique resampled
endpoints. The interval above is therefore retained as the original sealed-run
audit value, not the final publication-facing interval. Repair commit
`693c505fb55d287514530f2f6e92a50b96c8fa6a` changes only that traversal order.
The SHA-bound predictions, final labels, point estimate, MacroCSY estimate, and
interpretation are unchanged; the corrected interval is promoted only after
the inference-only repair record passes all hash checks.
