# Dominant-fragment terminology note

The final manuscript uses the term **dominant fragment** for the deterministic single-fragment representation sensitivity.

Some frozen implementation artifacts retain the historical filename/token `parent_fragment`, specifically:

- `PARENT_FRAGMENT_SENSITIVITY_PROTOCOL_V3.md`
- `scripts/15_run_parent_fragment_sensitivity_v3.py`
- `results/parent_fragment_sensitivity_v3/`

These names are preserved intentionally because the completed sensitivity jobs and frozen metadata record the SHA-256 digest of the protocol file used at execution time. Renaming or editing that frozen protocol after model fitting would unnecessarily invalidate provenance checks.

The operational definition is unchanged: disconnected fragments are ranked by heavy-atom count, then carbon-atom count, then canonical isomeric SMILES as a deterministic tie-break. The selected top-ranked component is described in the manuscript as the **dominant fragment**, not as a chemically guaranteed true parent molecule. This wording avoids implying that the algorithm resolves salts, mixtures, co-crystals, or metal complexes to an authoritative biological parent structure.

This terminology note changes no data, split, model, statistic, or result.
