# RACER-C data acquisition and cleaning runbook

Status: **pre-freeze preparation**. This runbook creates no model predictions.

## Fail-closed order

1. Read `data_provenance_license_manifest.csv` and reject any endpoint whose
   `analysis_use_status` is `pending_original_terms`.
2. Acquire the exact named source through its recorded access layer. Never replace
   an unavailable dataset with a similarly named endpoint.
3. Save the original bytes once, compute SHA256 before decompression or parsing,
   and compare any predeclared hash.
4. Record the downloader/package version, immutable source revision or Dataverse
   version, retrieval timestamp in UTC, final resolved URL, HTTP metadata, and
   raw byte count.
5. Apply only the frozen standardization contract. Preserve verbatim source IDs,
   SMILES, labels, units, and row numbers in the local audit lineage.
6. Write one rejection-log row for every discarded or duplicate-aggregated source
   record. The accounting identity must reconcile source rows with retained and
   rejected/aggregated rows.
7. Hash the cleaning contract, code commit, clean CSV, rejection log, and
   role-feasibility input. A second run in a fresh directory must reproduce all
   canonical hashes.
8. Generate only group/count feasibility outputs. Do not train, calibrate, tune,
   or inspect extension test predictions before the formal protocol tag.

## License interpretation

The TDC repository license covers TDC code, not automatically every dataset. The
official TDC pages are authoritative for the access-layer record. Where a page
literally states `Dataset License: Not Specified`, the endpoint remains blocked
even if the same page links to a generic Creative Commons deed. A paper citation,
public URL, or successful download is not a data license.

Allowed-with-attribution status permits analysis preparation but does not waive
the requirements to record the exact dataset version, cite TDC and the original
source, preserve license notices, and audit any upstream restrictions. Raw data
are never committed to this repository.

## Required local file layout

```text
data/raw/racer_c/<endpoint>/<immutable source bytes>
data/processed/racer_c/<endpoint>_clean.csv
data/processed/racer_c/<endpoint>_rejections.csv
data/processed/racer_c/role_inputs/<endpoint>_role_input.csv
data/manifests/racer_c/<endpoint>_acquisition.json
data/manifests/racer_c/<endpoint>_cleaning.json
```

The role input has exactly one row per clean standardized structure and requires:

```text
endpoint,structure_id,target,murcko_scaffold_id,similarity_cluster_id
```

`similarity_cluster_id` may remain blank until the label-blind clustering
algorithm is frozen. Random-grouped and scaffold feasibility can still be audited;
the cluster track must fail closed rather than substituting scaffold IDs.

## Freeze-1 evidence packet

- completed provenance/license manifest;
- acquisition and cleaning JSON manifests with byte hashes;
- endpoint-specific label/unit/species/assay decision records;
- complete rejection logs and reconciliation table;
- deterministic clean-hash reproduction report;
- grouped role-count and conformal-resolution outputs for every candidate
  allocation, track, and main seed;
- endpoint eligibility decision file signed by commit SHA.
