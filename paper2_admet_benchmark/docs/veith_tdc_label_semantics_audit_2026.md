# Veith CYP to TDC binary-label semantics audit

Status: **polarity supported; exact transformation provenance unresolved**

Audit date: 2026-08-03

TDC describes `CYP2C9_Veith`, `CYP2D6_Veith`, and `CYP3A4_Veith` as binary
inhibition datasets and distributes a binary `Y` column under CC BY 4.0. The
downloaded raw positive counts are 4,045, 2,514, and 5,110. Independent benchmark
summaries reproduce those exact active counts, which supports interpreting
`Y=1` as active/inhibitor and `Y=0` as inactive/non-inhibitor.

The original Veith qHTS study reports concentration-response curve classes,
activity calls, efficacy, and assay-specific quality-control logic. Current TDC
metadata and loader code map the endpoint name to the preprocessed CSV but do not
publish an executable rule that derives the binary `Y` value from the original
assay fields. PubChem confirms the assay identity and qHTS context but does not by
itself prove TDC's complete transformation path.

Accordingly:

- allowed claim: the TDC-distributed label polarity is active/inhibitor versus
  inactive/non-inhibitor;
- disallowed claim: the repository has independently reconstructed TDC's exact
  curve-class/flag-to-binary transformation;
- protocol consequence: these endpoints are not primary Freeze-1 candidates
  under the existing exact-semantic-recovery rule. They may become source-defined
  secondary analyses only if that narrower interpretation is frozen before model
  outcomes, or become primary only after the transformation is documented.

Sources:

- https://tdcommons.ai/single_pred_tasks/adme/
- https://pmc.ncbi.nlm.nih.gov/articles/PMC2783980/
- https://pubchem.ncbi.nlm.nih.gov/bioassay/883
- https://arxiv.org/html/2411.09820v1
