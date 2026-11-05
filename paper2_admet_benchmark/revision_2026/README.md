# CHEMOLAB complete revision implementation

Start with **[START_HERE_CN.md](START_HERE_CN.md)** for the full Chinese research protocol and operator-only instructions. This 5 November revision supersedes the previous partial handoff: the second author runs commands and returns artifacts; they are not asked to implement reference algorithms or choose analyses.

Download the [Chinese revision plan](CHEMOLAB_Full_Revision_Operator_Plan_20261105.pdf) or forward [OPERATOR_PROMPT_CN.txt](OPERATOR_PROMPT_CN.txt). The 4 November PDF is superseded historical material.

```sh
python -m pip install -r paper2_admet_benchmark/revision_2026/requirements_full.txt
python -m paper2_admet_benchmark.revision_2026.execute_revision --pilot --device auto --jobs 4 --output .local/chemolab_pilot_v2
python -m paper2_admet_benchmark.revision_2026.execute_revision --device auto --jobs 4 --output .local/chemolab_formal_v2
```

Use a fresh Python 3.12 environment. The chemical identity contract requires RDKit 2026.03.4. CPU and CUDA are supported; the full GNN matrix is intended for a server. Resume with identical arguments plus `--resume`. Pilot outputs are explicitly ineligible for formal reporting.

## Implemented scope

- Twelve Tox21 endpoints, development and retrospective EPA batches; three shared predictors, five fits per batch, three alpha values.
- AMES, standard hERG, DILI and CYP2D6_Veith, each with fixed random/scaffold source-target partitions and five internal fits. Thresholds transfer unchanged.
- Exact original-singleton-class budget matching against omitted-label probability, uncertainty, similarity and random deferral; fixed-budget CP/non-CP curves.
- Complete independently implemented CoDrug: AttentiveFP, auxiliary ZINC/energy loss, cross-validated KDE and query-mass weighted CP. Printed and swapped energy-margin configurations are both prespecified.
- Complete independently implemented KMM-CP with its own vanilla AttentiveFP, permutation MMD bandwidth selection, standard/joint QPs, target selection/refit, global and separately labelled class conventions.
- Query-weight alignment controls, discordant-error rescue analysis and synthetic mechanism diagnostics.
- Exact matrix/identity checks, complete/selected-population metrics, structure-cluster paired inference, figures, data-driven manuscript results and result packaging.

Paper-to-code details and disclosed choices are in [CODRUG_IMPLEMENTATION.md](full_baselines/CODRUG_IMPLEMENTATION.md), [KMM_IMPLEMENTATION.md](full_baselines/KMM_IMPLEMENTATION.md), [INDEPENDENT_VALIDATION.md](INDEPENDENT_VALIDATION.md), and [LITERATURE_AND_COMPARATORS.md](LITERATURE_AND_COMPARATORS.md). These are independent paper-based implementations, not claims of numerical reproduction of the authors' unavailable code.

`protocol.json` and `independent_sources.json` define the matrix before outcomes are scored. `execute_revision.py` orchestrates the run. `runner.py` remains available for diagnostic shared-predictor subruns. `density_baselines.py` remains a component-only diagnostic; it does not substitute for the full reference methods. The external adapter is optional infrastructure for future additional methods, not a required operator task.

The EPA labels were already inspected. These analyses are retrospective. Public additional-task evaluations are not newly collected or blinded clinical validation. Target outcomes do not enter prediction/ratio/selection APIs. Failed gates fall back by the declared algorithm; solver failure is an error, not an undocumented alternative method.

The complete research matrix has not been run during code delivery. See [DELIVERY_STATUS.md](DELIVERY_STATUS.md) for actual validation. The manuscript source in `paper2_latex/resubmission_20261104/` is a full pre-experiment revision draft; its new results are filled only from verified formal outputs.
