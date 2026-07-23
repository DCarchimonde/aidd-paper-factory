# Paper 2 literature audit

## Audit rule

Every reference in `references.bib` was admitted only after its title, venue, year, and DOI or official proceedings record were checked against a publisher or official conference page. Recent work from 2021--2026 is prioritized. Pre-2021 sources are retained only when they are foundational to the dataset, representation, calibration, uncertainty-quantification, or conformal method used in the study.

The table below records the intended evidentiary role of every citation. A reference is not cited merely because it is topically related; it must support a specific sentence or methodological choice.

| Key | Year | Type | Intended use in the manuscript | Verification anchor |
|---|---:|---|---|---|
| `xiong2021admetlab` | 2021 | ADMET platform | Establish recent large-scale multi-endpoint ADMET prediction systems | NAR publisher page; DOI 10.1093/nar/gkab255 |
| `wei2022interpretable` | 2022 | ADMET platform | Explain demand for interpretable prediction and molecular optimization | Bioinformatics publisher page; DOI 10.1093/bioinformatics/btac192 |
| `tian2022admetboost` | 2022 | ADMET platform | Document continued development of high-throughput ADMET prediction services | Springer DOI 10.1007/s00894-022-05373-8 |
| `zhang2022helix` | 2022 | ADMET platform | Support modern transferable, multi-endpoint ADMET modeling | Bioinformatics publisher page; DOI 10.1093/bioinformatics/btac342 |
| `swanson2024admetai` | 2024 | ADMET platform | Support large-library ADMET screening context | Bioinformatics publisher page; DOI 10.1093/bioinformatics/btae416 |
| `huang2021tdc` | 2021 | Benchmark/data commons | Establish standardized drug-discovery task collections | Official NeurIPS proceedings |
| `heid2024chemprop` | 2024 | Molecular ML software | Position current molecular property modeling ecosystem | ACS publisher page; DOI 10.1021/acs.jcim.3c01250 |
| `kim2024quantum` | 2024 | Representation learning | Show continuing gains from richer molecular pretraining | ACS publisher page; DOI 10.1021/acs.jcim.4c00772 |
| `fang2022geometry` | 2022 | Representation learning | Show modern 3D/geometry-aware molecular representation development | Nature Machine Intelligence publisher record |
| `oliveira2022grammar` | 2022 | Representation learning | Provide an additional recent molecular representation example | ACS publisher page; DOI 10.1021/acs.jcim.1c01573 |
| `hirschfeld2020molecularuq` | 2020 | Molecular UQ | Foundational molecular uncertainty benchmark predating the five-year emphasis | ACS publisher page; DOI 10.1021/acs.jcim.0c00502 |
| `scalia2020gnnuq` | 2020 | Molecular UQ | Foundational comparison of scalable GNN uncertainty methods | ACS publisher page; DOI 10.1021/acs.jcim.9b00975 |
| `lanini2024unique` | 2024 | Molecular UQ benchmark | Support multidimensional UQ benchmarking rather than a single score | ACS/PMC record; DOI 10.1021/acs.jcim.4c01578 |
| `jiang2024gnas` | 2024 | Molecular UQ | Show uncertainty estimation for molecular graph models | RSC publisher page; DOI 10.1039/D4DD00088A |
| `li2024conformalized` | 2024 | ADMET conformal | Establish recent conformalized ADMET prediction work | ACS publisher page; DOI 10.1021/acs.jcim.4c01139 |
| `liu2024reactionuq` | 2024 | Chemical UQ | Demonstrate uncertainty evaluation beyond equilibrium molecular endpoints | ACS publisher page; DOI 10.1021/acs.jcim.4c01358 |
| `tang2025polymeruq` | 2025 | UQ benchmark | Support endpoint- and evaluation-design dependence of UQ rankings | ACS publisher page; DOI 10.1021/acs.jcim.5c00550 |
| `koetter2025upgrading` | 2025 | Molecular UQ benchmark | Support dependence of UQ performance on chemical space and split design | ACS publisher page; DOI 10.1021/acs.jcim.5c00464 |
| `parrondo2026shifts` | 2026 | Molecular UQ benchmark | Direct comparison of UQ behavior under molecular data shifts | ACS publisher page; DOI 10.1021/acs.jcim.5c02381 |
| `kim2025distance` | 2025 | Applicability domain | Support relationship between domain distance and prediction difficulty while avoiding a universal cutoff claim | ACS publisher page; DOI 10.1021/acs.jcim.5c01037 |
| `komissarov2025explainable` | 2025 | Explainability and UQ | Support distinction between prediction attribution and uncertainty attribution | ACS publisher page; DOI 10.1021/acs.jcim.5c01003 |
| `ash2025comparison` | 2025 | Benchmark methodology | Support paired, practically meaningful method comparison and avoidance of model-as-replicate inference | ACS publisher page; DOI 10.1021/acs.jcim.5c01609 |
| `zhang2021toxicity` | 2021 | Molecular conformal | Support toxicity-focused conformal prediction context | ACS publisher page; DOI 10.1021/acs.jcim.1c00208 |
| `krstajic2021critical` | 2021 | Conformal critique | Support caution when applying conformal methods to binary classification | ACS publisher page; DOI 10.1021/acs.jcim.1c00549 |
| `laghuvarapu2023codrug` | 2023 | Shift-aware conformal | Closest recent molecular comparator for density-aware conformal prediction under covariate shift | Official NeurIPS proceedings |
| `gibbs2021adaptive` | 2021 | Distribution shift | Support distinction between exchangeable and adaptively shifted conformal settings | Official NeurIPS proceedings |
| `cauchois2021knowing` | 2021 | Conditional validity | Support validated confidence sets and subgroup-aware evaluation | JMLR official record |
| `ding2023classconditional` | 2023 | Class-conditional conformal | Support class-conditional coverage motivation | Official NeurIPS proceedings |
| `angelopoulos2023pid` | 2023 | Adaptive conformal | Broader recent adaptive conformal context | Official NeurIPS proceedings |
| `bastani2022multivalid` | 2022 | Group validity | Support multigroup validity as distinct from marginal validity | Official NeurIPS proceedings |
| `fisch2022trading` | 2022 | Coverage-efficiency trade-off | Support joint analysis of coverage and precision/informativeness | Official ICLR record |
| `teng2023feature` | 2023 | Feature-conditional conformal | Support feature-aware predictive inference context | Official ICLR record |
| `einbinder2023labelnoise` | 2023 | Conformal robustness | Support robustness boundary discussion | Official ICLR record |
| `gawlikowski2023survey` | 2023 | UQ survey | Define epistemic/aleatoric uncertainty and explain evaluation diversity | ACM publisher record; DOI 10.1145/3559752 |
| `hullermeier2021aleatoric` | 2021 | UQ concepts | Conceptual distinction between aleatoric and epistemic uncertainty | Springer publisher record; DOI 10.1007/s10994-021-05946-3 |
| `wu2018moleculenet` | 2018 | Dataset benchmark | Source and benchmark context for BBBP, ClinTox, ESOL, and Lipophilicity | RSC publisher page; DOI 10.1039/C7SC02664A |
| `rogers2010ecfp` | 2010 | Molecular representation | Original extended-connectivity fingerprint definition | ACS publisher page; DOI 10.1021/ci100050t |
| `delaney2004esol` | 2004 | Dataset | Original ESOL dataset/model context | ACS publisher page; DOI 10.1021/ci034243x |
| `martins2012bbbp` | 2012 | Dataset | Original BBBP modeling dataset context | ACS publisher page; DOI 10.1021/ci300124c |
| `tibshirani2019covariate` | 2019 | Weighted conformal | Foundational weighted conformal method under covariate shift | Official NeurIPS proceedings |
| `guo2017calibration` | 2017 | Calibration | Foundational modern probability-calibration reference | Official ICML proceedings |

## Citation-placement safeguards

1. Dataset citations support provenance, not claims of external validation.
2. Molecular UQ papers are used to position the reliability problem, not to imply that their models were reproduced.
3. Weighted conformal citations are accompanied by an explicit assumptions paragraph; estimated density ratios do not justify an unconditional exact-coverage claim.
4. Class-conditional and multivalid references motivate subgroup evaluation, but empirical ClinTox conclusions come only from the frozen outputs of this study.
5. Recent ADMET platforms establish the practical modeling ecosystem; they do not validate the present benchmark numerically.
6. Older references are not counted toward the requested recent-literature minimum.
