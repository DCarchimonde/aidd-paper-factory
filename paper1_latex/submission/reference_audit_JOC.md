# Reference Audit — Journal of Chemometrics Manuscript

## Audit standard

- Every cited item must be retrievable from a publisher, proceedings platform, DOI resolver, PubMed, OpenReview, or arXiv.
- Publication type is stated accurately; preprints are not presented as journal articles.
- Recent-literature target: at least 30 cited works from 2021–2026.
- Older works are retained only when foundational to scaffolds, fingerprints, QSAR/QSPR validation, applicability domains, or software/method definitions.
- A BibTeX entry is not considered accepted merely because it compiles; title, authorship, year, venue, pagination/article number, and DOI/identifier must agree with the source record.

## Recent references cited in the manuscript (2021–2026)

| Key | Year | Type | DOI / official identifier | Role in manuscript | Audit status |
|---|---:|---|---|---|---|
| hu2021ogblsc | 2021 | NeurIPS Datasets and Benchmarks proceedings | official proceedings URL | large-scale benchmark and standardized evaluation | verified |
| huang2021tdc | 2021 | NeurIPS Datasets and Benchmarks proceedings | arXiv:2102.09548 / proceedings record | therapeutics benchmark infrastructure | verified |
| liu2022graphmvp | 2022 | ICLR conference paper | OpenReview xQUe1pOKPam | 2D/3D cross-view pretraining | verified |
| wang2022molclr | 2022 | journal article | 10.1038/s42256-022-00447-x | graph contrastive molecular learning | verified |
| fang2022gem | 2022 | journal article | 10.1038/s42256-021-00438-4 | geometry-enhanced property prediction | verified |
| li2022adaptive | 2022 | journal article | 10.1038/s42256-022-00501-8 | adaptive graph learning | verified |
| ross2022molformer | 2022 | journal article | 10.1038/s42256-022-00580-7 | chemical language modelling | verified |
| zeng2022image | 2022 | journal article | 10.1038/s42256-022-00557-6 | image-based self-supervised representation | verified |
| vantilborg2022cliffs | 2022 | journal article | 10.1021/acs.jcim.2c01073 | activity-cliff limitations | verified |
| jimenez2022attribution | 2022 | journal article | 10.1021/acs.jcim.1c01163 | activity-cliff attribution benchmark | verified |
| oliveira2022sgvae | 2022 | journal article | 10.1021/acs.jcim.1c01573 | grammar-based molecular representation | verified |
| jiang2022multigran | 2022 | journal article | 10.1093/bioinformatics/btac550 | multigranular SMILES representation | verified |
| zhu2022unified | 2022 | KDD conference paper | 10.1145/3534678.3539368 | unified 2D/3D pretraining | verified |
| li2022imdrug | 2022 | preprint/benchmark release | arXiv:2209.07921 | imbalanced AIDD benchmark | verified; preprint explicitly labelled |
| li2023kpgt | 2023 | journal article | 10.1038/s41467-023-43214-1 | knowledge-guided graph-transformer pretraining | verified |
| zang2023hierarchical | 2023 | journal article | 10.1038/s42004-023-00825-5 | hierarchical molecular self-supervision | verified |
| jiang2023pharmacophore | 2023 | journal article | 10.1038/s42004-023-00857-x | pharmacophore-constrained graph transformer | verified |
| wu2023substructure | 2023 | journal article | 10.1038/s41467-023-38192-3 | substructure-based explanation | verified |
| dias2023limitations | 2023 | journal article | 10.1038/s41467-023-41967-3 | limitations on small molecular benchmarks | verified |
| fang2023kgcontrastive | 2023 | journal article | 10.1038/s42256-023-00654-0 | knowledge-graph contrastive learning | verified |
| zhou2023unimol | 2023 | ICLR conference paper | OpenReview 6K2RM6wVqKu | 3D molecular representation | verified |
| liu2023moleculestm | 2023 | journal article | 10.1038/s42256-023-00759-6 | structure–text multimodal representation | verified |
| zhang2023acnet | 2023 | preprint/dataset release | 10.48550/arXiv.2302.07541 | activity-cliff prediction benchmark | verified; preprint explicitly labelled |
| zhao2024gslmpp | 2024 | journal article | 10.1093/bioinformatics/btae304 | graph-structure learning for property prediction | verified |
| sultan2024transformers | 2024 | journal review | 10.1021/acs.jcim.4c00747 | recent transformer evidence and limitations | verified |
| gao2024knomol | 2024 | journal article | 10.1021/acs.jcim.4c01092 | knowledge-enhanced molecular transformer | verified |
| lasfar2024robustness | 2024 | journal article | 10.1002/cem.3530 | cross-validation versus bootstrap robustness | verified |
| guo2024scaffold | 2024 | preprint | arXiv:2406.00873 | cross-scaffold chemical similarity and screening bias | verified; preprint explicitly labelled |
| kiraly2025leakage | 2025 | journal article | 10.1002/cem.70026 | leakage and cross-validation scaling | verified |
| ezenarro2025validation | 2025 | journal systematic review | 10.1002/cem.70036 | empirical reporting quality of chemometric validation | verified |
| luo2025molclsp | 2025 | journal article | 10.1093/bioinformatics/btaf507 | recent multimodal contrastive property prediction | verified |
| camacho2026validation | 2026 | journal article | 10.1002/cem.70110 | current model-validation rules | verified |

**Recent cited total: 32 works (2021–2026).** Of these, 29 are journal or peer-reviewed conference/proceedings items and 3 are transparently labelled preprints/benchmark releases.

## Foundational references retained outside the five-year window

- Bemis and Murcko (1996): molecular scaffold definition.
- Rogers and Hahn (2010): extended-connectivity fingerprints.
- Tropsha et al. (2003) and Gramatica (2007): QSAR/QSPR validation principles.
- Netzeva et al. (2005) and Sahigara et al. (2012): applicability-domain foundations.
- Esbensen and Geladi (2010): proper validation and resampling principles.
- MoleculeNet (2018), learned-representation comparisons, and software/method papers are retained where they define the benchmark, algorithm, or implementation used.

## Remaining checks before submission

1. Run the clean build; no `??`, undefined citation, undefined reference, or BibTeX rerun warning is permitted.
2. Confirm every item listed in the generated bibliography is cited in the text and every text citation appears in the bibliography.
3. Check the final rendered PDF, not only the `.tex` source.
4. Keep preprints explicitly identified as preprints in the bibliography and prose.
5. Re-run this audit after any citation-key or bibliography change.
