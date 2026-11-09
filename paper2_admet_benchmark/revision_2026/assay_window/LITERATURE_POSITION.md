# 新方向与最近邻工作的关系

更新：2026-11-09。逐篇机制、版本与访问范围见 [完整文献复核](../literature_reaudit_20261109/README_CN.md) 和 [配对测定比较](../literature_reaudit_20261109/ASSAY_COMPARISON.md)。当前线性关联实现的负结果不排除整个研究方向。

原候选定义：选定问题为配对浓度反应起始间隔的结构预测；这是研究候选，近期工作已有大量可复用统计与毒理组件。

| 原始来源 | 已解决的部分 | 本轮具体检验 |
|---|---|---|
| [NCATS qHTS activity profiling pipeline, 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9540341/) | 整合重复、主读出和counter-screen形成活性判定；已处理cytotoxicity干扰。 | 使用完整配对浓度括区训练结构模型，比较关联与独立删失学习；官方规则保留作参考。 |
| [Cytotoxicity Burst?, EHP 2020](https://ehp.niehs.nih.gov/doi/10.1289/EHP6664) | EC/IC与specificity ratio已存在。 | 测量范围之外的阈值保留删失，检验两阈值联合分布对起始间隔预测的增量。 |
| [NICEATM paired cytotoxicity models, SOT 2024](https://ntp.niehs.nih.gov/sites/default/files/2024-03/SOT2024-Borrel-poster.pdf) | cell/time-specific cytotoxicity预测与492 endpoints的配对映射。 | 海报未实现双头或成对删失模型，不能据它排除此类研究；普通多任务组件本身仍不足以成立贡献。 |
| [A library-wide cytotoxicity filter is not uniform, 2026](https://www.sciencedirect.com/science/article/pii/S2468111326000575) | 公开摘要及部分索引正文讨论bound来源和测试范围；付费全文未读。 | 研究从配对测量证据学习，不再把默认1000μM问题当新发现。 |
| [Tox21mer, 2026预印本v1](https://pmc.ncbi.nlm.nih.gov/articles/PMC13308148/) | 浓度曲线与assay context的表示学习；其主体数据处理排除了互补viability读出。 | 将配对viability及删失区间直接纳入目标；模型预测时只用结构/设计变量，不能把输入曲线重建官方标签等同化学预测。 |

[QComp, JCIM 2025](https://doi.org/10.1021/acs.jcim.5c00059) 是补全与实验选择的重要邻居；本轮完整读取的是2024预印本v1及当前作者代码，正式全文未取得，不能混称同一版本。补全、cliff损失和DRO已有先例，只能界定具体机制的重合，不能据此排除整个方向。

更直接的测量前例包括 [Hsieh 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4568956/) 的双通道POD间隔规则，以及 [AMPL 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7189366/) 的删失复测聚合。起始间隔和处理删失本身均不能作为首创；本项目需检验配对监督相对直接分类及合理独立模型的实际增量。

联合类别模型、关联项和区间删失似然都采用既有统计思想。论文能否成立取决于化学与测量层的结果：保留范围信息和配对关联，是否帮助预测未见结构的起始间隔，并在复测与更强独立模型前保持收益。

官方数据与定义：

- [NCATS Tox21 Public Data](https://tripod.nih.gov/pubdata/)
- [字段定义：CONC为M、AC50为μM](https://tripod.nih.gov/pubdata/fieldCategory_fieldName_definition.txt)
- [通道与重复编号定义](https://tripod.nih.gov/pubdata/fieldName_fieldValue_definition.txt)
- [已核验的AR-luc antagonist p1原始包](https://opendata.ncats.nih.gov/public/tox21/assay/data/tox21-ar-mda-kb2-luc-antagonist-p1.zip)

与该原始包同行的description与实验协议明确same-well viability/reporter读数顺序。导出不含plate或batch标识，当前的重复编号只按官方定义使用。
