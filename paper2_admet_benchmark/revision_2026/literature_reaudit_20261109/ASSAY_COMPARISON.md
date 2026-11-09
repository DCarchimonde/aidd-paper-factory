# Assay方向：逐篇原文重审（2026-11-09）

结论：此前不能由“已有cytotoxicity过滤、双终点或曲线表示学习”推出“配对起始响应预测整个方向不能做”。最近邻具有明显不同的信息条件与预测目标。与此同时，本轮补足此前未展开比较的2015年POD间隔规则；它比旧文档重点引用的若干新论文更直接：起始浓度差、约3倍间隔与重复测量都不是新概念。应保留方向、收窄可主张的增量，而不是全盘否决或直接宣布创新。

本轮对照对象：本项目使用分子结构和已知浓度设计，训练时读成对reporter/viability原始曲线，保留起始阈值的区间/左右删失及重复，预测未见分子的联合起始浓度与范围内3倍间隔。当前已测试的是线性关联项；初验不支持其筛选优势，不等于方向不存在。

## 原有五篇：按同一维度比较

| 文献与原文定位 | 研究问题 | 输入与监督 | 机制与输出 | 前提及验证 | 实质重合 | 未重合与裁定 |
|---|---|---|---|---|---|---|
| Huang 2022, *A Quantitative High-Throughput Screening Data Analysis Pipeline for Activity Profiling*, DOI 10.1007/978-1-0716-2213-1_13；全文§2.1–2.4、Tables 3–4、Fig.2 | 测后整合多读出与复测，辨别真实活性及干扰 | 已测plate/曲线、多个readout、counter-screen、复测；无需结构预测标签 | Hill拟合、curve class/rank、复测一致性、规则hitcall；§2.3用AC50差<3倍或t-test p>0.05识别细胞毒假阳性 | 依赖对照、拟合质量与测定读出，应用于Tox21；无未见结构预测或分布无关保证 | 同孔counter-screen、重复整合、浓度分离判断已有 | **直接的测量处理前例，不是预测机制重合。** 不能声称首次用3倍分离；可以研究以这些数据监督未见结构的配对分布，并应把官方规则作基线。 |
| Escher et al.2020, *Cytotoxicity Burst?*, DOI 10.1289/EHP6664；全文PDF pp.3–4，Eqs.2–10、Fig.1、Table1 | 给已观察reporter效应量化特异性，与细胞损伤及baseline narcosis比较 | 已测EC10与IC10；或由脂质分配性质预测baseline IC10；重复曲线合并 | SR=IC10/EC10（Eq.7）；QSAR基准SR（Eq.8）；毒性比值与SR分级 | baseline QSAR有适用域；自做实验＋六种Tox21 assays；高相对误差浓度被排除 | 浓度比、阈值分离和结构辅助的毒性基准不是新概念 | **输出概念近、学习任务不同。** 未联合学习两个删失阈值，未评估新结构的复测稳定间隔；不能据此排除该学习方向。 |
| Borrel et al.2024, NICEATM SOT poster P168/abstract4383，*Machine Learning-based Development of Cytotoxicity Flags Within the Tox21/ToxCast Assays*；全海报Fig.3–6、Tables2–5 | 为缺少同步viability的bioactivity测定补充细胞类型/时间特异毒性背景 | RDKit结构描述符；按cell/time分组后的每化合物median cytotoxicity AC50 | RF/XGBoost回归、20次欠采样模型平均；492个直接配对终点映射及AC50差统计 | 两个24h细胞系proof-of-concept，80/20与10-fold；输出单毒性AC50，无覆盖保证 | 配对终点映射、结构→条件特异毒性预测已有 | **“paired cytotoxicity models”容易误读。** 海报并非成对起始浓度联合删失模型，也不是双头模型比较。旧文档从它推出“普通双头预测不是贡献”证据跨度过大。 |
| Li et al.2026, *Tox21mer*, bioRxiv v1 DOI 10.64898/2026.06.15.732308，PMC13308148；Methods/Dataset、Tox21mer Overview、Discussion | 学习可复用曲线表示，并对已测曲线做outcome/AC50 probing | 推断时仍输入15点浓度—响应和assay metadata；masked reconstruction主监督，少量outcome/AC50辅助 | Transformer曲线编码＋冻结表示上的probe；互补viability被排除；结构学生/蒸馏仅Discussion初步讨论 | 随机SMILES划分；非同行评审预印本；作者说明学生蒸馏对已有直接监督增益少 | 多assay曲线表示、辅助监督、化学学生概念已有 | **邻近表示方法，非同信息条件的直接竞争者。** 未展示结构-only成对删失间隔任务；保留该方向，但若用教师—学生须对照其方案。 |
| Msemakweli et al.2026, *A library-wide cytotoxicity filter is not uniform*, DOI 10.1016/j.comtox.2026.100456 | 审查全库单一cytotoxicity bound在来源与测试范围差异下的含义 | invitrodb v4.3已给出的bound、AC50、hitcall、浓度范围 | 对默认/估计bound分层，分析1/3/10倍过滤保留与family coverage | 出版商摘要与部分正文索引已核验；付费全文未获取。卷期标December2026，当前已有线上记录 | “默认高上界≠实测分离”、测试范围限制已有直接讨论 | **相邻数据解释问题；不是配对结构预测。** 不得把未核验全文当成排除联合学习的证据。 |

## 新增更近的原始工作

| 文献与定位 | 核验发现 | 对本项目的准确约束 |
|---|---|---|
| Hsieh et al.2015, *A Data Analysis Pipeline Accounting for Artifacts in Tox21 Quantitative High-Throughput Screening Assays*, DOI 10.1177/1087057115581317，PMC4568956；Methods的Signal quantification、Assay interference identification、Data Collapsing；Eq.1；Fig.3 | POD定义为超过assay噪声阈值的线性插值浓度。同孔primary与viability的POD分离<0.5 log10且t-test p>0.05时标干扰；0.5来自阳性对照potency变化。对三次运行取中位数，研究32个assay复测。 | **这是比AC50比值更直接的onset先例。** 不能把首次阈值跨越、约3倍POD间隔或复测处理当新贡献；仍未训练未见结构的成对删失概率分布。可作为测量定义、去噪和复测稳定性的重要基线。 |
| Minnich et al.2020, *AMPL: A Data-Driven Modeling Pipeline for Drug Discovery*, DOI 10.1021/acs.jcim.9b01053，PMC7189366；Methods/Data Curation及作者文档mle_censored_mean、aggregate_assay_data | 有限浓度导致IC50/EC50左右删失；含删失与非删失复测时以Gaussian测量分布MLE聚合均值，随后做结构模型。 | **删失＋复测用于分子建模并非空白。** 这是单终点聚合路线，不能替代成对联合事件学习；须与合理的删失聚合/独立预测基线比较。论文Methods相关全文索引与官方实现文档已读，未逐页复核所有结果。 |
| *Counting cells can accurately predict small-molecule bioactivity benchmarks*, Nature Communications2026, DOI 10.1038/s41467-026-68725-5；原文Results/Cell count matches…、Methods/Predicting compound activity from phenotypic profiles及curated Protein-Target data | cell count作为测后表型输入，可解释许多旧bioactivity基准；用独立蛋白靶标面板检验完整表型的增量。 | **这是任务与对照的启示，不是结构-only间隔模型。** 不能因其研究细胞毒性就排除此方向；若使用表型/viability辅助信息，应匹配信息成本与该类简单基线。原文主要Methods与Results索引已读，网页直接打开有访问错误。 |

## 对旧文档的修正

1. “NCATS已有处理cytotoxicity”正确，但它只划定了测后处理层的先例，不否定测前结构预测。新增Hsieh2015，否则首次onset/复测叙事会漏掉更直接前例。
2. “EHP已有SR”正确；把它扩大成“所有配对阈值学习都没有新意”不成立。SR是点估计比例，联合区间监督和预测信息条件不同。
3. NICEATM行应写“条件特异的单毒性AC50预测＋直接配对映射”，不能概括为已实现我们所说的联合双头/成对删失模型。
4. Tox21mer应注明预印本；互补viability排除的陈述成立。主任务需要实际曲线输入，与当前结构-only任务不可直接按F1比较。作者已讨论化学学生，故蒸馏概念也不能独占。
5. 2026 filter paper只能作已核验公开内容层面的邻近先例；未读全文的机制维度标为未核验，不能据其标题作否决。

## 重新开放的具体空间与尚需成立的贡献

可重新开放：从结构预测在实验浓度范围内、对复测变异仍稳定的成对响应事件；在训练中显式保留配对和删失，并解释它较独立删失学习、单终点聚合与直接分类究竟增加了什么信息。上述文献分别覆盖测后间隔规则、单毒性QSAR、曲线表征或单终点删失聚合，本轮没有找到把这些条件全部合在一起并完成同任务实证的原文。

这只是**有边界的候选空间**：联合似然、层次模型、删失和多任务学习本身均是既有工具；不能靠把它们排列在一起判定创新。需要解释配对信息如何改变目标事件估计，再在相同推断输入和化学划分下，检验它相对直接事件分类、合理的独立删失预测及聚合路线有何可归因的增量。不要求证明所有独立方法原则上都无法达到。当前线性rho初验失败只约束该实现，不足以排除整个空间；恢复研究资格不意味着恢复已失败模型的全面实验。

原文访问范围见 [SOURCES.json](SOURCES.json)。本分项复核未训练模型。
