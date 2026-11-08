# CHEMOLAB 技术候选与近期工作核查（2026-11-08）

## 研究选择

本轮选定：**在一个固定分子批次中，通过目标标签的随机抽样测定，同时约束两类错误，并减少完成整批判定所需的总测定量。**

算法联合处理三个决定：哪些区域自动判0、哪些自动判1、下一批真实测定投向哪里。免测收益扣除已经花掉的抽样测定；剩余未自动判定的分子全部计入后续测定需求。最终容量为 B 时，当前最小已认证总需求 C*≤B 才表示容量足够。

技术候选集中在**多区域双向联合决策 + 面向净免测收益的信息获取**。有限总体置信序列、主动获取标签、后验选择决策、成本敏感自动标注都有直接先例。下面据此区分已有方法和需要实验检验的增量。

当前已实现八方法的诊断矩阵、共享预测器、逐次标签接口、成本统计及自动报告；真实分子结果待执行。具体版本以 [RESEARCH_PLAN_CN.md](RESEARCH_PLAN_CN.md) 和 [THEORY_AND_POSITIONING.md](THEORY_AND_POSITIONING.md) 为准。

## 最近邻与具体差异

| 原始来源 | 已有结果 | 对本研究的约束与切入点 |
|---|---|---|
| Candès, Ilyas, Zrnic, **Probably Approximately Correct Labels**, ICML 2026：[论文](https://proceedings.mlr.press/v306/candes26a.html)、[代码](https://github.com/tijana-zrnic/pac-labels/) | 固定数据集、随机审核、自动标注与专家回退、高概率总体错误控制；并研究成本敏感路由和分组校准。 | 固定库与节省标签已有先例。本轮直接联合约束两个未知真实类别分母下的错误，选择非单链的区域动作，并优化获取信息的价值。 |
| Zhang, Huang, Jin, **CARED**, 2026-10-29预印本：[论文](https://arxiv.org/abs/2610.40252) | 随机审核、任意分布漂移、逐类错误及自动化效用；§3.3研究长期累计错误的几乎必然控制。 | Discussion将有限样本实际错误的置信界列为后续方向。本轮针对单个有限批次、所有审核时点的高概率错误事件。 |
| Xu, Karampatziakis, Mineiro, **Active, anytime-valid risk controlling prediction sets**, NeurIPS 2024：[论文](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6eb05d8bc6bd7bb6868c64b5802125bd-Abstract-Conference.html) | 主动查询、随时有效的风险保证、预测辅助，以及固定标签预算下的查询策略分析。 | 本轮采用固定批次内分层无放回查询，收益直接由双类风险下可省掉的测定量定义；“主动+anytime”属于已有框架。 |
| **Exact Finite-Population Sequential Auditing for Selective Release from Model-Proposed Candidate Sets**, Mathematics 2026：[论文](https://www.mdpi.com/2227-7390/14/18/3427) | 冻结候选集上的有限总体顺序审核、释放决策与回退；已处理错误分子与全局真实缺陷分母的耦合。 | 单区的分母耦合已有先例。可检验扩展是多层、双向动作及其联合测定分配，不能单独把耦合比值作为新贡献。 |
| Waudby-Smith, Ramdas, **Confidence sequences for sampling without replacement**, NeurIPS 2020：[论文](https://proceedings.neurips.cc/paper/2020/hash/e96c7de8f6390b1e6c71556e4e0a4959-Abstract.html) | 无放回有限总体的时间一致置信序列与计数推断。 | 本轮使用既有混合似然比构造，再推导双风险动作优化；均匀先验与固定边界混合先验作为效率对照。 |
| Bai, Jin, **SCoRE: Conformal Selective Prediction with General Risk Control**, 2026预印本：[论文](https://arxiv.org/abs/2603.24704) | 一般有界风险、选择性部署、协变量漂移，含药物发现中的成本和多样性实例。 | 化学成本与风险控制已结合。本轮以整批真实阳性/阴性为分母，采用付费目标测定获得有限批次证据。 |
| Bai等, **Conformal Selection for Efficient and Accurate Compound Screening in Drug Discovery**, JCIM 2025：[论文](https://pubs.acs.org/doi/10.1021/acs.jcim.5c01633) | 分子筛选中的错误发现与错误遗漏控制。 | 错误遗漏率的分母是判阴集合，FNR的分母是真实阳性；比较时分别报告这些指标。 |
| Yu, Liu, **A Joint Finite-Sample Certificate for Adaptive Selective Conformal Risk Control**, 2026预印本：[论文](https://arxiv.org/abs/2606.08517) | 联合风险、接受率与效用证书，允许数据驱动阈值选择和不可行结果；采用固定分布下的i.i.d.校准。 | 联合证书和预算不可行的概念已有先例。本轮依据固定总体的随机抽样，约束两个实际类别分母。 |

此外，[Provably Label-Efficient Conformal Prediction, ICML 2026](https://proceedings.mlr.press/v306/ilyas26a.html) 已研究校准标签数量和效率之间的停止决策。因此当前科学问题需要落实到“哪些测定带来多少净免测收益”，而非泛称标签效率。

## 候选1：多区域双风险联合动作（已实现）

对冻结模型分数与训练集化学相似度构造固定区域。每层剩余 n_h 个未测分子，未知阳性数为 K_h，同时置信界为 [L_h,U_h]；累计已测阳性和阴性分别为 O+、O−。

总体类别数为 N+=O++ΣK_h、N−=O−+Σ(n_h−K_h)。每层剩余分子统一自动判0、自动判1或测定。

若自动判0的指示为 z0_h、自动判1为 z1_h，则两个风险在共享计数矩形上的精确约束是：

- Σ z0_h[(1−α)U_h+αL_h] ≤ α(O++ΣL_h)。
- Σ z1_h[(1−β)(n_h−L_h)+β(n_h−U_h)] ≤ β(O−+Σ(n_h−U_h))。

在 z0_h+z1_h≤1 下最大化 Σn_h(z0_h+z1_h)，得到当前最少总测定量。相同的计数事件覆盖所有区域动作组合，因此可以随审核证据重新选择动作。公式推导和概率前提见理论文档。

关键比较是：相同计数界、风险水平和标签预算下，联合动作相比按分数排序的双阈值能省多少测定；保留分子分母依赖，相比独立取界能减少多少保守性。分层由标签揭示前的信息确定，层内查询采用均匀无放回抽样。

## 候选2：按预计净节省分配测定（已实现）

令 C*(state) 为当前最小已认证总测定需求。枚举下一小批测定的可能阳性数，预测各结果下的计数界并重新求解动作优化，选择预期降低 C* 最多的区域。C* 包含已查询成本，因而获得标签本身要付费。

当前实现使用最多4个分子的短期前瞻；结果概率来自冻结分数与已揭示标签的平滑估计。这个预测影响获取效率，最终风险界由实际随机测定更新。预计增益全部为零时探索计数区间较宽的区域。查询策略是待验证的启发式；当前状态的动作优化在固定区域策略类内精确求解。

八种诊断方法为：joint_lookahead、joint_uniform、joint_entropy、joint_neyman、threshold_lookahead、loose_lookahead、score_only_lookahead、uniform_cs_lookahead。前四种使用相同区域、统计界和动作优化器，直接检验信息获取策略的增量。

## 候选3：多终点共享实验预算（后续扩展）

一个实验面板可能同时揭示多个终点，获取价值因此跨终点共享。值得研究的对象是“实验动作—可揭示终点—联合风险—成本”的优化关系。

[Conformal Risk Control, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html) 及 [Data-Adaptive Tradeoffs among Multiple Risks](https://arxiv.org/abs/2403.19605) 已涵盖多风险与可调权衡。新问题应利用同一次测定揭示多个标签的结构，而非简单增加阈值数量。

Tox21可作为候选，但其标签缺失决定了哪些面板可被回放。需要先定义每次动作真实可揭示的标签集合；成本缺少实测依据时采用明确的规范化场景。这一方向留在单终点机制得到实证支持之后。

## 候选4：预测辅助或双重稳健校准（效率组件）

[Semi-Supervised Risk Control, TPAMI 2025](https://arxiv.org/abs/2412.11174)、[PPAT](https://arxiv.org/abs/2607.08347)、SCoRE及Active RCPS已经覆盖预测辅助、小样本校准、残差获取或双重稳健校正的不同组合。

这些工具可用于降低同一双风险任务的标签成本，作为后续强对照或统计效率改进。当前先检验联合动作和获取价值；若主要瓶颈确实来自计数区间宽度，再加入对应的预测辅助构造。

## 从诊断矩阵到论文级比较

当前96条分子轨迹检验机制是否产生值得继续研究的增益，完整1280条矩阵继续检验四任务两划分上的表现。现有八种方法包含强的同证书对照及机制消融；论文级比较还需加入最近邻原方法或明确命名的类风险改编。

PAC Labels优先级最高。其官方代码固定版本为 [b415b58756b14b384529ac9cf146bd5d4c8139aa](https://github.com/tijana-zrnic/pac-labels/tree/b415b58756b14b384529ac9cf146bd5d4c8139aa)，入口 `pac_labeling(Y,Yhat,loss,epsilon,alpha,uncertainty,pi,num_draws,asymptotic=True)`。非渐近比较需选 `asymptotic=False`。

官方接口接收完整Y并预计算损失，适合离线模拟。接入逐次标签接口时，应采用预知有界损失上限，仅在实际查询时揭示标签，并核对无阈值通过时的回退和数值搜索。所有已查询标签及最终回退测定都计入同一总成本。当前 `threshold_lookahead` 是相同计数界下的阈值策略，尚未实现作者版PAC Labels。

CoDrug/KMM等零目标标签方法可在后续同一三动作成本表中比较；它们的预测集合转为“自动决定/测定”后，需要计入全部回退测定。PACL的总体错误指标和本轮双类错误分别报告。

主要判据是：同一风险水平下，相比有竞争力的同证书方法产生可重复的净免测增益，并解释收益来自联合动作、查询策略还是统计界。当前任务均为既有公开数据的回顾性研究；新终点或新批次用于后续独立确认。结果若显示罕见阳性使几乎整批都需要测定，就定位造成成本的统计与化学条件，再决定效率改进的方向。
