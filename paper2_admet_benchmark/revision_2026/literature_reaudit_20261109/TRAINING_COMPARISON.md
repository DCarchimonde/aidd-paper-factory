# 训练机制文献复审（2026-11-09）

**结论：应撤销“activity-cliff-aware training / scaffold DRO 已有人做，所以整条训练路线没有空间”的排除。** 原文能排除的是特定损失和优化结构，不能排除所有化学局部机制。反过来，最接近“直接优化稀有类别的 conformal 集合”的前作其实是 ConfTr；它已经包含类别权重、集合组成与应用损失，不能仅把这些改名为新方法。

这五篇是定向核验，不是全领域穷尽检索；“未解决我们的组合问题”也不等于组合本身有创新。下面将原文事实与本项目判断分开。阅读覆盖和版本见第二表。

| 文献 | 原文问题、输入/监督、机制与输出 | 理论、实验 | 与本项目的实际重合边界 |
|---|---|---|---|
| **ACA / ACANet** | 分子图与性质标签；任务回归损失加活性差异定义的 soft-margin triplet loss，并挖掘违反约束的三元组；输出性质预测与表征。ACA 是损失/归纳偏置，ACANet 是框架，不能算两条独立前作。 | 正文报告 52 个性质/配对任务、4 类 GNN；已读方法未给 shift 下类别条件覆盖定理。 | “让活性不相似分子在表征空间分离”“hard-triplet mining”直接重合。固定复核预算下的稀有类漏失不是它的目标。**更正：基础 ACA 的候选三元组按活性差异选取；结构相似性门控是另行比较的 gated 变体，不能把基础 ACA 描述成固定 Tanimoto 门控。** |
| **LAC，ICLR 2025** | 分子、性质标签和 matched molecular pair（MMP）图；节点损失对 cliff 分子加权，逐步聚焦大损失节点；边损失为 \(-(y_i-y_j)(\hat y_i-\hat y_j)\)，再进行困难边课程学习；输出性质预测。 | Prop. 4.1 是边损失梯度的代数化简，不是 OOD 有效性保证。8 个 MoleculeNet 分类集和5个 ChEMBL 回归任务，报告 ROC-AUC/MAE。 | “MMP 关系图 + 难 cliff 节点/边课程学习”是直接重合；它没有因此覆盖所有化学关系监督。完整固定边损失可化为 cliff 度数加权；再提出类似图边损失必须检查是否也只是在重加权。 |
| **ConfTr，ICLR 2022** | 带标签样本；每个训练批次分为校准/预测两部分，平滑分位数与集合成员指示；优化集合大小及可配置损失矩阵，允许类别加权集合成本；输出分类器，训练后重新独立校准。 | 校准阶段继承相应 CP 假设下保证；训练不会产生任意 shift 下最差类别保证。WineQuality 少数类及图像任务展示类别效率和集合组成调整。 | “把 rare-class 代价写进可微 conformal 集合训练”已被明确涵盖，不只是沾边。固定总复核名额可能是更具体的决策问题，但仅换损失函数名称或类别权重不足以形成机制贡献。 |
| **CSCT / RWCE，AAAI 2026** | 普通带标签分类；以真实类别在模型分数中的排名乘交叉熵，排名停止梯度；输出分类器，再配合 HPS/APS/RAPS/SAPS 校准。 | §4.2 给出排名、预期集合大小与泛化界；§5 测试 CIFAR-100、Caltech101、iNaturalist、SST-5。 | 标题里的 cost-sensitive 指排名困难度，并不是化学漏失代价或固定复核成本。它直接覆盖“rank-weighted CE 缩小集合”，不能据标题排除所有决策成本训练。其界不等于 shift 下类别风险约束。 |
| **DrugOOD，AAAI 2023；此处读取 2022 扩展预印本** | 分子/靶点与亲和力记录；按 assay/scaffold/size 等定义域，整理成活性分类；已有 GroupDRO 基线，输出 OOD 分类预测。 | §4.2/Table 5 给出 GroupDRO 等域泛化实验；基准本身不提供本项目所需类别风险定理。§3.2 对删失值作固定偏移，对重复测量取平均。 | “按 scaffold 做 GroupDRO”不是新机制。该事实不能排除结合测量证据的局部关系学习；但仅采用原始重复数据也不算算法创新。还必须区别最差 scaffold 平均损失与稀有类决策漏失。 |

## 可回查的原文位置与版本

| 文献 | 精确身份与主要原文位置 | 本次访问范围 |
|---|---|---|
| ACA / ACANet | Cui 等，*Activity-cliff awareness enables robust graph learning for molecular property prediction*，Nature Communications 17, 9376 (2026)，[DOI 10.1038/s41467-026-75713-2](https://www.nature.com/articles/s41467-026-75713-2)。Fig. 1；Methods → “Activity cliff awareness loss function”；Discussion / Supplementary Note 1 所指的 gated ACA 比较。旧预印本题为 *Activity Cliff-Informed Contrastive Learning for Molecular Property Prediction*。 | 最终版 PMC/出版社索引 HTML 的摘要、结果、讨论与方法相关段落已读；直接最终 PDF 与补充材料未取得。对补充实验仅按正文说明报告，**未声称已核对补充文件全文**。公式编号未可靠取得，故不编造。 |
| LAC | Yang、Yao、Kwok，*Curriculum-aware Training for Discriminating Molecular Property Prediction Models*，[ICLR 2025 官方全文](https://proceedings.iclr.cc/paper_files/paper/2025/file/285cf10c8c6153d66b8cd6a3ab0d69ce-Paper-Conference.pdf)。§3 Def. 3.1–3.2；§4.1–4.4，pp. 5–7，Eq. (1)–(4)，Prop. 4.1，Algorithm 1；§5.1–5.3。 | 官方及 OpenReview PDF 的网页解析全文可读；方法、命题及主实验已核对，未逐条复核补充证明。无可靠 arXiv 编号，不填写猜测编号。 |
| ConfTr | Stutz 等，*Learning Optimal Conformal Classifiers*，[arXiv:2110.09192v3](https://arxiv.org/abs/2110.09192v3)，ICLR 2022。§2.2；§3.1–3.3，pp. 4–6，Eq. (3)–(8)；§4.2、Fig. 5；Appendix O。 | 下载并读取 v3 PDF 和 HTML；已核对训练目标、类别权重、训练后重校准及 minority-class 实验。PDF SHA-256 见 JSON。 |
| CSCT | Jia 等，*Cost-Sensitive Conformal Training with Provably Controllable Learning Bounds*，[AAAI 官方记录](https://ojs.aaai.org/index.php/AAAI/article/view/39384)，[arXiv:2511.17861v1](https://arxiv.org/abs/2511.17861v1)。§4.1 Eq. (4)–(5)、Algorithm 1；§4.2 Theorem 1–2、Corollary 1；§5.1。 | AAAI 出版身份经官方索引核验；完整阅读的是 v1 预印本，未逐字比对最终版。特别是 Eq. (6) 的加项含 \(K(1-\alpha+1/(n+1))\)，不能只把它转述成 \(O(K/n)\) 校准误差。 |
| DrugOOD | Ji 等，*DrugOOD: Out-of-Distribution (OOD) Dataset Curator and Benchmark for AI-aided Drug Discovery – A Focus on Affinity Prediction Problems with Noise Annotations*，[arXiv:2201.09637v1](https://arxiv.org/abs/2201.09637v1)，后发表于 AAAI 2023，[项目站](https://drug.ai.tencent.com/drugood/)。扩展稿 §3.2–3.4，pp. 14–15；§4.2、Table 5，pp. 19–20。 | 完整下载 v1 扩展稿并核对上述部分；此处页码属于扩展稿，不能冒充 AAAI 精简版页码。 |

## 必须修改的排除结论

1. **撤销大范围排除：**“cliff-aware loss 已有 → 化学局部训练都没有创新”；“scaffold GroupDRO 已有 → 所有 shift-aware 训练都没空间”；“cost-sensitive 在标题出现 → 固定复核成本问题已完整解决”。这些推理跨度过大。
2. **保留窄范围排除：**普通活性 triplet 加回归、MMP 困难边课程、按 scaffold 重加权、按真实类别排名加权 CE、在 ConfTr 里提高少数类权重，都不足以单独作为新核心机制。
3. **优先补齐最接近前作：**若继续做“直接训练 conformal 输出”，ConfTr 必须作为机制层比较对象；ACA/LAC 与 DrugOOD 不能替代它。比较应分开写优化对象、监督信息、输出决策和有效性条件，而非只列应用领域。

## 最多两个可继续推敲的机制候选（均未证实新颖）

**A. 用测量证据决定 cliff 约束，而不是用困难程度决定权重。** 将同一 assay 内的删失区间/重复测量保留为性质证据；只有证据支持活性顺序的分子对才施加定向分离，证据冲突则采用部分顺序/不确定约束。化学假设是：困难 cliff 挖掘可能把测量冲突当成学习重点。与 ACA/LAC 的区别须落在“约束何时成立、如何处理不确定关系”的数学对象，而不是换一组权重。还需系统核验 censored regression、uncertain ranking、noise-aware contrastive 的近邻工作；本次五篇不足以认证首创。需要用公式或小型构造说明所提目标在退化情况下与既有损失的关系，避免只换包装。

**B. 从真实复核决策反推训练，而不是把集合大小当成本替身。** 明确定义自动判定、送复核和名额约束，训练优化整个批次的稀有类漏失—复核分配，而非逐样本加权集合大小。可能的区别是受总预算耦合的样本间决策及其对化学局部关系的梯度；应检查其策略族、约束、信息与优化是否都等价于 ConfTr 的既有损失；仅出现拉格朗日形式不足以判定等价。若这些要素均可由既有方法直接复现，则须另找实质增量。需要与 selective classification / learning to defer / decision-focused learning 另行核验；本次不能断言它们未做。独立校准决定有效性，训练目标本身不能宣称恢复 shift 下条件覆盖。

就当前证据，A 的化学依据更具体；B 与现有通用决策学习的潜在重合更强。两者可先用公式、小型构造或轻量原型说明差异，再做匹配比较；方法贡献与可靠实证可以成立，只有主张新保证时才需要相应证明。不建议先扩大现有实验。
