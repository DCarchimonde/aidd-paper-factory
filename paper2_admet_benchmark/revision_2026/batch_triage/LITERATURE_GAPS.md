# CHEMOLAB 新方向文献核查与技术候选（2026-11-08）

本报告只提出可检验的增量，不宣称完成了穷尽性 novelty 检索。检索使用 primary papers / proceedings / author pages，覆盖检索时可得的 2026 年工作；新预印本与已发表论文分开标明。未改仓库代码。

## 结论与建议

建议主线是：**固定分子库、双类条件错误约束、硬性实验预算下，基于随机目标域审核的可认证三动作决策**。三动作是按预先定义的实验终点判为 0、判为 1、或做真实终点测定。研究对象从预测集合大小改成整个待筛分子库的决策错误与实验消耗。

最可辩护的技术贡献不是“又一个 CP 包装器”，而是：一个共享的分层阳性计数置信集合同时约束所有候选决策，保留错误分子数与真实类别分母之间的依赖；将此集合转换成可精确求解的 0/1 优化；再按**预期减少多少已认证的剩余实验需求**分配审核标签。单纯组合 class-conditional CP、成本矩阵和 active labeling 已不足以构成新方法。

这仍然是**待证明与待实证验证的新方向**。如果它只比很弱的随机审核好，而不优于同样拥有目标标签的 class-specific LTT / PAC-labeling 改编，不能据此宣称方法创新。原 TAME 的 +1.36 pp 不能作为此方向的支持证据。

## 最近邻：必须正面承认的已有结果

| Primary source | 已有贡献；我们不能再声称首次 | 与建议主线的准确边界 |
|---|---|---|
| **CARED**，Zhang, Huang, Jin，2026-10-29 预印本：[arXiv](https://arxiv.org/abs/2610.40252)，[PDF](https://arxiv.org/pdf/2610.40252)，[作者列表页](https://ying531.github.io/publication_year/) | 随机审核、任意分布漂移、逐类误差、自动化效用已经组合。§3.3 Theorem 3.3 / Corollary 3.4 给累计误差的几乎必然长期控制；类别累计样本数须发散。 | Discussion p.17 明确将有限样本 realized-error confidence bounds 列为未来方向。我们需要给**单个有限库、任意停止时点**的同时高概率证书及硬预算可行性；不是长期平均补偿。 |
| **Probably Approximately Correct Labels**，Candès, Ilyas, Zrnic，ICML 2026：[终版](https://proceedings.mlr.press/v306/candes26a.html)，[v1 全文](https://arxiv.org/html/2506.10908v1)，[官方代码](https://github.com/tijana-zrnic/pac-labels/) | 已将数据视为固定，随机性只来自审核；同时具有自动标签、专家回退、成本敏感多模型路由和 uncertainty multicalibration。§2 Algorithm 1 用不确定度阈值控制全库平均损失，单调性避免逐阈值多重修正。 | 不能宣称首个 fixed-pool 或 label-cost 方法。目标是**共享未知类计数分母的双风险**、非单链动作选择和 certifier-directed acquisition。必须做 PAC-labeling 的合理类风险改编，不能只比其原始全局错误指标。 |
| **Active, anytime-valid risk controlling prediction sets**，Xu, Karampatziakis, Mineiro，NeurIPS 2024：[arXiv](https://arxiv.org/abs/2406.10490)，[全文](https://arxiv.org/html/2406.10490v2) | 已有主动标签、anytime 风险保证、预测辅助降方差、预算下最优标签策略。§2.1 的查询是随到达 X 的 Bernoulli query，§3 优化 e-process 增长。 | 不可将“主动审核 + anytime”当增量。我们的数据是固定库中的自适应分层无放回取样；优化的是双类风险下的实际 assay savings，非单一嵌套风险参数的财富增长。 |
| **SCoRE: Conformal Selective Prediction with General Risk Control**，Bai, Jin，2026 预印本：[arXiv](https://arxiv.org/abs/2603.24704)，[全文](https://arxiv.org/html/2603.24704v1) | 已处理一般 bounded risk、选择性部署、covariate shift；§7.1 已有 ADMET/药物发现、合成可及性成本和多样性。附录 A.4–A.5 还有 doubly robust calibration。 | “化学成本 + 风险控制”也不是空白。它的 MDR/SDR 与全库真实阳性分母的 FNR 不同；我们的有限库审核不依赖 source-to-target 密度比。必须明确 estimand，不能以指标改名假装击败它。 |
| **Conformal Selection for Efficient and Accurate Compound Screening in Drug Discovery**，Bai et al.，JCIM 2025：[出版社](https://pubs.acs.org/doi/10.1021/acs.jcim.5c01633)，[PubMed](https://pubmed.ncbi.nlm.nih.gov/41363967/) | 分子筛选中基于 conformal p-values 控制 false discovery / omission rates，且做真实数据模拟。 | False omission rate 的分母是判阴集合，FNR 分母是真实阳性集合；两者不可混用。没有资格声称此前药物筛选只看 marginal coverage。 |
| **A Joint Finite-Sample Certificate for Adaptive Selective Conformal Risk Control**，Yu, Liu，2026 预印本：[arXiv](https://arxiv.org/abs/2606.08517)，[PDF](https://arxiv.org/pdf/2606.08517) | 已联合证明 selected risk、acceptance floor 和 utility 的有限样本证书；有限阈值网格可自适应选择，并明确返回 INFEASIBLE。原文 p.2 假定固定分布的 i.i.d. 校准数据。 | “联合证书 + 允许不可行”不新。我们的区别只能在 fixed-population randomization、两个未知真类分母、硬实验容量和重复审核决策。 |
| **Confidence sequences for sampling without replacement**，Waudby-Smith, Ramdas，NeurIPS 2020：[会议页](https://papers.nips.cc/paper_files/paper/2020/hash/e96c7de8f6390b1e6c71556e4e0a4959-Abstract.html)，[PDF](https://proceedings.neurips.cc/paper_files/paper/2020/file/e96c7de8f6390b1e6c71556e4e0a4959-Paper.pdf) | 无放回有限总体的 anytime confidence sequence 与超几何计数推断已有完整理论。 | 直接引用并实现这个统计部件；不要命名为新定理或新分布无关原理。新增命题应解释如何推到任意数据自适应三动作策略及联合优化。 |

特别注意：CARED 的网页接口此次受限，但原始 arXiv PDF 已直接下载并核对。`cared_2610.40252.pdf` / `.txt` 位于本报告同目录。Joint Certificate PDF 也已下载核对；这些不是只凭二手摘要得到的判断。

## 候选 1（推荐主线）：共享计数证书的有限库双风险三动作决策

### 定义需要做对

固定库中 `Y=1` 表示该实验终点阳性，而不是泛化的“临床有毒”。只有当终点化学含义支持时，才把自动判 0 称为放行、判 1 称为拦截。hERG、AMES 与 Tox21 的含义应分别写清，不能把 BBBP、binding activity 的正类一律当伤害。

将冻结模型的分数与 label-free support 信息划为 H 个固定 strata。审核前冻结这些分层。每层在当前时刻还有 n_h 个未测分子，其真实阳性数 K_h 未知；累计已测阳性和阴性分别为 O+ 与 O−。无放回随机审核得到同时、anytime 的计数界 `L_h ≤ K_h ≤ U_h`。动作对层内剩余分子统一执行，以保持下面的计数证书成立。

设 z0_h 表示自动判 0，z1_h 表示自动判 1，zA_h 表示测定，三者每层和为 1。记录 assay 结果后可认为相对于该测定标签没有模型判别错误；这不等于假定实验标签没有生物或测量误差。

总体真实类别分母是：

- `N+ = O+ + Σ K_h`
- `N− = O− + Σ(n_h−K_h)`

相对于整个库的类条件错误是：

- `FNR = Σ z0_h K_h / N+`
- `FPR = Σ z1_h (n_h−K_h) / N−`

没有某类时需预先定义指标约定，并始终报告类别计数范围；不能用一个空分母制造“验证成功”。

### 可证明的技术核心

在同一个计数矩形上，约束 `FNR≤α` 的最坏情况有精确线性形式：

`Σ z0_h[(1−α)U_h + αL_h] − αΣL_h − αO+ ≤ 0`。

约束 `FPR≤β` 的最坏情况同样为：

`Σ z1_h[(1−β)(n_h−L_h) + β(n_h−U_h)] − βΣ(n_h−U_h) − βO− ≤ 0`。

因此可以最小化 `已审核数 + Σ zA_h n_h`，或在总预算 B 下最大化已认证的自动化人数。未知分子既影响误差分子，也影响真实类别分母；上式保留这一耦合，比先独立上界 numerator、下界 denominator 再相除更紧。审核高阳性层即使不直接改变低分区的误差，也可能通过 O+ 提供有效分母信息。

只要 count event 同时覆盖所有层和所有审核次数，**任何依据已审核标签选择、扩张或收缩的动作方案**都可在此事件上验证；不需要对 `3^H` 个策略逐个 Bonferroni。这个推论及 exact robust counterpart 是可以诚实陈述的增量。计数 CS、随机审核或 MILP 本身都不是新发明。

严格限制：保证针对这一固定库及测定标签，不自动推到未来化学批次；分层不能在看见目标标签后任意重建；层内不能再按分数挑部分放行却沿用整层计数界，除非预先把这些子集合也纳入被认证的计数结构。

### 可行性与必须比较的基线

现有 repo 的 probability、standardized structure ID、source similarity / embedding、缓存的模型预测就足够，不必重新训练神经网络。新增部分是 oracle reveal ledger、WOR count CS、MILP 与预算评估。老 EPA 等已被反复看过的数据必须标 retrospective，不能重新称 sealed。

同预算基线至少包括：随机审核 + 同一 certifier；分层比例审核 + 同一 certifier；简单两阈值 + 类条件 LTT；PACL 按两个类风险合理改编；单一全局错误 PAC-labeling（作为指标差异演示）；直接概率排序三动作；CoDrug/KMM 集合转三动作且计入全部退回测定成本。无标签方法与有标签方法不能只按相同预测集合大小比较。

核心检验是：同一 `α,β,δ,B` 下的 certified-feasible rate、总测定数、实际双类错误、双类约束违反事件频率、自动化数量，以及判不可行时的最低已认证测定需求。需用重复随机审核来核对失败频率，不把 bootstrap CI 当作理论有效性的证明。

## 候选 2（适合作为主线技术增量）：面向认证成本的自适应审核

最近邻是 Active RCPS 的最优 query policy、PACL 的 uncertainty-based audit、PPAT 的 residual-variance acquisition；仅用 entropy、罕见类别概率或 disagreement 采样不新。

建议定义当前状态的最小已认证总实验数 `C*(state)`，先不施加 B，因为测定全部余下分子总是一个安全解。对于每层 h，分别模拟下一次审核得到 0 或 1，更新它的 count CS、O+/O− 和剩余人数，并重解 MILP。选择预期 `C*` 降幅最大的层：

`gain_h = C*(now) − [p_h C*(next_h,1) + (1−p_h)C*(next_h,0)]`。

`C*` 已含新增审核成本，不能再把这一个标签免费计算。p_h 可由冻结分数与已审核结果构造平滑估计；它只决定效率，不进入真实性证书。选好层后，必须从该层余下分子中均匀无放回抽样。这样可以主动寻找分母信息，而非仅审核当前拟放行边界。

可贡献之处是针对两类风险、硬容量和三动作优化的 value of information，而不是“主动学习”四个字。最小消融：随机、entropy、Neyman allocation、boundary-only、该成本下降规则。单步离散界可能导致所有 gain 为零，须预先固定 batch lookahead 或轮换探索 fallback；不要看全库标签后调规则。暂不声称 query policy 全局最优或有 regret 保证；它是需要证明近似性质或用严谨实验支持的算法增量。

PPAT primary：[arXiv 2607.08347](https://arxiv.org/abs/2607.08347)。它将 LURE 与预测控制变量结合，并研究方差导向获取；其主要区间结果是渐近有效，不等于这里的有限总体同时证书。

## 候选 3（更大但更慢）：多个安全终点共享实验预算

对同一分子库的多个毒理终点共同规划 assay allocation，控制各终点有害类漏失，同时计入多终点板式实验的共享开销。技术目标应是 endpoint × stratum 的计数区域与 assay-action 成本结构形成可求解的 robust program，以及跨终点获取价值，而不是简单把多个 CRC 阈值放在一起。

最近邻包括 CRC 的 multiple-risk 扩展、Data-Adaptive Tradeoffs among Multiple Risks、SCoRE 的一般成本/多样性以及既有 multi-assay active learning。因此“多风险”并不新；真正可能有价值的是**一个实验动作同时揭示多个终点、共享预算、存在缺失 assay 标签时的可认证决策**。

可行性：现有 Tox21 12 assays 有结构共享，适合研究；但旧表存在天然缺失。模拟 oracle 只能揭示真实已有记录，不能伪造缺失标签。需要预先限制到完整可审核子库，或明确建模不同实验面板可揭示哪些终点。真实实验成本若未公开，应给规范化 cost scenarios，不将任意权重写成真实费用。此方向工作量和假设成本都高，适合作为主线成立后的扩展。

Primary：[CRC ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html)；[Data-Adaptive Tradeoffs, arXiv 2403.19605](https://arxiv.org/abs/2403.19605)。后者已有 data-adaptive risk/tradeoff parameters 的有效控制，不能遗漏。

## 候选 4（不推荐独立立项）：少量 target labels + doubly robust / prediction-powered class calibration

这个方向容易实现，但近期已有成果极密集。Semi-Supervised RCPS 已用 PPI 改善小校准集；SCoRE 附录已有 DR shift correction；PPAT 已主动 residualization；Active RCPS 已联合预测辅助与主动标签；ReCIRC 已做局部风险曲线以改善组风险。因此将 source predictions、目标标签和 density ratio 合成一个新的 estimator，若没有明确新理论或显著效率优势，很容易重蹈“已知估计器周围的工程包装”。

可以把 PPI / DR 当候选 1–2 的效率基线或未来扩展，不能先给它一个新缩写就作为创新点。

Primary：[Semi-Supervised Risk Control, TPAMI 2025](https://ieeexplore.ieee.org/document/11106221/)，[arXiv](https://arxiv.org/abs/2412.11174)；[ReCIRC, 2026-09-29](https://arxiv.org/abs/2609.38112)。

## 另外应引用、但不能误读的近邻

- [Provably Label-Efficient Conformal Prediction, ICML 2026](https://proceedings.mlr.press/v306/ilyas26a.html)：校准标签数与集合效率的停止规则已经研究；不是我们的硬单批预算问题，但排除了宽泛的 label-efficient 新颖性主张。
- [High Probability Risk Control Under Covariate Shift, COPA 2025](https://proceedings.mlr.press/v266/almeida25a.html)：importance-weighted LTT 已存在。不能把加密度比的 LTT 当新原理。
- [Risk-Controlled Post-Processing of Decision Policies, 2026](https://arxiv.org/abs/2605.06479)：已有保持原策略与风险约束间的最优后处理，以及安全 fallback。不能声称首个风险受控最少干预策略。
- [Know When to Abstain, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/3fe2a777282299ecb4f9e7ebb531f0ab-Abstract-Conference.html)：likelihood-ratio selective classification under shift 已存在。
- [KMM-CP, UAI 2026](https://proceedings.mlr.press/v337/laghuvarapu26a.html)：选择支持重叠区域并不新；现有完整复现可保留作零目标标签基线。

## 建议的论文主张与停止标准

可写的目标句：我们研究固定分子库中，如何用有限随机测定同时认证两种类条件错误，并在这一证书约束下最小化实验数量；提出共享计数区域、精确 robust 三动作优化及认证成本导向获取。

不要写：新的 CP validity principle；首次保护 minority class；任意分布漂移下对未来分子保证；无需目标标签却验证了 conditional shift；首个 active / cost-aware / class-conditional risk control。

继续条件：在至少多个独立 ADMET 终点和明确 shift 方案上，以相同目标标签预算优于合理改编的强方法，或者在相同证书强度下明显减少实验数；并有理论证明支撑单批联合有效性。若只能提高 marginal coverage、将更多分子送测、或大量设置始终不可行，则应报告为方法限制并收缩主张，不能再次把保守退回当成新方法成功。

## PAC Labels 官方实现与可执行比较入口（补充）

官方仓库：[tijana-zrnic/pac-labels](https://github.com/tijana-zrnic/pac-labels/)。MIT 许可。主要入口是 [pac_utils.py](https://github.com/tijana-zrnic/pac-labels/blob/main/pac_utils.py)，配套 `pac_labels_single_model.ipynb`、`calibration.ipynb`、`pac_router.ipynb`。无需训练新 backbone，现有 p1 足以产生硬预测及 uncertainty。

API：`pac_labeling(Y, Yhat, loss, epsilon, alpha, uncertainty, pi, num_draws, asymptotic=True)`，返回最终标签、已标注 mask、初始审核 unique fraction。非渐近比较必须显式设 `asymptotic=False`。

直接代码核查发现，它接收完整 Y，预计算全库 losses，并用 `max(losses)/min(pi)` 作为损失界。这是离线仿真接口，不能原样作为严格隐藏目标标签的 oracle 接口。独立复现应使用预先已知的 bounded loss 上限 `1/min(pi)`，仅在 queried index 上读取标签。另需核对无阈值通过时 `lb_index-1` 的负索引边界，以及 binary search 对数值上界单调性的要求；稳妥做法是按论文 first-crossing 定义扫描，明确无可证自动区时测定全部。

忠实的方法级基线可以用：预定 m 次有放回随机索引抽取 + Bernoulli 查询；固定或预先定义 pi；有限样本 meanUB；审核标签可复用但只计一次实际测定；求得阈值后将所有高 uncertainty 分子送测，计入同一总预算。如果总需求超过 B，应报告预算不可行，不能免费追加测定。PACL 允许一般有效 meanUB，因此独立实现在理论上可选明确的保守上界，但要命名所用版本，不冒称逐数值官方复现。

现阶段 proposed controls `joint_uniform`、`joint_entropy`、`joint_Neyman`、`threshold_lookahead`、`loose_ratio_lookahead`、`score_only_lookahead` 是有价值的机制消融；`threshold_lookahead` 使用本研究同一个 count CS，属于自行定义的阈值基线，**不是官方 PAC Labels**。在宣称优于最新方法之前，仍需加入论文级 PACL 复现或相应强改编，并把原始 total-error estimand 和类条件指标分开报告。

来源固定信息：本次查看的 PAC Labels 官方 main commit 为 `b415b58756b14b384529ac9cf146bd5d4c8139aa`；可固定读取 [pac_utils.py](https://github.com/tijana-zrnic/pac-labels/blob/b415b58756b14b384529ac9cf146bd5d4c8139aa/pac_utils.py)。下载 PDF SHA256：PACL ICML2026 `972635f199188fbba46e628f2e2d2f90c77dadb4cd4c1ac302dc68bfb741641a`；CARED `d8dbd5ff5595aaf6244b41a294e3d584822d049d8c601930631ca6cb46f71f0a`；Active RCPS `b97bfcd633c592ab0b766c80378b1edefd12c0d5c145fa750c767a7df4ce180b`；Joint Certificate `154886ef919296f5bfafb29abe7caa2b710ce0ec56e2df1f15390f19176b36d0`。
