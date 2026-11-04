# CHEMOLAB-D-26-01118：实质修订与逐条回复工作稿

日期：2026-11-04。旧投稿版本：`d2e05ad1b948ea8c19b8d67783a659a385c54240`。

这是实验前的修订说明和回复底稿，**不能直接作为已完成的 response letter 提交**。新代码的存在、单元测试通过、真实实验完成、科学主张得到支持，是四种不同状态；此处不把它们混为一谈。`manuscript.tex` 已重写研究问题、数学定义、对照设计和讨论，但新增实验的结果与结论明确标为 pending。

## 一、编辑信的含义与本轮定位

本次决定是拒稿，允许 substantially revised version 作为新稿投稿，不是原稿的大修决定，也不承诺重投后送审或录用。重投 cover letter 必须注明原稿号并具体说明实质变化。第二位审稿人长期未交报告解释了处理延迟；不能据此推断现有审稿意见无效或结果应该被推翻。

建议保留“拒稿重投同刊”的方向，但把提交条件设为完成下面的公平比较和结果裁定。仅换标题、压缩文字、增加审计叙事不足以回应这份意见。

新标题工作稿：

> Class-Specific Errors and Deferral under Molecular Distribution Shift: A Budget-Controlled Comparison of Conformal and Abstention Rules

该标题把可检验的问题放在前面。TAME 将作为比较对象和待检验的经验性扩张规则，而不是预先宣布获胜的新原理。

## 二、不能忽略的原仓库证据

1. 原代码不只有普通 Mondrian：已包含 ECFP、physicochemical、score-view 三种 domain-logistic weighted Mondrian，以及 equal-logit 普通 Mondrian。真正不足是主要推断仅围绕 stacked Mondrian，并且缺少完整 CoDrug/KMM-CP 与同预算非 conformal 规则。回复不能错误承认“原研究根本没有任何 weighted comparator”。
2. 原 Stage I 和 Stage II 的二分类主分数都是 LAC。改变的是端点、基模型组合和第二阶段的 stack。回归分数当然不同。因此应澄清具体差异，不承认并未发生的二分类评分规则变更。
3. `final_summary_all_endpoints.csv` 的十二端点描述性结果中，equal-logit 普通 Mondrian 的 mean minimum-class coverage 为 0.844119，MacroCSY 为 0.414430；TAME 为 0.842958、0.413293。equal-logit 的 ambiguity 也更低（0.495739 vs 0.504006）。这不是显著性结论，也不是原六端点主要比较，但说明必须把简单基线放回正文，并重新做配对比较。
4. 原 Stage I 的 `13_mondrian_conformal_analysis.py` 将有限样本 rank 截到 n，极小类别校准样本下应为正无穷。Stage II 的 `racer_c4_core.py` 已正确处理。修订应统计受影响单元，单独输出修正结果，保留历史版本。
5. 原 EPA 数据已经打开标签，任何本轮新增方法和新种子结果都属于回顾性分析。不能再次称 sealed/prospective/independent validation。

## 三、逐条评论、修订动作与回复底稿

### 评论 1：边际覆盖与类别覆盖的差异是已知性质，ClinTox 案例不构成新原理

**裁定：核心成立。** 原文已经引用相关理论，但叙事仍把已知性质的实例化放在贡献中心。新稿把它作为背景，不再把“总体 90% 却少数类失效”本身列为方法创新。

**已完成的文字修改：** 新 Introduction 前三段明确承认既有理论；摘要和标题改为 class-specific error / deferral；原四端点审计转为背景及补充材料，主线收拢为 assay activity 分类。

**实验仍需完成：** Stage I 有限样本分位数影响审计与重算。修订前不把旧 ClinTox 精确数字作为新版算法的核验结果。

**英文回复底稿：**

> We agree that the distinction between marginal and class-conditional coverage is established in conformal prediction. The revised introduction treats this distinction as background rather than a new principle. We have reframed the study around a different empirical question: whether transport-based set expansion targets class-specific decision errors more effectively than simpler rules with the same deferral budget. The earlier molecular examples are retained as historical motivation and supplementary analyses. We also identified a finite-sample quantile issue in the Stage I implementation and are auditing the affected calibration groups; corrected results will be reported separately from the archived analysis.

完成重算后，把最后一句替换为具体受影响数量、修正范围和表号；现在不可写成“没有影响”。

### 评论 2：TAME 是 Mondrian 的保守 wrapper；没有新 validity result；审计门不能恢复条件覆盖

**裁定：成立，但要精确。** 旧正文已否认任意 shift guarantee 和 conditional guarantee，不能回信说原稿“错误证明了新定理”。真正要改的是贡献价值与证据强度。包含关系带来的经验覆盖不下降是构造性质，不是足够的创新证据。

**已完成的文字修改：** 新稿只称 empirical set-expansion rule；用一个简短公式说明 weighted quantile 的 test mass at infinity；删去证书有效性、promotion 成功即贡献成立的叙事。审计指标改称 diagnostics / gates。

**必须完成的比较：**

| 问题 | 比较 | 对应可下结论 |
|---|---|---|
| 是否只是多拒判 | 同 baseline、同新增拒判数、同原 singleton 类别组成的 omitted-label probability（主对照），及 random / uncertainty / AD（次要） | 是否有更好的错误定位 |
| 双视图是否必要 | 每个单视图、双视图 union、双视图 consensus | 多视图的增量价值 |
| 门槛是否必要 | 原 gates、全部取消、逐项取消/单因素敏感性 | gates 是否带来实证收益 |
| 是否只保护了容易改善的一类 | label 0、label 1、both | 任务目标与干预是否一致 |
| 是否依赖一个弱基模型 | logistic、stack、equal-logit，保持 uncertainty 层相同 | 基模型依赖性 |

预算必须分别匹配原 baseline singleton 0 和 singleton 1 的扩张数量。只匹配总拒判数仍可能混入预测类别组成差异。所有排序只能使用概率、结构、无标签密度信息及预设随机种子。

**明确主对照：** `matched_label1_active_omitted_label_probability`。对于原 `{0}`，按 `p1` 从高到低扩张；对于原 `{1}`，按 `1-p1` 从高到低扩张。Mondrian singleton 不一定等于概率 argmax，所以不能只用离 0.5 最近或 entropy 最大的样本作为最强简单对照。主比较固定为 stack、LAC、alpha=0.10、历史六端点，比较 active-label TAME 与上述规则的 false-inactive exposure；TAME 减去主对照的差值小于零才表示更好。uncertainty、AD、random 为次要对照。

**汇总顺序：** 每个 endpoint/seed 先分别计算指标，再对声明的端点和 seed 等权平均。不得先平均不同 seed 的概率或集合，再阈值化构造一套新的预测。Bootstrap 以 molecule/structure 为跨端点共享重采样单元，保留该分子的全部 seed 预测，再重新计算相同的 cell 指标与等权均值。

**英文回复底稿：**

> We agree that TAME is an empirical wrapper around Mondrian prediction and does not supply a new coverage theorem. The original manuscript stated this limitation, but its contribution was insufficiently separated from the deterministic benefit of set expansion. We have rewritten the algorithm description and replaced the main evaluation question with a budget-controlled comparison. The new primary comparison matches the numbers of expanded baseline class-0 and class-1 singletons and ranks the probability of the omitted label, with uncertainty, chemical-similarity, and random selection as secondary rules. The primary contrast uses the stacked predictor, LAC scores, alpha 0.10, and the historical six-assay panel to compare false-inactive exposure. This decision-specific simple comparator is needed because a Mondrian singleton need not equal the probability argmax. Gate-removal and single-view analyses will quantify the contribution of the transport components. Passing a diagnostic gate is no longer described as evidence that class-conditional coverage has been restored.

最终回复必须将 “will quantify” 换成真实结果与表号；若没有增益，应直接报告。

### 评论 3：化学动机薄弱；Stage I 强调稀有毒性阳性，Stage II 却保护 label 0

**裁定：成立。** 二者是不同的决策错误。原 TAME 的 0 类保护主要纠正 inactive 被错误判成 active 的 singleton；不能将其描述为防止 active 被清除，更不能把 Tox21 active 等同 clinical toxic。

**已完成的文字修改：** 每个 endpoint 使用 activity/inactivity 的原始标签语义；定义 `{0}` inactive decision、`{1}` active decision、双标签 defer。原 label-0 TAME 保留为历史算法，新的 label-1 变体作为独立、明确的 active-protection 规则。

**必须输出的指标：**

| 指标 | 公式 | 意义 |
|---|---|---|
| False-inactive exposure | P(S={0} \| Y=1) | 实际 active 中被作 inactive 决策的比例 |
| Active fraction among inactive decisions | P(Y=1 \| S={0}) | 已作 inactive 决策的样本里有多少是 active |
| False-active exposure | P(S={1} \| Y=0) | 实际 inactive 中被错误作 active 决策的比例 |
| Inactive-decision yield | P(S={0}) | 有多少样本得到 inactive 决策 |
| Deferral | P(非 singleton) | 后续需要证据的比例；同时另报原始空集 |
| 两类 coverage / CSY | 分别按 Y=0,1 计算 | 防止一个类的改善掩盖另一类损失 |

没有对应分母时报告 undefined 和样本数，不填 0。

**英文回复底稿：**

> We agree that the previous narrative did not align the Stage I motivation with the Stage II protected label. Label 0 was selected from a development coverage deficit and addressed erroneous active decisions for inactive compounds. It was not an assay-active protection rule. The revised manuscript states this explicitly and separates the historical label-0 method from a label-1 variant motivated by avoiding inactive decisions for assay-active compounds. We report false-inactive and false-active exposure, decision yield, and deferral together. We also replace generic toxicity language with the measured Tox21 assay-activity interpretation.

### 评论 4：EPA +1.36 pp 小、依赖协议、强方法未在统一协议比较

**裁定：成立。** 改进是否有价值取决于它与什么相比、拒判成本是多少、是否在不同端点/基模型/设置稳定，不能只用“CI > 0”结束讨论。

**已完成的文字修改：** 新稿保留旧六端点效应为 historical finding；并列给出十二端点所有旧方法的真实描述性表，明确 equal-logit 两个指标略高于 TAME。原 -5 pp 容忍线不再被称为 formal non-inferiority 或临床效用标准。

**必须完成的实验：**

1. 在统一输入、校准、测试行、alpha 下完成 LAC/APS、普通/weighted Mondrian、TAME 变体与 matched deferral。
2. 按完整方法而非名称复现 CoDrug 与 KMM-CP；官方实现/忠实复现先做小样本检查。CoDrug 包含模型/energy regularization 与 KDE，不能把普通 domain logistic 或单纯 KDE 改名为 CoDrug。KMM-CP 也不能仅凭一个 QP 就宣布完整复现。
3. 组件受控比较与完整 end-to-end 比较分开报告；不同表示学习带来的差别不能归给 uncertainty wrapper。
4. 增加六端点和十二端点的 assay-level 配对效果、所有端点 sample counts、同预算曲线和置信区间。
5. 重做统计时用统一 molecule/structure cluster 跨 endpoint 共享重采样权重，并保留同分子全部 seed 结果；不要把 5 个 seed 当 5 倍生物样本。

**英文回复底稿：**

> We agree that the original primary contrast against stacked Mondrian prediction was too narrow to establish practical advantage. The previous implementation also contained three domain-ratio-weighted Mondrian comparators and an equal-logit Mondrian comparator, but these were not integrated adequately into the main inference. In fact, the archived twelve-assay descriptive summary gives slightly better minimum-class coverage and macro correct-singleton yield for equal-logit Mondrian than for TAME. We now disclose this comparison and retain it in the revised analysis. The revised suite adds matched abstention rules, score and predictor sensitivities, and verified reproductions of directly relevant shift-aware methods. The new EPA analyses will be identified as retrospective, since these labels were already accessed in the original study.

最后版本只有完成完整 CoDrug/KMM-CP 后才能将 “adds ... verified reproductions” 写成已完成；未完成时保留 pending，不能提交一封声称全做完的信。

### 评论 5：算法被 firewall、hashing、promotion 叙事掩盖；阶段切换不清；六门槛不可迁移

**裁定：主要成立。** 复现记录有必要，但不能替代简明算法与化学问题。标签隔离的历史事实保留在补充材料，不再充当主要方法贡献。

**已完成的文字修改：** 新稿 Methods 按 数据→共同预测→score/quantile→weights→protected class→matched budget→metrics 排列；算法的核心是集合扩张与对照。原稿两个 classification stage 都用 LAC 的事实被保留，模型/数据角色差异明确说明。gate 只被称为 study-specific settings。

**实验仍需完成：** 跨阈值/视图/模型分析，报告启用率、fallback 率、类别错误与拒判成本。不凭同一个 EPA cohort 的通过率宣称 transferable。

**英文回复底稿：**

> We have reorganized the methods around the prediction problem, shared base predictions, candidate-label scores, set construction, and decision metrics. Hashes and historical label-access records are retained in the reproducibility material rather than used as the main argument for methodological contribution. Both classification stages used LAC scores; the principal stage differences were the endpoint collections and base-model construction, which are now made explicit. In the revised controlled comparison, predictors and scores are held fixed across uncertainty layers, with separate sensitivity analyses. The original gate values are presented as study-specific choices whose empirical effects must be measured, not transferable certificates.

## 四、投稿前必须完成的结果裁定

可交给二作的是可运行研究包，不是预先规定“必须跑出正结果”的任务。完成全矩阵后按证据决定论文主张：

| 实际结果 | 可采用的论文表述 |
|---|---|
| TAME 在同预算、同预测类组成下稳定优于强简单规则，并有强 CP 方法比较支持 | 可以主张在明确 assay/模型/设置内存在额外错误定位价值；不升级为普适定理 |
| TAME 只优于较少拒判的 baseline，匹配预算后优势消失 | 主要报告覆盖增长由拒判解释，不写算法优越性 |
| 简单 equal-logit/uncertainty/AD 更优 | 将其作为实质结论，论文转为经验比较和复杂度必要性评估 |
| 不同 endpoint/模型方向冲突 | 报告条件性结果，禁止挑最有利的 endpoint 充当总体结论 |
| 完整 CoDrug/KMM-CP 尚未完成 | 强基线缺口仍未关闭，不能声称审稿主要问题全部解决 |

现有 `manuscript.tex` 的 pending 段落必须在真实结果完成后替换；作者声明须由两位作者重新确认。编辑要求的 cover letter 应写明原稿号、上述研究问题重构、实际新增的实验与结果，而不使用“我们已全面解决所有问题”这类未经证实的总括句。

## 五、编译与文件关系

在本目录运行：

```bash
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
```

参考文献为本文件内的精简 `thebibliography`，无需 bibtex；正文与历史表数字均可直接编译。旧的 `paper2_latex/main.tex` 和投稿 PDF 保持原状以供追溯。新 PDF 仅用于审阅这一实验前工作稿，不应直接上传期刊。
