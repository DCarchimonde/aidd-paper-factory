# CHEMOLAB 拒稿重投：二作执行计划与交接

版本：2026-11-04。稿件：CHEMOLAB-D-26-01118。目标：Chemometrics and Intelligent Laboratory Systems 的 substantially revised new submission。

**本轮交付是经过代码验证的修订实现、实验方案和实验前论文改写稿。不是已经完成新实验的可投稿终稿。** 原稿、封存结果和旧分支保持原样；本目录中的新方法、新对照与新统计不能借用旧结果当作自己的结果。

## 1. 先作决定：值得认真重投，但必须更换论证中心

编辑明确允许大幅修改后作为新稿提交，这提供了重新评审的入口，并不是有条件接收。此次应当重投的是实质改进后的研究，而非把同样的证据换一套措辞。

现有研究并非全无价值：已有数据处理、统一角色划分、分布偏移案例、加权比较器及完整历史记录。问题在于，原文用大量程序控制与审计语言支撑一个尚未充分证明实用优势的包络算法。修订将重点放在**在相同拒判代价下，哪个方法能够更少漏掉 assay-active 分子，以及复杂迁移修正什么时候有额外价值**。

研究范围收窄到 Tox21 分子测定的跨批次分布变化。label 1 是该测定 active，label 0 是 inactive；二者均不能直接解释为临床有毒/安全。前稿 ClinTox/BBBP 等第一阶段案例可以作为历史诊断，但“marginal coverage 不保证 class coverage”不能再写成我们的新原理。第一阶段回归任务移至补充背景，避免正文把四种不同问题硬接为一种普适 ADMET 方法。

## 2. 已经核实的关键问题

| 问题 | 核实结果 | 本次处理 |
|---|---|---|
| 边际与类别覆盖差异已知 | 审稿人判断成立 | 缩短为已有知识与问题动机，贡献落在决策比较和条件分析 |
| 加标签即可增加 coverage | 固定样本上由集合包含机械保证 | 新增严格预算及原 singleton 类别匹配的简单对照，不能只报覆盖上升 |
| 原 TAME 保护 label 0 | 与原文“毒性阳性稀有类”的动机不一致 | 保留历史版本；新增事先指定 active=1 保护，以及双类保护，分别报告 |
| 原文完全没有加权对照 | 这个表述不准确；旧仓库有 ECFP、理化和 score 三种 weighted Mondrian | 回复中说明已有对照，同时承认原主要统计只与普通 Mondrian 比，缺完整 CoDrug/KMM-CP 和公平拒判对照 |
| 简单集成可能更好 | 旧十二端点均值：equal-logit 最差类覆盖 0.844119、MacroCSY 0.414430；TAME 为 0.842958、0.413293 | equal-logit 列为必需模型；不能据均值判断显著，但须重做六主端点配对分析 |
| 第一阶段有限样本分位数 | 旧脚本将 rank 截为 n；n 很小时应使用正无穷 | 新核心统一正确实现；审计旧逐行文件，确认实际影响，不直接覆写历史表 |
| 原统计未保留跨端点同分子依赖 | 各 endpoint 独立抽分子，seed 又共享测试分子 | 新推断按 structure_id 跨端点、种子、方法共用重采样次数 |
| EPA 是否可重新称盲测 | 不可以，标签和结果已被检查 | 本轮全部 EPA 比较标为 retrospective；新 seeds 也不改变这一点 |
| “六个审计阈值”能否迁移 | 只能视为启发式参数，不能认证 covariate-shift 假设 | 消融与单因素敏感性；不按 EPA 表现重选阈值 |

完整路径、原始数字与限制见 `BASELINE_AUDIT.md`；文献与完整方法复现界限见 `LITERATURE_AND_COMPARATORS.md`。

## 3. 交付的代码与各自职责

| 文件 | 职责 |
|---|---|
| `core.py` | marginal/Mondrian LAC、确定性 APS、∞ 分位数、三种保护标签、预算匹配、决策指标 |
| `runner.py` | 下载并核对固定来源、独立缓存、三个模型、两个批次、各对照/消融、逐行输出与断点校验 |
| `protocol.json` | 端点、种子、alpha、预算和阈值敏感性；执行前固定，不按结果改 |
| `density_baselines.py` | 独立 KDE/KMM 组件比较；不是完整 CoDrug 或完整 selective KMM-CP |
| `audit_legacy_quantiles.py` | 用旧校准/预测文件检查第一阶段有限样本分位数影响；缺输入时明确报未能判断 |
| `summarize.py` | 按分子聚类的配对 bootstrap，固定端点面板，seed 取平均，不冒充独立样本 |
| `external_adapter.py` | 验证完整外部方法的逐行集合与目标 ID、alpha、来源记录 |
| `score_external.py` | 外部预测固定后再接标签；分别输出全人口和所选子集指标 |
| `tests/` 与独立 `test_*.py` | 数值边界、输入契约、恢复运行、标签与身份对齐测试 |
| 仓库根目录 `paper2_latex/resubmission_20261104/` | 独立论文改写稿、逐点回复草稿；新实验结果处明确待填 |

所有实际数据、模型和大逐行结果在运行者指定的 `.local` 或独立磁盘中；不要提交原始受限数据，也不要覆盖投稿时的记录。

## 4. 二作开始：环境、代码与第一轮检查

在新的目录克隆；如已有仓库，先保存自己的修改，然后 fetch 并切到本修订分支。不要 reset、clean 或覆盖旧环境。

```bash
git clone --branch paper2-chemolab-resubmission-20261104 https://github.com/DCarchimonde/aidd-paper-factory.git
cd aidd-paper-factory
python -m venv .venv-chemolab
```

Windows PowerShell 激活：`.venv-chemolab\Scripts\Activate.ps1`。Linux 激活：`source .venv-chemolab/bin/activate`。使用 Python 3.12（与交付验证环境一致），先在新环境安装：

```bash
python -m pip install -r paper2_admet_benchmark/revision_2026/requirements.txt
python -m unittest discover -s paper2_admet_benchmark/revision_2026 -p "test_*.py" -v
python -m unittest discover -s paper2_admet_benchmark/revision_2026/tests -p "test_*.py" -v
python -m unittest discover -s paper2_admet_benchmark/tests/racer_c4 -p "test_*.py" -v
python -m paper2_admet_benchmark.revision_2026.runner smoke --jobs 2 --output .local/chemolab_smoke
```

`smoke` 使用合成输入，只证明代码能联通，不能写进论文当分子实验结果。默认 CPU 可运行；本版主模型不依赖用户笔记本，也不需要 GPU。二作可在自己电脑或服务器执行。先用 2 个线程确认运行，再按机器资源增加，不能在比较过程中任意更换数据处理版本。

旧缓存只作为参考。新 `prepare` 将保存原始字节哈希、处理代码及 RDKit 版本；如果网站不可达，先记录失败，不改 URL/内容来强行通过哈希。若二作已有同哈希源文件，可以按 `runner --help` 与旧 source lock 的文件名放入新缓存 sources 路径，程序会验证。

```bash
python -m paper2_admet_benchmark.revision_2026.runner prepare --cache .local/chemolab_cache
```

## 5. 阶段 A：历史核查和一个真实小单元

先运行旧分位数审计，按 `--help` 指定二作保留的旧 confirmatory predictions / summary 路径。当前 GitHub 不含完整逐行校准预测，所以本交付不能判断旧论文实际受影响多少。若报告 `missing_inputs_cannot_determine_historical_impact`，应寻找旧电脑/服务器的输出或重新构建该输入，不能写“零影响”。旧 Stage II 已使用正确的 ∞ 处理。

```bash
python -m paper2_admet_benchmark.revision_2026.audit_legacy_quantiles --help
python -m paper2_admet_benchmark.revision_2026.runner run --phase development --endpoints Tox21_NR_AhR --seeds 101 --models logistic stack equal_logit --alphas 0.1 --jobs 2 --cache .local/chemolab_cache --output .local/chemolab_pilot
```

检查 `run_complete.json`、每个 cell 的 `metrics.csv`、`audits.json`、`roles.csv` 和 `predictions.csv`。人工确认：同端点同种子三个模型的 fit/router/conformal 分子角色相同；各方法目标 ID 完全相同；重叠/无效结构只按预设规则排除；缺失标签不是阴性；TAME 与 matched 对照从原 {0} 和 {1} 单标签中转双标签的个数分别一致。

测量这个 pilot 的真实墙钟时间、峰值内存、下载时间和磁盘占用，再按单元数估算全量时间。没有 pilot 实测，不承诺“几小时跑完”。任何异常先修代码，再用新输出目录重做受影响单元，并记录版本变化。

## 6. 阶段 B：共同模型与分数的主矩阵

数据：官方全部 12 个 Tox21 端点；历史六主端点固定为 NR-AhR、NR-ER、SR-ARE、SR-ATAD5、SR-MMP、SR-p53。全部十二端点逐一报告；六端点主面板沿用历史选择，不能按本轮结果更换，其余端点作为小样本描述性压力测试。

模型：ECFP logistic、旧四模型堆叠 stack、相同四模型等权 logit 集成。三者共享分子角色；stack 的 router 仅在 router 集训练，conformal 校准集不能进入监督模型拟合。主方法比较在同一 predictor 内进行；跨 predictor 表单列，尤其完整 CoDrug 的自身表征不能被算成包络的贡献。

分数：主比较统一 LAC；确定性 APS 是单独明确的分数敏感性。alpha 主值 0.10，附加 0.05/0.20。种子 leaderboard=101–105，EPA=211–215，仅表示重复拟合。每批 12×5×3=180 个 predictor 单元；两批合计 360 个 predictor 单元，每单元三个 alpha 共享预测。多个方法的后处理共享这些模型输出，不需逐方法重复训练。

```bash
python -m paper2_admet_benchmark.revision_2026.runner run --phase development --jobs 4 --cache .local/chemolab_cache --output .local/chemolab_dev
python -m paper2_admet_benchmark.revision_2026.runner run --phase retrospective_epa --jobs 4 --cache .local/chemolab_cache --output .local/chemolab_epa
```

中断后用**完全相同参数**附加 `--resume`。完成文件逐项核对哈希后才跳过；代码、配置、数据或环境改变时拒绝混合结果，应换新输出目录。不要只把旧文件重命名来骗过检查。

最强的简单概率对照按“被原集合遗漏的标签的模型概率”排序：原 {0} 按 p1 降序，原 {1} 按 1−p1 降序。它比一般 entropy/max-confidence 更直接地针对被遗漏标签；Mondrian singleton 未必等于 argmax，因此两种排序都保留。

主对照包括：普通 marginal/Mondrian、三种 LR-weighted Mondrian、Mondrian APS、旧 TAME(label0)、active保护(label1)、双类保护、无门槛/单视图/union 消融、空集合修复、full set。对每种 TAME 保护方式，置信度、不相似性和随机对照都匹配原预测类别内的扩张数。

固定预算曲线按原 Mondrian singleton 的 0%、10%、25%、50%、75%、100% 扩张；同时有相同总拒判数的非 CP argmax-abstention 对照。full set 给出100%覆盖、0 singleton的极端参照，提醒读者覆盖率不能脱离代价解读。

## 7. 阶段 C：强文献方法是必补项，不得用组件冒名

CoDrug 包含 energy-regularized 分子分类器、无标签辅助信息、KDE 与加权 CP；只在旧 score 上做 KDE 不是完整 CoDrug。KMM-CP 包括其原文的 KMM 方法及 selective 版本；本代码的 KMM 组件不等于完整选择流程。

本环境未取得作者链接的代码（仓库读取返回404），因此完整 CoDrug/KMM-CP **尚未实现和运行**。本轮已经提供组件计算与外部逐行接入、核对和计分入口。二作需取得可用官方源码，或按论文独立实现并核对公式后执行；不能把 `component_only` 改名为原方法来完成表格。正式稿在这项未解决前不提交。

执行顺序：

1. 核对本目录文献中的最终版本，保存方法来源、代码 commit、依赖和许可。与文中一项已报告配置对照，先检查实现可复现性。
2. 固定 source fit/router/conformal 身份，与本套目标结构和 alpha 对齐。所有训练/超参选择仅使用允许的源数据/开发数据，禁止用 EPA 指标调参。
3. 若原方法需要自身深度模型，单列“完整方法比较”；共享 predictor 的组件比较另列，不能混为后处理优劣。
4. 按 `LITERATURE_AND_COMPARATORS.md` 导出完整目标 ID 的集合 CSV 和 provenance JSON，运行 `external_adapter`，再运行 `score_external`。其成功只代表输入与计分契约通过，仍需检查上游实现。
5. selective KMM 同时报告选择率、所选子集覆盖、各真类保留比例，以及未选择者视为 defer 的全人口指标。不能把被删掉的难例从分母中消失。

独立 KDE/KMM 组件可以先运行，明确标为组件实验。KMM 的二次规划可能消耗较多时间；先用一个端点试跑、检查 solver 收敛与约束残差。达到规模上限或不收敛时应报失败，不能静默采样/换算法/只报告成功单元。在上述 pilot 命令中加入 `--density-components kde kmm` 并使用新的输出目录即可执行；其余选项以 `runner --help` 和 `density_baselines.py` 为准。

## 8. 阶段 D：消融、阈值和方法贡献

```bash
python -m paper2_admet_benchmark.revision_2026.runner run --phase retrospective_epa --jobs 4 --sensitivities --cache .local/chemolab_cache --output .local/chemolab_epa_sensitivity
```

运行前可在 development 上完成同样设置并固定整个参数网格。原全部门槛、逐一改变门槛、取消门槛、单视图、双视图 all/any、保护类选择，都需报告。不能发现某阈值在 EPA 上更好就把它换成新的主规则。

要回答的不是“消融后有几个数字变差”，而是：

- gates 是否识别了加权会失败的单元，还是只减少方法使用次数？
- 两视图的一致性是否比一个视图有更好的同预算错误控制？
- label1 保护是否实质减少 missed activity；双类保护的额外代价多大？
- TAME 是否胜过同预算的置信度拒判；若不能，复杂视图的额外价值是什么？
- equal-logit 已经足够时，stack/router 是否值得加入？

若答案是否定的，应按预先问题报告负结果，并将正文改为方法比较研究；不能继续靠放宽阈值、换主要终点来追逐优胜。

## 9. 阶段 E：统计、图表与结论

主要比较固定为 stack、alpha=0.10、历史六端点面板，`tame_label1_active` 相对 `matched_label1_active_omitted_label_probability` 的 assay-active missed-activity/false-clearance 概率差，越低越好。uncertainty、AD、random、普通 Mondrian、完整强方法为预设关键比较；全部披露，不能选择测试集最弱方法作为唯一对照。logistic/equal-logit 验证模型敏感性。

```bash
python -m paper2_admet_benchmark.revision_2026.summarize --predictions .local/chemolab_epa --output .local/chemolab_analysis/tame_vs_omitted_probability.json --method tame_label1_active --reference matched_label1_active_omitted_label_probability --predictor stack --phase retrospective_epa --alpha 0.1 --replicates 2000
```

完整外部方法计分后，用 `--additional-predictions` 指向 scored 的 `predictions.csv` 或只含 scored 文件的目录；勿把没有 y 的原始导入目录混入。`--predictor` 使用外部 provenance 的 evaluation_model，`--reference-predictor` 指定 stack/equal_logit/logistic。

同样计算 matched uncertainty/AD/random 及其他预设对照；模型交叉比较可用 `--reference-predictor equal_logit`，但方法名必须与真实输出对应。summary 使用整个固定面板的端点-种子等权均值；每次抽取同一 structure_id 的次数，在各端点、各方法及所有种子间共享。CI 条件于已拟合模型和这组固定端点；没有声称外推所有化学任务，也没有把六端点或五种子当成大量独立新化合物。

所有区间为点式、探索性 bootstrap 区间，不写成多重比较后确认显著。主要结果同时展示每端点差异、样本数和效应大小；小分母导致区间不可用时显示 NA 及原因。若要声称正式非劣，须另设有依据的界值并计算相应区间；旧 −5 pp 点估计 guardrail 不能当正式非劣检验。

建议正文图表：

1. 简明算法与决策图：普通集合、何种单标签转 defer、如何计算真实代价；哈希与流程细节移补充。
2. 覆盖与拒判/信息性曲线，明确 alpha、实际预算、模型；包括 full-set 极端点。
3. active missed activity 和 inactive correct-singleton 的同预算配对结果。
4. equal-logit/stack/logistic 与完整 CoDrug/KMM-CP 的效应及失败案例。
5. audit/gate 敏感性与作用机制，报告门槛启用率及不收敛。

## 10. 投稿前的决策标准

这次不设“只要 p<0.05 就能投”的机械门槛。须同时满足：主要对照完整、同预算比较成立、原稿已知问题修正、化学测定语义清楚、结果有一个可以被复核的实质发现。

- 如果 TAME 相对简单拒判和强方法有稳定、有解释力的优势：按实际适用条件写方法贡献，不能扩展成普适保证。
- 如果 TAME 没有优势，但系统比较揭示可复现的适用条件/失败机制：以严谨的比较研究重写题目、摘要、结论和 cover letter，不强行维护算法冠军叙事。
- 如果只有旧现象重述、小幅波动或新增比较仍不支持清楚贡献：继续研究或改变投稿定位，不能将当前稿直接重投。

编辑信按 new submission 处理。新 cover letter 需引用 CHEMOLAB-D-26-01118，概述研究问题、实验、代码和结论的实质变化；附逐点说明。不要称为原稿 R1，不要在新结果未完成前写“we have demonstrated”。

## 11. 二作交回什么

交回一份执行报告，包含代码 commit、环境版本、真实命令、单元完成/失败清单、总耗时、数据哈希、原分位数影响审计、主要和敏感性汇总、外部方法实现差异、完整日志与逐行输出路径。大文件可打包交接，原始受限数据按来源许可处理。

第一阶段只需先反馈环境测试、真实 pilot 和原始数据是否可取得；随后继续完成已授权的主矩阵，不用等“允许训练”的重复确认。任何会改变研究问题、主要对照、主要终点或数据定义的改动，先写明原因、影响和新的版本，不能静默混入旧输出。

### 可直接发给二作

> 这篇 CHEMOLAB 稿件已收到拒稿后可大幅修改重投的意见。我们已在 GitHub 新分支 `paper2-chemolab-resubmission-20261104` 提交修订实现与计划，请从 `paper2_admet_benchmark/revision_2026/START_HERE_CN.md` 开始。先建新环境、跑测试和一个真实 pilot，再执行共同模型/分数的主矩阵及阈值敏感性。重点是 TAME 与按原预测类别匹配拒判预算的 uncertainty/AD/random 对照，以及 equal-logit 简单模型；完整 CoDrug/KMM-CP 仍需按文献核实实现并运行，现有组件不能冒名替代。EPA 已有标签结果，本轮只能称回顾性比较。请保留全部阴性结果、日志与逐行输出，先给我 pilot 耗时和数据可用情况，再按计划继续全量，最后交回实验报告和结果包。新论文稿和回复草稿已搭好，待真实结果填入后再决定提交。
