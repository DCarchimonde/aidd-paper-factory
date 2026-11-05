# CHEMOLAB 全面大修：研究方案与运行交接

更新：2026-11-05。原稿号：CHEMOLAB-D-26-01118。仓库：DCarchimonde/aidd-paper-factory。分支：paper2-chemolab-resubmission-20261104。

**本文件取代 2026-11-04 的交接。研究设计、完整比较方法、数据处理、统计、作图、论文与回复由我们完成；二作只负责运行、保存日志和交回结果，不承担补算法、选指标、修改阈值或撰写结论。**

本次交付包括完整实验程序与实验前全文。正式矩阵尚需服务器实际运行；任何试跑都不能充当正式研究结果。运行结束后程序自动生成真实结果表、图和带结果的全文，我们再核对解释、局限、回复与投稿文件。不会让二作手工补数字或自行判断怎么写。

## 一、这次到底改什么

审稿意见指出的是贡献与证据不相称，不意味着所有积累都没有价值。我们接受四个实质问题：已知的 marginal/class-conditional 区别不是新原理；TAME 的加标签性质不是新覆盖保证；小幅 coverage 增益必须与弃判代价一起比较；化学语义和实际筛选目标必须一致。

新研究的核心问题是：**在同样数量、同样原预测类别的分子被送去进一步核查时，迁移信息是否比简单概率排序更能找出错误？这种增量能否跨测定家族重复，什么条件下消失？**

研究由以下四块组成，缺一不能靠措辞补足：

1. 固定同一预测器和校准样本，比较 CP、加权 CP、TAME 与简单弃判，隔离后处理的贡献。
2. 以各自完整神经网络实现 CoDrug、标准与 selective KMM-CP，另列完整方法比较，不把表征变化算成 TAME 的功劳。
3. 增加 AMES、标准小规模 hERG、DILI、CYP2D6_Veith，原审计阈值不变，固定随机与骨架划分，检验跨任务稳定性。
4. 比较同预算分歧分子的错误救回数，加入迁移权重错配对照及合成机制诊断，解释有效与无效的原因。

正文不再以 ClinTox 的已知反例为主要贡献，不再把 firewall、hashing、promotion 当作算法创新，不再把原 +1.36 pp 对普通 Mondrian 的结果当成胜过强方法的证据。旧第一阶段数字不进入本轮主要证据；其脚本与历史审计入口保留，二作无需寻找旧逐行文件来填本轮结果。

## 二、逐条回应审稿意见

| 审稿问题 | 实质修改 | 程序或正文依据 |
|---|---|---|
| 类别覆盖不足是已知现象 | 引言简述已有理论，研究问题转为错误定位与弃判成本 | 新 Methods、同预算比较 |
| TAME 只是保守 wrapper | 明确其集合扩张性质，不声称新有效性定理；检验是否有额外信息价值 | 完整简单对照、权重错配、分歧分子分析 |
| 域分类器、clipping、gates 是启发式 | 逐一记录启用原因，固定敏感性与 ungated 对照，检验门槛能否识别失败 | audits、gate sensitivities |
| label 0 与毒性阳性动机矛盾 | 主策略固定保护 label 1；历史 label 0 与双类策略单列 | 三种策略同时输出 |
| 只与普通 Mondrian 比不够 | 补完整 CoDrug、KMM-CP、selective KMM-CP 和非 CP 弃判 | full_baselines 与共同模型矩阵 |
| EPA 增益小且局限单协议 | EPA 如实标回顾性；加入四个任务的固定随机/骨架评价 | independent_data |
| 模型和 score 跨阶段更换不清楚 | 主分析统一三个预测器的 LAC；APS 为明示敏感性；完整 GNN 方法另列 | protocol、roles、model 字段 |
| 工程语言压过算法 | 正文直接写预测集合、权重、扩张和决策指标；版本记录移实现说明 | 全文和逐点回复 |

另补两项自主纠错：新核心使用正确有限样本分位数，rank 超过校准数时取正无穷；统计对同一结构在不同端点、方法和训练种子上的预测共同重采样，不能把种子当作独立新分子。

## 三、数据、标签与划分已经固定

| 数据 | 外层评价 | 训练种子 | 标签 1 的意思 |
|---|---|---|---|
| Tox21 全部 12 端点 | 原 public leaderboard，development | 101–105 | 对应核受体或应激反应测定 active |
| Tox21 全部 12 端点 | 原 EPA，retrospective_epa | 211–215 | 同上 |
| AMES | 固定 random 与 scaffold 各一套 | 每套 311–315 | 细菌回复突变试验阳性/致突变标记 |
| 标准 hERG | 固定 random 与 scaffold 各一套 | 每套 311–315 | hERG blocker |
| DILI | 固定 random 与 scaffold 各一套 | 每套 311–315 | 数据集记录的肝损伤风险阳性 |
| CYP2D6_Veith | 固定 random 与 scaffold 各一套 | 每套 311–315 | CYP2D6 酶抑制阳性 |

Tox21 的 active 不能等同于临床有毒；inactive 不能证明安全。各任务的具体定义、原始版本和清洗数量由 `independent_sources.json`、`INDEPENDENT_VALIDATION.md` 与运行记录共同说明。下载原始行数可能与网站简介不同，必须报告实测清洗流程，不能把网站数字硬填成实际样本量。

新任务的外层 source/target 约 80/20；random 固定外层种子 43021，scaffold 固定 43022。同一外层的五个训练种子共享同一 target，只改变 source 内部 60/20/20 的 fit/router/calibration 分配。骨架分组不可拆开，实际比例可以偏离 80/20。结构冲突、无效结构、重叠与缺失值按事先规则处理，不按模型表现删端点或换划分。

新增数据来自公开基准；其标签用于数据清洗与最终评价，不能称新采集或新的盲测。EPA 已经看过，改 seed、换分支、重新 hash 都不会让它重新独立。新家族分析检验规则原样应用时的表现，不据其结果重选阈值或保护类别。hERG 与 DILI 的校准量以及 DILI 的目标量低于原 gate 要求，回退本身是要记录的现象，不能为让 TAME 启用而放宽门槛。基于这个样本量检查、在任何新任务模型结果出现前加入 CYP2D6_Veith，使 AMES 之外另有一个校准量充足的任务；保留两组小样本边界测试。它是本研究的新基准，不引用其他 CYP 项目的结果。

## 四、完整方法与公平性

### 4.1 共同预测器的比较

三个预测器为 ECFP logistic、原四模型 stack、四模型 equal-logit 集成；共享相同分子角色和目标分子。stack 的 router 仅用 router 集拟合，校准标签不参与监督训练。

全部 alpha 为 0.05、0.10、0.20，主值 0.10。计算 marginal/Mondrian LAC、APS、三种 LR-weighted Mondrian、TAME 的 label 0/label 1/双类保护、无门槛/单视图/union、全部固定门槛敏感性、full set、固定预算及非 CP 弃判。方法清单由程序在读取结果前生成，完成检查严格核对清单。

TAME 的简单对照必须分别匹配原 `{0}` 与 `{1}` singleton 被扩张的数量。最强的直接概率对照是：原 `{0}` 按 p1 从大到小；原 `{1}` 按 1−p1 从大到小。不能仅用 entropy 代替，因为 Mondrian singleton 不必等于概率 argmax。另有 uncertainty、训练集最大 Tanimoto 距离和确定性随机排序。无门槛、union、单视图及门槛敏感性也各自配同预算概率对照，避免把不同弃判数量造成的变化误当成组件作用。

固定预算曲线扩张原 singleton 的 0%、10%、25%、50%、75%、100%；full set 是完全弃判的极端参照。匹配使用所有 eligible 无标签 target，标签缺失的行也参与实际工作量计数；有标签子集的预算可能不同，程序另列完整预算和评价分母。

### 4.2 CoDrug 的完整实现

`full_baselines/codrug.py` 包含分子图 AttentiveFP、监督分类与 energy-margin 联合损失、ZINC250k 辅助无标签分子及源骨架排除、校准/目标输出密度各自交叉验证 KDE、包含 query 正无穷质量的加权预测集合。它不是在原 stack 输出上接一个 KDE 的简化版。

作者代码本环境未取得，本次是按论文与官方架构源码核对的独立实现。详细超参、原文未说明的固定实现选择、源码链接和测试见 `CODRUG_IMPLEMENTATION.md`。我们不声称它已逐数值复现作者原表。

原文印刷 margin 为 in=-5、out=-35，可能不能形成通常预期的能量分离。正式矩阵同时保留原文值与预先指定的反向 margin 敏感性（in=-35、out=-5）。两者都要报告，不能只挑较弱版本来衬托 TAME。

### 4.3 完整 KMM-CP 与 selective KMM-CP

`full_baselines/kmm_cp.py` 与独立 vanilla AttentiveFP 训练组合：置换标准化 MMD 选择带宽、标准有界 KMM 二次规划、源权重/目标选择权重联合优化、阈值选择目标、对选中目标重新拟合 KMM，再用加权经验分位数预测。

原文 predicted-group 与 candidate-label Mondrian 的记法有差异，程序分别输出原文字面版本和明确的 candidate 版本，另有 global 版本。不得静默把两者混称同一个算法。selective 方法同时保留原始集合、选择标记、选中子集结果和全目标结果；未选中分子在全目标决策中记作 defer，不能消失于分母。

标准/选择 QP 的求解状态、约束残差、有效样本数与选择率全部保存。未收敛不能用全 1 权重冒充完成。完整算法与原 `density_baselines.py` 中的 KDE/KMM 组件明确区分；组件只是可选诊断，不是强方法的替身。

### 4.4 机制与门槛诊断

主机制分析记录 TAME-only 与 probability-only 的弃判分子、其中阳性/阴性数及分箱。相同原 `{0}` 预算下，若救回阳性数分别为 eT、eP，则 false-inactive 差为 `(eP−eT)/N1`，阴性 correct-singleton yield 差为 `(eT−eP)/N0`。缺失标签时使用完整纠偏式，不能假定有标签子集仍严格预算相同。这是解释性恒等式，不是新覆盖定理。

五次固定置换在 p1 的固定区间 `[0,.2,.4,.6,.8,1]` 内打乱 query weight 与分子的对应；两个视图使用同一置换。源校准权重、目标密度估计和双视图联合权重分布保持不变。该实验只检验 query 权重对齐作用，不能宣称破坏了全部迁移信息。gated 与 ungated 分别产生自己的同类别预算概率对照。

固定 predictor 的二分类 LAC 下，如果每个视图的 query weight 对所有目标分子均为常数，TAME 对原 `{0}` 集合的追加规则只相当于 p1 超过一个固定阈值，因此与同预算概率排序相同。我们给出这个结构解释及带 ties 的数值测试，用它明确真正需要检验的是逐分子权重是否改变了有效排序，而非源权重改变后多弃判了多少。

合成机制诊断区分无偏移、已知 ratio 的 covariate shift、support loss 和相同 X 的 concept shift，使用 oracle 作为诊断。其作用是核查方法边界，不替代真实分子实验，不计入分子样本量。

## 五、二作实际只需做这些

推荐在独立 Linux GPU 服务器运行；CPU 也受支持，但完整神经网络矩阵耗时明显更长。不要在用户笔记本启动全量。先使用 Python 3.12 新环境，保留旧项目环境和历史结果。

```bash
git clone --branch paper2-chemolab-resubmission-20261104 https://github.com/DCarchimonde/aidd-paper-factory.git
cd aidd-paper-factory
python3.12 -m venv .venv-chemolab
source .venv-chemolab/bin/activate
python -m pip install -r paper2_admet_benchmark/revision_2026/requirements_full.txt
```

Windows 的激活命令为 `.venv-chemolab\Scripts\Activate.ps1`，其余 Python 命令一致。`--device auto` 自动选择可用 CUDA，否则 CPU；需要确保使用 GPU 时写 `--device cuda`，不可用会明确报错。无需改代码中的设备字符串。

先跑短试跑：

```bash
python -m paper2_admet_benchmark.revision_2026.execute_revision --pilot --device auto --jobs 4 --cache .local/chemolab_cache_v2 --output .local/chemolab_pilot_v2
```

试跑固定缩小数据和训练轮数，只检查数据、三种共同预测器、完整强方法、文件计分与恢复流程。输出明确标为 pilot，不能填入论文主表，也不能拿它的成绩调整方案。

正式运行：

```bash
python -m paper2_admet_benchmark.revision_2026.execute_revision --device auto --jobs 4 --cache .local/chemolab_cache_v2 --output .local/chemolab_formal_v2
```

该命令自动准备数据、列出全部任务、训练、运行所有方法、计算统计、生成图表/论文结果和打包。计划共 160 个数据/训练种子 case：120 个 Tox21 case 与 40 个新任务 case；每个 case 有 3 个共同预测器和 3 个完整 GNN 配置，共 960 个 predictor cell，其中 480 次神经网络训练。alpha 与后处理方法共享一次预测，不逐方法重复训练。

实际时间受 GPU、分子大小、早停与 QP 影响，不能在完整实跑前保证小时数。试跑给出测量信息；全量预留充足运行时间与磁盘，日志实时刷新。试跑中每个 CoDrug checkpoint 约 121 MB、KMM backbone checkpoint 约 149 MB，仅正式 480 个网络的 checkpoint 就约 62 GB。再计逐行 CSV、报告、压缩包与临时文件，建议预留至少 300 GB 可用磁盘；这是容量预算估计，实际用量由程序记录。不要只保留截图。

中断后原命令添加 `--resume`：

```bash
python -m paper2_admet_benchmark.revision_2026.execute_revision --device auto --jobs 4 --cache .local/chemolab_cache_v2 --output .local/chemolab_formal_v2 --resume
```

恢复会核对代码、配置、环境、输入和已完成文件。没有完整收据的 cell 继续或重做；已有完整收据的 cell 验证后跳过。若报版本或哈希不一致，把错误与日志交给我们，不自行删收据、改配置或替换数据来强行通过。

二作可决定硬件、保存路径、线程数和何时继续运行；不能决定丢哪些端点、改门槛、替换强方法、把失败结果改成 baseline 或按表现挑图。遇到技术错误，保留目录和日志，我们修代码后给出继续命令。无需他写论文或分析结果。

## 六、统计、自动报告与全文

预先固定的核心比较为 stack、alpha=.10、TAME label1 与同原预测类别预算的 omitted-label-probability 对照。EPA 历史六端点面板为既定回顾性主要比较；十二端点全表、新四个任务、两种划分、三个预测器、完整强方法与两种 CoDrug margin 均完整披露。新任务分别按任务与划分报告，不能把它们并入历史六端点来稀释不利结果。

主要量为 `P(prediction={0}|Y=1)`，即 false-inactive；同时报告反方向错误、两类覆盖、最差类覆盖、macro correct-singleton yield、弃判率、空集合率、阴性决策中的阳性比例。完整 GNN 方法的实际工作量未必相同，必须连同 error/deferral 前沿一起解释；同 alpha 不等于同预算。

置信区间使用固定面板、端点及训练种子等权均值的配对结构簇 bootstrap，2000 次。每次重采样，同一结构在不同端点、方法和种子的次数一致。重复训练种子不是独立样本，区间也不是适用于所有未来化学任务的保证。本轮区间为点式探索性结果，不能把多张图中的偶然显著升级成确认性发现。

自动报告检查预先列出的整个任务矩阵，而不是根据成功文件倒推计划。缺方法、缺 alpha、同目标身份不一致、标签错位、同预算不成立或文件被修改都会阻止生成完整研究状态。未完成时仍保留真实失败记录。

程序输出全部 cell 与 panel 指标、预设配对比较、覆盖/错误/弃判图、机制分析、运行与版本信息，以及由真实结果填充的 `manuscript_completed.tex`。若系统安装 LaTeX 则同时编译 PDF；缺 LaTeX 不要求二作写作，我们从完整源文件编译即可。正式论文最终解释、贡献表述与 cover letter 由我们审阅定稿。

结果无论正负，都围绕预先的问题解释：有增量就写清适用条件、代价和大小；没有增量就明确说明简单概率排序能否替代复杂修正、哪些条件造成失效。不能为了维持 TAME 名称而挑弱对照。编辑允许 substantially revised new submission，这是重审机会，不是录用承诺。

## 七、交回文件与责任分工

二作交回程序生成的结果包、完整日志、最终运行状态及服务器环境信息；优先交回轻量 review bundle；完整逐行与模型包由程序生成分卷及校验记录，保留原目录并交回这些分卷，不用手工重新组织文件。模型文件用于复核，不能删除后只交均值表。**不要手动编辑输出 CSV、图或 manuscript。**

我们收到结果后负责：复核所有矩阵与数据语义、阅读实际失败案例、检查图表和全文一致、完成逐条回复与新投稿 cover letter、编译投稿版本。cover letter 引用 CHEMOLAB-D-26-01118；按新投稿处理，不冒称原稿 R1。

### 可以直接转发给二作

> 这篇 CHEMOLAB 拒稿重投稿正在全面重做。研究方案、完整 CoDrug/KMM-CP、数据处理、统计和论文结果生成均已由我们写好，你只需要在独立环境按仓库 START_HERE_CN.md 跑短 pilot、正式矩阵并交回程序生成的结果包和日志。不要自行修改方法、阈值、划分或挑选结果，也不需要补代码或写论文。遇到错误保留输出目录，把完整日志发回，我们负责修复和给继续命令。正式实验完成后，我们负责核读结果、写结论、逐点回复和投稿。
