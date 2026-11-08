# 分子批次双风险测定分配：新一轮研究与运行入口

> **本路线已停止扩展（2026-11-08）：** 根据首轮结果与作者决定，不继续修补四步前瞻，也不运行formal/sensitivity。新机制与运行入口见 [assay_window/README.md](../assay_window/README.md)。以下内容保留用于复现既有实验。

> **2026-11-08运行后更新：** 96条screen已完成。当前主获取策略在20%预算下未超过随机获取，暂不启动formal/sensitivity。结果与下一步见 [SCREEN_REVIEW_20261108.md](SCREEN_REVIEW_20261108.md)。下方命令保留用于复现本轮实验。

本轮研究问题：**在一批发生分布变化的分子中，用有限的抽样测定，尽可能减少完成整批判定所需的总测定量，同时约束两类错误。**

已有 TAME 结果构成出发点：旧 label-0 覆盖增益可被同预算简单排序复制，新 label-1 主比较没有发生扩张。现在研究真实测定信息应当花在哪里、这些信息能够支持多少自动判定。

## 具体技术贡献候选

1. 把每个固定“预测分数 × 化学相似度”区域中的未知阳性总数作为共同状态。两类错误共享这些未知计数，分母随证据一起约束。
2. 用整数优化联合选择自动判为0、自动判为1、继续测定的区域，使满足双风险证书的免测数量最大。已付出的抽样测定不算作节省。
3. 根据下一小批测定对可认证免测数量的预期影响来选择测定区域；每个区域内部仍随机无放回抽样。

计数置信序列、有限总体抽样、整数规划都是已有工具。本轮需要证明的实际增量是这种联合决策与分配能否比强的同证书对照节省更多测定。完整文献地图见 [LITERATURE_GAPS.md](LITERATURE_GAPS.md)，研究方案见 [RESEARCH_PLAN_CN.md](RESEARCH_PLAN_CN.md)，公式与推导见 [THEORY_AND_POSITIONING.md](THEORY_AND_POSITIONING.md)。

## 八种预设方法

| 方法 | 测定分配 | 决策/统计差别 |
|---|---|---|
| joint_lookahead | 预计免测收益 | 联合双风险整数优化，主候选 |
| joint_uniform | 按剩余人数随机 | 同一个证书与优化器，主对照 |
| joint_entropy | 预测熵加权 | 同一个证书与优化器 |
| joint_neyman | 方差导向分配 | 同一个证书与优化器 |
| threshold_lookahead | 预计免测收益 | 仅允许按区域平均分数排序的双阈值动作 |
| loose_lookahead | 预计免测收益 | 独立上界分子/下界分母的保守消融 |
| score_only_lookahead | 预计免测收益 | 八个纯分数区域，去掉化学相似度分层 |
| uniform_cs_lookahead | 预计免测收益 | 换成均匀总计数先验的置信序列 |

前三种分配对照与主方法使用相同分层、置信序列、风险水平及优化器。阈值对照限定相同计数证书下的策略空间。原版 PAC Labels、CARED 等的目标与适用场景在文献地图逐项比较。

## 二作只执行这些命令

在新目录检出，不覆盖此前 v3 的结果或本地修复：

```bash
git clone --branch paper2-batch-triage-20261108 https://github.com/DCarchimonde/aidd-paper-factory.git aidd-paper-factory-batch-triage
cd aidd-paper-factory-batch-triage
python3.12 -m venv .venv-batch
source .venv-batch/bin/activate
python -m pip install -r paper2_admet_benchmark/revision_2026/requirements.txt
```

Windows 可用 `py -3.12 -m venv .venv-batch`，然后运行 `.venv-batch\Scripts\Activate.ps1`。如果原环境已经满足上述依赖，也可以沿用；不需要安装或切换 GPU/PyTorch。

先运行流程检查。它只使用合成数据：

```bash
python -m paper2_admet_benchmark.revision_2026.batch_triage.execute --mode pilot --jobs 4 --output .local/batch_pilot
```

随后运行本轮指定的分子诊断矩阵：

```bash
python -m paper2_admet_benchmark.revision_2026.batch_triage.execute --mode screen --jobs 4 --cache .local/batch_cache --output .local/batch_screen
```

该阶段包括 AMES/CYP2D6 两任务、random/scaffold 两划分、三个抽样种子及八种方法，共 **96条分配轨迹**；每条轨迹报告0%、5%、10%、20%、40%抽样预算处的结果。共享预测器每个数据划分只拟合一次，此后的预算、分配方法和抽样种子复用同一组分数。主比较是20%抽样预算、两类风险均不超过10%、联合失败概率5%。

中断后按原命令添加 `--resume`：

```bash
python -m paper2_admet_benchmark.revision_2026.batch_triage.execute --mode screen --jobs 4 --cache .local/batch_cache --output .local/batch_screen --resume
```

运行完成后先交回 `.local/batch_screen_review.zip` 和终端完整报错（如有）。原始目录与分数缓存保留。二作不需要补算法、调阈值、选结果、作图或写论文。

## 后续完整矩阵

我们核读96条诊断轨迹后再安排后续研究。完整命令已实现：

```bash
python -m paper2_admet_benchmark.revision_2026.batch_triage.execute --mode formal --jobs 4 --cache .local/batch_cache --output .local/batch_formal
```

完整矩阵为四任务、两划分、20个抽样种子、八方法，共1280条轨迹。`--sensitivity` 在独立输出目录增加三预测器及三个预设非对称风险组合，合计11520条轨迹；不是当前让二作立即启动的任务。

## 输出与解释

- `all_checkpoint_metrics.csv`：全部预算处的实际两类错误、认证上界、已测数量、后续仍需测定数量和总测定成本。
- `paired_savings.json`：同一批次、同一预测器和抽样种子下的配对免测差值。
- `runs/*.json`：逐条抽样记录、每个检查点的行动和计数界。
- `experiment_complete.json`：完整矩阵状态。
- 自动统计和图表：由 `report.py` 生成，抽样种子反映固定分子批次上的随机抽样差异，不作为新增独立分子。

抽样上限与完成整批的总测定成本是两个量：总成本为“已做抽样测定 + 尚未自动判定的分子数”。程序同时计算总测定预算20%、40%、60%、80%是否足够；若不够就报告不可满足，不能把剩余测定隐藏起来。

所有真实分子结果仍待执行。本次代码的数值穷举、信息隔离测试和合成流程检查属于实现验证，不是分子实验优势。
