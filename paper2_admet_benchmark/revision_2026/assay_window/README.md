# 配对起始响应间隔：新机制运行入口

> **2026-11-09更新：本轮真实初验完成，当前联合关联模型暂停扩展。** 8次拟合全部收敛，返回数据和指标重算通过。总体NLL略有改善，但明确阳性拟合恶化，阳性排序落后于简单二元分类器。详见[完整复核与裁定](SCREEN_REVIEW_20261109.md)。二作当前无需追加运行。

当前分支：`paper2-assay-window-20261108`。旧 TAME / batch-triage 路线停止扩展。本目录从原始配对浓度曲线重新建立监督目标与模型。

研究问题：给定未见化合物的结构，能否预测报告通道首次响应比细胞活性损失至少早三倍浓度，且实际测试范围覆盖这个间隔？模型保留未跨阈值曲线的删失信息，并学习两种起始浓度的关联。首次跨越后的回落另行记录。

本目录包含数据读取、目标变换、联合/独立删失模型、两个简化对照、分组划分、结果汇总与断点续跑。真实分子结果已于2026-11-09复核；下方保留原实验的重现方法。

- [完整研究与论文改写方案](RESEARCH_PLAN_CN.md)
- [最近邻文献与候选贡献](LITERATURE_POSITION.md)
- [二作可直接照做的说明](OPERATOR_PROMPT_CN.txt)
- [软件检查记录](VALIDATION.json)

## 开发前数据检查与正式结果

正式运行经过结构处理后保留19,179对曲线、5,387个parent组，明确yes 451、明确no 17,871、未决857；三次均明确yes的sample为22。以下19,724对为化学标准化前的开发检查数量，两者的545对差异均记录为结构处理失败。

默认 NCATS AR-MDA-kb2-luc antagonist p1 原始 ZIP 约14 MB。已完成全文件读取、单位与配对核查，未训练模型。A/B纯度、唯一通道、浓度一致后保留19,724对曲线，来自7,206个sample ID；这些数量尚未经过RDKit化学标准化。

固定30%报告响应、20%活性损失、3倍间隔后，461对明确满足、18,393对明确不满足、870对未决。5,741个三次重复完整的sample中，只有23个三次均明确满足，247个出现明确yes/no冲突。461是重复测量行数，不是461个稳定阳性化合物。详见[evidence/real_protocol_window_feasibility.json](evidence/real_protocol_window_feasibility.json)。

## 安装

在新目录检出本分支，旧结果目录保留：

```bash
git clone --single-branch --branch paper2-assay-window-20261108 https://github.com/DCarchimonde/aidd-paper-factory.git aidd-paper-factory-assay-window
cd aidd-paper-factory-assay-window
python3.12 -m venv .local/venv-assay
source .local/venv-assay/bin/activate
python -m pip install -r paper2_admet_benchmark/revision_2026/requirements.txt
```

Windows对应环境命令为 `py -3.12 -m venv .local/venv-assay`，随后 `.local\venv-assay\Scripts\Activate.ps1`。此前运行batch screen的Python 3.12环境满足同一份依赖时可以直接沿用。CPU即可，无需GPU或PyTorch。

## 既有初验复现命令（本轮已完成）

从仓库根目录执行；`--jobs 4`限制数值库线程数，并非同时启动4个大模型。

```bash
python -m paper2_admet_benchmark.revision_2026.assay_window.execute --mode pilot --jobs 4 --output .local/assay_window_pilot --resume
python -m paper2_admet_benchmark.revision_2026.assay_window.execute --mode prepare --jobs 4 --cache .local/assay_window_cache --output .local/assay_window_prepare --resume
python -m paper2_admet_benchmark.revision_2026.assay_window.execute --mode screen --jobs 4 --cache .local/assay_window_cache --output .local/assay_window_screen --resume
```

1. `pilot`：合成曲线，检查全流程，8次轻量拟合。
2. `prepare`：下载/复用原始包、标准化结构、配对曲线、整理区间与复测统计，0次模型拟合。
3. `screen`：同一协议，random/scaffold两种分组划分，joint_censored、independent_censored、point_imputed、binary_window四模型，共8次拟合。目标测试比例25%，实际random为25.00%、scaffold因保留完整大骨架桶达到43.57%。分组使用标准化parent ID；同一parent的盐形式和全部重复一起进入训练或测试。结构指纹SVD只在训练结构上拟合。

正常结束会打印 `complete_declared_matrix` 或数据准备完成。若打印 `complete_with_nonconverged_fits`，直接回传结果，交由本项目处理数值问题。遇到异常保留完整报错；中断后重跑同一条命令即可，已完成拟合不重复运行。更改代码、参数或环境后使用新输出目录。

## 原回传内容（已收到并核验）

回传 `.local/assay_window_prepare_review.zip` 和 `.local/assay_window_screen_review.zip`；pilot异常时再附pilot包/报错。包内包含：

- `READOUT_CN.md`：结果概览。
- `data_profile.json`、`source_diagnostics.json`：实际化学分组、删失/参考状态、复测冲突、排除数量。
- `paired_records.json.gz`、`derived_records.json.gz`：原始配对读数、起始浓度括区及参考状态。
- `split_*.json`：训练/测试parent分组、训练内特征处理信息。
- `metrics.json`、`test_predictions.csv`：各模型指标、逐行概率分辨率界与mask NLL。
- `configuration.json`、`runs/`、`experiment_complete.json`：实际配置、逐次结果和收敛状态。

二作无需调参、改阈值、加协议或写稿。我们根据相同测试化合物上的联合/独立模型差值、明确阳性预测和复测表现判断机制是否值得继续；总体NLL改善不单独决定方向成立。

## 本地软件测试

```bash
python -m unittest discover -s paper2_admet_benchmark/revision_2026/assay_window -t . -p 'test_*.py'
```

数据解析测试使用内嵌小样例；模型测试使用合成数据。下载的原始数据和运行结果留在`.local/`，不进入Git提交。
