# TAME 实质大修交付 — 2026-11-12

稿件：**Separating coverage gains from error targeting in molecular prediction sets**

目标期刊：Chemometrics and Intelligent Laboratory Systems。原稿号 CHEMOLAB-D-26-01118；本轮按大幅修改后的新稿准备，尚未向期刊提交。

## 共同第一作者

署名为 **Siyuan Tong（第一顺位、共同第一作者、通讯作者）** 和 **Yuechen Wang（第二顺位、共同第一作者）**。主稿、补充材料和逐点回复均标注共同第一作者，投稿信与 CRediT 声明同步说明。通讯邮箱仍为 `25064241@siswa.um.edu.my`。具体 CRediT 分工沿用原有记录。

## 本轮完成内容

正文、补充材料、对原审稿意见的逐点回复、投稿信、Highlights 均已完成重写和编译。主线是同类别、同弃判预算的比较，以及加权集合追加标签的精确触发条件。全文区分门槛通过、实际追加标签和错误救回。

摘要和讨论以三项贡献展开：同类别弃判预算的比较、加权集合追加标签的精确条件、以及遗漏标签排序的决策价值。在历史六测定面板和相同预算下，遗漏标签排序的最弱类覆盖与正确单标签产出分别为 85.10% 和 46.57%，通用不确定性排序为 84.75% 和 44.61%。遗漏标签排序还复现了 TAME 的 1.3649 个百分点覆盖增益。主要主动标签比较的零干预结果、完整参照配置、额外测定家族与相关区间均保留。文字调整记录见 `EDITORIAL_REVIEW.md`。

## 实际运行源码已补齐

两份实际运行源码已收到并与正式实验来源记录逐字节核对一致；其参考版本也已与 GitHub 的实验参考提交核对。此前的源码缺口已关闭，相关实现已同步到仓库的实际调用路径，并纳入本次完整数值附件。

- CoDrug：补入骨架提取遇到立体化学异常后的重试，以及恢复 CUDA 随机状态前转回 CPU 的处理。
- KMM-CP：补入二次规划目标的正比例缩放，并将报告的目标值还原到原始尺度；SI 已说明该实现。
- 修复前的 CoDrug 源码由参考版本和已识别的 CUDA 改动恢复，字节哈希与前一阶段正式记录一致，单独保存在 `runtime_sources/pre_scaffold_fix/`。

原始回包源码、参考源码、差异补丁、收集记录和说明均在 `runtime_sources/`。核对结果见 `RUNTIME_SOURCE_VERIFICATION.json`，固定数值与图表的逐文件核对见 `FROZEN_CONTENT_VERIFICATION.json`。本次没有重新训练，也没有修改既有科学结果。

本包为更新后的完整投稿准备材料，尚未向期刊提交。作者顺序、单位和通讯信息沿用原稿；两位作者新增共同第一作者说明。

## 投稿文件对应

| 文件 | 用途 |
|---|---|
| `manuscript.pdf` | 主稿阅读版 |
| `supplementary.pdf` | 补充材料 |
| `response.pdf` | 对原审稿意见的逐点回复，按意见主题标明摘要 |
| `cover_letter.pdf` | 投稿信 |
| `highlights.txt` | 五条 Highlights |
| `TAME_LaTeX_Source.zip`（完整交付包内） | LaTeX 源文件、图表及四份文档编译入口 |
| `TAME_Numerical_Supplement.zip`（完整交付包内） | 原始返回聚合结果、来源记录、重建代码及固定输入 |

系统的具体上传分类以实际投稿页面为准；不将整份内部交付 ZIP 当作主稿上传。

## 数据与版本来源

- 本轮代码基线：`db485b3ac7721d191576801286a4b046ae9a982b`。
- 正式实验参考提交：`ae9c3927f0e34f86f58005c4ccea9d50ff5a7376`。
- 正式回包：`chemolab_formal_v3_review.zip`。
- 回包 SHA-256：`b4b30da53a8ea1a89d681d737571e1f971d7a51dfbd301075febbe70897d6083`。
- 发布分支：`paper2-tame-resubmission-20261112`。
- 完整原始聚合输入包括 343,680 行单元指标、77,328 行面板指标和 7,700 行成对比较指标，压缩前字节与回包一致。来源清单见 `inputs/SOURCE_MANIFEST.json` 及数值附件。

## 重建与核验

使用 Python 3.10 或更新版本，安装 `requirements.txt`，另需提供 `pdflatex` 的 TeX Live/MiKTeX。运行：

```text
python rebuild.py
```

该命令读取固定聚合结果，重建表格和两张图，验证激活条件，然后将四份 LaTeX 文档分别编译两次。它不训练模型，也不重跑分子层 bootstrap。原回包不含模型权重和逐分子预测；这些操作须使用执行机保存的完整导出。

核验文件：`NUMERICAL_VERIFICATION.json`（30 组 EPA 汇总、每组 7 项指标）、`ACTIVATION_IDENTITY_CHECK.json`（80,000 次成员条件对照与边界/排序检查）、`BUILD_VERIFICATION.json`（四份文档编译）、`COLLECTOR_VERIFICATION.json`（取件脚本）、`RUNTIME_SOURCE_VERIFICATION.json`（实际运行源码与相关行为检查）。

GitHub 保留可重建本稿的精简固定输入；完整原始聚合 CSV 随数值附件交付。两者的用途和变换记录在来源清单中。
