# TAME 实质大修交付 — 2026-11-12

稿件：**Separating coverage gains from error targeting in molecular prediction sets**

目标期刊：Chemometrics and Intelligent Laboratory Systems。原稿号 CHEMOLAB-D-26-01118；本轮按大幅修改后的新稿准备，尚未向期刊提交。

## 本轮完成内容

正文、补充材料、对原审稿意见的逐点回复、投稿信、Highlights 均已完成重写和编译。主线是同类别、同弃判预算的比较，以及加权集合追加标签的精确触发条件。全文区分门槛通过、实际追加标签和错误救回。

历史六测定面板的最弱类覆盖从 83.739% 升至 85.104%，但同预算的遗漏标签概率排序复现了这部分增益。因此本稿的贡献是解释覆盖增益和错误定位能力的关系，而不是宣称 TAME 在相同成本下独有优势。960 个预测器单元的正式回包、完整参照配置、额外测定家族与反向结果均保留。

## 正式提交前的实际源码核对

**当前状态：成稿及数值附件已完成；CoDrug 和 KMM-CP 两份实际运行源码尚待取回并核对。**

正式回包的来源记录显示，执行机上的两份源码相对参考提交发生过修改，但回包未附这两份实际文件。当前仓库中的旧实现不能冒充实际运行版本。请在产生正式 v3 结果的原电脑/服务器收集文件，不需要重训。

1. 将本目录的 `collect_runtime_sources.py` 复制到原运行仓库根目录。
2. 在该目录执行：

   ```text
   python collect_runtime_sources.py --repo . --output TAME_Runtime_Sources_20261112.zip
   ```

3. 回传生成的 `TAME_Runtime_Sources_20261112.zip`。脚本只读取两份源码、相关 Git 差异及可选来源记录，不切换分支、不覆盖源码、不启动训练。

收集前请保持实际运行文件，不先执行分支切换或用新版本覆盖这两份文件。若脚本报告哈希不一致，仍回传 ZIP，由我们核对差异，不自行重跑实验。若另存有 scaffold 修复前的 `codrug.py`，可额外使用 `--pre-fix-codrug 文件路径`；已知结果目录可用 `--result-dir 目录路径`。

核对后需要将实际实现及必要差异说明纳入最终复现附件，再进行正式提交。新修订稿沿用原稿的作者、单位和通讯信息；本交付不声称两位作者已重新批准新版本。

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

核验文件：`NUMERICAL_VERIFICATION.json`（30 组 EPA 汇总、每组 7 项指标）、`ACTIVATION_IDENTITY_CHECK.json`（80,000 次成员条件对照与边界/排序检查）、`BUILD_VERIFICATION.json`（四份文档编译）、`COLLECTOR_VERIFICATION.json`（取件脚本）。

GitHub 保留可重建本稿的精简固定输入；完整原始聚合 CSV 随数值附件交付。两者的用途和变换记录在来源清单中。
