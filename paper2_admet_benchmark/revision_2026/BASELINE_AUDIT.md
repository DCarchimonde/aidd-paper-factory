# CHEMOLAB-D-26-01118：投稿版本独立核查

核查对象：`d2e05ad1b948ea8c19b8d67783a659a385c54240`，旧分支 `paper2-racer-c4-development-2026`。日期：2026-11-04。以下旧文件行号均指该提交。新实验尚未完成；本文不把修订方案当作结果。

## 1. 首要裁定

审稿意见的主要问题成立：原稿没有证明 TAME 在相同决策成本下优于简单拒判或较强的 shift-aware 方法；保护 label 0 的开发规则与前文 ClinTox 阳性漏覆盖的动机之间缺少明确的筛选任务连接。仅扩大集合的覆盖增益不能单独支持方法创新或实际价值。

但有两点应准确回应，不能为了表示接受意见而承认并未发生的事实：

1. 原稿确已说明没有任意 shift 的条件覆盖保证，且不是新覆盖定理（`paper2_latex/sections/methods.tex:127`、`:157`，`discussion.tex:47`）。问题是研究定位、证据强度与篇幅安排，不是隐瞒了一个已经证明的保证。
2. Stage I 与 Stage II 的**分类**非一致性分数均为 LAC：`1-p_y`（`methods.tex:61–69`、`:114`）。不同的是模型组合与拟合流程，以及 Stage I 另含回归残差分数。重写应清楚分开任务、统一比较协议，不把分类分数说成已经改变。

## 2. 已有结果揭示的强基线问题

旧代码确实计算了普通 stack Mondrian、equal-logit Mondrian 以及三种 density-weighted Mondrian（`run_prospective_racer_c4.py:345–351`）。因此“只有 ordinary comparator”并不准确；准确说法是：**主要差值推断只有相对普通 stack Mondrian，其他强方法没有在同一确认方案中充分比较**。

`results/racer_c4_independent_final/final_summary_all_endpoints.csv:2–7` 的 12 端点 × 5 seeds 描述性均值为：

| 方法 | Minimum-class coverage | MacroCSY | Ambiguity |
|---|---:|---:|---:|
| TAME | 0.84295754 | 0.41329278 | 0.50400570 |
| 普通 equal-logit Mondrian | 0.84411917 | 0.41443021 | 0.49573931 |
| 普通 stack Mondrian | 0.83345642 | 0.42375486 | 0.49050782 |
| Physchem-weighted Mondrian | 0.83602184 | 0.43088930 | 0.46977063 |
| Score-weighted Mondrian | 0.82484424 | 0.42524889 | 0.45952473 |

equal-logit 的覆盖和 MacroCSY 均略高于 TAME，ambiguity 更低。这是已有次要面板上的描述性支配，不等于显著优于，更不能由此推断六个主端点也有同样结果。六主端点的 equal-logit 配对差值及其区间需从逐行预测重新计算；当前提交中的 `primary_endpoint_means.csv` 只提供 TAME 对 ordinary stack 的差值，不能补造缺失比较。

开发集也已有信息：`results/racer_c4_development/development_summary.csv` 中 physchem-weighted 的 minimum coverage 为 0.850062、MacroCSY 为 0.411991；TAME 为 0.848173、0.375612。两者类别覆盖侧重不同，这再次说明只比较“最弱类 + 固定宽容的效率界限”不足以说明 TAME 的决策价值。开发集不能与 EPA 合并当作独立验证。

修订主比较应包括上述全部已有方法，并增加：同预算 confidence defer、相似性 defer、随机 defer；标准加权 CP；有可核实实现与一致数据访问条件的 CoDrug/KMM-CP 等。不可把自写的通用 KMM 权重算法直接命名为完整 KMM-CP 复现。

## 3. 标签语义与科学问题

`scripts/racer_c/prepare_tox21_challenge.py:324` 明确记载 `1=active; 0=inactive; absent property=unavailable`。Tox21 的 assay active 不是“已证实临床有毒”，inactive 也不是“安全”。

旧 TAME 固定保护 0，依据是开发集平均 coverage deficit（`configs/racer_c4/prospective_lock_v1.yaml:116–124`）。在 baseline singleton `{1}` 上添加 0，会把 active 判断变成 defer；它能减少 true-inactive 被自信判为 active 的错误。它**不直接解决 active 被自信判为 inactive 的漏检**。如果修订聚焦避免漏掉 assay-active，则应事先固定保护 1，并明确这是新的候选规则，重新评估，不可重写旧算法历史。

同预算比较的具体含义：在相同 baseline 的可扩张 singleton 中，按照 label-free 排序选相同数量分子变成 `{0,1}`；比较各真类 wrong-singleton exposure、correct-singleton yield、总 defer 和类别 defer。预算不由 target 标签决定；目标标签只用于评价。增加双标签本身不能被计作错误减少的无成本收益。

## 4. 已确认实现问题与局限

### P1：Stage I 小校准类的有限样本分位数被截断

`scripts/13_mondrian_conformal_analysis.py:129–136` 计算 `ceil((n+1)*(1-alpha))` 后将 rank 截到 n；`scripts/07_conformal_analysis.py:94–103` 同样处理 marginal 分位数。正文 `methods.tex:67` 也写了这个截断。标准保守构造在 rank > n 时应为 +∞，不能取最大的有限校准分数。alpha=0.10 时，正样本数 n<9 会触发；alpha=0.20 时，正 n<4 会触发。n=0 的旧 Mondrian runner 会跳过整个 cell，需另报遗漏而非给出零影响。

Stage II `scripts/racer_c4/racer_c4_core.py:40–49` 已正确处理 +∞。因此不是所有旧结果都错，也不能直接推断论文数值发生变化。阈值更改是否改变测试集合，必须检查对应的逐行 calibration/test 概率；只有汇总类数时只能判断潜在影响。

新增脚本 `audit_legacy_quantiles.py`：读取旧预测 `results/predictions/*_predictions.csv`，不重新训练，不覆盖任何历史文件。默认只审查 `split_confirm_*` 的分类数据；每个模型/划分分别检查 marginal 和各类 Mondrian；报告阈值变化、真实集合元素变化和新覆盖真标签数。若无逐行文件，尝试读取 `mondrian_conformal_summary_confirmatory.csv` 的类数；若两者均缺失，明确返回 `missing_inputs_cannot_determine_historical_impact`，退出码 2。

```bash
python paper2_admet_benchmark/revision_2026/audit_legacy_quantiles.py \
  --prediction-dir paper2_admet_benchmark/results/predictions \
  --summary paper2_admet_benchmark/results/conformal/mondrian_conformal_summary_confirmatory.csv \
  --output .local/chemolab_revision/legacy_quantile_audit
```

只选择同一个历史运行目录；重复的 endpoint/split/model/role/row_index 会被拒绝，不能把同一预测文件的拷贝当作额外样本。重新运行需指定新的空输出目录。`--mode all` 可额外审查 MVP；MVP 不能代替 confirmatory 的影响结论。

### P1：旧 bootstrap 没有保持跨 assay 的分子依赖

`run_prospective_racer_c4.py:565–644` 对端点重采样后，在各端点内独立按真类重采样化合物，保持五个 seed。五个 seed 没被当作五套独立化合物，这一点是对的；但同一化合物出现在多个 assay 时，独立端点抽样不保留联合结果依赖。函数在 `:605` 只比较 labels 数组而未比较 sample ID，若输入次序被改变且相同标签连续出现，无法充分证明行配对。

新分析应先按 `(endpoint, seed, sample_id, method)` 严格唯一性与键连接，再以全局标准化 structure ID 为 cluster，所有端点共享同次抽样权重；seeds 固定平均。缺失类的 bootstrap draw 应标记不可估，不应 silently 用单类指标替代 minimum-class 指标。原六端点只是固定分析面板，不能由六端点区间宣称覆盖全部 assay universe。旧 CI 保留为历史分析，新敏感性区间另列。

### P2：可运行复现的环境与缓存

旧仓库有源码、摘要、图和清洗 manifests，但不含完整逐行 EPA 预测与训练模型。`results/racer_c4_independent_final/README.md:23` 明确说约 39 MB 预测不提交 git。`environment/requirements.txt` 未固定所有版本；Windows wrapper 固定了部分环境但不等价于跨平台逐位复现保证。

`run_prospective_racer_c4.py:116–121` 若 processed CSV 已存在即直接复用，未在该路径校验其内容与原 raw/清洗版本关系。第二作者应使用干净的新 processed 目录，记录原始输入哈希、清洗后数据哈希和实际依赖版本，避免沿用不明缓存。不能为了匹配历史哈希去调模型或删样本。

## 5. 门限与实验解释

transport checks 的 ESS、clipping、AUC、balance 门限存在于配置 `:107–114`，但“提前固定”本身不证明可迁移性。工作权重在完整 source/target covariates 上拟合；交叉拟合只用于 domain AUC（`racer_c4_core.py:284–305`）。不能称全部工作权重已独立 cross-fitted。以相同目标 X 拟合权重是 transductive 设置，应给全部可比方法同等访问。

开发 gate 的八项中，baseline inclusion、无空集、无新 singleton、类 coverage 不下降、wrong singleton 不增加主要属于构造与实现正确性检查（runner `:541–552`），不是五项独立实证发现。实质经验检查是效率损失和 active-view 比例等。新版应把构造不变量放算法说明，把相同成本下的收益放主结果。

原稿 `methods.tex:157` 已明确 5 pp 是开发选择容忍度，不是临床或监管效用界限，也不构成 CI-based non-inferiority。修订不要恢复旧 machine-readable `noninferiority` 词语作为统计结论。应报告预设预算曲线、配对差值与区间，不按 EPA 结果选择更有利阈值。

## 6. 旧 EPA 证据可以怎样复用

旧 EPA 标签已经打开，重新跑模型、换 seed、换脚本或重新 hash 都不能让它重新变成独立盲测。可以在明确标为 retrospective reanalysis 的阶段上做统一基线、误差排查和消融；修订若要新独立验证，需新的未用于规则选择的 assay family/cohort，明确排重、端点语义和数据角色。没有新队列时，就把结论限制为已知公共数据上的比较研究，不用“重新封存”暗示恢复独立性。

## 7. 本轮已做的验证与尚缺证据

- 旧 `tests/racer_c4` 的 21 项 unittest 在本环境通过（2026-11-04，约 12 秒）；这验证已有数值单元与合同检查，不是科研优势证据。
- 本次未运行真实数据完整训练矩阵；没有生成新 EPA/ADMET 性能结论。
- 新 quantile audit 有有限样本边界、实际集合变化与真覆盖变化区分、重复行拒绝、counts-only 与 missing-input 区分、禁止覆盖输出测试。
- 六主端点 equal-logit 的配对比较、Stage I 小类分位数实际影响、同预算强对照、新队列结果均需要第二作者执行并回传逐行证据。
