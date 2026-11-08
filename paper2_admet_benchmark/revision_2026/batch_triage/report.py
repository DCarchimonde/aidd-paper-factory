"""Per-batch assay-cost reporting; audit repeats are conditional Monte Carlo.

No task, class, predictor or risk cap is pooled into a purported population
effect. Missing-class risks remain NA. A completed report is not a scientific
acceptance gate, and synthetic pilot rows never become molecular evidence.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from scipy.stats import t as student_t


GROUP = ["case_id", "predictor", "method", "eps_fn", "eps_fp", "audit_budget"]
PAIR_GROUP = ["case_id", "predictor", "reference", "eps_fn", "eps_fp", "audit_budget"]
CELL = ["case_id", "predictor", "eps_fn", "eps_fp"]
PAIR_KEY = ["case_id", "predictor", "audit_seed", "eps_fn", "eps_fp", "audit_budget"]
METHOD_ORDER = ["joint_lookahead", "joint_uniform", "joint_entropy", "joint_neyman",
                "threshold_lookahead", "loose_lookahead", "score_only_lookahead", "uniform_cs_lookahead"]
SHORT_NAMES = {"joint_lookahead": "Joint + lookahead", "joint_uniform": "Joint + uniform",
               "joint_entropy": "Joint + entropy", "joint_neyman": "Joint + Neyman",
               "threshold_lookahead": "Threshold + lookahead", "loose_lookahead": "Loose + lookahead",
               "score_only_lookahead": "Score only + lookahead", "uniform_cs_lookahead": "Uniform CS + lookahead"}
VIOLATION = "violated_realized_class_risk"


def _boolean(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    raise ValueError("risk-violation flags must be explicit booleans")


def _frame(rows):
    data = pd.DataFrame(rows).copy()
    required = {*GROUP, "audit_seed", "n_target", "queried", "saved_fraction",
                "total_assays_required", "fn_rate", "fp_rate", VIOLATION}
    if data.empty or not required.issubset(data.columns):
        raise ValueError(f"nonempty report rows require fields {sorted(required)}")
    if data[GROUP + ["audit_seed", "n_target"]].isna().any().any():
        raise ValueError("report group identities and target counts cannot be missing")
    if data.duplicated(GROUP + ["audit_seed"]).any():
        raise ValueError("duplicate exact-case/method/budget/audit-seed observation")
    numeric = ["n_target", "audit_budget", "queried", "saved_fraction", "total_assays_required",
               "fn_rate", "fp_rate", "eps_fn", "eps_fp"]
    for optional in ("fn_upper", "fp_upper", "class_0_n", "class_1_n"):
        if optional not in data:
            data[optional] = np.nan
        numeric.append(optional)
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="raise")
        if np.isinf(data[column].to_numpy(dtype=float)).any():
            raise ValueError(f"nonfinite numeric value in {column}")
    for column in ("n_target", "audit_budget", "queried", "saved_fraction", "total_assays_required"):
        if data[column].isna().any():
            raise ValueError(f"required assay accounting cannot be missing: {column}")
    if (data["n_target"] <= 0).any() or data.groupby(CELL)["n_target"].nunique().max() != 1:
        raise ValueError("each exact batch must have one positive fixed target size")
    if ((data["audit_budget"] < 0) | (data["audit_budget"] > data["n_target"]) |
        (data["queried"] < 0) | (data["queried"] > data["audit_budget"]) |
        (data["total_assays_required"] < data["queried"]) |
        (data["total_assays_required"] > data["n_target"])).any():
        raise ValueError("audit and total-assay accounting is inconsistent")
    if not np.allclose(data["saved_fraction"], 1 - data["total_assays_required"] / data["n_target"],
                       rtol=0, atol=1e-12):
        raise ValueError("saved fraction must count all queried and remaining deferred assays")
    for risk in ("fn_rate", "fp_rate"):
        values = data[risk].dropna()
        if ((values < 0) | (values > 1)).any():
            raise ValueError("class risks must be probabilities or NA")
    for label, risk in ((1, "fn_rate"), (0, "fp_rate")):
        # Absent classes must not silently become perfect zero-error results.
        data.loc[data[f"class_{label}_n"] == 0, risk] = np.nan
    data[VIOLATION] = data[VIOLATION].map(_boolean)
    data["planned_audit_fraction"] = data["audit_budget"] / data["n_target"]
    data["total_assay_fraction"] = data["total_assays_required"] / data["n_target"]
    return data


def aggregate_rows(rows) -> pd.DataFrame:
    """Aggregate only random audit repeats of the exact same finite-batch case."""
    data = _frame(rows)
    output = []
    for keys, group in data.groupby(GROUP, sort=True, dropna=False):
        row = dict(zip(GROUP, keys))
        n = len(group)
        std = float(group["saved_fraction"].std(ddof=1)) if n >= 2 else np.nan
        row.update(n_audit_seeds=n, n_target=int(group["n_target"].iloc[0]),
                   planned_audit_fraction=float(group["planned_audit_fraction"].iloc[0]),
                   mean_queried=float(group["queried"].mean()),
                   mean_saved_fraction=float(group["saved_fraction"].mean()),
                   mc_std_saved_fraction=std,
                   mc_se_saved_fraction=std / math.sqrt(n) if n >= 2 else np.nan,
                   mean_total_assays_required=float(group["total_assays_required"].mean()),
                   mean_total_assay_fraction=float(group["total_assay_fraction"].mean()),
                   pointwise_violation_frequency=float(group[VIOLATION].mean()))
        for risk in ("fn_rate", "fp_rate", "fn_upper", "fp_upper"):
            valid = group[risk].dropna()
            row[f"n_defined_{risk}"] = len(valid)
            row[f"mean_{risk}"] = float(valid.mean()) if len(valid) else np.nan
        row["inference_scope"] = "audit_randomization_conditional_on_fixed_batch_and_predictor"
        if "n_strata" in group:
            if group["n_strata"].nunique(dropna=False) != 1:
                raise ValueError("the fixed partition cannot vary across audit seeds")
            row["n_strata"] = int(group["n_strata"].iloc[0])
        output.append(row)
    return pd.DataFrame(output)


def summarize_pairs(paired) -> pd.DataFrame:
    """Pointwise paired t intervals quantify MC mean, not molecular transfer."""
    fields = [*PAIR_GROUP, "n_paired_audit_seeds", "mean_delta_saved_fraction",
              "mc_std_delta_saved_fraction", "mc_se_delta_saved_fraction",
              "pointwise_mc_t95_low", "pointwise_mc_t95_high", "inference_scope"]
    data = pd.DataFrame(paired)
    if data.empty:
        return pd.DataFrame(columns=fields)
    required = {*PAIR_GROUP, "audit_seed", "delta_saved_fraction"}
    if not required.issubset(data.columns) or data[list(required)].isna().any().any():
        raise ValueError("paired comparisons require complete same-case pairing keys and savings differences")
    if data.duplicated(PAIR_GROUP + ["audit_seed"]).any():
        raise ValueError("duplicate paired audit-seed comparison")
    data["delta_saved_fraction"] = pd.to_numeric(data["delta_saved_fraction"], errors="raise")
    if not np.isfinite(data["delta_saved_fraction"]).all():
        raise ValueError("paired savings differences must be finite")
    output = []
    for keys, group in data.groupby(PAIR_GROUP, sort=True, dropna=False):
        row = dict(zip(PAIR_GROUP, keys))
        values = group["delta_saved_fraction"]
        n, mean = len(values), float(values.mean())
        std = float(values.std(ddof=1)) if n >= 2 else np.nan
        se = std / math.sqrt(n) if n >= 2 else np.nan
        half = float(student_t.ppf(.975, n - 1)) * se if n >= 2 else np.nan
        row.update(n_paired_audit_seeds=n, mean_delta_saved_fraction=mean,
                   mc_std_delta_saved_fraction=std, mc_se_delta_saved_fraction=se,
                   pointwise_mc_t95_low=mean - half, pointwise_mc_t95_high=mean + half,
                   inference_scope="pointwise_approximate_t_interval_for_audit_MC_mean_fixed_batch_not_population_effect")
        output.append(row)
    return pd.DataFrame(output, columns=fields)


def _validate_pairs(data, paired):
    lookup = {tuple(row[k] for k in PAIR_KEY) + (row["method"],): row
              for row in data.to_dict("records")}
    for pair in paired:
        key = tuple(pair[k] for k in PAIR_KEY)
        if pair["reference"] == "joint_lookahead":
            raise ValueError("self-comparison is not a baseline contrast")
        focal = lookup.get(key + ("joint_lookahead",))
        reference = lookup.get(key + (pair["reference"],))
        if focal is None or reference is None:
            raise ValueError("paired comparison does not match both exact observed audit-seed rows")
        difference = focal["saved_fraction"] - reference["saved_fraction"]
        if not math.isclose(difference, float(pair["delta_saved_fraction"]), abs_tol=1e-12, rel_tol=0):
            raise ValueError("paired savings difference disagrees with observed assay accounting")


def _filename(parts):
    value = "__".join(map(str, parts))
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", value)[:140]
    return stem + "__" + hashlib.sha256(value.encode()).hexdigest()[:8]


def _plot_case(group, destination: Path, mode: str) -> list[str]:
    first = group.iloc[0]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True)
    methods = sorted(group["method"].unique(), key=lambda name: (METHOD_ORDER.index(name)
                     if name in METHOD_ORDER else len(METHOD_ORDER), name))
    colors = plt.get_cmap("tab10")
    panels = [("mean_saved_fraction", "Assays saved / original batch", (0, 1)),
              ("mean_total_assay_fraction", "Queries + deferred assays / batch", (0, 1)),
              ("mean_fn_rate", "False-negative fraction among all actives", None),
              ("mean_fp_rate", "False-positive fraction among all inactives", None)]
    for index, method in enumerate(methods):
        points = group[group["method"] == method].sort_values("audit_budget")
        for axis, (metric, ylabel, limits) in zip(axes.flat, panels):
            axis.plot(points["planned_audit_fraction"], points[metric], marker="o", markersize=3,
                      linewidth=1.5, color=colors(index % 10), label=SHORT_NAMES.get(method, method))
    for axis, (metric, ylabel, limits) in zip(axes.flat, panels):
        axis.set_ylabel(ylabel, fontsize=9)
        axis.set_xlabel("Planned audit budget / original batch", fontsize=9)
        axis.xaxis.set_major_formatter(PercentFormatter(1))
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.grid(alpha=.2)
        axis.axvline(.2, color=".6", linewidth=.8, linestyle=":")
        if limits:
            axis.set_ylim(*limits)
        if metric in ("mean_fn_rate", "mean_fp_rate"):
            cap = first["eps_fn"] if metric == "mean_fn_rate" else first["eps_fp"]
            axis.axhline(cap, color="black", linewidth=1, linestyle="--")
            valid = group[metric].dropna()
            axis.set_ylim(0, min(1., max(cap * 1.5, float(valid.max()) * 1.15 if len(valid) else 0)))
            if not len(valid):
                axis.text(.5, .5, "NA: class-risk denominator absent", transform=axis.transAxes,
                          ha="center", fontsize=9)
    kind = "SYNTHETIC PILOT — NOT MOLECULAR EVIDENCE" if mode == "pilot" else "RETROSPECTIVE FIXED-BATCH EVALUATION"
    fig.suptitle(f"{kind}\n{first['case_id']} | {first['predictor']} | caps: FN={first['eps_fn']:g}, FP={first['eps_fp']:g}",
                 fontsize=11, y=.98)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .035), ncol=4, fontsize=8)
    fig.text(.5, .012, "Audit-seed means conditional on this fixed batch; risk curves are not certificate upper bounds.",
             ha="center", fontsize=8)
    fig.subplots_adjust(top=.87, bottom=.19, hspace=.32, wspace=.26)
    paths = []
    for suffix in (".png", ".pdf"):
        path = destination.with_suffix(suffix)
        fig.savefig(path, dpi=160, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def _number(value, *, percent=False):
    if pd.isna(value):
        return "NA"
    return f"{100 * value:.2f}%" if percent else f"{value:.2f}"


def build_report(rows, paired, output, mode):
    """Write reviewable summaries and figures after the declared matrix completes."""
    if mode not in ("pilot", "screen", "formal"):
        raise ValueError("report mode must be pilot, screen or formal")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = _frame(rows)
    _validate_pairs(data, paired)
    aggregate = aggregate_rows(rows)
    pairs = summarize_pairs(paired)
    main = aggregate[aggregate["audit_budget"] == np.floor(.2 * aggregate["n_target"])].copy()
    main["requested_primary_audit_fraction"] = .2
    primary_pair = pairs.rename(columns={"reference": "method"})
    if not primary_pair.empty:
        main = main.merge(primary_pair, on=GROUP, how="left", suffixes=("", "_paired"), validate="one_to_one")
    aggregate.to_csv(output / "aggregate.csv", index=False, na_rep="")
    pairs.to_csv(output / "paired_summary.csv", index=False, na_rep="")
    main.to_csv(output / "main_20pct.csv", index=False, na_rep="")
    figure_folder = output / "figures"
    figure_folder.mkdir(exist_ok=True)
    figures = []
    for keys, group in aggregate.groupby(CELL, sort=True):
        figures.extend(_plot_case(group, figure_folder / _filename(keys), mode))

    status = ("当前仅完成合成数据软件检查；以下数字不是分子实验结果。" if mode == "pilot" else
              "当前结果来自既有公开任务的回顾性固定批次实验；不属于新盲测或未来批次保证。")
    lines = ["# 有限批次审计结果读表说明", "", status, "",
             f"运行模式：`{mode}`。共 {data['case_id'].nunique()} 个批次；汇总保留每个预测器、风险阈值、方法和预算。",
             "总实验数 = 已查询标签数 + 仍需实验的分子数；节省比例 = 1 − 总实验数 / 原始批次数。已审计分子不能再次计为自动分类节省。", "",
             "`aggregate.csv` 中均值、标准差及标准误只概括同一批次上审计随机性的 Monte Carlo 重复。它们不是独立化学样本，跨任务、随机/骨架批次和模型均未合并。", "",
             "`paired_summary.csv` 中差值为 joint_lookahead − reference 的节省比例；正值仅表示该精确条件下节省较多。配对保留同一批次、预测器、审计 seed、双类阈值和预算。", "",
             "n≥2 才给出近似的逐点 t 区间；n=1 的标准差、标准误和区间留空。区间描述固定批次的审计 MC 均值，未校正多重比较，也不表示任务总体效果或跨化学空间泛化。", "",
             "类不存在时对应 fn_rate/fp_rate 保留 NA，同时输出有效重复数；NA 不能解释为零错误。风险曲线画的是经验率均值，不是证书上界。", "",
             "pointwise_violation_frequency 是该预算点任一有定义类风险超阈值的频率；不能把它当作跨全部时间点的联合失效频率。", "",
             "## 固定 20% 审计预算：所有方法", "",
             "与执行器一致，预算取 floor(0.20 × N)；它是审计上限，不是最终总实验预算。以下各行均保留原批次，不按类别稀有程度加权合并。", "",
             "| 批次 | 预测器 | FN/FP 阈值 | 方法 | 审计重复 | 平均节省 | 平均总实验数 | FN / FP 经验率 | 点位超阈频率 |",
             "|---|---|---|---|---:|---:|---:|---|---:|"]
    if main.empty:
        lines.append("| 未提供20%预算检查点 | — | — | — | — | — | — | — | — |")
    for row in main.to_dict("records"):
        case = str(row["case_id"]).replace("|", "\\|")
        lines.append(f"| {case} | {row['predictor']} | {row['eps_fn']:g}/{row['eps_fp']:g} | {row['method']} | "
                     f"{row['n_audit_seeds']} | {_number(row['mean_saved_fraction'], percent=True)} | "
                     f"{_number(row['mean_total_assays_required'])} | {_number(row['mean_fn_rate'], percent=True)} / "
                     f"{_number(row['mean_fp_rate'], percent=True)} | {_number(row['pointwise_violation_frequency'], percent=True)} |")
    lines.extend(["", "## 图表与比较顺序", "",
                  "`figures/` 为每个精确批次、预测器与风险阈值组合分别输出 PNG 和 PDF：节省比例、总实验比例、两类经验错误曲线。pilot 图显式标记为合成软件检查。", "",
                  "优先查看 joint_lookahead 与同证书 joint_uniform 的配对差值，再查看 entropy、Neyman、阈值策略、分离分子/分母界的保守对照、score-only 与 uniform-CS 对照，并比较增益在不同预算和任务间的一致性。", "",
                  "n_strata 记录实际非空层数：分数×相似度最多8层，score-only最多8层，空交叉单元会使前者少于8层。", "",
                  "正类/负类对应当前 benchmark 的测定标签；成本按获得一个参考标签所需的一次测定计。", ""])
    (output / "READOUT_CN.md").write_text("\n".join(lines), encoding="utf-8")
    return {"aggregate": str(output / "aggregate.csv"), "paired_summary": str(output / "paired_summary.csv"),
            "main_20pct": str(output / "main_20pct.csv"), "readout": str(output / "READOUT_CN.md"),
            "figures": figures}
