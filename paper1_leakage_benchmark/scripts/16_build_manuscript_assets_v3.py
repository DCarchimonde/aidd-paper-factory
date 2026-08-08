from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "paper1_leakage_benchmark"
TABLE_DIR = PAPER_DIR / "results" / "tables"
PARENT_DIR = PAPER_DIR / "results" / "parent_fragment_sensitivity_v3" / "tables"
FIG_DIR = PAPER_DIR / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PRIMARY = TABLE_DIR / "primary_inference_summary_v3.csv"
SINGLETON = TABLE_DIR / "acyclic_singleton_sensitivity_v3.csv"
PARENT_COMPARISON = PARENT_DIR / "parent_fragment_vs_main_comparison_v3.csv"

MODEL_ORDER = {"LR": 0, "Ridge": 0, "RF": 1, "XGB": 2}
DATASET_ORDER = {
    "BACE": 0,
    "BBBP": 1,
    "ClinTox": 2,
    "HIV": 3,
    "ESOL": 4,
    "FreeSolv": 5,
}


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required result file: {path}")


def save(fig: plt.Figure, stem: str) -> None:
    pdf = FIG_DIR / f"{stem}.pdf"
    png = FIG_DIR / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(pdf)
    print(png)


def figure1_framework() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    ax.set_axis_off()
    labels = [
        "Audited molecular\nuniverse",
        "Explicit scaffold\nsemantics",
        "Fixed-budget target-blind\ncandidate pool",
        "Exact-size paired\nbenchmark perturbation",
        "Frozen manifests +\npartition-level inference",
        "Protocol-sensitivity\naudit",
    ]
    xs = np.linspace(0.06, 0.94, len(labels))
    width = 0.135
    height = 0.34
    y = 0.50
    for x, label in zip(xs, labels):
        box = FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.015",
            linewidth=1.2,
            fill=False,
            transform=ax.transAxes,
        )
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center", fontsize=10, transform=ax.transAxes)
    for left, right in zip(xs[:-1], xs[1:]):
        ax.annotate(
            "",
            xy=(right - width / 2 - 0.008, y),
            xytext=(left + width / 2 + 0.008, y),
            xycoords=ax.transAxes,
            arrowprops={"arrowstyle": "->", "linewidth": 1.2},
        )
    ax.text(
        0.5,
        0.93,
        "Benchmark construction treated as a chemometric measurement process",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.12,
        "Primary contrast: same dataset, same seed, same candidate pool, same test size; target balance is the designed perturbation",
        ha="center",
        va="center",
        fontsize=10,
        transform=ax.transAxes,
    )
    save(fig, "figure1_audit_framework_v3")


def _sort_primary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dataset_order"] = out["dataset"].map(DATASET_ORDER)
    out["model_order"] = out["model"].map(MODEL_ORDER)
    return out.sort_values(["dataset_order", "model_order"], kind="mergesort").reset_index(drop=True)


def _forest(ax: plt.Axes, df: pd.DataFrame, xlabel: str) -> None:
    y = np.arange(len(df))
    means = df["mean_effect"].astype(float).to_numpy()
    lo = df["bootstrap_ci_low"].astype(float).to_numpy()
    hi = df["bootstrap_ci_high"].astype(float).to_numpy()
    labels = [f"{d} – {m}" for d, m in zip(df["dataset"], df["model"])]
    ax.errorbar(
        means,
        y,
        xerr=np.vstack([means - lo, hi - means]),
        fmt="o",
        capsize=3,
        linewidth=1.1,
    )
    ax.axvline(0.0, linewidth=1.0, linestyle="--")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", linewidth=0.4, alpha=0.35)


def figure2_primary() -> None:
    require(PRIMARY)
    df = _sort_primary(pd.read_csv(PRIMARY, keep_default_na=False))
    if len(df) != 18:
        raise AssertionError(f"Expected 18 primary cells; found {len(df)}")
    cls = df.loc[df["task_type"].eq("classification")].copy()
    reg = df.loc[df["task_type"].eq("regression")].copy()
    if len(cls) != 12 or len(reg) != 6:
        raise AssertionError("Primary task counts are not 12 classification + 6 regression")

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.6), gridspec_kw={"width_ratios": [1.15, 1.0]})
    _forest(axes[0], cls, "AUC effect: balanced − size-matched")
    axes[0].set_title("Classification")
    _forest(axes[1], reg, "RMSE improvement: size-matched − balanced")
    axes[1].set_title("Regression")
    fig.suptitle("Exact-size paired target-balance effects over 20 unique partition pairs", fontsize=13)
    fig.tight_layout()
    save(fig, "figure2_primary_effects_v3")


def figure3_acyclic() -> None:
    require(PRIMARY)
    require(SINGLETON)
    primary = pd.read_csv(PRIMARY, keep_default_na=False)
    primary = primary.loc[primary["dataset"].isin(["ESOL", "FreeSolv"])].copy()
    singleton = pd.read_csv(SINGLETON, keep_default_na=False).copy()
    if len(primary) != 6 or len(singleton) != 6:
        raise AssertionError("Expected six primary and six singleton regression cells")

    primary = primary[["dataset", "model", "mean_effect", "bootstrap_ci_low", "bootstrap_ci_high"]].copy()
    primary["semantics"] = "single-group (primary)"
    singleton = singleton.rename(
        columns={
            "mean_effect_positive_is_balanced_better": "mean_effect",
            "bootstrap_ci_low": "bootstrap_ci_low",
            "bootstrap_ci_high": "bootstrap_ci_high",
        }
    )[["dataset", "model", "mean_effect", "bootstrap_ci_low", "bootstrap_ci_high"]]
    singleton["semantics"] = "singleton sensitivity"

    rows = []
    for dataset in ("ESOL", "FreeSolv"):
        for model in ("Ridge", "RF", "XGB"):
            for semantics, frame in (("single-group (primary)", primary), ("singleton sensitivity", singleton)):
                row = frame.loc[frame["dataset"].eq(dataset) & frame["model"].eq(model)]
                if len(row) != 1:
                    raise AssertionError(f"Missing {dataset}/{model}/{semantics}")
                rows.append(row.iloc[0].to_dict())
    plot = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.1), sharex=False)
    for ax, dataset in zip(axes, ("ESOL", "FreeSolv")):
        sub = plot.loc[plot["dataset"].eq(dataset)].copy()
        base_y = np.arange(3)
        for offset, semantics, marker in ((-0.10, "single-group (primary)", "o"), (0.10, "singleton sensitivity", "s")):
            vals = sub.loc[sub["semantics"].eq(semantics)].copy()
            vals["model_order"] = vals["model"].map(MODEL_ORDER)
            vals = vals.sort_values("model_order")
            means = vals["mean_effect"].astype(float).to_numpy()
            lo = vals["bootstrap_ci_low"].astype(float).to_numpy()
            hi = vals["bootstrap_ci_high"].astype(float).to_numpy()
            ax.errorbar(
                means,
                base_y + offset,
                xerr=np.vstack([means - lo, hi - means]),
                fmt=marker,
                capsize=3,
                linewidth=1.1,
                label=semantics,
            )
        ax.axvline(0.0, linestyle="--", linewidth=1.0)
        ax.set_yticks(base_y, ["Ridge", "RF", "XGB"])
        ax.invert_yaxis()
        ax.set_title(dataset)
        ax.set_xlabel("RMSE improvement: size-matched − balanced")
        ax.grid(axis="x", linewidth=0.4, alpha=0.35)
    axes[0].legend(loc="best", fontsize=9)
    fig.suptitle("Regression effect depends on the treatment of acyclic scaffold identity", fontsize=13)
    fig.tight_layout()
    save(fig, "figure3_acyclic_sensitivity_v3")


def figure4_fragment() -> None:
    require(PARENT_COMPARISON)
    df = pd.read_csv(PARENT_COMPARISON, keep_default_na=False)
    if len(df) != 9:
        raise AssertionError(f"Expected nine dominant-fragment comparison cells; found {len(df)}")
    df["dataset_order"] = df["dataset"].map(DATASET_ORDER)
    df["model_order"] = df["model"].map(MODEL_ORDER)
    df = df.sort_values(["dataset_order", "model_order"], kind="mergesort").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9.3, 5.8))
    y = np.arange(len(df))
    main = df["main_mean_effect"].astype(float).to_numpy()
    parent = df["parent_mean_effect"].astype(float).to_numpy()
    for yi, x1, x2 in zip(y, main, parent):
        ax.plot([x1, x2], [yi, yi], linewidth=1.0)
    ax.scatter(main, y, marker="o", label="source-faithful primary")
    ax.scatter(parent, y, marker="s", label="dominant-fragment sensitivity")
    ax.axvline(0.0, linestyle="--", linewidth=1.0)
    ax.set_yticks(y, [f"{d} – {m}" for d, m in zip(df["dataset"], df["model"])])
    ax.invert_yaxis()
    ax.set_xlabel("AUC effect: balanced − size-matched")
    ax.set_title("Classification point estimates are representation-sensitive")
    ax.grid(axis="x", linewidth=0.4, alpha=0.35)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    save(fig, "figure4_dominant_fragment_sensitivity_v3")


def main() -> None:
    print("Building Paper 1 manuscript assets from frozen v3 result tables")
    figure1_framework()
    figure2_primary()
    figure3_acyclic()
    figure4_fragment()
    print("\nMANUSCRIPT ASSETS V3 COMPLETED")


if __name__ == "__main__":
    main()
