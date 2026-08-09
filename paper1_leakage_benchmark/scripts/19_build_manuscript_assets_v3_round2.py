from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "paper1_leakage_benchmark"
TABLE_DIR = PAPER_DIR / "results" / "tables"
PARENT_DIR = PAPER_DIR / "results" / "parent_fragment_sensitivity_v3" / "tables"
FIG_DIR = PAPER_DIR / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PRIMARY = TABLE_DIR / "primary_inference_summary_v3.csv"
SINGLETON = TABLE_DIR / "acyclic_singleton_sensitivity_v3.csv"
PARENT_COMPARISON = PARENT_DIR / "parent_fragment_vs_main_comparison_v3.csv"

DATASETS = ["BACE", "BBBP", "ClinTox", "HIV", "ESOL", "FreeSolv"]
CLASSIFICATION = ["BACE", "BBBP", "ClinTox", "HIV"]
REGRESSION = ["ESOL", "FreeSolv"]
MODELS_CLS = ["LR", "RF", "XGB"]
MODELS_REG = ["Ridge", "RF", "XGB"]
DATASET_N = {"BACE": 1513, "BBBP": 1965, "ClinTox": 1442, "HIV": 41120, "ESOL": 1117, "FreeSolv": 642}
RAW_N = {"BACE": 1513, "BBBP": 2050, "ClinTox": 1484, "HIV": 41127, "ESOL": 1128, "FreeSolv": 642}
TEST_N = {"BACE": 303, "BBBP": 393, "ClinTox": 288, "HIV": 8224, "ESOL": 223, "FreeSolv": 128}
MULTICOMPONENT = {"BBBP": 105, "ClinTox": 14, "HIV": 3086}
SCAFFOLD_CHANGED = {"BBBP": 5, "ClinTox": 1, "HIV": 235}
SIM_LT_090 = {"BBBP": 18, "ClinTox": 5, "HIV": 640}
CONFLICT_GROUPS = {"BBBP": 1, "ClinTox": 1, "HIV": 17}
BUDGET_SINGLE_GROUP = {
    "ESOL": {3000: 0.034622, 5000: 0.018366, 10000: 0.010563, 20000: 0.001787},
    "FreeSolv": {3000: 1.076491, 5000: 1.044452, 10000: 0.787491, 20000: 0.686528},
}
BUDGET_SINGLETON = {
    "ESOL": {100: 0.003687, 300: 0.001535, 500: 0.000600, 1000: 0.000600, 3000: 0.000301, 5000: 0.000194},
    "FreeSolv": {100: 0.019467, 300: 0.006230, 500: 0.005645, 1000: 0.005416, 3000: 0.001609, 5000: 0.000471},
}

C = {
    "ink": "#1F2D35",
    "navy": "#315B73",
    "navy2": "#24485D",
    "teal": "#2B8C82",
    "teal2": "#176B64",
    "cyan": "#75C0C8",
    "orange": "#D58A43",
    "orange2": "#A85F28",
    "sage": "#AFCBBD",
    "pale_teal": "#E9F4F0",
    "pale_blue": "#EBF2F6",
    "pale_orange": "#FBF1E7",
    "pale_gray": "#F4F6F7",
    "gray": "#6C7A82",
    "mid": "#B7C1C6",
    "grid": "#E5EAEC",
    "white": "#FFFFFF",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10.0,
    "axes.titlesize": 11.5,
    "axes.labelsize": 10.0,
    "xtick.labelsize": 9.0,
    "ytick.labelsize": 9.0,
    "legend.fontsize": 8.8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": C["mid"],
    "axes.labelcolor": C["ink"],
    "text.color": C["ink"],
    "xtick.color": C["ink"],
    "ytick.color": C["ink"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.10)
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)
    print(FIG_DIR / f"{stem}.pdf")


def clean_axis(ax: plt.Axes, grid: str | None = None) -> None:
    ax.spines["left"].set_color(C["mid"])
    ax.spines["bottom"].set_color(C["mid"])
    if grid:
        ax.grid(axis=grid, color=C["grid"], linewidth=0.8, zorder=0)


def label_panel(ax: plt.Axes, letter: str, title: str | None = None, x: float = -0.08, y: float = 1.08) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=15, fontweight="bold", ha="left", va="top")
    if title:
        ax.text(0.0, 1.04, title, transform=ax.transAxes, fontsize=11.2, fontweight="bold", ha="left", va="bottom")


def box(ax: plt.Axes, xy: tuple[float, float], w: float, h: float, text: str, face: str, edge: str,
        fontsize: float = 8.8, weight: str = "normal", lw: float = 1.15, radius: float = 0.025) -> None:
    p = FancyBboxPatch(xy, w, h, boxstyle=f"round,pad=0.012,rounding_size={radius}",
                       transform=ax.transAxes, facecolor=face, edgecolor=edge, linewidth=lw)
    ax.add_patch(p)
    ax.text(xy[0] + w/2, xy[1] + h/2, text, transform=ax.transAxes,
            ha="center", va="center", fontsize=fontsize, fontweight=weight, linespacing=1.1)


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = C["gray"], lw: float = 1.4) -> None:
    ax.add_patch(FancyArrowPatch(start, end, transform=ax.transAxes, arrowstyle="-|>",
                                 mutation_scale=13, linewidth=lw, color=color))


def read_primary() -> pd.DataFrame:
    require(PRIMARY)
    df = pd.read_csv(PRIMARY, keep_default_na=False)
    if "inference_label" not in df.columns:
        raise KeyError("primary table requires inference_label")
    return df


def read_singleton() -> pd.DataFrame:
    require(SINGLETON)
    df = pd.read_csv(SINGLETON, keep_default_na=False).copy()
    if "mean_effect_positive_is_balanced_better" in df.columns:
        df = df.rename(columns={"mean_effect_positive_is_balanced_better": "mean_effect"})
    return df


def figure1() -> None:
    fig = plt.figure(figsize=(14.8, 9.2))
    gs = fig.add_gridspec(2, 3, wspace=0.30, hspace=0.40)
    fig.suptitle("Benchmark construction is treated as a chemometric measurement process",
                 fontsize=15.5, fontweight="bold", y=0.985)

    ax = fig.add_subplot(gs[0, 0])
    label_panel(ax, "A", "Audited molecular universe")
    y = np.arange(len(DATASETS))
    vals = [DATASET_N[d] for d in DATASETS]
    cols = [C["navy"] if d in CLASSIFICATION else C["teal"] for d in DATASETS]
    ax.barh(y, vals, color=cols, height=0.60, zorder=2)
    ax.set_yticks(y, DATASETS); ax.invert_yaxis(); ax.set_xscale("log")
    ax.set_xlabel("Clean molecules (log scale)"); clean_axis(ax, "x")
    for yi, v in zip(y, vals):
        ax.text(v*1.07, yi, f"{v:,}", va="center", fontsize=8.3)
    ax.legend(handles=[Rectangle((0,0),1,1,color=C["navy"],label="Classification"),
                       Rectangle((0,0),1,1,color=C["teal"],label="Regression")],
              frameon=False, loc="lower right")

    ax = fig.add_subplot(gs[0, 1]); ax.set_axis_off(); label_panel(ax, "B", "Exact-size paired perturbation")
    box(ax, (0.03,0.70),0.25,0.15,"Target-blind\ncandidate pool",C["pale_blue"],C["navy"],8.7,"bold")
    arrow(ax,(0.29,0.775),(0.39,0.775))
    box(ax,(0.40,0.67),0.25,0.21,"Size-matched\nbaseline",C["pale_gray"],C["mid"],9.0,"bold")
    box(ax,(0.72,0.67),0.25,0.21,"Target-balanced\ncounterpart",C["pale_teal"],C["teal"],9.0,"bold")
    arrow(ax,(0.66,0.775),(0.71,0.775),C["teal2"])
    box(ax,(0.10,0.42),0.80,0.12,"controlled: dataset · seed · candidate pool · scaffold rule",C["white"],C["mid"],8.2)
    box(ax,(0.18,0.22),0.64,0.13,"exactly the same test-set size",C["pale_orange"],C["orange"],9.0,"bold")
    ax.text(0.50,0.08,"Designed perturbation: target-distribution mismatch",transform=ax.transAxes,ha="center",fontsize=8.8,color=C["gray"])

    ax = fig.add_subplot(gs[0, 2]); ax.set_axis_off(); label_panel(ax, "C", "Freeze before model outcomes")
    steps=[("Candidate\nbudget",C["pale_orange"],C["orange"]),("Molecule-level\nmanifest + hash",C["pale_blue"],C["navy"]),("LR / Ridge\nRF · XGB",C["pale_gray"],C["mid"]),("Paired effect\nper partition",C["pale_teal"],C["teal"])]
    x0=0.02
    for i,(txt,fc,ec) in enumerate(steps):
        box(ax,(x0+i*0.245,0.67),0.20,0.18,txt,fc,ec,8.3,"bold")
        if i<3: arrow(ax,(x0+i*0.245+0.205,0.76),(x0+(i+1)*0.245-0.01,0.76))
    box(ax,(0.06,0.39),0.27,0.12,"20 unique\npartition pairs",C["white"],C["mid"],8.3)
    box(ax,(0.365,0.39),0.27,0.12,"10,000 paired\nbootstrap draws",C["white"],C["mid"],8.3)
    box(ax,(0.67,0.39),0.27,0.12,"Wilcoxon +\nHolm correction",C["white"],C["mid"],8.3)
    box(ax,(0.13,0.14),0.74,0.12,"Inference is attached to unique paired partitions—not model seeds",C["pale_teal"],C["teal"],8.4,"bold")

    ax = fig.add_subplot(gs[1, 0]); ax.set_axis_off(); label_panel(ax, "D", "Acyclic scaffold semantics")
    ax.text(0.23,0.84,"single-group",transform=ax.transAxes,ha="center",fontweight="bold",fontsize=9.2)
    ax.text(0.77,0.84,"singleton",transform=ax.transAxes,ha="center",fontweight="bold",fontsize=9.2)
    for k in range(4):
        yy=0.66-k*0.115
        ax.plot([0.07,0.14,0.22,0.30],[yy,yy+0.035,yy-0.015,yy+0.025],transform=ax.transAxes,color=C["navy"],lw=2.1)
        ax.plot([0.59,0.66,0.74,0.82],[yy,yy+0.035,yy-0.015,yy+0.025],transform=ax.transAxes,color=C["teal"],lw=2.1)
        box(ax,(0.86,yy-0.03),0.09,0.06,f"S{k+1}",C["pale_teal"],C["teal"],7.5,radius=0.010)
    box(ax,(0.05,0.11),0.36,0.12,"one ACYCLIC\nscaffold identity",C["pale_blue"],C["navy"],8.5,"bold")
    box(ax,(0.57,0.11),0.38,0.12,"each acyclic molecule\ngets its own identity",C["pale_teal"],C["teal"],8.5,"bold")

    ax = fig.add_subplot(gs[1, 1]); ax.set_axis_off(); label_panel(ax, "E", "Disconnected-component representation")
    ax.text(0.20,0.84,"source-faithful record",transform=ax.transAxes,ha="center",fontweight="bold",fontsize=8.7)
    ax.scatter([0.12,0.20,0.29],[0.61,0.66,0.56],s=[360,210,120],c=[C["teal"],C["cyan"],C["orange"]],transform=ax.transAxes,clip_on=False)
    arrow(ax,(0.37,0.62),(0.55,0.62),C["orange"])
    ax.text(0.46,0.68,"deterministic\nselection",transform=ax.transAxes,ha="center",fontsize=8.0,color=C["gray"])
    ax.scatter([0.70],[0.62],s=[650],c=[C["teal"]],transform=ax.transAxes,clip_on=False)
    ax.text(0.70,0.84,"dominant fragment",transform=ax.transAxes,ha="center",fontweight="bold",fontsize=8.7)
    box(ax,(0.06,0.31),0.88,0.15,"can change fingerprints · scaffold identities · duplicate mappings",C["pale_orange"],C["orange"],8.3,"bold")
    ax.text(0.50,0.15,"Sensitivity representation; not asserted as the chemically correct parent",transform=ax.transAxes,ha="center",fontsize=8.1,color=C["gray"])

    ax = fig.add_subplot(gs[1, 2]); ax.set_axis_off(); label_panel(ax, "F", "What is tested for stability")
    box(ax,(0.07,0.68),0.86,0.14,"Target balance changes benchmark composition",C["pale_blue"],C["navy"],8.8,"bold")
    arrow(ax,(0.50,0.66),(0.50,0.56))
    box(ax,(0.07,0.43),0.86,0.14,"Does predictive evidence change?",C["pale_teal"],C["teal"],9.0,"bold")
    arrow(ax,(0.50,0.41),(0.50,0.31))
    box(ax,(0.07,0.16),0.39,0.13,"Scaffold-semantics\nperturbation",C["pale_orange"],C["orange"],8.5,"bold")
    box(ax,(0.54,0.16),0.39,0.13,"Molecular-representation\nperturbation",C["pale_orange"],C["orange"],8.5,"bold")
    save(fig,"figure1_audit_framework_v3")


def forest_panel(ax: plt.Axes, df: pd.DataFrame, task: str) -> None:
    if task == "classification":
        order=[(d,m) for d in CLASSIFICATION for m in MODELS_CLS]
        xlabel="AUC effect: balanced − size-matched"
    else:
        order=[(d,m) for d in REGRESSION for m in MODELS_REG]
        xlabel="RMSE improvement: size-matched − balanced"
    rows=[]
    for d,m in order:
        r=df[(df.dataset==d)&(df.model==m)]
        if len(r)!=1: raise AssertionError(f"Missing {d}/{m}")
        rows.append(r.iloc[0])
    p=pd.DataFrame(rows).reset_index(drop=True)
    y=np.arange(len(p))
    means=p.mean_effect.astype(float).to_numpy(); lo=p.bootstrap_ci_low.astype(float).to_numpy(); hi=p.bootstrap_ci_high.astype(float).to_numpy()
    labels=[f"{d} · {m}" for d,m in zip(p.dataset,p.model)]
    if task=="classification":
        bounds=[0,3,6,9,12]
    else:
        bounds=[0,3,6]
    for i in range(len(bounds)-1):
        if i%2==0:
            ax.axhspan(bounds[i]-0.5,bounds[i+1]-0.5,color=C["pale_gray"],zorder=0)
    colors=[C["teal2"] if s=="target_balanced_better" else C["navy"] for s in p.inference_label.astype(str)]
    for yi,mn,l,h,col in zip(y,means,lo,hi,colors):
        ax.plot([l,h],[yi,yi],color=col,lw=1.8,zorder=2)
        ax.plot([l,l],[yi-0.11,yi+0.11],color=col,lw=1.1)
        ax.plot([h,h],[yi-0.11,yi+0.11],color=col,lw=1.1)
        ax.scatter(mn,yi,s=38,color=col,edgecolor="white",linewidth=0.6,zorder=3)
    ax.axvline(0,color=C["gray"],ls="--",lw=1.1,zorder=1)
    ax.set_yticks(y,labels); ax.invert_yaxis(); ax.set_xlabel(xlabel); clean_axis(ax,"x")
    return p


def figure2() -> None:
    df=read_primary()
    fig=plt.figure(figsize=(14.8,8.4))
    gs=fig.add_gridspec(1,2,width_ratios=[1.35,1.0],wspace=0.34)
    fig.suptitle("Exact-size paired target-balance effects across 20 unique partition pairs",
                 fontsize=15.2,fontweight="bold",y=0.985)
    ax=fig.add_subplot(gs[0,0]); label_panel(ax,"A","Classification · 12 dataset–model cells")
    pcls=forest_panel(ax,df[df.task_type=="classification"],"classification")
    box(ax,(0.05,0.01),0.46,0.09,"0/12 corrected\nbalanced advantages",C["pale_blue"],C["navy"],8.6,"bold")
    box(ax,(0.54,0.01),0.41,0.09,"11/12 mean effects\nwere negative",C["pale_orange"],C["orange"],8.6,"bold")
    ax=fig.add_subplot(gs[0,1]); label_panel(ax,"B","Regression · primary single-group semantics")
    preg=forest_panel(ax,df[df.task_type=="regression"],"regression")
    box(ax,(0.12,0.01),0.76,0.09,"6/6 cells met the pre-specified corrected decision rule",C["pale_teal"],C["teal"],8.6,"bold")
    fig.subplots_adjust(bottom=0.13,top=0.90)
    save(fig,"figure2_primary_effects_v3")


def figure3() -> None:
    primary=read_primary(); single=read_singleton()
    primary=primary[primary.dataset.isin(REGRESSION)].copy()
    fig=plt.figure(figsize=(14.8,8.6))
    gs=fig.add_gridspec(2,2,height_ratios=[0.34,1.0],hspace=0.34,wspace=0.28)
    fig.suptitle("Regression effect depends on the structural semantics assigned to acyclic molecules",
                 fontsize=15.0,fontweight="bold",y=0.985)
    ax=fig.add_subplot(gs[0,:]); ax.set_axis_off(); label_panel(ax,"A","What changes when the empty Bemis–Murcko framework is interpreted differently?",x=-0.03,y=1.04)
    box(ax,(0.06,0.47),0.34,0.26,"single-group\nall acyclic molecules → one identity",C["pale_blue"],C["navy"],9.2,"bold")
    arrow(ax,(0.41,0.60),(0.59,0.60),C["orange"],1.7)
    box(ax,(0.60,0.47),0.34,0.26,"singleton\neach acyclic molecule → own identity",C["pale_teal"],C["teal"],9.2,"bold")
    box(ax,(0.34,0.12),0.32,0.16,"unchanged: endpoints · models · 20 seeds\nexact-size paired selection logic",C["pale_orange"],C["orange"],8.2)
    legend=[Line2D([0],[0],marker='o',color=C["navy"],lw=1.5,label="single-group (primary)"),Line2D([0],[0],marker='s',color=C["orange"],lw=1.5,label="singleton sensitivity")]
    for j,dataset in enumerate(REGRESSION):
        ax=fig.add_subplot(gs[1,j]); label_panel(ax,chr(ord('B')+j),dataset,x=-0.10,y=1.06)
        base=np.arange(3)
        for model_i,model in enumerate(MODELS_REG):
            pr=primary[(primary.dataset==dataset)&(primary.model==model)].iloc[0]
            sr=single[(single.dataset==dataset)&(single.model==model)].iloc[0]
            for yoff,row,col,marker in [(-0.11,pr,C["navy"],'o'),(0.11,sr,C["orange"],'s')]:
                mn=float(row.mean_effect); lo=float(row.bootstrap_ci_low); hi=float(row.bootstrap_ci_high)
                yy=model_i+yoff
                ax.plot([lo,hi],[yy,yy],color=col,lw=1.8); ax.scatter(mn,yy,s=42,color=col,marker=marker,zorder=3)
        ax.axvline(0,color=C["gray"],ls='--',lw=1.0); ax.set_yticks(base,MODELS_REG); ax.invert_yaxis(); clean_axis(ax,"x")
        ax.set_xlabel("RMSE improvement: size-matched − balanced")
        if dataset=="ESOL":
            box(ax,(0.55,0.04),0.40,0.10,"Primary gains\nstrongly attenuate",C["pale_orange"],C["orange"],8.2,"bold")
        else:
            box(ax,(0.49,0.04),0.46,0.10,"All three singleton\npoint estimates reverse sign",C["pale_orange"],C["orange"],8.2,"bold")
    fig.legend(handles=legend,frameon=False,loc="lower center",ncol=2,bbox_to_anchor=(0.5,0.015))
    fig.subplots_adjust(bottom=0.12,top=0.90)
    save(fig,"figure3_acyclic_sensitivity_v3")


def figure4() -> None:
    require(PARENT_COMPARISON)
    comp=pd.read_csv(PARENT_COMPARISON,keep_default_na=False)
    fig=plt.figure(figsize=(14.8,9.2))
    gs=fig.add_gridspec(2,3,height_ratios=[0.48,1.0],wspace=0.30,hspace=0.38)
    fig.suptitle("Disconnected-component representation changes benchmark composition without changing corrected classification inference",
                 fontsize=14.6,fontweight="bold",y=0.985)
    ax=fig.add_subplot(gs[0,0]); ax.set_axis_off(); label_panel(ax,"A","Representation perturbation")
    ax.scatter([0.18,0.29,0.38],[0.59,0.65,0.52],s=[330,190,100],c=[C["teal"],C["cyan"],C["orange"]],transform=ax.transAxes)
    arrow(ax,(0.49,0.60),(0.66,0.60),C["orange"])
    ax.scatter([0.79],[0.60],s=[620],c=[C["teal"]],transform=ax.transAxes)
    ax.text(0.27,0.82,"source-faithful",transform=ax.transAxes,ha="center",fontweight="bold",fontsize=8.6)
    ax.text(0.79,0.82,"dominant fragment",transform=ax.transAxes,ha="center",fontweight="bold",fontsize=8.6)
    box(ax,(0.15,0.15),0.70,0.16,"not a lossless formatting step",C["pale_orange"],C["orange"],8.7,"bold")
    affected=["BBBP","ClinTox","HIV"]
    ax=fig.add_subplot(gs[0,1]); label_panel(ax,"B","Multi-component records")
    vals=[MULTICOMPONENT[d] for d in affected]; bars=ax.bar(affected,vals,color=[C["navy"],C["cyan"],C["teal"]],width=0.58)
    ax.set_yscale("log"); ax.set_ylabel("Count (log scale)"); clean_axis(ax,"y")
    for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v*1.16,f"{v:,}",ha="center",fontsize=8.5)
    ax=fig.add_subplot(gs[0,2]); label_panel(ax,"C","Structural consequences")
    x=np.arange(3); w=0.32
    b1=ax.bar(x-w/2,[SCAFFOLD_CHANGED[d] for d in affected],width=w,label="Scaffold changed",color=C["navy"])
    b2=ax.bar(x+w/2,[SIM_LT_090[d] for d in affected],width=w,label="Similarity < 0.90",color=C["orange"])
    ax.set_yscale("log"); ax.set_xticks(x,affected); ax.set_ylabel("Count (log scale)"); clean_axis(ax,"y"); ax.legend(frameon=False,loc="upper left")
    for bars in (b1,b2):
        for b in bars: ax.text(b.get_x()+b.get_width()/2,b.get_height()*1.14,f"{int(b.get_height()):,}",ha="center",fontsize=8.2)
    ax=fig.add_subplot(gs[1,:]); label_panel(ax,"D","Effect direction is representation-sensitive")
    order=[(d,m) for d in affected for m in MODELS_CLS]
    rows=[]
    for d,m in order:
        r=comp[(comp.dataset==d)&(comp.model==m)]
        if len(r)!=1: raise AssertionError(f"Missing parent comparison {d}/{m}")
        rows.append(r.iloc[0])
    p=pd.DataFrame(rows).reset_index(drop=True); y=np.arange(len(p))
    main=p.main_mean_effect.astype(float).to_numpy(); parent=p.parent_mean_effect.astype(float).to_numpy()
    reverse=np.sign(main)!=np.sign(parent)
    for yi,a,b,rev in zip(y,main,parent,reverse):
        ax.plot([a,b],[yi,yi],color=C["orange"] if rev else C["mid"],lw=1.8,zorder=1)
        ax.scatter(a,yi,s=42,color=C["navy"],zorder=2); ax.scatter(b,yi,s=42,color=C["orange"],marker='s',zorder=2)
    ax.axvline(0,color=C["gray"],ls='--',lw=1.0); ax.set_yticks(y,[f"{d} · {m}" for d,m in order]); ax.invert_yaxis(); clean_axis(ax,"x")
    ax.set_xlabel("AUC effect: balanced − size-matched")
    ax.legend(handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor=C["navy"],label="source-faithful primary"),Line2D([0],[0],marker='s',color='none',markerfacecolor=C["orange"],label="dominant-fragment sensitivity")],frameon=False,loc="lower right")
    box(ax,(0.72,0.83),0.25,0.10,"7/9 mean-effect\nsigns reverse",C["pale_orange"],C["orange"],8.4,"bold")
    box(ax,(0.72,0.69),0.25,0.10,"9/9 corrected\ninferences remain inconclusive",C["pale_blue"],C["navy"],8.1,"bold")
    save(fig,"figure4_dominant_fragment_sensitivity_v3")


def figure5() -> None:
    fig=plt.figure(figsize=(14.8,9.2)); gs=fig.add_gridspec(2,2,wspace=0.28,hspace=0.38)
    fig.suptitle("Candidate-search budget is a benchmark-construction hyperparameter and was frozen before model fitting",
                 fontsize=14.6,fontweight="bold",y=0.985)
    for j,d in enumerate(REGRESSION):
        ax=fig.add_subplot(gs[0,j]); label_panel(ax,chr(ord('A')+j),f"{d} · single-group acyclic semantics")
        budgets=np.array(list(BUDGET_SINGLE_GROUP[d])); vals=np.array(list(BUDGET_SINGLE_GROUP[d].values()))
        ax.plot(budgets,vals,color=C["teal"] if d=="ESOL" else C["navy"],marker='o',lw=2.0,ms=5)
        ax.fill_between(budgets,vals,0,color=C["pale_teal"] if d=="ESOL" else C["pale_blue"],alpha=0.75)
        ax.axvline(20000,color=C["orange"],ls='--',lw=1.2)
        ax.annotate(f"frozen cap\n20,000",xy=(20000,vals[-1]),xytext=(13200,vals.max()*0.55),arrowprops=dict(arrowstyle='->',color=C["orange"]),fontsize=8.3,color=C["orange2"])
        ax.set_xscale('log'); ax.set_xlabel("Candidate budget"); ax.set_ylabel("Mean balanced target gap"); clean_axis(ax,"both")
        box(ax,(0.04,0.05),0.45,0.10,"continued improvement\nthrough largest audited budget",C["pale_orange"],C["orange"],8.0,"bold")
    ax=fig.add_subplot(gs[1,0]); label_panel(ax,"C","Singleton sensitivity · normalized target-gap trajectory")
    for d,col,marker in [("ESOL",C["teal"],'o'),("FreeSolv",C["orange"],'s')]:
        budgets=np.array(list(BUDGET_SINGLETON[d])); vals=np.array(list(BUDGET_SINGLETON[d].values())); rel=vals/vals[0]
        ax.plot(budgets,rel,color=col,marker=marker,lw=2.0,ms=5,label=d)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel("Candidate budget"); ax.set_ylabel("Gap relative to 100-candidate value"); clean_axis(ax,"both"); ax.legend(frameon=False)
    ax=fig.add_subplot(gs[1,1]); label_panel(ax,"D","Exact-size pairing holds for every primary dataset")
    y=np.arange(len(DATASETS)); vals=np.array([TEST_N[d] for d in DATASETS],float)
    ax.hlines(y,vals,vals,color=C["mid"],lw=8)
    ax.scatter(vals,y,s=70,color=C["navy"],label="size-matched",zorder=3)
    ax.scatter(vals,y,s=34,color=C["teal"],marker='s',label="target-balanced",zorder=4)
    ax.set_xscale('log'); ax.set_yticks(y,DATASETS); ax.invert_yaxis(); ax.set_xlabel("Test molecules (log scale)"); clean_axis(ax,"x")
    for yi,v in zip(y,vals): ax.text(v*1.09,yi,f"{int(v):,}",va='center',fontsize=8.4)
    ax.legend(frameon=False,loc="lower right")
    box(ax,(0.50,0.03),0.46,0.10,"same n_test within every seed\nand every paired protocol",C["pale_teal"],C["teal"],8.2,"bold")
    save(fig,"figure5_candidate_budget_audit_v3")


def figure6() -> None:
    primary=read_primary(); single=read_singleton(); require(PARENT_COMPARISON); comp=pd.read_csv(PARENT_COMPARISON,keep_default_na=False)
    fig=plt.figure(figsize=(15.2,8.9)); gs=fig.add_gridspec(1,3,width_ratios=[0.95,1.35,1.05],wspace=0.30)
    fig.suptitle("The scientific claim changes at different levels when benchmark-construction rules are perturbed",
                 fontsize=14.8,fontweight="bold",y=0.985)
    ax=fig.add_subplot(gs[0,0]); ax.set_axis_off(); label_panel(ax,"A","Primary classification")
    box(ax,(0.08,0.80),0.84,0.12,"0/12 supported\nbalanced advantages",C["pale_blue"],C["navy"],10.0,"bold")
    cls=primary[primary.task_type=="classification"].copy(); order=[(d,m) for d in CLASSIFICATION for m in MODELS_CLS]
    for i,(d,m) in enumerate(order):
        r=cls[(cls.dataset==d)&(cls.model==m)].iloc[0]; neg=float(r.mean_effect)<0
        row=i//3; col=i%3; x=0.12+col*0.27; y=0.62-row*0.12
        fc=C["pale_orange"] if neg else C["pale_teal"]; ec=C["orange"] if neg else C["teal"]
        box(ax,(x,y),0.19,0.075,m,fc,ec,7.8,"bold",radius=0.015)
        if col==0: ax.text(0.03,y+0.037,d,transform=ax.transAxes,ha='right',va='center',fontsize=8.0,fontweight='bold')
    box(ax,(0.08,0.09),0.84,0.12,"11/12 point estimates < 0\nInference: no reproducible classification gain",C["pale_orange"],C["orange"],8.8,"bold")

    ax=fig.add_subplot(gs[0,1]); label_panel(ax,"B","Regression depends on scaffold semantics")
    order=[(d,m) for d in REGRESSION for m in MODELS_REG]; y=np.arange(len(order))
    for yi,(d,m) in enumerate(order):
        pr=primary[(primary.dataset==d)&(primary.model==m)].iloc[0]; sr=single[(single.dataset==d)&(single.model==m)].iloc[0]
        a=float(pr.mean_effect); b=float(sr.mean_effect); col=C["orange"] if np.sign(a)!=np.sign(b) else C["mid"]
        ax.plot([a,b],[yi,yi],color=col,lw=2.0); ax.scatter(a,yi,s=45,color=C["navy"]); ax.scatter(b,yi,s=45,color=C["orange"],marker='s')
    ax.axvline(0,color=C["gray"],ls='--',lw=1.0); ax.set_yticks(y,[f"{d} · {m}" for d,m in order]); ax.invert_yaxis(); clean_axis(ax,"x")
    ax.set_xlabel("Effect (positive favors target balancing)")
    ax.legend(handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor=C["navy"],label="primary single-group"),Line2D([0],[0],marker='s',color='none',markerfacecolor=C["orange"],label="singleton sensitivity")],frameon=False,loc="lower right")
    box(ax,(0.05,0.05),0.43,0.10,"6/6 supported\nunder primary semantics",C["pale_teal"],C["teal"],8.2,"bold")
    box(ax,(0.53,0.05),0.42,0.10,"3/3 FreeSolv effects\nreverse under singleton",C["pale_orange"],C["orange"],8.2,"bold")

    ax=fig.add_subplot(gs[0,2]); label_panel(ax,"C","Dominant-fragment sensitivity")
    order=[(d,m) for d in ["BBBP","ClinTox","HIV"] for m in MODELS_CLS]; y=np.arange(len(order))
    rev_count=0
    for yi,(d,m) in enumerate(order):
        r=comp[(comp.dataset==d)&(comp.model==m)].iloc[0]; a=float(r.main_mean_effect); b=float(r.parent_mean_effect); rev=np.sign(a)!=np.sign(b); rev_count+=int(rev)
        ax.annotate('',xy=(b,yi),xytext=(a,yi),arrowprops=dict(arrowstyle='-|>',lw=1.7,color=C["orange"] if rev else C["mid"],mutation_scale=11))
        ax.scatter(a,yi,s=28,color=C["navy"],zorder=3)
    ax.axvline(0,color=C["gray"],ls='--',lw=1.0); ax.set_yticks(y,[f"{d} · {m}" for d,m in order]); ax.invert_yaxis(); clean_axis(ax,"x")
    ax.set_xlabel("AUC effect")
    box(ax,(0.08,0.08),0.84,0.11,"7/9 point-estimate directions reverse",C["pale_orange"],C["orange"],8.8,"bold")
    box(ax,(0.08,0.22),0.84,0.11,"9/9 corrected inferences remain inconclusive",C["pale_blue"],C["navy"],8.5,"bold")
    fig.subplots_adjust(bottom=0.11,top=0.90)
    save(fig,"figure6_claim_stability_map_v3")


def figure_s1() -> None:
    fig,ax=plt.subplots(figsize=(12.8,5.8))
    x=np.arange(len(DATASETS)); raw=np.array([RAW_N[d] for d in DATASETS]); final=np.array([DATASET_N[d] for d in DATASETS]); removed=raw-final
    ax.bar(x,raw,color=C["pale_blue"],edgecolor=C["mid"],width=0.62,label="Raw rows")
    ax.bar(x,final,color=[C["navy"] if d in CLASSIFICATION else C["teal"] for d in DATASETS],width=0.46,label="Final clean rows")
    ax.set_yscale('log'); ax.set_xticks(x,DATASETS); ax.set_ylabel("Rows (log scale)"); clean_axis(ax,"y")
    for xi,r,f,rm in zip(x,raw,final,removed):
        if rm>0: ax.annotate(f"−{rm}",xy=(xi,f),xytext=(xi,r*1.15),ha='center',fontsize=8.5,color=C["orange2"])
        else: ax.text(xi,r*1.10,"no removal",ha='center',fontsize=8.0,color=C["gray"])
    ax.set_title("Audited raw-to-clean molecular-data construction",fontweight='bold',fontsize=14)
    ax.legend(frameon=False,ncol=2,loc='upper right')
    save(fig,"figureS1_dataset_construction_v3")


def figure_s2() -> None:
    fig,axes=plt.subplots(1,2,figsize=(13.4,5.6))
    for ax,d in zip(axes,REGRESSION):
        for source,col,marker,label in [(BUDGET_SINGLE_GROUP,C["navy"],'o','single-group'),(BUDGET_SINGLETON,C["orange"],'s','singleton')]:
            budgets=np.array(list(source[d])); vals=np.array(list(source[d].values())); ax.plot(budgets,vals,color=col,marker=marker,lw=2.0,ms=5,label=label)
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel("Candidate budget"); ax.set_ylabel("Mean balanced target gap"); clean_axis(ax,"both"); ax.set_title(d,fontweight='bold'); ax.legend(frameon=False)
    fig.suptitle("Candidate-budget behavior differs under alternative acyclic-scaffold semantics",fontsize=14,fontweight='bold')
    fig.tight_layout(rect=[0,0,1,0.94]); save(fig,"figureS2_budget_semantics_v3")


def figure_s3() -> None:
    datasets=["BBBP","ClinTox","HIV"]
    metrics=["Multi-component","Scaffold changed","Similarity < 0.90","Conflict groups"]
    mat=np.array([[MULTICOMPONENT[d],SCAFFOLD_CHANGED[d],SIM_LT_090[d],CONFLICT_GROUPS[d]] for d in datasets],float)
    score=np.log10(mat+1)
    fig,ax=plt.subplots(figsize=(10.8,5.8)); im=ax.imshow(score,cmap="YlGnBu",aspect='auto')
    ax.set_xticks(np.arange(len(metrics)),metrics); ax.set_yticks(np.arange(len(datasets)),datasets)
    ax.set_title("Disconnected-component structural-audit summary",fontweight='bold',fontsize=14)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j,i,f"{int(mat[i,j]):,}",ha='center',va='center',fontsize=10,fontweight='bold',color=C["ink"] if score[i,j]<2.6 else C["white"])
    cb=fig.colorbar(im,ax=ax,fraction=0.035,pad=0.03); cb.set_label("log10(count + 1)")
    for spine in ax.spines.values(): spine.set_visible(False)
    fig.tight_layout(); save(fig,"figureS3_multicomponent_audit_v3")


def main() -> None:
    print("Building Paper 1 round-2 publication figures from frozen v3 result tables")
    figure1(); figure2(); figure3(); figure4(); figure5(); figure6(); figure_s1(); figure_s2(); figure_s3()
    print("\nROUND-2 MANUSCRIPT FIGURE BUILD: PASS")


if __name__ == "__main__":
    main()
