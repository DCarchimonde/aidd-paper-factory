from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper1_leakage_benchmark"
TABLES = PAPER / "results" / "tables"
PARENT = PAPER / "results" / "parent_fragment_sensitivity_v3" / "tables"
FIG = PAPER / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

PRIMARY = TABLES / "primary_inference_summary_v3.csv"
SINGLETON = TABLES / "acyclic_singleton_sensitivity_v3.csv"
PARENT_CMP = PARENT / "parent_fragment_vs_main_comparison_v3.csv"
SUPPORT = TABLES / "supporting_metric_effects_v3.csv"
COLLATERAL = TABLES / "q1_collateral_partition_diagnostics_v3.csv"
MEAN_ONLY = TABLES / "q1_mean_only_regression_summary_v3.csv"
SEED = TABLES / "q1_model_seed_summary_v3.csv"
CLEAN = TABLES / "q1_cleaning_accounting_v3.csv"

DATASETS = ["BACE", "BBBP", "ClinTox", "HIV", "ESOL", "FreeSolv"]
CLS = ["BACE", "BBBP", "ClinTox", "HIV"]
REG = ["ESOL", "FreeSolv"]
MODEL_ORDER = {"LR": 0, "Ridge": 0, "RF": 1, "XGB": 2}
DATASET_ORDER = {d: i for i, d in enumerate(DATASETS)}
N = {"BACE": 1513, "BBBP": 1965, "ClinTox": 1442, "HIV": 41120, "ESOL": 1117, "FreeSolv": 642}
TEST_N = {"BACE": 303, "BBBP": 393, "ClinTox": 288, "HIV": 8224, "ESOL": 223, "FreeSolv": 128}
MULTI = {"BBBP": 105, "ClinTox": 14, "HIV": 3086}
SCAFF_CHANGED = {"BBBP": 5, "ClinTox": 1, "HIV": 235}
SIM090 = {"BBBP": 18, "ClinTox": 5, "HIV": 640}
CONFLICT = {"BBBP": 1, "ClinTox": 1, "HIV": 17}
BUDGET_SINGLE = {
    "ESOL": {3000: 0.034622, 5000: 0.018366, 10000: 0.010563, 20000: 0.001787},
    "FreeSolv": {3000: 1.076491, 5000: 1.044452, 10000: 0.787491, 20000: 0.686528},
}
BUDGET_SINGLETON = {
    "ESOL": {100: 0.003687, 300: 0.001535, 500: 0.000600, 1000: 0.000600, 3000: 0.000301, 5000: 0.000194},
    "FreeSolv": {100: 0.019467, 300: 0.006230, 500: 0.005645, 1000: 0.005416, 3000: 0.001609, 5000: 0.000471},
}

C = {
    "ink": "#20313A", "navy": "#315B73", "navy2": "#24485D",
    "teal": "#2B8C82", "teal2": "#176B64", "orange": "#D58A43",
    "orange2": "#A85F28", "pale_blue": "#EAF1F5", "pale_teal": "#E8F3EF",
    "pale_orange": "#FBF0E5", "pale_gray": "#F4F6F7", "gray": "#6D7A81",
    "mid": "#B8C2C7", "grid": "#E5EAEC", "white": "#FFFFFF",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.3, "axes.titlesize": 9.0,
    "axes.labelsize": 8.3, "xtick.labelsize": 7.4, "ytick.labelsize": 7.4,
    "legend.fontsize": 7.0, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": C["mid"], "text.color": C["ink"], "axes.labelcolor": C["ink"],
    "xtick.color": C["ink"], "ytick.color": C["ink"], "pdf.fonttype": 42,
    "ps.fonttype": 42, "figure.facecolor": "white", "savefig.facecolor": "white",
})


def need(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(FIG / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(FIG / f"{stem}.pdf")


def panel(ax: plt.Axes, letter: str, title: str | None = None) -> None:
    ax.text(-0.09, 1.07, letter, transform=ax.transAxes, fontsize=10.8, fontweight="bold", va="top")
    if title:
        ax.text(0.0, 1.035, title, transform=ax.transAxes, fontsize=8.7, fontweight="bold", va="bottom")


def box(ax, xy, w, h, text, fc, ec, fs=7.2, bold=False):
    p = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                       transform=ax.transAxes, facecolor=fc, edgecolor=ec, linewidth=0.9)
    ax.add_patch(p)
    ax.text(xy[0]+w/2, xy[1]+h/2, text, transform=ax.transAxes, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal", linespacing=1.05)


def arrow(ax, start, end, color=None):
    ax.add_patch(FancyArrowPatch(start, end, transform=ax.transAxes, arrowstyle="-|>",
                                 mutation_scale=10, linewidth=1.0, color=color or C["gray"]))


def clean(ax, axis="x"):
    ax.grid(axis=axis, color=C["grid"], lw=0.7, zorder=0)


def primary_frame() -> pd.DataFrame:
    df = pd.read_csv(need(PRIMARY), keep_default_na=False)
    df["do"] = df["dataset"].map(DATASET_ORDER)
    df["mo"] = df["model"].map(MODEL_ORDER)
    return df.sort_values(["do", "mo"]).reset_index(drop=True)


def figure1() -> None:
    fig = plt.figure(figsize=(7.15, 5.35))
    gs = fig.add_gridspec(2, 2, wspace=0.28, hspace=0.40)

    ax = fig.add_subplot(gs[0,0]); panel(ax, "A", "Audited molecular universe")
    y = np.arange(6); vals = [N[d] for d in DATASETS]
    cols = [C["navy"] if d in CLS else C["teal"] for d in DATASETS]
    ax.barh(y, vals, color=cols, height=0.58, zorder=2)
    ax.set_yticks(y, DATASETS); ax.invert_yaxis(); ax.set_xscale("log"); ax.set_xlabel("Clean molecules")
    clean(ax, "x")
    for yy,v in zip(y,vals): ax.text(v*1.06, yy, f"{v:,}", va="center", fontsize=6.5)
    ax.legend(handles=[Rectangle((0,0),1,1,color=C["navy"],label="Classification"), Rectangle((0,0),1,1,color=C["teal"],label="Regression")], frameon=False, loc="lower right")

    ax = fig.add_subplot(gs[0,1]); ax.set_axis_off(); panel(ax, "B", "Exact-size target-mean perturbation")
    box(ax,(0.02,0.62),0.25,0.18,"Target-blind\ncandidate pool",C["pale_blue"],C["navy"],7.0,True)
    arrow(ax,(0.29,0.71),(0.39,0.71))
    box(ax,(0.40,0.61),0.25,0.20,"Size-matched\nbaseline",C["pale_gray"],C["mid"],7.0,True)
    box(ax,(0.72,0.61),0.25,0.20,"Lowest target-mean\ngap at same n_test",C["pale_teal"],C["teal"],6.8,True)
    arrow(ax,(0.66,0.71),(0.71,0.71),C["teal2"])
    box(ax,(0.11,0.34),0.78,0.12,"fixed: dataset · seed · scaffold rule · candidate budget",C["white"],C["mid"],6.5)
    box(ax,(0.20,0.13),0.60,0.13,"exactly the same test-set size",C["pale_orange"],C["orange"],7.1,True)

    ax = fig.add_subplot(gs[1,0]); ax.set_axis_off(); panel(ax, "C", "Pre-outcome freeze and inference")
    steps=[("Budget\nfrozen",C["pale_orange"],C["orange"]),("Manifest\n+ hash",C["pale_blue"],C["navy"]),("Model\nfit",C["pale_gray"],C["mid"]),("Paired\neffect",C["pale_teal"],C["teal"])]
    for i,(txt,fc,ec) in enumerate(steps):
        x=0.02+i*0.245; box(ax,(x,0.58),0.19,0.19,txt,fc,ec,7.0,True)
        if i<3: arrow(ax,(x+0.195,0.675),(x+0.235,0.675))
    box(ax,(0.06,0.27),0.26,0.13,"20 unique\npartition pairs",C["white"],C["mid"],6.8)
    box(ax,(0.37,0.27),0.26,0.13,"10,000 paired\nbootstrap draws",C["white"],C["mid"],6.8)
    box(ax,(0.68,0.27),0.26,0.13,"Wilcoxon +\nHolm",C["white"],C["mid"],6.8)
    ax.text(0.50,0.08,"Inferential N = unique partition pairs, not model seeds",transform=ax.transAxes,ha="center",fontsize=6.8,color=C["gray"])

    ax = fig.add_subplot(gs[1,1]); ax.set_axis_off(); panel(ax, "D", "Protocol sensitivities")
    box(ax,(0.04,0.57),0.40,0.21,"Acyclic semantics\nsingle-group ↔ singleton",C["pale_blue"],C["navy"],7.1,True)
    box(ax,(0.56,0.57),0.40,0.21,"Molecular record\nsource-faithful ↔ dominant fragment",C["pale_orange"],C["orange"],6.7,True)
    arrow(ax,(0.24,0.53),(0.43,0.34),C["navy"]); arrow(ax,(0.76,0.53),(0.57,0.34),C["orange"])
    box(ax,(0.25,0.18),0.50,0.16,"Does the scientific claim survive?",C["pale_teal"],C["teal"],7.2,True)
    ax.text(0.5,0.03,"Sensitivity disagreements are reported rather than resolved post hoc.",transform=ax.transAxes,ha="center",fontsize=6.5,color=C["gray"])

    fig.suptitle("Benchmark construction as a controlled chemometric measurement process", fontsize=10.6, fontweight="bold", y=0.995)
    save(fig,"figure1_audit_framework_v3")


def forest(ax, df, task, title, letter):
    panel(ax,letter,title)
    y=np.arange(len(df)); eff=df["mean_effect"].to_numpy(float); lo=df["bootstrap_ci_low"].to_numpy(float); hi=df["bootstrap_ci_high"].to_numpy(float)
    labels=[f"{r.dataset} · {r.model}" for r in df.itertuples(index=False)]
    for i,r in enumerate(df.itertuples(index=False)):
        if i%3==0: ax.axhspan(i-0.48,min(i+2.48,len(df)-0.52),color=C["pale_gray"],zorder=0)
    col=[C["teal2"] if str(x)=="target_balanced_better" else C["navy"] for x in df["inference_label"]]
    for yy,e,l,h,c in zip(y,eff,lo,hi,col): ax.errorbar(e,yy,xerr=[[e-l],[h-e]],fmt="o",ms=3.2,lw=1.0,capsize=2.0,color=c,zorder=3)
    ax.axvline(0,color=C["gray"],ls="--",lw=0.9); ax.set_yticks(y,labels); ax.invert_yaxis(); clean(ax,"x")
    ax.set_xlabel("AUC effect: balanced − size-matched" if task=="classification" else "RMSE improvement: size-matched − balanced")


def figure2() -> None:
    df=primary_frame(); cls=df[df["task_type"].eq("classification")]; reg=df[df["task_type"].eq("regression")]
    fig,axs=plt.subplots(1,2,figsize=(7.15,4.35),gridspec_kw={"width_ratios":[1.12,0.88],"wspace":0.45})
    forest(axs[0],cls,"classification","Classification · 12 cells","A"); forest(axs[1],reg,"regression","Regression · primary semantics","B")
    fig.suptitle("Exact-size paired target-mean selection effects across 20 partition pairs",fontsize=10.2,fontweight="bold",y=0.995)
    fig.text(0.25,0.015,"0/12 cells met the corrected target-balanced-advantage criterion",ha="center",fontsize=6.9,color=C["navy2"],fontweight="bold")
    fig.text(0.76,0.015,"6/6 regression cells met the corrected criterion under primary semantics",ha="center",fontsize=6.9,color=C["teal2"],fontweight="bold")
    fig.subplots_adjust(bottom=0.13,top=0.87)
    save(fig,"figure2_primary_effects_v3")


def figure3() -> None:
    p=primary_frame(); s=pd.read_csv(need(SINGLETON),keep_default_na=False).copy()
    if "mean_effect_positive_is_balanced_better" in s.columns: s=s.rename(columns={"mean_effect_positive_is_balanced_better":"mean_effect"})
    fig=plt.figure(figsize=(7.15,4.75)); gs=fig.add_gridspec(2,2,height_ratios=[0.42,1.0],hspace=0.48,wspace=0.38)
    ax=fig.add_subplot(gs[0,:]); ax.set_axis_off(); panel(ax,"A","Acyclic scaffold semantics is the only designed change")
    box(ax,(0.06,0.42),0.34,0.28,"single-group\nall acyclic molecules → one identity",C["pale_blue"],C["navy"],7.1,True); arrow(ax,(0.41,0.56),(0.59,0.56),C["orange"]); box(ax,(0.60,0.42),0.34,0.28,"singleton\neach acyclic molecule → own identity",C["pale_teal"],C["teal"],7.1,True)
    ax.text(0.50,0.15,"unchanged: endpoints · models · 20 partition seeds · exact-size paired selection logic",transform=ax.transAxes,ha="center",fontsize=6.7,color=C["gray"])
    for j,ds in enumerate(REG):
        ax=fig.add_subplot(gs[1,j]); panel(ax,"B" if j==0 else "C",ds)
        pp=p[p["dataset"].eq(ds)].sort_values("mo"); ss=s[s["dataset"].eq(ds)].copy(); ss["mo"]=ss["model"].map(MODEL_ORDER); ss=ss.sort_values("mo")
        y=np.arange(3)
        for k,m in enumerate(["Ridge","RF","XGB"]):
            pr=pp[pp["model"].eq(m)].iloc[0]; sr=ss[ss["model"].eq(m)].iloc[0]
            ax.errorbar(float(pr.mean_effect),k,xerr=[[float(pr.mean_effect-pr.bootstrap_ci_low)],[float(pr.bootstrap_ci_high-pr.mean_effect)]],fmt="o",ms=3.6,color=C["navy"],lw=1.0,capsize=2)
            ax.errorbar(float(sr.mean_effect),k,xerr=[[float(sr.mean_effect-sr.bootstrap_ci_low)],[float(sr.bootstrap_ci_high-sr.mean_effect)]],fmt="s",ms=3.5,color=C["orange"],lw=1.0,capsize=2)
        ax.axvline(0,color=C["gray"],ls="--",lw=0.8); ax.set_yticks(y,["Ridge","RF","XGB"]); ax.invert_yaxis(); clean(ax,"x"); ax.set_xlabel("RMSE improvement")
    fig.legend(handles=[Line2D([0],[0],marker="o",color=C["navy"],lw=0,label="single-group (primary)"),Line2D([0],[0],marker="s",color=C["orange"],lw=0,label="singleton sensitivity")],loc="lower center",ncol=2,frameon=False,bbox_to_anchor=(0.5,0.005))
    fig.suptitle("Regression effects depend on the structural semantics assigned to acyclic molecules",fontsize=10.1,fontweight="bold",y=0.995); fig.subplots_adjust(bottom=0.14,top=0.88)
    save(fig,"figure3_acyclic_sensitivity_v3")


def figure4() -> None:
    comp=pd.read_csv(need(PARENT_CMP),keep_default_na=False)
    # normalize common column names used by the frozen comparison table
    main_col=next(c for c in ["main_mean_effect","primary_mean_effect","source_faithful_mean_effect"] if c in comp.columns)
    frag_col=next(c for c in ["parent_mean_effect","fragment_mean_effect","dominant_fragment_mean_effect"] if c in comp.columns)
    comp["do"]=comp["dataset"].map(DATASET_ORDER); comp["mo"]=comp["model"].map(MODEL_ORDER); comp=comp.sort_values(["do","mo"])
    fig=plt.figure(figsize=(7.15,5.25)); gs=fig.add_gridspec(2,2,height_ratios=[0.72,1.30],wspace=0.33,hspace=0.43)
    ax=fig.add_subplot(gs[0,0]); ax.set_axis_off(); panel(ax,"A","Representation perturbation")
    ax.scatter([0.14,0.22,0.30],[0.56,0.62,0.50],s=[75,45,28],c=[C["teal"],"#75C0C8",C["orange"]],transform=ax.transAxes,clip_on=False); arrow(ax,(0.39,0.57),(0.60,0.57),C["orange"]); ax.scatter([0.75],[0.57],s=125,c=[C["teal"]],transform=ax.transAxes,clip_on=False)
    ax.text(0.22,0.78,"source-faithful",transform=ax.transAxes,ha="center",fontsize=7.1,fontweight="bold"); ax.text(0.75,0.78,"dominant fragment",transform=ax.transAxes,ha="center",fontsize=7.1,fontweight="bold"); box(ax,(0.18,0.15),0.64,0.16,"algorithmic sensitivity; not a lossless formatting step",C["pale_orange"],C["orange"],6.5,True)
    ax=fig.add_subplot(gs[0,1]); panel(ax,"B","Structural consequences")
    x=np.arange(3); w=0.24; ds=["BBBP","ClinTox","HIV"]
    ax.bar(x-w,[SCAFF_CHANGED[d] for d in ds],w,color=C["navy"],label="Scaffold changed"); ax.bar(x,[SIM090[d] for d in ds],w,color=C["orange"],label="Similarity < 0.90"); ax.bar(x+w,[CONFLICT[d] for d in ds],w,color=C["teal"],label="Conflict groups"); ax.set_yscale("symlog",linthresh=1); ax.set_xticks(x,ds); ax.set_ylabel("Count"); clean(ax,"y"); ax.legend(frameon=False,ncol=1,loc="upper left")
    ax=fig.add_subplot(gs[1,:]); panel(ax,"C","Effect direction is representation-sensitive")
    y=np.arange(len(comp)); a=comp[main_col].astype(float).to_numpy(); b=comp[frag_col].astype(float).to_numpy(); labels=[f"{r.dataset} · {r.model}" for r in comp.itertuples(index=False)]
    for yy,x1,x2 in zip(y,a,b): ax.plot([x1,x2],[yy,yy],color=C["orange"],lw=1.0,zorder=1)
    ax.scatter(a,y,s=16,color=C["navy"],label="source-faithful primary",zorder=2); ax.scatter(b,y,s=18,marker="s",color=C["orange"],label="dominant-fragment sensitivity",zorder=2); ax.axvline(0,color=C["gray"],ls="--",lw=0.8); ax.set_yticks(y,labels); ax.invert_yaxis(); ax.set_xlabel("AUC effect: balanced − size-matched"); clean(ax,"x"); ax.legend(frameon=False,loc="lower right")
    fig.suptitle("Disconnected-component representation changes benchmark composition and point estimates",fontsize=10.1,fontweight="bold",y=0.995); fig.subplots_adjust(top=0.88,bottom=0.08)
    save(fig,"figure4_dominant_fragment_sensitivity_v3")


def figure5() -> None:
    fig,axs=plt.subplots(2,2,figsize=(7.15,5.25)); fig.subplots_adjust(wspace=0.34,hspace=0.50,top=0.89,bottom=0.10)
    for j,ds in enumerate(REG):
        ax=axs[0,j]; panel(ax,"A" if j==0 else "B",f"{ds} · single-group")
        x=np.array(list(BUDGET_SINGLE[ds])); y=np.array(list(BUDGET_SINGLE[ds].values())); ax.plot(x,y,marker="o",ms=3.2,lw=1.2,color=C["teal"] if ds=="ESOL" else C["navy"]); ax.fill_between(x,y,0,alpha=0.08,color=C["teal"] if ds=="ESOL" else C["navy"]); ax.axvline(20000,color=C["orange"],ls="--",lw=0.8); ax.annotate("frozen cap\n20,000",xy=(20000,y[-1]),xytext=(13000,y[-1]+0.25*(max(y)-min(y))),fontsize=6.4,color=C["orange2"],arrowprops=dict(arrowstyle="->",lw=0.7,color=C["orange2"])); ax.set_xlabel("Candidate budget"); ax.set_ylabel("Mean target-mean gap"); clean(ax,"both")
    ax=axs[1,0]; panel(ax,"C","Singleton sensitivity · normalized trajectory")
    for ds,col,mark in [("ESOL",C["teal"],"o"),("FreeSolv",C["orange"],"s")]:
        x=np.array(list(BUDGET_SINGLETON[ds])); y=np.array(list(BUDGET_SINGLETON[ds].values())); ax.plot(x,y/y[0],marker=mark,ms=3,lw=1.1,color=col,label=ds)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Candidate budget"); ax.set_ylabel("Gap relative to 100-candidate value"); clean(ax,"both"); ax.legend(frameon=False)
    ax=axs[1,1]; panel(ax,"D","Exact test-size pairing")
    yy=np.arange(6); vals=np.array([TEST_N[d] for d in DATASETS],float); ax.scatter(vals,yy,s=26,color=C["navy"],label="size-matched",zorder=3); ax.scatter(vals,yy,s=15,marker="s",facecolor=C["white"],edgecolor=C["teal"],linewidth=1.0,label="target-balanced",zorder=4); ax.set_yticks(yy,DATASETS); ax.invert_yaxis(); ax.set_xscale("log"); ax.set_xlabel("Test molecules"); clean(ax,"x"); ax.legend(frameon=False,loc="lower right")
    for x0,y0 in zip(vals,yy): ax.text(x0*1.08,y0,f"{int(x0):,}",va="center",fontsize=6.3)
    fig.suptitle("Candidate-search budget is a frozen benchmark-construction hyperparameter",fontsize=10.2,fontweight="bold",y=0.985)
    save(fig,"figure5_candidate_budget_audit_v3")


def figure6() -> None:
    p=primary_frame(); mo=pd.read_csv(need(MEAN_ONLY),keep_default_na=False); cd=pd.read_csv(need(COLLATERAL),keep_default_na=False)
    mo=mo[mo["freeze_label"].eq("main_regression")]
    fig=plt.figure(figsize=(7.15,5.45)); gs=fig.add_gridspec(2,2,wspace=0.40,hspace=0.46); fig.subplots_adjust(top=0.89,bottom=0.09)
    ax=fig.add_subplot(gs[0,0]); panel(ax,"A","Mean-only control vs learned models")
    rows=[]
    for ds in REG:
        m=mo[mo["dataset"].eq(ds)].iloc[0]; rows.append((ds,"Mean-only",float(m.mean_effect_size_minus_balanced_rmse),float(m.bootstrap_ci_low),float(m.bootstrap_ci_high),C["orange"]))
        for model in ["Ridge","RF","XGB"]:
            r=p[(p["dataset"].eq(ds))&(p["model"].eq(model))].iloc[0]; rows.append((ds,model,float(r.mean_effect),float(r.bootstrap_ci_low),float(r.bootstrap_ci_high),C["teal2"]))
    y=np.arange(len(rows)); labels=[]
    for yy,(ds,m,e,l,h,col) in zip(y,rows): ax.errorbar(e,yy,xerr=[[e-l],[h-e]],fmt="o" if m=="Mean-only" else "s",ms=3.1,color=col,lw=0.9,capsize=1.8); labels.append(f"{ds} · {m}")
    ax.axvline(0,color=C["gray"],ls="--",lw=0.8); ax.set_yticks(y,labels); ax.invert_yaxis(); ax.set_xlabel("RMSE improvement"); clean(ax,"x")

    ax=fig.add_subplot(gs[0,1]); panel(ax,"B","Target-mean gap reduction")
    rng=np.random.default_rng(3)
    for i,ds in enumerate(DATASETS):
        g=cd[cd["dataset"].eq(ds)].copy(); s=g["size_abs_target_mean_gap"].to_numpy(float); b=g["balanced_abs_target_mean_gap"].to_numpy(float); ratio=np.divide(b,s,out=np.full_like(b,np.nan),where=s>0); finite=ratio[np.isfinite(ratio)]; x=np.full(len(finite),i)+rng.normal(0,0.045,len(finite)); ax.scatter(x,finite,s=9,alpha=0.55,color=C["teal"]); ax.scatter([i],[np.median(finite)],s=28,marker="D",color=C["navy2"],zorder=4)
    ax.axhline(1,color=C["gray"],ls="--",lw=0.8); ax.set_yscale("log"); ax.set_xticks(range(6),DATASETS,rotation=25,ha="right"); ax.set_ylabel("Balanced / size target-mean gap"); clean(ax,"y")

    ax=fig.add_subplot(gs[1,0]); panel(ax,"C","Collateral change: largest scaffold fraction")
    for i,ds in enumerate(DATASETS):
        g=cd[cd["dataset"].eq(ds)]; d=g["delta_balanced_minus_size_largest_test_scaffold_fraction"].to_numpy(float); x=np.full(len(d),i)+rng.normal(0,0.045,len(d)); ax.scatter(x,d,s=9,alpha=0.55,color=C["orange"]); ax.scatter([i],[np.mean(d)],s=28,marker="D",color=C["navy2"])
    ax.axhline(0,color=C["gray"],ls="--",lw=0.8); ax.set_xticks(range(6),DATASETS,rotation=25,ha="right"); ax.set_ylabel("Balanced − size"); clean(ax,"y")

    ax=fig.add_subplot(gs[1,1]); panel(ax,"D","Collateral change: effective scaffold number")
    for i,ds in enumerate(DATASETS):
        g=cd[cd["dataset"].eq(ds)]; s=g["size_effective_test_scaffolds"].to_numpy(float); b=g["balanced_effective_test_scaffolds"].to_numpy(float); v=np.log2(b/s); x=np.full(len(v),i)+rng.normal(0,0.045,len(v)); ax.scatter(x,v,s=9,alpha=0.55,color=C["navy"]); ax.scatter([i],[np.mean(v)],s=28,marker="D",color=C["teal2"])
    ax.axhline(0,color=C["gray"],ls="--",lw=0.8); ax.set_xticks(range(6),DATASETS,rotation=25,ha="right"); ax.set_ylabel("log2(balanced / size)"); clean(ax,"y")
    fig.suptitle("Target-mean-aware selection changes predictive difficulty and other benchmark properties",fontsize=10.0,fontweight="bold",y=0.99)
    save(fig,"figure6_collateral_diagnostics_v3")


def supplementary_figures() -> None:
    cleaning=pd.read_csv(need(CLEAN),keep_default_na=False)
    fig,ax=plt.subplots(figsize=(7.15,3.45)); x=np.arange(6); raw=cleaning["raw_rows"].to_numpy(float); final=cleaning["final_clean_unique_molecules"].to_numpy(float)
    ax.bar(x,raw,width=0.62,color=C["pale_blue"],edgecolor=C["mid"],label="Raw rows"); ax.bar(x,final,width=0.48,color=[C["navy"]]*4+[C["teal"]]*2,label="Final unique molecules"); ax.set_yscale("log"); ax.set_xticks(x,DATASETS); ax.set_ylabel("Rows"); clean(ax,"y"); ax.legend(frameon=False,ncol=2)
    for i,r in cleaning.iterrows(): ax.text(i,float(r.final_clean_unique_molecules)*1.08,f"−{int(r.raw_rows-r.final_clean_unique_molecules)}" if r.raw_rows!=r.final_clean_unique_molecules else "0",ha="center",fontsize=6.4,color=C["orange2"])
    ax.set_title("Audited raw-to-clean molecular-data construction",fontweight="bold"); save(fig,"figureS1_dataset_construction_v3")

    fig,axs=plt.subplots(1,2,figsize=(7.15,3.25),wspace=0.33)
    for j,ds in enumerate(REG):
        ax=axs[j];
        for vals,col,mark,label in [(BUDGET_SINGLE[ds],C["navy"],"o","single-group"),(BUDGET_SINGLETON[ds],C["orange"],"s","singleton")]:
            x=np.array(list(vals)); y=np.array(list(vals.values())); ax.plot(x,y,marker=mark,ms=3,lw=1.1,color=col,label=label)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(ds,fontweight="bold"); ax.set_xlabel("Candidate budget"); ax.set_ylabel("Mean target-mean gap"); clean(ax,"both"); ax.legend(frameon=False)
    fig.suptitle("Candidate-budget behavior under alternative acyclic-scaffold semantics",fontsize=9.6,fontweight="bold"); save(fig,"figureS2_budget_semantics_v3")

    vals=np.array([[MULTI[d],SCAFF_CHANGED[d],SIM090[d],CONFLICT[d]] for d in ["BBBP","ClinTox","HIV"]],float); log=np.log10(vals+1)
    cmap=LinearSegmentedColormap.from_list("audit",["#FFF8DB","#92D3B5","#2E9DB6","#153B73"])
    fig,ax=plt.subplots(figsize=(7.15,2.85)); im=ax.imshow(log,cmap=cmap,aspect="auto"); ax.set_xticks(range(4),["Multi-component","Scaffold changed","Similarity < 0.90","Conflict groups"]); ax.set_yticks(range(3),["BBBP","ClinTox","HIV"])
    for i in range(3):
        for j in range(4): ax.text(j,i,f"{int(vals[i,j]):,}",ha="center",va="center",fontsize=7.5,color="white" if log[i,j]>2.2 else C["ink"],fontweight="bold")
    fig.colorbar(im,ax=ax,fraction=0.025,pad=0.02,label="log10(count + 1)"); ax.set_title("Disconnected-component structural-audit summary",fontweight="bold"); save(fig,"figureS3_multicomponent_audit_v3")

    sup=pd.read_csv(need(SUPPORT),keep_default_na=False)
    # Classification heatmap: 12 rows x 6 metrics, normalized within each metric by max absolute mean effect.
    cls=sup[sup["task_type"].eq("classification")].copy(); cls["do"]=cls["dataset"].map(DATASET_ORDER); cls["mo"]=cls["model"].map(MODEL_ORDER); rows=cls[["dataset","model"]].drop_duplicates().sort_values(["dataset","model"],key=lambda col: col.map(DATASET_ORDER) if col.name=="dataset" else col.map(MODEL_ORDER)); metrics=["roc_auc","average_precision","f1","accuracy","balanced_accuracy","brier_score"]
    arr=[]; labels=[]
    for _,rr in rows.iterrows():
        labels.append(f"{rr.dataset} · {rr.model}"); valsrow=[]
        for m in metrics:
            v=float(cls[(cls["dataset"].eq(rr.dataset))&(cls["model"].eq(rr.model))&(cls["metric"].eq(m))]["mean_effect_positive_is_balanced_better"].iloc[0]); scale=float(cls[cls["metric"].eq(m)]["mean_effect_positive_is_balanced_better"].abs().max()); valsrow.append(v/scale if scale>0 else 0)
        arr.append(valsrow)
    fig,ax=plt.subplots(figsize=(7.15,4.4)); div=LinearSegmentedColormap.from_list("div",[C["orange"],"#FFFFFF",C["teal"]]); im=ax.imshow(np.array(arr),cmap=div,norm=TwoSlopeNorm(vmin=-1,vcenter=0,vmax=1),aspect="auto"); ax.set_yticks(range(len(labels)),labels); ax.set_xticks(range(len(metrics)),["AUC","AP","F1","Accuracy","Bal. acc.","Brier"]); ax.set_title("Supporting classification metrics: signed mean effects (column-normalized)",fontweight="bold"); fig.colorbar(im,ax=ax,fraction=0.025,pad=0.02,label="Signed relative effect"); save(fig,"figureS4_supporting_metrics_v3")

    seed=pd.read_csv(need(SEED),keep_default_na=False); seed=seed[seed["freeze_label"].isin(["main_classification","main_regression","acyclic_singleton_sensitivity"])]
    cells=seed[["freeze_label","dataset","model"]].drop_duplicates(); fig,axs=plt.subplots(1,3,figsize=(7.15,3.35),wspace=0.38)
    for ax,(label,title) in zip(axs,[("main_classification","Primary classification"),("main_regression","Primary regression"),("acyclic_singleton_sensitivity","Singleton regression")]):
        sub=seed[seed["freeze_label"].eq(label)]; combos=sub[["dataset","model"]].drop_duplicates();
        for _,r in combos.iterrows():
            g=sub[(sub["dataset"].eq(r.dataset))&(sub["model"].eq(r.model))].sort_values("model_seed"); ax.plot(g["model_seed"],g["mean_effect"],marker="o",ms=2.8,lw=0.9,label=f"{r.dataset}·{r.model}")
        ax.axhline(0,color=C["gray"],ls="--",lw=0.8); ax.set_xticks([17,29,43]); ax.set_xlabel("Model seed"); ax.set_title(title,fontweight="bold",fontsize=8.2); clean(ax,"y")
        if label=="main_classification": ax.set_ylabel("Mean paired effect")
    handles,labels=axs[0].get_legend_handles_labels(); fig.legend(handles,labels,loc="lower center",ncol=4,frameon=False,bbox_to_anchor=(0.5,-0.01),fontsize=6.0); fig.suptitle("Predeclared RF/XGB model-seed sensitivity on five fixed partition seeds",fontsize=9.5,fontweight="bold"); fig.subplots_adjust(bottom=0.22,top=0.83); save(fig,"figureS5_model_seed_sensitivity_v3")


def main() -> None:
    for path in [PRIMARY,SINGLETON,PARENT_CMP,SUPPORT,COLLATERAL,MEAN_ONLY,SEED,CLEAN]: need(path)
    figure1(); figure2(); figure3(); figure4(); figure5(); figure6(); supplementary_figures()
    print("\nROUND-3 PUBLICATION-SIZE FIGURES: PASS")


if __name__ == "__main__":
    main()
