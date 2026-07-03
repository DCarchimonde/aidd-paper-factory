from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "paper1_leakage_benchmark"
TABLE_DIR = PAPER_DIR / "results" / "tables"
FIGURE_DIR = PAPER_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

main_gap_path = TABLE_DIR / "paper1_main_gap_table.csv"
shift_path = TABLE_DIR / "paper1_split_shift_table.csv"
sim_path = TABLE_DIR / "paper1_similarity_audit_compact.csv"

if not main_gap_path.exists():
    raise FileNotFoundError(f"Missing {main_gap_path}. Run compare_splits.py first.")
if not shift_path.exists():
    raise FileNotFoundError(f"Missing {shift_path}. Run compare_splits.py first.")
if not sim_path.exists():
    raise FileNotFoundError(f"Missing {sim_path}. Run similarity_audit.py first.")

main_gap = pd.read_csv(main_gap_path)
shift = pd.read_csv(shift_path)
sim = pd.read_csv(sim_path)

# Figure 1: ordinary vs balanced generalization gap.
plot_gap = main_gap.copy()
plot_gap["label"] = plot_gap["dataset"] + " / " + plot_gap["model"]
plot_gap = plot_gap.sort_values(["dataset", "model"]).reset_index(drop=True)
x = range(len(plot_gap))
width = 0.4

plt.figure(figsize=(14, 6))
plt.bar([i - width / 2 for i in x], plot_gap["ordinary_gap_mean"], width=width, label="ordinary scaffold gap")
plt.bar([i + width / 2 for i in x], plot_gap["balanced_gap_mean"], width=width, label="balanced scaffold gap")
plt.axhline(0, linewidth=1)
plt.xticks(list(x), plot_gap["label"], rotation=75, ha="right")
plt.ylabel("Generalization gap")
plt.title("Random split gap before and after target-balanced scaffold splitting")
plt.legend()
plt.tight_layout()
out_path = FIGURE_DIR / "figure1_gap_ordinary_vs_balanced.png"
plt.savefig(out_path, dpi=300)
plt.close()
print("saved", out_path)

# Figure 2: split-induced target shift.
pivot = shift.pivot(index="dataset", columns="split_type", values="mean_abs_target_gap").reset_index()
for col in ["split_random", "split_scaffold", "balanced_scaffold"]:
    if col not in pivot.columns:
        pivot[col] = 0.0
x = range(len(pivot))
width = 0.25

plt.figure(figsize=(10, 5))
plt.bar([i - width for i in x], pivot["split_random"], width=width, label="random")
plt.bar(list(x), pivot["split_scaffold"], width=width, label="ordinary scaffold")
plt.bar([i + width for i in x], pivot["balanced_scaffold"], width=width, label="balanced scaffold")
plt.xticks(list(x), pivot["dataset"])
plt.ylabel("Mean absolute target gap")
plt.title("Target distribution shift by split strategy")
plt.legend()
plt.tight_layout()
out_path = FIGURE_DIR / "figure2_split_target_shift.png"
plt.savefig(out_path, dpi=300)
plt.close()
print("saved", out_path)

# Figure 3: test-to-train chemical similarity.
sim_label = {
    "random": "random",
    "ordinary_scaffold": "ordinary scaffold",
    "balanced_scaffold": "balanced scaffold",
}
sim = sim[sim["split"].isin(sim_label.keys())].copy()
sim["split_label"] = sim["split"].map(sim_label)
sim_pivot = sim.pivot(index="dataset", columns="split_label", values="mean_max_tanimoto").reset_index()
for col in ["random", "ordinary scaffold", "balanced scaffold"]:
    if col not in sim_pivot.columns:
        sim_pivot[col] = 0.0
sim_pivot = sim_pivot.sort_values("dataset").reset_index(drop=True)
x = range(len(sim_pivot))
width = 0.25

plt.figure(figsize=(10, 5))
plt.bar([i - width for i in x], sim_pivot["random"], width=width, label="random")
plt.bar(list(x), sim_pivot["ordinary scaffold"], width=width, label="ordinary scaffold")
plt.bar([i + width for i in x], sim_pivot["balanced scaffold"], width=width, label="balanced scaffold")
plt.xticks(list(x), sim_pivot["dataset"])
plt.ylabel("Mean max test-to-train Tanimoto")
plt.title("Chemical similarity between test molecules and training set")
plt.legend()
plt.tight_layout()
out_path = FIGURE_DIR / "figure3_tanimoto_similarity.png"
plt.savefig(out_path, dpi=300)
plt.close()
print("saved", out_path)

# Manuscript-friendly rounded tables.
rounded_gap = main_gap.copy()
for col in ["ordinary_gap_mean", "ordinary_gap_std", "balanced_gap_mean", "balanced_gap_std", "gap_reduction_after_balancing"]:
    rounded_gap[col] = rounded_gap[col].round(3)
rounded_gap.to_csv(TABLE_DIR / "paper1_main_gap_table_rounded.csv", index=False)

rounded_shift = shift.copy()
rounded_shift["mean_abs_target_gap"] = rounded_shift["mean_abs_target_gap"].round(4)
rounded_shift.to_csv(TABLE_DIR / "paper1_split_shift_table_rounded.csv", index=False)

rounded_sim = sim.copy()
for col in ["mean_max_tanimoto", "median_max_tanimoto", "p90_max_tanimoto", "p95_max_tanimoto", "frac_test_ge_0_7", "frac_test_ge_0_8", "frac_test_ge_0_9"]:
    rounded_sim[col] = rounded_sim[col].round(3)
rounded_sim.to_csv(TABLE_DIR / "paper1_similarity_audit_compact_rounded.csv", index=False)

print("saved rounded manuscript tables")
