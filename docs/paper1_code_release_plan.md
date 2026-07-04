# Paper 1 Code Release Plan

Date: 2026-07-05

Goal: prepare a clean public repository containing only Paper 1 code, small result tables, figures, and manuscript source files.

Recommended repository name:

`paper1-split-diagnostic-molecular-benchmark`

Recommended contents:

```text
README.md
requirements.txt
shared_utils/
paper1_leakage_benchmark/scripts/
paper1_leakage_benchmark/results/tables/
paper1_leakage_benchmark/figures/
paper1_latex/
```

Do not include:

```text
raw downloaded datasets
large model files
local build files
private planning notes
unrelated future-paper folders
```

Recommended local preparation workflow:

```powershell
E:
cd E:\AIDD_Paper_Factory
mkdir E:\paper1_release
robocopy shared_utils E:\paper1_release\shared_utils /E
robocopy paper1_leakage_benchmark\scripts E:\paper1_release\paper1_leakage_benchmark\scripts /E
robocopy paper1_leakage_benchmark\results\tables E:\paper1_release\paper1_leakage_benchmark\results\tables /E
robocopy paper1_leakage_benchmark\figures E:\paper1_release\paper1_leakage_benchmark\figures /E
robocopy paper1_latex E:\paper1_release\paper1_latex /E /XD __pycache__ /XF *.aux *.bbl *.blg *.fdb_latexmk *.fls *.out *.pdf
```

After creating a new GitHub repository manually, push with:

```powershell
cd E:\paper1_release
git init
git add .
git commit -m "Initial Paper 1 reproducibility release"
git branch -M main
git remote add origin git@github.com:DCarchimonde/paper1-split-diagnostic-molecular-benchmark.git
git push -u origin main
```

Reason: the current `aidd-paper-factory` repository is a working factory. The submission repository should be smaller, cleaner, and easier for reviewers to reproduce.
