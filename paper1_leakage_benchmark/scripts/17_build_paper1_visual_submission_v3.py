from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "paper1_leakage_benchmark"
LATEX_DIR = ROOT / "paper1_latex"
FIG_SCRIPT = PAPER_DIR / "scripts" / "19_build_manuscript_assets_v3_round2.py"
BUILD_DIR = LATEX_DIR / "build_visual_v3_round2"

EXPECTED_FIGURES = [
    "figure1_audit_framework_v3.pdf",
    "figure2_primary_effects_v3.pdf",
    "figure3_acyclic_sensitivity_v3.pdf",
    "figure4_dominant_fragment_sensitivity_v3.pdf",
    "figure5_candidate_budget_audit_v3.pdf",
    "figure6_claim_stability_map_v3.pdf",
    "figureS1_dataset_construction_v3.pdf",
    "figureS2_budget_semantics_v3.pdf",
    "figureS3_multicomponent_audit_v3.pdf",
]


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("\n>>>", " ".join(cmd))
    completed = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def build_figures() -> None:
    run([sys.executable, str(FIG_SCRIPT)], cwd=ROOT)
    fig_dir = PAPER_DIR / "results" / "figures"
    missing = [name for name in EXPECTED_FIGURES if not (fig_dir / name).exists()]
    if missing:
        raise FileNotFoundError("Figure generation completed but outputs are missing: " + ", ".join(missing))
    print(f"\nVerified {len(EXPECTED_FIGURES)}/{len(EXPECTED_FIGURES)} expected PDF figures.")


def compile_with_latexmk() -> bool:
    latexmk = shutil.which("latexmk")
    if not latexmk:
        return False
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    for source in ("main.tex", "supplementary.tex"):
        run([
            latexmk,
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            f"-outdir={BUILD_DIR.name}",
            source,
        ], cwd=LATEX_DIR)
    return True


def compile_with_pdflatex() -> None:
    pdflatex = shutil.which("pdflatex")
    bibtex = shutil.which("bibtex")
    if not pdflatex:
        raise RuntimeError(
            "No LaTeX compiler was found. Figures were generated successfully, but manuscript PDFs "
            "cannot be rebuilt. Install latexmk (preferred) or pdflatex/bibtex and rerun this script."
        )

    run([pdflatex, "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", "main.tex"], cwd=LATEX_DIR)
    if not bibtex:
        raise RuntimeError("pdflatex was found but bibtex was not; main manuscript bibliography cannot be rebuilt.")
    run([bibtex, "main"], cwd=LATEX_DIR)
    run([pdflatex, "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", "main.tex"], cwd=LATEX_DIR)
    run([pdflatex, "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", "main.tex"], cwd=LATEX_DIR)
    run([pdflatex, "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", "supplementary.tex"], cwd=LATEX_DIR)
    run([pdflatex, "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", "supplementary.tex"], cwd=LATEX_DIR)

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("main.pdf", "supplementary.pdf"):
        source = LATEX_DIR / name
        if not source.exists():
            raise FileNotFoundError(f"Expected compiled PDF not found: {source}")
        shutil.copy2(source, BUILD_DIR / name)


def verify_pdfs() -> None:
    for path in (BUILD_DIR / "main.pdf", BUILD_DIR / "supplementary.pdf"):
        if not path.exists() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"Missing or empty compiled PDF: {path}")
        print(f"OK: {path} ({path.stat().st_size / (1024 * 1024):.2f} MiB)")


def main() -> None:
    print("=" * 78)
    print("PAPER 1 V3 — ROUND-2 FIGURE + LAYOUT POLISH")
    print("=" * 78)
    print("No models are refit. Frozen partitions, statistics, and reported numerical results are unchanged.")
    build_figures()
    if compile_with_latexmk():
        print("\nLaTeX build completed with latexmk.")
    else:
        print("\nlatexmk not found; using pdflatex/bibtex fallback.")
        compile_with_pdflatex()
    verify_pdfs()
    print("\n" + "=" * 78)
    print("PAPER 1 ROUND-2 VISUAL BUILD: PASS")
    print(f"Main manuscript: {BUILD_DIR / 'main.pdf'}")
    print(f"Supporting information: {BUILD_DIR / 'supplementary.pdf'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
