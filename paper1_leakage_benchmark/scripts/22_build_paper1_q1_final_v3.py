from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper1_leakage_benchmark"
LATEX = ROOT / "paper1_latex"
SCRIPTS = PAPER / "scripts"
BUILD = LATEX / "build_q1_final_v3"
BUNDLE = ROOT / "paper1_submission_q1_final_v3"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("\n>>>", " ".join(str(x) for x in cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def capture(cmd: list[str], cwd: Path | None = None) -> str:
    return subprocess.check_output(cmd, cwd=str(cwd) if cwd else None, text=True).strip()


def run_model_seed_sensitivity() -> None:
    script = SCRIPTS / "09_run_models_v3.py"
    for label in ["main_classification", "main_regression", "acyclic_singleton_sensitivity"]:
        run([
            sys.executable, str(script),
            "--freeze-label", label,
            "--run-type", "model_seed_sensitivity",
            "--protocols", "size_matched_scaffold,target_balanced_scaffold",
            "--models", "RF,XGB",
        ], cwd=ROOT)


def scientific_pipeline() -> None:
    run([sys.executable, str(SCRIPTS / "12_analyze_partition_effects_v3.py")], cwd=ROOT)
    run([sys.executable, str(SCRIPTS / "20a_build_q1_scientific_controls_safe_v3.py")], cwd=ROOT)
    run([sys.executable, str(SCRIPTS / "20b_build_q1_tex_tables_v3.py")], cwd=ROOT)
    run([sys.executable, str(SCRIPTS / "20d_capture_raw_source_provenance_v3.py")], cwd=ROOT)
    run([sys.executable, str(SCRIPTS / "20c_write_q1_result_text_v3.py")], cwd=ROOT)
    run([sys.executable, str(SCRIPTS / "21a_build_manuscript_assets_v3_round3_compat.py")], cwd=ROOT)
    run([sys.executable, str(SCRIPTS / "23_polish_q1_diagnostic_figures_v3.py")], cwd=ROOT)
    run([sys.executable, str(SCRIPTS / "25_finalize_submission_figures_v3.py")], cwd=ROOT)
    run([sys.executable, str(SCRIPTS / "26_final_artwork_qc_v3.py")], cwd=ROOT)
    run([sys.executable, str(SCRIPTS / "27_joc_submission_artwork_v3.py")], cwd=ROOT)
    run([sys.executable, str(SCRIPTS / "24_q1_submission_gate_v3.py")], cwd=ROOT)


def compile_latex() -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    latexmk = shutil.which("latexmk")
    if latexmk:
        for source in ["main.tex", "supplementary.tex"]:
            run([
                latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", "-file-line-error",
                f"-outdir={BUILD.name}", source,
            ], cwd=LATEX)
        return

    pdflatex = shutil.which("pdflatex")
    if not pdflatex:
        raise RuntimeError("LaTeX compiler not found. Install latexmk (preferred) or pdflatex.")

    main_text = (LATEX / "main.tex").read_text(encoding="utf-8")
    run([pdflatex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=LATEX)
    if "\\bibliography{" in main_text:
        bibtex = shutil.which("bibtex")
        if not bibtex:
            raise RuntimeError("BibTeX is required by main.tex but was not found.")
        run([bibtex, "main"], cwd=LATEX)
    run([pdflatex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=LATEX)
    run([pdflatex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=LATEX)
    run([pdflatex, "-interaction=nonstopmode", "-halt-on-error", "supplementary.tex"], cwd=LATEX)
    run([pdflatex, "-interaction=nonstopmode", "-halt-on-error", "supplementary.tex"], cwd=LATEX)
    for name in ["main.pdf", "supplementary.pdf"]:
        src = LATEX / name
        if src.exists():
            shutil.copy2(src, BUILD / name)


def audit_latex_logs() -> dict:
    report = {"overfull_hbox_lines": [], "undefined_reference_lines": [], "undefined_citation_lines": []}
    for name in ["main.log", "supplementary.log"]:
        path = BUILD / name
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            lower = line.lower()
            if "overfull \\hbox" in lower:
                report["overfull_hbox_lines"].append(f"{name}: {line}")
            if "undefined references" in lower or ("reference" in lower and "undefined" in lower):
                report["undefined_reference_lines"].append(f"{name}: {line}")
            if "citation" in lower and "undefined" in lower:
                report["undefined_citation_lines"].append(f"{name}: {line}")
    return report


def package_submission() -> None:
    BUNDLE.mkdir(parents=True, exist_ok=True)
    for name in ["main.pdf", "supplementary.pdf"]:
        src = BUILD / name
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, BUNDLE / name)

    fig_dir = PAPER / "results" / "figures"
    expected = [
        "figure1_audit_framework_v3", "figure2_primary_effects_v3",
        "figure3_acyclic_sensitivity_v3", "figure4_dominant_fragment_sensitivity_v3",
        "figure5_candidate_budget_audit_v3", "figure6_collateral_diagnostics_v3",
        "figureS1_dataset_construction_v3", "figureS2_budget_semantics_v3",
        "figureS3_multicomponent_audit_v3", "figureS4_supporting_metrics_v3",
        "figureS5_model_seed_sensitivity_v3",
    ]
    outfig = BUNDLE / "figures"
    outfig.mkdir(exist_ok=True)
    for stem in expected:
        for ext in [".pdf", ".tiff"]:
            src = fig_dir / f"{stem}{ext}"
            if not src.exists():
                raise FileNotFoundError(src)
            shutil.copy2(src, outfig / src.name)

    tables = PAPER / "results" / "tables"
    key_tables = [
        "primary_inference_summary_v3.csv", "supporting_metric_effects_v3.csv",
        "acyclic_singleton_sensitivity_v3.csv", "q1_mean_only_regression_summary_v3.csv",
        "q1_collateral_diagnostics_summary_v3.csv", "q1_model_seed_summary_v3.csv",
        "q1_cleaning_accounting_v3.csv", "q1_raw_source_provenance_v3.csv",
        "q1_environment_versions_v3.txt",
    ]
    outtables = BUNDLE / "tables"
    outtables.mkdir(exist_ok=True)
    for name in key_tables:
        src = tables / name
        if src.exists():
            shutil.copy2(src, outtables / name)

    # Keep the journal-formatted LaTeX sources together with the exact review PDFs.
    outsrc = BUNDLE / "latex_source"
    outsrc.mkdir(exist_ok=True)
    for name in [
        "main.tex", "supplementary.tex", "statements.tex", "appendix_chemometrics.tex",
        "supplementary_figures_v3.tex", "references_joc.tex",
    ]:
        src = LATEX / name
        if src.exists():
            shutil.copy2(src, outsrc / name)
    for dirname in ["sections", "generated"]:
        src = LATEX / dirname
        dst = outsrc / dirname
        if dst.exists():
            shutil.rmtree(dst)
        if src.exists():
            shutil.copytree(src, dst)

    git_sha = capture(["git", "rev-parse", "HEAD"], ROOT)
    git_branch = capture(["git", "branch", "--show-current"], ROOT)
    audit = audit_latex_logs()
    manifest = {
        "branch": git_branch,
        "commit": git_sha,
        "main_pdf": str((BUNDLE / "main.pdf").relative_to(ROOT)),
        "supplementary_pdf": str((BUNDLE / "supplementary.pdf").relative_to(ROOT)),
        "figures_pdf_and_tiff": expected,
        "reference_style": "Journal of Chemometrics / ACS-abbreviated journal titles",
        "artwork_font": "Arial-family sans serif",
        "artwork_max_size_mm": [140, 200],
        "latex_audit": audit,
    }
    (BUNDLE / "BUILD_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if audit["undefined_reference_lines"] or audit["undefined_citation_lines"]:
        raise AssertionError("Undefined LaTeX references/citations remain; inspect BUILD_MANIFEST.json")
    if audit["overfull_hbox_lines"]:
        print(f"\nNOTICE: {len(audit['overfull_hbox_lines'])} overfull-hbox log lines were detected; inspect if visually material.")


def main() -> None:
    print("=" * 78)
    print("PAPER 1 Q1 FINAL SCIENTIFIC + JOC FORMAT + ARTWORK BUILD")
    print("=" * 78)
    run_model_seed_sensitivity()
    scientific_pipeline()
    compile_latex()
    run([sys.executable, str(SCRIPTS / "24_q1_submission_gate_v3.py"), "--post-build"], cwd=ROOT)
    package_submission()
    print("\n" + "=" * 78)
    print("PAPER 1 Q1 FINAL BUILD: PASS")
    print("Main PDF:", BUNDLE / "main.pdf")
    print("SI PDF  :", BUNDLE / "supplementary.pdf")
    print("Manifest:", BUNDLE / "BUILD_MANIFEST.json")
    print("=" * 78)


if __name__ == "__main__":
    main()
