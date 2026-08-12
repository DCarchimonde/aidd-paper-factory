from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LATEX = ROOT / "paper1_latex"
PAPER = ROOT / "paper1_leakage_benchmark"


def strip_tex(text: str) -> str:
    text = re.sub(r"%.*", " ", text)
    text = re.sub(r"\\cite\w*\{[^}]*\}", " ", text)
    text = re.sub(r"\\[A-Za-z@]+(?:\[[^\]]*\])?\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\[A-Za-z@]+", " ", text)
    text = re.sub(r"[{}$~]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def abstract_word_count() -> int:
    text = (LATEX / "sections" / "abstract_chemometrics.tex").read_text(encoding="utf-8")
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text, re.S)
    if not m:
        raise AssertionError("Abstract environment not found")
    plain = strip_tex(m.group(1))
    return len(re.findall(r"\b[\w’'-]+\b", plain))


def keyword_count() -> int:
    text = (LATEX / "sections" / "abstract_chemometrics.tex").read_text(encoding="utf-8")
    m = re.search(r"Keywords:\}\s*(.+)", text)
    if not m:
        raise AssertionError("Keywords line not found")
    return len([x for x in m.group(1).split(";") if x.strip()])


def running_title_length() -> int:
    text = (LATEX / "main.tex").read_text(encoding="utf-8")
    m = re.search(r"\\newcommand\{\\runningtitle\}\{([^}]*)\}", text)
    if not m:
        raise AssertionError("Running title macro not found")
    return len(m.group(1))


def count_environments() -> tuple[int, int]:
    sources = [
        LATEX / "sections" / "methods_chemometrics.tex",
        LATEX / "sections" / "results_chemometrics.tex",
        LATEX / "sections" / "discussion_chemometrics.tex",
    ]
    text = "\n".join(p.read_text(encoding="utf-8") for p in sources)
    return text.count("\\begin{figure}"), text.count("\\begin{table}")


def manuscript_language_gate() -> list[str]:
    paths = [
        LATEX / "sections" / "abstract_chemometrics.tex",
        LATEX / "sections" / "introduction_chemometrics.tex",
        LATEX / "sections" / "methods_chemometrics.tex",
        LATEX / "sections" / "results_chemometrics.tex",
        LATEX / "sections" / "discussion_chemometrics.tex",
        LATEX / "statements.tex",
    ]
    combined = "\n".join(p.read_text(encoding="utf-8") for p in paths).lower()
    banned = [
        "isolates one such design factor",
        "isolate the contribution of target-distribution mismatch",
        "no reproducible classification gain",
        "proves equivalence",
        "numerical convergence was achieved",
        "intended to be preserved as an immutable tagged release",
    ]
    return [phrase for phrase in banned if phrase in combined]


def supplementary_language_gate() -> list[str]:
    paths = [LATEX / "appendix_chemometrics.tex", LATEX / "supplementary_figures_v3.tex"]
    generated = LATEX / "generated" / "q1_collateral_table_v3.tex"
    if generated.exists():
        paths.append(generated)
    combined = "\n".join(p.read_text(encoding="utf-8") for p in paths).lower()
    banned = [
        "supporting metrics preserve direction",
        "\\caption{supporting classification metrics.",
        "balanced/size ratio",
        "script 20 for the q1 scientific controls; and script 21 for publication-size figures",
    ]
    return [phrase for phrase in banned if phrase in combined]


def journal_format_gate() -> None:
    main = (LATEX / "main.tex").read_text(encoding="utf-8")
    results = (LATEX / "sections" / "results_chemometrics.tex").read_text(encoding="utf-8")
    refs = (LATEX / "references_joc.tex").read_text(encoding="utf-8")

    required_main = [
        "\\usepackage[margin=3cm]{geometry}",
        "\\doublespacing",
        "\\renewcommand{\\thetable}{\\Roman{table}}",
        "\\input{references_joc}",
    ]
    missing = [token for token in required_main if token not in main]
    if missing:
        raise AssertionError("Journal of Chemometrics manuscript formatting missing: " + "; ".join(missing))
    if "\\bibliography{" in main:
        raise AssertionError("Legacy BibTeX reference output remains in submission main.tex")
    if results.count("width=0.895\\textwidth") != 6:
        raise AssertionError("All six main figures must be placed at the journal reproduction width")

    bibitems = refs.count("\\bibitem{")
    if bibitems != 24:
        raise AssertionError(f"Expected 24 audited references, found {bibitems}")
    reference_checks = [
        "Landrum GA, Beckers M, Lanini J, Schneider N, Stiefl N, Riniker S.",
        "Netzeva TI, Worth A, Aldenberg T, et al.",
        "doi:10.5281/zenodo.21291217",
        "\\textit{J. Chemom.}",
        "\\textit{J. Chem. Inf. Model.}",
    ]
    missing_refs = [token for token in reference_checks if token not in refs]
    if missing_refs:
        raise AssertionError("Audited reference metadata/style missing: " + "; ".join(missing_refs))


def pdf_pages(path: Path) -> int | None:
    pdfinfo = shutil.which("pdfinfo")
    if pdfinfo:
        out = subprocess.check_output([pdfinfo, str(path)], text=True, errors="ignore")
        m = re.search(r"^Pages:\s+(\d+)", out, re.M)
        if m:
            return int(m.group(1))
    try:
        from pypdf import PdfReader  # type: ignore
        return len(PdfReader(str(path)).pages)
    except Exception:
        try:
            from PyPDF2 import PdfReader  # type: ignore
            return len(PdfReader(str(path)).pages)
        except Exception:
            return None


def source_gate() -> None:
    words = abstract_word_count()
    keywords = keyword_count()
    running = running_title_length()
    figures, tables = count_environments()
    banned = manuscript_language_gate()
    si_banned = supplementary_language_gate()

    print(f"Abstract words: {words}")
    print(f"Keywords: {keywords}")
    print(f"Running-title characters: {running}")
    print(f"Main figures: {figures}")
    print(f"Main tables: {tables}")

    if words > 250:
        raise AssertionError(f"Abstract exceeds 250 words: {words}")
    if not 3 <= keywords <= 5:
        raise AssertionError(f"Keyword count must be 3–5: {keywords}")
    if running > 70:
        raise AssertionError(f"Running title exceeds 70 characters: {running}")
    if figures > 7:
        raise AssertionError(f"Main figure count exceeds 7: {figures}")
    if tables > 4:
        raise AssertionError(f"Main table count exceeds 4: {tables}")
    if banned:
        raise AssertionError("Overstrong/outdated manuscript wording remains: " + "; ".join(banned))
    if si_banned:
        raise AssertionError("Outdated/ambiguous Supporting Information wording remains: " + "; ".join(si_banned))

    journal_format_gate()

    figure_stems = [
        "figure1_audit_framework_v3", "figure2_primary_effects_v3",
        "figure3_acyclic_sensitivity_v3", "figure4_dominant_fragment_sensitivity_v3",
        "figure5_candidate_budget_audit_v3", "figure6_collateral_diagnostics_v3",
        "figureS1_dataset_construction_v3", "figureS2_budget_semantics_v3",
        "figureS3_multicomponent_audit_v3", "figureS4_supporting_metrics_v3",
        "figureS5_model_seed_sensitivity_v3",
    ]
    required = [
        PAPER / "results" / "tables" / "q1_mean_only_regression_summary_v3.csv",
        PAPER / "results" / "tables" / "q1_collateral_diagnostics_summary_v3.csv",
        PAPER / "results" / "tables" / "q1_model_seed_summary_v3.csv",
        PAPER / "results" / "tables" / "q1_cleaning_accounting_v3.csv",
        PAPER / "results" / "tables" / "q1_raw_source_provenance_v3.csv",
        PAPER / "scripts" / "25_finalize_submission_figures_v3.py",
        PAPER / "scripts" / "26_final_artwork_qc_v3.py",
        PAPER / "scripts" / "27_joc_submission_artwork_v3.py",
        LATEX / "references_joc.tex",
    ]
    for stem in figure_stems:
        required.extend([
            PAPER / "results" / "figures" / f"{stem}.pdf",
            PAPER / "results" / "figures" / f"{stem}.tiff",
        ])
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Q1 final artifacts missing: " + ", ".join(missing))


def post_build_gate() -> None:
    build = LATEX / "build_q1_final_v3"
    main = build / "main.pdf"
    si = build / "supplementary.pdf"
    for p in [main, si]:
        if not p.exists() or p.stat().st_size == 0:
            raise FileNotFoundError(p)
    pages = pdf_pages(main)
    if pages is None:
        raise AssertionError("Main-manuscript page count could not be verified automatically")
    print(f"Main-manuscript pages: {pages}")
    if pages > 25:
        raise AssertionError(
            f"Journal of Chemometrics Original Research limit exceeded: {pages} > 25 double-spaced pages"
        )

    for log_name in ["main.log", "supplementary.log"]:
        log = build / log_name
        if not log.exists():
            continue
        text = log.read_text(encoding="utf-8", errors="ignore").lower()
        if "undefined references" in text or "there were undefined references" in text:
            raise AssertionError(f"Undefined references remain in {log_name}")
        if "citation" in text and "undefined" in text:
            raise AssertionError(f"Undefined citations remain in {log_name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--post-build", action="store_true")
    args = parser.parse_args()
    source_gate()
    if args.post_build:
        post_build_gate()
    print("Q1 + JOURNAL OF CHEMOMETRICS SUBMISSION GATE: PASS")


if __name__ == "__main__":
    main()
