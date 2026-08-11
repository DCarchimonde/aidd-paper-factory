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
    ]
    combined = "\n".join(p.read_text(encoding="utf-8") for p in paths).lower()
    banned = [
        "isolates one such design factor",
        "isolate the contribution of target-distribution mismatch",
        "no reproducible classification gain",
        "proves equivalence",
        "numerical convergence was achieved",
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

    figure_names = [
        "figure1_audit_framework_v3.pdf", "figure2_primary_effects_v3.pdf",
        "figure3_acyclic_sensitivity_v3.pdf", "figure4_dominant_fragment_sensitivity_v3.pdf",
        "figure5_candidate_budget_audit_v3.pdf", "figure6_collateral_diagnostics_v3.pdf",
        "figureS1_dataset_construction_v3.pdf", "figureS2_budget_semantics_v3.pdf",
        "figureS3_multicomponent_audit_v3.pdf", "figureS4_supporting_metrics_v3.pdf",
        "figureS5_model_seed_sensitivity_v3.pdf",
    ]
    required = [
        PAPER / "results" / "tables" / "q1_mean_only_regression_summary_v3.csv",
        PAPER / "results" / "tables" / "q1_collateral_diagnostics_summary_v3.csv",
        PAPER / "results" / "tables" / "q1_model_seed_summary_v3.csv",
        PAPER / "results" / "tables" / "q1_cleaning_accounting_v3.csv",
        PAPER / "results" / "tables" / "q1_raw_source_provenance_v3.csv",
        PAPER / "scripts" / "25_finalize_submission_figures_v3.py",
    ] + [PAPER / "results" / "figures" / name for name in figure_names]
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
        print("Main-manuscript page count: unavailable automatically; visual audit required.")
    else:
        print(f"Main-manuscript pages: {pages}")
        if pages > 25:
            print(
                f"PAGE-COUNT REVIEW: main manuscript is {pages} pages. Keep the compiled artifacts, but reduce the journal submission copy if the target journal counts embedded figures/tables toward its 25-page guidance."
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
    print("Q1 SUBMISSION GATE: PASS")


if __name__ == "__main__":
    main()
