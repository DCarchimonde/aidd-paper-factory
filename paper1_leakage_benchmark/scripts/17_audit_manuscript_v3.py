from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LATEX = ROOT / "paper1_latex"
PAPER = ROOT / "paper1_leakage_benchmark"
FIG_DIR = PAPER / "results" / "figures"

ACTIVE_TEX = [
    LATEX / "main.tex",
    LATEX / "sections" / "abstract_chemometrics.tex",
    LATEX / "sections" / "introduction_chemometrics.tex",
    LATEX / "sections" / "related_work_chemometrics.tex",
    LATEX / "sections" / "methods_chemometrics.tex",
    LATEX / "sections" / "results_chemometrics.tex",
    LATEX / "sections" / "discussion_chemometrics.tex",
    LATEX / "statements.tex",
    LATEX / "supplementary.tex",
    LATEX / "appendix_chemometrics.tex",
]

BIBS = [
    LATEX / "references.bib",
    LATEX / "references_extra.bib",
    LATEX / "references_recent.bib",
    LATEX / "references_v3.bib",
]

REQUIRED_RESULTS = [
    PAPER / "results" / "tables" / "primary_inference_summary_v3.csv",
    PAPER / "results" / "tables" / "acyclic_singleton_sensitivity_v3.csv",
    PAPER
    / "results"
    / "parent_fragment_sensitivity_v3"
    / "tables"
    / "parent_fragment_vs_main_comparison_v3.csv",
]

REQUIRED_FIGURES = [
    FIG_DIR / "figure1_audit_framework_v3.pdf",
    FIG_DIR / "figure2_primary_effects_v3.pdf",
    FIG_DIR / "figure3_acyclic_sensitivity_v3.pdf",
    FIG_DIR / "figure4_dominant_fragment_sensitivity_v3.pdf",
]

BANNED_ACTIVE_PHRASES = [
    "nine smaller gaps",
    "six larger gaps",
    "three inconclusive changes",
    "generalization-gap reduction",
    "paired generalization-gap effects",
    "5,000-draw random-scaffold",
    "ordinary scaffold gap",
]

EXPECTED_TITLE = (
    "Dissecting Molecular Benchmark Construction: A Paired Chemometric Audit of "
    "Target Balance, Scaffold Semantics, and Molecular Representation"
)


def read(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")


def collect_bib_keys() -> tuple[set[str], list[str]]:
    all_keys: list[str] = []
    for bib in BIBS:
        text = read(bib)
        all_keys.extend(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", text))
    duplicates = sorted({key for key in all_keys if all_keys.count(key) > 1})
    return set(all_keys), duplicates


def citation_keys(text: str) -> set[str]:
    found: set[str] = set()
    for payload in re.findall(r"\\cite(?:p|t)?\{([^}]+)\}", text):
        found.update(key.strip() for key in payload.split(",") if key.strip())
    return found


def main() -> None:
    problems: list[str] = []
    texts = {path: read(path) for path in ACTIVE_TEX}
    active = "\n".join(texts.values())

    if EXPECTED_TITLE not in texts[LATEX / "main.tex"]:
        problems.append("Current v3 title not found in main.tex")

    lower = active.lower()
    for phrase in BANNED_ACTIVE_PHRASES:
        if phrase.lower() in lower:
            problems.append(f"Stale active-manuscript phrase found: {phrase!r}")

    if "results_chemometrics_target" in texts[LATEX / "sections" / "results_chemometrics.tex"]:
        problems.append("Active Results still inputs an obsolete results subfile")
    if "results_chemometrics_performance" in texts[LATEX / "sections" / "results_chemometrics.tex"]:
        problems.append("Active Results still inputs an obsolete results subfile")
    if "results_chemometrics_similarity" in texts[LATEX / "sections" / "results_chemometrics.tex"]:
        problems.append("Active Results still inputs an obsolete results subfile")

    bib_keys, duplicate_bib_keys = collect_bib_keys()
    if duplicate_bib_keys:
        problems.append(f"Duplicate bibliography keys: {duplicate_bib_keys}")
    cites = citation_keys(active)
    missing_citations = sorted(cites - bib_keys)
    if missing_citations:
        problems.append(f"Missing bibliography entries: {missing_citations}")

    for path in REQUIRED_RESULTS:
        if not path.exists():
            problems.append(f"Missing authoritative result table: {path}")
    for path in REQUIRED_FIGURES:
        if not path.exists():
            problems.append(
                f"Missing manuscript figure: {path}. Run scripts/16_build_manuscript_assets_v3.py"
            )

    required_terms = [
        "exact-size",
        "partition hash",
        "dominant-fragment",
        "protocol-conditioned",
        "20 unique",
    ]
    for term in required_terms:
        if term.lower() not in lower:
            problems.append(f"Expected v3 manuscript concept missing: {term}")

    print(f"Active TeX files checked: {len(ACTIVE_TEX)}")
    print(f"Bibliography keys: {len(bib_keys)}")
    print(f"Cited keys: {len(cites)}")
    print(f"Required result tables: {len(REQUIRED_RESULTS)}")
    print(f"Required figures: {len(REQUIRED_FIGURES)}")

    if problems:
        print("\nMANUSCRIPT AUDIT V3 FAILED")
        for item in problems:
            print(f"- {item}")
        raise SystemExit(1)

    print("\nMANUSCRIPT AUDIT V3 PASSED")


if __name__ == "__main__":
    main()
