from __future__ import annotations

import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LATEX = ROOT / "paper1_latex"
SCRIPTS = ROOT / "paper1_leakage_benchmark" / "scripts"
LEGACY = SCRIPTS / "22_build_paper1_q1_final_v3.py"
LEGACY_GATE = SCRIPTS / "24_q1_submission_gate_v3.py"
ARTWORK = SCRIPTS / "31_publication_artwork_final_v3.py"
REFERENCES = SCRIPTS / "build_paper1_references_final_v3.py"
FINAL_REFS = LATEX / "references_joc_submission.tex"
SI_SOURCE = LATEX / "appendix_chemometrics.tex"
SI_FINAL = LATEX / "appendix_chemometrics_submission.tex"
MEAN_ONLY_TEX = LATEX / "generated" / "q1_mean_only_table_v3.tex"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def citation_order() -> list[str]:
    paths = [
        LATEX / "sections" / "abstract_chemometrics.tex",
        LATEX / "sections" / "introduction_chemometrics.tex",
        LATEX / "sections" / "methods_chemometrics.tex",
        LATEX / "sections" / "results_chemometrics.tex",
        LATEX / "sections" / "discussion_chemometrics.tex",
        LATEX / "statements.tex",
    ]
    seen: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"\\cite\w*\{([^}]*)\}", text):
            for key in match.group(1).split(","):
                key = key.strip()
                if key and key not in seen:
                    seen.append(key)
    return seen


def _hash_lines(value: str) -> str:
    if len(value) != 64:
        raise AssertionError(f"Expected 64-character SHA-256, found {len(value)} characters")
    return f"\\texttt{{{value[:32]}}}\\\\\n\\texttt{{{value[32:]}}}"


def _fit_acyclic_table(text: str) -> str:
    """Constrain the one direct SI table that is naturally wider than the text block."""
    anchor = "\\label{tab:si-acyclic}"
    if anchor not in text:
        raise AssertionError("Acyclic-sensitivity SI table label not found")
    left, right = text.split(anchor, 1)
    begin = "\\begin{tabular}{llrrrrr}"
    end = "\\end{tabular}\n\\end{table}"
    if begin not in right or end not in right:
        raise AssertionError("Acyclic-sensitivity SI table structure not found")
    right = right.replace(begin, "\\resizebox{\\textwidth}{!}{%\n" + begin, 1)
    right = right.replace(end, "\\end{tabular}%\n}\n\\end{table}", 1)
    return left + anchor + right


def build_si_submission_source() -> None:
    text = SI_SOURCE.read_text(encoding="utf-8")

    hash_re = re.compile(
        r"The frozen production registry recorded partition-registry SHA-256 "
        r"\\texttt\{([0-9a-f]+)\} and artifact-registry SHA-256 "
        r"\\texttt\{([0-9a-f]+)\}\."
    )
    match = hash_re.search(text)
    if match is None:
        raise AssertionError("Frozen registry hashes were not found in Supporting Information source")
    partition_hash, artifact_hash = match.groups()
    hash_block = (
        "The frozen production registry hashes are preserved below for exact artifact verification:\n\n"
        "\\begin{quote}\n\\small\n"
        "\\textbf{Partition registry SHA-256}\\\\[0.20em]\n"
        + _hash_lines(partition_hash)
        + "\\\\[0.60em]\n"
        "\\textbf{Artifact registry SHA-256}\\\\[0.20em]\n"
        + _hash_lines(artifact_hash)
        + "\n\\end{quote}"
    )
    text = hash_re.sub(lambda _: hash_block, text, count=1)
    text = _fit_acyclic_table(text)

    marker = "\\section{Reproducibility and Artifact Map}"
    if marker not in text:
        raise AssertionError("Supporting Information reproducibility section not found")
    prefix = text.split(marker, 1)[0]

    repro = r"""\section{Reproducibility and Artifact Map}
\label{sec:si-repro}

The scientific workflow separates frozen analysis from publication-only rendering. Dedicated shared utilities define molecular identity and row lineage, scaffold semantics and partition hashing, and target-blind candidate generation with exact-size pairing. The frozen protocol documents record the split construction, predictive-model configuration, and dominant-fragment sensitivity. Scripts 05--15 perform the scientific audit, protocol freezing, model fitting, completeness checks, partition-level inference, and disconnected-component sensitivity.

The authoritative publication-final pass is implemented by script 32, with all eleven publication figures rendered exactly once by script 31. This pass consumes only frozen scientific result artifacts, regenerates manuscript tables and result narratives, constructs the cited reference list, applies source and post-build quality gates, compiles the main manuscript and Supporting Information, and packages the submission bundle. It does not refit models, regenerate partitions, or recompute inferential statistics. Exact script paths, source files, and the validated Git commit are recorded in the machine-readable \texttt{BUILD\_MANIFEST.json} supplied with the submission bundle.
"""
    SI_FINAL.write_text(prefix + repro, encoding="utf-8")
    print("FINAL SUPPORTING-INFORMATION SOURCE: PASS")


def source_gate(gate) -> None:
    words = gate.abstract_word_count()
    keywords = gate.keyword_count()
    running = gate.running_title_length()
    figures, tables = gate.count_environments()
    banned = gate.manuscript_language_gate()
    si_banned = gate.supplementary_language_gate()
    print(f"Abstract words: {words}")
    print(f"Keywords: {keywords}")
    print(f"Running-title characters: {running}")
    print(f"Main figures: {figures}")
    print(f"Main tables: {tables}")
    if words > 250 or not 3 <= keywords <= 5 or running > 70 or figures > 7 or tables > 4:
        raise AssertionError("Journal manuscript count/length gate failed")
    if banned or si_banned:
        raise AssertionError(f"Outdated manuscript wording remains: main={banned}; SI={si_banned}")

    main = (LATEX / "main.tex").read_text(encoding="utf-8")
    supplementary = (LATEX / "supplementary.tex").read_text(encoding="utf-8")
    si_final = SI_FINAL.read_text(encoding="utf-8")
    mean_only = MEAN_ONLY_TEX.read_text(encoding="utf-8")
    results = (LATEX / "sections" / "results_chemometrics.tex").read_text(encoding="utf-8")
    refs = FINAL_REFS.read_text(encoding="utf-8")
    required = ["\\usepackage[margin=3cm]{geometry}", "\\doublespacing", "\\usepackage{xurl}",
                "\\renewcommand{\\thetable}{\\Roman{table}}", "\\input{references_joc_submission}"]
    missing = [token for token in required if token not in main]
    if missing:
        raise AssertionError("Submission format tokens missing: " + "; ".join(missing))
    si_required = ["\\usepackage{xurl}", "\\input{appendix_chemometrics_submission}"]
    si_missing = [token for token in si_required if token not in supplementary]
    if si_missing:
        raise AssertionError("Supporting Information formatting tokens missing: " + "; ".join(si_missing))
    if "publication-final pass is implemented by script 32" not in si_final or "rendered exactly once by script 31" not in si_final:
        raise AssertionError("Authoritative publication-final workflow is not documented in Supporting Information")
    if "22_build_paper1_q1_final_v3.py" in si_final or "The final submission build uses" in si_final:
        raise AssertionError("Outdated final-build wording remains in publication Supporting Information")
    if "\\label{tab:si-acyclic}\n\\resizebox{\\textwidth}{!}{%" not in si_final:
        raise AssertionError("Acyclic-sensitivity SI table is not width-constrained")
    if "\\resizebox{\\textwidth}{!}{%" not in mean_only:
        raise AssertionError("Mean-only SI table is not width-constrained")
    if results.count("width=0.895\\textwidth") != 6:
        raise AssertionError("All six main figures must retain the intended manuscript placement width")

    bib_order = re.findall(r"\\bibitem\{([^}]+)\}", refs)
    cited = citation_order()
    if bib_order != cited:
        raise AssertionError(f"Reference order does not match first citation order: bib={bib_order}; cited={cited}")
    if len(bib_order) != 19:
        raise AssertionError(f"Expected 19 cited references; found {len(bib_order)}")
    checks = [
        "doi:10.1002/cem.1310", "doi:10.1186/s13321-023-00787-9",
        "doi:10.1021/acs.jcim.5c02465", "doi:10.1021/acs.jcim.6c00514",
        "doi:10.5281/zenodo.21291217", "\\textit{J. Chemom.}",
        "\\textit{J. Chem. Inf. Model.}",
    ]
    absent = [token for token in checks if token not in refs]
    if absent:
        raise AssertionError("Audited reference metadata missing: " + "; ".join(absent))
    print("PUBLICATION SOURCE + REFERENCE GATE: PASS")


def overfull_box_gate() -> None:
    build = LATEX / "build_q1_final_v3"
    failures: list[str] = []
    pattern = re.compile(r"Overfull \\[hv]box \(([0-9.]+)pt too (?:wide|high)\)")
    for log_name in ["main.log", "supplementary.log"]:
        path = build / log_name
        if not path.exists():
            continue
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        material: list[tuple[float, str]] = []
        for i, line in enumerate(lines):
            m = pattern.search(line)
            if not m:
                continue
            amount = float(m.group(1))
            if amount <= 0.5:
                continue
            context = " | ".join(lines[i:i + 3])[:350]
            material.append((amount, context))
        if material:
            print(f"MATERIAL OVERFULL BOXES IN {log_name}")
            for amount, context in material:
                print(f"  {amount:.3f} pt :: {context}")
            failures.append(f"{log_name}: max overfull box {max(x[0] for x in material):.3f} pt; count={len(material)}")
    if failures:
        raise AssertionError("Material LaTeX overfull boxes remain: " + "; ".join(failures))
    print("LATEX OVERFULL-BOX GATE: PASS")


def main() -> None:
    r = load(LEGACY, "paper1_submission_library")
    gate = load(LEGACY_GATE, "paper1_gate_library")
    print("=" * 78)
    print("PAPER 1 PUBLICATION-FINAL BUILD — FROZEN SCIENCE / ONE ARTWORK PASS")
    print("=" * 78)
    r.verify_frozen_inputs()
    r.reset_outputs()
    r.refresh_manuscript_assets()
    build_si_submission_source()
    r.run([sys.executable, str(REFERENCES)], cwd=ROOT)
    r.run([sys.executable, str(ARTWORK)], cwd=ROOT)
    source_gate(gate)
    r.compile_latex()
    gate.post_build_gate()
    overfull_box_gate()
    r.package_submission()

    outsrc = r.BUNDLE / "latex_source"
    shutil.copy2(FINAL_REFS, outsrc / FINAL_REFS.name)
    shutil.copy2(SI_FINAL, outsrc / SI_FINAL.name)
    manifest_path = r.BUNDLE / "BUILD_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["final_artwork_builder"] = "paper1_leakage_benchmark/scripts/31_publication_artwork_final_v3.py"
    manifest["reference_file"] = "paper1_latex/references_joc_submission.tex"
    manifest["reference_count"] = 19
    manifest["si_source_file"] = "paper1_latex/appendix_chemometrics_submission.tex"
    manifest["pipeline"] = "frozen science -> final SI source -> generated cited references -> fixed-size artwork -> source gate -> LaTeX -> post-build/overfull gates -> bundle"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print("PAPER 1 PUBLICATION-FINAL BUILD: PASS")
    print("Main PDF:", r.BUNDLE / "main.pdf")
    print("SI PDF  :", r.BUNDLE / "supplementary.pdf")
    print("Manifest:", manifest_path)
    print("=" * 78)


if __name__ == "__main__":
    main()
