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
    results = (LATEX / "sections" / "results_chemometrics.tex").read_text(encoding="utf-8")
    refs = FINAL_REFS.read_text(encoding="utf-8")
    required = ["\\usepackage[margin=3cm]{geometry}", "\\doublespacing",
                "\\renewcommand{\\thetable}{\\Roman{table}}", "\\input{references_joc_submission}"]
    missing = [token for token in required if token not in main]
    if missing:
        raise AssertionError("Submission format tokens missing: " + "; ".join(missing))
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


def main() -> None:
    r = load(LEGACY, "paper1_submission_library")
    gate = load(LEGACY_GATE, "paper1_gate_library")
    print("=" * 78)
    print("PAPER 1 PUBLICATION-FINAL BUILD — FROZEN SCIENCE / ONE ARTWORK PASS")
    print("=" * 78)
    r.verify_frozen_inputs()
    r.reset_outputs()
    r.refresh_manuscript_assets()
    r.run([sys.executable, str(REFERENCES)], cwd=ROOT)
    r.run([sys.executable, str(ARTWORK)], cwd=ROOT)
    source_gate(gate)
    r.compile_latex()
    gate.post_build_gate()
    r.package_submission()

    outsrc = r.BUNDLE / "latex_source"
    shutil.copy2(FINAL_REFS, outsrc / FINAL_REFS.name)
    manifest_path = r.BUNDLE / "BUILD_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["final_artwork_builder"] = "paper1_leakage_benchmark/scripts/31_publication_artwork_final_v3.py"
    manifest["reference_file"] = "paper1_latex/references_joc_submission.tex"
    manifest["reference_count"] = 19
    manifest["pipeline"] = "frozen science -> generated cited references -> fixed-size artwork -> source gate -> LaTeX -> post-build gate -> bundle"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print("PAPER 1 PUBLICATION-FINAL BUILD: PASS")
    print("Main PDF:", r.BUNDLE / "main.pdf")
    print("SI PDF  :", r.BUNDLE / "supplementary.pdf")
    print("Manifest:", manifest_path)
    print("=" * 78)


if __name__ == "__main__":
    main()
