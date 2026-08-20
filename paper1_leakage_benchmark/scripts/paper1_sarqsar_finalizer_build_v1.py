from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import textwrap
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from paper1_sarqsar_finalizer_core_v1 import *


def write_wrappers(primary: pd.DataFrame, singleton: pd.DataFrame, summary: pd.DataFrame, bridge: pd.DataFrame) -> None:
    shutil.copy2(require(SOURCE / "main_body.tex"), LATEX / "main_body.tex")
    shutil.copy2(require(SOURCE / "references.tex"), LATEX / "references.tex")
    shutil.copytree(require(SOURCE / "sections"), LATEX / "sections")

    preamble = r"""\documentclass[11pt]{article}
\usepackage[margin=3cm]{geometry}
\usepackage{graphicx,booktabs,array,longtable,adjustbox,amsmath,amssymb,microtype,setspace,lineno,xurl,caption,float,placeins}
\usepackage[numbers,sort&compress]{natbib}
\usepackage[colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue]{hyperref}
\doublespacing
\linenumbers
\setlength{\emergencystretch}{2em}
\captionsetup{font=small,labelfont=bf,labelsep=period,skip=6pt}
\newif\ifanonymous
"""
    anonymous = preamble + rf"""\anonymoustrue
\title{{{TITLE}}}\author{{}}\date{{}}
\begin{{document}}\maketitle\input{{main_body.tex}}\end{{document}}
"""
    authors = preamble + rf"""\anonymousfalse
\title{{{TITLE}}}
\author{{\begin{{minipage}}{{0.94\textwidth}}\centering
Siyuan Tong$^{{1,*}}$ and Yuechen Wang$^{{2}}$\\[0.4em]
\small $^1$Department of Artificial Intelligence, Faculty of Computer Science and Information Technology, University of Malaya, 50603 Kuala Lumpur, Malaysia\\
\small $^2$Faculty of Data Science, City University of Macau, Macau SAR, China\\[0.25em]
\small $^*$Corresponding author: \texttt{{25064241@siswa.um.edu.my}}
\end{{minipage}}}}\date{{}}
\begin{{document}}\maketitle\input{{main_body.tex}}\end{{document}}
"""
    (LATEX / "main_anonymous.tex").write_text(anonymous, encoding="utf-8")
    (LATEX / "main_with_authors.tex").write_text(authors, encoding="utf-8")

    si = preamble.replace("\\linenumbers\n", "") + rf"""\anonymoustrue
\title{{Supporting Information\\[0.4em]\large {TITLE}}}
\author{{Double-anonymous review version}}\date{{}}
\begin{{document}}\maketitle
\section{{Molecular endpoint-permutation integrity}}
The final run contained 200 endpoint permutations per dataset and 20 partition seeds. Partition-seed effects were averaged within endpoint permutation. Exact-size pairing, non-worsening target gap, constant-score AUC invariance, and MSE decomposition were enforced as hard gates.
\begin{{table}}[!htbp]\centering\caption{{Maximum-budget regression null effects. Brackets are central 2.5th--97.5th percentiles.}}\resizebox{{\textwidth}}{{!}}{{\input{{generated/null_regression_table.tex}}}}\end{{table}}
\begin{{table}}[!htbp]\centering\caption{{Classification null effects at 300 requested draws.}}\input{{generated/null_classification_table.tex}}\end{{table}}
\clearpage
\section{{Exploratory empirical--null bridge}}
\begin{{table}}[!htbp]\centering\caption{{Observed mean-only RMSE effects compared with matched null distributions.}}\resizebox{{\textwidth}}{{!}}{{\input{{generated/bridge_table.tex}}}}\end{{table}}
\clearpage
\section{{Frozen empirical paired inference}}
\begin{{table}}[!htbp]\centering\scriptsize\caption{{Complete primary empirical inference summary.}}\resizebox{{\textwidth}}{{!}}{{\input{{generated/primary_table.tex}}}}\end{{table}}
\begin{{table}}[!htbp]\centering\scriptsize\caption{{Acyclic singleton regression sensitivity.}}\resizebox{{\textwidth}}{{!}}{{\input{{generated/singleton_table.tex}}}}\end{{table}}
\end{{document}}
"""
    (LATEX / "supporting_information_anonymous.tex").write_text(si, encoding="utf-8")

    title_page = rf"""\documentclass[11pt]{{article}}\usepackage[margin=3cm]{{geometry}}\begin{{document}}
\begin{{center}}{{\Large\bfseries {TITLE}\par}}\vspace{{1.5em}}Siyuan Tong$^{{1,*}}$ and Yuechen Wang$^{{2}}$\end{{center}}
\noindent $^1$Department of Artificial Intelligence, Faculty of Computer Science and Information Technology, University of Malaya, 50603 Kuala Lumpur, Malaysia

\noindent $^2$Faculty of Data Science, City University of Macau, Macau SAR, China

\noindent $^*$Corresponding author: 25064241@siswa.um.edu.my

\noindent ORCID: 0009-0004-4450-083X

\section*{{Running title}}Metric coupling in QSAR validation
\section*{{Article type}}Research Article
\section*{{Funding}}The authors received no specific funding for this work.
\section*{{Competing interests}}The authors declare no competing interests.
\section*{{Related manuscript disclosure}}A related manuscript under consideration elsewhere addresses a distinct conformal uncertainty intervention under molecular distribution shift. The present manuscript studies benchmark split construction, endpoint permutation, metric coupling, scaffold semantics, and molecular-record representation. The hypotheses, analyses, figures, tables, and conclusions are distinct.
\end{{document}}
"""
    (LATEX / "title_page.tex").write_text(title_page, encoding="utf-8")

    cover = rf"""\documentclass[11pt]{{letter}}\usepackage[margin=2.7cm]{{geometry}}
\signature{{Siyuan Tong\\Corresponding Author\\25064241@siswa.um.edu.my}}
\address{{Department of Artificial Intelligence\\Faculty of Computer Science and Information Technology\\University of Malaya\\50603 Kuala Lumpur, Malaysia}}
\begin{{document}}\begin{{letter}}{{Editor\\SAR and QSAR in Environmental Research}}\opening{{Dear Editor,}}
Please consider our Research Article entitled ``{TITLE}'' for publication in \textit{{SAR and QSAR in Environmental Research}}.

The manuscript addresses a validation problem central to QSAR methodology: when endpoint information is used to select among candidate scaffold-disjoint test sets, the split objective can align mathematically with downstream metrics. We combine a pre-specified molecular endpoint-permutation experiment (six public datasets, 200 permutations per dataset, 20 partition seeds, and nested search budgets) with an exact-size paired empirical audit of learned models, a response-only control, scaffold-semantic sensitivity, and molecular-record representation sensitivity.

The null experiment distinguishes deterministic coupling of aligned metric components from contingent changes in headline metrics. An explicitly exploratory matched-budget bridge connects the mechanism to the observed regression effects without redefining the original empirical hypothesis family. The manuscript concludes with a practical reporting checklist for response-aware QSAR benchmark construction.

The work is original, is not under consideration elsewhere, and has been approved by both authors. We disclose related work on a distinct conformal uncertainty intervention in the title-page materials. The authors declare no competing interests and no specific funding. Public datasets are cited, and an anonymized reproducibility archive will be supplied for double-anonymous review.
\closing{{Sincerely,}}\end{{letter}}\end{{document}}
"""
    (LATEX / "cover_letter.tex").write_text(cover, encoding="utf-8")


def reference_gate() -> None:
    refs = require(SOURCE / "references.tex").read_text(encoding="utf-8")
    body = require(SOURCE / "main_body.tex").read_text(encoding="utf-8")
    body += "\n" + "\n".join(path.read_text(encoding="utf-8") for path in sorted(require(SOURCE / "sections").glob("*.tex")))
    keys = re.findall(r"\\bibitem\{([^}]+)\}", refs)
    cited: set[str] = set()
    for match in re.finditer(r"\\cite[pt]?\{([^}]+)\}", body):
        cited.update(part.strip() for part in match.group(1).split(","))
    missing = cited.difference(keys)
    if missing or len(keys) < 20 or len(keys) != len(set(keys)):
        raise AssertionError(f"Reference gate failed: missing={sorted(missing)}, entries={len(keys)}, unique={len(set(keys))}")
    print(f"REFERENCE GATE: PASS ({len(keys)} entries; all cited keys resolved)", flush=True)


def source_anonymity_gate() -> None:
    source_paths = [LATEX / "main_anonymous.tex", LATEX / "main_body.tex", LATEX / "supporting_information_anonymous.tex"]
    source_paths.extend(sorted((LATEX / "sections").glob("*.tex")))
    text = "\n".join(path.read_text(encoding="utf-8").lower() for path in source_paths)
    hits = [token for token in ANON_TOKENS if token in text]
    if hits:
        raise AssertionError(f"Double-anonymous source leak: {hits}")
    print("DOUBLE-ANONYMOUS SOURCE GATE: PASS", flush=True)


def compile_tex(name: str) -> Path:
    latexmk = shutil.which("latexmk")
    if not latexmk:
        raise RuntimeError("latexmk is required")
    run([latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", name], LATEX)
    pdf = require(LATEX / f"{Path(name).stem}.pdf")
    log = require(LATEX / f"{Path(name).stem}.log").read_text(encoding="utf-8", errors="replace")
    if "LaTeX Error" in log or "Undefined control sequence" in log:
        raise AssertionError(f"LaTeX error remains in {name}")
    if re.search(r"LaTeX Warning: (Reference|Citation).*undefined", log):
        raise AssertionError(f"Undefined reference or citation remains in {name}")
    return pdf


def pdf_text(path: Path) -> str:
    tool = shutil.which("pdftotext")
    if tool:
        out = path.with_suffix(".txt")
        run([tool, str(path), str(out)])
        return out.read_text(encoding="utf-8", errors="replace")
    from pypdf import PdfReader  # type: ignore
    return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)


def pdf_anonymity_gate(paths: list[Path]) -> None:
    failures = {}
    for path in paths:
        text = pdf_text(path).lower()
        hits = [token for token in ANON_TOKENS if token in text]
        if hits:
            failures[path.name] = hits
    if failures:
        raise AssertionError(f"Double-anonymous PDF leak: {failures}")
    print("DOUBLE-ANONYMOUS PDF GATE: PASS", flush=True)


def package(pdfs: dict[str, Path], summary: pd.DataFrame, quality: pd.DataFrame, bridge: pd.DataFrame) -> None:
    reset_dir(OUT)
    copies = {
        "01_Manuscript_with_author_details.pdf": pdfs["authors"],
        "02_Manuscript_anonymous.pdf": pdfs["anonymous"],
        "03_Title_Page.pdf": pdfs["title"],
        "04_Supporting_Information_anonymous.pdf": pdfs["si"],
        "05_Cover_Letter.pdf": pdfs["cover"],
    }
    for name, source in copies.items():
        shutil.copy2(source, OUT / name)
    figures = OUT / "Figures"
    figures.mkdir()
    for number in range(1, 8):
        for suffix in [".pdf", ".tiff"]:
            shutil.copy2(FIGS / f"Figure_{number}{suffix}", figures / f"Figure_{number}{suffix}")
    tables = OUT / "Tables"
    tables.mkdir()
    summary.to_csv(tables / "null_metric_effect_summary_v1.csv", index=False)
    quality.to_csv(tables / "null_simulation_quality_gate_summary_v1.csv", index=False)
    bridge.to_csv(tables / "empirical_null_bridge_v1.csv", index=False)
    for name in ["primary_inference_summary_v3.csv", "acyclic_singleton_sensitivity_v3.csv"]:
        shutil.copy2(BUILD / name, tables / name)

    src = OUT / "LaTeX_Source"
    src.mkdir()
    for name in [
        "main_anonymous.tex", "main_with_authors.tex", "main_body.tex",
        "supporting_information_anonymous.tex", "title_page.tex", "cover_letter.tex", "references.tex",
    ]:
        shutil.copy2(LATEX / name, src / name)
    shutil.copytree(GEN, src / "generated")
    shutil.copytree(FIGS, src / "figures")
    shutil.copytree(LATEX / "sections", src / "sections")

    (OUT / "00_UPLOAD_MAP.txt").write_text(textwrap.dedent("""
        SAR and QSAR in Environmental Research upload map
        =================================================
        Manuscript with author details: 01_Manuscript_with_author_details.pdf
        Anonymous manuscript:           02_Manuscript_anonymous.pdf
        Title page:                     03_Title_Page.pdf
        Anonymous Supporting Info:     04_Supporting_Information_anonymous.pdf
        Cover letter:                   05_Cover_Letter.pdf
        Separate figures:              Figures/Figure_1.tiff ... Figure_7.tiff
        LaTeX source:                   LaTeX_Source/
    """).strip() + "\n", encoding="utf-8")

    audit = textwrap.dedent(f"""
        # Science and submission audit
        - Endpoint permutations per dataset: 200
        - Partition seeds: 20
        - Permutation-level cells: {int(quality['permutation_level_rows'].iloc[0]):,}
        - Seed-level cells: {int(quality['raw_partition_seed_rows'].iloc[0]):,}
        - Maximum MSE decomposition residual: {float(quality['max_abs_mse_decomposition_residual'].iloc[0]):.3e}
        - Empirical primary decisions: 0/12 classification; 6/6 primary regression
        - Empirical-null bridge: exploratory and budget-matched
        - Double-anonymous source gate: PASS
        - Double-anonymous PDF gate: PASS
        - Reference key gate: PASS
        - Build commit: {git('rev-parse', 'HEAD')}
    """).strip() + "\n"
    (OUT / "SCIENCE_AND_SUBMISSION_AUDIT.md").write_text(audit, encoding="utf-8")

    manifest = {"status": "complete", "title": TITLE, "build_commit": git("rev-parse", "HEAD"), "null_science_commit": SCIENCE_COMMIT, "created_at_unix": time.time(), "files": {}}
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            manifest["files"][path.relative_to(OUT).as_posix()] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    (OUT / "FINAL_BUILD_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if OUT_ZIP.exists():
        OUT_ZIP.unlink()
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(OUT).as_posix())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Paper 1 SAR/QSAR submission package from frozen outputs.")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 92)
    print("PAPER 1 SAR/QSAR SUBMISSION-FINAL BUILD")
    print("=" * 92)
    branch = git("branch", "--show-current")
    if branch not in {EXPECTED_BRANCH, "unknown"}:
        raise AssertionError(f"Wrong branch: {branch}")
    summary, perm, quality, manifest = audit_null()
    ensure_empirical()
    mean_only, primary, singleton = audit_empirical()
    bridge = bridge_table(mean_only, perm)
    if args.preflight_only:
        print("SUBMISSION-FINAL PREFLIGHT ONLY: PASS")
        return

    reset_dir(BUILD)
    LATEX.mkdir(parents=True)
    GEN.mkdir(parents=True)
    FIGS.mkdir(parents=True)
    write_generated(summary, quality, bridge, primary, singleton)
    copy_figures()
    write_wrappers(primary, singleton, summary, bridge)
    source_anonymity_gate()
    reference_gate()
    pdfs = {
        "anonymous": compile_tex("main_anonymous.tex"),
        "authors": compile_tex("main_with_authors.tex"),
        "si": compile_tex("supporting_information_anonymous.tex"),
        "title": compile_tex("title_page.tex"),
        "cover": compile_tex("cover_letter.tex"),
    }
    pdf_anonymity_gate([pdfs["anonymous"], pdfs["si"]])
    package(pdfs, summary, quality, bridge)
    print("\n" + "=" * 92)
    print("PAPER 1 SAR/QSAR SUBMISSION-FINAL BUILD: PASS")
    print("Submission folder:", OUT)
    print("Submission ZIP   :", OUT_ZIP)
    print("=" * 92)


if __name__ == "__main__":
    main()
