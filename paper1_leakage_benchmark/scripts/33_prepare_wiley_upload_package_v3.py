from __future__ import annotations

import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "paper1_submission_q1_final_v3"
LATEX_SOURCE = BUNDLE / "latex_source"
OUT = ROOT / "paper1_wiley_upload_v1_2"

MAIN_FIGURES = [
    "figure1_audit_framework_v3",
    "figure2_primary_effects_v3",
    "figure3_acyclic_sensitivity_v3",
    "figure4_dominant_fragment_sensitivity_v3",
    "figure5_candidate_budget_audit_v3",
    "figure6_collateral_diagnostics_v3",
]


def require(path: Path) -> Path:
    if not path.exists() or (path.is_file() and path.stat().st_size == 0):
        raise FileNotFoundError(path)
    return path


def current_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()


def verify_bundle(expected_commit: str) -> dict:
    manifest_path = require(BUNDLE / "BUILD_MANIFEST.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    commit = str(manifest.get("commit", "")).strip()
    if commit != expected_commit:
        raise AssertionError(
            f"Submission bundle commit mismatch: found {commit or '<missing>'}, expected current HEAD {expected_commit}. "
            "Run script 32 after pulling the final branch, then run script 33 again."
        )
    require(BUNDLE / "main.pdf")
    require(BUNDLE / "supplementary.pdf")
    require(LATEX_SOURCE / "main.tex")
    for stem in MAIN_FIGURES:
        require(BUNDLE / "figures" / f"{stem}.pdf")
        require(BUNDLE / "figures" / f"{stem}.tiff")
    print(f"VERIFIED FINAL BUNDLE: {commit}")
    return manifest


def resolve_input(name: str) -> Path:
    rel = Path(name)
    if rel.suffix == "":
        rel = rel.with_suffix(".tex")
    path = LATEX_SOURCE / rel
    return require(path)


def inline_tex(path: Path, stack: tuple[Path, ...] = ()) -> str:
    resolved = path.resolve()
    if resolved in stack:
        chain = " -> ".join(p.name for p in (*stack, resolved))
        raise RuntimeError(f"Circular LaTeX input detected: {chain}")
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(r"\\input\{([^}]+)\}")

    def repl(match: re.Match[str]) -> str:
        child = resolve_input(match.group(1))
        body = inline_tex(child, (*stack, resolved))
        return f"\n% BEGIN INLINED: {match.group(1)}\n{body}\n% END INLINED: {match.group(1)}\n"

    return pattern.sub(repl, text)


def write_standalone_main() -> Path:
    source = inline_tex(require(LATEX_SOURCE / "main.tex"))
    old_graphicspath = (
        r"\graphicspath{{../paper1_leakage_benchmark/figures/}"
        r"{../paper1_leakage_benchmark/results/figures/}}"
    )
    if old_graphicspath not in source:
        raise AssertionError("Expected manuscript graphicspath was not found")
    source = source.replace(old_graphicspath, r"\graphicspath{{figures/}}", 1)
    path = OUT / "01_Main_Document_LaTeX.tex"
    path.write_text(source, encoding="utf-8")
    print("STANDALONE MAIN TEX: PASS")
    return path


def zip_tree(source_dir: Path, output_zip: Path) -> None:
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(source_dir).as_posix())


def compile_tex(tex_path: Path, cwd: Path, output_name: str) -> Path:
    latexmk = shutil.which("latexmk")
    pdflatex = shutil.which("pdflatex")
    build = cwd / "_compile_check"
    if build.exists():
        shutil.rmtree(build)
    build.mkdir()
    if latexmk:
        cmd = [
            latexmk,
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            f"-outdir={build.name}",
            tex_path.name,
        ]
        subprocess.run(cmd, cwd=str(cwd), check=True)
    elif pdflatex:
        for _ in range(2):
            subprocess.run(
                [pdflatex, "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={build.name}", tex_path.name],
                cwd=str(cwd),
                check=True,
            )
    else:
        raise RuntimeError("Neither latexmk nor pdflatex is available")
    compiled = require(build / f"{tex_path.stem}.pdf")
    destination = OUT / output_name
    shutil.copy2(compiled, destination)
    shutil.rmtree(build)
    return destination


def build_main_archive(standalone_tex: Path, expected_commit: str) -> None:
    stage = OUT / "_main_latex_bundle"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir()
    shutil.copy2(standalone_tex, stage / "main.tex")
    shutil.copy2(BUNDLE / "main.pdf", stage / "Main_Manuscript_Peer_Review.pdf")
    figures = stage / "figures"
    figures.mkdir()
    for stem in MAIN_FIGURES:
        shutil.copy2(BUNDLE / "figures" / f"{stem}.pdf", figures / f"{stem}.pdf")
    (stage / "README.txt").write_text(
        "Journal of Chemometrics main-manuscript LaTeX bundle\n"
        f"Frozen manuscript commit: {expected_commit}\n"
        "Compile: latexmk -pdf main.tex\n"
        "The compiled peer-review PDF is included as Main_Manuscript_Peer_Review.pdf.\n",
        encoding="utf-8",
    )

    # Prove that the staged archive compiles independently before creating it.
    compile_tex(stage / "main.tex", stage, "02_Main_Manuscript_Peer_Review_Recompiled.pdf")
    zip_tree(stage, OUT / "01_Main_Manuscript_LaTeX_Bundle.zip")

    support_stage = OUT / "_latex_support"
    if support_stage.exists():
        shutil.rmtree(support_stage)
    support_stage.mkdir()
    support_figures = support_stage / "figures"
    support_figures.mkdir()
    for stem in MAIN_FIGURES:
        shutil.copy2(BUNDLE / "figures" / f"{stem}.pdf", support_figures / f"{stem}.pdf")
    (support_stage / "README.txt").write_text(
        "Supporting files for 01_Main_Document_LaTeX.tex.\n"
        "These six PDF figures are required to compile the LaTeX main document.\n",
        encoding="utf-8",
    )
    zip_tree(support_stage, OUT / "05_LaTeX_Supplementary_Files.zip")
    shutil.rmtree(stage)
    shutil.rmtree(support_stage)
    print("SELF-CONTAINED LATEX ARCHIVE: PASS")


def build_cover_letter() -> None:
    cover = r"""\documentclass[11pt]{letter}
\usepackage[margin=2.6cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\signature{Siyuan Tong\\Corresponding Author}
\address{Department of Artificial Intelligence\\Faculty of Computer Science and Information Technology\\University of Malaya\\50603 Kuala Lumpur, Malaysia\\25064241@siswa.um.edu.my}
\date{15 August 2026}
\begin{document}
\begin{letter}{Editor\\Journal of Chemometrics}
\opening{Dear Editor,}

Please consider our Research Article entitled ``Dissecting Molecular Benchmark Construction: A Paired Chemometric Audit of Target-Mean Balance, Scaffold Semantics, and Molecular Representation'' for publication in the \textit{Journal of Chemometrics}.

The manuscript treats molecular benchmark construction as a chemometric measurement problem. Within fixed target-blind scaffold candidate pools, we compare exact-size paired partitions selected with and without a target-mean criterion, freeze search budgets and partition manifests before model fitting, and use partition-level inference. Mechanistic and protocol-sensitivity analyses show that the apparent regression benefit contains a response-geometry component visible to a mean-only predictor and depends strongly on acyclic-scaffold semantics, while a deterministic molecular-record perturbation changes most classification point-estimate directions without changing the corrected inferential conclusion. We believe this combination of experimental-design discipline, validation analysis, and reproducible molecular-data auditing is directly relevant to the journal's readership.

The work is original, has not been published previously, and is not under consideration elsewhere. All authors have reviewed and approved the manuscript and agree with its submission. The authors declare no competing interests. The datasets are public, and the frozen workflow, code, manifests, result tables, and reproducibility materials are openly available. Use of OpenAI ChatGPT for language, organization, debugging, LaTeX, figure-layout, and reproducibility-material assistance is disclosed in the Materials and Methods; all scientific analyses and outputs were executed and verified by the authors.

Thank you for your consideration.

\closing{Sincerely,}
\end{letter}
\end{document}
"""
    tex = OUT / "04_Cover_Letter.tex"
    tex.write_text(cover, encoding="utf-8")
    compile_tex(tex, OUT, "04_Cover_Letter.pdf")
    print("COVER LETTER: PASS")


def copy_submission_files() -> None:
    shutil.copy2(BUNDLE / "main.pdf", OUT / "02_Main_Manuscript_Peer_Review.pdf")
    shutil.copy2(BUNDLE / "supplementary.pdf", OUT / "03_Supporting_Information.pdf")
    figures = OUT / "Figures"
    figures.mkdir()
    for index, stem in enumerate(MAIN_FIGURES, start=1):
        shutil.copy2(BUNDLE / "figures" / f"{stem}.tiff", figures / f"Figure_{index}.tiff")


def write_upload_map(expected_commit: str) -> None:
    text = f"""JOURNAL OF CHEMOMETRICS — WILEY UPLOAD MAP

Verified source commit: {expected_commit}

Main Document - LaTeX .tex File -> 01_Main_Document_LaTeX.tex
Main Document - LaTeX PDF -> 02_Main_Manuscript_Peer_Review.pdf
LaTeX Supplementary File -> 05_LaTeX_Supplementary_Files.zip
Supplementary Material for Review -> 03_Supporting_Information.pdf
Figure -> Figures/Figure_1.tiff through Figures/Figure_6.tiff

Do not upload PNG versions, result CSV files, BUILD_MANIFEST.json, build directories, or supplementary figures separately. They are already contained in the final Supporting Information PDF or are not publication files.
"""
    (OUT / "00_READ_ME_FIRST_UPLOAD_MAP.txt").write_text(text, encoding="utf-8")


def main() -> None:
    expected_commit = current_head()
    verify_bundle(expected_commit)
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    standalone = write_standalone_main()
    copy_submission_files()
    build_main_archive(standalone, expected_commit)
    build_cover_letter()
    write_upload_map(expected_commit)
    (OUT / "FINAL_SOURCE_COMMIT.txt").write_text(expected_commit + "\n", encoding="utf-8")
    print("\nWILEY UPLOAD PACKAGE: PASS")
    print("Source commit:", expected_commit)
    print("Folder:", OUT)
    for path in sorted(OUT.iterdir()):
        if path.is_file():
            print(f"  {path.name} ({path.stat().st_size:,} bytes)")
        else:
            print(f"  {path.name}/")


if __name__ == "__main__":
    main()
