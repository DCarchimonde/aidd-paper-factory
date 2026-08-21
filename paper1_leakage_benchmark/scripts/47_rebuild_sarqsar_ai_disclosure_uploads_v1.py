from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper1_leakage_benchmark"
SOURCE = ROOT / "paper1_sarqsar_submission_source_v1"
SECTION = SOURCE / "sections" / "06_end.tex"
OUT = ROOT / "paper1_sarqsar_submission_v1"
OUT_ZIP = ROOT / "paper1_sarqsar_submission_v1.zip"
NULL_ROOT = PAPER / "results" / "sarqsar_metric_coupling_v1"
NULL_TABLES = NULL_ROOT / "tables"
EMP_TABLES = PAPER / "results" / "tables"
BUILDER = PAPER / "scripts" / "44_build_paper1_sarqsar_submission_v1.py"
EXPECTED_BRANCH = "paper1-sarqsar-metric-coupling-2026"
AI_MARKER = "OpenAI ChatGPT (GPT-5.6 Pro, web application; accessed 22 August 2026)"
ANON_TOKENS = [
    "siyuan tong",
    "yuechen wang",
    "25064241",
    "d25091100346",
    "university of malaya",
    "city university of macau",
    "0009-0004-4450-083x",
    "dcarchimonde",
    "aidd-paper-factory",
]


def require(path: Path) -> Path:
    if not path.exists() or (path.is_file() and path.stat().st_size == 0):
        raise FileNotFoundError(path)
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=str(ROOT), text=True).strip()


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def zip_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        destination.unlink()
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source).as_posix())


def pdf_text(path: Path) -> str:
    pdftotext = shutil.which("pdftotext")
    if pdftotext:
        target = path.with_suffix(".ai_disclosure_check.txt")
        subprocess.run([pdftotext, str(path), str(target)], check=True)
        text = target.read_text(encoding="utf-8", errors="replace")
        target.unlink(missing_ok=True)
        return text
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as exc:
        raise RuntimeError("pdftotext or pypdf is required for PDF verification") from exc
    return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)


def verify_source_and_pdfs() -> None:
    source_text = require(SECTION).read_text(encoding="utf-8")
    if source_text.count(AI_MARKER) != 1:
        raise AssertionError("Updated AI disclosure marker is absent or duplicated in 06_end.tex")
    if "OpenAI ChatGPT was used" in source_text:
        raise AssertionError("The superseded unversioned AI disclosure remains in 06_end.tex")

    for name in ["01_Manuscript_with_author_details.pdf", "02_Manuscript_anonymous.pdf"]:
        text = pdf_text(require(OUT / name))
        if "GPT-5.6 Pro" not in text or "22 August 2026" not in text:
            raise AssertionError(f"Updated AI disclosure is missing from {name}")
    print("AI DISCLOSURE PDF GATE: PASS", flush=True)


def build_anonymous_source_zip() -> Path:
    source_root = require(OUT / "LaTeX_Source")
    stage = ROOT / "paper1_sarqsar_anonymous_source_upload_v1"
    reset_dir(stage)
    files = [
        "main_anonymous.tex",
        "main_body.tex",
        "supporting_information_anonymous.tex",
        "references.tex",
    ]
    for name in files:
        shutil.copy2(require(source_root / name), stage / name)
    for directory in ["generated", "figures", "sections"]:
        shutil.copytree(require(source_root / directory), stage / directory)

    joined = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore").lower()
        for path in stage.rglob("*")
        if path.is_file() and path.suffix.lower() in {".tex", ".txt", ".md", ".csv", ".json"}
    )
    hits = [token for token in ANON_TOKENS if token in joined]
    if hits:
        raise AssertionError(f"Anonymous LaTeX source contains identity tokens: {hits}")

    target = OUT / "06_LaTeX_Source_Anonymous.zip"
    zip_tree(stage, target)
    shutil.rmtree(stage)
    return target


def sanitized_run_manifest() -> dict:
    manifest = json.loads(require(NULL_ROOT / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    allowed = [
        "status",
        "protocol_sha256",
        "config_sha256",
        "protocol_version",
        "n_permutations",
        "partition_seeds",
        "completed_at_unix",
    ]
    return {key: manifest.get(key) for key in allowed if key in manifest}


def build_reproducibility_archive(source_zip: Path) -> Path:
    stage = ROOT / "paper1_sarqsar_anonymized_reproducibility_v1"
    reset_dir(stage)
    (stage / "README.txt").write_text(
        textwrap.dedent(
            """
            Anonymized reproducibility archive for double-anonymous review
            =================================================================
            This archive contains the frozen molecular-null protocol/configuration,
            anonymized run metadata, permutation-level outputs, empirical summary
            tables, matched empirical-null bridge, publication figures, and anonymous
            manuscript/source materials. It contains no author, institution, email,
            ORCID, public-repository account, or related-manuscript identifiers.
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    shutil.copy2(require(PAPER / "SARQSAR_METRIC_COUPLING_PROTOCOL_V1.md"), stage / "protocol_v1.md")
    shutil.copy2(require(PAPER / "SARQSAR_METRIC_COUPLING_CONFIG_V1.json"), stage / "config_v1.json")
    (stage / "run_manifest_anonymized.json").write_text(
        json.dumps(sanitized_run_manifest(), indent=2), encoding="utf-8"
    )

    tables = stage / "tables"
    tables.mkdir()
    null_names = [
        "null_simulation_permutation_level_effects.csv",
        "null_metric_effect_summary.csv",
        "null_simulation_quality_gate_summary.csv",
        "qsar_benchmark_minimum_reporting_checklist.csv",
    ]
    for name in null_names:
        shutil.copy2(require(NULL_TABLES / name), tables / name)
    empirical_names = [
        "primary_inference_summary_v3.csv",
        "acyclic_singleton_sensitivity_v3.csv",
        "q1_mean_only_regression_summary_v3.csv",
    ]
    for name in empirical_names:
        shutil.copy2(require(EMP_TABLES / name), tables / name)
    shutil.copy2(require(OUT / "Tables" / "empirical_null_bridge_v1.csv"), tables / "empirical_null_bridge_v1.csv")

    figures = stage / "figures"
    figures.mkdir()
    for number in range(1, 8):
        shutil.copy2(require(OUT / "Figures" / f"Figure_{number}.pdf"), figures / f"Figure_{number}.pdf")

    manuscript = stage / "anonymous_manuscript"
    manuscript.mkdir()
    shutil.copy2(require(OUT / "02_Manuscript_anonymous.pdf"), manuscript / "02_Manuscript_anonymous.pdf")
    shutil.copy2(
        require(OUT / "04_Supporting_Information_anonymous.pdf"),
        manuscript / "04_Supporting_Information_anonymous.pdf",
    )
    shutil.copy2(source_zip, manuscript / source_zip.name)

    textual = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore").lower()
        for path in stage.rglob("*")
        if path.is_file() and path.suffix.lower() in {".txt", ".md", ".csv", ".json", ".tex"}
    )
    hits = [token for token in ANON_TOKENS if token in textual]
    if hits:
        raise AssertionError(f"Reproducibility archive contains identity tokens: {hits}")

    manifest = {}
    for path in sorted(stage.rglob("*")):
        if path.is_file():
            manifest[path.relative_to(stage).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    (stage / "ARCHIVE_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    target = OUT / "07_Anonymized_Reproducibility_Archive.zip"
    zip_tree(stage, target)
    shutil.rmtree(stage)
    return target


def update_upload_map(source_zip: Path, archive_zip: Path) -> None:
    map_path = OUT / "00_UPLOAD_MAP.txt"
    map_path.write_text(
        textwrap.dedent(
            f"""
            SAR and QSAR in Environmental Research upload map
            =================================================
            Manuscript with author details: 01_Manuscript_with_author_details.pdf
            Anonymous manuscript:           02_Manuscript_anonymous.pdf
            Title page:                     03_Title_Page.pdf
            Anonymous Supporting Info:     04_Supporting_Information_anonymous.pdf
            Cover letter:                   05_Cover_Letter.pdf
            Anonymous LaTeX source:         {source_zip.name}
            Reproducibility archive:        {archive_zip.name}
            Separate figures:              Figures/Figure_1.tiff ... Figure_7.tiff

            Files changed by the AI-version disclosure update:
            - 01_Manuscript_with_author_details.pdf
            - 02_Manuscript_anonymous.pdf
            - {source_zip.name}
            - {archive_zip.name}
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )


def refresh_manifests_and_outer_zip() -> None:
    manifest_path = OUT / "FINAL_BUILD_MANIFEST.json"
    previous = {}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = {}
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path != manifest_path:
            files[path.relative_to(OUT).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    previous["status"] = "complete"
    previous["ai_disclosure"] = AI_MARKER
    previous["ai_disclosure_commit"] = git("rev-parse", "HEAD")
    previous["files"] = files
    manifest_path.write_text(json.dumps(previous, indent=2), encoding="utf-8")

    if OUT_ZIP.exists():
        OUT_ZIP.unlink()
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(OUT).as_posix())


def main() -> None:
    branch = git("branch", "--show-current")
    if branch != EXPECTED_BRANCH:
        raise AssertionError(f"Wrong branch: {branch!r}; expected {EXPECTED_BRANCH!r}")
    require(BUILDER)
    require(SECTION)
    if AI_MARKER not in SECTION.read_text(encoding="utf-8"):
        raise AssertionError("Pull the disclosure-update commit before running this script")

    subprocess.run([sys.executable, "-u", str(BUILDER)], cwd=str(ROOT), check=True)
    verify_source_and_pdfs()
    source_zip = build_anonymous_source_zip()
    archive_zip = build_reproducibility_archive(source_zip)
    update_upload_map(source_zip, archive_zip)
    refresh_manifests_and_outer_zip()

    print("\n" + "=" * 92)
    print("SAR/QSAR AI DISCLOSURE UPDATE: PASS")
    print("Replace exactly these four uploaded files:")
    print("  01_Manuscript_with_author_details.pdf")
    print("  02_Manuscript_anonymous.pdf")
    print(f"  {source_zip.name}")
    print(f"  {archive_zip.name}")
    print("Do not replace Title Page, Supporting Information, Cover Letter, or Figures.")
    print("Output folder:", OUT)
    print("Outer backup ZIP:", OUT_ZIP)
    print("=" * 92)


if __name__ == "__main__":
    main()
