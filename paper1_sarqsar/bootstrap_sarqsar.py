from __future__ import annotations

import argparse
import base64
import hashlib
import io
import py_compile
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "paper1_sarqsar"
PARTS = [PROJECT / "runtime_bundle" / f"runtime_bundle.part{index:02d}.b64" for index in range(6)]
EXPECTED_B64_SHA256 = "b344ae0373994f93eaab0b9f6c132715a81315ddf281ee81190cb92e879521c1"
EXPECTED_TAR_SHA256 = "769b4b48bda6ed4ffd2422927359ebe2f754cb0bdcd2cbe9c73e4de058da24ff"
EXPECTED_FILES = {
    "scripts/00_preflight_sarqsar.py": "339df27dee3ade21c6ea8d5ada2723ceb3e029a5a7f42f216f2ac9ce8230383f",
    "scripts/01_run_null_permutation_simulation.py": "22343ea7a593738fff29bdd4b6aa807c4ff7a6677e4139ba2ff2de03461167e0",
    "scripts/02_summarize_null_permutation.py": "00b697953ac5d0a69a620fcc1c98fce28f13c4c6545df81c193804ccd3521b3e",
    "scripts/03_build_sarqsar_figures.py": "320e561f618d28d1ad6c777dd1487b984be9953157498216548fecfb6dbb9cf0",
    "scripts/04_build_sarqsar_manuscript.py": "71a527e7b8a84f0280f4b255f041b05fd86c3cff5b033d3eae1002a29233b967",
    "scripts/05_build_sarqsar_submission_package.py": "8a4ec921237b8b2f847af5c0be8328eacad6999aa8f66d81bec54cf3f112cf0c",
    "scripts/99_run_all_sarqsar.py": "27cad8e6b39615c1c99a8bbc6dc4fad0574974380e95a3f930638fca840ce355",
    "scripts/__init__.py": "63b79a8351c7c6fbb89d3eaac8975f534f22e6e138c00a7c029b3620468c8cdc",
    "scripts/simulation_common.py": "3d44c3270b9c777fa58e6dda3c0b59ec70b4b6063a3c4a2e1fe4ccc417bbf8da",
}
MARKER = PROJECT / "scripts" / ".runtime_bundle_sha256"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_materialized() -> bool:
    if not MARKER.exists() or MARKER.read_text(encoding="utf-8").strip() != EXPECTED_TAR_SHA256:
        return False
    for relative, expected in EXPECTED_FILES.items():
        path = PROJECT / relative
        if not path.exists() or sha256_file(path) != expected:
            return False
    return True


def safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    destination_resolved = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        try:
            target.relative_to(destination_resolved)
        except ValueError as exc:
            raise RuntimeError(f"Unsafe archive member: {member.name}") from exc
        if member.issym() or member.islnk():
            raise RuntimeError(f"Links are not allowed in runtime bundle: {member.name}")
    archive.extractall(destination)


def materialize_runtime(force: bool = False) -> None:
    if not force and validate_materialized():
        print("SAR/QSAR RUNTIME BUNDLE: VERIFIED / REUSE")
        return
    missing = [str(path) for path in PARTS if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing runtime bundle parts: " + ", ".join(missing))
    encoded = "".join(path.read_text(encoding="ascii").strip() for path in PARTS)
    if sha256_bytes(encoded.encode("ascii")) != EXPECTED_B64_SHA256:
        raise AssertionError("Runtime base64 bundle hash mismatch")
    raw = base64.b64decode(encoded.encode("ascii"), validate=True)
    if sha256_bytes(raw) != EXPECTED_TAR_SHA256:
        raise AssertionError("Runtime tar.gz bundle hash mismatch")
    scripts = PROJECT / "scripts"
    if scripts.exists():
        shutil.rmtree(scripts)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        safe_extract(archive, PROJECT)
    for relative, expected in EXPECTED_FILES.items():
        path = PROJECT / relative
        observed = sha256_file(path)
        if observed != expected:
            raise AssertionError(f"Materialized source mismatch: {relative}: {observed} != {expected}")
    for path in sorted(scripts.glob("*.py")):
        py_compile.compile(str(path), doraise=True)
    MARKER.write_text(EXPECTED_TAR_SHA256 + "\n", encoding="utf-8")
    print("SAR/QSAR RUNTIME BUNDLE: MATERIALIZED + HASH VERIFIED + PY_COMPILE PASS")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize the verified SAR/QSAR runtime and run the resumable full pipeline."
    )
    parser.add_argument("--force-runtime-refresh", action="store_true")
    args, remaining = parser.parse_known_args()
    materialize_runtime(force=args.force_runtime_refresh)
    runner = PROJECT / "scripts" / "99_run_all_sarqsar.py"
    command = [sys.executable, str(runner), *remaining]
    print("RUN:", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=str(ROOT))
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
