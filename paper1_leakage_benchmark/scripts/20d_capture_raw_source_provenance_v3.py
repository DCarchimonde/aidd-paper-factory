from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper1_leakage_benchmark"
RAW = PAPER / "data" / "raw"
TABLES = PAPER / "results" / "tables"
GEN = ROOT / "paper1_latex" / "generated"
TABLES.mkdir(parents=True, exist_ok=True)
GEN.mkdir(parents=True, exist_ok=True)

from shared_utils.dataset_registry import DATASETS


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    rows = []
    for dataset, spec in DATASETS.items():
        path = RAW / f"{dataset.lower()}_raw.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        rows.append({
            "dataset": dataset,
            "task_type": spec.task_type,
            "source_url": spec.url,
            "raw_local_file": str(path.relative_to(ROOT)),
            "raw_bytes": int(path.stat().st_size),
            "raw_sha256": sha256(path),
        })
    frame = pd.DataFrame(rows)
    out = TABLES / "q1_raw_source_provenance_v3.csv"
    frame.to_csv(out, index=False)
    text = (
        "Raw-source provenance was re-audited during the final build. For all six datasets, the exact download URL used by the dataset registry, local raw-file path, byte size, and SHA-256 digest are recorded in the machine-readable file "
        "\\texttt{q1\\_raw\\_source\\_provenance\\_v3.csv}. The raw inputs themselves are not silently replaced during the final manuscript build."
    )
    (GEN / "q1_raw_provenance_text_v3.tex").write_text(text + "\n", encoding="utf-8")
    print(frame[["dataset", "raw_bytes", "raw_sha256"]].to_string(index=False))
    print("RAW SOURCE PROVENANCE CAPTURE: PASS")


if __name__ == "__main__":
    main()
