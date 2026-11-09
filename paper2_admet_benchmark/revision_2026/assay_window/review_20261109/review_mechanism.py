"""Describe an existing assay-window return package without refitting models.

Run from the repository root, or invoke this file from its published location
inside the repository. --return-root contains the extracted screen/ directory.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
from pathlib import Path
import sys

import numpy as np

# Support both direct-script and python -m invocation after publication.
for candidate in (Path.cwd(), *Path(__file__).resolve().parents):
    if (candidate / "paper2_admet_benchmark/revision_2026/assay_window/targets.py").is_file():
        sys.path.insert(0, str(candidate))
        break
from paper2_admet_benchmark.revision_2026.assay_window.targets import pair_allowed_mask


def analyze(return_root: Path) -> dict:
    screen = return_root / "screen"
    records = json.loads(gzip.decompress((screen / "derived_records.json.gz").read_bytes()))
    metrics = json.loads((screen / "metrics.json").read_text(encoding="utf-8"))
    lookup = {(r["split"], r["method"]): r for r in metrics}
    by_sample = collections.defaultdict(list)
    by_mask = collections.defaultdict(list)
    for record in records:
        by_sample[(record["sample_id"], record["protocol"])].append(record)
        mask = pair_allowed_mask(record["target_onset"], record["viability_onset"])
        by_mask[np.packbits(mask).tobytes()].append(record)
    complete = [
        rows for rows in by_sample.values()
        if len(rows) == 3 and {r["replicate"] for r in rows} == {1, 2, 3}
    ]
    by_repeat = collections.defaultdict(collections.Counter)
    for rows in complete:
        for record in rows:
            by_repeat[record["replicate"]][record["window_reference"]["status"]] += 1
    out = {
        "analysis_kind": "post_hoc_existing_outputs_only",
        "real_model_refits": 0,
        "complete_three_replicate_sample_protocol_count": len(complete),
        "same_complete_samples_by_repeat": {
            str(key): {
                "counts": dict(value),
                "definite_yes_fraction": value["definite_yes"] / len(complete),
            }
            for key, value in sorted(by_repeat.items())
        },
        "grid_diagnostic": {
            "unique_observation_masks": len(by_mask),
            "masks_shared_by_definite_yes_and_no": sum(
                {"definite_yes", "definite_no"}.issubset(
                    {r["window_reference"]["status"] for r in rows}
                )
                for rows in by_mask.values()
            ),
        },
        "comparisons": [],
    }
    for split in ["random", "scaffold"]:
        joint = lookup[split, "joint_censored"]
        independent = lookup[split, "independent_censored"]
        binary = lookup[split, "binary_window"]
        joint_nll = joint["masked_nll_molecule_weighted"]
        independent_nll = independent["masked_nll_molecule_weighted"]
        out["comparisons"].append({
            "split": split,
            "joint_nll": joint_nll,
            "independent_nll": independent_nll,
            "relative_nll_reduction_pct": 100 * (independent_nll - joint_nll) / independent_nll,
            "yes_nll_joint_minus_independent": (
                joint["by_reference_status"]["definite_yes"]["masked_nll_molecule_weighted"]
                - independent["by_reference_status"]["definite_yes"]["masked_nll_molecule_weighted"]
            ),
            "repeat3_ap_joint": joint["repeat3_reference"]["average_precision"],
            "repeat3_ap_independent": independent["repeat3_reference"]["average_precision"],
            "repeat3_ap_binary": binary["repeat3_reference"]["average_precision"],
            "all_repeats_ap_joint": joint["all_repeats"]["average_precision"],
            "all_repeats_ap_binary": binary["all_repeats"]["average_precision"],
        })
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--return-root", type=Path, required=True,
                        help="directory containing the extracted screen/ package")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="directory to create for mechanism_observations.json")
    args = parser.parse_args(argv)
    result = analyze(args.return_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / "mechanism_observations.json"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {destination}; model refits: 0")


if __name__ == "__main__":
    main()
