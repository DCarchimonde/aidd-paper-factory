from __future__ import annotations

"""Safely refresh the final Paper 2 manuscript figures.

The final layout pass only needs to overwrite Figures 1 and 2.  Figures 3--6 are
already frozen manuscript assets.  A previous entry point unnecessarily invoked
the full figure builder first, which required a large row-level selective-curve
CSV even when the frozen Figure 5 PDF/PNG already existed.  This wrapper:

1. performs a full rebuild when all raw dependencies are available;
2. otherwise reuses the existing frozen Figures 3--6;
3. always rebuilds the layout-corrected Figures 1 and 2;
4. validates and refreshes the 12-file figure integrity manifest.

No model fitting or statistical re-analysis is performed.
"""

from pathlib import Path

import figure35_layout_fix as layout


REUSABLE_STEMS = [
    "figure_3_classification_conformal_tradeoff",
    "figure_4_applicability_domain_diagnostics",
    "figure_5_selective_prediction_diagnostics",
    "figure_6_regression_conformal_tradeoff",
]


def missing_reusable_assets() -> list[Path]:
    missing: list[Path] = []
    for stem in REUSABLE_STEMS:
        for suffix in (".pdf", ".png"):
            path = layout.base.FIGURE_DIR / f"{stem}{suffix}"
            if not path.exists():
                missing.append(path)
    return missing


def main() -> None:
    selective_source = (
        layout.base.SELECTIVE_DIR
        / "selective_prediction_v2_curves_confirmatory_full.csv"
    )

    if selective_source.exists():
        print("Raw selective curves found; rebuilding the complete six-figure package.")
        layout.base.main()
    else:
        missing = missing_reusable_assets()
        if missing:
            joined = "\n".join(str(path) for path in missing)
            raise FileNotFoundError(
                "The raw selective-curve source is absent and one or more frozen "
                "Figures 3--6 are also missing. Restore the frozen manuscript "
                "assets or regenerate the selective-prediction outputs before a "
                f"full rebuild. Missing assets:\n{joined}"
            )
        print(
            "Raw selective curves are absent; reusing frozen Figures 3--6 and "
            "rebuilding layout-corrected Figures 1 and 2 only."
        )

    layout.rebuild_workflow_figure()
    layout.rebuild_performance_figure()
    layout.refresh_manifest()
    print("Safe final manuscript figure refresh complete.")


if __name__ == "__main__":
    main()
