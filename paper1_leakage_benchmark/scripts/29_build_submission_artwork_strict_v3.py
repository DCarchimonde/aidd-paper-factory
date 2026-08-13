from __future__ import annotations

"""Authoritative publication-artwork pass for Paper 1.

All scientific inputs are already frozen. This entrypoint redraws each main/SI
figure once at fixed physical size, creates PDF + 600-dpi TIFF companions, and
runs physical-size and rendered-text checks. It never fits models or regenerates
partitions/inference.
"""

import publication_artwork_common_v3 as common
import publication_figure1_v3 as f1
import publication_figures_primary_v3 as primary
import publication_figures_diagnostics_v3 as diagnostic
import publication_figures_si_v3 as si


def main() -> None:
    common.configure()
    f1.build()
    primary.figure2()
    primary.figure3()
    diagnostic.figure4()
    diagnostic.figure5()
    diagnostic.figure6()
    si.figure_s1()
    si.figure_s2()
    si.figure_s3()
    si.figure_s4()
    si.figure_s5()
    common.finish()
    print("SUBMISSION-FINAL PUBLICATION ARTWORK: PASS")


if __name__ == "__main__":
    main()
