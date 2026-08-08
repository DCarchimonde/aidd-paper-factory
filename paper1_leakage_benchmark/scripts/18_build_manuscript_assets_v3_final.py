from __future__ import annotations

import importlib.util
from pathlib import Path

BASE_SCRIPT = Path(__file__).with_name("16_build_manuscript_assets_v3.py")


def load_base_module():
    spec = importlib.util.spec_from_file_location("paper1_assets_v3", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import visual asset builder: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = load_base_module()
    original_read_csv = module.pd.read_csv

    def read_csv_with_inference_alias(*args, **kwargs):
        frame = original_read_csv(*args, **kwargs)
        if "inference_label" in frame.columns and "inference" not in frame.columns:
            frame = frame.copy()
            frame["inference"] = frame["inference_label"].astype(str).str.replace("_", " ", regex=False)
        return frame

    # The authoritative v3 analysis table uses `inference_label`.  The redesigned
    # plotting layer accepts a human-readable alias so visual encoding remains
    # compatible without modifying any frozen result CSV.
    module.pd.read_csv = read_csv_with_inference_alias
    module.main()


if __name__ == "__main__":
    main()
