# Windows Environment Setup Notes

These notes are for the local `aidd_paper` conda environment on Windows.

## Current policy

Paper 2 should keep the main environment lightweight and reproducible.

Required packages:

- rdkit
- pandas
- numpy
- scikit-learn
- matplotlib
- tqdm
- joblib
- openpyxl
- xgboost
- shap
- requests
- pyyaml

`pytdc` is optional for the MVP. Do not install it in the main Windows environment unless we specifically decide to use PyTDC loaders.

## Why PyTDC is optional

Some newer PyTDC releases can pull heavy biomedical dependencies on Windows, including packages related to single-cell data infrastructure. This may require building native components such as `tiledbsoma`, which can fail on Windows and can also attempt to change core scientific packages.

For Paper 2 MVP, we prefer direct CSV/source loaders where possible. This keeps the environment stable and avoids unnecessary dependency risk.

## Safe environment check

From the repository root:

```powershell
python paper2_admet_benchmark/scripts/00_test_environment.py
```

The environment is acceptable if required packages pass and the script ends with:

```text
RDKit Morgan fingerprint test: 2048 bits
Environment test passed.
```

A warning for `tdc` is acceptable at this stage.

## If pip cache permission fails

If pip reports a permission error in the cache, use:

```powershell
python -m pip install <package-name> --no-cache-dir
```

Do not use `--user` inside this conda environment unless there is no alternative, because it can mix user-site packages with the conda environment.
