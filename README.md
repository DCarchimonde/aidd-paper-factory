# AIDD Paper Factory

This repository is a research project pool for AI-driven drug discovery papers.

## Project Structure

- `paper1_leakage_benchmark`: Leakage-aware molecular property prediction benchmark
- `paper2_admet_benchmark`: ADMET and molecular property benchmark extension
- `paper3_target_screening`: Target-specific interpretable ML + virtual screening
- `shared_utils`: Shared Python utilities
- `environment`: Conda / pip environment files
- `docs`: Planning documents, journal notes, and research logs

## Suggested Workflow

1. Create conda environment: `conda create -n aidd_paper python=3.10 -y`
2. Activate environment: `conda activate aidd_paper`
3. Install dependencies: `pip install -r environment/requirements.txt`
4. Run RDKit test: `python paper1_leakage_benchmark/scripts/00_test_rdkit.py`
