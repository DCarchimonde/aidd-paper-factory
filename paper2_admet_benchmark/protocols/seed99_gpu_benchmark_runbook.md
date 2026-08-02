# RACER-C seed-99 RTX-4090 benchmark runbook

Status: **pre-freeze technical benchmark only**.

This workflow may read labels only for `D_dev`. It must not generate policy,
conformal, or test predictions, compute a performance metric, run seeds 101--110,
or create a protocol tag.

## Target environment

Use a fresh Linux RTX-4090 instance. The candidate lock is
`configs/racer_c/gpu_environment_lock.yaml`. It remains a candidate until the
runtime audit and benchmark both pass on the target GPU.

```bash
cd /root/autodl-tmp/aidd-paper-factory
git fetch origin
git switch paper2-reliability-extension-2026
git pull --ff-only origin paper2-reliability-extension-2026

conda create -n racer_c_gpu python=3.11.13 -y
conda activate racer_c_gpu

python -m pip install --upgrade pip
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r paper2_admet_benchmark/environment/racer_c_gpu_requirements.txt
```

The cleaned Tox21 files are intentionally not stored in Git. Rebuild them from
the locked NCATS archive with the existing acquisition/cleaning runbook, or copy
the already audited local files without changing bytes. Before continuing, these
two hashes must match the committed manifests:

- `Tox21_NR_ER_clean.csv`: `2a6217e66e3300e437d11fad68637b291526abc610c091effbbef4955d7d54a0`
- `Tox21_NR_ER_role_input.csv`: `edbe26eeee9cb9aa188e941f5884967b1775b3fe36d92349656a42b5b6bee900`

## Fail-closed preflight

```bash
python paper2_admet_benchmark/scripts/racer_c/capture_gpu_environment.py
python paper2_admet_benchmark/scripts/racer_c/prepare_seed99_gpu_benchmark.py
python -m unittest discover -s paper2_admet_benchmark/tests -v
```

Stop if any command fails. In particular, do not substitute another package
version, GPU, MoLFormer revision, token truncation rule, endpoint, track, or seed.
Report the generated `environment_audit.json` instead.

## Component timing benchmark

```bash
python paper2_admet_benchmark/scripts/racer_c/run_seed99_gpu_component_benchmark.py \
  2>&1 | tee paper2_admet_benchmark/results/logs/racer_c_seed99_gpu_component.log
```

The command measures frozen MoLFormer embedding extraction on the seed-99
development role and one representative Chemprop outer-final fit/prediction. It
validates prediction finiteness and transitive fit lineage but does not calculate
AUC, Brier score, calibration, policy feasibility, conformal coverage, or any
other scientific outcome.

Expected small outputs:

- `results/racer_c_phase3_preflight/environment/environment_audit.json`;
- `results/racer_c_phase3_preflight/seed99_gpu_benchmark_plan.json`;
- `results/racer_c_phase3_preflight/seed99_gpu_component_benchmark.json`.

Models, embeddings, row-level input tables, and component probabilities remain in
ignored local directories. After the benchmark, inspect only timing, environment,
hash, token-length, finiteness, and lineage fields. Formal freeze and seeds
101--110 remain blocked until the projected GPU budget is reported to and approved
by the user.
