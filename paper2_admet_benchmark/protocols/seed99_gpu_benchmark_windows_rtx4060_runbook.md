# RACER-C seed-99 Windows RTX-4060 Laptop benchmark runbook

Status: **pre-freeze technical benchmark only**.

This is the current approved pre-freeze runbook for the user's local laptop. It
fixes the execution platform and MoLFormer inference batch size for the RTX 4060.
Endpoint,
roles, seed, model revision, token policy, Chemprop architecture, optimization,
and scientific failure rules are unchanged.

The workflow may read labels only for `D_dev`. It must not generate policy,
conformal, or test predictions, calculate a performance metric, run seeds
101--110, or create a protocol tag.

## 1. Update the extension branch

Open **Anaconda PowerShell Prompt**:

```powershell
Set-Location E:\AIDD_Paper_Factory
git fetch origin
git switch paper2-reliability-extension-2026
git pull --ff-only origin paper2-reliability-extension-2026
```

Do not use `main` for this benchmark.

## 2. Create the isolated Windows environment

```powershell
conda create -n racer_c_gpu python=3.11.13 -y
conda activate racer_c_gpu

python -m pip install --upgrade pip
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r paper2_admet_benchmark\environment\racer_c_gpu_requirements.txt
```

The candidate runtime is Windows AMD64, one NVIDIA GeForce RTX 4060 Laptop GPU,
at least 7 GiB visible VRAM, PyTorch 2.13.0+cu130, Chemprop 2.3.0, and the exact
versions in `gpu_environment_windows_rtx4060.yaml`. The MoLFormer extraction batch
is fixed at 8 for the laptop; embeddings remain float32 and truncation remains
forbidden.

## 3. Restore the audited NR-ER inputs

The following ignored local files must exist:

```text
paper2_admet_benchmark\data\processed\racer_c\Tox21_NR_ER_clean.csv
paper2_admet_benchmark\data\processed\racer_c\role_inputs\Tox21_NR_ER_role_input.csv
```

If the audited files already exist, continue to the hash check below. Otherwise,
download the locked NCATS archive and rebuild the files:

```powershell
New-Item -ItemType Directory -Force .local\racer_c_sources | Out-Null
Invoke-WebRequest `
  -Uri "https://tripod.nih.gov/tox21/challenge/download?id=tox21_10k_data_allsdf&sec=" `
  -OutFile .local\racer_c_sources\tox21_10k_data_allsdf.zip

Expand-Archive `
  -Path .local\racer_c_sources\tox21_10k_data_allsdf.zip `
  -DestinationPath .local\racer_c_sources\tox21_10k_data_allsdf `
  -Force

python paper2_admet_benchmark\scripts\racer_c\prepare_tox21_challenge.py `
  --archive .local\racer_c_sources\tox21_10k_data_allsdf.zip `
  --sdf .local\racer_c_sources\tox21_10k_data_allsdf\tox21_10k_data_all.sdf `
  --manifest-dir paper2_admet_benchmark\data\processed\racer_c\runtime_manifests

python paper2_admet_benchmark\scripts\racer_c\build_similarity_clusters.py `
  --endpoints Tox21_NR_ER `
  --manifest-dir paper2_admet_benchmark\data\processed\racer_c\runtime_manifests
```

The raw archive and SDF are local-only and must not be committed.

Verify all four locked hashes:

```powershell
$Expected = @{
  ".local\racer_c_sources\tox21_10k_data_allsdf.zip" = "024a3ae2690bcd4a593e6e0b10b455470b9bcb1d8f299dd36f220a250181517b"
  ".local\racer_c_sources\tox21_10k_data_allsdf\tox21_10k_data_all.sdf" = "d66e1f9ec945ee528b1bea6e49af9c10d0bad546c2b304eb96004c8228824206"
  "paper2_admet_benchmark\data\processed\racer_c\Tox21_NR_ER_clean.csv" = "2a6217e66e3300e437d11fad68637b291526abc610c091effbbef4955d7d54a0"
  "paper2_admet_benchmark\data\processed\racer_c\role_inputs\Tox21_NR_ER_role_input.csv" = "edbe26eeee9cb9aa188e941f5884967b1775b3fe36d92349656a42b5b6bee900"
}
foreach ($Path in $Expected.Keys) {
  if (-not (Test-Path $Path)) { throw "Missing locked input: $Path" }
  $Observed = (Get-FileHash -Algorithm SHA256 $Path).Hash.ToLower()
  if ($Observed -ne $Expected[$Path]) {
    throw "SHA256 mismatch for ${Path}: $Observed"
  }
  Write-Host "PASS $Path $Observed"
}
```

If the two processed files already existed but the raw source folder did not,
verify the two processed-file hashes directly; the benchmark runner independently
rechecks them and will fail closed on any mismatch.

## 4. Run the fail-closed preflight

```powershell
$Config = "paper2_admet_benchmark\configs\racer_c\gpu_environment_windows_rtx4060.yaml"
$EnvironmentDir = "paper2_admet_benchmark\results\racer_c_phase3_preflight\environment_windows_rtx4060"

python paper2_admet_benchmark\scripts\racer_c\capture_gpu_environment.py `
  --config $Config `
  --output-dir $EnvironmentDir
if ($LASTEXITCODE -ne 0) { throw "GPU environment audit failed" }

python paper2_admet_benchmark\scripts\racer_c\prepare_seed99_gpu_benchmark.py
if ($LASTEXITCODE -ne 0) { throw "Seed-99 plan generation failed" }

python -m unittest discover -s paper2_admet_benchmark\tests -v
if ($LASTEXITCODE -ne 0) { throw "Repository tests failed" }

python paper2_admet_benchmark\scripts\racer_c\run_seed99_gpu_component_benchmark.py `
  --config $Config `
  --environment-audit "$EnvironmentDir\environment_audit.json" `
  --dry-run
if ($LASTEXITCODE -ne 0) { throw "Component dry-run failed" }
```

Stop if any command fails. Do not substitute a package, model revision, endpoint,
track, seed, token truncation rule, or training hyperparameter after a failure.

## 5. Run the actual seed-99 component benchmark

Close memory-heavy GPU applications first. The program records MoLFormer torch
peak allocation/reservation and samples whole-device memory during Chemprop.

```powershell
$Log = "paper2_admet_benchmark\results\logs\racer_c_seed99_windows_rtx4060.log"
$Result = "paper2_admet_benchmark\results\racer_c_phase3_preflight\seed99_gpu_component_benchmark_windows_rtx4060.json"

python paper2_admet_benchmark\scripts\racer_c\run_seed99_gpu_component_benchmark.py `
  --config $Config `
  --environment-audit "$EnvironmentDir\environment_audit.json" `
  --output $Result `
  2>&1 | Tee-Object -FilePath $Log
if ($LASTEXITCODE -ne 0) { throw "GPU component benchmark failed" }
```

This executes frozen MoLFormer embedding extraction over 2,928 NR-ER development
rows and one representative Chemprop outer-final fit/prediction. It does not
calculate AUC, Brier score, calibration, policy feasibility, conformal coverage,
or any other paper outcome.

## 6. Return the small audit packet

Return these files for review:

- `environment_windows_rtx4060\environment_audit.json`;
- `environment_windows_rtx4060\nvidia_smi.txt`;
- `seed99_gpu_component_benchmark_windows_rtx4060.json`;
- `racer_c_seed99_windows_rtx4060.log`.

Model checkpoints, embeddings, raw/processed molecular tables, and probabilities
remain ignored locally. Formal freeze and seeds 101--110 stay blocked until this
packet passes timing, memory, finiteness, token-length, hash, and lineage review.
