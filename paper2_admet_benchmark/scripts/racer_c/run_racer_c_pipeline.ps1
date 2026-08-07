[CmdletBinding()]
param(
    [ValidateSet("Validate", "Benchmark", "FreezeReview", "Full")]
    [string]$Mode = "Benchmark",
    [switch]$ForceRerun,
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][scriptblock]$Command
    )
    Write-Host "`n==> $Name" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

function Read-JsonFile {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
}

function Test-PassedBenchmark {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$ConfigPath,
        [Parameter(Mandatory = $true)][string]$ScriptPath
    )
    $Record = Read-JsonFile -Path $Path
    return (
        $null -ne $Record -and
        $Record.status -eq "pass_gpu_component_benchmark" -and
        $Record.config_sha256 -eq (Get-Sha256 -Path $ConfigPath) -and
        $Record.script_sha256 -eq (Get-Sha256 -Path $ScriptPath)
    )
}

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Assert-LockedFile {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$ExpectedSha256
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Missing locked input after restoration: $Path"
    }
    $ObservedSha256 = Get-Sha256 -Path $Path
    if ($ObservedSha256 -ne $ExpectedSha256) {
        throw "SHA256 mismatch for ${Path}: expected=$ExpectedSha256 observed=$ObservedSha256"
    }
    Write-Host "PASS $Path $ObservedSha256"
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Set-Location $RepoRoot

$Config = Join-Path $RepoRoot "paper2_admet_benchmark\configs\racer_c\gpu_environment_lock.yaml"
$BenchmarkScript = Join-Path $RepoRoot "paper2_admet_benchmark\scripts\racer_c\run_seed99_gpu_component_benchmark.py"
$EnvironmentDir = Join-Path $RepoRoot "paper2_admet_benchmark\results\racer_c_phase3_preflight\environment_windows_rtx4060"
$Result = Join-Path $RepoRoot "paper2_admet_benchmark\results\racer_c_phase3_preflight\seed99_gpu_component_benchmark_windows_rtx4060.json"
$FreezeReviewResult = Join-Path $RepoRoot "paper2_admet_benchmark\results\racer_c_phase4_freeze_review\formal_freeze_review_windows_rtx4060.json"
$LogDir = Join-Path $RepoRoot "paper2_admet_benchmark\results\logs"
$RunId = Get-Date -Format "yyyyMMdd_HHmmss"
$Log = Join-Path $LogDir "racer_c_${Mode}_${RunId}.log"
$WorkDir = Join-Path $RepoRoot "paper2_admet_benchmark\data\processed\racer_c\gpu_benchmark_seed99\attempt_${RunId}"
$ProcessedDir = Join-Path $RepoRoot "paper2_admet_benchmark\data\processed\racer_c"
$RuntimeManifestDir = Join-Path $ProcessedDir "runtime_manifests"
$CleanInput = Join-Path $ProcessedDir "Tox21_NR_ER_clean.csv"
$RoleInput = Join-Path $ProcessedDir "role_inputs\Tox21_NR_ER_role_input.csv"
$SourceDir = Join-Path $RepoRoot ".local\racer_c_sources"
$Archive = Join-Path $SourceDir "tox21_10k_data_allsdf.zip"
$ExtractDir = Join-Path $SourceDir "tox21_10k_data_allsdf"
$SourceSdf = Join-Path $ExtractDir "tox21_10k_data_all.sdf"
$SourceUrl = "https://tripod.nih.gov/tox21/challenge/download?id=tox21_10k_data_allsdf&sec="
$ExpectedArchiveSha256 = "024a3ae2690bcd4a593e6e0b10b455470b9bcb1d8f299dd36f220a250181517b"
$ExpectedSdfSha256 = "d66e1f9ec945ee528b1bea6e49af9c10d0bad546c2b304eb96004c8228824206"
$ExpectedCleanSha256 = "2a6217e66e3300e437d11fad68637b291526abc610c091effbbef4955d7d54a0"
$ExpectedRoleSha256 = "edbe26eeee9cb9aa188e941f5884967b1775b3fe36d92349656a42b5b6bee900"
$FreezePrimaryEndpoints = @(
    "Tox21_NR_AhR",
    "Tox21_NR_ER",
    "Tox21_SR_ARE",
    "Tox21_SR_MMP"
)
$CommittedManifestDir = Join-Path $RepoRoot "paper2_admet_benchmark\data\manifests\racer_c"

function Ensure-LockedTox21Source {
    New-Item -ItemType Directory -Force $SourceDir | Out-Null
    if (-not (Test-Path -LiteralPath $Archive -PathType Leaf)) {
        $PartialArchive = "${Archive}.partial"
        if (Test-Path -LiteralPath $PartialArchive) {
            throw "An incomplete prior download exists; remove it only after inspection: $PartialArchive"
        }
        Write-Host "Downloading locked NCATS Tox21 archive..."
        Invoke-WebRequest -Uri $SourceUrl -OutFile $PartialArchive
        Assert-LockedFile -Path $PartialArchive -ExpectedSha256 $ExpectedArchiveSha256
        Move-Item -LiteralPath $PartialArchive -Destination $Archive
    }
    Assert-LockedFile -Path $Archive -ExpectedSha256 $ExpectedArchiveSha256

    if (-not (Test-Path -LiteralPath $SourceSdf -PathType Leaf)) {
        New-Item -ItemType Directory -Force $ExtractDir | Out-Null
        Expand-Archive -LiteralPath $Archive -DestinationPath $ExtractDir -Force
    }
    Assert-LockedFile -Path $SourceSdf -ExpectedSha256 $ExpectedSdfSha256
}

New-Item -ItemType Directory -Force $LogDir | Out-Null
Start-Transcript -LiteralPath $Log -Force | Out-Null

try {
    Write-Host "RACER-C pipeline mode: $Mode" -ForegroundColor Green
    Write-Host "Repository: $RepoRoot"
    Write-Host "Log: $Log"

    if ($env:CONDA_DEFAULT_ENV -ne "racer_c_gpu") {
        throw "Activate the racer_c_gpu Conda environment before running this command."
    }

    Write-Host "`n==> Restore and verify locked NR-ER inputs" -ForegroundColor Cyan
    $CleanReady = (
        (Test-Path -LiteralPath $CleanInput -PathType Leaf) -and
        ((Get-Sha256 -Path $CleanInput) -eq $ExpectedCleanSha256)
    )
    $RoleReady = (
        (Test-Path -LiteralPath $RoleInput -PathType Leaf) -and
        ((Get-Sha256 -Path $RoleInput) -eq $ExpectedRoleSha256)
    )

    if (-not ($CleanReady -and $RoleReady)) {
        if ($Mode -eq "FreezeReview") {
            Write-Host (
                "NR-ER is not ready; recovery is deferred to the staged " +
                "four-endpoint freeze-input gate."
            ) -ForegroundColor Yellow
        }
        else {
            if ((Test-Path -LiteralPath $CleanInput -PathType Leaf) -and -not $CleanReady) {
                throw "Existing clean input has the wrong SHA256; refusing to overwrite it: $CleanInput"
            }
            if ((Test-Path -LiteralPath $RoleInput -PathType Leaf) -and -not $RoleReady) {
                throw "Existing role input has the wrong SHA256; refusing to overwrite it: $RoleInput"
            }

            Ensure-LockedTox21Source

            Invoke-Checked -Name "Deterministic NCATS Tox21 cleaning" -Command {
                python paper2_admet_benchmark\scripts\racer_c\prepare_tox21_challenge.py --archive $Archive --sdf $SourceSdf --manifest-dir $RuntimeManifestDir
            }
            Invoke-Checked -Name "Label-blind NR-ER similarity clustering" -Command {
                python paper2_admet_benchmark\scripts\racer_c\build_similarity_clusters.py --endpoints Tox21_NR_ER --manifest-dir $RuntimeManifestDir
            }
        }
    }
    else {
        Write-Host "Locked processed inputs already exist; data restoration skipped."
    }
    if ($Mode -ne "FreezeReview") {
        Assert-LockedFile -Path $CleanInput -ExpectedSha256 $ExpectedCleanSha256
        Assert-LockedFile -Path $RoleInput -ExpectedSha256 $ExpectedRoleSha256
    }

    if ($Mode -eq "FreezeReview") {
        Write-Host "`n==> Verify all four primary freeze inputs" -ForegroundColor Cyan
        $PrimaryInputRecords = @()
        $PrimaryInputMismatches = @()
        foreach ($Endpoint in $FreezePrimaryEndpoints) {
            $ManifestPath = Join-Path $CommittedManifestDir "${Endpoint}_cleaning.json"
            $Manifest = Read-JsonFile -Path $ManifestPath
            if ($null -eq $Manifest) {
                throw "Missing committed cleaning manifest: $ManifestPath"
            }
            if ($Manifest.endpoint -ne $Endpoint) {
                throw "Cleaning manifest endpoint mismatch: expected=$Endpoint observed=$($Manifest.endpoint)"
            }
            if ($Manifest.similarity_cluster_status -ne "complete") {
                throw "Committed role input is not cluster-complete for $Endpoint"
            }
            foreach ($HashField in @(
                "cleaned_byte_sha256",
                "rejections_byte_sha256",
                "role_input_byte_sha256"
            )) {
                if ([string]($Manifest.$HashField) -notmatch "^[0-9a-f]{64}$") {
                    throw "Invalid $HashField in committed manifest for $Endpoint"
                }
            }

            $Record = [PSCustomObject]@{
                Endpoint = $Endpoint
                CleanPath = Join-Path $ProcessedDir "${Endpoint}_clean.csv"
                RejectionPath = Join-Path $ProcessedDir "${Endpoint}_rejections.csv"
                RolePath = Join-Path $ProcessedDir "role_inputs\${Endpoint}_role_input.csv"
                CleanSha256 = [string]$Manifest.cleaned_byte_sha256
                RejectionSha256 = [string]$Manifest.rejections_byte_sha256
                RoleSha256 = [string]$Manifest.role_input_byte_sha256
            }
            $PrimaryInputRecords += $Record

            $Checks = @(
                [PSCustomObject]@{
                    Path = $Record.CleanPath
                    Expected = $Record.CleanSha256
                    Label = "clean"
                }
                [PSCustomObject]@{
                    Path = $Record.RejectionPath
                    Expected = $Record.RejectionSha256
                    Label = "rejections"
                }
                [PSCustomObject]@{
                    Path = $Record.RolePath
                    Expected = $Record.RoleSha256
                    Label = "cluster-complete role"
                }
            )
            foreach ($Check in $Checks) {
                $Observed = if (Test-Path -LiteralPath $Check.Path -PathType Leaf) {
                    Get-Sha256 -Path $Check.Path
                }
                else {
                    "missing"
                }
                if ($Observed -ne $Check.Expected) {
                    $PrimaryInputMismatches += (
                        "{0} {1}: expected={2} observed={3}" -f `
                        $Endpoint, $Check.Label, $Check.Expected, $Observed
                    )
                }
            }
        }

        if ($PrimaryInputMismatches.Count -gt 0) {
            Write-Host "Primary freeze inputs require deterministic recovery:" -ForegroundColor Yellow
            $PrimaryInputMismatches | ForEach-Object { Write-Host "  $_" }
            Ensure-LockedTox21Source

            # Rebuild away from the live processed directory. Nothing is replaced until
            # all four clean, rejection, and cluster-complete role files match the
            # committed manifests byte-for-byte.
            $RecoveryRoot = Join-Path $RepoRoot (
                "paper2_admet_benchmark\results\racer_c_phase4_freeze_review\input_recovery_attempt_${RunId}"
            )
            $StagedProcessedDir = Join-Path $RecoveryRoot "staged_processed"
            $StagedManifestDir = Join-Path $RecoveryRoot "staged_manifests"
            $BackupDir = Join-Path $RecoveryRoot "replaced_input_backup"
            New-Item -ItemType Directory -Force $StagedProcessedDir | Out-Null
            New-Item -ItemType Directory -Force $StagedManifestDir | Out-Null
            New-Item -ItemType Directory -Force $BackupDir | Out-Null

            Invoke-Checked -Name "Deterministic staged Tox21 rebuild" -Command {
                python paper2_admet_benchmark\scripts\racer_c\prepare_tox21_challenge.py `
                    --archive $Archive `
                    --sdf $SourceSdf `
                    --processed-dir $StagedProcessedDir `
                    --manifest-dir $StagedManifestDir
            }
            $PrimaryEndpointCsv = $FreezePrimaryEndpoints -join ","
            Invoke-Checked -Name "Label-blind staged clustering for all four primary endpoints" -Command {
                python paper2_admet_benchmark\scripts\racer_c\build_similarity_clusters.py `
                    --endpoints $PrimaryEndpointCsv `
                    --processed-dir $StagedProcessedDir `
                    --manifest-dir $StagedManifestDir
            }

            foreach ($Record in $PrimaryInputRecords) {
                $StagedCleanPath = Join-Path $StagedProcessedDir "$($Record.Endpoint)_clean.csv"
                $StagedRejectionPath = Join-Path $StagedProcessedDir "$($Record.Endpoint)_rejections.csv"
                $StagedRolePath = Join-Path $StagedProcessedDir "role_inputs\$($Record.Endpoint)_role_input.csv"
                Assert-LockedFile -Path $StagedCleanPath -ExpectedSha256 $($Record.CleanSha256)
                Assert-LockedFile -Path $StagedRejectionPath -ExpectedSha256 $($Record.RejectionSha256)
                Assert-LockedFile -Path $StagedRolePath -ExpectedSha256 $($Record.RoleSha256)
            }

            foreach ($Record in $PrimaryInputRecords) {
                $StagedCleanPath = Join-Path $StagedProcessedDir "$($Record.Endpoint)_clean.csv"
                $StagedRejectionPath = Join-Path $StagedProcessedDir "$($Record.Endpoint)_rejections.csv"
                $StagedRolePath = Join-Path $StagedProcessedDir "role_inputs\$($Record.Endpoint)_role_input.csv"
                $EndpointBackupDir = Join-Path $BackupDir $Record.Endpoint
                New-Item -ItemType Directory -Force $EndpointBackupDir | Out-Null
                foreach ($LivePath in @($Record.CleanPath, $Record.RejectionPath, $Record.RolePath)) {
                    if (Test-Path -LiteralPath $LivePath -PathType Leaf) {
                        Copy-Item -LiteralPath $LivePath -Destination $EndpointBackupDir -Force
                    }
                }
                New-Item -ItemType Directory -Force (Split-Path -Parent $Record.CleanPath) | Out-Null
                New-Item -ItemType Directory -Force (Split-Path -Parent $Record.RolePath) | Out-Null
                Copy-Item -LiteralPath $StagedCleanPath -Destination $Record.CleanPath -Force
                Copy-Item -LiteralPath $StagedRejectionPath -Destination $Record.RejectionPath -Force
                Copy-Item -LiteralPath $StagedRolePath -Destination $Record.RolePath -Force
            }
            Write-Host "Verified recovery complete; replaced files were preserved at $BackupDir"
        }

        foreach ($Record in $PrimaryInputRecords) {
            Assert-LockedFile -Path $($Record.CleanPath) -ExpectedSha256 $($Record.CleanSha256)
            Assert-LockedFile -Path $($Record.RejectionPath) -ExpectedSha256 $($Record.RejectionSha256)
            Assert-LockedFile -Path $($Record.RolePath) -ExpectedSha256 $($Record.RoleSha256)
        }
    }

    Invoke-Checked -Name "Python and CUDA driver diagnostic" -Command {
        python -c "import json, platform, subprocess, torch; print(json.dumps({'python': platform.python_version(), 'torch': torch.__version__, 'torch_cuda_build': torch.version.cuda, 'cuda_available': torch.cuda.is_available(), 'device_count': torch.cuda.device_count()}, sort_keys=True)); subprocess.run(['nvidia-smi', '--query-gpu=driver_version,name,memory.total', '--format=csv,noheader,nounits'], check=True)"
    }

    Invoke-Checked -Name "Fail-closed locked environment audit" -Command {
        python paper2_admet_benchmark\scripts\racer_c\capture_gpu_environment.py --config $Config --output-dir $EnvironmentDir
    }

    Invoke-Checked -Name "Seed-99 benchmark plan" -Command {
        python paper2_admet_benchmark\scripts\racer_c\prepare_seed99_gpu_benchmark.py
    }

    if (-not $SkipTests) {
        Invoke-Checked -Name "RACER-C contract tests" -Command {
            python -m unittest discover -s paper2_admet_benchmark\tests -v
        }
    }

    Invoke-Checked -Name "Seed-99 component dry-run" -Command {
        python paper2_admet_benchmark\scripts\racer_c\run_seed99_gpu_component_benchmark.py --config $Config --environment-audit "$EnvironmentDir\environment_audit.json" --work-dir $WorkDir --dry-run
    }

    if ($Mode -eq "Validate") {
        Write-Host "`nVALIDATION COMPLETE. No GPU model fit was started." -ForegroundColor Green
        exit 0
    }

    if ($Mode -eq "FreezeReview") {
        Invoke-Checked -Name "Prediction-free four-endpoint formal freeze review" -Command {
            python paper2_admet_benchmark\scripts\racer_c\prepare_formal_freeze_review.py `
                --config $Config `
                --benchmark $Result `
                --output $FreezeReviewResult
        }
        $FreezeRecord = Read-JsonFile -Path $FreezeReviewResult
        if (
            $null -eq $FreezeRecord -or
            $FreezeRecord.status -ne "pass_prediction_free_formal_freeze_review" -or
            $FreezeRecord.scientific_predictions_generated -ne $false -or
            $FreezeRecord.track_seed_cell_count -ne 60
        ) {
            throw "Formal freeze-review command returned without the required prediction-free pass record."
        }
        Write-Host "`nFORMAL FREEZE REVIEW COMPLETE: $FreezeReviewResult" -ForegroundColor Green
        Write-Host "No confirmatory model was fit. Explicit user approval remains required before the protocol tag or seeds 101-110."
        exit 0
    }

    if ($Mode -eq "Full") {
        $StudyDesign = Get-Content "paper2_admet_benchmark\configs\racer_c\study_design.yaml" -Raw
        if ($StudyDesign -match "protocol_status:\s*draft_pre_freeze") {
            throw "Full confirmatory execution is blocked: the RACER-C protocol is still draft_pre_freeze. FreezeReview, explicit approval, and the formal protocol tag are required before seeds 101-110."
        }
        $ProductionRunner = "paper2_admet_benchmark\scripts\racer_c\run_confirmatory_racer_c.py"
        if (-not (Test-Path -LiteralPath $ProductionRunner)) {
            throw "Full confirmatory execution is blocked: the production RACER-C runner has not been implemented or frozen."
        }
        Invoke-Checked -Name "Frozen full RACER-C confirmatory study" -Command {
            python $ProductionRunner --resume
        }
        Write-Host "`nFULL RACER-C STUDY COMPLETE." -ForegroundColor Green
        exit 0
    }

    if ((Test-PassedBenchmark -Path $Result -ConfigPath $Config -ScriptPath $BenchmarkScript) -and -not $ForceRerun) {
        Write-Host "`nBENCHMARK ALREADY PASSED: $Result" -ForegroundColor Yellow
        Write-Host "Use -ForceRerun only when a scientifically documented rerun is required."
        exit 0
    }

    Invoke-Checked -Name "Actual seed-99 RTX-4060 component benchmark" -Command {
        python paper2_admet_benchmark\scripts\racer_c\run_seed99_gpu_component_benchmark.py --config $Config --environment-audit "$EnvironmentDir\environment_audit.json" --work-dir $WorkDir --output $Result
    }

    if (-not (Test-PassedBenchmark -Path $Result -ConfigPath $Config -ScriptPath $BenchmarkScript)) {
        throw "Benchmark command returned but the required pass record is missing."
    }
    Write-Host "`nSEED-99 GPU BENCHMARK COMPLETE: $Result" -ForegroundColor Green
    Write-Host "Formal protocol review/freeze is the next gate; seeds 101-110 were not run."
}
finally {
    Stop-Transcript | Out-Null
}
