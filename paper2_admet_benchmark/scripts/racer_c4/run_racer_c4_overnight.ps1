[CmdletBinding()]
param(
    [string]$CondaEnv = "aidd_paper",
    [ValidateSet("primary", "all")]
    [string]$Scope = "all",
    [ValidateRange(1, 3)]
    [int]$MaximumPasses = 2,
    [switch]$SkipEnvironmentRepair
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Set-Location $RepoRoot

$LogDir = Join-Path $RepoRoot "paper2_admet_benchmark\results\logs"
$OutputDir = Join-Path $RepoRoot ".local\racer_c4_results"
$FinalReport = Join-Path $OutputDir "final_report.json"
$RequiredRdkitRuntime = "2026.03.4"
$RequiredRdkitDistribution = "2026.3.4"
New-Item -ItemType Directory -Force $LogDir | Out-Null
$LogPath = Join-Path $LogDir (
    "racer_c4_overnight_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss")
)

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "Conda is not available in this PowerShell session."
}

function Get-RdkitRuntimeVersion {
    $ProbeOutput = @(
        & conda run -n $CondaEnv python -c `
            "from rdkit import rdBase; print(rdBase.rdkitVersion)" 2>&1
    )
    if ($LASTEXITCODE -ne 0) {
        return ""
    }
    $Versions = @(
        $ProbeOutput |
            ForEach-Object { "$_".Trim() } |
            Where-Object { $_ -match '^\d{4}\.\d{2}\.\d+$' }
    )
    if ($Versions.Count -eq 0) {
        return ""
    }
    return [string]$Versions[-1]
}

function Install-LockedRdkit {
    Write-Host (
        "Installing the locked RDKit $RequiredRdkitDistribution wheel " +
        "without changing its dependencies..."
    ) -ForegroundColor Yellow
    & conda run --no-capture-output -n $CondaEnv python -m pip install `
        --disable-pip-version-check `
        --no-cache-dir `
        --no-deps `
        "--only-binary=:all:" `
        --upgrade `
        "rdkit==$RequiredRdkitDistribution"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install locked RDKit $RequiredRdkitDistribution in $CondaEnv."
    }
}

$TrackedChanges = @(git status --porcelain --untracked-files=no)
if ($LASTEXITCODE -ne 0) {
    throw "Cannot audit the Git worktree."
}
if ($TrackedChanges.Count -gt 0) {
    throw (
        "Tracked worktree changes would invalidate the sealed run: " +
        ($TrackedChanges -join ", ")
    )
}

Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class RacerC4ExecutionState {
    [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
"@
$ES_CONTINUOUS = [Convert]::ToUInt32("80000000", 16)
$ES_SYSTEM_REQUIRED = [uint32]0x00000001
$ES_DISPLAY_REQUIRED = [uint32]0x00000002
$KeepAwake = $ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED -bor $ES_DISPLAY_REQUIRED

Start-Transcript -LiteralPath $LogPath -Force | Out-Null
try {
    $State = [RacerC4ExecutionState]::SetThreadExecutionState($KeepAwake)
    if ($State -eq 0) {
        throw "Windows refused the process-scoped keep-awake request."
    }

    Write-Host "RACER-C4/TAME independent prospective pipeline" -ForegroundColor Green
    Write-Host "Repository: $RepoRoot"
    Write-Host "Conda environment: $CondaEnv"
    Write-Host "Endpoint scope: $Scope"
    Write-Host "Log: $LogPath"

    Write-Host "`n==> Locked RDKit environment preflight" -ForegroundColor Cyan
    $RdkitBefore = Get-RdkitRuntimeVersion
    if ($RdkitBefore -ne $RequiredRdkitRuntime) {
        $Observed = if ([string]::IsNullOrWhiteSpace($RdkitBefore)) {
            "unavailable"
        }
        else {
            $RdkitBefore
        }
        if ($SkipEnvironmentRepair) {
            throw (
                "RACER-C4 requires RDKit $RequiredRdkitRuntime, observed $Observed. " +
                "Rerun without -SkipEnvironmentRepair to install only the locked " +
                "RDKit wheel."
            )
        }
        Write-Host (
            "RDKit mismatch: required=$RequiredRdkitRuntime observed=$Observed."
        ) -ForegroundColor Yellow
        Install-LockedRdkit
    }
    $RdkitAfter = Get-RdkitRuntimeVersion
    if ($RdkitAfter -ne $RequiredRdkitRuntime) {
        throw (
            "Locked RDKit verification failed: required=$RequiredRdkitRuntime " +
            "observed=$RdkitAfter."
        )
    }
    Write-Host "RDKit runtime verified: $RdkitAfter" -ForegroundColor Green

    Write-Host "`n==> Contract and numerical tests" -ForegroundColor Cyan
    conda run --no-capture-output -n $CondaEnv python -m unittest discover `
        -s paper2_admet_benchmark\tests -p "test_*.py"
    if ($LASTEXITCODE -ne 0) {
        throw "Unit tests failed with exit code $LASTEXITCODE."
    }

    $Completed = $false
    for ($Pass = 1; $Pass -le $MaximumPasses; $Pass++) {
        Write-Host "`n==> Sealed pipeline pass $Pass/$MaximumPasses" -ForegroundColor Cyan
        conda run --no-capture-output -n $CondaEnv python -u `
            paper2_admet_benchmark\scripts\racer_c4\run_prospective_racer_c4.py `
            --mode full --scope $Scope
        $Code = $LASTEXITCODE
        if (Test-Path -LiteralPath $FinalReport -PathType Leaf) {
            $Report = Get-Content -LiteralPath $FinalReport -Raw | ConvertFrom-Json
            if (
                $Report.status -eq "complete_independent_final_epa_evaluation" -and
                $Report.predictions_sealed_before_final_labels -eq $true -and
                $Report.final_labels_opened_after_promotion -eq $true
            ) {
                $Completed = $true
                break
            }
        }
        if ($Pass -lt $MaximumPasses) {
            Write-Host (
                "Pass $Pass ended with code $Code. Locked downloads are retained; " +
                "retrying after 20 seconds."
            ) -ForegroundColor Yellow
            Start-Sleep -Seconds 20
        }
    }
    if (-not $Completed) {
        throw (
            "RACER-C4 did not reach a complete sealed final report. " +
            "Inspect $LogPath and .local\racer_c4_results\failure.json."
        )
    }

    Write-Host "`nRACER-C4 INDEPENDENT EVALUATION COMPLETE" -ForegroundColor Green
    Write-Host "Interpretation: $($Report.result_interpretation)"
    Write-Host "Final report: $FinalReport"
}
finally {
    [void][RacerC4ExecutionState]::SetThreadExecutionState($ES_CONTINUOUS)
    Stop-Transcript | Out-Null
}
