[CmdletBinding()]
param(
    [ValidateRange(1, 5)]
    [int]$MaximumPasses = 3
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Set-Location $RepoRoot

$ExpectedTag = "paper2-racer-protocol-freeze-v1.0"
$SummaryPath = Join-Path $RepoRoot `
    "paper2_admet_benchmark\results\racer_c_confirmatory_v1\run_summary.json"
$RegistryPath = Join-Path $RepoRoot `
    "paper2_admet_benchmark\results\racer_c_confirmatory_v1\run_registry.json"
$LogDir = Join-Path $RepoRoot "paper2_admet_benchmark\results\logs"
New-Item -ItemType Directory -Force $LogDir | Out-Null
$LogPath = Join-Path $LogDir (
    "racer_c_overnight_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss")
)

if ($env:CONDA_DEFAULT_ENV -ne "racer_c_gpu") {
    throw "Activate the racer_c_gpu Conda environment before the overnight run."
}

$Head = (git rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Cannot resolve the checked-out Git commit."
}
$Tags = @(git tag --points-at HEAD)
if ($LASTEXITCODE -ne 0 -or $Tags -notcontains $ExpectedTag) {
    throw "HEAD $Head is not tagged $ExpectedTag; confirmatory predictions are blocked."
}

# Prevent Windows display/system sleep for this process lifetime. The prior power
# plan is not modified. ES_CONTINUOUS alone in finally releases the request.
Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class RacerExecutionState {
    [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
"@
$ES_CONTINUOUS = [uint32]0x80000000
$ES_SYSTEM_REQUIRED = [uint32]0x00000001
$ES_DISPLAY_REQUIRED = [uint32]0x00000002
$KeepAwake = $ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED -bor $ES_DISPLAY_REQUIRED

Start-Transcript -LiteralPath $LogPath -Force | Out-Null
try {
    $State = [RacerExecutionState]::SetThreadExecutionState($KeepAwake)
    if ($State -eq 0) {
        throw "Windows refused the process-scoped keep-awake request."
    }

    Write-Host "RACER-C v1.0 overnight confirmatory run" -ForegroundColor Green
    Write-Host "Frozen commit: $Head"
    Write-Host "Protocol tag: $ExpectedTag"
    Write-Host "Log: $LogPath"

    $Completed = $false
    for ($Pass = 1; $Pass -le $MaximumPasses; $Pass++) {
        Write-Host "`n==> Resume pass $Pass of $MaximumPasses" -ForegroundColor Cyan
        powershell -NoProfile -ExecutionPolicy Bypass -File `
            paper2_admet_benchmark\scripts\racer_c\run_racer_c_pipeline.ps1 `
            -Mode Full
        $Code = $LASTEXITCODE

        if (Test-Path -LiteralPath $SummaryPath -PathType Leaf) {
            $Summary = Get-Content -LiteralPath $SummaryPath -Raw | ConvertFrom-Json
            if (
                $Summary.status -eq "complete_confirmatory_primary_study" -and
                $Summary.primary_cell_count -eq 60 -and
                $Summary.method_cell_count -eq 540 -and
                $Summary.failed_cell_count -eq 0
            ) {
                $Completed = $true
                break
            }
        }

        if ($Pass -lt $MaximumPasses) {
            Write-Host (
                "Pass $Pass ended with code $Code; retained artifacts will be " +
                "hash-checked and resumed after 30 seconds."
            ) -ForegroundColor Yellow
            Start-Sleep -Seconds 30
        }
    }

    if (-not $Completed) {
        if (Test-Path -LiteralPath $RegistryPath -PathType Leaf) {
            $Registry = Get-Content -LiteralPath $RegistryPath -Raw | ConvertFrom-Json
            Write-Host "Final registry status: $($Registry.status)" -ForegroundColor Red
            foreach ($Failure in @($Registry.failures)) {
                Write-Host (
                    "FAILED {0} stage={1} attempt={2}: {3}" -f `
                    $Failure.cell, $Failure.stage, $Failure.attempt, $Failure.error
                ) -ForegroundColor Red
            }
        }
        throw (
            "RACER-C remains incomplete after $MaximumPasses safe resume passes. " +
            "Completed cells were retained; do not delete the result directory."
        )
    }

    Write-Host "`nRACER-C CONFIRMATORY PRIMARY STUDY COMPLETE" -ForegroundColor Green
    Write-Host "Primary cells: 60/60"
    Write-Host "Method-cell results: 540/540"
    Write-Host "Failed cells: 0"
    Write-Host "Summary: $SummaryPath"
}
finally {
    [void][RacerExecutionState]::SetThreadExecutionState($ES_CONTINUOUS)
    Stop-Transcript | Out-Null
}
