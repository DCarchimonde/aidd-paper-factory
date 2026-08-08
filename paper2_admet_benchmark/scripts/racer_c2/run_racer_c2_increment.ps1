[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Set-Location $RepoRoot

$ExpectedBranch = "paper2-racer-c2-development-2026"
$SourceRoot = Join-Path $RepoRoot `
    "paper2_admet_benchmark\results\racer_c_confirmatory_v1"
$CellRoot = Join-Path $SourceRoot "cells"
$DevelopmentOutput = Join-Path $RepoRoot `
    "paper2_admet_benchmark\results\racer_c2_development\development_only_score_selection.json"
$OutputRoot = Join-Path $RepoRoot `
    "paper2_admet_benchmark\results\racer_c2_retrospective_extension_v0"
$SummaryPath = Join-Path $OutputRoot "run_summary.json"
$LogDir = Join-Path $RepoRoot "paper2_admet_benchmark\results\logs"
New-Item -ItemType Directory -Force $LogDir | Out-Null
$LogPath = Join-Path $LogDir (
    "racer_c2_increment_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss")
)

$ObservedBranch = (git branch --show-current).Trim()
if ($LASTEXITCODE -ne 0 -or $ObservedBranch -ne $ExpectedBranch) {
    throw "Run RACER-C2 from branch $ExpectedBranch; observed=$ObservedBranch"
}

if (-not (Test-Path -LiteralPath $CellRoot -PathType Container)) {
    throw "Completed RACER-C v1 cells are missing: $CellRoot"
}

$Cells = @(
    Get-ChildItem -LiteralPath $CellRoot -Directory |
    Where-Object {
        (Test-Path -LiteralPath (Join-Path $_.FullName "raw_predictions.csv")) -and
        (Test-Path -LiteralPath (Join-Path $_.FullName "test_predictions.csv")) -and
        (Test-Path -LiteralPath (Join-Path $_.FullName "final_manifest.json"))
    }
)
if ($Cells.Count -ne 60) {
    throw "Completed RACER-C v1 cell count is $($Cells.Count)/60"
}

$env:PYTHONUTF8 = "1"
$env:PYTHONHASHSEED = "0"

Start-Transcript -LiteralPath $LogPath -Force | Out-Null
try {
    Write-Host "RACER-C2 additive experiment" -ForegroundColor Green
    Write-Host "Existing v1 cells: 60/60"
    Write-Host "Base-model retraining: no"
    Write-Host "GPU required: no"
    Write-Host "Log: $LogPath"

    Write-Host "`nSTEP 1/3: Contract and regression tests" -ForegroundColor Cyan
    & python -m unittest discover `
        -s paper2_admet_benchmark/tests `
        -p "test_*.py" -v
    if ($LASTEXITCODE -ne 0) {
        throw "RACER-C2 tests failed with exit code $LASTEXITCODE"
    }

    Write-Host "`nSTEP 2/3: Reproduce development-only selection" -ForegroundColor Cyan
    & python `
        paper2_admet_benchmark/scripts/racer_c2/run_development_audit.py `
        --cell-root $CellRoot `
        --output $DevelopmentOutput
    if ($LASTEXITCODE -ne 0) {
        throw "RACER-C2 development selection failed with exit code $LASTEXITCODE"
    }
    $Development = Get-Content -LiteralPath $DevelopmentOutput -Raw |
        ConvertFrom-Json
    $Selected = $Development.selected_configuration
    if (
        $Development.status -ne "complete_development_only_score_selection" -or
        $Development.discovered_cell_count -ne 60 -or
        $Development.selection_cell_count -ne 40 -or
        $Development.evaluation_row_count -ne 720 -or
        [double]$Selected.t_max -ne 1.5 -or
        [double]$Selected.gamma_0 -ne 0.1 -or
        [double]$Selected.gamma_1 -ne -0.1 -or
        [double]$Selected.counterfactual_blend -ne 0.0 -or
        [bool]$Development.policy_labels_used -or
        [bool]$Development.conformal_labels_used -or
        [bool]$Development.test_labels_used
    ) {
        throw "RACER-C2 development selection contract failed"
    }

    Write-Host "`nSTEP 3/3: Add RACER-C2 to all 60 completed cells" `
        -ForegroundColor Cyan
    & python `
        paper2_admet_benchmark/scripts/racer_c2/run_retrospective_extension.py `
        --source $SourceRoot `
        --output-dir $OutputRoot `
        --resume
    if ($LASTEXITCODE -ne 0) {
        throw "RACER-C2 extension failed with exit code $LASTEXITCODE"
    }

    if (-not (Test-Path -LiteralPath $SummaryPath -PathType Leaf)) {
        throw "RACER-C2 summary is missing: $SummaryPath"
    }
    $Summary = Get-Content -LiteralPath $SummaryPath -Raw | ConvertFrom-Json
    if (
        $Summary.status -ne "complete_retrospective_racer_c2_extension" -or
        $Summary.source_cell_count -ne 60 -or
        $Summary.source_method_cell_count -ne 540 -or
        $Summary.method_count -ne 5 -or
        $Summary.policy_method_cell_count -ne 300 -or
        $Summary.test_method_cell_count -ne 300 -or
        $Summary.metric_row_count -ne 600 -or
        $Summary.base_models_retrained -ne $false -or
        $Summary.old_methods_rerun -ne $false -or
        $Summary.v1_source_modified -ne $false -or
        $Summary.confirmatory_claim_authorized -ne $false
    ) {
        throw "RACER-C2 final summary failed the completeness/scope contract"
    }

    Write-Host "`nRACER-C2 ADDITIVE EXPERIMENT COMPLETE" -ForegroundColor Green
    Write-Host "Existing v1 cells reused: 60/60"
    Write-Host "New C2 methods/ablations: 5"
    Write-Host "New test method-cell results: 300/300"
    Write-Host "Base models retrained: false"
    Write-Host "Old 540 results rerun: false"
    Write-Host "v1 source modified: false"
    Write-Host "Summary: $SummaryPath"
}
finally {
    Stop-Transcript | Out-Null
}

