$ErrorActionPreference = "Stop"

$LatexDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $LatexDir

Write-Host "============================================================"
Write-Host "PAPER 1 V3 MANUSCRIPT BUILD"
Write-Host "Repo: $RepoRoot"
Write-Host "============================================================"

Set-Location $RepoRoot

Write-Host "`n[1/4] Building manuscript figures from authoritative v3 tables..."
python paper1_leakage_benchmark\scripts\16_build_manuscript_assets_v3.py
if ($LASTEXITCODE -ne 0) { throw "Manuscript asset build failed." }

Write-Host "`n[2/4] Running manuscript consistency audit..."
python paper1_leakage_benchmark\scripts\17_audit_manuscript_v3.py
if ($LASTEXITCODE -ne 0) { throw "Manuscript audit failed." }

Set-Location $LatexDir

Write-Host "`n[3/4] Compiling main manuscript..."
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
if ($LASTEXITCODE -ne 0) { throw "Main manuscript LaTeX build failed." }

Write-Host "`n[4/4] Compiling Supporting Information..."
latexmk -pdf -interaction=nonstopmode -halt-on-error supplementary.tex
if ($LASTEXITCODE -ne 0) { throw "Supporting Information LaTeX build failed." }

if (-not (Test-Path "main.pdf")) { throw "main.pdf was not created." }
if (-not (Test-Path "supplementary.pdf")) { throw "supplementary.pdf was not created." }

Write-Host "`n============================================================"
Write-Host "PAPER 1 V3 MANUSCRIPT BUILD PASSED"
Write-Host "Main PDF: $LatexDir\main.pdf"
Write-Host "SI PDF:   $LatexDir\supplementary.pdf"
Write-Host "============================================================"
