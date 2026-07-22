$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "[1/4] Cleaning stale LaTeX and BibTeX artifacts..."
latexmk -C main.tex | Out-Host
Remove-Item main.aux, main.bbl, main.blg, main.fdb_latexmk, main.fls, main.log, main.out -Force -ErrorAction SilentlyContinue

Write-Host "[2/4] Building manuscript with BibTeX..."
latexmk -pdf -bibtex -interaction=nonstopmode -halt-on-error main.tex | Out-Host

Write-Host "[3/4] Checking unresolved citations/references and hard LaTeX errors..."
$patterns = @(
    "LaTeX Error",
    "Undefined control sequence",
    "Citation.*undefined",
    "Reference.*undefined",
    "There were undefined references",
    "Please \(re\)run BibTeX"
)
$hits = Select-String -Path main.log -Pattern $patterns
if ($hits) {
    $hits | Format-Table -AutoSize | Out-Host
    throw "Build completed with unresolved citations/references or LaTeX errors."
}

Write-Host "[4/4] Checking visible question-mark citation artifacts in extracted PDF text..."
if (Get-Command pdftotext -ErrorAction SilentlyContinue) {
    pdftotext main.pdf main_extracted_check.txt
    $pdfHits = Select-String -Path main_extracted_check.txt -Pattern "\?\?|\d+\?"
    Remove-Item main_extracted_check.txt -Force -ErrorAction SilentlyContinue
    if ($pdfHits) {
        $pdfHits | Format-Table -AutoSize | Out-Host
        throw "Visible unresolved citation markers were found in the PDF."
    }
} else {
    Write-Host "pdftotext is unavailable; visual PDF citation check is still required."
}

Write-Host "CLEAN BUILD PASSED: main.pdf contains no log-level unresolved citations or references."
