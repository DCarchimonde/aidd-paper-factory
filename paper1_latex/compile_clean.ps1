$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$documents = @("main", "supplementary")

Write-Host "[1/6] Cleaning stale LaTeX and BibTeX artifacts..."
foreach ($doc in $documents) {
    latexmk -C "$doc.tex" | Out-Host
    Remove-Item "$doc.aux", "$doc.bbl", "$doc.blg", "$doc.fdb_latexmk", "$doc.fls", "$doc.log", "$doc.out" -Force -ErrorAction SilentlyContinue
}

Write-Host "[2/6] Building main manuscript with BibTeX..."
latexmk -pdf -bibtex -interaction=nonstopmode -halt-on-error main.tex | Out-Host

Write-Host "[3/6] Building Supporting Information..."
latexmk -pdf -interaction=nonstopmode -halt-on-error supplementary.tex | Out-Host

Write-Host "[4/6] Checking hard LaTeX, citation, reference, and overfull-box errors..."
$patterns = @(
    "LaTeX Error",
    "Undefined control sequence",
    "Citation.*undefined",
    "Reference.*undefined",
    "There were undefined references",
    "Please \(re\)run BibTeX",
    "Overfull \\hbox",
    "Overfull \\vbox"
)
foreach ($doc in $documents) {
    $hits = Select-String -Path "$doc.log" -Pattern $patterns
    if ($hits) {
        $hits | Format-Table -AutoSize | Out-Host
        throw "$doc.pdf contains unresolved citations/references, LaTeX errors, or overfull boxes."
    }
}

Write-Host "[5/6] Checking visible question-mark citation artifacts in extracted PDF text..."
if (Get-Command pdftotext -ErrorAction SilentlyContinue) {
    foreach ($doc in $documents) {
        $checkPath = "${doc}_extracted_check.txt"
        pdftotext "$doc.pdf" $checkPath
        $pdfHits = Select-String -Path $checkPath -Pattern "\?\?|\d+\?"
        Remove-Item $checkPath -Force -ErrorAction SilentlyContinue
        if ($pdfHits) {
            $pdfHits | Format-Table -AutoSize | Out-Host
            throw "Visible unresolved citation markers were found in $doc.pdf."
        }
    }
} else {
    Write-Host "pdftotext is unavailable; visual PDF citation checks remain mandatory."
}

Write-Host "[6/6] Checking the 25-page main-manuscript limit..."
if (Get-Command pdfinfo -ErrorAction SilentlyContinue) {
    $pagesLine = pdfinfo main.pdf | Select-String '^Pages:'
    $pages = [int](($pagesLine -split ':')[1].Trim())
    Write-Host "Main manuscript pages: $pages"
    if ($pages -gt 25) {
        throw "main.pdf exceeds the 25-page limit."
    }
} else {
    Write-Host "pdfinfo is unavailable; confirm manually that main.pdf is no more than 25 pages."
}

Write-Host "CLEAN BUILD PASSED: manuscript and Supporting Information contain no unresolved citations, references, overfull boxes, or visible question-mark markers."
