# Build script: run .\build.ps1 inside the paper directory.
# Requires MiKTeX/TeX Live with ctex, fontspec, tikz and XeLaTeX.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "==> XeLaTeX pass 1"
xelatex -interaction=nonstopmode -halt-on-error main.tex
Write-Host "==> XeLaTeX pass 2 (resolve refs)"
xelatex -interaction=nonstopmode -halt-on-error main.tex

if (Test-Path "main.pdf") {
    Write-Host ""
    Write-Host "Build OK: $PSScriptRoot\main.pdf"
} else {
    Write-Error "Build failed, check main.log"
}
