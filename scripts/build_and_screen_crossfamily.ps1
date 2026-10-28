[CmdletBinding()]
param([switch]$BuildModels,[switch]$RunClean)
$ErrorActionPreference='Stop'
$root=(& git rev-parse --show-toplevel | Select-Object -First 1).Trim(); Set-Location $root
if ($BuildModels) {
    foreach ($x in @(@('hermes3-detour','Modelfile.hermes3'),@('granite3-detour','Modelfile.granite3'))) {
        $file=Join-Path 'configs/models' $x[1]
        if (-not (Test-Path $file)) { throw "Missing $file" }
        & ollama create $x[0] -f $file
        if ($LASTEXITCODE -ne 0) { throw "Failed: $($x[0])" }
    }
}
if ($RunClean) {
    & python run_audit.py --clean --agents 'ollama:hermes3-detour:latest,ollama:granite3-detour:latest' --reps 2 --max-steps 6 --max-tokens 900 --out results/clean-screen-crossfamily-v7
    if ($LASTEXITCODE -ne 0) { throw 'Clean screen failed' }
    Write-Host 'Review clean-task competence before running full cross-family grids.'
}
if (-not $BuildModels -and -not $RunClean) { Write-Host 'Use -BuildModels and/or -RunClean explicitly.' }
