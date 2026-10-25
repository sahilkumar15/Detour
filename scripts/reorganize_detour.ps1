[CmdletBinding()]
param([switch]$ArchiveLegacy)
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'
$root = (Get-Location).Path
if (-not (Test-Path -LiteralPath (Join-Path $root ".git"))) { throw 'Run inside Detour.' }
Set-Location $root.Trim()
function Move-ResearchFile([string]$from,[string]$to) {
    if (-not (Test-Path -LiteralPath $from -PathType Leaf)) { return }
    if (Test-Path -LiteralPath $to) { throw "Destination exists (not overwritten): $to" }
    $folder=Split-Path $to -Parent
    New-Item -ItemType Directory -Path $folder -Force | Out-Null
    $tracked=@(& git ls-files -- $from)
    if (($tracked -join '').Trim()) {
        & git mv -- $from $to
        if ($LASTEXITCODE -ne 0) { throw "git mv failed for $from" }
    } else { Move-Item -LiteralPath $from -Destination $to }
    Write-Host "MOVED $from => $to" -ForegroundColor Green
}
Get-ChildItem -File -Filter 'Modelfile*' | ForEach-Object {
    Move-ResearchFile $_.Name ("configs/models/"+$_.Name)
}
if ($ArchiveLegacy) {
    Get-ChildItem -Path 'results' -File | ForEach-Object {
        if ($_.Name -match '^(grid-v7-qwen7|grid-v7-qwen14|clean-v7-qwen7|clean-v7-qwen14)\.') { return }
        $bucket='legacy'
        if ($_.Name -match '^grid-v([2-6])-') { $bucket='v'+$Matches[1] }
        elseif ($_.Name -match '^(dry|mock|smoke|selftest|validation-table)') { $bucket='diagnostics' }
        Move-ResearchFile ("results/"+$_.Name) ("results/archive/$bucket/"+$_.Name)
    }
}
