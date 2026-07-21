# Environment bootstrap for the vm-llm-mem experiment suite (Windows / Vulkan).
#
#   1. Creates a Python venv at .\venv with the runtime deps.
#   2. Unpacks the prebuilt llama.cpp Vulkan binaries into bin\ (drop the
#      official llama-<tag>-bin-win-vulkan-x64.zip into third_party\ first,
#      or keep the one already there).
#   3. Models are downloaded with huggingface-cli from scripts\models.txt.

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent

Write-Host "== 1/3 Python venv =="
if (-not (Test-Path "$root\venv")) { python -m venv "$root\venv" }
& "$root\venv\Scripts\pip" install --upgrade pip numpy huggingface_hub

Write-Host "== 2/3 llama.cpp Vulkan binaries =="
$zip = Get-ChildItem "$root\third_party\*vulkan*.zip" | Select-Object -First 1
if ($zip) {
    New-Item -ItemType Directory -Force "$root\bin" | Out-Null
    Expand-Archive $zip.FullName -DestinationPath "$root\bin" -Force
    Write-Host "   unpacked $($zip.Name) -> bin\"
} else {
    Write-Warning "no third_party\*vulkan*.zip found - download a llama.cpp win-vulkan release zip first"
}

Write-Host "== 3/3 Models =="
Get-Content "$root\scripts\models.txt" | Where-Object { $_ -notmatch '^\s*(#|$)' } | ForEach-Object {
    $repo, $file = $_ -split '\s+', 2
    if (Test-Path "$root\models\$file") {
        Write-Host "   $file already present, skipping"
    } else {
        & "$root\venv\Scripts\huggingface-cli" download $repo $file --local-dir "$root\models"
    }
}

Write-Host "Done. Try: venv\Scripts\python experiments\bateria6.py models\<model>.gguf <tag>"
