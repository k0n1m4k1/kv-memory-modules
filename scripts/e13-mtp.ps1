# E13 — characterize MTP speculation over RESTORED KV state (stock binaries).
#
# llama.cpp's MTP draft context shares KV cells with the target context, and all
# state ops on it silently no-op ([TAG_KV_CACHE_SHARE_CELLS]). Hypothesis: after
# a slot restore, the target KV is correct but the MTP head's KV for restored
# positions is uninitialized garbage -> draft acceptance should collapse while
# answers stay correct (the target model verifies every draft).
#
#   A) baseline: full prefill + MTP generation        -> normal acceptance
#   B) save slot after prefilling the memory
#   C) fresh server, restore slot, same question      -> acceptance over restored KV
#
# Usage: .\scripts\e13-mtp.ps1  (expects the MTP GGUF path below)

$ErrorActionPreference = 'Stop'

$root   = Split-Path $PSScriptRoot -Parent
$server = Join-Path $root 'bin\llama-server.exe'
$model  = Join-Path $root 'models\Qwopus3.5-4B-Coder-MTP-Q6_K.gguf'  # see scripts/models.txt
$slots  = Join-Path $root 'slots'
$logs   = Join-Path $root 'results\logs'
$port   = 8093
$base   = "http://127.0.0.1:$port"

New-Item -ItemType Directory -Force $logs | Out-Null

$commonArgs = @('-m', $model, '-ngl', '99', '-c', '8192', '-fa', 'off',
                '--spec-type', 'draft-mtp',
                '--slot-save-path', $slots, '--cache-ram', '0',
                '--host', '127.0.0.1', '--port', "$port", '-np', '1', '--no-webui')

$mem      = Get-Content (Join-Path $root 'data\memoria-agente.md') -Raw -Encoding UTF8
$question = "`n`n---`nPregunta: ¿cuál es la URL de staging, cuándo se refresca y qué bug intermitente hay relacionado con ese refresco? Responde en dos frases.`n`nRespuesta: "

function Start-Server([string]$tag) {
    $proc = Start-Process -FilePath $server -ArgumentList $commonArgs -PassThru -NoNewWindow `
        -RedirectStandardError (Join-Path $logs "e13-$tag.err.log") `
        -RedirectStandardOutput (Join-Path $logs "e13-$tag.out.log")
    $deadline = (Get-Date).AddSeconds(300)
    while ((Get-Date) -lt $deadline) {
        try {
            $h = Invoke-RestMethod "$base/health" -TimeoutSec 2
            if ($h.status -eq 'ok') { return $proc }
        } catch { Start-Sleep -Milliseconds 500 }
        if ($proc.HasExited) { throw "server ($tag) died on startup; see results\logs\e13-$tag.err.log" }
    }
    throw "timeout waiting for /health ($tag)"
}

function Stop-Server($proc) {
    if ($proc -and -not $proc.HasExited) { Stop-Process -Id $proc.Id -Force -Confirm:$false }
    Start-Sleep -Seconds 1
}

function Post-Json([string]$path, $obj, [int]$timeoutSec = 600) {
    $json = $obj | ConvertTo-Json -Depth 5
    return Invoke-RestMethod -Method Post -Uri "$base$path" -Body $json `
        -ContentType 'application/json; charset=utf-8' -TimeoutSec $timeoutSec
}

function Gen-Stats($c) {
    [ordered]@{
        prompt_n         = $c.timings.prompt_n
        predicted_n      = $c.timings.predicted_n
        gen_tps          = [math]::Round($c.timings.predicted_per_second, 1)
        draft_n          = $c.timings.draft_n
        draft_n_accepted = $c.timings.draft_n_accepted
        acceptance       = if ($c.timings.draft_n) { [math]::Round($c.timings.draft_n_accepted / $c.timings.draft_n, 3) } else { $null }
        answer           = $c.content.Trim()
    }
}

$r = [ordered]@{ model = Split-Path $model -Leaf }

# ---------- A: baseline, full prefill + MTP ----------
Write-Host '== A: baseline MTP (full prefill) =='
$p = Start-Server 'baseline'
try {
    $c = Post-Json '/completion' @{ prompt = ($mem + $question); n_predict = 120; cache_prompt = $false; temperature = 0 }
    $r.baseline = Gen-Stats $c
    # ---------- B: prefill memory only, save slot ----------
    Write-Host '== B: save slot =='
    $null = Post-Json '/completion' @{ prompt = $mem; n_predict = 1; cache_prompt = $true; temperature = 0 }
    $s = Post-Json '/slots/0?action=save' @{ filename = 'e13-mtp.bin' }
    $r.save = [ordered]@{ n_saved = $s.n_saved; file_MB = [math]::Round((Get-Item (Join-Path $slots 'e13-mtp.bin')).Length / 1MB, 1) }
} finally { Stop-Server $p }

# ---------- C: fresh server, restore, MTP over restored KV ----------
Write-Host '== C: restore + MTP =='
$p = Start-Server 'restore'
try {
    $s = Post-Json '/slots/0?action=restore' @{ filename = 'e13-mtp.bin' }
    $r.restore = [ordered]@{ n_restored = $s.n_restored }
    $c = Post-Json '/completion' @{ prompt = ($mem + $question); n_predict = 120; cache_prompt = $true; temperature = 0 }
    $r.restored = Gen-Stats $c
} finally { Stop-Server $p }

$out = Join-Path $root 'results\resultados-e13-mtp.json'
$r | ConvertTo-Json -Depth 5 | Set-Content $out -Encoding UTF8
Write-Host "`n== E13 =="
Write-Host ("baseline : acc {0}  ({1}/{2})  {3} t/s" -f $r.baseline.acceptance, $r.baseline.draft_n_accepted, $r.baseline.draft_n, $r.baseline.gen_tps)
Write-Host ("restored : acc {0}  ({1}/{2})  {3} t/s" -f $r.restored.acceptance, $r.restored.draft_n_accepted, $r.restored.draft_n, $r.restored.gen_tps)
Write-Host "results -> $out"
