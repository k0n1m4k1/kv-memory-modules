# PoC Fase A — "precompilar" una memoria MD a estado KV y restaurarla en frío sin re-prefill.
# Tres pasadas: (1) compilar+guardar, (2) restaurar en frío, (3) línea base en frío sin restaurar.

$ErrorActionPreference = 'Stop'

$root   = Split-Path $PSScriptRoot -Parent
$server = Join-Path $root 'bin\llama-server.exe'
$model  = Join-Path $root 'models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf'
$slots  = Join-Path $root 'slots'
$logs   = Join-Path $root 'results\logs'
$port   = 8090
$base   = "http://127.0.0.1:$port"

New-Item -ItemType Directory -Force $logs | Out-Null

# -fa off fijado a propósito: v_trans depende de flash-attn y con 'auto' el layout del
# estado guardado podría variar entre backends/GPUs, rompiendo la compatibilidad del módulo.
$commonArgs = @('-m', $model, '-ngl', '99', '-c', '8192', '-fa', 'off',
                '--slot-save-path', $slots, '--cache-ram', '0',
                '--host', '127.0.0.1', '--port', "$port", '-np', '1', '--no-webui')

$mem      = Get-Content (Join-Path $root 'data\memoria-agente.md') -Raw -Encoding UTF8
$question = "`n`n---`nPregunta: ¿cuál es la URL de staging, cuándo se refresca y qué bug intermitente hay relacionado con ese refresco? Responde en dos frases.`n`nRespuesta: "

function Start-Server([string]$tag) {
    $proc = Start-Process -FilePath $server -ArgumentList $commonArgs -PassThru -NoNewWindow `
        -RedirectStandardError (Join-Path $logs "server-$tag.err.log") `
        -RedirectStandardOutput (Join-Path $logs "server-$tag.out.log")
    $deadline = (Get-Date).AddSeconds(300)
    while ((Get-Date) -lt $deadline) {
        try {
            $h = Invoke-RestMethod "$base/health" -TimeoutSec 2
            if ($h.status -eq 'ok') { return $proc }
        } catch { Start-Sleep -Milliseconds 500 }
        if ($proc.HasExited) { throw "El servidor ($tag) murió al arrancar; revisa logs\server-$tag.err.log" }
    }
    throw "Timeout esperando /health ($tag)"
}

function Stop-Server($proc) {
    if ($proc -and -not $proc.HasExited) { Stop-Process -Id $proc.Id -Force -Confirm:$false }
    Start-Sleep -Seconds 1
}

function Post-Json([string]$path, $obj, [int]$timeoutSec = 300) {
    $json = $obj | ConvertTo-Json -Depth 5
    return Invoke-RestMethod -Method Post -Uri "$base$path" -Body $json `
        -ContentType 'application/json; charset=utf-8' -TimeoutSec $timeoutSec
}

$r = [ordered]@{}

# ---------- PASADA 1: compilación (prefill) + guardado del módulo ----------
Write-Host "== PASADA 1: compilar memoria (prefill) y guardar módulo =="
$p1 = Start-Server 'compile'
try {
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $c1 = Post-Json '/completion' @{ prompt = $mem; n_predict = 1; cache_prompt = $true; temperature = 0 }
    $sw.Stop()
    $r.compile = [ordered]@{
        prompt_n   = $c1.timings.prompt_n
        prompt_ms  = [math]::Round($c1.timings.prompt_ms, 1)
        pp_tps     = [math]::Round($c1.timings.prompt_per_second, 1)
        wall_ms    = $sw.ElapsedMilliseconds
    }
    $sw.Restart()
    $s1 = Post-Json '/slots/0?action=save' @{ filename = 'memoria.bin' }
    $sw.Stop()
    $r.save = [ordered]@{
        n_saved    = $s1.n_saved
        n_written  = $s1.n_written
        save_ms    = if ($s1.timings.save_ms) { [math]::Round($s1.timings.save_ms, 1) } else { $null }
        wall_ms    = $sw.ElapsedMilliseconds
        file_MB    = [math]::Round((Get-Item (Join-Path $slots 'memoria.bin')).Length / 1MB, 1)
    }
} finally { Stop-Server $p1 }

# ---------- PASADA 2: arranque frío + restauración del módulo ----------
Write-Host "== PASADA 2: arranque frío, restaurar módulo, preguntar =="
$p2 = Start-Server 'restore'
try {
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $s2 = Post-Json '/slots/0?action=restore' @{ filename = 'memoria.bin' }
    $sw.Stop()
    $r.restore = [ordered]@{
        n_restored = $s2.n_restored
        n_read     = $s2.n_read
        restore_ms = if ($s2.timings.restore_ms) { [math]::Round($s2.timings.restore_ms, 1) } else { $null }
        wall_ms    = $sw.ElapsedMilliseconds
    }
    $sw.Restart()
    $c2 = Post-Json '/completion' @{ prompt = ($mem + $question); n_predict = 80; cache_prompt = $true; temperature = 0 }
    $sw.Stop()
    $r.query_restored = [ordered]@{
        prompt_n   = $c2.timings.prompt_n      # <- debería ser ~tokens de la pregunta, no de toda la memoria
        prompt_ms  = [math]::Round($c2.timings.prompt_ms, 1)
        wall_ms    = $sw.ElapsedMilliseconds
        answer     = $c2.content.Trim()
    }
} finally { Stop-Server $p2 }

# ---------- PASADA 3: línea base, arranque frío SIN restaurar ----------
Write-Host "== PASADA 3: arranque frío sin restaurar (línea base) =="
$p3 = Start-Server 'baseline'
try {
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $c3 = Post-Json '/completion' @{ prompt = ($mem + $question); n_predict = 80; cache_prompt = $true; temperature = 0 }
    $sw.Stop()
    $r.query_baseline = [ordered]@{
        prompt_n   = $c3.timings.prompt_n
        prompt_ms  = [math]::Round($c3.timings.prompt_ms, 1)
        wall_ms    = $sw.ElapsedMilliseconds
        answer     = $c3.content.Trim()
    }
} finally { Stop-Server $p3 }

$r | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $root 'results\resultados.json') -Encoding UTF8
Write-Host "`n== RESULTADOS =="
$r | ConvertTo-Json -Depth 5
