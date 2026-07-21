# PoC extra — portabilidad de backend: restaurar en CPU (-ngl 0) el módulo compilado en Vulkan/GPU.

$ErrorActionPreference = 'Stop'

$root   = Split-Path $PSScriptRoot -Parent
$server = Join-Path $root 'bin\llama-server.exe'
$model  = Join-Path $root 'models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf'
$slots  = Join-Path $root 'slots'
$logs   = Join-Path $root 'results\logs'
$port   = 8090
$base   = "http://127.0.0.1:$port"

$commonArgs = @('-m', $model, '-ngl', '0', '-c', '8192', '-fa', 'off',
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

function Post-Json([string]$path, $obj, [int]$timeoutSec = 600) {
    $json = $obj | ConvertTo-Json -Depth 5
    return Invoke-RestMethod -Method Post -Uri "$base$path" -Body $json `
        -ContentType 'application/json; charset=utf-8' -TimeoutSec $timeoutSec
}

$r = [ordered]@{}

Write-Host "== CPU: restaurar módulo compilado en Vulkan =="
$p1 = Start-Server 'cpu-restore'
try {
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $s = Post-Json '/slots/0?action=restore' @{ filename = 'memoria.bin' }
    $sw.Stop()
    $r.cpu_restore = [ordered]@{
        n_restored = $s.n_restored
        restore_ms = if ($s.timings.restore_ms) { [math]::Round($s.timings.restore_ms, 1) } else { $null }
        wall_ms    = $sw.ElapsedMilliseconds
    }
    $sw.Restart()
    $c = Post-Json '/completion' @{ prompt = ($mem + $question); n_predict = 60; cache_prompt = $true; temperature = 0 }
    $sw.Stop()
    $r.cpu_query_restored = [ordered]@{
        prompt_n  = $c.timings.prompt_n
        prompt_ms = [math]::Round($c.timings.prompt_ms, 1)
        wall_ms   = $sw.ElapsedMilliseconds
        answer    = $c.content.Trim()
    }
} finally { Stop-Server $p1 }

Write-Host "== CPU: línea base sin restaurar =="
$p2 = Start-Server 'cpu-baseline'
try {
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $c = Post-Json '/completion' @{ prompt = ($mem + $question); n_predict = 60; cache_prompt = $true; temperature = 0 }
    $sw.Stop()
    $r.cpu_query_baseline = [ordered]@{
        prompt_n  = $c.timings.prompt_n
        prompt_ms = [math]::Round($c.timings.prompt_ms, 1)
        wall_ms   = $sw.ElapsedMilliseconds
        answer    = $c.content.Trim()
    }
} finally { Stop-Server $p2 }

$r | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $root 'results\resultados-cpu.json') -Encoding UTF8
Write-Host "`n== RESULTADOS CPU =="
$r | ConvertTo-Json -Depth 5
