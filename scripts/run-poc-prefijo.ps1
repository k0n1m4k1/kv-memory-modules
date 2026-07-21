# PoC — escenario real: system prompt + parte variable (fecha/hora) + memoria MD.
# Mide qué pasa HOY (semántica solo-prefijo) cuando el módulo se compiló sin ese prefijo delante.

$ErrorActionPreference = 'Stop'

$root   = Split-Path $PSScriptRoot -Parent
$server = Join-Path $root 'bin\llama-server.exe'
$model  = Join-Path $root 'models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf'
$slots  = Join-Path $root 'slots'
$logs   = Join-Path $root 'results\logs'
$port   = 8090
$base   = "http://127.0.0.1:$port"

$commonArgs = @('-m', $model, '-ngl', '99', '-c', '8192', '-fa', 'off',
                '--slot-save-path', $slots, '--cache-ram', '0',
                '--host', '127.0.0.1', '--port', "$port", '-np', '1', '--no-webui')

$mem      = Get-Content (Join-Path $root 'data\memoria-agente.md') -Raw -Encoding UTF8
$system   = "Eres un asistente de ingeniería. Responde de forma breve y precisa.`n"
$fecha    = "Fecha y hora actuales: sábado 19 de julio de 2026, 14:30 CET.`n`n"
$question = "`n`n---`nPregunta: ¿cuántos días faltan para el próximo refresco de datos de staging y a qué hora ocurre? Responde en una frase.`n`nRespuesta: "

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
        if ($proc.HasExited) { throw "El servidor ($tag) murió al arrancar" }
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

# El módulo memoria.bin ya existe (compilado con la memoria sola, posiciones 0..n).
# Restaurarlo y lanzar la conversación real: system + fecha + memoria + pregunta.
Write-Host "== Restaurar módulo y consultar con prefijo variable delante =="
$p1 = Start-Server 'prefijo'
try {
    $s = Post-Json '/slots/0?action=restore' @{ filename = 'memoria.bin' }
    $r.restore = [ordered]@{ n_restored = $s.n_restored; restore_ms = [math]::Round($s.timings.restore_ms, 1) }

    $c = Post-Json '/completion' @{ prompt = ($system + $fecha + $mem + $question); n_predict = 60; cache_prompt = $true; temperature = 0 }
    $r.query_con_prefijo = [ordered]@{
        prompt_n  = $c.timings.prompt_n     # si sale ~todo el prompt, el módulo no sirvió de nada
        prompt_ms = [math]::Round($c.timings.prompt_ms, 1)
        answer    = $c.content.Trim()
    }
} finally { Stop-Server $p1 }

$r | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $root 'results\resultados-prefijo.json') -Encoding UTF8
Write-Host "`n== RESULTADOS =="
$r | ConvertTo-Json -Depth 5
