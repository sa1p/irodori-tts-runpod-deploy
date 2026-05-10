param(
    [string]$BaseUrl = "http://127.0.0.1:8010",
    [string]$Registry = "configs\model_registry.smoke.json",
    [string]$OutputDir = "",
    [string]$Text,
    [string]$ModelCsv = "",
    [ValidateSet("wav", "mp3")]
    [string]$Format = "wav",
    [int]$NumSteps = 32,
    [int]$Seed = 20260419,
    [double]$Seconds = 30.0,
    [bool]$AutoSplit = $true,
    [int]$MaxChunkChars = 100,
    [double]$ChunkSeconds = 0.0,
    [int]$ChunkSilenceMs = 350,
    [int]$ChunkTailPaddingMs = 250,
    [bool]$TrimTail = $true,
    [double]$CfgScaleText = 3.0,
    [double]$CfgScaleSpeaker = 5.0,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

Set-Location (Resolve-Path (Join-Path $PSScriptRoot ".."))

if ([string]::IsNullOrWhiteSpace($Text)) {
    $Text = "これは長めの品質確認用サンプルです。落ち着いた会話の調子で、声の高さ、息の入り方、語尾の自然さ、文章の途中での間の取り方を確認しています。最後まで聞いたときに声質が崩れず、聞き取りやすく自然に届いていれば成功です。"
}

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputDir = "outputs\model_smoke_samples_$stamp"
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$manifest = Join-Path $OutputDir "manifest.jsonl"
$summary = Join-Path $OutputDir "summary.json"
$promptPath = Join-Path $OutputDir "prompt.txt"

if ($Overwrite -and (Test-Path $manifest)) {
    Remove-Item -LiteralPath $manifest -Force
}

$Text | Set-Content -LiteralPath $promptPath -Encoding UTF8

$registryObj = Get-Content -LiteralPath $Registry -Raw -Encoding UTF8 | ConvertFrom-Json
$models = @($registryObj.models | Where-Object { $_.enabled -eq $true })
if (-not [string]::IsNullOrWhiteSpace($ModelCsv)) {
    $wanted = @{}
    foreach ($raw in $ModelCsv.Split(",")) {
        $id = $raw.Trim()
        if ($id) {
            $wanted[$id] = $true
        }
    }
    $models = @($models | Where-Object { $wanted.ContainsKey([string]$_.id) })
}

if ($models.Count -eq 0) {
    throw "No enabled models selected from registry: $Registry"
}

$results = @()
Write-Host "models=$($models.Count) output=$OutputDir format=$Format num_steps=$NumSteps"

foreach ($model in $models) {
    $modelId = [string]$model.id
    $safeName = $modelId -replace '[^A-Za-z0-9_.-]', '_'
    $outFile = Join-Path $OutputDir "$safeName.$Format"
    $started = Get-Date
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $row = [ordered]@{
        model_id = $modelId
        output = (Resolve-Path $OutputDir).Path + "\" + "$safeName.$Format"
        status = "running"
        started_at = $started.ToString("o")
        elapsed_sec = $null
        size_bytes = $null
        duration_sec = $null
        sample_rate = $null
        channels = $null
        error = $null
    }

    try {
        if ((Test-Path $outFile) -and -not $Overwrite) {
            Write-Host "skip existing $modelId -> $outFile"
            $row.status = "skipped"
        } else {
            $bodyPayload = @{
                model_id = $modelId
                text = $Text
                format = $Format
                seed = $Seed
                num_steps = $NumSteps
                cfg_scale_text = $CfgScaleText
                cfg_scale_speaker = $CfgScaleSpeaker
                seconds = $Seconds
                auto_split = $AutoSplit
                max_chunk_chars = $MaxChunkChars
                chunk_silence_ms = $ChunkSilenceMs
                chunk_tail_padding_ms = $ChunkTailPaddingMs
                trim_tail = $TrimTail
            }
            if ($ChunkSeconds -gt 0) {
                $bodyPayload.chunk_seconds = $ChunkSeconds
            }
            $body = $bodyPayload | ConvertTo-Json -Depth 5

            Write-Host "generate $modelId -> $outFile"
            Invoke-WebRequest `
                -Uri "$($BaseUrl.TrimEnd('/'))/v1/tts" `
                -Method Post `
                -Body $body `
                -ContentType "application/json; charset=utf-8" `
                -OutFile $outFile `
                -TimeoutSec 1200 | Out-Null
            $row.status = "ok"
        }

        if (Test-Path $outFile) {
            $item = Get-Item -LiteralPath $outFile
            $row.size_bytes = $item.Length
            $ffprobe = Get-Command ffprobe -ErrorAction SilentlyContinue
            if ($ffprobe) {
                $probeRaw = & ffprobe -v error -show_entries format=duration -show_entries stream=sample_rate,channels -of json $outFile
                if ($probeRaw) {
                    $probe = $probeRaw | ConvertFrom-Json
                    if ($probe.format.duration) {
                        $row.duration_sec = [Math]::Round([double]$probe.format.duration, 3)
                    }
                    if ($probe.streams.Count -gt 0) {
                        $row.sample_rate = $probe.streams[0].sample_rate
                        $row.channels = $probe.streams[0].channels
                    }
                }
            }
        }
    } catch {
        $row.status = "failed"
        $row.error = $_.Exception.Message
        Write-Host "failed ${modelId}: $($row.error)"
    } finally {
        $sw.Stop()
        $row.elapsed_sec = [Math]::Round($sw.Elapsed.TotalSeconds, 3)
        $results += [pscustomobject]$row
        ([pscustomobject]$row | ConvertTo-Json -Compress -Depth 8) | Add-Content -LiteralPath $manifest -Encoding UTF8
    }
}

$payload = [ordered]@{
    generated_at = (Get-Date).ToString("o")
    base_url = $BaseUrl
    registry = (Resolve-Path $Registry).Path
    output_dir = (Resolve-Path $OutputDir).Path
    text = $Text
    format = $Format
    num_steps = $NumSteps
    seed = $Seed
    seconds = $Seconds
    auto_split = $AutoSplit
    max_chunk_chars = $MaxChunkChars
    chunk_seconds = $ChunkSeconds
    chunk_silence_ms = $ChunkSilenceMs
    chunk_tail_padding_ms = $ChunkTailPaddingMs
    trim_tail = $TrimTail
    total = $results.Count
    ok = @($results | Where-Object { $_.status -eq "ok" }).Count
    failed = @($results | Where-Object { $_.status -eq "failed" }).Count
    skipped = @($results | Where-Object { $_.status -eq "skipped" }).Count
    results = $results
}

$payload | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $summary -Encoding UTF8
Write-Host "summary=$summary"
Write-Host "manifest=$manifest"
