param(
    [string]$ProductionLogDir = "",
    [string]$Device = "cuda",
    [int]$MaxSteps = 1000,
    [int]$PollSeconds = 300,
    [switch]$StartApiWhenDone
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

Set-Location (Resolve-Path (Join-Path $PSScriptRoot ".."))

if ([string]::IsNullOrWhiteSpace($ProductionLogDir)) {
    $ProductionLogDir = Get-ChildItem -LiteralPath "logs" -Directory -Filter "production_pipeline_*" |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1 -ExpandProperty FullName
}
if ([string]::IsNullOrWhiteSpace($ProductionLogDir) -or -not (Test-Path -LiteralPath $ProductionLogDir)) {
    throw "ProductionLogDir not found: $ProductionLogDir"
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogDir = "logs\aina_lora_$stamp"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$logPath = Join-Path $LogDir "run.log"

function Write-Log {
    param([string]$Message)
    "[$(Get-Date -Format o)] $Message" | Tee-Object -FilePath $logPath -Append
}

function Invoke-Step {
    param([string]$Name, [string[]]$ArgsList)
    Write-Log $Name
    Write-Log ".venv\Scripts\python.exe $($ArgsList -join ' ')"
    $stepName = $Name -replace "[^A-Za-z0-9_.-]", "_"
    $stdoutPath = Join-Path $LogDir "$stepName.stdout.tmp"
    $stderrPath = Join-Path $LogDir "$stepName.stderr.tmp"
    $proc = Start-Process -FilePath ".\.venv\Scripts\python.exe" `
        -ArgumentList $ArgsList `
        -WorkingDirectory (Get-Location) `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath `
        -NoNewWindow `
        -Wait `
        -PassThru
    if (Test-Path -LiteralPath $stdoutPath) {
        Get-Content -LiteralPath $stdoutPath | Tee-Object -FilePath $logPath -Append
        Remove-Item -LiteralPath $stdoutPath -Force -ErrorAction SilentlyContinue
    }
    if (Test-Path -LiteralPath $stderrPath) {
        Get-Content -LiteralPath $stderrPath | Tee-Object -FilePath $logPath -Append
        Remove-Item -LiteralPath $stderrPath -Force -ErrorAction SilentlyContinue
    }
    if ($proc.ExitCode -ne 0) {
        throw "$Name failed with exit code $($proc.ExitCode)"
    }
}

function Stop-IrodoriApi {
    $targets = Get-CimInstance Win32_Process | Where-Object {
        ($_.Name -eq "uvicorn.exe" -or $_.Name -eq "python.exe") -and
        $_.CommandLine -like "*api_server:app*"
    }
    foreach ($proc in $targets) {
        Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
    }
}

function Get-ProductionStatus {
    $inventoryPath = Join-Path $ProductionLogDir "inventory.json"
    if (-not (Test-Path -LiteralPath $inventoryPath)) {
        throw "production inventory not found: $inventoryPath"
    }
    $payload = & ".\.venv\Scripts\python.exe" "scripts\pipeline_status.py" "--inventory" $inventoryPath "--json" |
        ConvertFrom-Json
    return $payload.summary
}

Write-Log "waiting for production pipeline: $ProductionLogDir"
while ($true) {
    $statePath = Join-Path $ProductionLogDir "state.json"
    $state = $null
    if (Test-Path -LiteralPath $statePath) {
        $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
        if ($state.status -eq "failed") {
            throw "production pipeline failed; aborting Aina training"
        }
    }
    $summary = Get-ProductionStatus
    Write-Log "production merged=$($summary.merged)/$($summary.total) data_ready=$($summary.data_ready)/$($summary.total) status=$($state.status)"
    if ($state -and $state.status -eq "ok") {
        break
    }
    if (-not $state -and $summary.total -gt 0 -and $summary.merged -ge $summary.total) {
        break
    }
    Start-Sleep -Seconds $PollSeconds
}

Stop-IrodoriApi

Invoke-Step "prepare_aina_dataset" @(
    "scripts\prepare_aina_dataset.py"
)

Invoke-Step "prepare_aina_manifest" @(
    "scripts\prepare_local_manifest.py",
    "--input-jsonl", "data\irodori_training\aina001\train.jsonl",
    "--output-manifest", "data\irodori_training\aina001\train_manifest.jsonl",
    "--latent-dir", "data\irodori_training\aina001\latents",
    "--speaker-id-prefix", "aina001",
    "--device", $Device,
    "--target-sample-rate", "48000"
)

$checkpointDir = "outputs\irodori_loras\aina001\checkpoint"
$resume = Get-ChildItem -LiteralPath $checkpointDir -Directory -Filter "checkpoint_*" -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match "^checkpoint_\d+$" } |
    Sort-Object { [int]($_.Name -replace "^checkpoint_", "") } -Descending |
    Select-Object -First 1

$trainArgs = @(
    "train.py",
    "--config", "configs\train_500m_v3_lora.yaml",
    "--manifest", "data\irodori_training\aina001\train_manifest.jsonl",
    "--output-dir", $checkpointDir,
    "--device", $Device,
    "--max-steps", "$MaxSteps"
)
if ($resume) {
    $trainArgs += @("--resume", $resume.FullName)
} else {
    $trainArgs += @("--init-checkpoint", "models\Irodori-TTS-500M-v3\model.safetensors")
}
Invoke-Step "train_aina_lora" $trainArgs

Invoke-Step "convert_aina_lora" @(
    "convert_checkpoint_to_safetensors.py",
    "outputs\irodori_loras\aina001\checkpoint\checkpoint_final",
    "--base-checkpoint", "models\Irodori-TTS-500M-v3\model.safetensors",
    "--output", "outputs\irodori_loras\aina001\aina001_merged.safetensors"
)

$modelCsv = & ".\.venv\Scripts\python.exe" -c "import sys; sys.path.insert(0, 'scripts'); from irodori_targets import DEFAULT_MODEL_FOLDERS; print(','.join(DEFAULT_MODEL_FOLDERS + ['aina001']))"
Invoke-Step "build_registry_with_aina" @(
    "scripts\build_model_registry.py",
    "--models", $modelCsv,
    "--training-data-root", "data\irodori_training",
    "--distilled-root", "data\distilled",
    "--checkpoint-root", "outputs\irodori_loras",
    "--output", "configs\model_registry.local.json"
)

if ($StartApiWhenDone) {
    Stop-IrodoriApi
    $env:IRODORI_MODEL_REGISTRY = "configs\model_registry.local.json"
    $env:IRODORI_PRELOAD_MODELS = "true"
    $apiOut = Join-Path $LogDir "api.out.log"
    $apiErr = Join-Path $LogDir "api.err.log"
    Start-Process -FilePath ".venv\Scripts\uvicorn.exe" `
        -ArgumentList @("api_server:app", "--host", "127.0.0.1", "--port", "8000") `
        -WorkingDirectory (Get-Location) `
        -RedirectStandardOutput $apiOut `
        -RedirectStandardError $apiErr | Out-Null
}

Write-Log "Aina training complete"
