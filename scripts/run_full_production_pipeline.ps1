param(
    [string]$ModelCsv = "",
    [string]$ModelAssetsRoot = "D:\sbv2\Style-Bert-VITS2\model_assets",
    [string]$DataRoot = "D:\sbv2\Style-Bert-VITS2\Data",
    [string]$Sbv2Root = "C:\sbv2\Style-Bert-VITS2",
    [string]$Sbv2Python = "C:\sbv2\Style-Bert-VITS2\venv\Scripts\python.exe",
    [int]$DistillLimit = 160,
    [int]$MaxSteps = 1000,
    [string]$Device = "cuda",
    [string]$LogDir = "",
    [switch]$SkipDistill,
    [switch]$SkipTrain,
    [switch]$StartApiWhenDone
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

Set-Location (Resolve-Path (Join-Path $PSScriptRoot ".."))

if ([string]::IsNullOrWhiteSpace($LogDir)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $LogDir = "logs\production_pipeline_$stamp"
}
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$env:SBV2_ROOT = $Sbv2Root
$env:SBV2_PYTHON = $Sbv2Python
$env:SBV2_MODEL_ASSETS_ROOT = $ModelAssetsRoot
$env:SBV2_DATA_ROOT = $DataRoot

function Write-State {
    param([string]$Stage, [string]$Model = "", [string]$Status = "running")
    $state = [ordered]@{
        timestamp = (Get-Date).ToString("o")
        stage = $Stage
        model = $Model
        status = $Status
        log_dir = (Resolve-Path $LogDir).Path
    }
    $state | ConvertTo-Json -Depth 5 | Set-Content -Path (Join-Path $LogDir "state.json") -Encoding UTF8
}

function Invoke-Native {
    param([string]$Name, [string[]]$ArgsList)
    $logPath = Join-Path $LogDir "$Name.log"
    Write-State -Stage $Name
    "[$(Get-Date -Format o)] $Name" | Tee-Object -FilePath $logPath -Append
    "uv $($ArgsList -join ' ')" | Tee-Object -FilePath $logPath -Append
    & uv @ArgsList 2>&1 | Tee-Object -FilePath $logPath -Append
    if ($LASTEXITCODE -ne 0) {
        Write-State -Stage $Name -Status "failed"
        throw "$Name failed with exit code $LASTEXITCODE"
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

$inventoryPath = Join-Path $LogDir "inventory.json"
$inventoryArgs = @(
    "run", "python", "scripts\inventory_irodori_targets.py",
    "--model-assets-root", $ModelAssetsRoot,
    "--data-root", $DataRoot,
    "--output", $inventoryPath,
    "--pretty"
)
if (-not [string]::IsNullOrWhiteSpace($ModelCsv)) {
    $inventoryArgs += @("--models", $ModelCsv)
}
Invoke-Native -Name "01_inventory" -ArgsList $inventoryArgs

$inventory = Get-Content -Path $inventoryPath -Raw -Encoding UTF8 | ConvertFrom-Json
$distillModels = @($inventory.items | Where-Object { $_.source_type -eq "distill" } | Select-Object -ExpandProperty model_id)

Stop-IrodoriApi

if (-not $SkipDistill) {
    foreach ($model in $distillModels) {
        Write-State -Stage "distill" -Model $model
        $safeName = $model -replace '[^A-Za-z0-9_.-]', '_'
        $distillLog = Join-Path $LogDir "distill_$safeName.log"
        $distillArgs = @(
            "run", "python", "scripts\distill_sbv2_dataset.py",
            "--models", $model,
            "--mode", "subprocess",
            "--sbv2-root", $Sbv2Root,
            "--sbv2-python", $Sbv2Python,
            "--model-assets-root", $ModelAssetsRoot,
            "--limit", "$DistillLimit",
            "--use-gpu",
            "--skip-existing",
            "--continue-on-error"
        )
        "[$(Get-Date -Format o)] distill $model" | Tee-Object -FilePath $distillLog -Append
        & uv @distillArgs 2>&1 | Tee-Object -FilePath $distillLog -Append
        if ($LASTEXITCODE -ne 0) {
            Write-State -Stage "distill" -Model $model -Status "failed"
            throw "distill failed for $model with exit code $LASTEXITCODE"
        }
    }
}

if (-not $SkipTrain) {
    Stop-IrodoriApi
    $trainArgs = @(
        "run", "python", "scripts\batch_train_irodori_loras.py",
        "--inventory", $inventoryPath,
        "--training-data-root", "data\irodori_training",
        "--distilled-root", "data\distilled",
        "--output-root", "outputs\irodori_loras",
        "--config", "configs\train_500m_v2_lora_kohaku.yaml",
        "--base-checkpoint", "models\Irodori-TTS-500M-v2\model.safetensors",
        "--device", $Device,
        "--max-steps", "$MaxSteps",
        "--skip-existing",
        "--resume-existing"
    )
    if (-not [string]::IsNullOrWhiteSpace($ModelCsv)) {
        $trainArgs += @("--models", $ModelCsv)
    }
    Invoke-Native -Name "02_batch_train" -ArgsList $trainArgs
}

$registryArgs = @(
    "run", "python", "scripts\build_model_registry.py",
    "--inventory", $inventoryPath,
    "--training-data-root", "data\irodori_training",
    "--distilled-root", "data\distilled",
    "--checkpoint-root", "outputs\irodori_loras",
    "--output", "configs\model_registry.local.json"
)
if (-not [string]::IsNullOrWhiteSpace($ModelCsv)) {
    $registryArgs += @("--models", $ModelCsv)
}
Invoke-Native -Name "03_build_registry" -ArgsList $registryArgs

Invoke-Native -Name "04_status" -ArgsList @(
    "run", "python", "scripts\pipeline_status.py",
    "--inventory", $inventoryPath,
    "--distill-target-rows", "$DistillLimit",
    "--json"
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

Write-State -Stage "complete" -Status "ok"
