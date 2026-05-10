param(
    [Parameter(Mandatory = $true)]
    [string]$Image,

    [string]$PodName = "irodori-tts-prod",
    [string]$GpuId = "NVIDIA GeForce RTX 4090",
    [string]$DataCenterIds = "",
    [string]$NetworkVolumeId = "",
    [string]$VolumeMountPath = "/workspace",
    [int]$ContainerDiskGB = 40,
    [string]$Ports = "8000/http,22/tcp",
    [string]$RegistryPath = "/workspace/irodori_artifacts/configs/model_registry.runpod.json",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

if (-not $env:RUNPOD_API_KEY) {
    throw "RUNPOD_API_KEY is not set. Set it in the current shell before running this script."
}

$envJson = @{
    IRODORI_MODEL_REGISTRY = $RegistryPath
    IRODORI_MODEL_DEVICE = "cuda"
    IRODORI_MODEL_PRECISION = "bf16"
    IRODORI_CODEC_DEVICE = "cuda"
    IRODORI_CODEC_PRECISION = "fp32"
    IRODORI_LORA_LOAD_MODE = "delta"
    IRODORI_CHUNK_BATCH_SIZE = "2"
    IRODORI_MAX_CACHED_RUNTIMES = "1"
    IRODORI_PRELOAD_MODELS = "false"
} | ConvertTo-Json -Compress

$argsList = @(
    "pod", "create",
    "--name", $PodName,
    "--image", $Image,
    "--gpu-id", $GpuId,
    "--ports", $Ports,
    "--container-disk-in-gb", "$ContainerDiskGB",
    "--volume-mount-path", $VolumeMountPath,
    "--env", $envJson
)

if ($NetworkVolumeId) {
    $argsList += @("--network-volume-id", $NetworkVolumeId)
}
if ($DataCenterIds) {
    $argsList += @("--data-center-ids", $DataCenterIds)
}

if ($DryRun) {
    "runpodctl " + ($argsList -join " ")
    exit 0
}

& runpodctl @argsList
