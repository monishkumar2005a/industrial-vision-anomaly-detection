$ErrorActionPreference = "Stop"

$DATA_ROOT = "D:\mvtec_anomaly_detection"
$MODEL_ROOT = ".\models"

$categories = @(
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper"
)

foreach ($category in $categories) {

    Write-Host ""
    Write-Host "========================================"
    Write-Host "CATEGORY: $category"
    Write-Host "========================================"

    $modelDir = Join-Path $MODEL_ROOT $category
    $metaFile = Join-Path $modelDir "meta.json"

    if (-not (Test-Path $metaFile)) {

        Write-Host "Training $category..."

        anomaly-detect train `
            --data-root $DATA_ROOT `
            --category $category `
            --model-dir $modelDir
    }
    else {
        Write-Host "Model already exists for $category - skipping training."
    }

    Write-Host "Evaluating $category..."

    anomaly-detect evaluate `
        --data-root $DATA_ROOT `
        --category $category `
        --model-dir $modelDir

    Write-Host "Finished: $category"
}

Write-Host ""
Write-Host "========================================"
Write-Host "ALL 15 CATEGORIES COMPLETED"
Write-Host "========================================"