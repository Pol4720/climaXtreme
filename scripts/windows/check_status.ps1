<#
.SYNOPSIS
    Verifica el estado del sistema climaXtreme

.DESCRIPTION
    Este script verifica:
    - Estado de contenedores Docker
    - Archivos en HDFS
    - Archivos procesados localmente
    - Tamaños y estadísticas
#>

$ErrorActionPreference = "Stop"

function Write-Header {
    param([string]$text)
    Write-Host ""
    Write-Host "═══════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "  $text" -ForegroundColor Cyan
    Write-Host "═══════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host ""
}

function Write-Success { Write-Host "✓ $args" -ForegroundColor Green }
function Write-Fail { Write-Host "✗ $args" -ForegroundColor Red }
function Write-Info { Write-Host "  $args" -ForegroundColor Gray }

Write-Header "VERIFICACIÓN DEL SISTEMA climaXtreme"

# ============================================================================
# 1. Verificar Docker
# ============================================================================

Write-Header "1. Docker"

try {
    docker info | Out-Null
    Write-Success "Docker está corriendo"
} catch {
    Write-Fail "Docker no está corriendo o no está disponible"
    exit 1
}

# ============================================================================
# 2. Verificar Contenedores
# ============================================================================

Write-Header "2. Contenedores"

$containers = @("climaxtreme-namenode", "climaxtreme-datanode", "climaxtreme-processor")

foreach ($container in $containers) {
    $status = docker inspect $container --format '{{.State.Status}}' 2>$null
    
    if ($status -eq 'running') {
        Write-Success "$container está corriendo"
        
        # Get container stats
        $stats = docker stats $container --no-stream --format "table {{.CPUPerc}}\t{{.MemUsage}}" | Select-Object -Skip 1
        Write-Info "Stats: $stats"
    } elseif ($status) {
        Write-Fail "$container existe pero no está corriendo (estado: $status)"
    } else {
        Write-Fail "$container no existe"
    }
}

# ============================================================================
# 3. Verificar Archivos en HDFS
# ============================================================================

Write-Header "3. Archivos en HDFS"

$hdfsCheck = docker exec climaxtreme-namenode hdfs dfs -ls -h /data/climaxtreme/ 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Success "Directorio /data/climaxtreme/ existe en HDFS"
    Write-Host ""
    Write-Host "Contenido:" -ForegroundColor Yellow
    docker exec climaxtreme-namenode hdfs dfs -ls -h /data/climaxtreme/
    Write-Host ""
    
    # Check for main dataset
    $mainDataset = docker exec climaxtreme-namenode hdfs dfs -test -e /data/climaxtreme/GlobalLandTemperaturesByCity.csv 2>$null
    if ($LASTEXITCODE -eq 0) {
        $fileSize = docker exec climaxtreme-namenode hdfs dfs -du -h /data/climaxtreme/GlobalLandTemperaturesByCity.csv | Select-Object -First 1
        Write-Success "Dataset completo encontrado: $fileSize"
    } else {
        $sampleDataset = docker exec climaxtreme-namenode hdfs dfs -test -e /data/climaxtreme/GlobalLandTemperaturesByCity_sample.csv 2>$null
        if ($LASTEXITCODE -eq 0) {
            $fileSize = docker exec climaxtreme-namenode hdfs dfs -du -h /data/climaxtreme/GlobalLandTemperaturesByCity_sample.csv | Select-Object -First 1
            Write-Info "Solo sample encontrado: $fileSize"
        } else {
            Write-Fail "No se encontró dataset en HDFS"
        }
    }
} else {
    Write-Fail "No se puede acceder a HDFS o el directorio no existe"
}

# ============================================================================
# 4. Verificar Archivos Procesados en HDFS
# ============================================================================

Write-Header "4. Archivos Procesados en HDFS"

$processedCheck = docker exec climaxtreme-namenode hdfs dfs -ls -h /data/climaxtreme/processed/ 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Success "Directorio /data/climaxtreme/processed/ existe"
    Write-Host ""
    Write-Host "Contenido:" -ForegroundColor Yellow
    docker exec climaxtreme-namenode hdfs dfs -ls -h /data/climaxtreme/processed/
    Write-Host ""
    
    $artifacts = @("monthly.parquet", "yearly.parquet", "anomalies.parquet")
    foreach ($artifact in $artifacts) {
        $artifactCheck = docker exec climaxtreme-namenode hdfs dfs -test -e "/data/climaxtreme/processed/$artifact" 2>$null
        if ($LASTEXITCODE -eq 0) {
            $size = docker exec climaxtreme-namenode hdfs dfs -du -h "/data/climaxtreme/processed/$artifact" | Select-Object -First 1
            Write-Success "$artifact encontrado: $size"
        } else {
            Write-Fail "$artifact NO encontrado"
        }
    }
} else {
    Write-Fail "No se encontraron archivos procesados en HDFS"
    Write-Info "Ejecuta: .\scripts\process_full_dataset.ps1"
}

# ============================================================================
# 5. Verificar Archivos Locales
# ============================================================================

Write-Header "5. Archivos Locales"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$DataDir = Join-Path $RepoRoot "DATA"
$ProcessedDir = Join-Path $DataDir "processed"

# Check raw CSV
$csvPath = Join-Path $DataDir "GlobalLandTemperaturesByCity.csv"
if (Test-Path $csvPath) {
    $size = (Get-Item $csvPath).Length
    $sizeMB = [math]::Round($size / 1MB, 2)
    Write-Success "CSV original encontrado: $sizeMB MB"
} else {
    Write-Fail "CSV original NO encontrado en $csvPath"
}

# Check processed files
if (Test-Path $ProcessedDir) {
    Write-Success "Directorio processed/ existe"
    Write-Host ""
    Write-Host "Contenido:" -ForegroundColor Yellow
    
    $artifacts = @("monthly.parquet", "yearly.parquet", "anomalies.parquet")
    foreach ($artifact in $artifacts) {
        $artifactPath = Join-Path $ProcessedDir $artifact
        if (Test-Path $artifactPath) {
            $size = (Get-ChildItem $artifactPath -Recurse | Measure-Object -Property Length -Sum).Sum
            $sizeMB = [math]::Round($size / 1MB, 2)
            Write-Success "$artifact encontrado: $sizeMB MB"
        } else {
            Write-Fail "$artifact NO encontrado"
        }
    }
} else {
    Write-Fail "Directorio processed/ NO existe"
    Write-Info "Ejecuta: .\scripts\process_full_dataset.ps1"
}

# ============================================================================
# 6. Resumen y Recomendaciones
# ============================================================================

Write-Header "6. Resumen"

Write-Host "Estado del Sistema:" -ForegroundColor Yellow
Write-Host ""

# Count issues
$issues = 0

# Check if dataset is uploaded
$datasetUploaded = docker exec climaxtreme-namenode hdfs dfs -test -e /data/climaxtreme/GlobalLandTemperaturesByCity.csv 2>$null
if ($LASTEXITCODE -ne 0) {
    $issues++
    Write-Info "⚠ Dataset completo no está en HDFS"
}

# Check if processed
$processed = docker exec climaxtreme-namenode hdfs dfs -test -e /data/climaxtreme/processed/monthly.parquet 2>$null
if ($LASTEXITCODE -ne 0) {
    $issues++
    Write-Info "⚠ Dataset no ha sido procesado"
}

# Check if downloaded
if (-not (Test-Path (Join-Path $ProcessedDir "monthly.parquet"))) {
    $issues++
    Write-Info "⚠ Resultados procesados no están descargados localmente"
}

Write-Host ""

if ($issues -eq 0) {
    Write-Success "¡Todo está configurado correctamente!"
    Write-Host ""
    Write-Info "Puedes iniciar el dashboard con:"
    Write-Info "  cd Tools"
    Write-Info "  python -m climaxtreme.cli dashboard --data-dir ../DATA/processed"
} else {
    Write-Host "Se encontraron $issues problemas." -ForegroundColor Yellow
    Write-Host ""
    Write-Info "Para configurar todo desde cero, ejecuta:"
    Write-Info "  .\scripts\process_full_dataset.ps1"
}

Write-Host ""
