<#
.SYNOPSIS
    Script completo para cargar y procesar el dataset completo en HDFS + Spark

.DESCRIPTION
    Este script automatiza todo el proceso:
    1. Verifica que Docker esté corriendo
    2. Inicia HDFS (namenode + datanode)
    3. Sube el dataset COMPLETO a HDFS
    4. Procesa el dataset con PySpark
    5. Guarda los resultados procesados
    6. Descarga los resultados a la carpeta local DATA/processed

.PARAMETER CsvPath
    Ruta al archivo CSV (default: DATA/GlobalLandTemperaturesByCity.csv)

.PARAMETER SkipUpload
    Si se especifica, asume que el archivo ya está en HDFS y solo ejecuta el procesamiento

.EXAMPLE
    .\process_full_dataset.ps1
    # Procesa todo desde cero

.EXAMPLE
    .\process_full_dataset.ps1 -SkipUpload
    # Solo procesa (asume que el archivo ya está en HDFS)
#>

param(
    [string]$CsvPath = "DATA/GlobalLandTemperaturesByCity.csv",
    [switch]$SkipUpload
)

$ErrorActionPreference = "Stop"

# Colores para output
function Write-Success { Write-Host $args -ForegroundColor Green }
function Write-Info { Write-Host $args -ForegroundColor Cyan }
function Write-Warning { Write-Host $args -ForegroundColor Yellow }
function Write-Error { param($msg) Write-Host $msg -ForegroundColor Red }

Write-Host ""
Write-Host "========================================" -ForegroundColor Magenta
Write-Host "  climaXtreme - Procesamiento Completo" -ForegroundColor Magenta
Write-Host "========================================" -ForegroundColor Magenta
Write-Host ""

# Resolve paths
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$HdfsSetupScript = Join-Path $PSScriptRoot "hdfs_setup_and_load.ps1"
$OutputDir = Join-Path $RepoRoot "DATA\processed"

# HDFS paths
$HdfsInputPath = "hdfs://climaxtreme-namenode:9000/data/climaxtreme/GlobalLandTemperaturesByCity.csv"
$HdfsOutputPath = "hdfs://climaxtreme-namenode:9000/data/climaxtreme/processed"
$LocalDownloadPath = $OutputDir

Write-Info "Configuración:"
Write-Host "  Repo Root: $RepoRoot"
Write-Host "  CSV Path: $CsvPath"
Write-Host "  HDFS Input: $HdfsInputPath"
Write-Host "  HDFS Output: $HdfsOutputPath"
Write-Host "  Local Output: $LocalDownloadPath"
Write-Host ""

# ============================================================================
# PASO 1: Cargar dataset a HDFS (si no se saltó)
# ============================================================================

if (-not $SkipUpload) {
    Write-Info "PASO 1/4: Cargando dataset COMPLETO a HDFS..."
    Write-Host ""
    
    & $HdfsSetupScript -CsvPath $CsvPath -FullFile
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Error al cargar el dataset a HDFS"
        exit 1
    }
    
    Write-Success "✓ Dataset cargado exitosamente a HDFS"
    Write-Host ""
} else {
    Write-Warning "PASO 1/4: Saltado (se asume que el archivo ya está en HDFS)"
    Write-Host ""
}

# ============================================================================
# PASO 2: Verificar que el contenedor processor esté corriendo
# ============================================================================

Write-Info "PASO 2/4: Verificando contenedor processor..."
Write-Host ""

$processorStatus = docker inspect climaxtreme-processor --format '{{.State.Status}}' 2>$null

if ($processorStatus -ne 'running') {
    Write-Warning "Contenedor processor no está corriendo. Iniciándolo..."
    docker start climaxtreme-processor | Out-Null
    Start-Sleep -Seconds 5
}

Write-Success "✓ Contenedor processor está listo"
Write-Host ""

# ============================================================================
# PASO 3: Procesar dataset con PySpark
# ============================================================================

Write-Info "PASO 3/4: Procesando dataset con PySpark..."
Write-Info "  (Esto puede tardar varios minutos para datasets grandes)"
Write-Host ""

$sparkCmd = @"
python -m climaxtreme.cli preprocess \
    --input-path '$HdfsInputPath' \
    --output-path '$HdfsOutputPath' \
    --format city-csv
"@

Write-Host "Ejecutando comando Spark en contenedor..."
docker exec climaxtreme-processor bash -c $sparkCmd

if ($LASTEXITCODE -ne 0) {
    Write-Error "Error al procesar el dataset con PySpark"
    exit 1
}

Write-Success "✓ Procesamiento completado"
Write-Host ""

# ============================================================================
# PASO 4: Descargar resultados procesados
# ============================================================================

Write-Info "PASO 4/4: Descargando resultados procesados a carpeta local..."
Write-Host ""

# Crear directorio de salida
New-Item -ItemType Directory -Force -Path $LocalDownloadPath | Out-Null

# Descargar archivos Parquet desde HDFS
$artifacts = @("monthly.parquet", "yearly.parquet", "anomalies.parquet", "climatology.parquet", "seasonal.parquet", "extreme_thresholds.parquet")

foreach ($artifact in $artifacts) {
    Write-Host "  Descargando $artifact..."
    
    # Get from HDFS to container temp
    $containerTempPath = "/tmp/$artifact"
    docker exec climaxtreme-namenode hdfs dfs -get "$HdfsOutputPath/$artifact" $containerTempPath 2>$null
    
    # Copy from container to local
    $localArtifactPath = Join-Path $LocalDownloadPath $artifact
    
    # Remove if exists
    if (Test-Path $localArtifactPath) {
        Remove-Item -Recurse -Force $localArtifactPath
    }
    
    docker cp "climaxtreme-namenode:$containerTempPath" $localArtifactPath | Out-Null
    
    if (Test-Path $localArtifactPath) {
        Write-Success "    ✓ $artifact descargado"
    } else {
        Write-Warning "    ⚠ No se pudo descargar $artifact"
    }
}

Write-Host ""
Write-Success "========================================="
Write-Success "  ✓ PROCESAMIENTO COMPLETADO"
Write-Success "========================================="
Write-Host ""
Write-Info "Resultados disponibles en:"
Write-Host "  Local:  $LocalDownloadPath"
Write-Host "  HDFS:   $HdfsOutputPath"
Write-Host ""
Write-Info "Para ver el dashboard con los datos procesados:"
Write-Host "  cd Tools"
Write-Host "  python -m climaxtreme.cli dashboard --data-dir ../DATA/processed"
Write-Host ""
Write-Info "O para ver análisis específicos:"
Write-Host "  python -m climaxtreme.cli analyze --data-path DATA/processed"
Write-Host ""
