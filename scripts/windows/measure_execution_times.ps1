<#
.SYNOPSIS
    Script para medir y documentar tiempos de ejecución del pipeline de procesamiento.

.DESCRIPTION
    Este script ejecuta el pipeline de procesamiento completo midiendo el tiempo
    de cada operación, generando un reporte detallado para el informe técnico.

.PARAMETER SubsetSize
    Número de registros a procesar. Usar 0 para dataset completo.
    Por defecto: 0 (completo)

.PARAMETER CompareVolumes
    Si se debe ejecutar comparativa con diferentes volúmenes de datos.

.EXAMPLE
    .\measure_execution_times.ps1
    .\measure_execution_times.ps1 -SubsetSize 100000
    .\measure_execution_times.ps1 -CompareVolumes
#>

param(
    [int]$SubsetSize = 0,
    [switch]$CompareVolumes
)

$ErrorActionPreference = "Continue"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent (Split-Path -Parent $scriptDir)

# Colores
function Write-Step { param([string]$Message) Write-Host "`n🔷 $Message" -ForegroundColor Cyan }
function Write-Success { param([string]$Message) Write-Host "   ✅ $Message" -ForegroundColor Green }
function Write-Info { param([string]$Message) Write-Host "   ℹ️  $Message" -ForegroundColor White }
function Write-Timing { param([string]$Operation, [double]$Seconds) 
    Write-Host "   ⏱️  $Operation`: " -NoNewline -ForegroundColor Yellow
    Write-Host "$([math]::Round($Seconds, 2)) segundos" -ForegroundColor White
}

# Banner
Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Blue
Write-Host "║        MEDICIÓN DE TIEMPOS DE EJECUCIÓN - CLIMAXTREME       ║" -ForegroundColor Blue
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Blue
Write-Host ""

# Verificar Docker
Write-Step "Verificando infraestructura..."
$dockerCheck = docker ps --filter "name=climaxtreme" --format "{{.Names}}" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker no está corriendo o los contenedores no están activos" -ForegroundColor Red
    exit 1
}
Write-Success "Contenedores activos"

# Crear directorio de resultados
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$resultsDir = Join-Path $projectRoot "DATA\performance"
if (-not (Test-Path $resultsDir)) {
    New-Item -ItemType Directory -Path $resultsDir -Force | Out-Null
}
$resultsFile = Join-Path $resultsDir "execution_times_$timestamp.txt"
$csvFile = Join-Path $resultsDir "execution_times_$timestamp.csv"

# Inicializar archivo de resultados
$header = @"
================================================================================
              REPORTE DE TIEMPOS DE EJECUCIÓN - CLIMAXTREME
================================================================================

Fecha: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
Host: $env:COMPUTERNAME
Subset: $(if ($SubsetSize -eq 0) { "Dataset completo (~8.6M registros)" } else { "$SubsetSize registros" })

================================================================================
TIEMPOS POR OPERACIÓN
================================================================================

"@
$header | Out-File -FilePath $resultsFile -Encoding UTF8

# CSV header
"operation,duration_seconds,records_processed,throughput_records_per_sec" | Out-File -FilePath $csvFile -Encoding UTF8

# Función para medir tiempo de ejecución
function Measure-Operation {
    param(
        [string]$Name,
        [scriptblock]$Operation,
        [int]$Records = 0
    )
    
    Write-Info "Ejecutando: $Name"
    
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    
    try {
        & $Operation
        $stopwatch.Stop()
        $duration = $stopwatch.Elapsed.TotalSeconds
        
        $throughput = if ($Records -gt 0) { [math]::Round($Records / $duration, 0) } else { 0 }
        
        Write-Timing $Name $duration
        
        # Guardar en archivo
        $line = "$(if ($Records -gt 0) { "$Name ($Records registros)" } else { $Name }): $([math]::Round($duration, 2))s"
        if ($throughput -gt 0) { $line += " ($throughput reg/s)" }
        $line | Out-File -FilePath $resultsFile -Append -Encoding UTF8
        
        # CSV
        "$Name,$([math]::Round($duration, 4)),$Records,$throughput" | Out-File -FilePath $csvFile -Append -Encoding UTF8
        
        return @{
            Name = $Name
            Duration = $duration
            Records = $Records
            Throughput = $throughput
            Success = $true
        }
    }
    catch {
        $stopwatch.Stop()
        Write-Host "   ❌ Error en $Name`: $_" -ForegroundColor Red
        return @{
            Name = $Name
            Duration = $stopwatch.Elapsed.TotalSeconds
            Success = $false
            Error = $_.Exception.Message
        }
    }
}

# Array para guardar resultados
$results = @()

# ==================== MEDICIONES ====================

Write-Step "Iniciando mediciones de tiempos..."

# 1. Verificación de HDFS
$results += Measure-Operation -Name "Verificación HDFS" -Operation {
    docker exec climaxtreme-namenode hdfs dfs -ls /data 2>&1 | Out-Null
}

# 2. Ejecutar pipeline completo
Write-Step "Ejecutando pipeline de procesamiento..."

# El comando Python para ejecutar el procesamiento
$pythonCmd = @"
import time
import sys
sys.path.insert(0, '/app/Tools/src')

from climaxtreme.preprocessing.spark.spark_processor import SparkPreprocessor
from climaxtreme.preprocessing.spark.spark_session_manager import SparkSessionManager

# Inicializar
print('Iniciando Spark Session...')
start = time.time()
processor = SparkPreprocessor('climaXtreme-Benchmark')
spark_init_time = time.time() - start
print(f'TIMING:spark_init:{spark_init_time:.4f}')

# Leer datos
print('Leyendo CSV desde HDFS...')
start = time.time()
df = processor.read_city_temperature_csv_path('hdfs://climaxtreme-namenode:9000/data/raw/GlobalLandTemperaturesByCity.csv')
read_time = time.time() - start
count = df.count()
print(f'TIMING:read_csv:{read_time:.4f}:{count}')

# Limpieza
print('Limpiando datos...')
start = time.time()
df_clean = processor.clean_temperature_data(df)
clean_count = df_clean.count()
clean_time = time.time() - start
print(f'TIMING:clean:{clean_time:.4f}:{clean_count}')

# Agregación mensual
print('Agregando datos mensuales...')
start = time.time()
df_monthly = processor.aggregate_monthly_data(df_clean)
monthly_count = df_monthly.count()
monthly_time = time.time() - start
print(f'TIMING:monthly_agg:{monthly_time:.4f}:{monthly_count}')

# Agregación anual
print('Agregando datos anuales...')
start = time.time()
df_yearly = processor.aggregate_yearly_data(df_clean)
yearly_count = df_yearly.count()
yearly_time = time.time() - start
print(f'TIMING:yearly_agg:{yearly_time:.4f}:{yearly_count}')

# Detección de anomalías
print('Detectando anomalías...')
start = time.time()
df_anomalies = processor.detect_anomalies(df_clean)
anomaly_count = df_anomalies.count()
anomaly_time = time.time() - start
print(f'TIMING:anomalies:{anomaly_time:.4f}:{anomaly_count}')

# Climatología
print('Calculando climatología...')
start = time.time()
df_clim = processor.compute_climatology_stats(df_clean)
clim_count = df_clim.count()
clim_time = time.time() - start
print(f'TIMING:climatology:{clim_time:.4f}:{clim_count}')

# Estacional
print('Calculando estadísticas estacionales...')
start = time.time()
df_seasonal = processor.compute_seasonal_stats(df_clean)
seasonal_count = df_seasonal.count()
seasonal_time = time.time() - start
print(f'TIMING:seasonal:{seasonal_time:.4f}:{seasonal_count}')

# Cerrar Spark
processor.stop_spark_session()
print('TIMING:complete')
"@

# Ejecutar y capturar output
$pipelineStart = Get-Date
Write-Info "Ejecutando pipeline en contenedor processor..."

$output = docker exec climaxtreme-processor python3 -c $pythonCmd 2>&1

$pipelineEnd = Get-Date
$totalPipelineTime = ($pipelineEnd - $pipelineStart).TotalSeconds

Write-Timing "Pipeline Total" $totalPipelineTime

# Parsear tiempos del output
Write-Step "Parseando resultados..."

$output | ForEach-Object {
    if ($_ -match "TIMING:(\w+):([0-9.]+)(:([0-9]+))?") {
        $opName = $Matches[1]
        $opTime = [double]$Matches[2]
        $opRecords = if ($Matches[4]) { [int]$Matches[4] } else { 0 }
        
        $displayName = switch ($opName) {
            "spark_init" { "Inicialización Spark" }
            "read_csv" { "Lectura CSV desde HDFS" }
            "clean" { "Limpieza de datos" }
            "monthly_agg" { "Agregación mensual" }
            "yearly_agg" { "Agregación anual" }
            "anomalies" { "Detección de anomalías" }
            "climatology" { "Cálculo de climatología" }
            "seasonal" { "Estadísticas estacionales" }
            default { $opName }
        }
        
        $throughput = if ($opRecords -gt 0) { [math]::Round($opRecords / $opTime, 0) } else { 0 }
        
        Write-Timing $displayName $opTime
        
        # Guardar
        $line = "$displayName`: $([math]::Round($opTime, 2))s"
        if ($opRecords -gt 0) { $line += " ($opRecords registros, $throughput reg/s)" }
        $line | Out-File -FilePath $resultsFile -Append -Encoding UTF8
        
        "$displayName,$([math]::Round($opTime, 4)),$opRecords,$throughput" | Out-File -FilePath $csvFile -Append -Encoding UTF8
    }
}

# Resumen final
$summary = @"

================================================================================
RESUMEN
================================================================================

Tiempo total del pipeline: $([math]::Round($totalPipelineTime, 2)) segundos

================================================================================
NOTAS PARA EL INFORME
================================================================================

1. Los tiempos incluyen la latencia de HDFS y la comunicación entre contenedores.
2. El primer job de Spark incluye el tiempo de calentamiento de la JVM.
3. Para una comparación justa, ejecute múltiples veces y calcule el promedio.
4. El throughput se calcula como: registros_procesados / tiempo_en_segundos

Archivos generados:
- Reporte: $resultsFile
- CSV: $csvFile

"@

$summary | Out-File -FilePath $resultsFile -Append -Encoding UTF8

Write-Host ""
Write-Host "=================================================================================" -ForegroundColor Green
Write-Host "                         MEDICIÓN COMPLETADA" -ForegroundColor Green
Write-Host "=================================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "📁 Archivos generados:" -ForegroundColor Yellow
Write-Host "   - Reporte: $resultsFile" -ForegroundColor White
Write-Host "   - CSV: $csvFile" -ForegroundColor White
Write-Host ""
Write-Host "💡 Tip: Use el CSV para generar gráficos comparativos en el informe." -ForegroundColor Cyan
Write-Host ""

# Si se pidió comparación de volúmenes
if ($CompareVolumes) {
    Write-Host ""
    Write-Host "⚠️  La comparación de volúmenes requiere ejecutar múltiples veces el pipeline" -ForegroundColor Yellow
    Write-Host "   con diferentes tamaños de dataset. Esta funcionalidad está pendiente." -ForegroundColor Yellow
}
