<#
.SYNOPSIS
    Script de monitoreo de métricas del clúster Docker para climaXtreme.

.DESCRIPTION
    Este script captura métricas de CPU, RAM y disco de los contenedores Docker
    durante la ejecución de jobs de Spark. Genera archivos CSV con las métricas
    y gráficos para incluir en el informe técnico.

.PARAMETER Duration
    Duración del monitoreo en segundos. Por defecto: 300 (5 minutos).

.PARAMETER Interval
    Intervalo entre capturas en segundos. Por defecto: 5.

.PARAMETER OutputDir
    Directorio donde guardar los resultados. Por defecto: DATA/metrics.

.EXAMPLE
    .\monitor_cluster_metrics.ps1 -Duration 600 -Interval 10
#>

param(
    [int]$Duration = 300,
    [int]$Interval = 5,
    [string]$OutputDir = "..\DATA\metrics"
)

# Colores para output
$colors = @{
    Info = "Cyan"
    Success = "Green"
    Warning = "Yellow"
    Error = "Red"
}

function Write-ColoredOutput {
    param([string]$Message, [string]$Type = "Info")
    Write-Host $Message -ForegroundColor $colors[$Type]
}

# Crear directorio de salida
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent (Split-Path -Parent $scriptDir)
$outputPath = Join-Path $projectRoot "DATA\metrics"

if (-not (Test-Path $outputPath)) {
    New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
    Write-ColoredOutput "📁 Directorio de métricas creado: $outputPath" "Success"
}

# Timestamp para archivos
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$metricsFile = Join-Path $outputPath "cluster_metrics_$timestamp.csv"
$summaryFile = Join-Path $outputPath "metrics_summary_$timestamp.txt"

Write-ColoredOutput "========================================" "Info"
Write-ColoredOutput "  MONITOREO DE MÉTRICAS - CLIMAXTREME" "Info"
Write-ColoredOutput "========================================" "Info"
Write-ColoredOutput ""
Write-ColoredOutput "⏱️  Duración: $Duration segundos" "Info"
Write-ColoredOutput "📊 Intervalo: $Interval segundos" "Info"
Write-ColoredOutput "💾 Archivo de salida: $metricsFile" "Info"
Write-ColoredOutput ""

# Verificar que Docker está corriendo
try {
    $dockerCheck = docker info 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-ColoredOutput "❌ Docker no está corriendo. Por favor inicie Docker Desktop." "Error"
        exit 1
    }
} catch {
    Write-ColoredOutput "❌ Error al verificar Docker: $_" "Error"
    exit 1
}

# Verificar contenedores del proyecto
$containers = @(
    "climaxtreme-namenode",
    "climaxtreme-datanode1",
    "climaxtreme-datanode2", 
    "climaxtreme-datanode3",
    "climaxtreme-processor",
    "climaxtreme-dashboard"
)

Write-ColoredOutput "🔍 Verificando contenedores..." "Info"

$activeContainers = @()
foreach ($container in $containers) {
    $status = docker inspect -f '{{.State.Running}}' $container 2>&1
    if ($status -eq "true") {
        $activeContainers += $container
        Write-ColoredOutput "  ✅ $container - Activo" "Success"
    } else {
        Write-ColoredOutput "  ⚠️  $container - Inactivo" "Warning"
    }
}

if ($activeContainers.Count -eq 0) {
    Write-ColoredOutput "❌ No hay contenedores activos. Ejecute primero:" "Error"
    Write-ColoredOutput "   cd infra; docker-compose up -d" "Info"
    exit 1
}

Write-ColoredOutput ""
Write-ColoredOutput "📈 Iniciando captura de métricas..." "Info"
Write-ColoredOutput "   Presione Ctrl+C para detener" "Info"
Write-ColoredOutput ""

# Cabecera del CSV
$header = "timestamp,container,cpu_percent,mem_usage_mb,mem_limit_mb,mem_percent,net_io_rx_mb,net_io_tx_mb,block_io_read_mb,block_io_write_mb"
$header | Out-File -FilePath $metricsFile -Encoding UTF8

# Variables para estadísticas
$allMetrics = @()
$iterations = [math]::Ceiling($Duration / $Interval)
$currentIteration = 0

# Función para parsear métricas de docker stats
function Get-ContainerMetrics {
    param([string]$ContainerName)
    
    try {
        $stats = docker stats $ContainerName --no-stream --format "{{.CPUPerc}},{{.MemUsage}},{{.MemPerc}},{{.NetIO}},{{.BlockIO}}" 2>&1
        
        if ($LASTEXITCODE -eq 0 -and $stats) {
            $parts = $stats -split ","
            
            # CPU (remover %)
            $cpu = [double]($parts[0] -replace '%', '')
            
            # Memoria (formato: "123.4MiB / 1GiB")
            $memParts = $parts[1] -split " / "
            $memUsage = Parse-Size $memParts[0]
            $memLimit = Parse-Size $memParts[1]
            
            # Memoria %
            $memPercent = [double]($parts[2] -replace '%', '')
            
            # Network IO (formato: "1.23MB / 4.56MB")
            $netParts = $parts[3] -split " / "
            $netRx = Parse-Size $netParts[0]
            $netTx = Parse-Size $netParts[1]
            
            # Block IO (formato: "1.23MB / 4.56MB")
            $blockParts = $parts[4] -split " / "
            $blockRead = Parse-Size $blockParts[0]
            $blockWrite = Parse-Size $blockParts[1]
            
            return @{
                CPU = $cpu
                MemUsage = $memUsage
                MemLimit = $memLimit
                MemPercent = $memPercent
                NetRx = $netRx
                NetTx = $netTx
                BlockRead = $blockRead
                BlockWrite = $blockWrite
            }
        }
    } catch {
        return $null
    }
    return $null
}

# Función para parsear tamaños (KB, MB, GB, etc.)
function Parse-Size {
    param([string]$SizeStr)
    
    $SizeStr = $SizeStr.Trim()
    
    if ($SizeStr -match "^([\d.]+)([KMGT]?i?B?)$") {
        $value = [double]$Matches[1]
        $unit = $Matches[2].ToUpper()
        
        switch -Regex ($unit) {
            "^K" { return $value / 1024 }           # KB to MB
            "^M" { return $value }                   # MB
            "^G" { return $value * 1024 }           # GB to MB
            "^T" { return $value * 1024 * 1024 }   # TB to MB
            "^B$" { return $value / (1024 * 1024) } # B to MB
            default { return $value }
        }
    }
    return 0
}

# Loop principal de monitoreo
$startTime = Get-Date

while ($currentIteration -lt $iterations) {
    $currentTime = Get-Date
    $elapsed = ($currentTime - $startTime).TotalSeconds
    $remaining = $Duration - $elapsed
    
    # Barra de progreso
    $progress = [math]::Min(100, [math]::Round(($elapsed / $Duration) * 100))
    Write-Progress -Activity "Capturando métricas" -Status "$progress% completado - Tiempo restante: $([math]::Round($remaining))s" -PercentComplete $progress
    
    $timestamp = $currentTime.ToString("yyyy-MM-dd HH:mm:ss")
    
    foreach ($container in $activeContainers) {
        $metrics = Get-ContainerMetrics -ContainerName $container
        
        if ($metrics) {
            # Escribir al CSV
            $line = "$timestamp,$container,$($metrics.CPU),$([math]::Round($metrics.MemUsage, 2)),$([math]::Round($metrics.MemLimit, 2)),$($metrics.MemPercent),$([math]::Round($metrics.NetRx, 4)),$([math]::Round($metrics.NetTx, 4)),$([math]::Round($metrics.BlockRead, 4)),$([math]::Round($metrics.BlockWrite, 4))"
            $line | Out-File -FilePath $metricsFile -Append -Encoding UTF8
            
            # Guardar para estadísticas
            $allMetrics += @{
                Container = $container
                CPU = $metrics.CPU
                MemUsage = $metrics.MemUsage
                MemPercent = $metrics.MemPercent
            }
        }
    }
    
    $currentIteration++
    
    if ($currentIteration -lt $iterations) {
        Start-Sleep -Seconds $Interval
    }
}

Write-Progress -Activity "Capturando métricas" -Completed

Write-ColoredOutput ""
Write-ColoredOutput "✅ Captura completada!" "Success"
Write-ColoredOutput ""

# Generar resumen de estadísticas
Write-ColoredOutput "📊 Generando resumen de estadísticas..." "Info"

$summary = @"
================================================================================
              RESUMEN DE MÉTRICAS DEL CLÚSTER - CLIMAXTREME
================================================================================

Fecha de captura: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
Duración del monitoreo: $Duration segundos
Intervalo de muestreo: $Interval segundos
Número de muestras: $($allMetrics.Count)

--------------------------------------------------------------------------------
ESTADÍSTICAS POR CONTENEDOR
--------------------------------------------------------------------------------

"@

# Calcular estadísticas por contenedor
foreach ($container in $activeContainers) {
    $containerMetrics = $allMetrics | Where-Object { $_.Container -eq $container }
    
    if ($containerMetrics.Count -gt 0) {
        $cpuValues = $containerMetrics | ForEach-Object { $_.CPU }
        $memValues = $containerMetrics | ForEach-Object { $_.MemUsage }
        $memPercentValues = $containerMetrics | ForEach-Object { $_.MemPercent }
        
        $cpuAvg = ($cpuValues | Measure-Object -Average).Average
        $cpuMax = ($cpuValues | Measure-Object -Maximum).Maximum
        $memAvg = ($memValues | Measure-Object -Average).Average
        $memMax = ($memValues | Measure-Object -Maximum).Maximum
        $memPercentAvg = ($memPercentValues | Measure-Object -Average).Average
        
        $summary += @"

[$container]
  CPU:
    - Promedio: $([math]::Round($cpuAvg, 2))%
    - Máximo:   $([math]::Round($cpuMax, 2))%
  
  Memoria:
    - Promedio: $([math]::Round($memAvg, 2)) MB ($([math]::Round($memPercentAvg, 2))%)
    - Máximo:   $([math]::Round($memMax, 2)) MB

"@
    }
}

$summary += @"
--------------------------------------------------------------------------------
ARCHIVOS GENERADOS
--------------------------------------------------------------------------------

- Métricas detalladas (CSV): $metricsFile
- Este resumen: $summaryFile

--------------------------------------------------------------------------------
NOTAS PARA EL INFORME
--------------------------------------------------------------------------------

1. El archivo CSV puede importarse en Excel o Python para generar gráficos.
2. Los valores de CPU son porcentajes del total de CPUs disponibles.
3. Los valores de memoria están en MB.
4. Para capturar métricas durante el procesamiento completo:
   - Abra una terminal y ejecute este script
   - En otra terminal, ejecute: .\process_full_dataset.ps1

================================================================================
"@

$summary | Out-File -FilePath $summaryFile -Encoding UTF8

Write-ColoredOutput ""
Write-ColoredOutput "📁 Archivos generados:" "Success"
Write-ColoredOutput "   - $metricsFile" "Info"
Write-ColoredOutput "   - $summaryFile" "Info"
Write-ColoredOutput ""

# Mostrar resumen en consola
Write-ColoredOutput "================== RESUMEN RÁPIDO ==================" "Info"

foreach ($container in $activeContainers) {
    $containerMetrics = $allMetrics | Where-Object { $_.Container -eq $container }
    if ($containerMetrics.Count -gt 0) {
        $cpuAvg = ($containerMetrics | ForEach-Object { $_.CPU } | Measure-Object -Average).Average
        $memAvg = ($containerMetrics | ForEach-Object { $_.MemUsage } | Measure-Object -Average).Average
        Write-ColoredOutput "  $container" "Success"
        Write-ColoredOutput "    CPU: $([math]::Round($cpuAvg, 1))%  |  RAM: $([math]::Round($memAvg, 0)) MB" "Info"
    }
}

Write-ColoredOutput "=====================================================" "Info"
Write-ColoredOutput ""
Write-ColoredOutput "💡 Tip: Para generar gráficos, use el script:" "Info"
Write-ColoredOutput "   python scripts/generate_metrics_charts.py $metricsFile" "Info"
