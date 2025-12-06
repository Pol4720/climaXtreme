<#
.SYNOPSIS
    Script de demostración para la exposición del proyecto ClimaXtreme.

.DESCRIPTION
    Este script automatiza el flujo de demostración para la presentación oral,
    incluyendo verificación de infraestructura, procesamiento de datos y
    apertura del dashboard.

.PARAMETER Mode
    Modo de ejecución:
    - "full": Demostración completa (procesamiento + dashboard)
    - "quick": Demostración rápida (solo verificación + dashboard)
    - "status": Solo verificar estado del sistema

.PARAMETER OpenBrowser
    Si se debe abrir automáticamente el navegador con las UIs.

.EXAMPLE
    .\demo_presentation.ps1 -Mode quick -OpenBrowser
    .\demo_presentation.ps1 -Mode full
    .\demo_presentation.ps1 -Mode status
#>

param(
    [ValidateSet("full", "quick", "status")]
    [string]$Mode = "quick",
    [switch]$OpenBrowser = $true
)

# Configuración
$ErrorActionPreference = "Continue"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent (Split-Path -Parent $scriptDir)
$infraDir = Join-Path $projectRoot "infra"

# Colores
function Write-Step { param([string]$Message) Write-Host "`n🔷 $Message" -ForegroundColor Cyan }
function Write-Success { param([string]$Message) Write-Host "   ✅ $Message" -ForegroundColor Green }
function Write-Warning { param([string]$Message) Write-Host "   ⚠️  $Message" -ForegroundColor Yellow }
function Write-Error { param([string]$Message) Write-Host "   ❌ $Message" -ForegroundColor Red }
function Write-Info { param([string]$Message) Write-Host "   ℹ️  $Message" -ForegroundColor White }

# Banner
function Show-Banner {
    Write-Host ""
    Write-Host "╔══════════════════════════════════════════════════════════════════╗" -ForegroundColor Blue
    Write-Host "║                                                                  ║" -ForegroundColor Blue
    Write-Host "║   ██████╗██╗     ██╗███╗   ███╗ █████╗ ██╗  ██╗████████╗██████╗ ║" -ForegroundColor Cyan
    Write-Host "║  ██╔════╝██║     ██║████╗ ████║██╔══██╗╚██╗██╔╝╚══██╔══╝██╔══██╗║" -ForegroundColor Cyan
    Write-Host "║  ██║     ██║     ██║██╔████╔██║███████║ ╚███╔╝    ██║   ██████╔╝║" -ForegroundColor Cyan
    Write-Host "║  ██║     ██║     ██║██║╚██╔╝██║██╔══██║ ██╔██╗    ██║   ██╔══██╗║" -ForegroundColor Cyan
    Write-Host "║  ╚██████╗███████╗██║██║ ╚═╝ ██║██║  ██║██╔╝ ██╗   ██║   ██║  ██║║" -ForegroundColor Cyan
    Write-Host "║   ╚═════╝╚══════╝╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝║" -ForegroundColor Cyan
    Write-Host "║                                                                  ║" -ForegroundColor Blue
    Write-Host "║      Análisis Climático y Modelado de Eventos Extremos          ║" -ForegroundColor White
    Write-Host "║             Procesamiento de Grandes Volúmenes de Datos         ║" -ForegroundColor White
    Write-Host "║                                                                  ║" -ForegroundColor Blue
    Write-Host "╚══════════════════════════════════════════════════════════════════╝" -ForegroundColor Blue
    Write-Host ""
}

# Verificar Docker
function Test-Docker {
    Write-Step "Verificando Docker Desktop..."
    
    try {
        $dockerInfo = docker info 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Docker está corriendo"
            return $true
        }
    } catch {}
    
    Write-Error "Docker no está disponible"
    Write-Info "Por favor, inicie Docker Desktop y espere a que esté listo"
    return $false
}

# Verificar contenedores
function Test-Containers {
    Write-Step "Verificando contenedores del clúster..."
    
    $containers = @{
        "climaxtreme-namenode" = @{ Port = 9870; Name = "HDFS NameNode" }
        "climaxtreme-datanode1" = @{ Port = $null; Name = "DataNode 1" }
        "climaxtreme-datanode2" = @{ Port = $null; Name = "DataNode 2" }
        "climaxtreme-datanode3" = @{ Port = $null; Name = "DataNode 3" }
        "climaxtreme-processor" = @{ Port = 4040; Name = "Spark Processor" }
        "climaxtreme-dashboard" = @{ Port = 8501; Name = "Streamlit Dashboard" }
    }
    
    $allRunning = $true
    $results = @()
    
    foreach ($container in $containers.Keys) {
        $info = $containers[$container]
        $status = docker inspect -f '{{.State.Running}}' $container 2>&1
        
        if ($status -eq "true") {
            $portInfo = if ($info.Port) { "(puerto $($info.Port))" } else { "" }
            Write-Success "$($info.Name) $portInfo"
            $results += @{ Name = $container; Running = $true; Port = $info.Port }
        } else {
            Write-Warning "$($info.Name) - No está corriendo"
            $results += @{ Name = $container; Running = $false; Port = $info.Port }
            $allRunning = $false
        }
    }
    
    return @{ AllRunning = $allRunning; Containers = $results }
}

# Iniciar contenedores
function Start-Containers {
    Write-Step "Iniciando contenedores..."
    
    Push-Location $infraDir
    
    Write-Info "Ejecutando docker-compose up -d..."
    docker-compose up -d 2>&1 | Out-Null
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Contenedores iniciados"
        
        Write-Info "Esperando a que los servicios estén listos (30 segundos)..."
        $progress = 0
        while ($progress -lt 30) {
            Write-Progress -Activity "Esperando servicios" -Status "$progress/30 segundos" -PercentComplete (($progress / 30) * 100)
            Start-Sleep -Seconds 1
            $progress++
        }
        Write-Progress -Activity "Esperando servicios" -Completed
        
        Write-Success "Servicios listos"
    } else {
        Write-Error "Error al iniciar contenedores"
    }
    
    Pop-Location
}

# Verificar HDFS
function Test-HDFS {
    Write-Step "Verificando HDFS..."
    
    # Verificar estructura de directorios
    $hdfsCheck = docker exec climaxtreme-namenode hdfs dfs -ls / 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "HDFS operativo"
        
        # Verificar datos procesados
        $processedCheck = docker exec climaxtreme-namenode hdfs dfs -ls /data/processed 2>&1
        if ($LASTEXITCODE -eq 0) {
            $parquetCount = ($processedCheck | Select-String "\.parquet" | Measure-Object).Count
            Write-Success "Datos procesados encontrados: $parquetCount archivos Parquet"
        } else {
            Write-Warning "No se encontraron datos procesados en HDFS"
            Write-Info "Ejecute el procesamiento con: .\scripts\windows\process_full_dataset.ps1"
        }
        
        return $true
    } else {
        Write-Error "HDFS no responde"
        return $false
    }
}

# Mostrar métricas del clúster
function Show-ClusterMetrics {
    Write-Step "Métricas del clúster HDFS..."
    
    # Obtener reporte de HDFS
    $report = docker exec climaxtreme-namenode hdfs dfsadmin -report 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        # Extraer información relevante
        $liveNodes = ($report | Select-String "Live datanodes" | ForEach-Object { $_.Line })
        $capacity = ($report | Select-String "Configured Capacity:" | ForEach-Object { $_.Line } | Select-Object -First 1)
        $used = ($report | Select-String "DFS Used:" | ForEach-Object { $_.Line } | Select-Object -First 1)
        
        Write-Info $liveNodes
        Write-Info $capacity
        Write-Info $used
    }
}

# Abrir UIs en navegador
function Open-WebInterfaces {
    Write-Step "Abriendo interfaces web..."
    
    $urls = @{
        "HDFS NameNode" = "http://localhost:9870"
        "Dashboard" = "http://localhost:8501"
    }
    
    foreach ($name in $urls.Keys) {
        $url = $urls[$name]
        Write-Info "$name - $url"
        
        if ($OpenBrowser) {
            Start-Process $url
            Start-Sleep -Milliseconds 500
        }
    }
    
    # Verificar si Spark UI está disponible
    try {
        $sparkCheck = Invoke-WebRequest -Uri "http://localhost:4040" -TimeoutSec 2 -ErrorAction SilentlyContinue
        if ($sparkCheck.StatusCode -eq 200) {
            Write-Info "Spark UI - http://localhost:4040 (activo durante jobs)"
            if ($OpenBrowser) {
                Start-Process "http://localhost:4040"
            }
        }
    } catch {
        Write-Info "Spark UI (puerto 4040) - disponible durante ejecución de jobs"
    }
}

# Ejecutar procesamiento de demostración
function Start-DemoProcessing {
    Write-Step "Ejecutando procesamiento de demostración..."
    
    Write-Info "Este proceso puede tomar varios minutos"
    Write-Info "Puede monitorear el progreso en Spark UI: http://localhost:4040"
    
    $processScript = Join-Path $projectRoot "scripts\windows\process_full_dataset.ps1"
    
    if (Test-Path $processScript) {
        Write-Info "Iniciando procesamiento..."
        & $processScript -SkipDownload
    } else {
        Write-Warning "Script de procesamiento no encontrado"
    }
}

# Mostrar resumen para presentación
function Show-PresentationSummary {
    Write-Step "Resumen para la Presentación"
    
    Write-Host ""
    Write-Host "┌──────────────────────────────────────────────────────────────────┐" -ForegroundColor White
    Write-Host "│                    PUNTOS CLAVE PARA EXPONER                     │" -ForegroundColor White
    Write-Host "├──────────────────────────────────────────────────────────────────┤" -ForegroundColor White
    Write-Host "│                                                                  │" -ForegroundColor White
    Write-Host "│  1. ARQUITECTURA:                                                │" -ForegroundColor White
    Write-Host "│     - HDFS con 1 NameNode + 3 DataNodes                         │" -ForegroundColor Cyan
    Write-Host "│     - Factor de replicación: 3                                   │" -ForegroundColor Cyan
    Write-Host "│     - Procesamiento con Apache Spark (PySpark)                   │" -ForegroundColor Cyan
    Write-Host "│                                                                  │" -ForegroundColor White
    Write-Host "│  2. DATASET:                                                     │" -ForegroundColor White
    Write-Host "│     - Berkeley Earth Climate Data                                │" -ForegroundColor Cyan
    Write-Host "│     - ~8.6 millones de registros                                 │" -ForegroundColor Cyan
    Write-Host "│     - Temperaturas globales desde 1750                           │" -ForegroundColor Cyan
    Write-Host "│                                                                  │" -ForegroundColor White
    Write-Host "│  3. PROCESAMIENTO:                                               │" -ForegroundColor White
    Write-Host "│     - 11 archivos Parquet generados                              │" -ForegroundColor Cyan
    Write-Host "│     - Agregaciones temporales y espaciales                       │" -ForegroundColor Cyan
    Write-Host "│     - Análisis estadístico (EDA)                                 │" -ForegroundColor Cyan
    Write-Host "│                                                                  │" -ForegroundColor White
    Write-Host "│  4. VISUALIZACIÓN:                                               │" -ForegroundColor White
    Write-Host "│     - Dashboard Streamlit con 13 páginas                         │" -ForegroundColor Cyan
    Write-Host "│     - Mapas de calor, tendencias, predicciones                   │" -ForegroundColor Cyan
    Write-Host "│                                                                  │" -ForegroundColor White
    Write-Host "│  5. MACHINE LEARNING:                                            │" -ForegroundColor White
    Write-Host "│     - Modelos: Linear, Ridge, Random Forest                      │" -ForegroundColor Cyan
    Write-Host "│     - Ensemble: VotingRegressor                                  │" -ForegroundColor Cyan
    Write-Host "│     - Predicción de intensidad de eventos                        │" -ForegroundColor Cyan
    Write-Host "│                                                                  │" -ForegroundColor White
    Write-Host "└──────────────────────────────────────────────────────────────────┘" -ForegroundColor White
    Write-Host ""
    
    Write-Host "🔗 URLs para mostrar:" -ForegroundColor Yellow
    Write-Host "   • HDFS Web UI:    http://localhost:9870" -ForegroundColor White
    Write-Host "   • Spark UI:       http://localhost:4040 (durante jobs)" -ForegroundColor White
    Write-Host "   • Dashboard:      http://localhost:8501" -ForegroundColor White
    Write-Host ""
}

# Flujo principal
function Main {
    Show-Banner
    
    Write-Host "📋 Modo de ejecución: $Mode" -ForegroundColor Yellow
    Write-Host ""
    
    # 1. Verificar Docker
    if (-not (Test-Docker)) {
        Write-Host ""
        Write-Host "⚠️  Por favor, inicie Docker Desktop y ejecute este script nuevamente." -ForegroundColor Yellow
        return
    }
    
    # 2. Verificar contenedores
    $containerStatus = Test-Containers
    
    if (-not $containerStatus.AllRunning) {
        Write-Host ""
        $response = Read-Host "¿Desea iniciar los contenedores? (S/N)"
        if ($response -eq "S" -or $response -eq "s") {
            Start-Containers
            $containerStatus = Test-Containers
        }
    }
    
    # 3. Verificar HDFS
    Test-HDFS
    
    # 4. Mostrar métricas
    Show-ClusterMetrics
    
    # 5. Modo específico
    switch ($Mode) {
        "full" {
            Start-DemoProcessing
            Open-WebInterfaces
        }
        "quick" {
            Open-WebInterfaces
        }
        "status" {
            # Solo mostrar estado (ya hecho arriba)
        }
    }
    
    # 6. Mostrar resumen
    Show-PresentationSummary
    
    Write-Host ""
    Write-Host "✅ Demo lista!" -ForegroundColor Green
    Write-Host ""
}

# Ejecutar
Main
