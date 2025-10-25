param(
  [string]$CsvPath = "DATA/GlobalLandTemperaturesByCity.csv",
  [string]$HdfsDir = "/data/climaxtreme",
  [int]$Head = 0,  # 0 = upload full file, otherwise upload first N rows as sample
  [switch]$FullFile  # Explicitly upload the complete file
)

$ErrorActionPreference = "Stop"

# Resolve repo root based on script location
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot ".."))
$ComposeFile = Join-Path $RepoRoot "infra\docker-compose.yml"

if (-not (Test-Path $ComposeFile)) {
  Write-Error "No se encontró el archivo de Docker Compose en: $ComposeFile"
}

# Resolve CSV absolute path (allow relative to repo root)
if (-not ([System.IO.Path]::IsPathRooted($CsvPath))) {
  $CsvPath = Join-Path $RepoRoot $CsvPath
}
if (-Not (Test-Path $CsvPath)) {
  Write-Error "No se encontró el archivo CSV: $CsvPath"; exit 1
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
  Write-Error "Docker no está instalado o no está en PATH. Instala Docker Desktop y vuelve a intentarlo."; exit 1
}

try {
  docker info | Out-Null
} catch {
  Write-Error "Docker Desktop no está en ejecución. Ábrelo y espera a que esté 'Running' antes de continuar."; exit 1
}

function Invoke-Compose {
  param([Parameter(Mandatory=$true)][string[]]$Args)
  if (Get-Command docker-compose -ErrorAction SilentlyContinue) {
    & docker-compose @Args
    return $LASTEXITCODE
  } else {
    & docker compose @Args
    return $LASTEXITCODE
  }
}

Write-Host "Iniciando HDFS (Docker Compose)..."
Write-Host "Descargando imágenes si es necesario (esto puede tardar 2-3 min la primera vez)..."
$composeResult = Invoke-Compose -Args @('-f', "$ComposeFile", 'up', '-d', 'namenode', 'datanode')

if ($composeResult -ne 0) {
  Write-Host ""
  Write-Host "ERROR: Docker Compose falló al iniciar los contenedores." -ForegroundColor Red
  Write-Host ""
  Write-Host "Posibles causas:" -ForegroundColor Yellow
  Write-Host "  1. Docker Desktop no está corriendo correctamente"
  Write-Host "  2. Error al descargar imágenes (verifica tu conexión a internet)"
  Write-Host "  3. Puerto 9870 o 9000 ya está en uso"
  Write-Host ""
  Write-Host "Soluciones:" -ForegroundColor Yellow
  Write-Host "  - Reinicia Docker Desktop y vuelve a intentar"
  Write-Host "  - Verifica logs: docker compose -f '$ComposeFile' logs"
  Write-Host "  - Limpia contenedores previos: docker compose -f '$ComposeFile' down -v"
  Write-Host ""
  exit 1
}

Write-Host "Esperando a que HDFS inicialice..."
# Espera a que el contenedor de NameNode aparezca y esté corriendo
$maxWait = 90
$start = Get-Date
$ok = $false

do {
  Start-Sleep -Seconds 3
  $status = docker inspect climaxtreme-namenode --format '{{.State.Status}}' 2>$null
  
  if ($status -eq 'running') {
    # Container is running, check if it has a healthcheck
    $health = docker inspect climaxtreme-namenode --format '{{.State.Health.Status}}' 2>$null
    
    if ($health -eq 'healthy') {
      $ok = $true
    } elseif (!$health) {
      # No healthcheck defined, wait a bit and assume it's ok
      Write-Host "Contenedor iniciado (sin healthcheck), esperando 10s adicionales..."
      Start-Sleep -Seconds 10
      $ok = $true
    } elseif ($health -eq 'starting') {
      Write-Host "NameNode iniciando (health: starting)..."
    } else {
      Write-Host "NameNode health: $health"
    }
  } elseif ($status) {
    Write-Host "NameNode status: $status (esperando 'running')..."
  }
} while(-not $ok -and ((Get-Date) - $start).TotalSeconds -lt $maxWait)

if (-not $ok) {
  Write-Host ""
  Write-Host "ERROR: El contenedor NameNode no arrancó correctamente en $maxWait segundos." -ForegroundColor Red
  Write-Host ""
  Write-Host "Ver logs del contenedor:" -ForegroundColor Yellow
  Write-Host "  docker logs climaxtreme-namenode"
  Write-Host "  docker logs climaxtreme-datanode"
  Write-Host ""
  Write-Host "Ver estado:" -ForegroundColor Yellow
  Write-Host "  docker ps -a | Select-String climaxtreme"
  Write-Host ""
  exit 1
}

# Determine if uploading full file or sample
$UploadFullFile = $FullFile -or ($Head -eq 0)

if ($UploadFullFile) {
  Write-Host "Modo: Subir archivo COMPLETO a HDFS (esto puede tardar varios minutos para archivos grandes)"
  $FileToUpload = $CsvPath
  $HdfsFileName = "GlobalLandTemperaturesByCity.csv"
} else {
  Write-Host "Modo: Subir SAMPLE ($Head filas) a HDFS"
  $sample = Join-Path $env:TEMP "climaxtreme_sample.csv"
  Write-Host "Creando sample ($Head filas) desde $CsvPath -> $sample"
  Get-Content -Path $CsvPath -TotalCount ($Head + 1) | Set-Content -Path $sample
  $FileToUpload = $sample
  $HdfsFileName = "GlobalLandTemperaturesByCity_sample.csv"
}

Write-Host "Creando directorio en HDFS: $HdfsDir"
docker exec climaxtreme-namenode hdfs dfs -mkdir -p $HdfsDir | Out-Null

Write-Host "Subiendo archivo a HDFS..."
Write-Host "  Origen: $FileToUpload"
Write-Host "  Destino HDFS: $HdfsDir/$HdfsFileName"

# Get file size for progress indication
$FileSize = (Get-Item $FileToUpload).Length
$FileSizeMB = [math]::Round($FileSize / 1MB, 2)
Write-Host "  Tamaño: $FileSizeMB MB"

if ($FileSizeMB -gt 100) {
  Write-Host "  NOTA: Este archivo es grande. La carga puede tardar varios minutos..."
}

# Copy file to container's temp directory
docker cp $FileToUpload climaxtreme-namenode:/tmp/upload_file.csv | Out-Null

# Upload to HDFS with progress
Write-Host "  Copiando a HDFS..."
docker exec climaxtreme-namenode hdfs dfs -put -f /tmp/upload_file.csv "$HdfsDir/$HdfsFileName"

# Clean up temp file in container
docker exec climaxtreme-namenode rm -f /tmp/upload_file.csv | Out-Null

Write-Host "Contenido de ${HdfsDir}:"
docker exec climaxtreme-namenode hdfs dfs -ls $HdfsDir

Write-Host ""
Write-Host "✓ Listo. Archivo subido exitosamente a HDFS" -ForegroundColor Green
Write-Host "  URL HDFS: hdfs://climaxtreme-namenode:9000$HdfsDir/$HdfsFileName"
Write-Host ""

if ($UploadFullFile) {
  Write-Host "Para procesar este dataset completo, ejecuta:" -ForegroundColor Cyan
  Write-Host "  docker exec -it climaxtreme-processor python -m climaxtreme.cli preprocess \"
    --input-path hdfs://climaxtreme-namenode:9000$HdfsDir/$HdfsFileName \"
    --output-path hdfs://climaxtreme-namenode:9000$HdfsDir/processed \"
    --format city-csv" -ForegroundColor Yellow
}
