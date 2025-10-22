param(
  [string]$CsvPath = "DATA/GlobalLandTemperaturesByCity.csv",
  [string]$HdfsDir = "/data/climaxtreme",
  [int]$Head = 100000
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
    & docker-compose @Args | Out-Null
  } else {
    & docker compose @Args | Out-Null
  }
}

Write-Host "Iniciando HDFS (Docker Compose)..."
Invoke-Compose -Args @('-f', "$ComposeFile", 'up', '-d', 'namenode', 'datanode')

Write-Host "Esperando a que HDFS inicialice..."
# Espera a que el contenedor de NameNode aparezca en 'docker ps' y esté "healthy"
$maxWait = 60
$start = Get-Date
do {
  Start-Sleep -Seconds 3
  $status = docker inspect climaxtreme-namenode --format '{{.State.Status}}' 2>$null
  $health = docker inspect climaxtreme-namenode --format '{{.State.Health.Status}}' 2>$null
  $ok = ($status -eq 'running') -and (($health -eq 'healthy') -or (!$health))
  if ($status -eq 'running' -and !$health) {
    # Container doesn't have healthcheck; just wait a bit more
    Start-Sleep -Seconds 5
    $ok = $true
  }
} while(-not $ok -and ((Get-Date) - $start).TotalSeconds -lt $maxWait)

if (-not $ok) {
  Write-Error "No se pudo iniciar el contenedor 'climaxtreme-namenode'. Revisa Docker Desktop y logs con: docker logs climaxtreme-namenode"; exit 1
}

$sample = Join-Path $env:TEMP "climaxtreme_sample.csv"
Write-Host "Creando sample ($Head filas) desde $CsvPath -> $sample"
Get-Content -Path $CsvPath -TotalCount ($Head + 1) | Set-Content -Path $sample

Write-Host "Creando directorio en HDFS: $HdfsDir"
docker exec climaxtreme-namenode hdfs dfs -mkdir -p $HdfsDir | Out-Null

Write-Host "Subiendo sample a HDFS..."
docker cp $sample climaxtreme-namenode:/tmp/sample.csv | Out-Null
docker exec climaxtreme-namenode hdfs dfs -put -f /tmp/sample.csv "$HdfsDir/GlobalLandTemperaturesByCity_sample.csv"

Write-Host "Contenido de ${HdfsDir}:"
docker exec climaxtreme-namenode hdfs dfs -ls $HdfsDir

Write-Host "Listo. URL HDFS: hdfs://climaxtreme-namenode:9000$HdfsDir/GlobalLandTemperaturesByCity_sample.csv"
