# Scripts de climaXtreme

Este directorio contiene scripts para configurar y ejecutar el sistema climaXtreme, organizados por sistema operativo.

## Estructura

```
scripts/
├── windows/                       # Scripts para Windows (PowerShell)
│   ├── check_status.ps1           # Verificar estado del sistema
│   ├── hdfs_setup_and_load.ps1    # Configurar HDFS y cargar datos
│   ├── process_full_dataset.ps1   # Pipeline completo de procesamiento
│   ├── monitor_cluster_metrics.ps1 # Monitoreo de métricas del clúster
│   ├── measure_execution_times.ps1 # Medir tiempos de ejecución
│   └── demo_presentation.ps1      # Script para exposición/demo
│
├── linux/                         # Scripts para Linux/macOS (Bash)
│   ├── check_status.sh
│   ├── hdfs_setup_and_load.sh
│   └── process_full_dataset.sh
│
└── generate_metrics_charts.py     # Generar gráficos de métricas (Python)
```

## Scripts Disponibles

### 1. `check_status` - Verificar Estado del Sistema

Verifica el estado completo del sistema climaXtreme:
- Contenedores Docker (namenode, datanode, processor, dashboard)
- Archivos en HDFS
- Archivos procesados
- Estadísticas y tamaños

**Windows:**
```powershell
.\scripts\windows\check_status.ps1
```

**Linux/macOS:**
```bash
bash scripts/linux/check_status.sh
# O darle permisos de ejecución:
chmod +x scripts/linux/check_status.sh
./scripts/linux/check_status.sh
```

---

### 2. `hdfs_setup_and_load` - Configurar HDFS y Cargar Datos

Inicia HDFS y carga datos al cluster.

**Windows:**
```powershell
# Cargar archivo completo
.\scripts\windows\hdfs_setup_and_load.ps1 -FullFile

# Cargar muestra (100k filas)
.\scripts\windows\hdfs_setup_and_load.ps1 -Head 100000

# Especificar archivo CSV
.\scripts\windows\hdfs_setup_and_load.ps1 -CsvPath "path\to\file.csv" -FullFile
```

**Linux/macOS:**
```bash
# Cargar archivo completo
bash scripts/linux/hdfs_setup_and_load.sh --full-file

# Cargar muestra (100k filas)
bash scripts/linux/hdfs_setup_and_load.sh --head 100000

# Especificar archivo CSV
bash scripts/linux/hdfs_setup_and_load.sh --csv-path "path/to/file.csv" --full-file
```

---

### 3. `process_full_dataset` - Pipeline Completo Automático

Ejecuta el pipeline completo: carga, procesamiento y descarga (opcional).

**Windows:**
```powershell
# Pipeline completo (upload + procesamiento + download)
.\scripts\windows\process_full_dataset.ps1

# Sin descarga (HDFS como única fuente - RECOMENDADO)
.\scripts\windows\process_full_dataset.ps1 -SkipDownload

# Solo procesamiento (datos ya en HDFS)
.\scripts\windows\process_full_dataset.ps1 -SkipUpload

# Solo procesamiento (sin upload ni download)
.\scripts\windows\process_full_dataset.ps1 -SkipUpload -SkipDownload
```

**Linux/macOS:**
```bash
# Pipeline completo (upload + procesamiento + download)
bash scripts/linux/process_full_dataset.sh

# Sin descarga (HDFS como única fuente - RECOMENDADO)
bash scripts/linux/process_full_dataset.sh --skip-download

# Solo procesamiento (datos ya en HDFS)
bash scripts/linux/process_full_dataset.sh --skip-upload

# Solo procesamiento (sin upload ni download)
bash scripts/linux/process_full_dataset.sh --skip-upload --skip-download
```

---

### 4. `monitor_cluster_metrics` - Monitoreo de Métricas (Windows)

Captura métricas de CPU, RAM y disco de los contenedores Docker durante la ejecución de jobs.
Genera archivos CSV para análisis posterior.

```powershell
# Monitoreo de 5 minutos con intervalo de 5 segundos (default)
.\scripts\windows\monitor_cluster_metrics.ps1

# Personalizar duración e intervalo
.\scripts\windows\monitor_cluster_metrics.ps1 -Duration 600 -Interval 10

# Los resultados se guardan en DATA/metrics/
```

**Archivos generados:**
- `cluster_metrics_YYYYMMDD_HHMMSS.csv` - Métricas detalladas
- `metrics_summary_YYYYMMDD_HHMMSS.txt` - Resumen estadístico

---

### 5. `measure_execution_times` - Medir Tiempos de Ejecución (Windows)

Ejecuta el pipeline de procesamiento midiendo el tiempo de cada operación.
Útil para el informe técnico y análisis de rendimiento.

```powershell
# Medir tiempos del pipeline completo
.\scripts\windows\measure_execution_times.ps1

# Los resultados se guardan en DATA/performance/
```

**Métricas capturadas:**
- Tiempo de inicialización de Spark
- Tiempo de lectura CSV desde HDFS
- Tiempo de limpieza de datos
- Tiempo de cada agregación
- Throughput (registros/segundo)

---

### 6. `demo_presentation` - Script de Demostración (Windows)

Script automatizado para la exposición oral del proyecto.
Verifica infraestructura, abre interfaces web y muestra puntos clave.

```powershell
# Demo rápida (solo verificación + dashboard)
.\scripts\windows\demo_presentation.ps1 -Mode quick -OpenBrowser

# Demo completa (incluye procesamiento)
.\scripts\windows\demo_presentation.ps1 -Mode full

# Solo verificar estado
.\scripts\windows\demo_presentation.ps1 -Mode status
```

**Características:**
- Banner visual del proyecto
- Verificación de Docker y contenedores
- Estado de HDFS y datos procesados
- Apertura automática de interfaces web
- Resumen de puntos clave para la presentación

---

### 7. `generate_metrics_charts.py` - Generar Gráficos (Python)

Genera gráficos de métricas a partir del CSV capturado por `monitor_cluster_metrics.ps1`.

```bash
# Usar archivo más reciente
python scripts/generate_metrics_charts.py

# Especificar archivo
python scripts/generate_metrics_charts.py DATA/metrics/cluster_metrics_XXXXXXXX.csv
```

**Gráficos generados:**
- `cpu_usage_chart.png` - Uso de CPU por contenedor
- `memory_usage_chart.png` - Uso de memoria por contenedor
- `metrics_dashboard.png` - Dashboard combinado
- `statistics_summary.csv` - Tabla de estadísticas
- `statistics_summary.md` - Estadísticas en Markdown

---

## Flujo de Trabajo Recomendado

### Primera Vez (Setup Completo)

**Windows:**
```powershell
# 1. Levantar contenedores
cd infra
docker-compose up -d

# 2. Procesar dataset completo
cd ..
.\scripts\windows\process_full_dataset.ps1 -SkipDownload

# 3. Dashboard ya está corriendo en http://localhost:8501
```

**Linux/macOS:**
```bash
# 1. Levantar contenedores
cd infra
docker-compose up -d

# 2. Procesar dataset completo
cd ..
bash scripts/linux/process_full_dataset.sh --skip-download

# 3. Dashboard ya está corriendo en http://localhost:8501
```

### Desarrollo (Cambios en Código)

**Windows:**
```powershell
# 1. Modificar código en Tools/src/climaxtreme/

# 2. Reconstruir contenedor
cd infra
docker-compose build processor
docker-compose restart processor

# 3. Reprocesar (sin re-upload)
cd ..
.\scripts\windows\process_full_dataset.ps1 -SkipUpload -SkipDownload

# 4. Ver estado
.\scripts\windows\check_status.ps1
```

**Linux/macOS:**
```bash
# 1. Modificar código en Tools/src/climaxtreme/

# 2. Reconstruir contenedor
cd infra
docker-compose build processor
docker-compose restart processor

# 3. Reprocesar (sin re-upload)
cd ..
bash scripts/linux/process_full_dataset.sh --skip-upload --skip-download

# 4. Ver estado
bash scripts/linux/check_status.sh
```

### Testing Rápido (Muestra Pequeña)

**Windows:**
```powershell
# Cargar solo 100k filas
.\scripts\windows\hdfs_setup_and_load.ps1 -Head 100000

# Procesar manualmente
docker exec climaxtreme-processor climaxtreme preprocess `
  --input-path "hdfs://climaxtreme-namenode:9000/data/climaxtreme/GlobalLandTemperaturesByCity_sample.csv" `
  --output-path "hdfs://climaxtreme-namenode:9000/data/climaxtreme/processed" `
  --format city-csv
```

**Linux/macOS:**
```bash
# Cargar solo 100k filas
bash scripts/linux/hdfs_setup_and_load.sh --head 100000

# Procesar manualmente
docker exec climaxtreme-processor climaxtreme preprocess \
  --input-path "hdfs://climaxtreme-namenode:9000/data/climaxtreme/GlobalLandTemperaturesByCity_sample.csv" \
  --output-path "hdfs://climaxtreme-namenode:9000/data/climaxtreme/processed" \
  --format city-csv
```

---

## Permisos (Linux/macOS)

Si necesitas dar permisos de ejecución a los scripts:

```bash
chmod +x scripts/linux/*.sh
```

## Diferencias entre Plataformas

| Característica | Windows | Linux/macOS |
|----------------|---------|-------------|
| **Lenguaje** | PowerShell (.ps1) | Bash (.sh) |
| **Paths** | `\` (backslash) | `/` (forward slash) |
| **Comandos** | `docker-compose` o `docker compose` | `docker-compose` o `docker compose` |
| **Colors** | `-ForegroundColor` | ANSI escape codes |
| **Flags** | `-ParameterName` | `--parameter-name` |

## Solución de Problemas

### Windows: "No se puede ejecutar el script"

```powershell
# Cambiar política de ejecución (solo primera vez)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Linux/macOS: "Permission denied"

```bash
# Dar permisos de ejecución
chmod +x scripts/linux/*.sh
```

### "Docker no está corriendo"

1. Abre Docker Desktop
2. Espera a que esté en estado "Running"
3. Verifica: `docker info`

### "Contenedor no arranca"

```bash
# Ver logs
docker logs climaxtreme-namenode
docker logs climaxtreme-processor

# Reiniciar desde cero
cd infra
docker-compose down -v
docker-compose up -d
```

## Documentación Adicional

- **Guía de Setup HDFS**: Ver `HDFS_SETUP_GUIDE.md`
- **Estructura de Parquets**: Ver `PARQUETS.md`
- **Análisis EDA**: Ver `EDA_IMPLEMENTATION.md`
- **Dashboard en Docker**: Ver `DOCKER_DASHBOARD.md`

## Contribuir

Al agregar nuevos scripts:

1. Crear versión Windows (.ps1) en `scripts/windows/`
2. Crear versión Linux (.sh) en `scripts/linux/`
3. Mantener funcionalidad equivalente entre ambas versiones
4. Actualizar este README con el nuevo script
5. Actualizar `HDFS_SETUP_GUIDE.md` si es necesario
