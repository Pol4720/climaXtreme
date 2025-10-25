# Scripts de climaXtreme

Este directorio contiene scripts para configurar y ejecutar el sistema climaXtreme, organizados por sistema operativo.

## Estructura

```
scripts/
├── windows/          # Scripts para Windows (PowerShell)
│   ├── check_status.ps1
│   ├── hdfs_setup_and_load.ps1
│   └── process_full_dataset.ps1
│
└── linux/            # Scripts para Linux/macOS (Bash)
    ├── check_status.sh
    ├── hdfs_setup_and_load.sh
    └── process_full_dataset.sh
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
