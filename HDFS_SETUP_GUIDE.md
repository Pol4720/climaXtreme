# Guía de Configuración HDFS para climaXtreme

> **Nota**: Esta guía soporta **Windows (PowerShell)** y **Linux/macOS (Bash)**. Los comandos específicos están claramente marcados.

## � Selecciona tu Sistema Operativo

Esta guía proporciona comandos para ambas plataformas:

| Sistema Operativo | Shell | Scripts ubicados en |
|-------------------|-------|---------------------|
| **Windows** | PowerShell | `scripts/windows/` |
| **Linux/macOS** | Bash | `scripts/linux/` |

### Diferencias Clave

| Característica | Windows | Linux/macOS |
|----------------|---------|-------------|
| **Scripts** | `.ps1` (PowerShell) | `.sh` (Bash) |
| **Paths** | `\` (backslash) | `/` (forward slash) |
| **Flags** | `-ParameterName` | `--parameter-name` |
| **Ejecución** | `.\script.ps1` | `bash script.sh` o `./script.sh` |

### Configuración Inicial

**Windows**: No requiere configuración adicional (PowerShell está incluido)

**Linux/macOS**: Dar permisos de ejecución a los scripts (solo primera vez)
```bash
chmod +x scripts/linux/*.sh
```

---

## �🚀 Quick Start (2 comandos)

### Windows (PowerShell)
```powershell
# 1. Levantar todos los contenedores (HDFS + Dashboard)
cd infra
docker-compose up -d

# 2. Procesar dataset completo (automático)
cd ..
.\scripts\windows\process_full_dataset.ps1 -SkipDownload

# 3. Abrir dashboard: http://localhost:8501
```

### Linux/macOS (Bash)
```bash
# 1. Levantar todos los contenedores (HDFS + Dashboard)
cd infra
docker-compose up -d

# 2. Procesar dataset completo (automático)
cd ..
bash scripts/linux/process_full_dataset.sh --skip-download

# 3. Abrir dashboard: http://localhost:8501
```

**Resultado**: Abre http://localhost:8501 → Selecciona "HDFS" en el sidebar → ¡Listo! 🎉

**Tiempo total: ~30-40 minutos** (la mayor parte es procesamiento automático en Docker)

**Nota:** Todo se ejecuta en **Docker** (HDFS + Processor + Dashboard). El dashboard ya está levantado con `docker-compose up -d`.

---

## Requisitos Previos

1. **Docker Desktop** instalado y en ejecución (WSL2 recomendado)
2. **Dataset** en `DATA/GlobalLandTemperaturesByCity.csv`

**NOTA**: Todo el procesamiento se realiza dentro de contenedores Docker. No necesitas instalar Python, PySpark ni Java en tu máquina local.

## Paso 1: Verificar Docker Desktop

Antes de ejecutar cualquier comando, asegúrate que Docker Desktop está corriendo:

```powershell
docker info
```

Si ves un error, abre Docker Desktop y espera a que muestre "Running" en verde.

## Paso 2: Iniciar Contenedores HDFS y Processor

Desde el directorio `infra` en PowerShell:

```powershell
cd infra
docker-compose up -d
```

Esto iniciará **3 contenedores**:
- **climaxtreme-namenode**: HDFS NameNode (puerto 9870 para UI, 9000 para datos)
- **climaxtreme-datanode**: HDFS DataNode
- **climaxtreme-processor**: Contenedor con Python 3.9, PySpark y Java 17 para procesamiento

**Tiempo esperado**: 
- Primera vez: 5-10 minutos (descarga de imágenes Docker + build del processor)
- Siguientes veces: 10-30 segundos

### Verificar que los contenedores están corriendo:

```powershell
docker-compose ps
```

Deberías ver los 3 contenedores en estado "Running" o "Up", y namenode/datanode con "(healthy)".

## Paso 3: Procesamiento Completo Automático (Recomendado)

**Usa el script unificado que hace TODO automáticamente:**

### Windows (PowerShell)
```powershell
# Desde el directorio raíz del proyecto
.\scripts\windows\process_full_dataset.ps1
```

### Linux/macOS (Bash)
```bash
# Desde el directorio raíz del proyecto
bash scripts/linux/process_full_dataset.sh
```

Este script ejecuta el **pipeline completo**:
1. ✅ Verifica que Docker esté corriendo
2. ✅ Inicia contenedores HDFS (namenode + datanode + processor)
3. ✅ Sube el dataset completo a HDFS
4. ✅ Ejecuta procesamiento Spark (genera 11 archivos Parquet)
5. ✅ (Opcional) Descarga resultados a `DATA/processed/`

**Parámetros disponibles:**

### Windows (PowerShell)
```powershell
# Procesar todo (default: sube, procesa, y descarga)
.\scripts\windows\process_full_dataset.ps1

# Procesar sin descargar (datos quedan solo en HDFS - RECOMENDADO)
.\scripts\windows\process_full_dataset.ps1 -SkipDownload

# Solo procesar (asume que datos ya están en HDFS)
.\scripts\windows\process_full_dataset.ps1 -SkipUpload

# Solo procesar (sin upload ni download)
.\scripts\windows\process_full_dataset.ps1 -SkipUpload -SkipDownload

# Especificar ruta del CSV
.\scripts\windows\process_full_dataset.ps1 -CsvPath "DATA\MiArchivo.csv"
```

### Linux/macOS (Bash)
```bash
# Procesar todo (default: sube, procesa, y descarga)
bash scripts/linux/process_full_dataset.sh

# Procesar sin descargar (datos quedan solo en HDFS - RECOMENDADO)
bash scripts/linux/process_full_dataset.sh --skip-download

# Solo procesar (asume que datos ya están en HDFS)
bash scripts/linux/process_full_dataset.sh --skip-upload

# Solo procesar (sin upload ni download)
bash scripts/linux/process_full_dataset.sh --skip-upload --skip-download

# Especificar ruta del CSV
bash scripts/linux/process_full_dataset.sh --csv-path "DATA/MiArchivo.csv"
```

**Salida esperada:**
```
========================================
  climaXtreme - Procesamiento Completo
========================================

PASO 1/4: Cargando dataset COMPLETO a HDFS...
✓ Dataset cargado exitosamente a HDFS

PASO 2/4: Ejecutando procesamiento con PySpark...
✓ Procesamiento completado (11 archivos generados)

PASO 3/4: Verificando archivos en HDFS...
✓ Todos los archivos Parquet verificados

PASO 4/4: Descargando resultados...
  ✓ monthly.parquet descargado
  ✓ yearly.parquet descargado
  ... (11 archivos total)

=========================================
  ✓ PROCESAMIENTO COMPLETADO
=========================================
```

**Archivos generados en HDFS:**
- `/data/climaxtreme/processed/monthly.parquet`
- `/data/climaxtreme/processed/yearly.parquet`
- `/data/climaxtreme/processed/anomalies.parquet`
- `/data/climaxtreme/processed/climatology.parquet`
- `/data/climaxtreme/processed/seasonal.parquet`
- `/data/climaxtreme/processed/extreme_thresholds.parquet`
- `/data/climaxtreme/processed/regional.parquet`
- `/data/climaxtreme/processed/continental.parquet`
- `/data/climaxtreme/processed/correlation_matrix.parquet`
- `/data/climaxtreme/processed/descriptive_stats.parquet`
- `/data/climaxtreme/processed/chi_square_tests.parquet`

---

## Paso 3 Alternativo: Carga Manual (Solo para desarrollo)

Si necesitas cargar datos manualmente sin procesamiento:

### Windows (PowerShell)
```powershell
# Cargar solo una muestra (100,000 filas)
.\scripts\windows\hdfs_setup_and_load.ps1 -CsvPath "DATA\GlobalLandTemperaturesByCity.csv" -Head 100000

# Cargar archivo completo
.\scripts\windows\hdfs_setup_and_load.ps1 -CsvPath "DATA\GlobalLandTemperaturesByCity.csv" -FullFile
```

**Parámetros del script manual (Windows)**:
- `-CsvPath`: Ruta al archivo CSV original
- `-Head`: Número de filas a cargar (para muestras)
- `-FullFile`: Cargar archivo completo (sin límite de filas)

### Linux/macOS (Bash)
```bash
# Cargar solo una muestra (100,000 filas)
bash scripts/linux/hdfs_setup_and_load.sh --csv-path "DATA/GlobalLandTemperaturesByCity.csv" --head 100000

# Cargar archivo completo
bash scripts/linux/hdfs_setup_and_load.sh --csv-path "DATA/GlobalLandTemperaturesByCity.csv" --full-file
```

**Parámetros del script manual (Linux/macOS)**:
- `--csv-path`: Ruta al archivo CSV original
- `--head`: Número de filas a cargar (para muestras)
- `--full-file`: Cargar archivo completo (sin límite de filas)

## Paso 4: Verificar HDFS

### Opción A: UI Web
Abre en tu navegador: http://localhost:9870

Ve a "Utilities" → "Browse the file system" → Navega a `/data/climaxtreme`

### Opción B: Línea de comandos
```powershell
# Ver archivos raw (CSV original)
docker exec climaxtreme-namenode hdfs dfs -ls /data/climaxtreme

# Ver archivos procesados (Parquet)
docker exec climaxtreme-namenode hdfs dfs -ls /data/climaxtreme/processed

# Ver detalles de un archivo específico
docker exec climaxtreme-namenode hdfs dfs -ls /data/climaxtreme/processed/monthly.parquet

# Ver primeras líneas del CSV
docker exec climaxtreme-namenode hdfs dfs -tail /data/climaxtreme/GlobalLandTemperaturesByCity.csv
```

---

## Paso 5: Lanzar Dashboard de Streamlit

**NOTA**: Para usar el dashboard necesitas instalar el paquete climaxtreme en tu máquina local.

### Instalación local (solo para el dashboard):

```powershell
# Desde el directorio Tools
cd Tools
pip install -e .
```

### Ejecutar dashboard:

```powershell
# El dashboard se ejecuta en DOCKER (contenedor dedicado)
# Ya está levantado si hiciste: docker-compose up -d

# Para iniciarlo por separado:
cd infra
docker-compose up -d dashboard

# Para ver logs del dashboard:
docker logs -f climaxtreme-dashboard
```

Abre: http://localhost:8501

### Configurar fuente de datos:

En el **sidebar del dashboard** verás un selector de fuente de datos:

**Opción 1 - HDFS (Recomendado para Big Data):**
1. Seleccionar: **HDFS (Recommended)**
2. Configurar:
   - HDFS Host: `namenode`
   - HDFS Port: `9000`
   - HDFS Base Path: `/data/climaxtreme/processed`
3. El dashboard leerá directo desde HDFS sin descargar archivos

**Archivos disponibles para visualizar:**
- **monthly.parquet**: Análisis de tendencias mensuales
- **yearly.parquet**: Análisis de tendencias anuales con línea de tendencia
- **anomalies.parquet**: Detección de anomalías climáticas
- **seasonal.parquet**: Patrones estacionales
- **extreme_thresholds.parquet**: Umbrales de eventos extremos
- **regional.parquet**: Análisis por región geográfica (16 regiones) + mapa interactivo 🗺️
- **continental.parquet**: Análisis por continente (7 continentes) + mapa global 🌍
- **correlation_matrix.parquet**: Matriz de correlación de Pearson (EDA) 📊
- **descriptive_stats.parquet**: Estadísticas descriptivas completas (EDA) 📈
- **chi_square_tests.parquet**: Pruebas de independencia Chi-cuadrado (EDA) 🧪

**Pestañas del dashboard (7 en total):**
1. 🌡️ **Temperature Trends**: Tendencias de temperatura a lo largo del tiempo
2. 🗺️ **Heatmaps**: Mapas de calor de temperatura
3. 📈 **Seasonal Analysis**: Análisis estacional
4. ⚡ **Extreme Events**: Eventos climáticos extremos
5. 🌍 **Regional Analysis**: Análisis por región + mapa mundial interactivo
6. 🌐 **Continental Analysis**: Análisis por continente + mapa global con burbujas
7. 📊 **Exploratory Analysis (EDA)**: Correlaciones, estadísticas descriptivas y pruebas Chi-cuadrado

**Opción 2 - Local Files (Para desarrollo/demos):**
1. Seleccionar: **Local Files**
2. Primero descarga los archivos desde HDFS (ver sección de comandos útiles)
3. El dashboard leerá desde `DATA/processed/`

**Ventajas del modo HDFS:**
- ✅ Sin descargas innecesarias
- ✅ HDFS como única fuente de verdad (principio Big Data)
- ✅ Siempre datos actualizados
- ✅ Ahorro de espacio en disco local
- ✅ Acceso a todos los 11 archivos Parquet generados
- ✅ Visualizaciones interactivas (mapas, heatmaps, estadísticas)

## Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────┐
│                    Docker Network (hdfs)                 │
│                                                          │
│  ┌──────────────┐      ┌──────────────┐                │
│  │  NameNode    │◄────►│  DataNode    │                │
│  │  (Port 9870) │      │              │                │
│  │  (Port 9000) │      │              │                │
│  └──────────────┘      └──────────────┘                │
│         ▲                                                │
│         │                                                │
│         │ HDFS Protocol                                 │
│         │                                                │
│  ┌──────▼──────────────────────────────┐                │
│  │       Processor Container           │                │
│  │  - Python 3.9                       │                │
│  │  - PySpark 3.4+                     │                │
│  │  - Java 17 (OpenJDK)                │                │
│  │  - climaxtreme package              │                │
│  └─────────────────────────────────────┘                │
│                                                          │
└─────────────────────────────────────────────────────────┘
         ▲
         │ docker exec
         │
    ┌────┴─────┐
    │ Windows  │
    │ PowerShell│
    └──────────┘
```

## Solución de Problemas

### Error: "image not found" o build muy lento
**Problema**: Docker no puede descargar las imágenes o el build del processor es lento.

**Solución**:
1. Verifica tu conexión a internet
2. La primera vez, el build del processor descarga muchas dependencias Python (~500MB)
3. Si estás detrás de un proxy corporativo, configúralo en Docker Desktop → Settings → Resources → Proxies

### Error: "Unable to load native-hadoop library"
**Problema**: Advertencia al ejecutar PySpark (no es crítico).

**Solución**: Es solo una advertencia, el procesamiento funciona correctamente. Hadoop usa librerías Java en su lugar.

### Error: "TypeError: bad operand type for abs(): 'Column'"
**Problema**: Versión antigua del código en el contenedor.

**Solución**: Reconstruir la imagen del processor:
```powershell
cd infra
docker-compose build processor
docker-compose stop processor
docker-compose up -d processor
```

### Error: "Connection refused" al conectar con HDFS
**Problema**: Intentas ejecutar `climaxtreme` desde tu máquina local en lugar del contenedor.

**Solución**: **SIEMPRE** ejecuta los comandos de procesamiento dentro del contenedor:
```powershell
docker exec climaxtreme-processor climaxtreme preprocess ...
```

NO ejecutes directamente:
```powershell
# ❌ INCORRECTO - No funcionará desde Windows
climaxtreme preprocess --input-path "hdfs://..."

# ✅ CORRECTO - Ejecutar dentro del contenedor
docker exec climaxtreme-processor climaxtreme preprocess --input-path "hdfs://..."
```

### Error: "The system cannot find the file specified"
**Problema**: Docker Desktop no está corriendo.

**Solución**: Abre Docker Desktop y espera a que esté en estado "Running"

### Error: "No se pudo iniciar el contenedor" o contenedores no saludables
**Problema**: Los contenedores no arrancan correctamente.

**Solución**:
1. Ver logs de los contenedores:
   ```powershell
   docker logs climaxtreme-namenode
   docker logs climaxtreme-datanode
   docker logs climaxtreme-processor
   ```
2. Verificar el estado:
   ```powershell
   cd infra
   docker-compose ps
   ```
3. Reiniciar desde cero:
   ```powershell
   docker-compose down -v
   docker-compose up -d
   ```

### Los contenedores se detienen después de unos segundos
**Problema**: El processor está configurado para mantenerse corriendo indefinidamente.

**Solución**: Si el processor se detiene, verifica los logs:
```powershell
docker logs climaxtreme-processor
```

El contenedor debe ejecutar `tail -f /dev/null` para mantenerse activo.

### El dashboard no carga los datos
**Problema**: No se ven archivos disponibles en el dashboard.

**Solución**: 
- **Modo HDFS**: Verifica que los contenedores estén corriendo y que los archivos existan en HDFS:
  ```powershell
  docker exec climaxtreme-namenode hdfs dfs -ls /data/climaxtreme/processed
  ```
- **Modo Local Files**: Verifica que `DATA/processed/` contenga archivos `.parquet`. Si no existen, descárgalos desde HDFS:
  ```powershell
  docker exec climaxtreme-namenode hdfs dfs -get /data/climaxtreme/processed/*.parquet /tmp/
  docker cp climaxtreme-namenode:/tmp/monthly.parquet ./DATA/processed/
  ```

### Error: "Could not connect to HDFS" en el dashboard
**Problema**: El dashboard no puede conectarse al namenode.

**Solución**:
1. Verifica que los contenedores estén corriendo:
   ```powershell
   docker ps | Select-String "namenode"
   ```
2. Verifica la configuración en el sidebar:
   - HDFS Host debe ser: `namenode` (o `localhost` si estás en Windows)
   - HDFS Port debe ser: `9000`
3. Si usas `localhost` como host, puede que necesites usar la IP del contenedor:
   ```powershell
   docker inspect climaxtreme-namenode -f '{{.NetworkSettings.Networks.hdfs.IPAddress}}'
   ```

## Detener HDFS

```powershell
cd infra
docker-compose down
```

Para eliminar también los volúmenes (datos persistentes):
```powershell
docker-compose down -v
```

**NOTA**: Si ejecutas `down -v`, perderás todos los datos en HDFS y tendrás que volver a cargarlos.

## Comandos Útiles

### Docker & Contenedores

```powershell
# Ver todos los contenedores corriendo
docker ps

# Ver logs en tiempo real
docker logs -f climaxtreme-namenode
docker logs -f climaxtreme-processor

# Entrar al contenedor (bash interactivo)
docker exec -it climaxtreme-namenode bash
docker exec -it climaxtreme-processor bash

# Reiniciar un contenedor específico
cd infra
docker-compose restart processor
```

### HDFS (ejecutar desde cualquier lugar)

```powershell
# Listar archivos en HDFS
docker exec climaxtreme-namenode hdfs dfs -ls /
docker exec climaxtreme-namenode hdfs dfs -ls /data/climaxtreme

# Ver las últimas líneas de un archivo
docker exec climaxtreme-namenode hdfs dfs -tail /data/climaxtreme/GlobalLandTemperaturesByCity_sample.csv

# Ver espacio usado en HDFS
docker exec climaxtreme-namenode hdfs dfs -df -h

# Descargar archivo desde HDFS a tu máquina (solo si usas modo Local Files)
docker exec climaxtreme-namenode hdfs dfs -get /data/climaxtreme/processed/monthly.parquet /tmp/
docker cp climaxtreme-namenode:/tmp/monthly.parquet ./

# NOTA: Con modo HDFS en el dashboard, NO necesitas descargar archivos

# Eliminar archivo/directorio en HDFS
docker exec climaxtreme-namenode hdfs dfs -rm -r /data/climaxtreme/processed
```

### Procesamiento (ejecutar desde cualquier lugar)

```powershell
# Ejecutar preprocesamiento
docker exec climaxtreme-processor climaxtreme preprocess `
  --input-path "hdfs://climaxtreme-namenode:9000/data/climaxtreme/GlobalLandTemperaturesByCity_sample.csv" `
  --output-path "hdfs://climaxtreme-namenode:9000/data/climaxtreme/processed" `
  --format city-csv

# Ver la ayuda del CLI
docker exec climaxtreme-processor climaxtreme --help
docker exec climaxtreme-processor climaxtreme preprocess --help

# Ejecutar Python interactivo dentro del processor
docker exec -it climaxtreme-processor python3
```

### Reconstruir el Processor (cuando cambias código)

```powershell
cd infra
docker-compose build processor
docker-compose stop processor
docker-compose up -d processor
```

## Próximos Pasos

1. **Añadir más datos**: Modifica el parámetro `-Head` del script para cargar más filas, o elimínalo para cargar todo el dataset
2. **Dashboard con HDFS**: Usa el modo HDFS en el dashboard para visualizar datos sin descargas
3. **Explorar EDA**: Carga los archivos `correlation_matrix.parquet`, `descriptive_stats.parquet` o `chi_square_tests.parquet` para ver análisis estadístico completo
4. **Mapas interactivos**: Usa `regional.parquet` o `continental.parquet` para ver visualizaciones geográficas del mundo
5. **Análisis avanzado**: Usa PySpark dentro del processor para analizar los datos procesados
6. **Procesamiento en batch**: Crea scripts para procesar múltiples archivos automáticamente
7. **Queries SQL con Spark**: Lee los Parquet desde HDFS y ejecuta queries SQL
8. **Machine Learning**: Entrena modelos usando los datos procesados en formato Parquet

## Flujo de Trabajo Completo

### 🚀 Inicio Rápido (Setup + Procesamiento + Dashboard)

**Un solo comando para todo:**

### Windows (PowerShell)
```powershell
# 1. Levantar TODO (namenode + datanode + processor + dashboard)
cd infra
docker-compose up -d

# 2. Procesar dataset completo (modo HDFS-first)
cd ..
.\scripts\windows\process_full_dataset.ps1 -SkipDownload

# 3. Abrir dashboard (ya está levantado)
# http://localhost:8501
```

### Linux/macOS (Bash)
```bash
# 1. Levantar TODO (namenode + datanode + processor + dashboard)
cd infra
docker-compose up -d

# 2. Procesar dataset completo (modo HDFS-first)
cd ..
bash scripts/linux/process_full_dataset.sh --skip-download

# 3. Abrir dashboard (ya está levantado)
# http://localhost:8501
```

En el dashboard:
- Selecciona "HDFS (Recommended)"
- Host: `namenode`, Port: `9000`, Path: `/data/climaxtreme/processed`
- ¡Listo! Ya puedes visualizar los 11 archivos Parquet

---

### 📋 Setup Inicial (solo una vez)

```powershell
# Levantar todos los contenedores
cd infra
docker-compose up -d

# Esto levanta 4 contenedores:
#   - climaxtreme-namenode (HDFS NameNode)
#   - climaxtreme-datanode (HDFS DataNode)
#   - climaxtreme-processor (PySpark processor)
#   - climaxtreme-dashboard (Streamlit dashboard en puerto 8501)

# Esperar a que estén healthy (~30 segundos)
docker-compose ps

# Verificar que el dashboard esté corriendo
docker logs climaxtreme-dashboard

# Volver al directorio raíz
cd ..
```

---

### ⚙️ Procesamiento Completo (método recomendado)

### Windows (PowerShell)
```powershell
# Opción 1: Todo automático (sube, procesa, descarga)
.\scripts\windows\process_full_dataset.ps1

# Opción 2: Solo HDFS (sin descarga local - RECOMENDADO)
.\scripts\windows\process_full_dataset.ps1 -SkipDownload

# Opción 3: Reprocesar (datos ya en HDFS)
.\scripts\windows\process_full_dataset.ps1 -SkipUpload
```

### Linux/macOS (Bash)
```bash
# Opción 1: Todo automático (sube, procesa, descarga)
bash scripts/linux/process_full_dataset.sh

# Opción 2: Solo HDFS (sin descarga local - RECOMENDADO)
bash scripts/linux/process_full_dataset.sh --skip-download

# Opción 3: Reprocesar (datos ya en HDFS)
bash scripts/linux/process_full_dataset.sh --skip-upload
```

**Tiempo estimado (dataset completo ~8.6M registros):**
- Upload a HDFS: ~2-5 min
- Procesamiento Spark: ~20-30 min
- Download (opcional): ~2-3 min
- **Total: ~25-40 minutos**

---

### 🎨 Visualización con Dashboard

```powershell
# El dashboard ya está corriendo si hiciste: docker-compose up -d
# Simplemente abre: http://localhost:8501

# Para iniciarlo manualmente:
cd infra
docker-compose up -d dashboard

# Ver logs del dashboard:
docker logs -f climaxtreme-dashboard

# Reiniciar dashboard (si haces cambios en el código):
docker-compose restart dashboard

# En el navegador (http://localhost:8501):
#    - Seleccionar "HDFS (Recommended)" en sidebar
#    - Configurar: Host=namenode, Port=9000, Path=/data/climaxtreme/processed
#    - Seleccionar archivo para visualizar (11 opciones disponibles)
#    - Navegar por las 7 pestañas:
#      * Temperature Trends
#      * Heatmaps
#      * Seasonal Analysis
#      * Extreme Events
#      * Regional Analysis (con mapa interactivo del mundo)
#      * Continental Analysis (con mapa global)
#      * Exploratory Analysis (EDA: correlaciones, stats, chi-cuadrado)
```

---

### 🔧 Desarrollo (cuando modificas código)

### Windows (PowerShell)
```powershell
# 1. Hacer cambios en Tools/src/climaxtreme/

# 2a. Si modificas procesamiento (preprocessing, ml, etc):
cd infra
docker-compose build processor
docker-compose restart processor

# 2b. Si modificas dashboard (dashboard/app.py):
docker-compose build dashboard
docker-compose restart dashboard

# 3. Reprocesar datos (sin re-upload)
cd ..
.\scripts\windows\process_full_dataset.ps1 -SkipUpload -SkipDownload

# 4. Verificar resultados
docker exec climaxtreme-namenode hdfs dfs -ls /data/climaxtreme/processed

# 5. Refrescar dashboard en el navegador (Ctrl+R)
```

### Linux/macOS (Bash)
```bash
# 1. Hacer cambios en Tools/src/climaxtreme/

# 2a. Si modificas procesamiento (preprocessing, ml, etc):
cd infra
docker-compose build processor
docker-compose restart processor

# 2b. Si modificas dashboard (dashboard/app.py):
docker-compose build dashboard
docker-compose restart dashboard

# 3. Reprocesar datos (sin re-upload)
cd ..
bash scripts/linux/process_full_dataset.sh --skip-upload --skip-download

# 4. Verificar resultados
docker exec climaxtreme-namenode hdfs dfs -ls /data/climaxtreme/processed

# 5. Refrescar dashboard en el navegador (Ctrl+R)
```

---

### 🧪 Testing y Desarrollo Rápido

Para desarrollar con muestras pequeñas:

### Windows (PowerShell)
```powershell
# 1. Cargar solo 100k filas (rápido)
.\scripts\windows\hdfs_setup_and_load.ps1 -CsvPath "DATA\GlobalLandTemperaturesByCity.csv" -Head 100000

# 2. Procesar manualmente
docker exec climaxtreme-processor climaxtreme preprocess `
  --input-path "hdfs://climaxtreme-namenode:9000/data/climaxtreme/GlobalLandTemperaturesByCity_sample.csv" `
  --output-path "hdfs://climaxtreme-namenode:9000/data/climaxtreme/processed" `
  --format city-csv

# Tiempo: ~2-5 minutos (vs ~30 minutos para dataset completo)
```

### Linux/macOS (Bash)
```bash
# 1. Cargar solo 100k filas (rápido)
bash scripts/linux/hdfs_setup_and_load.sh --csv-path "DATA/GlobalLandTemperaturesByCity.csv" --head 100000

# 2. Procesar manualmente
docker exec climaxtreme-processor climaxtreme preprocess \
  --input-path "hdfs://climaxtreme-namenode:9000/data/climaxtreme/GlobalLandTemperaturesByCity_sample.csv" \
  --output-path "hdfs://climaxtreme-namenode:9000/data/climaxtreme/processed" \
  --format city-csv

# Tiempo: ~2-5 minutos (vs ~30 minutos para dataset completo)
```

---

### 📦 Gestión de Datos

```powershell
# Ver archivos en HDFS
docker exec climaxtreme-namenode hdfs dfs -ls /data/climaxtreme/processed

# Limpiar datos procesados (para reprocesar)
docker exec climaxtreme-namenode hdfs dfs -rm -r /data/climaxtreme/processed

# Limpiar todo (raw + processed)
docker exec climaxtreme-namenode hdfs dfs -rm -r /data/climaxtreme

# Descargar archivo específico
docker exec climaxtreme-namenode hdfs dfs -get /data/climaxtreme/processed/monthly.parquet /tmp/
docker cp climaxtreme-namenode:/tmp/monthly.parquet ./DATA/processed/
```

---

### 🛑 Detener Sistema

```powershell
# Detener solo el dashboard
cd infra
docker-compose stop dashboard

# Detener contenedores (mantiene datos)
docker-compose stop

# Detener y eliminar contenedores (mantiene volúmenes)
docker-compose down

# Detener TODO y eliminar datos (⚠️ Cuidado!)
docker-compose down -v
```

## Recursos Adicionales

- **HDFS Web UI**: http://localhost:9870
- **Documentación PySpark**: https://spark.apache.org/docs/latest/api/python/
- **Docker Compose Docs**: https://docs.docker.com/compose/
- **Documentación Parquet**: Ver `PARQUETS.md` para detalles de estructura de archivos
- **Guía de EDA**: Ver `EDA_IMPLEMENTATION.md` para análisis exploratorio
- **Mapas Interactivos**: Ver `MAPAS_INTERACTIVOS.md` para visualizaciones geográficas
- **Scripts README**: Ver `scripts/README.md` para documentación detallada de scripts

---

## 📚 Referencia Rápida de Comandos

### Gestión de Contenedores

| Acción | Windows (PowerShell) | Linux/macOS (Bash) |
|--------|---------------------|-------------------|
| **Iniciar todo** | `cd infra; docker-compose up -d` | `cd infra && docker-compose up -d` |
| **Detener todo** | `cd infra; docker-compose down` | `cd infra && docker-compose down` |
| **Ver estado** | `.\scripts\windows\check_status.ps1` | `bash scripts/linux/check_status.sh` |
| **Ver logs** | `docker logs -f climaxtreme-namenode` | `docker logs -f climaxtreme-namenode` |

### Procesamiento

| Acción | Windows (PowerShell) | Linux/macOS (Bash) |
|--------|---------------------|-------------------|
| **Pipeline completo** | `.\scripts\windows\process_full_dataset.ps1` | `bash scripts/linux/process_full_dataset.sh` |
| **Sin descarga** | `.\scripts\windows\process_full_dataset.ps1 -SkipDownload` | `bash scripts/linux/process_full_dataset.sh --skip-download` |
| **Solo procesamiento** | `.\scripts\windows\process_full_dataset.ps1 -SkipUpload` | `bash scripts/linux/process_full_dataset.sh --skip-upload` |
| **Cargar muestra** | `.\scripts\windows\hdfs_setup_and_load.ps1 -Head 100000` | `bash scripts/linux/hdfs_setup_and_load.sh --head 100000` |

### HDFS

| Acción | Comando (Mismo en ambos SO) |
|--------|----------------------------|
| **Listar archivos** | `docker exec climaxtreme-namenode hdfs dfs -ls /data/climaxtreme` |
| **Ver procesados** | `docker exec climaxtreme-namenode hdfs dfs -ls /data/climaxtreme/processed` |
| **Eliminar procesados** | `docker exec climaxtreme-namenode hdfs dfs -rm -r /data/climaxtreme/processed` |
| **Ver espacio usado** | `docker exec climaxtreme-namenode hdfs dfs -df -h` |

### Dashboard

| Acción | Comando (Mismo en ambos SO) |
|--------|----------------------------|
| **Iniciar dashboard** | `cd infra && docker-compose up -d dashboard` |
| **Ver logs** | `docker logs -f climaxtreme-dashboard` |
| **Reiniciar** | `docker-compose restart dashboard` |
| **Acceder** | http://localhost:8501 |

---

## 🔍 Solución de Problemas Específicos por SO

### Windows

**"No se puede ejecutar el script"**
```powershell
# Cambiar política de ejecución (solo primera vez)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Paths con espacios**
```powershell
# Usa comillas
.\scripts\windows\process_full_dataset.ps1 -CsvPath "C:\Mi Carpeta\archivo.csv"
```

### Linux/macOS

**"Permission denied"**
```bash
# Dar permisos de ejecución
chmod +x scripts/linux/*.sh
```

**Scripts no usan colores**
```bash
# Asegúrate de estar usando un terminal con soporte ANSI
# O ejecuta: export TERM=xterm-256color
```

**macOS: bc command not found**
```bash
# Instalar bc con Homebrew
brew install bc
```

## Análisis Exploratorio de Datos (EDA)

El sistema genera automáticamente 3 archivos adicionales con análisis estadístico avanzado:

### 1. Matriz de Correlación (correlation_matrix.parquet)

**Contenido:**
- Correlaciones de Pearson entre todas las variables numéricas
- Variables: `year`, `avg_temperature`, `min_temperature`, `max_temperature`, `temperature_range`

**Visualización en dashboard:**
- Heatmap interactivo con escala de colores RdBu_r
- Top 10 correlaciones más fuertes (positivas y negativas)
- Valores de correlación de -1 (negativa perfecta) a +1 (positiva perfecta)

**Ejemplo de uso:**
```powershell
# Cargar en dashboard
# 1. Seleccionar: correlation_matrix.parquet
# 2. Ir a Tab 7: "Exploratory Analysis (EDA)"
# 3. Ver heatmap de correlaciones
```

**Interpretación:**
- |r| > 0.7: Correlación fuerte
- |r| > 0.4: Correlación moderada
- |r| < 0.3: Correlación débil

---

### 2. Estadísticas Descriptivas (descriptive_stats.parquet)

**Contenido:**
- 11 estadísticas por cada variable numérica
- Métricas: count, mean, std_dev, min, Q1, median, Q3, max, IQR, skewness, kurtosis

**Visualización en dashboard:**
- Tabla estilizada con gradiente de color
- Gráfico de barras con error bars (media ± desviación estándar)
- Box plots interactivos mostrando los 5 números resumen

**Ejemplo de uso:**
```powershell
# Cargar en dashboard
# 1. Seleccionar: descriptive_stats.parquet
# 2. Ir a Tab 7: "Exploratory Analysis (EDA)"
# 3. Ver tabla y gráficos de distribución
```

**Interpretación:**
- **Skewness < -1**: Distribución sesgada izquierda
- **Skewness -0.5 a 0.5**: Aproximadamente simétrica
- **Skewness > 1**: Distribución sesgada derecha
- **Kurtosis < 0**: Platicúrtica (colas ligeras)
- **Kurtosis ≈ 0**: Mesocúrtica (normal)
- **Kurtosis > 0**: Leptocúrtica (colas pesadas)

---

### 3. Pruebas Chi-Cuadrado (chi_square_tests.parquet)

**Contenido:**
- Pruebas de independencia entre variables categóricas
- Tests: Continente vs Temperatura, Estación vs Temperatura, Período vs Temperatura
- Incluye: estadístico χ², p-value, grados de libertad, significancia

**Visualización en dashboard:**
- Tabla de resultados con resaltado de tests significativos
- Gráfico de barras del estadístico χ²
- Interpretación textual automática

**Ejemplo de uso:**
```powershell
# Cargar en dashboard
# 1. Seleccionar: chi_square_tests.parquet
# 2. Ir a Tab 7: "Exploratory Analysis (EDA)"
# 3. Ver resultados de pruebas de independencia
```

**Interpretación:**
- **p-value < 0.05**: Variables dependientes (relación significativa) ✅
- **p-value ≥ 0.05**: Variables independientes (no hay relación) ❌

**Ejemplo de resultado:**
```
Test: Continent vs Temperature Category
χ² = 145,678.23
p-value = 0.0000 (< 0.05)
→ Significativo: La temperatura depende del continente
```

---

### Casos de Uso del EDA

**1. Análisis Pre-Modelado:**
- Detectar multicolinealidad antes de entrenar modelos ML
- Identificar variables dependientes
- Verificar distribuciones para decidir transformaciones

**2. Validación de Hipótesis:**
- ¿Las temperaturas están aumentando? → Ver correlación `year` ↔ `avg_temperature`
- ¿Hay diferencias por continente? → Ver chi-square continente vs temperatura
- ¿Las estaciones afectan? → Ver chi-square estación vs temperatura

**3. Detección de Anomalías:**
- Usar IQR para detectar outliers: valores fuera de [Q1 - 1.5×IQR, Q3 + 1.5×IQR]
- Verificar skewness para distribuciones asimétricas
- Analizar kurtosis para eventos extremos

**4. Reportes Científicos:**
- Exportar tablas de estadísticas descriptivas
- Incluir heatmaps de correlación
- Reportar resultados de tests estadísticos

---

## Script Completo de Procesamiento

Para procesar el dataset completo y generar todos los 11 archivos Parquet automáticamente:

```powershell
# Desde el directorio raíz del proyecto
.\scripts\process_full_dataset.ps1
```

Este script ejecuta todo el pipeline:
1. ✅ Verifica Docker
2. ✅ Inicia HDFS (namenode + datanode)
3. ✅ Sube el dataset completo a HDFS
4. ✅ Procesa con PySpark generando 11 archivos:
   - 8 agregaciones climáticas
   - 3 análisis EDA
5. ✅ (Opcional) Descarga resultados a local

**Parámetros disponibles:**
```powershell
# Procesar todo y descargar
.\scripts\process_full_dataset.ps1

# Solo procesar (datos quedan en HDFS)
.\scripts\process_full_dataset.ps1 -SkipDownload

# Solo procesar (asume datos ya en HDFS)
.\scripts\process_full_dataset.ps1 -SkipUpload

# Solo procesar sin upload ni download
.\scripts\process_full_dataset.ps1 -SkipUpload -SkipDownload
```

**Tiempo estimado (dataset completo ~8.6M registros):**
- Upload a HDFS: ~2-5 minutos
- Procesamiento Spark: ~20-30 minutos
  - Agregaciones: ~15 minutos
  - EDA: ~10-15 minutos adicionales
- Download (opcional): ~2-3 minutos
- **Total: ~25-40 minutos**

**Archivos generados (11 total, ~150 MB):**
```
/data/climaxtreme/processed/
├── monthly.parquet              (~80 MB)
├── yearly.parquet               (~30 MB)
├── anomalies.parquet            (~15 MB)
├── climatology.parquet          (~10 MB)
├── seasonal.parquet             (~8 MB)
├── extreme_thresholds.parquet   (~5 MB)
├── regional.parquet             (~1 MB)
├── continental.parquet          (~500 KB)
├── correlation_matrix.parquet   (<1 MB)
├── descriptive_stats.parquet    (<1 MB)
└── chi_square_tests.parquet     (<1 MB)
```
