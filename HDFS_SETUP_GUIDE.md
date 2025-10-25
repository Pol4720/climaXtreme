# Guía de Configuración HDFS para climaXtreme

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

## Paso 3: Cargar Datos a HDFS

Usa el script PowerShell para cargar tus datos:

```powershell
# Volver al directorio raíz del proyecto
cd ..

# Cargar datos (ejemplo con 100,000 filas)
.\scripts\hdfs_setup_and_load.ps1 -CsvPath "DATA\GlobalLandTemperaturesByCity.csv" -Head 100000
```

Esto creará:
- Directorio `/data/climaxtreme` en HDFS
- Archivo `/data/climaxtreme/GlobalLandTemperaturesByCity_sample.csv` con las primeras 100k filas

**Parámetros del script**:
- `-CsvPath`: Ruta al archivo CSV original
- `-Head`: Número de filas a cargar (omitir para cargar todo el archivo)

## Paso 4: Verificar HDFS

### Opción A: UI Web
Abre en tu navegador: http://localhost:9870

Ve a "Utilities" → "Browse the file system" → Navega a `/data/climaxtreme`

### Opción B: Línea de comandos
```powershell
docker exec climaxtreme-namenode hdfs dfs -ls /data/climaxtreme
docker exec climaxtreme-namenode hdfs dfs -tail /data/climaxtreme/GlobalLandTemperaturesByCity_sample.csv
```

## Paso 5: Procesar Datos con PySpark

**IMPORTANTE**: El procesamiento se ejecuta **dentro del contenedor processor**, no en tu máquina local.

```powershell
docker exec climaxtreme-processor climaxtreme preprocess `
  --input-path "hdfs://climaxtreme-namenode:9000/data/climaxtreme/GlobalLandTemperaturesByCity_sample.csv" `
  --output-path "hdfs://climaxtreme-namenode:9000/data/climaxtreme/processed" `
  --format city-csv
```

Este comando:
1. Ejecuta el CLI `climaxtreme` dentro del contenedor `processor`
2. Lee datos desde HDFS (usando el hostname interno `climaxtreme-namenode`)
3. Procesa los datos con PySpark
4. Detecta anomalías de temperatura
5. Guarda resultados en HDFS en formato Parquet

**Salidas generadas en HDFS**:
- `/data/climaxtreme/processed/monthly.parquet` - Agregaciones mensuales
- `/data/climaxtreme/processed/yearly.parquet` - Agregaciones anuales
- `/data/climaxtreme/processed/anomalies.parquet` - Anomalías detectadas

**Tiempo esperado**: 1-3 minutos dependiendo del tamaño de los datos

### Verificar archivos procesados:

```powershell
docker exec climaxtreme-namenode hdfs dfs -ls /data/climaxtreme/processed
docker exec climaxtreme-namenode hdfs dfs -ls /data/climaxtreme/processed/monthly.parquet
```

## Paso 6: Lanzar Dashboard de Streamlit (Opcional)

**NOTA**: Para usar el dashboard necesitas instalar el paquete climaxtreme en tu máquina local.

### Instalación local (solo para el dashboard):

```powershell
# Desde el directorio Tools
cd Tools
pip install -e .
```

### Ejecutar dashboard:

```powershell
# El dashboard ahora soporta HDFS y Local Files
python -m climaxtreme.cli dashboard
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

**Opción 2 - Local Files (Para desarrollo/demos):**
1. Seleccionar: **Local Files**
2. Primero descarga los archivos desde HDFS (ver sección de comandos útiles)
3. El dashboard leerá desde `DATA/processed/`

**Ventajas del modo HDFS:**
- ✅ Sin descargas innecesarias
- ✅ HDFS como única fuente de verdad (principio Big Data)
- ✅ Siempre datos actualizados
- ✅ Ahorro de espacio en disco local

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
3. **Análisis exploratorio**: Usa PySpark dentro del processor para analizar los datos procesados
4. **Procesamiento en batch**: Crea scripts para procesar múltiples archivos automáticamente
5. **Queries SQL con Spark**: Lee los Parquet desde HDFS y ejecuta queries SQL
6. **Machine Learning**: Entrena modelos usando los datos procesados en formato Parquet

## Flujo de Trabajo Completo

### Setup Inicial (solo una vez):

```powershell
# 1. Levantar contenedores
cd infra
docker-compose up -d

# 2. Esperar a que estén healthy (30 segundos aprox)
docker-compose ps

# 3. Cargar datos a HDFS
cd ..
.\scripts\hdfs_setup_and_load.ps1 -CsvPath "DATA\GlobalLandTemperaturesByCity.csv" -Head 100000
```

### Procesamiento (cada vez que necesites procesar datos):

```powershell
# 1. Procesar datos
docker exec climaxtreme-processor climaxtreme preprocess `
  --input-path "hdfs://climaxtreme-namenode:9000/data/climaxtreme/GlobalLandTemperaturesByCity_sample.csv" `
  --output-path "hdfs://climaxtreme-namenode:9000/data/climaxtreme/processed" `
  --format city-csv

# 2. Verificar resultados
docker exec climaxtreme-namenode hdfs dfs -ls /data/climaxtreme/processed
```

### Visualización con Dashboard:

```powershell
# 1. Instalar paquete (solo primera vez)
cd Tools
pip install -e .

# 2. Lanzar dashboard
python -m climaxtreme.cli dashboard

# 3. En el navegador (http://localhost:8501):
#    - Seleccionar "HDFS (Recommended)" en sidebar
#    - Configurar: Host=namenode, Port=9000, Path=/data/climaxtreme/processed
#    - Seleccionar archivo para visualizar
```

### Desarrollo (cuando modificas código):

```powershell
# 1. Hacer cambios en Tools/src/climaxtreme/

# 2. Reconstruir processor
cd infra
docker-compose build processor
docker-compose restart processor

# 3. Probar cambios
docker exec climaxtreme-processor climaxtreme preprocess ...
```

## Recursos Adicionales

- **HDFS Web UI**: http://localhost:9870
- **Documentación PySpark**: https://spark.apache.org/docs/latest/api/python/
- **Docker Compose Docs**: https://docs.docker.com/compose/
