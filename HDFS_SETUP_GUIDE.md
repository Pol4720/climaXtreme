# Guía de Configuración HDFS para climaXtreme

## Requisitos Previos

1. **Docker Desktop** instalado y en ejecución (WSL2 recomendado)
2. **Python 3.9+** con el paquete climaxtreme instalado
3. **Dataset** en `DATA/GlobalLandTemperaturesByCity.csv`

## Paso 1: Verificar Docker Desktop

Antes de ejecutar cualquier comando, asegúrate que Docker Desktop está corriendo:

```powershell
docker info
```

Si ves un error, abre Docker Desktop y espera a que muestre "Running" en verde.

## Paso 2: Iniciar HDFS y Cargar Datos

Desde la raíz del repositorio en PowerShell:

```powershell
.\scripts\hdfs_setup_and_load.ps1 -CsvPath "DATA\GlobalLandTemperaturesByCity.csv" -Head 100000
```

Esto hará:
- Descargar imágenes Docker de Apache Hadoop (primera vez ~500MB)
- Iniciar NameNode (puerto 9870) y DataNode
- Crear directorio `/data/climaxtreme` en HDFS
- Subir un sample de 100k filas del CSV

**Tiempo esperado**: 1-3 minutos la primera vez (descarga de imágenes)

## Paso 3: Verificar HDFS

### Opción A: UI Web
Abre en tu navegador: http://localhost:9870

Ve a "Utilities" → "Browse the file system" → Navega a `/data/climaxtreme`

### Opción B: Línea de comandos
```powershell
docker exec climaxtreme-namenode hdfs dfs -ls /data/climaxtreme
docker exec climaxtreme-namenode hdfs dfs -tail /data/climaxtreme/GlobalLandTemperaturesByCity_sample.csv
```

## Paso 4: Procesar Datos desde HDFS

```powershell
climaxtreme preprocess `
  --input-path "hdfs://climaxtreme-namenode:9000/data/climaxtreme/GlobalLandTemperaturesByCity_sample.csv" `
  --output-path "hdfs://climaxtreme-namenode:9000/data/climaxtreme/processed" `
  --format city-csv
```

**Salidas Parquet en HDFS**:
- `/data/climaxtreme/processed/monthly.parquet`
- `/data/climaxtreme/processed/yearly.parquet`
- `/data/climaxtreme/processed/anomalies.parquet`

## Paso 5: Lanzar Dashboard de Streamlit

```powershell
climaxtreme dashboard --data-dir "DATA"
```

Abre: http://localhost:8501

## Solución de Problemas

### Error: "image not found"
**Problema**: Docker no puede descargar las imágenes.

**Solución**:
1. Verifica tu conexión a internet
2. Intenta descargar manualmente:
   ```powershell
   docker pull apache/hadoop:3
   ```
3. Si estás detrás de un proxy corporativo, configúralo en Docker Desktop → Settings → Resources → Proxies

### Error: "The system cannot find the file specified"
**Problema**: Docker Desktop no está corriendo.

**Solución**: Abre Docker Desktop y espera a que esté en estado "Running"

### Error: "No se pudo iniciar el contenedor"
**Problema**: El contenedor no arrancó correctamente.

**Solución**:
1. Ver logs del contenedor:
   ```powershell
   docker logs climaxtreme-namenode
   docker logs climaxtreme-datanode
   ```
2. Reiniciar desde cero:
   ```powershell
   docker compose -f infra\docker-compose.yml down -v
   .\scripts\hdfs_setup_and_load.ps1
   ```

### El dashboard no carga los datos
**Problema**: La ruta de datos no es correcta.

**Solución**: 
- Verifica que `DATA/GlobalLandTemperaturesByCity.csv` existe
- En el dashboard, ajusta la ruta en el sidebar a la ubicación correcta

## Detener HDFS

```powershell
docker compose -f infra\docker-compose.yml down
```

Para eliminar también los volúmenes (datos persistentes):
```powershell
docker compose -f infra\docker-compose.yml down -v
```

## Comandos Útiles

```powershell
# Ver contenedores corriendo
docker ps

# Ver logs en tiempo real
docker logs -f climaxtreme-namenode

# Entrar al contenedor
docker exec -it climaxtreme-namenode bash

# Listar archivos en HDFS
docker exec climaxtreme-namenode hdfs dfs -ls /

# Ver espacio usado
docker exec climaxtreme-namenode hdfs dfs -df -h

# Descargar archivo desde HDFS
docker exec climaxtreme-namenode hdfs dfs -get /data/climaxtreme/processed/monthly.parquet /tmp/
docker cp climaxtreme-namenode:/tmp/monthly.parquet ./
```

## Próximos Pasos

1. **Añadir más datos**: Modifica el parámetro `-Head` para cargar más filas
2. **Procesamiento en batch**: Crea scripts para procesar múltiples archivos
3. **Integración con YARN**: Para clusters más grandes (documentación futura)
4. **Queries SQL en Spark**: Usa Spark SQL sobre los Parquet en HDFS
