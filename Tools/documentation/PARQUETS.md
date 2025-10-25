# 📊 Parquet Files Documentation

Este documento describe la estructura y el proceso de generación de todos los archivos Parquet producidos por el sistema climaXtreme.

## 📁 Resumen General

El sistema genera **11 archivos Parquet** que contienen diferentes niveles de agregación y análisis de datos climáticos históricos. Todos estos archivos se generan mediante **Apache Spark** en el contenedor `processor` y se almacenan en **HDFS** bajo el directorio `/data/processed/`.

### Lista de Archivos Generados

| Archivo | Descripción | Registros Aprox. | Función Generadora |
|---------|-------------|------------------|-------------------|
| `monthly.parquet` | Agregaciones mensuales por ciudad | ~8.6M | `aggregate_monthly_data()` |
| `yearly.parquet` | Agregaciones anuales por ciudad | ~350K | `aggregate_yearly_data()` |
| `anomalies.parquet` | Desviaciones de temperatura respecto a la media climatológica | ~350K | `detect_anomalies()` |
| `climatology.parquet` | Valores climatológicos promedio por ciudad y mes | ~170K | `compute_climatology_stats()` |
| `seasonal.parquet` | Agregaciones por estación del año | ~1M | `compute_seasonal_stats()` |
| `extreme_thresholds.parquet` | Umbrales de temperaturas extremas (P10, P90) | ~170K | `compute_extreme_thresholds()` |
| `regional.parquet` | Agregaciones por región geográfica | ~2K | `compute_regional_aggregations()` |
| `continental.parquet` | Agregaciones por continente | ~300 | `compute_continental_aggregations()` |
| `correlation_matrix.parquet` | Matriz de correlación de Pearson | ~25 | `compute_correlation_matrix()` |
| `descriptive_stats.parquet` | Estadísticas descriptivas completas | ~4 | `compute_descriptive_statistics()` |
| `chi_square_tests.parquet` | Pruebas de independencia Chi-cuadrado | ~3 | `compute_chi_square_tests()` |

---

## 📋 Descripción Detallada de Cada Parquet

### 1️⃣ `monthly.parquet`

**Propósito:** Proporciona agregaciones mensuales de temperatura para cada ciudad, incluyendo estadísticas descriptivas completas.

**Esquema:**

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `City` | string | Nombre de la ciudad |
| `Country` | string | Nombre del país |
| `Latitude` | string | Latitud en formato original (ej: "57.05N") |
| `Longitude` | string | Longitud en formato original (ej: "10.33E") |
| `year` | integer | Año |
| `month` | integer | Mes (1-12) |
| `avg_temperature` | double | Temperatura promedio mensual (°C) |
| `min_temperature` | double | Temperatura mínima mensual (°C) |
| `max_temperature` | double | Temperatura máxima mensual (°C) |
| `std_temperature` | double | Desviación estándar de la temperatura |
| `record_count` | long | Número de registros diarios utilizados |

**Generación:**
```python
# Función: compute_monthly_aggregations()
# Archivo: Tools/src/climaxtreme/preprocessing/spark_processor.py (líneas 309-353)

df_monthly = df.groupBy("City", "Country", "Latitude", "Longitude", "year", "month").agg(
    F.mean("AverageTemperature").alias("avg_temperature"),
    F.min("AverageTemperature").alias("min_temperature"),
    F.max("AverageTemperature").alias("max_temperature"),
    F.stddev("AverageTemperature").alias("std_temperature"),
    F.count("*").alias("record_count")
)
```

**Caso de Uso:** Análisis detallado de tendencias mensuales, series temporales, visualización de ciclos estacionales por ciudad.

---

### 2️⃣ `yearly.parquet`

**Propósito:** Agrega datos a nivel anual para facilitar análisis de tendencias a largo plazo.

**Esquema:**

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `City` | string | Nombre de la ciudad |
| `Country` | string | Nombre del país |
| `Latitude` | string | Latitud en formato original |
| `Longitude` | string | Longitud en formato original |
| `year` | integer | Año |
| `avg_temperature` | double | Temperatura promedio anual (°C) |
| `min_temperature` | double | Temperatura mínima anual (°C) |
| `max_temperature` | double | Temperatura máxima anual (°C) |
| `temperature_range` | double | Rango de temperatura (max - min) |
| `record_count` | long | Número de registros mensuales utilizados |

**Generación:**
```python
# Función: compute_yearly_aggregations()
# Archivo: Tools/src/climaxtreme/preprocessing/spark_processor.py (líneas 355-400)

df_yearly = df_monthly.groupBy("City", "Country", "Latitude", "Longitude", "year").agg(
    F.mean("avg_temperature").alias("avg_temperature"),
    F.min("min_temperature").alias("min_temperature"),
    F.max("max_temperature").alias("max_temperature"),
    (F.max("max_temperature") - F.min("min_temperature")).alias("temperature_range"),
    F.sum("record_count").alias("record_count")
)
```

**Caso de Uso:** Visualización de tendencias anuales, detección de cambios climáticos a largo plazo, comparaciones interanuales.

---

### 3️⃣ `anomalies.parquet`

**Propósito:** Calcula las desviaciones (anomalías) de temperatura respecto a la media climatológica de referencia.

**Esquema:**

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `City` | string | Nombre de la ciudad |
| `Country` | string | Nombre del país |
| `Latitude` | string | Latitud en formato original |
| `Longitude` | string | Longitud en formato original |
| `year` | integer | Año |
| `avg_temperature` | double | Temperatura promedio anual observada (°C) |
| `climatology` | double | Temperatura climatológica de referencia (°C) |
| `anomaly` | double | Anomalía (observada - climatología) (°C) |
| `record_count` | long | Número de registros utilizados |

**Generación:**
```python
# Función: compute_anomalies()
# Archivo: Tools/src/climaxtreme/preprocessing/spark_processor.py (líneas 442-488)

# Se calcula la climatología (promedio de todos los años)
df_climatology = df_yearly.groupBy("City", "Country", "Latitude", "Longitude").agg(
    F.mean("avg_temperature").alias("climatology")
)

# Se hace JOIN y se calcula la anomalía
df_anomalies = df_yearly.join(df_climatology, on=["City", "Country", "Latitude", "Longitude"])
df_anomalies = df_anomalies.withColumn("anomaly", 
    F.col("avg_temperature") - F.col("climatology")
)
```

**Caso de Uso:** Identificación de años atípicamente cálidos o fríos, análisis de variabilidad climática, estudios de cambio climático.

---

### 4️⃣ `climatology.parquet`

**Propósito:** Proporciona valores climatológicos de referencia (promedios históricos) por ciudad y mes.

**Esquema:**

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `City` | string | Nombre de la ciudad |
| `Country` | string | Nombre del país |
| `Latitude` | string | Latitud en formato original |
| `Longitude` | string | Longitud en formato original |
| `month` | integer | Mes (1-12) |
| `climatology` | double | Temperatura climatológica mensual (°C) |
| `record_count` | long | Número de años utilizados en el cálculo |

**Generación:**
```python
# Función: compute_climatology()
# Archivo: Tools/src/climaxtreme/preprocessing/spark_processor.py (líneas 402-440)

df_climatology = df_monthly.groupBy("City", "Country", "Latitude", "Longitude", "month").agg(
    F.mean("avg_temperature").alias("climatology"),
    F.count("*").alias("record_count")
)
```

**Caso de Uso:** Línea base para cálculo de anomalías, comparación de condiciones actuales vs. históricas, normalización de datos.

---

### 5️⃣ `seasonal.parquet`

**Propósito:** Agrega datos por estaciones del año para analizar patrones estacionales.

**Esquema:**

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `City` | string | Nombre de la ciudad |
| `Country` | string | Nombre del país |
| `Latitude` | string | Latitud en formato original |
| `Longitude` | string | Longitud en formato original |
| `year` | integer | Año |
| `season` | string | Estación: "Winter", "Spring", "Summer", "Fall" |
| `avg_temperature` | double | Temperatura promedio estacional (°C) |
| `min_temperature` | double | Temperatura mínima estacional (°C) |
| `max_temperature` | double | Temperatura máxima estacional (°C) |
| `record_count` | long | Número de registros mensuales utilizados |

**Generación:**
```python
# Función: compute_seasonal_aggregations()
# Archivo: Tools/src/climaxtreme/preprocessing/spark_processor.py (líneas 490-526)

# Se asigna la estación según el mes (hemisferio norte)
df_seasonal = df_monthly.withColumn("season",
    F.when((F.col("month") >= 12) | (F.col("month") <= 2), "Winter")
     .when((F.col("month") >= 3) & (F.col("month") <= 5), "Spring")
     .when((F.col("month") >= 6) & (F.col("month") <= 8), "Summer")
     .otherwise("Fall")
)

# Se agregan por estación
df_seasonal = df_seasonal.groupBy("City", "Country", "Latitude", "Longitude", 
                                   "year", "season").agg(...)
```

**Caso de Uso:** Análisis de variaciones estacionales, identificación de cambios en patrones estacionales, estudios de fenología.

---

### 6️⃣ `extreme_thresholds.parquet`

**Propósito:** Calcula umbrales estadísticos (percentiles 10 y 90) para identificar eventos extremos de temperatura.

**Esquema:**

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `City` | string | Nombre de la ciudad |
| `Country` | string | Nombre del país |
| `Latitude` | string | Latitud en formato original |
| `Longitude` | string | Longitud en formato original |
| `month` | integer | Mes (1-12) |
| `p10_temperature` | double | Percentil 10 de temperatura (frío extremo) (°C) |
| `p90_temperature` | double | Percentil 90 de temperatura (calor extremo) (°C) |

**Generación:**
```python
# Función: compute_extreme_thresholds()
# Archivo: Tools/src/climaxtreme/preprocessing/spark_processor.py (líneas 672-733)

df_extreme = df_monthly.groupBy("City", "Country", "Latitude", "Longitude", "month").agg(
    F.expr("percentile(avg_temperature, 0.1)").alias("p10_temperature"),
    F.expr("percentile(avg_temperature, 0.9)").alias("p90_temperature")
)
```

**Caso de Uso:** Detección de olas de calor o frío extremo, alertas tempranas, análisis de eventos climáticos extremos.

---

### 7️⃣ `regional.parquet`

**Propósito:** Agrega datos por **16 regiones geográficas** del mundo, clasificadas automáticamente mediante latitud y longitud.

**Esquema:**

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `region` | string | Nombre de la región (ej: "Central Europe", "East Asia") |
| `continent` | string | Continente al que pertenece la región |
| `year` | integer | Año |
| `avg_temperature` | double | Temperatura promedio regional (°C) |
| `min_temperature` | double | Temperatura mínima regional (°C) |
| `max_temperature` | double | Temperatura máxima regional (°C) |
| `record_count` | long | Número de ciudades utilizadas |

**Regiones Geográficas (16):**
1. **Europe:** Northern Europe, Central Europe, Southern Europe
2. **Asia:** Northern Asia, Central Asia, South Asia, East Asia
3. **Africa:** Northern Africa, Central Africa, Southern Africa
4. **North America:** Northern North America, Central North America, Caribbean & Central America
5. **South America:** Northern South America, Central South America, Southern South America
6. **Oceania:** Northern Oceania, Southern Oceania
7. **Antarctica:** Antarctica

**Generación:**
```python
# Función: compute_regional_aggregations()
# Archivo: Tools/src/climaxtreme/preprocessing/spark_processor.py (líneas 621-670)

# Paso 1: Parsear coordenadas (N/S/E/W → numérico)
df_with_coords = df_yearly.withColumn("lat_numeric", parse_coordinates_udf(F.col("Latitude"), F.lit("lat")))
df_with_coords = df_with_coords.withColumn("lon_numeric", parse_coordinates_udf(F.col("Longitude"), F.lit("lon")))

# Paso 2: Asignar continente y región
df_with_coords = df_with_coords.withColumn("continent", assign_continent_udf(F.col("lat_numeric"), F.col("lon_numeric")))
df_with_coords = df_with_coords.withColumn("region", assign_region_udf(F.col("lat_numeric"), F.col("lon_numeric")))

# Paso 3: Agregar por región y año
df_regional = df_with_coords.groupBy("region", "continent", "year").agg(
    F.mean("avg_temperature").alias("avg_temperature"),
    F.min("min_temperature").alias("min_temperature"),
    F.max("max_temperature").alias("max_temperature"),
    F.sum("record_count").alias("record_count")
)
```

**Lógica de Clasificación Regional:**
- Se parsean las coordenadas de formato "57.05N, 10.33E" a valores numéricos con signo
- Se aplica una función que clasifica según rangos de latitud/longitud
- Cada región tiene límites geográficos específicos definidos en `assign_region()` (líneas 583-619)

**Caso de Uso:** Comparación de tendencias climáticas entre regiones geográficas, mapas de calor regionales, dashboard interactivo con selección de regiones.

---

### 8️⃣ `continental.parquet`

**Propósito:** Agrega datos al nivel más alto: **7 continentes**, facilitando comparaciones globales.

**Esquema:**

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `continent` | string | Nombre del continente: "Europe", "Asia", "Africa", "North America", "South America", "Oceania", "Antarctica" |
| `year` | integer | Año |
| `avg_temperature` | double | Temperatura promedio continental (°C) |
| `min_temperature` | double | Temperatura mínima continental (°C) |
| `max_temperature` | double | Temperatura máxima continental (°C) |
| `record_count` | long | Número de ciudades utilizadas |

**Generación:**
```python
# Función: compute_continental_aggregations()
# Archivo: Tools/src/climaxtreme/preprocessing/spark_processor.py (líneas 672-733)

# Similar a regional, pero solo agrupa por continente (sin región)
df_continental = df_with_coords.groupBy("continent", "year").agg(
    F.mean("avg_temperature").alias("avg_temperature"),
    F.min("min_temperature").alias("min_temperature"),
    F.max("max_temperature").alias("max_temperature"),
    F.sum("record_count").alias("record_count")
)
```

**Lógica de Clasificación Continental:**
```python
# Función: assign_continent() (líneas 547-581)
# Clasifica según rangos de latitud y longitud:
- Europe: lat 36-72, lon -10-40
- Asia: lat -10-81, lon 26-180
- Africa: lat -35-37, lon -18-52
- North America: lat 15-72, lon -169-(-52)
- South America: lat -56-13, lon -82-(-34)
- Oceania: lat -47-(-10), lon 110-180
- Antarctica: lat < -60
```

**Caso de Uso:** Comparaciones globales, visualización de tendencias continentales, mapas interactivos con burbujas por continente.

---

## 🔄 Proceso de Generación Completo

### Flujo de Procesamiento

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. INGESTA (HDFS)                                               │
│    - Archivo: GlobalLandTemperaturesByCity.csv (~500 MB)       │
│    - Ubicación: hdfs://namenode:9000/data/raw/                 │
│    - Registros: ~8.6 millones                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. PROCESAMIENTO SPARK (Container: processor)                  │
│    - Script: spark_processor.py                                │
│    - Función principal: process_path()                          │
│    - Orden de ejecución:                                        │
│      a) aggregate_monthly_data()            → monthly.parquet  │
│      b) aggregate_yearly_data()             → yearly.parquet   │
│      c) compute_climatology_stats()         → climatology.parq │
│      d) detect_anomalies()                  → anomalies.parq   │
│      e) compute_seasonal_stats()            → seasonal.parquet │
│      f) compute_extreme_thresholds()        → extreme_thresh.p │
│      g) compute_regional_aggregations()     → regional.parquet │
│      h) compute_continental_aggregations()  → continental.parq │
│      i) compute_correlation_matrix()        → correlation_mat. │
│      j) compute_descriptive_statistics()    → descriptive_st.  │
│      k) compute_chi_square_tests()          → chi_square_tests │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. ALMACENAMIENTO (HDFS)                                        │
│    - Ubicación: hdfs://namenode:9000/data/processed/           │
│    - Formato: Parquet (columnar, comprimido)                   │
│    - Tamaño total estimado: ~100-150 MB (11 archivos)          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. VISUALIZACIÓN (Dashboard Streamlit)                         │
│    - Acceso directo desde HDFS vía PyArrow                     │
│    - 7 pestañas de análisis:                                   │
│      • Temperature Trends                                       │
│      • Heatmaps                                                 │
│      • Seasonal Analysis                                        │
│      • Extreme Events                                           │
│      • Regional Analysis (con mapa interactivo)                │
│      • Continental Analysis (con mapa global)                  │
│      • Exploratory Analysis (EDA) ← NUEVO                      │
│      • Continental Analysis (con mapa global)                  │
└─────────────────────────────────────────────────────────────────┘
```

### Script de Ejecución

**Archivo:** `scripts/process_full_dataset.ps1`

**Comando:**
```powershell
.\scripts\process_full_dataset.ps1
```

**Pasos internos del script:**

1. **Upload a HDFS:**
   ```bash
   docker exec -it namenode hdfs dfs -mkdir -p /data/raw
   docker exec -it namenode hdfs dfs -put /data/GlobalLandTemperaturesByCity.csv /data/raw/
   ```

2. **Ejecución Spark:**
   ```bash
   docker exec processor spark-submit \
     --master local[*] \
     /opt/climaxtreme/src/climaxtreme/preprocessing/spark_processor.py \
     --input-path hdfs://namenode:9000/data/raw/GlobalLandTemperaturesByCity.csv \
     --output-path hdfs://namenode:9000/data/processed
   ```

3. **Verificación:**
   ```bash
   docker exec namenode hdfs dfs -ls /data/processed/
   ```

4. **Descarga (Opcional):**
   ```bash
   docker exec namenode hdfs dfs -get /data/processed/*.parquet /data/processed/
   ```

---

## 🛠️ Funciones de Procesamiento

### Referencia Rápida

| Función | Líneas | Dependencias | Output |
|---------|--------|--------------|--------|
| `parse_coordinates()` | 528-545 | - | UDF para parsear "57.05N" → 57.05 |
| `assign_continent()` | 547-581 | parse_coordinates | UDF para asignar continente |
| `assign_region()` | 583-619 | parse_coordinates | UDF para asignar región |
| `compute_monthly_aggregations()` | 309-353 | df (raw) | monthly.parquet |
| `compute_yearly_aggregations()` | 355-400 | monthly.parquet | yearly.parquet |
| `compute_climatology()` | 402-440 | monthly.parquet | climatology.parquet |
| `compute_anomalies()` | 442-488 | yearly + climatology | anomalies.parquet |
| `compute_seasonal_aggregations()` | 490-526 | monthly.parquet | seasonal.parquet |
| `compute_extreme_thresholds()` | 672-733 | monthly.parquet | extreme_thresholds.parquet |
| `compute_regional_aggregations()` | 621-670 | yearly.parquet | regional.parquet |
| `compute_continental_aggregations()` | 672-733 | yearly.parquet (con coords) | continental.parquet |
| `process_path()` | 735-903 | Todas las anteriores | Orquesta todo el proceso |

### Dependencias entre Archivos

```
raw CSV
   ├─→ monthly.parquet (base)
   │      ├─→ yearly.parquet
   │      │      ├─→ anomalies.parquet (yearly + climatology)
   │      │      ├─→ regional.parquet (yearly + coords)
   │      │      └─→ continental.parquet (yearly + coords)
   │      ├─→ climatology.parquet
   │      ├─→ seasonal.parquet
   │      └─→ extreme_thresholds.parquet
```

---

## 📈 Dashboard Integration

Los archivos Parquet se visualizan en el dashboard Streamlit mediante acceso directo a HDFS:

**Clase:** `HDFSReader` (app.py, líneas 50-135)

**Ejemplo de lectura:**
```python
# Modo HDFS
reader = HDFSReader(namenode_host="namenode", namenode_port=9000)
df = reader.read_parquet("hdfs://namenode:9000/data/processed/regional.parquet")

# Modo Local
df = pd.read_parquet("./DATA/processed/regional.parquet")
```

**Pestañas del Dashboard:**

1. **Monthly Analysis** → `monthly.parquet`
2. **Yearly Analysis** → `yearly.parquet`
3. **Anomalies Analysis** → `anomalies.parquet`
4. **Seasonal Analysis** → `seasonal.parquet`
5. **Regional Analysis** → `regional.parquet` + **Mapa interactivo del mundo**
6. **Continental Analysis** → `continental.parquet` + **Mapa global con burbujas**

### Visualizaciones de Mapas (Nuevas)

**Regional Analysis Tab:**
- **Mapa Scatter Geo** con 16 regiones
- Burbujas proporcionales a temperatura
- Escala de colores: RdYlBu_r (azul=frío, rojo=caliente)
- Proyección: Natural Earth
- Tooltip: Región, Temperatura, # Registros

**Continental Analysis Tab:**
- **Mapa Scatter Geo** con 7 continentes
- Burbujas más grandes con texto del continente
- Misma escala de colores
- Tooltip: Continente, Temperatura, # Registros

**Biblioteca Utilizada:** Plotly Express con soporte de `go.Scattergeo`

---

## ✅ Validación de Datos

### Checks de Calidad

Cada función de procesamiento incluye:
- **Filtrado de valores nulos:** `.filter(F.col("column").isNotNull())`
- **Validación de rango:** Temperaturas entre -90°C y 60°C
- **Conteo de registros:** Columna `record_count` para auditoría

### Logs de Procesamiento

```python
logger.info(f"Monthly aggregations: {df_monthly.count()} records")
logger.info(f"Regional aggregations: {df_regional.count()} records")
logger.info(f"Continental aggregations: {df_continental.count()} records")
```

---

## 🔧 Configuración

**Archivo:** `configs/default_config.yml`

```yaml
spark:
  app_name: "ClimaXtreme-Processor"
  master: "local[*]"
  input_path: "hdfs://namenode:9000/data/raw/GlobalLandTemperaturesByCity.csv"
  output_path: "hdfs://namenode:9000/data/processed"
```

---

### 9️⃣ `correlation_matrix.parquet`

**Propósito:** Matriz de correlación de Pearson entre variables numéricas climáticas para análisis exploratorio.

**Esquema:**

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `variable_1` | string | Nombre de la primera variable |
| `variable_2` | string | Nombre de la segunda variable |
| `correlation` | double | Coeficiente de correlación de Pearson (-1 a 1) |
| `abs_correlation` | double | Valor absoluto de la correlación |

**Generación:**
```python
# Función: compute_correlation_matrix()
# Archivo: Tools/src/climaxtreme/preprocessing/spark_processor.py (líneas 757-811)

# Variables analizadas:
# - year
# - avg_temperature
# - min_temperature
# - max_temperature
# - temperature_range (calculado como max - min)

# Cálculo de correlaciones pairwise
for var1 in numeric_vars:
    for var2 in numeric_vars:
        correlation = df.stat.corr(var1, var2)
        # Se almacenan tanto la correlación como su valor absoluto
```

**Interpretación:**
- **r = 1**: Correlación positiva perfecta
- **r = -1**: Correlación negativa perfecta
- **r = 0**: Sin correlación lineal
- **|r| > 0.7**: Correlación fuerte
- **|r| > 0.4**: Correlación moderada
- **|r| < 0.3**: Correlación débil

**Caso de Uso:** 
- Identificar relaciones lineales entre variables
- Detectar multicolinealidad antes de modelado
- Validar hipótesis sobre dependencias climáticas
- Visualización mediante heatmap de correlaciones

---

### 🔟 `descriptive_stats.parquet`

**Propósito:** Estadísticas descriptivas completas para todas las variables numéricas del dataset.

**Esquema (Formato Pivotado):**

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `variable` | string | Nombre de la variable analizada |
| `count` | double | Número de observaciones válidas |
| `mean` | double | Media aritmética |
| `std_dev` | double | Desviación estándar |
| `min` | double | Valor mínimo |
| `q1` | double | Primer cuartil (percentil 25) |
| `median` | double | Mediana (percentil 50) |
| `q3` | double | Tercer cuartil (percentil 75) |
| `max` | double | Valor máximo |
| `iqr` | double | Rango intercuartílico (Q3 - Q1) |
| `skewness` | double | Asimetría de la distribución |
| `kurtosis` | double | Curtosis (exceso de kurtosis) |

**Variables Analizadas:**
- `avg_temperature`
- `min_temperature`
- `max_temperature`
- `uncertainty` (incertidumbre de medición)

**Generación:**
```python
# Función: compute_descriptive_statistics()
# Archivo: Tools/src/climaxtreme/preprocessing/spark_processor.py (líneas 813-920)

# Para cada variable numérica:
stats = df.agg(
    count(var),
    mean(var),
    stddev(var),
    min(var),
    max(var),
    skewness(var),
    kurtosis(var),
    percentile_approx(var, 0.25),  # Q1
    percentile_approx(var, 0.50),  # Mediana
    percentile_approx(var, 0.75)   # Q3
)

iqr = Q3 - Q1  # Rango intercuartílico
```

**Interpretación:**

**Skewness (Asimetría):**
- **< -1**: Distribución muy sesgada a la izquierda (cola larga izquierda)
- **-1 a -0.5**: Moderadamente sesgada izquierda
- **-0.5 a 0.5**: Aproximadamente simétrica
- **0.5 a 1**: Moderadamente sesgada derecha
- **> 1**: Muy sesgada a la derecha (cola larga derecha)

**Kurtosis (Curtosis):**
- **< 0**: Platicúrtica (colas más ligeras que normal)
- **≈ 0**: Mesocúrtica (similar a distribución normal)
- **> 0**: Leptocúrtica (colas más pesadas, picos más pronunciados)

**Caso de Uso:**
- Comprender la distribución de las variables
- Detectar outliers usando IQR
- Validar supuestos de normalidad
- Comparar dispersión entre variables
- Base para transformaciones de datos

---

### 1️⃣1️⃣ `chi_square_tests.parquet`

**Propósito:** Resultados de pruebas de independencia Chi-cuadrado para variables categóricas.

**Esquema:**

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `test` | string | Nombre descriptivo del test |
| `variable_1` | string | Primera variable categórica |
| `variable_2` | string | Segunda variable categórica |
| `chi_square_statistic` | double | Estadístico χ² calculado |
| `p_value` | double | Valor p del test |
| `degrees_of_freedom` | integer | Grados de libertad |
| `is_significant` | boolean | True si p < 0.05 (rechaza H₀) |

**Tests Realizados:**

1. **Continent vs Temperature Category**
   - Variables: `continent` (7 categorías) × `temp_category` (Cold/Moderate/Hot)
   - Hipótesis: ¿La distribución de temperaturas es independiente del continente?

2. **Season vs Temperature Category**
   - Variables: `season` (4 estaciones) × `temp_category` (Cold/Moderate/Hot)
   - Hipótesis: ¿Las temperaturas varían significativamente por estación?

3. **Time Period vs Temperature Category**
   - Variables: `time_period` (Early/Late) × `temp_category` (Cold/Moderate/Hot)
   - Hipótesis: ¿Ha cambiado la distribución de temperaturas a lo largo del tiempo?

**Generación:**
```python
# Función: compute_chi_square_tests()
# Archivo: Tools/src/climaxtreme/preprocessing/spark_processor.py (líneas 922-1044)

# 1. Crear categorías de temperatura
df_categorized = df.withColumn('temp_category',
    when(col('avg_temperature') < 10, 'Cold')
    .when(col('avg_temperature') < 20, 'Moderate')
    .otherwise('Hot')
)

# 2. Crear tabla de contingencia
contingency = df.groupBy('var1', 'var2').count()

# 3. Calcular estadístico Chi-cuadrado
chi_square = Σ ((Observed - Expected)² / Expected)

# 4. Calcular p-value usando distribución χ²
from scipy.stats import chi2
p_value = 1 - chi2.cdf(chi_square_stat, df)
```

**Interpretación:**

**Hipótesis:**
- **H₀ (Nula)**: Las variables son independientes (no hay relación)
- **H₁ (Alternativa)**: Las variables son dependientes (existe relación)

**Criterio de Decisión:**
- **p-value < 0.05**: Rechazar H₀ → Variables dependientes (**Significativo**)
- **p-value ≥ 0.05**: No rechazar H₀ → Variables independientes (No significativo)

**Estadístico χ²:**
- Valores más altos indican mayor discrepancia entre frecuencias observadas y esperadas
- Depende de los grados de libertad: `df = (filas - 1) × (columnas - 1)`

**Caso de Uso:**
- Validar hipótesis sobre relaciones categóricas
- Detectar dependencias entre variables climáticas y geográficas
- Análisis de varianza categórica
- Preparación para modelos de clasificación

---

## 📚 Referencias

- **Código fuente:** `Tools/src/climaxtreme/preprocessing/spark_processor.py`
- **Dashboard:** `Tools/src/climaxtreme/dashboard/app.py`
- **Script de procesamiento:** `scripts/process_full_dataset.ps1`
- **Configuración:** `Tools/configs/default_config.yml`

---

## 🚀 Próximos Pasos

- [ ] Agregar tests unitarios para funciones de clasificación geográfica
- [ ] Implementar particionamiento de Parquet por año para optimizar lecturas
- [ ] Crear índices espaciales para queries geográficas más rápidas
- [ ] Añadir soporte para datos en tiempo real (streaming)

---

**Última actualización:** 2024
**Autor:** Equipo climaXtreme
