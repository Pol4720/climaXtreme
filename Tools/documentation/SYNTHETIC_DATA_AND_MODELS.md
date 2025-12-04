# 📊 Datos Sintéticos y Modelos - climaXtreme

## Resumen Ejecutivo

Este documento describe el sistema de generación de datos sintéticos y los modelos utilizados para soportar las visualizaciones en tiempo real del dashboard de climaXtreme.

---

## 1. 🎯 Objetivos

El dataset original (`GlobalLandTemperaturesByCity.csv`) contiene:
- Temperatura promedio mensual
- Incertidumbre de medición
- Ciudad, País, Latitud, Longitud
- Fechas desde 1743

**Necesidades adicionales para visualizaciones avanzadas:**
| Visualización | Datos Requeridos | Estado Original |
|--------------|------------------|-----------------|
| Mapas de calor climáticos | Temp, Lat, Lon, Tiempo | ✅ Parcial |
| Evolución de tormentas | Storm ID, Trayectoria, Intensidad | ❌ No existe |
| Predicción de intensidad | Métricas de intensidad, Features | ❌ No existe |
| Alertas activas | Nivel alerta, Tipo evento, Timestamp | ❌ No existe |
| Comparación histórica | Series completas, Anomalías | ✅ Parcial |
| Series temperatura/lluvia | Precipitación, Temp horaria | ❌ No existe |

---

## 2. 📋 Esquema de Datos Sintéticos

### 2.1 Esquema Principal Extendido

```
SyntheticClimateData
├── Campos Originales (del dataset)
│   ├── dt: date                      # Fecha original
│   ├── AverageTemperature: float     # Temperatura promedio
│   ├── AverageTemperatureUncertainty: float
│   ├── City: string
│   ├── Country: string
│   ├── Latitude: float               # Convertido a decimal
│   └── Longitude: float              # Convertido a decimal
│
├── Campos Sintéticos Temporales
│   ├── timestamp: timestamp          # Timestamp con resolución horaria
│   ├── hour: int                     # Hora del día (0-23)
│   ├── year: int
│   ├── month: int
│   └── day_of_week: int
│
├── Campos Meteorológicos Sintéticos
│   ├── temperature_hourly: float     # Temp horaria interpolada
│   ├── rain_mm: float                # Precipitación (mm)
│   ├── humidity_pct: float           # Humedad relativa (%)
│   ├── wind_speed_kmh: float         # Velocidad viento (km/h)
│   ├── wind_direction_deg: float     # Dirección viento (grados)
│   ├── pressure_hpa: float           # Presión atmosférica (hPa)
│   └── cloud_cover_pct: float        # Cobertura nubosa (%)
│
├── Campos de Eventos Extremos
│   ├── storm_id: string (nullable)   # ID único de tormenta
│   ├── storm_category: int (0-5)     # Categoría Saffir-Simpson
│   ├── storm_name: string (nullable)
│   ├── event_type: string            # NORMAL, STORM, HEATWAVE, COLDSNAP, FLOOD
│   ├── event_intensity: float (0-1)  # Intensidad normalizada
│   └── event_duration_hours: int
│
├── Campos de Alertas
│   ├── alert_active: boolean
│   ├── alert_level: string           # NONE, WATCH, WARNING, EMERGENCY
│   ├── alert_type: string            # HEAT, COLD, STORM, FLOOD, WIND
│   └── alert_issued_at: timestamp
│
└── Campos de Análisis
    ├── anomaly_score: float          # Desviación vs climatología
    ├── trend_direction: string       # UP, DOWN, STABLE
    ├── climate_zone: string          # TROPICAL, TEMPERATE, POLAR, etc.
    └── season: string                # Estación del año
```

### 2.2 Esquema de Tormentas (Tracking)

```
StormTrackData
├── storm_id: string              # UUID único
├── storm_name: string            # Nombre asignado
├── timestamp: timestamp          # Punto temporal
├── latitude: float               # Posición actual
├── longitude: float
├── category: int (0-5)           # Categoría actual
├── max_wind_kmh: float           # Viento máximo sostenido
├── central_pressure_hpa: float   # Presión central
├── movement_speed_kmh: float     # Velocidad de desplazamiento
├── movement_direction_deg: float # Dirección de movimiento
├── radius_km: float              # Radio de afectación
├── affected_countries: array<string>
├── affected_cities: array<string>
└── lifecycle_stage: string       # FORMING, INTENSIFYING, MATURE, WEAKENING, DISSIPATING
```

---

## 3. 🔬 Modelos y Técnicas de Generación

### 3.1 Generación de Series Temporales Horarias

**Técnica: Interpolación + Ruido Estocástico**

```python
# Modelo de temperatura horaria
T_hourly(h) = T_daily_mean + A_diurnal * sin(2π(h - h_max)/24) + ε

Donde:
- T_daily_mean: Temperatura media diaria (del dataset original)
- A_diurnal: Amplitud diurna (función de latitud y estación)
- h_max: Hora de temperatura máxima (~14:00 local)
- ε ~ N(0, σ²): Ruido gaussiano con σ proporcional a uncertainty
```

**Parámetros por zona climática:**
| Zona | A_diurnal (°C) | σ (°C) | h_max |
|------|---------------|--------|-------|
| Tropical | 8-12 | 0.5 | 14 |
| Templada | 10-18 | 1.0 | 15 |
| Árida | 15-25 | 0.8 | 14 |
| Polar | 5-10 | 1.5 | 13 |

### 3.2 Generación de Precipitación

**Técnica: Cadena de Markov + Distribución Gamma**

```python
# Modelo de precipitación diaria
1. Estado wet/dry: Cadena de Markov orden 1
   P(wet|dry) = p_01(month, latitude)  # Probabilidad transición a lluvia
   P(dry|wet) = p_10(month, latitude)  # Probabilidad fin de lluvia

2. Cantidad si wet: Distribución Gamma
   rain_mm ~ Gamma(α, β)
   Donde α, β varían según clima y estación

3. Desagregación horaria: Fragmentación estocástica
   - Distribución temporal basada en patrones de tormenta
```

**Matriz de transición típica (clima templado, verano):**
```
         Dry    Wet
Dry    [ 0.85,  0.15 ]
Wet    [ 0.60,  0.40 ]
```

### 3.3 Generación de Tormentas

**Técnica: Proceso de Poisson + Simulación de Trayectorias**

```python
# Modelo de ocurrencia de tormentas
N_storms(region, year) ~ Poisson(λ_region)

# Trayectoria: Random Walk con drift geofísico
lat(t+1) = lat(t) + v_lat * Δt + σ_lat * W_lat
lon(t+1) = lon(t) + v_lon * Δt + σ_lon * W_lon

Donde:
- v_lat, v_lon: Velocidades medias (influenciadas por Coriolis, corrientes)
- W: Proceso de Wiener (movimiento browniano)
```

**Parámetros de intensidad:**
```python
# Evolución de intensidad (Holland 1980 modificado)
I(t) = I_max * f(SST, shear, moisture) * g(lifecycle_stage)

# Categoría Saffir-Simpson
category = floor(max_wind_kmh / 33)  # Simplificado
```

### 3.4 Generación de Alertas

**Técnica: Sistema de Reglas + Umbrales Adaptativos**

```python
# Umbrales de alerta
ALERT_THRESHOLDS = {
    'HEAT': {
        'WATCH': percentile_95 + 2°C,
        'WARNING': percentile_99,
        'EMERGENCY': percentile_99 + 3°C
    },
    'COLD': {
        'WATCH': percentile_5 - 2°C,
        'WARNING': percentile_1,
        'EMERGENCY': percentile_1 - 3°C
    },
    'STORM': {
        'WATCH': category >= 1,
        'WARNING': category >= 3,
        'EMERGENCY': category >= 4
    },
    'WIND': {
        'WATCH': wind_kmh >= 60,
        'WARNING': wind_kmh >= 90,
        'EMERGENCY': wind_kmh >= 120
    }
}
```

### 3.5 Detección y Predicción de Anomalías

**Modelo: Z-Score + Seasonal Decomposition**

```python
# Anomaly Score
anomaly_score = (T_observed - T_climatology) / σ_climatology

# Clasificación
if abs(anomaly_score) < 1.5: event_type = 'NORMAL'
elif anomaly_score >= 2.5: event_type = 'HEATWAVE'
elif anomaly_score <= -2.5: event_type = 'COLDSNAP'
```

---

## 4. 🚀 Modelos de Machine Learning

### 4.1 Predicción de Intensidad de Eventos

**Modelo Principal: Gradient Boosting (XGBoost/LightGBM)**

```yaml
Algoritmo: LightGBM Regressor
Target: event_intensity (0-1)
Features:
  - Temporales: hour, day_of_week, month, season
  - Geográficas: latitude, longitude, climate_zone
  - Meteorológicas: temperature, humidity, pressure, wind_speed
  - Históricas: anomaly_score_lag1, anomaly_score_lag7, trend_30d
  
Hiperparámetros:
  n_estimators: 500
  max_depth: 8
  learning_rate: 0.05
  num_leaves: 31
  min_child_samples: 20
  
Validación: TimeSeriesSplit (5 folds)
Métricas: RMSE, MAE, R²
```

### 4.2 Clasificación de Tipo de Evento

**Modelo: Random Forest Classifier**

```yaml
Algoritmo: RandomForestClassifier
Target: event_type (NORMAL, STORM, HEATWAVE, COLDSNAP, FLOOD)
Features: Similar a predicción de intensidad

Hiperparámetros:
  n_estimators: 200
  max_depth: 12
  min_samples_split: 10
  class_weight: 'balanced'  # Para desbalance de clases
  
Métricas: F1-macro, Precision, Recall por clase
```

### 4.3 Predicción de Trayectorias de Tormentas

**Modelo: LSTM Sequence-to-Sequence**

```yaml
Arquitectura:
  Encoder: LSTM(128) → LSTM(64)
  Decoder: LSTM(64) → Dense(2)  # [lat, lon]
  
Input: Secuencia de 24h de posiciones + features
Output: Predicción de próximas 12-48h
Seq_length: 24
Prediction_horizon: 12, 24, 48 horas

Entrenamiento:
  Optimizer: Adam (lr=0.001)
  Loss: MSE + Haversine distance penalty
  Epochs: 100
  Early_stopping: patience=10
```

### 4.4 Ensemble para Dashboard

**Modelo Productivo: VotingRegressor/Classifier**

```yaml
Ensemble:
  - LinearRegression (baseline)
  - Ridge (regularización)
  - RandomForest (no-linealidad)
  - LightGBM (boosting)

Pesos: Optimizados por validación cruzada
Incertidumbre: Desviación estándar entre predicciones
```

---

## 5. 📊 Implementación en Spark

### 5.1 Generador Batch (PySpark)

```python
# Pseudocódigo del pipeline
def generate_synthetic_data(spark, original_df, config):
    # 1. Expandir a resolución horaria
    hourly_df = expand_to_hourly(original_df)
    
    # 2. Generar variables meteorológicas
    weather_df = generate_weather_variables(hourly_df)
    
    # 3. Simular eventos extremos
    events_df = simulate_extreme_events(weather_df, config.event_rates)
    
    # 4. Generar tormentas
    storms_df = simulate_storms(events_df, config.storm_params)
    
    # 5. Calcular alertas
    alerts_df = compute_alerts(storms_df, config.thresholds)
    
    # 6. Escribir a HDFS/Parquet particionado
    write_partitioned(alerts_df, output_path, ['year', 'month', 'country'])
    
    return alerts_df
```

### 5.2 Streaming (Structured Streaming)

```python
# Para demo de tiempo real
def create_streaming_generator(spark, rate_per_second=100):
    return spark.readStream \
        .format("rate") \
        .option("rowsPerSecond", rate_per_second) \
        .load() \
        .withColumn("synthetic_data", generate_row_udf())
```

---

## 6. 🎨 Integración con Dashboard

### 6.1 Visualizaciones Soportadas

| Visualización | Datos Usados | Librerías |
|--------------|--------------|-----------|
| Mapa de calor global | hourly temps, lat/lon | Plotly/Folium |
| Tracking tormentas | StormTrackData | Plotly animations |
| Gauge intensidad | event_intensity | Streamlit metrics |
| Tabla alertas | alerts con filtros | Streamlit dataframe |
| Series temporales | temps, rain, wind | Plotly time series |
| Comparación histórica | anomaly_score | Plotly overlays |

### 6.2 Flujo de Datos

```
HDFS/Parquet
    │
    ▼
DataSource (utils.py)
    │
    ├──► load_parquet('synthetic_hourly.parquet')
    ├──► load_parquet('storm_tracks.parquet')
    └──► load_parquet('alerts.parquet')
           │
           ▼
    Dashboard Pages
           │
           ▼
    Visualizaciones Interactivas
```

---

## 7. 📁 Archivos de Salida (Parquet)

| Archivo | Descripción | Particionamiento |
|---------|-------------|------------------|
| `synthetic_hourly.parquet` | Datos horarios completos | year/month/country |
| `storm_tracks.parquet` | Trayectorias de tormentas | year/storm_id |
| `alerts_history.parquet` | Historial de alertas | year/month |
| `event_summary.parquet` | Resumen de eventos | year/event_type |
| `predictions.parquet` | Predicciones ML | year/month |

---

## 8. ⚙️ Configuración

Ver `configs/default_config.yml` sección `synthetic_generation`:

```yaml
synthetic_generation:
  enabled: true
  seed: 42
  
  # Resolución temporal
  hourly_interpolation: true
  hours_per_day: 24
  
  # Tasas de eventos
  event_rates:
    storm_per_year_per_region: 
      tropical: 12
      temperate: 4
      polar: 1
    heatwave_probability: 0.02
    coldsnap_probability: 0.02
  
  # Parámetros meteorológicos
  weather_params:
    rain_gamma_shape: 2.0
    rain_gamma_scale: 5.0
    wind_weibull_shape: 2.0
    wind_weibull_scale: 15.0
  
  # Rutas de salida
  output:
    hdfs_path: "/data/climaxtreme/synthetic"
    local_path: "DATA/synthetic"
    partitions: ["year", "month", "country"]
```

---

## 9. 🔄 Ejecución

```bash
# Generar datos sintéticos (batch)
climaxtreme generate-synthetic --input-path DATA/GlobalLandTemperaturesByCity.csv --output-path DATA/synthetic

# Iniciar streaming demo
climaxtreme stream-synthetic --rate 100 --duration 3600

# Entrenar modelos
climaxtreme train-models --data-path DATA/synthetic --model-type intensity

# Lanzar dashboard
climaxtreme dashboard --port 8501
```

---

## 10. 📈 Métricas de Calidad

### Validación de Datos Sintéticos

| Métrica | Criterio | Umbral |
|---------|----------|--------|
| Correlación temp horaria vs diaria | Pearson | > 0.95 |
| Distribución precipitación | KS test | p > 0.05 |
| Frecuencia tormentas | χ² test | p > 0.05 |
| Cobertura geográfica | % ciudades | 100% |
| Consistencia temporal | Gaps | 0 |

---

## Apéndice A: Dependencias

```
pyspark>=3.4.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
lightgbm>=4.0.0
plotly>=5.15.0
streamlit>=1.28.0
pyarrow>=12.0.0
```

## Apéndice B: Referencias

1. Holland, G. J. (1980). An analytic model of the wind and pressure profiles in hurricanes.
2. Wilks, D. S. (2011). Statistical Methods in the Atmospheric Sciences.
3. Stern, R. D. (1980). The calculation of probability distributions for models of daily precipitation.
