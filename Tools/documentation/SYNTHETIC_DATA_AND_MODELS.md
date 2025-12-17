# 📊 Datos Sintéticos y Modelos - climaXtreme

## Resumen Ejecutivo

Este documento describe el sistema de generación de datos sintéticos en tiempo real y los modelos de ML implementados en el proyecto climaXtreme. El sistema utiliza **Apache Kafka** para streaming y genera datos basándose en las **3,463 ciudades** del dataset histórico almacenado en HDFS.

---

## 1. 🎯 Arquitectura de Streaming

### 1.1 Componentes Implementados

```
┌─────────────────────────────────────────────────────────────────┐
│                    STREAMING ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  HDFS (anomalies.parquet)                                       │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────┐                                           │
│  │ KafkaStreamingManager │◄── Dashboard Control (Streamlit)    │
│  │ (kafka_manager.py)     │                                     │
│  └──────────────────┘                                           │
│         │                                                        │
│         │  Genera eventos cada 100ms-1000ms                     │
│         ▼                                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │               KAFKA BROKER (:9092)                        │   │
│  │  ┌─────────────────┐  ┌─────────────────┐                │   │
│  │  │climaxtreme-weather│  │climaxtreme-alerts│              │   │
│  │  │  weather_update  │  │  alert events   │                │   │
│  │  └─────────────────┘  └─────────────────┘                │   │
│  │  ┌─────────────────┐  ┌─────────────────┐                │   │
│  │  │climaxtreme-storms│  │climaxtreme-progress│            │   │
│  │  │  storm tracking │  │  stats/metrics  │                │   │
│  │  └─────────────────┘  └─────────────────┘                │   │
│  └──────────────────────────────────────────────────────────┘   │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │           STREAMLIT DASHBOARD (:8501)                     │   │
│  │  • Live Streaming       • Storm Tracking                  │   │
│  │  • Active Alerts        • Weather TimeSeries              │   │
│  │  • Intensity Prediction • EDA Validation                  │   │
│  │  • Historical Comparison                                  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Fuente de Datos: HDFS

El sistema carga dinámicamente las ciudades desde HDFS:

```python
# Archivo: kafka_manager.py
def load_cities_from_hdfs() -> list:
    """
    Carga 3,463 ciudades únicas desde anomalies.parquet.
    Detecta automáticamente si corre dentro o fuera del contenedor.
    """
    if _is_running_in_container():
        return _load_cities_direct_spark()  # Spark directo
    else:
        return _load_cities_via_subprocess()  # docker exec
```

**Datos extraídos de HDFS por ciudad:**
- `city`: Nombre de la ciudad
- `country`: País
- `lat_numeric`: Latitud decimal
- `lon_numeric`: Longitud decimal
- `continent`: Continente
- `zone`: Zona climática (inferida)

---

## 2. 📋 Esquemas de Datos en Streaming

### 2.1 Evento Weather (climaxtreme-weather)

```json
{
  "event_type": "weather_update",
  "timestamp": "2025-12-17T10:30:00",
  "city": "Madrid",
  "City": "Madrid",
  "country": "Spain",
  "Country": "Spain",
  "latitude": 40.42,
  "longitude": -3.70,
  "lat_decimal": 40.42,
  "lon_decimal": -3.70,
  "temperature": 18.5,
  "temperature_c": 18.5,
  "temperature_hourly": 18.5,
  "humidity": 65.0,
  "humidity_pct": 65.0,
  "pressure": 1015.2,
  "pressure_hpa": 1015.2,
  "wind_speed": 12.3,
  "wind_speed_kmh": 12.3,
  "wind_direction": 180.0,
  "rain_mm": 0.0,
  "cloud_cover": 45.0,
  "climate_zone": "MEDITERRANEAN",
  "event_intensity": 0.15,
  "year": 2025,
  "month": 12,
  "day": 17,
  "hour": 10,
  "day_of_week": 2,
  "generated_at": "2025-12-17T10:30:00"
}
```

### 2.2 Evento Alert (climaxtreme-alerts)

```json
{
  "event_type": "alert",
  "alert_id": "ALT-MAD-1734432600000",
  "timestamp": "2025-12-17T10:30:00",
  "city": "Madrid",
  "country": "Spain",
  "latitude": 40.42,
  "longitude": -3.70,
  "alert_type": "HEAT",
  "alert_level": "WARNING",
  "temperature": 42.5,
  "wind_speed": 15.0,
  "rain_mm": 0.0,
  "humidity_pct": 30.0,
  "event_intensity": 0.85,
  "climate_zone": "MEDITERRANEAN",
  "description": "HEAT alert for Madrid: WARNING",
  "generated_at": "2025-12-17T10:30:00"
}
```

### 2.3 Evento Storm (climaxtreme-storms)

```json
{
  "event_type": "storm_update",
  "storm_id": "STM-20251217-001",
  "storm_name": "Alpha",
  "timestamp": "2025-12-17T10:30:00",
  "latitude": 25.50,
  "longitude": -75.30,
  "category": 3,
  "max_wind_kmh": 185.5,
  "central_pressure": 965.2,
  "movement_speed": 25.0,
  "movement_direction": 315.0,
  "created_at": "2025-12-17T08:00:00",
  "update_number": 15,
  "generated_at": "2025-12-17T10:30:00"
}
```

---

## 3. 🔬 Modelos de Generación Implementados

### 3.1 Modelo de Temperatura

**Archivo:** `kafka_manager.py` - método `_run_simple_producer()`

```python
# Temperatura base por zona climática
base_temps = {
    "TROPICAL": 28,
    "SUBTROPICAL": 22,
    "TEMPERATE": 15,
    "CONTINENTAL": 8,
    "MEDITERRANEAN": 18,
    "DESERT": 32,
    "POLAR": -5
}

# Modelo implementado:
temp = base_temp + season_adj + diurnal + noise

Donde:
- base_temp: Temperatura base según zona climática
- season_adj: Ajuste estacional (cos((day_of_year - 172) * 2π/365) * 10)
             Invertido para hemisferio sur
- diurnal: Variación diurna (6 * sin((hour - 6) * π/12))
- noise: Ruido gaussiano N(0, 2²)
```

### 3.2 Modelo de Humedad

```python
humidity_base = {
    "TROPICAL": 80,
    "SUBTROPICAL": 70,
    "TEMPERATE": 65,
    "CONTINENTAL": 55,
    "MEDITERRANEAN": 50,
    "DESERT": 25,
    "POLAR": 60
}

humidity = humidity_base[zone] + N(0, 10²)
humidity = clamp(humidity, 10, 100)
```

### 3.3 Modelo de Viento

```python
wind_speed = |Weibull(shape=2, scale=15)|  # km/h
wind_direction = Uniform(0, 360)  # grados
```

### 3.4 Modelo de Precipitación

```python
rain_prob = 0.1 + (humidity - 50) / 200
rain_mm = Exponential(λ=1/8) if random() < rain_prob else 0
```

### 3.5 Modelo de Tormentas (Storm Tracking)

**Inicialización:**
```python
storm = {
    "latitude": Uniform(-20, 30),      # Zonas tropicales
    "longitude": Uniform(-180, 180),
    "category": randint(1, 3),
    "direction": Uniform(0, 360),
    "speed": Uniform(15, 35),          # km/h
    "max_wind_kmh": Uniform(120, 200),
    "central_pressure": Uniform(960, 1000)
}
```

**Actualización por tick:**
```python
# Movimiento (incremento significativo para trayectorias visibles)
move_dist = 0.15 + Uniform(0, 0.2)  # ~0.15-0.35 grados
dir_rad = radians(storm["direction"])
storm["latitude"] += cos(dir_rad) * move_dist
storm["longitude"] += sin(dir_rad) * move_dist

# Variación de intensidad
storm["category"] = clamp(category + choice([-1, 0, 0, 1]), 1, 5)
storm["max_wind_kmh"] = 60 + category * 40 + N(0, 10²)
storm["central_pressure"] = 1010 - category * 15 + N(0, 5²)
storm["direction"] += N(0, 10²)

# Ciclo de vida: termina si updates > 100 o |lat| > 60°
```

### 3.6 Modelo de Alertas

**Umbrales implementados:**

| Tipo | WATCH | WARNING | EMERGENCY |
|------|-------|---------|-----------|
| HEAT | T > 38°C | T > 40°C | T > 42°C |
| COLD | T < 0°C | T < -5°C | T < -15°C |
| WIND | V > 60 km/h | V > 80 km/h | V > 100 km/h |
| FLOOD | Rain > 20mm | Rain > 35mm | Rain > 50mm |

**Alertas aleatorias (configurable):**
```python
if random() < alert_probability:  # Default: 0.1
    alert_type = choice(["HEAT", "COLD", "WIND", "STORM", "FLOOD"])
    alert_level = choices(
        ["WATCH", "WARNING", "EMERGENCY"],
        weights=[0.6, 0.3, 0.1]
    )[0]
```

### 3.7 Intensidad del Evento

```python
# Normalización de anomalías (0-1)
temp_anomaly = |temp - base_temp| / 20
wind_anomaly = wind_speed / 50
rain_anomaly = min(rain_mm / 30, 1)

event_intensity = min(1.0, (temp_anomaly + wind_anomaly + rain_anomaly) / 3)
```

### 3.8 Inferencia de Zona Climática

**Archivo:** `kafka_manager.py` - función `_infer_climate_zone()`

```python
def _infer_climate_zone(lat: float, lon: float, continent: str) -> str:
    abs_lat = abs(lat)
    
    if abs_lat >= 66.5:
        return "POLAR"
    
    if abs_lat <= 23.5:
        if continent == "Africa" and 15 < abs_lat < 30:
            return "DESERT"
        if continent == "Asia" and 35 < lon < 75 and abs_lat > 20:
            return "DESERT"
        return "TROPICAL"
    
    if abs_lat <= 35:
        if continent == "Europe" or (continent == "Africa" and lat > 25):
            if -10 < lon < 40:
                return "MEDITERRANEAN"
        if continent in ["Africa", "Asia"] and abs_lat > 20:
            return "DESERT"
        return "SUBTROPICAL"
    
    if abs_lat <= 55:
        if continent in ["Asia", "North America"] and (lon > 90 or lon < -90):
            return "CONTINENTAL"
        if continent == "Europe" and lon > 20:
            return "CONTINENTAL"
        return "TEMPERATE"
    
    return "CONTINENTAL"
```

---

## 4. 🚀 Modelos de Machine Learning

### 4.1 Modelos Base (BaselineModel)

**Archivo:** `ml/baseline.py`

```python
class BaselineModel:
    """
    Modelos de regresión para predicción de temperatura.
    """
    MODELS = {
        'linear': LinearRegression(),
        'ridge': Ridge(alpha=1.0),
        'lasso': Lasso(alpha=1.0),
        'random_forest': RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),
        'gradient_boosting': GradientBoostingRegressor(
            n_estimators=100,
            random_state=42
        )
    }
```

**Features utilizadas:**
```python
features_df['year'] = df['year']
features_df['month'] = df['month']
features_df['year_normalized'] = (year - year_min) / (year_max - year_min)
features_df['month_sin'] = sin(2π * month / 12)
features_df['month_cos'] = cos(2π * month / 12)
```

### 4.2 Ensemble (ClimatePredictor)

**Archivo:** `ml/predictor.py`

```python
class ClimatePredictor:
    """
    Combina múltiples modelos usando VotingRegressor.
    Incluye Time Series Cross-Validation.
    """
    
    def __init__(self, models=['linear', 'ridge', 'random_forest']):
        self.ensemble_model = VotingRegressor(estimators=[
            ('linear', LinearRegression()),
            ('ridge', Ridge(alpha=1.0)),
            ('random_forest', RandomForestRegressor(n_estimators=100))
        ])
    
    def train_ensemble(self, df, n_splits=5):
        # Time Series Cross-Validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        # ...
```

**Métricas de evaluación:**
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- R² (Coefficient of Determination)

### 4.3 Predicción de Intensidad (Dashboard)

**Archivo:** `pages/13_🔮_Intensity_Prediction.py`

Modelo heurístico basado en anomalías históricas:

```python
# Modelo de intensidad (0-10)
intensity = (
    abs(temp_zscore) * 0.6 +      # Z-score de temperatura (60%)
    season_factor * 0.2 +          # Factor estacional (20%)
    uncertainty_factor * 0.2       # Incertidumbre histórica (20%)
) * 10

# Factor estacional: mayor intensidad en meses extremos
season_factor = abs(month - 6.5) / 6.5  # 0 en junio/julio, 1 en enero/diciembre

# Factor de incertidumbre
uncertainty_factor = uncertainty / uncertainty_max
```

---

## 5. 📊 Integración con Dashboard

### 5.1 Páginas de Streaming (Kafka)

| Página | Descripción | Topics Consumidos |
|--------|-------------|-------------------|
| 8_Streaming_Hub | Control del productor Kafka | Ninguno (productor) |
| 9_Live_Streaming | Visualización en tiempo real | weather, alerts |
| 11_Storm_Tracking | Seguimiento de tormentas | storms |
| 12_Active_Alerts | Monitor de alertas activas | alerts |

### 5.2 Páginas Híbridas (Kafka + HDFS)

| Página | Descripción | Fuentes de Datos |
|--------|-------------|------------------|
| 13_Intensity_Prediction | Predicción de intensidad | anomalies.parquet |
| 14_Weather_TimeSeries | Series temporales | weather + HDFS |
| 15_Streaming_Forecast | Pronóstico | weather + modelos |
| 16_EDA_Validation | Validación streaming vs histórico | Kafka + climatology.parquet |
| 17_Historical_Comparison | Comparación | Kafka + anomalies.parquet |

### 5.3 Configuración del Productor

```python
@dataclass
class KafkaStreamingConfig:
    n_cities: int = 100          # Ciudades a usar (max 3,463)
    interval_seconds: float = 1.0 # Intervalo entre batches
    include_alerts: bool = True
    include_storms: bool = True
    alert_probability: float = 0.1
    storm_probability: float = 0.15
    bootstrap_servers: str = "climaxtreme-kafka:9092"
```

---

## 6. 📁 Archivos de Código

| Archivo | Descripción |
|---------|-------------|
| `dashboard/components/kafka_manager.py` | Gestor de streaming Kafka (1,167 líneas) |
| `dashboard/components/kafka_realtime.py` | Consumidor Kafka para dashboard |
| `streaming/kafka_producer.py` | Productor standalone |
| `streaming/kafka_consumer.py` | Consumidor standalone |
| `ml/baseline.py` | Modelos base de ML (413 líneas) |
| `ml/predictor.py` | Ensemble y predictor avanzado (1,245 líneas) |

---

## 7. ⚙️ Ejecución

### Iniciar Streaming desde Dashboard

1. Abrir página **8_🌊_Streaming_Hub**
2. Verificar que Kafka esté corriendo (indicador verde)
3. Configurar parámetros (ciudades, intervalo, probabilidades)
4. Click en **🚀 Iniciar Productor**

### Iniciar Streaming Manual

```bash
# Desde el contenedor processor
docker exec -it climaxtreme-processor bash
cd /app/Tools
python -m climaxtreme.streaming.kafka_producer
```

### Ver Datos en Tiempo Real

```bash
# Consumir topic weather
docker exec climaxtreme-kafka kafka-console-consumer \
    --bootstrap-server localhost:9092 \
    --topic climaxtreme-weather \
    --from-beginning

# Consumir topic alerts
docker exec climaxtreme-kafka kafka-console-consumer \
    --bootstrap-server localhost:9092 \
    --topic climaxtreme-alerts
```

---

## 8. 📈 Métricas de Calidad

### Validación Implementada (EDA Validation)

| Test | Criterio | Umbral |
|------|----------|--------|
| Rango Válido | % temperaturas en [hist_min, hist_max] | ≥95% |
| Media Razonable | \|mean_streaming - mean_historical\| | ≤2σ |
| Variabilidad Suficiente | σ_streaming | ≥0.3 * σ_historical |
| Sin Extremos Imposibles | Temperaturas fuera de [-60, 60]°C | 0 |
| Humedad en Rango | % humedad en [0, 100] | ≥99% |
| Múltiples Ciudades | Ciudades únicas | ≥10 |
| Múltiples Zonas | Zonas climáticas únicas | ≥3 |

---

## 9. 🔧 Dependencias

```
kafka-python>=2.0.2
pyspark>=3.4.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
plotly>=5.15.0
streamlit>=1.28.0
```

---

## Apéndice: Ciudades de Respaldo

Si HDFS no está disponible, se usan 20 ciudades predefinidas:

```python
FALLBACK_CITIES = [
    {"city": "Madrid", "country": "Spain", "lat": 40.42, "lon": -3.70, "zone": "MEDITERRANEAN"},
    {"city": "London", "country": "United Kingdom", "lat": 51.51, "lon": -0.13, "zone": "TEMPERATE"},
    {"city": "Paris", "country": "France", "lat": 48.86, "lon": 2.35, "zone": "TEMPERATE"},
    {"city": "Tokyo", "country": "Japan", "lat": 35.68, "lon": 139.69, "zone": "TEMPERATE"},
    {"city": "New York", "country": "United States", "lat": 40.71, "lon": -74.01, "zone": "CONTINENTAL"},
    {"city": "Sydney", "country": "Australia", "lat": -33.87, "lon": 151.21, "zone": "SUBTROPICAL"},
    {"city": "Dubai", "country": "UAE", "lat": 25.20, "lon": 55.27, "zone": "DESERT"},
    {"city": "Mumbai", "country": "India", "lat": 19.08, "lon": 72.88, "zone": "TROPICAL"},
    {"city": "Cairo", "country": "Egypt", "lat": 30.04, "lon": 31.24, "zone": "DESERT"},
    {"city": "Moscow", "country": "Russia", "lat": 55.75, "lon": 37.62, "zone": "CONTINENTAL"},
    # ... 10 más
]
```
