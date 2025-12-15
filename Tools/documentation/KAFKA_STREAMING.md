# 📡 Sistema de Streaming con Apache Kafka - climaXtreme

## Descripción General

El sistema de streaming de climaXtreme utiliza **Apache Kafka** como tubería principal para la transmisión de datos meteorológicos en tiempo real. Este diseño permite que múltiples páginas del dashboard consuman datos simultáneamente y muestren visualizaciones que se actualizan automáticamente.

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INFRAESTRUCTURA DOCKER                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────┐     │
│  │  Zookeeper   │◄───►│    Kafka     │◄───►│   Kafka UI       │     │
│  │  (2181)      │     │  (9092/29092)│     │   (8080)         │     │
│  └──────────────┘     └──────────────┘     └──────────────────┘     │
│                              │                                       │
│                              ▼                                       │
│                    ┌─────────────────┐                              │
│                    │  Kafka Topics   │                              │
│                    │ - weather       │                              │
│                    │ - alerts        │                              │
│                    │ - storms        │                              │
│                    │ - predictions   │                              │
│                    └─────────────────┘                              │
│                         │       ▲                                    │
│                         ▼       │                                    │
│  ┌──────────────────────────────┴──────────────────────────┐        │
│  │                    PROCESSOR CONTAINER                   │        │
│  │  ┌────────────────────┐    ┌────────────────────┐       │        │
│  │  │ KafkaStreamingProd │───►│  ClimateKafkaProd  │       │        │
│  │  │ (Data Generator)   │    │  (Low-level API)   │       │        │
│  │  └────────────────────┘    └────────────────────┘       │        │
│  └─────────────────────────────────────────────────────────┘        │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │                    DASHBOARD CONTAINER                    │       │
│  │  ┌────────────────┐    ┌────────────────────────────┐   │       │
│  │  │ KafkaStreamStat│◄───│  Streamlit Pages           │   │       │
│  │  │ (Consumer)     │    │  - Live Streaming          │   │       │
│  │  └────────────────┘    │  - Storm Tracking          │   │       │
│  │                        │  - Active Alerts           │   │       │
│  │                        │  - Streaming Forecast      │   │       │
│  │                        └────────────────────────────────┘   │       │
│  └──────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────┘
```

## 📁 Estructura de Archivos

```
Tools/src/climaxtreme/
├── streaming/
│   ├── kafka_producer.py      # Productor de datos a Kafka
│   ├── kafka_consumer.py      # Consumidor de datos de Kafka
│   └── __init__.py            # Exports de Kafka
│
├── dashboard/
│   ├── components/
│   │   ├── kafka_realtime.py  # Componentes UI tiempo real
│   │   ├── kafka_manager.py   # Manager del streaming
│   │   └── __init__.py        # Exports
│   │
│   └── pages/
│       ├── 6_🌊_Streaming_Hub.py    # Control central Kafka
│       ├── 9_🌀_Storm_Tracking.py    # Tormentas (con pestaña Live)
│       ├── 10_🚨_Active_Alerts.py    # Alertas (con pestaña Live)
│       ├── 14_🌊_Streaming_Forecast.py # Pronóstico streaming
│       └── 16_🔴_Live_Streaming.py   # Dashboard tiempo real
│
scripts/windows/
└── manage_kafka_streaming.ps1  # Script de gestión

infra/
├── docker-compose.yml          # Servicios Kafka
└── Dockerfile.processor        # Dependencias Kafka
```

## 🚀 Inicio Rápido

### 1. Iniciar el Clúster Kafka

```bash
cd infra
docker-compose up -d zookeeper kafka
```

### 2. Verificar el Estado

```bash
# Windows PowerShell
.\scripts\windows\manage_kafka_streaming.ps1 status
```

### 3. Crear Topics

```bash
.\scripts\windows\manage_kafka_streaming.ps1 topics
```

### 4. Iniciar el Productor

**Opción A - Desde el script:**
```bash
.\scripts\windows\manage_kafka_streaming.ps1 producer
```

**Opción B - Desde el Dashboard:**
1. Abre el Streaming Hub (página 6)
2. Ve a la pestaña "🔴 Kafka Streaming"
3. Configura el número de ciudades e intervalo
4. Haz clic en "▶️ Iniciar Productor"

### 5. Ver Datos en Tiempo Real

1. Abre cualquiera de estas páginas:
   - **🔴 Live Streaming** - Dashboard principal tiempo real
   - **🌊 Streaming Forecast** - Pronósticos en vivo
   - **🌀 Storm Tracking** - Pestaña "En Vivo"
   - **🚨 Active Alerts** - Pestaña "En Vivo"

2. Conecta al consumer haciendo clic en "▶️ Iniciar Stream"

3. Los gráficos se actualizarán automáticamente

## 📊 Topics de Kafka

| Topic | Descripción | Contenido |
|-------|-------------|-----------|
| `climaxtreme-weather` | Datos meteorológicos | Temperatura, humedad, viento, etc. |
| `climaxtreme-alerts` | Alertas activas | Emergencias, advertencias, vigilancia |
| `climaxtreme-storms` | Tormentas | Trayectorias, categorías, vientos |
| `climaxtreme-predictions` | Predicciones | Pronósticos ML |
| `climaxtreme-progress` | Progreso | Estado de generación |

## 🎛️ Configuración

### Variables de Entorno

```env
KAFKA_BOOTSTRAP_SERVERS=climaxtreme-kafka:9092
```

### Configuración del Productor

```python
from climaxtreme.streaming import KafkaStreamingProducer

producer = KafkaStreamingProducer(
    n_cities=50,              # Número de ciudades
    interval_seconds=1.0,     # Intervalo entre batches
    include_alerts=True,      # Generar alertas
    include_storms=True,      # Generar tormentas
    bootstrap_servers="climaxtreme-kafka:9092"
)

producer.start_streaming()
```

### Configuración del Consumer

```python
from climaxtreme.streaming import ClimateKafkaConsumer

consumer = ClimateKafkaConsumer(
    topics=['climaxtreme-weather', 'climaxtreme-alerts'],
    bootstrap_servers="climaxtreme-kafka:9092"
)

# Consumir eventos
for event in consumer.consume_stream():
    print(event)
```

## 📈 Métricas y Monitoreo

### Kafka UI

```bash
docker-compose --profile monitoring up -d kafka-ui
# Accede a http://localhost:8080
```

### Estadísticas del Dashboard

Cada página muestra:
- Total de eventos recibidos
- Eventos en buffer
- Última actualización
- Errores de conexión

## 🔧 Troubleshooting

### Kafka no arranca

```bash
# Verificar logs
docker logs climaxtreme-kafka

# Verificar Zookeeper
docker logs climaxtreme-zookeeper
```

### No se reciben eventos

1. Verificar que el productor esté corriendo
2. Verificar que los topics existan
3. Verificar conectividad de red entre contenedores

### Eventos lentos

1. Reducir el intervalo del productor
2. Aumentar el número de particiones
3. Verificar recursos del contenedor

## 🏗️ Componentes Principales

### KafkaStreamState (Singleton)

Mantiene el estado del consumer entre reruns de Streamlit:
- Buffer de eventos thread-safe
- Estadísticas de consumo
- Control de start/stop

### KafkaStreamingProducer

Genera eventos sintéticos en tiempo real:
- Thread de background
- Generación por batches
- Múltiples tipos de eventos

### Dashboard Components

- `render_kafka_status_card()` - Estado de conexión
- `create_realtime_weather_chart()` - Gráfico temperatura
- `create_realtime_map()` - Mapa de eventos
- `create_realtime_alerts_panel()` - Panel de alertas

## 📚 Referencias

- [Apache Kafka Documentation](https://kafka.apache.org/documentation/)
- [kafka-python Library](https://kafka-python.readthedocs.io/)
- [Streamlit Auto-refresh](https://docs.streamlit.io/)
