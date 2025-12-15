# 📜 Scripts de climaXtreme

Scripts Bash para gestionar el pipeline completo de climaXtreme desde cero.

## 🚀 Inicio Rápido

```bash
# 1. Configurar entorno (solo primera vez)
./01_setup_environment.sh

# 2. Iniciar infraestructura
./02_start_infrastructure.sh

# 3. Cargar datos
./03_load_data.sh --all

# 4. (Opcional) Iniciar streaming en tiempo real
./04_start_streaming.sh
```

Una vez completado, accede al dashboard en: **http://localhost:8501**

---

## 📋 Descripción de Scripts

### `01_setup_environment.sh` - Configuración Inicial

Prepara el entorno para ejecutar climaXtreme por primera vez.

**Acciones:**
- Verifica que Docker esté instalado y corriendo
- Verifica la estructura del proyecto
- Descarga las imágenes Docker necesarias (~3GB total)
- Construye las imágenes del proyecto (processor, dashboard)
- Crea directorios necesarios

**Uso:**
```bash
./01_setup_environment.sh [opciones]
```

**Opciones:**
| Opción | Descripción |
|--------|-------------|
| `--skip-pull` | Omitir descarga de imágenes (usar existentes) |
| `--clean` | Limpiar todo y empezar desde cero |
| `-h, --help` | Mostrar ayuda |

**Ejemplo:**
```bash
# Primera instalación
./01_setup_environment.sh

# Reinstalar desde cero
./01_setup_environment.sh --clean
```

---

### `02_start_infrastructure.sh` - Iniciar Servicios

Inicia todos los contenedores necesarios.

**Servicios iniciados:**
- **HDFS**: NameNode + 3 DataNodes (almacenamiento distribuido)
- **Kafka**: Zookeeper + Broker (streaming en tiempo real)
- **Processor**: Contenedor con Spark para procesamiento
- **Dashboard**: Aplicación Streamlit

**Uso:**
```bash
./02_start_infrastructure.sh [opciones]
```

**Opciones:**
| Opción | Descripción |
|--------|-------------|
| `--no-kafka` | Iniciar sin Kafka |
| `--only-hdfs` | Iniciar solo HDFS |
| `--only-kafka` | Iniciar solo Kafka |
| `-h, --help` | Mostrar ayuda |

**Puertos expuestos:**
| Servicio | Puerto | URL |
|----------|--------|-----|
| Dashboard | 8501 | http://localhost:8501 |
| HDFS NameNode UI | 9870 | http://localhost:9870 |
| Spark UI | 4040 | http://localhost:4040 |
| Kafka | 9092 | climaxtreme-kafka:9092 |

**Ejemplo:**
```bash
# Iniciar todo
./02_start_infrastructure.sh

# Solo HDFS (sin Kafka)
./02_start_infrastructure.sh --no-kafka
```

---

### `03_load_data.sh` - Cargar Datos

Carga el dataset de temperaturas en HDFS y genera datos sintéticos.

**Requisitos:**
- Dataset CSV en `DATA/GlobalLandTemperaturesByCity.csv`
- Descargar de: https://www.kaggle.com/berkeleyearth/climate-change-earth-surface-temperature-data

**Uso:**
```bash
./03_load_data.sh [opciones]
```

**Opciones:**
| Opción | Descripción |
|--------|-------------|
| `--sample N` | Cargar solo N filas (para pruebas rápidas) |
| `--full` | Cargar dataset completo (~500MB) |
| `--synthetic` | Generar datos sintéticos con Spark |
| `--all` | Dataset completo + sintéticos (recomendado) |
| `--csv PATH` | Ruta alternativa al CSV |
| `-h, --help` | Mostrar ayuda |

**Ejemplos:**
```bash
# Carga completa recomendada
./03_load_data.sh --all

# Solo muestra para pruebas rápidas
./03_load_data.sh --sample 100000

# Solo generar sintéticos (si ya hay datos)
./03_load_data.sh --synthetic
```

**Estructura HDFS creada:**
```
/data/climaxtreme/
├── raw/                    # Datos originales
├── processed/              # Datos procesados
├── synthetic/              # Datos sintéticos generados
│   ├── synthetic_hourly.parquet
│   └── storm_tracks.parquet
└── streaming/              # Checkpoints de streaming
```

---

### `04_start_streaming.sh` - Streaming en Tiempo Real

Inicia el pipeline de streaming Kafka para visualización en tiempo real.

**Arquitectura:**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ SyntheticClimate│    │     Kafka       │    │    Dashboard    │
│ Generator       │───▶│     Broker      │───▶│    Streamlit    │
│ (Spark)         │    │     Topics      │    │    (Real-time)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

**Uso:**
```bash
./04_start_streaming.sh [opciones]
```

**Opciones:**
| Opción | Descripción |
|--------|-------------|
| `--cities N` | Número de ciudades a generar (default: 50) |
| `--interval S` | Segundos entre batches (default: 1.0) |
| `--continuous` | Modo continuo (loop infinito) |
| `--background` | Ejecutar en segundo plano |
| `-h, --help` | Mostrar ayuda |

**Topics de Kafka:**
| Topic | Contenido |
|-------|-----------|
| `climaxtreme-weather` | Datos meteorológicos |
| `climaxtreme-alerts` | Alertas activas |
| `climaxtreme-storms` | Tracking de tormentas |
| `climaxtreme-predictions` | Predicciones |
| `climaxtreme-progress` | Estado del pipeline |

**Ejemplos:**
```bash
# Generación única
./04_start_streaming.sh

# Streaming continuo con 100 ciudades
./04_start_streaming.sh --continuous --cities 100

# En segundo plano
./04_start_streaming.sh --continuous --background
```

---

### `05_stop_all.sh` - Detener Todo

Detiene todos los servicios de forma ordenada.

**Uso:**
```bash
./05_stop_all.sh [opciones]
```

**Opciones:**
| Opción | Descripción |
|--------|-------------|
| `--keep-data` | Mantener volúmenes de datos |
| `--remove-images` | Eliminar también imágenes Docker |
| `--force` | No pedir confirmación |
| `-h, --help` | Mostrar ayuda |

**Ejemplos:**
```bash
# Detener manteniendo datos
./05_stop_all.sh --keep-data

# Limpieza completa
./05_stop_all.sh --remove-images --force
```

---

### `check_status.sh` - Verificar Estado

Muestra el estado actual de todos los componentes.

**Uso:**
```bash
./check_status.sh
```

**Muestra:**
- Estado de contenedores Docker
- Archivos en HDFS
- Estado de Kafka y topics
- Métricas de recursos

---

## 🔧 Comandos Útiles

### Docker
```bash
# Ver todos los contenedores de climaXtreme
docker ps --filter "name=climaxtreme"

# Ver logs de un contenedor
docker logs -f climaxtreme-dashboard

# Ejecutar comando en contenedor
docker exec -it climaxtreme-processor bash
```

### HDFS
```bash
# Listar archivos
docker exec climaxtreme-namenode hdfs dfs -ls -R /data/climaxtreme/

# Ver tamaño de archivos
docker exec climaxtreme-namenode hdfs dfs -du -h /data/climaxtreme/

# Descargar archivo de HDFS
docker exec climaxtreme-namenode hdfs dfs -get /data/climaxtreme/file.csv /tmp/
docker cp climaxtreme-namenode:/tmp/file.csv ./
```

### Kafka
```bash
# Listar topics
docker exec climaxtreme-kafka kafka-topics.sh --list --bootstrap-server localhost:9092

# Consumir mensajes de un topic
docker exec climaxtreme-kafka kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic climaxtreme-weather \
  --from-beginning --max-messages 10

# Ver detalles de un topic
docker exec climaxtreme-kafka kafka-topics.sh --describe \
  --bootstrap-server localhost:9092 \
  --topic climaxtreme-weather
```

---

## 🐛 Solución de Problemas

### Docker no inicia
```bash
# Verificar que Docker esté corriendo
docker info

# En Linux, iniciar servicio
sudo systemctl start docker
```

### Puertos en uso
```bash
# Ver qué usa un puerto
lsof -i :8501
# o en Windows
netstat -ano | findstr :8501

# Cambiar puerto en docker-compose.yml si es necesario
```

### HDFS no está healthy
```bash
# Ver logs del namenode
docker logs climaxtreme-namenode

# Reiniciar HDFS
docker-compose -f infra/docker-compose.yml restart namenode datanode1
```

### Kafka no conecta
```bash
# Verificar que Zookeeper esté corriendo primero
docker logs climaxtreme-zookeeper

# Reiniciar Kafka
docker-compose -f infra/docker-compose.yml restart kafka
```

---

## 📁 Estructura del Proyecto

```
climaXtreme/
├── DATA/                           # Datasets
│   ├── GlobalLandTemperaturesByCity.csv
│   ├── processed/
│   └── synthetic/
├── infra/                          # Docker configuration
│   ├── docker-compose.yml
│   ├── Dockerfile.processor
│   └── hadoop.env
├── scripts/                        # Scripts de gestión (este directorio)
│   ├── 01_setup_environment.sh
│   ├── 02_start_infrastructure.sh
│   ├── 03_load_data.sh
│   ├── 04_start_streaming.sh
│   ├── 05_stop_all.sh
│   ├── check_status.sh
│   └── README.md
└── Tools/                          # Código Python
    └── src/climaxtreme/
        ├── dashboard/              # Streamlit app
        ├── preprocessing/          # Procesamiento Spark
        └── streaming/              # Kafka streaming
```

---

## 📞 Requisitos del Sistema

- **Docker**: 20.10+
- **docker-compose**: 2.0+ (o plugin de Docker)
- **RAM**: Mínimo 8GB, recomendado 16GB
- **Disco**: ~10GB para imágenes + datos
- **SO**: Linux, macOS, Windows (con WSL2)

---

## 🎯 Flujo Típico de Uso

```
Primera vez:
  01_setup_environment.sh → 02_start_infrastructure.sh → 03_load_data.sh --all

Uso diario:
  02_start_infrastructure.sh → (usar dashboard) → 05_stop_all.sh --keep-data

Streaming:
  02_start_infrastructure.sh → 04_start_streaming.sh --continuous
```
