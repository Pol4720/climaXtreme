#!/bin/bash

# ============================================================================
# climaXtreme - Script 04: Iniciar Streaming Kafka
# ============================================================================
#
# Este script inicia el pipeline de streaming en tiempo real:
#   Spark (genera datos) → Kafka (transmite) → Dashboard (visualiza)
#
# Uso:
#   ./04_start_streaming.sh [--cities N] [--interval S] [--continuous]
#
# Opciones:
#   --cities N       Número de ciudades a generar (default: 50)
#   --interval S     Segundos entre batches (default: 1.0)
#   --continuous     Ejecutar en modo continuo (loop infinito)
#   --background     Ejecutar en segundo plano
#
# Ejemplos:
#   ./04_start_streaming.sh                     # Generación única
#   ./04_start_streaming.sh --continuous        # Streaming continuo
#   ./04_start_streaming.sh --cities 100        # 100 ciudades
#
# ============================================================================

set -e

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

N_CITIES=50
INTERVAL=1.0
CONTINUOUS=false
BACKGROUND=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cities)
            N_CITIES="$2"
            shift 2
            ;;
        --interval)
            INTERVAL="$2"
            shift 2
            ;;
        --continuous)
            CONTINUOUS=true
            shift
            ;;
        --background)
            BACKGROUND=true
            shift
            ;;
        -h|--help)
            head -28 "$0" | tail -23
            exit 0
            ;;
        *)
            echo "Opción desconocida: $1"
            exit 1
            ;;
    esac
done

write_header() {
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
}

write_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

write_fail() {
    echo -e "${RED}✗ $1${NC}"
}

write_info() {
    echo -e "${YELLOW}➤ $1${NC}"
}

# Detectar directorio raíz
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

write_header "climaXtreme - Streaming Kafka"

# ============================================================================
# Verificar Kafka
# ============================================================================

echo "Verificando Kafka..."

if ! docker ps --format '{{.Names}}' | grep -q "climaxtreme-kafka"; then
    write_fail "Kafka no está corriendo"
    echo ""
    echo "Ejecuta primero: ./02_start_infrastructure.sh"
    exit 1
fi

write_success "Kafka está corriendo"

# Verificar topics
echo "Verificando topics..."
TOPICS=$(docker exec climaxtreme-kafka kafka-topics.sh --list --bootstrap-server localhost:9092 2>/dev/null | grep climaxtreme || true)

if [ -z "$TOPICS" ]; then
    write_info "Creando topics..."
    
    for topic in climaxtreme-weather climaxtreme-alerts climaxtreme-storms climaxtreme-predictions climaxtreme-progress; do
        docker exec climaxtreme-kafka kafka-topics.sh --create \
            --bootstrap-server localhost:9092 \
            --topic "$topic" \
            --partitions 3 \
            --replication-factor 1 \
            --if-not-exists 2>/dev/null || true
    done
fi

write_success "Topics listos"

# ============================================================================
# Verificar Procesador
# ============================================================================

if ! docker ps --format '{{.Names}}' | grep -q "climaxtreme-processor"; then
    write_fail "Contenedor de procesamiento no está corriendo"
    exit 1
fi

write_success "Procesador listo"

# ============================================================================
# Ejecutar Streaming
# ============================================================================

write_header "Iniciando Pipeline de Streaming"

echo "Configuración:"
echo "  Ciudades: $N_CITIES"
echo "  Intervalo: ${INTERVAL}s"
echo "  Modo: $([ "$CONTINUOUS" = true ] && echo "Continuo" || echo "Una vez")"
echo ""

# Script Python para streaming
STREAMING_SCRIPT="
from climaxtreme.streaming.spark_kafka_producer import (
    SparkKafkaStreamingProducer,
    ContinuousSparkKafkaProducer,
    SparkKafkaConfig
)
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

config = SparkKafkaConfig(
    kafka_bootstrap_servers='climaxtreme-kafka:9092',
    max_cities=$N_CITIES,
    batch_size=100,
    delay_between_batches=$INTERVAL,
    input_csv_path='/data/GlobalLandTemperaturesByCity.csv'
)

if $( [ "$CONTINUOUS" = true ] && echo "True" || echo "False" ):
    print('Iniciando streaming CONTINUO (Ctrl+C para detener)...')
    producer = ContinuousSparkKafkaProducer(config)
    producer.start_continuous(max_loops=0)
else:
    print('Iniciando generación única...')
    producer = SparkKafkaStreamingProducer(config)
    producer.start()
    producer.stop()

print('Streaming completado')
"

if [ "$BACKGROUND" = true ]; then
    echo "Ejecutando en segundo plano..."
    docker exec -d climaxtreme-processor python -c "$STREAMING_SCRIPT"
    write_success "Streaming iniciado en segundo plano"
    echo ""
    echo "Para ver logs: docker logs -f climaxtreme-processor"
    echo "Para detener: docker exec climaxtreme-processor pkill -f spark_kafka"
else
    echo "Ejecutando streaming..."
    echo ""
    docker exec climaxtreme-processor python -c "$STREAMING_SCRIPT"
fi

# ============================================================================
# Información
# ============================================================================

write_header "📊 Monitoreo"

echo "El streaming está enviando datos a Kafka."
echo ""
echo "Para ver los datos en tiempo real:"
echo ""
echo -e "  1. Abre el dashboard: ${CYAN}http://localhost:8501${NC}"
echo -e "  2. Ve a ${YELLOW}Streaming Hub${NC} → pestaña ${YELLOW}Kafka Streaming${NC}"
echo -e "  3. O ve directamente a ${YELLOW}Live Streaming${NC}"
echo ""
echo "Comandos útiles:"
echo ""
echo "  # Ver mensajes en un topic"
echo "  docker exec climaxtreme-kafka kafka-console-consumer.sh \\"
echo "    --bootstrap-server localhost:9092 \\"
echo "    --topic climaxtreme-weather --from-beginning --max-messages 5"
echo ""
echo "  # Ver estado de topics"
echo "  docker exec climaxtreme-kafka kafka-topics.sh --describe \\"
echo "    --bootstrap-server localhost:9092 --topic climaxtreme-weather"
echo ""
