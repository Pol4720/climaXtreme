#!/bin/bash

# ============================================================================
# climaXtreme - Script 02: Iniciar Infraestructura
# ============================================================================
#
# Este script inicia todos los servicios necesarios para climaXtreme:
#   - HDFS (NameNode + DataNodes)
#   - Apache Kafka + Zookeeper
#   - Contenedor de procesamiento Spark
#   - Dashboard Streamlit
#
# Uso:
#   ./02_start_infrastructure.sh [--no-kafka] [--only-hdfs] [--only-kafka]
#
# Opciones:
#   --no-kafka     Iniciar sin Kafka (solo HDFS + Dashboard)
#   --only-hdfs    Iniciar solo HDFS
#   --only-kafka   Iniciar solo Kafka (requiere HDFS ya iniciado)
#   --detach       No mostrar logs (ejecutar en background)
#
# ============================================================================

set -e

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

NO_KAFKA=false
ONLY_HDFS=false
ONLY_KAFKA=false
DETACH=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-kafka)
            NO_KAFKA=true
            shift
            ;;
        --only-hdfs)
            ONLY_HDFS=true
            shift
            ;;
        --only-kafka)
            ONLY_KAFKA=true
            shift
            ;;
        --detach)
            DETACH=true
            shift
            ;;
        --logs)
            DETACH=false
            shift
            ;;
        -h|--help)
            head -25 "$0" | tail -20
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

write_info() {
    echo -e "${YELLOW}➤ $1${NC}"
}

# Detectar directorio raíz
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Detectar comando de compose
if command -v docker-compose &>/dev/null; then
    COMPOSE_CMD="docker-compose"
else
    COMPOSE_CMD="docker compose"
fi

cd "$REPO_ROOT/infra"

write_header "climaXtreme - Iniciando Infraestructura"

# ============================================================================
# Función para esperar que un contenedor esté healthy
# ============================================================================

wait_for_container() {
    local container=$1
    local max_wait=${2:-120}
    local start=$(date +%s)
    
    echo -n "Esperando $container"
    
    while true; do
        local status=$(docker inspect "$container" --format '{{.State.Status}}' 2>/dev/null || echo "not-found")
        local health=$(docker inspect "$container" --format '{{.State.Health.Status}}' 2>/dev/null || echo "none")
        
        if [ "$status" = "running" ]; then
            if [ "$health" = "healthy" ] || [ "$health" = "none" ]; then
                echo -e " ${GREEN}✓${NC}"
                return 0
            fi
        fi
        
        local elapsed=$(($(date +%s) - start))
        if [ $elapsed -ge $max_wait ]; then
            echo -e " ${RED}TIMEOUT${NC}"
            return 1
        fi
        
        echo -n "."
        sleep 2
    done
}

# ============================================================================
# Iniciar HDFS
# ============================================================================

if [ "$ONLY_KAFKA" = false ]; then
    write_header "1. Iniciando HDFS"
    
    echo "Iniciando NameNode y DataNodes..."
    $COMPOSE_CMD up -d namenode datanode1 datanode2 datanode3
    
    wait_for_container "climaxtreme-namenode" 120
    wait_for_container "climaxtreme-datanode1" 60
    
    write_success "HDFS iniciado"
    echo ""
    echo -e "  NameNode UI: ${CYAN}http://localhost:9870${NC}"
fi

# ============================================================================
# Iniciar Kafka
# ============================================================================

if [ "$ONLY_HDFS" = false ] && [ "$NO_KAFKA" = false ]; then
    write_header "2. Iniciando Kafka"
    
    echo "Iniciando Zookeeper y Kafka..."
    $COMPOSE_CMD up -d zookeeper kafka
    
    wait_for_container "climaxtreme-zookeeper" 60
    wait_for_container "climaxtreme-kafka" 90
    
    # Crear topics
    echo ""
    echo "Creando topics de Kafka..."
    sleep 5  # Esperar a que Kafka esté completamente listo
    
    topics=("climaxtreme-weather" "climaxtreme-alerts" "climaxtreme-storms" "climaxtreme-predictions" "climaxtreme-progress")
    
    for topic in "${topics[@]}"; do
        docker exec climaxtreme-kafka kafka-topics.sh --create \
            --bootstrap-server localhost:9092 \
            --topic "$topic" \
            --partitions 3 \
            --replication-factor 1 \
            --if-not-exists 2>/dev/null || true
        echo -e "  ${GREEN}✓${NC} $topic"
    done
    
    write_success "Kafka iniciado"
    echo ""
    echo -e "  Kafka Broker: ${CYAN}climaxtreme-kafka:9092${NC}"
fi

# ============================================================================
# Iniciar Procesador y Dashboard
# ============================================================================

if [ "$ONLY_HDFS" = false ] && [ "$ONLY_KAFKA" = false ]; then
    write_header "3. Iniciando Procesador y Dashboard"
    
    echo "Iniciando contenedor de procesamiento Spark..."
    $COMPOSE_CMD up -d processor
    wait_for_container "climaxtreme-processor" 60
    write_success "Procesador iniciado"
    
    echo ""
    echo "Iniciando Dashboard Streamlit..."
    $COMPOSE_CMD up -d dashboard
    wait_for_container "climaxtreme-dashboard" 60
    write_success "Dashboard iniciado"
    
    echo ""
    echo -e "  Dashboard: ${CYAN}http://localhost:8501${NC}"
    echo -e "  Spark UI:  ${CYAN}http://localhost:4040${NC} (cuando hay jobs activos)"
fi

# ============================================================================
# Resumen
# ============================================================================

write_header "✅ INFRAESTRUCTURA INICIADA"

echo "Contenedores activos:"
docker ps --filter "name=climaxtreme" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | head -10

echo ""
echo -e "Siguiente paso:"
echo ""
echo -e "  ${CYAN}./03_load_data.sh${NC}"
echo ""
echo "Esto cargará el dataset en HDFS y generará datos sintéticos."
echo ""
