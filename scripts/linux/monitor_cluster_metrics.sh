#!/bin/bash
#
# Script de monitoreo de métricas del clúster Docker para climaXtreme
# 
# Este script captura métricas de CPU, RAM y disco de los contenedores Docker
# durante la ejecución de jobs de Spark. Genera archivos CSV con las métricas
# y resumen para incluir en el informe técnico.
#
# Uso:
#   ./monitor_cluster_metrics.sh [duration_seconds] [interval_seconds]
#   ./monitor_cluster_metrics.sh 300 5
#

set -e

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Parámetros
DURATION=${1:-300}
INTERVAL=${2:-5}

# Directorios
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
OUTPUT_DIR="$PROJECT_ROOT/DATA/metrics"

# Crear directorio de salida
mkdir -p "$OUTPUT_DIR"

# Timestamp para archivos
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
METRICS_FILE="$OUTPUT_DIR/cluster_metrics_$TIMESTAMP.csv"
SUMMARY_FILE="$OUTPUT_DIR/metrics_summary_$TIMESTAMP.txt"

echo ""
echo -e "${CYAN}========================================"
echo "  MONITOREO DE MÉTRICAS - CLIMAXTREME"
echo -e "========================================${NC}"
echo ""
echo -e "⏱️  Duración: ${DURATION} segundos"
echo -e "📊 Intervalo: ${INTERVAL} segundos"
echo -e "💾 Archivo de salida: ${METRICS_FILE}"
echo ""

# Verificar que Docker está corriendo
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker no está corriendo. Por favor inicie Docker.${NC}"
    exit 1
fi

# Lista de contenedores a monitorear
CONTAINERS=(
    "climaxtreme-namenode"
    "climaxtreme-datanode1"
    "climaxtreme-datanode2"
    "climaxtreme-datanode3"
    "climaxtreme-processor"
    "climaxtreme-dashboard"
)

# Verificar contenedores activos
echo -e "${CYAN}🔍 Verificando contenedores...${NC}"
ACTIVE_CONTAINERS=()

for container in "${CONTAINERS[@]}"; do
    if docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null | grep -q "true"; then
        ACTIVE_CONTAINERS+=("$container")
        echo -e "  ${GREEN}✅ $container - Activo${NC}"
    else
        echo -e "  ${YELLOW}⚠️  $container - Inactivo${NC}"
    fi
done

if [ ${#ACTIVE_CONTAINERS[@]} -eq 0 ]; then
    echo -e "${RED}❌ No hay contenedores activos. Ejecute primero:${NC}"
    echo "   cd infra && docker-compose up -d"
    exit 1
fi

echo ""
echo -e "${CYAN}📈 Iniciando captura de métricas...${NC}"
echo "   Presione Ctrl+C para detener"
echo ""

# Cabecera del CSV
echo "timestamp,container,cpu_percent,mem_usage_mb,mem_limit_mb,mem_percent,net_io_rx_mb,net_io_tx_mb,block_io_read_mb,block_io_write_mb" > "$METRICS_FILE"

# Variables para estadísticas
declare -A CPU_SUM
declare -A MEM_SUM
declare -A COUNT

for container in "${ACTIVE_CONTAINERS[@]}"; do
    CPU_SUM[$container]=0
    MEM_SUM[$container]=0
    COUNT[$container]=0
done

# Función para parsear tamaño
parse_size() {
    local size_str=$1
    local value=$(echo "$size_str" | grep -oP '^[\d.]+')
    local unit=$(echo "$size_str" | grep -oP '[A-Za-z]+$')
    
    case "$unit" in
        "kB"|"KiB"|"KB") echo "scale=4; $value / 1024" | bc ;;
        "MB"|"MiB") echo "$value" ;;
        "GB"|"GiB") echo "scale=4; $value * 1024" | bc ;;
        "B") echo "scale=4; $value / 1048576" | bc ;;
        *) echo "$value" ;;
    esac
}

# Loop principal
START_TIME=$(date +%s)
ITERATIONS=$((DURATION / INTERVAL))
CURRENT=0

while [ $CURRENT -lt $ITERATIONS ]; do
    TIMESTAMP_NOW=$(date +"%Y-%m-%d %H:%M:%S")
    
    for container in "${ACTIVE_CONTAINERS[@]}"; do
        # Obtener stats
        STATS=$(docker stats "$container" --no-stream --format "{{.CPUPerc}},{{.MemUsage}},{{.MemPerc}},{{.NetIO}},{{.BlockIO}}" 2>/dev/null)
        
        if [ -n "$STATS" ]; then
            # Parsear valores
            CPU=$(echo "$STATS" | cut -d',' -f1 | tr -d '%')
            MEM_USAGE_RAW=$(echo "$STATS" | cut -d',' -f2 | cut -d'/' -f1 | tr -d ' ')
            MEM_LIMIT_RAW=$(echo "$STATS" | cut -d',' -f2 | cut -d'/' -f2 | tr -d ' ')
            MEM_PERCENT=$(echo "$STATS" | cut -d',' -f3 | tr -d '%')
            NET_RX=$(echo "$STATS" | cut -d',' -f4 | cut -d'/' -f1 | tr -d ' ')
            NET_TX=$(echo "$STATS" | cut -d',' -f4 | cut -d'/' -f2 | tr -d ' ')
            BLOCK_READ=$(echo "$STATS" | cut -d',' -f5 | cut -d'/' -f1 | tr -d ' ')
            BLOCK_WRITE=$(echo "$STATS" | cut -d',' -f5 | cut -d'/' -f2 | tr -d ' ')
            
            # Convertir a MB
            MEM_USAGE=$(parse_size "$MEM_USAGE_RAW")
            MEM_LIMIT=$(parse_size "$MEM_LIMIT_RAW")
            NET_RX_MB=$(parse_size "$NET_RX")
            NET_TX_MB=$(parse_size "$NET_TX")
            BLOCK_READ_MB=$(parse_size "$BLOCK_READ")
            BLOCK_WRITE_MB=$(parse_size "$BLOCK_WRITE")
            
            # Escribir al CSV
            echo "$TIMESTAMP_NOW,$container,$CPU,$MEM_USAGE,$MEM_LIMIT,$MEM_PERCENT,$NET_RX_MB,$NET_TX_MB,$BLOCK_READ_MB,$BLOCK_WRITE_MB" >> "$METRICS_FILE"
            
            # Acumular para estadísticas
            CPU_SUM[$container]=$(echo "${CPU_SUM[$container]} + $CPU" | bc)
            MEM_SUM[$container]=$(echo "${MEM_SUM[$container]} + $MEM_USAGE" | bc)
            COUNT[$container]=$((${COUNT[$container]} + 1))
        fi
    done
    
    CURRENT=$((CURRENT + 1))
    ELAPSED=$((CURRENT * INTERVAL))
    PERCENT=$((ELAPSED * 100 / DURATION))
    
    echo -ne "\r  Progreso: $PERCENT% ($ELAPSED/${DURATION}s)   "
    
    if [ $CURRENT -lt $ITERATIONS ]; then
        sleep $INTERVAL
    fi
done

echo ""
echo ""
echo -e "${GREEN}✅ Captura completada!${NC}"
echo ""

# Generar resumen
cat > "$SUMMARY_FILE" << EOF
================================================================================
              RESUMEN DE MÉTRICAS DEL CLÚSTER - CLIMAXTREME
================================================================================

Fecha de captura: $(date +"%Y-%m-%d %H:%M:%S")
Duración del monitoreo: $DURATION segundos
Intervalo de muestreo: $INTERVAL segundos

--------------------------------------------------------------------------------
ESTADÍSTICAS POR CONTENEDOR
--------------------------------------------------------------------------------

EOF

for container in "${ACTIVE_CONTAINERS[@]}"; do
    if [ ${COUNT[$container]} -gt 0 ]; then
        CPU_AVG=$(echo "scale=2; ${CPU_SUM[$container]} / ${COUNT[$container]}" | bc)
        MEM_AVG=$(echo "scale=2; ${MEM_SUM[$container]} / ${COUNT[$container]}" | bc)
        
        cat >> "$SUMMARY_FILE" << EOF

[$container]
  CPU Promedio: ${CPU_AVG}%
  RAM Promedio: ${MEM_AVG} MB
  Muestras: ${COUNT[$container]}

EOF
    fi
done

cat >> "$SUMMARY_FILE" << EOF
--------------------------------------------------------------------------------
ARCHIVOS GENERADOS
--------------------------------------------------------------------------------

- Métricas detalladas (CSV): $METRICS_FILE
- Este resumen: $SUMMARY_FILE

================================================================================
EOF

echo -e "${CYAN}📁 Archivos generados:${NC}"
echo "   - $METRICS_FILE"
echo "   - $SUMMARY_FILE"
echo ""

# Mostrar resumen rápido
echo -e "${CYAN}================== RESUMEN RÁPIDO ==================${NC}"
for container in "${ACTIVE_CONTAINERS[@]}"; do
    if [ ${COUNT[$container]} -gt 0 ]; then
        CPU_AVG=$(echo "scale=1; ${CPU_SUM[$container]} / ${COUNT[$container]}" | bc)
        MEM_AVG=$(echo "scale=0; ${MEM_SUM[$container]} / ${COUNT[$container]}" | bc)
        echo -e "  ${GREEN}$container${NC}"
        echo "    CPU: ${CPU_AVG}%  |  RAM: ${MEM_AVG} MB"
    fi
done
echo -e "${CYAN}=====================================================${NC}"
echo ""
echo -e "💡 Tip: Para generar gráficos, use:"
echo "   python scripts/generate_metrics_charts.py $METRICS_FILE"
