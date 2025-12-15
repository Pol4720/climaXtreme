#!/bin/bash
#
# Script de demostración para presentación oral - climaXtreme
#
# Este script automatiza la demostración del proyecto para la presentación.
# Incluye verificación de infraestructura, apertura de interfaces web,
# y un resumen de puntos clave para exponer.
#
# Uso:
#   ./demo_presentation.sh [mode]
#   Modos: quick, full, status
#

set -e

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Parámetros
MODE=${1:-quick}

# Directorios
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# ==============================================================================
# FUNCIONES DE UTILIDAD
# ==============================================================================

print_banner() {
    clear
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                                                                              ║"
    echo "║            ██████╗██╗     ██╗███╗   ███╗ █████╗ ██╗  ██╗████████╗██████╗     ║"
    echo "║           ██╔════╝██║     ██║████╗ ████║██╔══██╗╚██╗██╔╝╚══██╔══╝██╔══██╗    ║"
    echo "║           ██║     ██║     ██║██╔████╔██║███████║ ╚███╔╝    ██║   ██████╔╝    ║"
    echo "║           ██║     ██║     ██║██║╚██╔╝██║██╔══██║ ██╔██╗    ██║   ██╔══██╗    ║"
    echo "║           ╚██████╗███████╗██║██║ ╚═╝ ██║██║  ██║██╔╝ ██╗   ██║   ██║  ██║    ║"
    echo "║            ╚═════╝╚══════╝╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝    ║"
    echo "║                                                                              ║"
    echo "║                    🌡️  EXTREME CLIMATE ANALYSIS PLATFORM  🌡️                 ║"
    echo "║                                                                              ║"
    echo "║                 Procesamiento de Grandes Volúmenes de Datos                  ║"
    echo "║                               HDFS + PySpark                                 ║"
    echo "║                                                                              ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo ""
}

print_section() {
    echo ""
    echo -e "${MAGENTA}════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}   $1${NC}"
    echo -e "${MAGENTA}════════════════════════════════════════════════════════════════════${NC}"
    echo ""
}

print_status() {
    local status=$1
    local message=$2
    if [ "$status" = "OK" ]; then
        echo -e "   ${GREEN}✅ $message${NC}"
    elif [ "$status" = "WARN" ]; then
        echo -e "   ${YELLOW}⚠️  $message${NC}"
    elif [ "$status" = "ERROR" ]; then
        echo -e "   ${RED}❌ $message${NC}"
    else
        echo -e "   ${CYAN}ℹ️  $message${NC}"
    fi
}

check_docker() {
    if docker info > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

check_container() {
    local container=$1
    if docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null | grep -q "true"; then
        return 0
    else
        return 1
    fi
}

wait_for_key() {
    echo ""
    echo -e "${YELLOW}   Presione ENTER para continuar...${NC}"
    read -r
}

open_url() {
    local url=$1
    if command -v xdg-open > /dev/null; then
        xdg-open "$url" 2>/dev/null &
    elif command -v open > /dev/null; then
        open "$url" 2>/dev/null &
    else
        echo "   Abra manualmente: $url"
    fi
}

# ==============================================================================
# VERIFICACIÓN DE INFRAESTRUCTURA
# ==============================================================================

verify_infrastructure() {
    print_section "VERIFICACIÓN DE INFRAESTRUCTURA"
    
    # Docker
    echo -e "${CYAN}🐳 Docker Engine:${NC}"
    if check_docker; then
        print_status "OK" "Docker está corriendo"
    else
        print_status "ERROR" "Docker no está corriendo"
        echo ""
        echo "   Por favor inicie Docker Desktop y vuelva a ejecutar este script."
        exit 1
    fi
    echo ""
    
    # Contenedores
    echo -e "${CYAN}📦 Contenedores del Clúster:${NC}"
    
    CONTAINERS=(
        "climaxtreme-namenode:NameNode HDFS"
        "climaxtreme-datanode1:DataNode 1"
        "climaxtreme-datanode2:DataNode 2"
        "climaxtreme-datanode3:DataNode 3"
        "climaxtreme-processor:Spark Processor"
        "climaxtreme-dashboard:Dashboard Streamlit"
    )
    
    ALL_RUNNING=true
    for item in "${CONTAINERS[@]}"; do
        container="${item%%:*}"
        label="${item##*:}"
        if check_container "$container"; then
            print_status "OK" "$label - Activo"
        else
            print_status "ERROR" "$label - Inactivo"
            ALL_RUNNING=false
        fi
    done
    
    if [ "$ALL_RUNNING" = false ]; then
        echo ""
        echo -e "${YELLOW}   Iniciando contenedores...${NC}"
        cd "$PROJECT_ROOT/infra"
        docker-compose up -d
        echo ""
        echo "   Esperando 30 segundos para que los servicios inicien..."
        sleep 30
    fi
    
    echo ""
    echo -e "${CYAN}📊 Recursos de Contenedores:${NC}"
    docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" 2>/dev/null | head -10
}

# ==============================================================================
# VERIFICACIÓN DE HDFS
# ==============================================================================

verify_hdfs() {
    print_section "VERIFICACIÓN DE HDFS"
    
    echo -e "${CYAN}💾 Estado del Sistema de Archivos:${NC}"
    
    # Verificar NameNode
    if check_container "climaxtreme-namenode"; then
        # Verificar safemode
        SAFEMODE=$(docker exec climaxtreme-namenode hdfs dfsadmin -safemode get 2>/dev/null || echo "Error")
        if echo "$SAFEMODE" | grep -q "OFF"; then
            print_status "OK" "NameNode activo - Safe mode OFF"
        else
            print_status "WARN" "NameNode en safe mode: $SAFEMODE"
        fi
        
        # Listar datos
        echo ""
        echo -e "${CYAN}📁 Contenido de /data:${NC}"
        docker exec climaxtreme-namenode hdfs dfs -ls -R /data 2>/dev/null | head -20 || echo "   No hay datos cargados"
    else
        print_status "ERROR" "NameNode no está corriendo"
    fi
}

# ==============================================================================
# ABRIR INTERFACES WEB
# ==============================================================================

open_web_interfaces() {
    print_section "INTERFACES WEB"
    
    echo -e "${CYAN}🌐 Abriendo interfaces web...${NC}"
    echo ""
    
    INTERFACES=(
        "http://localhost:9870:HDFS NameNode Web UI"
        "http://localhost:8501:Dashboard Streamlit"
    )
    
    for item in "${INTERFACES[@]}"; do
        url="${item%%:*}:${item#*:}"
        url="${url%:*}"
        label="${item##*:}"
        echo -e "   ${GREEN}🔗 $label${NC}"
        echo "      $url"
        open_url "$url"
        sleep 2
    done
}

# ==============================================================================
# PUNTOS CLAVE PARA LA PRESENTACIÓN
# ==============================================================================

show_presentation_points() {
    print_section "PUNTOS CLAVE PARA LA PRESENTACIÓN"
    
    echo -e "${CYAN}📋 1. ARQUITECTURA${NC}"
    echo "   • Clúster HDFS: 1 NameNode + 3 DataNodes"
    echo "   • Factor de replicación: 3"
    echo "   • Procesamiento: PySpark en contenedor dedicado"
    echo "   • Visualización: Dashboard Streamlit (13 páginas)"
    echo ""
    
    echo -e "${CYAN}📊 2. DATASET${NC}"
    echo "   • Fuente: Berkeley Earth Climate Data (Kaggle)"
    echo "   • Volumen: ~8.6 millones de registros"
    echo "   • Período: 1743-2013 (270 años)"
    echo "   • Variables: Temperatura media, incertidumbre, ubicación"
    echo ""
    
    echo -e "${CYAN}⚙️ 3. PROCESAMIENTO${NC}"
    echo "   • Limpieza: Manejo de nulls, outliers, fechas"
    echo "   • Agregaciones: Mensual, anual, estacional, regional"
    echo "   • Análisis: Tendencias, anomalías, correlaciones"
    echo "   • ML: Predicción de temperatura y eventos extremos"
    echo ""
    
    echo -e "${CYAN}🤖 4. MODELOS DE MACHINE LEARNING${NC}"
    echo "   • VotingRegressor (ensemble)"
    echo "   • RandomForestRegressor"
    echo "   • GradientBoostingRegressor"
    echo "   • IntensityPredictor para eventos extremos"
    echo ""
    
    echo -e "${CYAN}📈 5. RESULTADOS GENERADOS (Parquets)${NC}"
    echo "   1. monthly.parquet     - Agregaciones mensuales"
    echo "   2. yearly.parquet      - Agregaciones anuales"
    echo "   3. seasonal.parquet    - Patrones estacionales"
    echo "   4. anomalies.parquet   - Eventos anómalos"
    echo "   5. regional.parquet    - Análisis regional"
    echo "   6. continental.parquet - Análisis continental"
    echo "   7. climate_zones.parquet"
    echo "   8. temperature_bins.parquet"
    echo "   9. correlation_matrix.parquet"
    echo "  10. descriptive_stats.parquet"
    echo "  11. chi_squared_results.parquet"
    echo ""
    
    echo -e "${CYAN}🎯 6. MÉTRICAS CLAVE A MOSTRAR${NC}"
    echo "   • Tiempo de procesamiento del dataset completo"
    echo "   • Uso de CPU/RAM durante el procesamiento"
    echo "   • Throughput (registros/segundo)"
    echo "   • Precisión de modelos ML (RMSE, R²)"
}

# ==============================================================================
# MODO DEMO COMPLETO
# ==============================================================================

run_full_demo() {
    print_banner
    echo -e "${YELLOW}   Modo: DEMOSTRACIÓN COMPLETA${NC}"
    echo "   Este modo ejecuta el pipeline completo y muestra todas las interfaces."
    wait_for_key
    
    verify_infrastructure
    wait_for_key
    
    verify_hdfs
    wait_for_key
    
    print_section "EJECUTANDO PIPELINE DE PROCESAMIENTO"
    echo -e "${CYAN}   Esto puede tomar varios minutos...${NC}"
    
    # Ejecutar procesamiento
    cd "$PROJECT_ROOT"
    bash scripts/linux/process_full_dataset.sh 2>&1 | tail -30
    
    wait_for_key
    
    open_web_interfaces
    wait_for_key
    
    show_presentation_points
}

# ==============================================================================
# MODO DEMO RÁPIDO
# ==============================================================================

run_quick_demo() {
    print_banner
    echo -e "${YELLOW}   Modo: DEMOSTRACIÓN RÁPIDA${NC}"
    echo "   Este modo verifica la infraestructura y abre las interfaces."
    wait_for_key
    
    verify_infrastructure
    wait_for_key
    
    verify_hdfs
    wait_for_key
    
    open_web_interfaces
    wait_for_key
    
    show_presentation_points
}

# ==============================================================================
# MODO VERIFICACIÓN DE ESTADO
# ==============================================================================

run_status_check() {
    print_banner
    echo -e "${YELLOW}   Modo: VERIFICACIÓN DE ESTADO${NC}"
    echo ""
    
    verify_infrastructure
    verify_hdfs
    
    print_section "RESUMEN"
    echo -e "${GREEN}✅ Verificación completada${NC}"
}

# ==============================================================================
# MAIN
# ==============================================================================

case "$MODE" in
    quick)
        run_quick_demo
        ;;
    full)
        run_full_demo
        ;;
    status)
        run_status_check
        ;;
    *)
        echo "Uso: $0 [quick|full|status]"
        echo ""
        echo "Modos:"
        echo "  quick  - Verificación + abrir interfaces (por defecto)"
        echo "  full   - Ejecutar pipeline completo + interfaces"
        echo "  status - Solo verificar estado"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}   ✅ Demostración finalizada${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════════════${NC}"
echo ""
