#!/bin/bash

# ============================================================================
# climaXtreme - Script 01: Configuración del Entorno
# ============================================================================
#
# Este script prepara el entorno para ejecutar climaXtreme por primera vez.
# Verifica dependencias, descarga imágenes Docker y prepara la estructura.
#
# Uso:
#   ./01_setup_environment.sh [--skip-pull] [--clean]
#
# Opciones:
#   --skip-pull    Omitir descarga de imágenes Docker (usar las existentes)
#   --clean        Limpiar todo y empezar desde cero
#
# Requisitos previos:
#   - Docker instalado y corriendo
#   - docker-compose instalado
#   - Git (para clonar el repositorio si no existe)
#
# ============================================================================

set -e

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
GRAY='\033[0;37m'
NC='\033[0m'

SKIP_PULL=false
CLEAN_START=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-pull)
            SKIP_PULL=true
            shift
            ;;
        --clean)
            CLEAN_START=true
            shift
            ;;
        -h|--help)
            head -30 "$0" | tail -25
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
    echo -e "${GRAY}  $1${NC}"
}

write_warn() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Detectar directorio raíz del proyecto
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

write_header "climaXtreme - Configuración del Entorno"
echo -e "Directorio del proyecto: ${CYAN}$REPO_ROOT${NC}"
echo ""

# ============================================================================
# 1. Verificar Docker
# ============================================================================

write_header "1. Verificando Docker"

if ! command -v docker &>/dev/null; then
    write_fail "Docker no está instalado"
    echo ""
    echo "Instala Docker desde: https://docs.docker.com/get-docker/"
    exit 1
fi
write_success "Docker está instalado"

if ! docker info &>/dev/null; then
    write_fail "Docker no está corriendo"
    echo ""
    echo "Inicia Docker Desktop o el servicio de Docker y vuelve a ejecutar este script."
    exit 1
fi
write_success "Docker está corriendo"

# Verificar docker-compose
if command -v docker-compose &>/dev/null; then
    COMPOSE_CMD="docker-compose"
    write_success "docker-compose está instalado"
elif docker compose version &>/dev/null; then
    COMPOSE_CMD="docker compose"
    write_success "docker compose (plugin) está instalado"
else
    write_fail "docker-compose no está instalado"
    exit 1
fi

# ============================================================================
# 2. Verificar estructura del proyecto
# ============================================================================

write_header "2. Verificando estructura del proyecto"

# Archivos requeridos
required_files=(
    "infra/docker-compose.yml"
    "infra/Dockerfile.processor"
    "Tools/requirements-docker.txt"
    "Tools/src/climaxtreme/__init__.py"
)

for file in "${required_files[@]}"; do
    if [ -f "$REPO_ROOT/$file" ]; then
        write_success "$file"
    else
        write_fail "$file no encontrado"
        exit 1
    fi
done

# ============================================================================
# 3. Verificar dataset
# ============================================================================

write_header "3. Verificando dataset"

DATA_FILE="$REPO_ROOT/DATA/GlobalLandTemperaturesByCity.csv"

if [ -f "$DATA_FILE" ]; then
    FILE_SIZE=$(du -h "$DATA_FILE" | cut -f1)
    LINE_COUNT=$(wc -l < "$DATA_FILE")
    write_success "Dataset encontrado: $FILE_SIZE ($LINE_COUNT líneas)"
else
    write_warn "Dataset no encontrado en: $DATA_FILE"
    echo ""
    echo "Descarga el dataset de Kaggle:"
    echo "  https://www.kaggle.com/berkeleyearth/climate-change-earth-surface-temperature-data"
    echo ""
    echo "Y coloca el archivo GlobalLandTemperaturesByCity.csv en la carpeta DATA/"
    echo ""
    read -p "¿Deseas continuar sin el dataset? (s/n): " continue_without
    if [ "$continue_without" != "s" ]; then
        exit 1
    fi
fi

# ============================================================================
# 4. Limpiar si se solicitó
# ============================================================================

if [ "$CLEAN_START" = true ]; then
    write_header "4. Limpiando instalación anterior"
    
    cd "$REPO_ROOT/infra"
    
    echo "Deteniendo y eliminando contenedores..."
    $COMPOSE_CMD down -v --remove-orphans 2>/dev/null || true
    
    echo "Eliminando imágenes del proyecto..."
    docker rmi infra-processor infra-dashboard 2>/dev/null || true
    
    write_success "Limpieza completada"
fi

# ============================================================================
# 5. Descargar imágenes Docker
# ============================================================================

write_header "5. Descargando imágenes Docker"

if [ "$SKIP_PULL" = false ]; then
    echo "Esto puede tardar varios minutos la primera vez..."
    echo ""
    
    cd "$REPO_ROOT/infra"
    
    # Pull de imágenes base
    images=(
        "bde2020/hadoop-namenode:2.0.0-hadoop3.2.1-java8"
        "bde2020/hadoop-datanode:2.0.0-hadoop3.2.1-java8"
        "bitnami/spark:3.5"
        "python:3.10-slim"
        "bitnami/zookeeper:3.8"
        "bitnami/kafka:3.6"
    )
    
    for image in "${images[@]}"; do
        echo -e "Descargando ${YELLOW}$image${NC}..."
        docker pull "$image" || write_warn "No se pudo descargar $image (puede no ser necesaria)"
    done
    
    write_success "Imágenes descargadas"
else
    write_info "Omitiendo descarga de imágenes (--skip-pull)"
fi

# ============================================================================
# 6. Construir imágenes del proyecto
# ============================================================================

write_header "6. Construyendo imágenes del proyecto"

cd "$REPO_ROOT/infra"

echo "Construyendo imagen del procesador Spark..."
$COMPOSE_CMD build processor

echo "Construyendo imagen del dashboard..."
$COMPOSE_CMD build dashboard

write_success "Imágenes construidas"

# ============================================================================
# 7. Crear directorios necesarios
# ============================================================================

write_header "7. Creando directorios"

directories=(
    "$REPO_ROOT/DATA/processed"
    "$REPO_ROOT/DATA/synthetic"
    "$REPO_ROOT/logs"
)

for dir in "${directories[@]}"; do
    mkdir -p "$dir"
    write_success "$(basename $dir)/"
done

# ============================================================================
# Resumen
# ============================================================================

write_header "✅ CONFIGURACIÓN COMPLETADA"

echo -e "El entorno está listo. Siguiente paso:"
echo ""
echo -e "  ${CYAN}./02_start_infrastructure.sh${NC}"
echo ""
echo "Esto iniciará todos los servicios (HDFS, Kafka, Dashboard)."
echo ""
