#!/bin/bash

# ============================================================================
# climaXtreme - Script 05: Detener Todo
# ============================================================================
#
# Este script detiene todos los servicios de climaXtreme de forma ordenada.
#
# Uso:
#   ./05_stop_all.sh [--keep-data] [--remove-images]
#
# Opciones:
#   --keep-data       Mantener volúmenes de datos (HDFS, Kafka)
#   --remove-images   Eliminar también las imágenes Docker
#   --force           No pedir confirmación
#
# ============================================================================

set -e

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

KEEP_DATA=false
REMOVE_IMAGES=false
FORCE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --keep-data)
            KEEP_DATA=true
            shift
            ;;
        --remove-images)
            REMOVE_IMAGES=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        -h|--help)
            head -20 "$0" | tail -15
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

write_header "climaXtreme - Detener Servicios"

# Confirmación
if [ "$FORCE" = false ]; then
    echo "Esto detendrá todos los servicios de climaXtreme."
    
    if [ "$KEEP_DATA" = false ]; then
        echo -e "${YELLOW}⚠ Los volúmenes de datos serán eliminados.${NC}"
    fi
    
    echo ""
    read -p "¿Continuar? (s/n): " confirm
    if [ "$confirm" != "s" ]; then
        echo "Cancelado."
        exit 0
    fi
fi

cd "$REPO_ROOT/infra"

# ============================================================================
# Detener contenedores
# ============================================================================

write_header "1. Deteniendo contenedores"

echo "Deteniendo servicios..."

if [ "$KEEP_DATA" = true ]; then
    $COMPOSE_CMD down --remove-orphans
    write_success "Contenedores detenidos (datos preservados)"
else
    $COMPOSE_CMD down -v --remove-orphans
    write_success "Contenedores detenidos y volúmenes eliminados"
fi

# ============================================================================
# Eliminar imágenes si se solicitó
# ============================================================================

if [ "$REMOVE_IMAGES" = true ]; then
    write_header "2. Eliminando imágenes"
    
    images=(
        "infra-processor"
        "infra-dashboard"
    )
    
    for image in "${images[@]}"; do
        if docker image inspect "$image" &>/dev/null; then
            docker rmi "$image" 2>/dev/null || true
            write_success "Eliminada: $image"
        fi
    done
fi

# ============================================================================
# Verificar
# ============================================================================

write_header "Estado Final"

remaining=$(docker ps --filter "name=climaxtreme" --format '{{.Names}}' | wc -l)

if [ "$remaining" -eq 0 ]; then
    write_success "Todos los servicios detenidos"
else
    echo "Contenedores aún activos:"
    docker ps --filter "name=climaxtreme" --format "table {{.Names}}\t{{.Status}}"
fi

echo ""
echo "Para volver a iniciar:"
echo ""
echo -e "  ${CYAN}./02_start_infrastructure.sh${NC}"
echo ""
