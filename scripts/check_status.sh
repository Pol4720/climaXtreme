#!/bin/bash

# Verifica el estado del sistema climaXtreme
# 
# Este script verifica:
# - Estado de contenedores Docker
# - Archivos en HDFS
# - Archivos procesados localmente
# - Tamaños y estadísticas

set -e

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
GRAY='\033[0;37m'
NC='\033[0m' # No Color

write_header() {
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════${NC}"
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

write_header "VERIFICACIÓN DEL SISTEMA climaXtreme"

# ============================================================================
# 1. Verificar Docker
# ============================================================================

write_header "1. Docker"

if docker info &>/dev/null; then
    write_success "Docker está corriendo"
else
    write_fail "Docker no está corriendo o no está disponible"
    exit 1
fi

# ============================================================================
# 2. Verificar Contenedores
# ============================================================================

write_header "2. Contenedores"

containers=("climaxtreme-namenode" "climaxtreme-datanode" "climaxtreme-processor" "climaxtreme-dashboard")

for container in "${containers[@]}"; do
    status=$(docker inspect "$container" --format '{{.State.Status}}' 2>/dev/null || echo "not-found")
    
    if [ "$status" == "running" ]; then
        write_success "$container está corriendo"
        
        # Get container stats
        stats=$(docker stats "$container" --no-stream --format "{{.CPUPerc}}\t{{.MemUsage}}" 2>/dev/null)
        write_info "Stats: $stats"
    elif [ "$status" != "not-found" ]; then
        write_fail "$container existe pero no está corriendo (estado: $status)"
    else
        write_fail "$container no existe"
    fi
done

# ============================================================================
# 3. Verificar Archivos en HDFS
# ============================================================================

write_header "3. Archivos en HDFS"

if docker exec climaxtreme-namenode hdfs dfs -ls -h /data/climaxtreme/ &>/dev/null; then
    write_success "Directorio /data/climaxtreme/ existe en HDFS"
    echo ""
    echo -e "${YELLOW}Contenido:${NC}"
    docker exec climaxtreme-namenode hdfs dfs -ls -h /data/climaxtreme/
    echo ""
    
    # Check for main dataset
    if docker exec climaxtreme-namenode hdfs dfs -test -e /data/climaxtreme/GlobalLandTemperaturesByCity.csv 2>/dev/null; then
        fileSize=$(docker exec climaxtreme-namenode hdfs dfs -du -h /data/climaxtreme/GlobalLandTemperaturesByCity.csv | head -1)
        write_success "Dataset completo encontrado: $fileSize"
    else
        if docker exec climaxtreme-namenode hdfs dfs -test -e /data/climaxtreme/GlobalLandTemperaturesByCity_sample.csv 2>/dev/null; then
            fileSize=$(docker exec climaxtreme-namenode hdfs dfs -du -h /data/climaxtreme/GlobalLandTemperaturesByCity_sample.csv | head -1)
            write_info "Solo sample encontrado: $fileSize"
        else
            write_fail "No se encontró dataset en HDFS"
        fi
    fi
else
    write_fail "No se puede acceder a HDFS o el directorio no existe"
fi

# ============================================================================
# 4. Verificar Archivos Procesados en HDFS
# ============================================================================

write_header "4. Archivos Procesados en HDFS"

if docker exec climaxtreme-namenode hdfs dfs -ls -h /data/climaxtreme/processed/ &>/dev/null; then
    write_success "Directorio /data/climaxtreme/processed/ existe"
    echo ""
    echo -e "${YELLOW}Contenido:${NC}"
    docker exec climaxtreme-namenode hdfs dfs -ls -h /data/climaxtreme/processed/
    echo ""
    
    artifacts=("monthly.parquet" "yearly.parquet" "anomalies.parquet" "correlation_matrix.parquet" "descriptive_stats.parquet" "chi_square_tests.parquet")
    for artifact in "${artifacts[@]}"; do
        if docker exec climaxtreme-namenode hdfs dfs -test -e "/data/climaxtreme/processed/$artifact" 2>/dev/null; then
            size=$(docker exec climaxtreme-namenode hdfs dfs -du -h "/data/climaxtreme/processed/$artifact" | head -1)
            write_success "$artifact encontrado: $size"
        else
            write_fail "$artifact NO encontrado"
        fi
    done
else
    write_fail "No se encontraron archivos procesados en HDFS"
    write_info "Ejecuta: ./scripts/linux/process_full_dataset.sh"
fi

# ============================================================================
# 5. Verificar Archivos Locales
# ============================================================================

write_header "5. Archivos Locales"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="$REPO_ROOT/DATA"
PROCESSED_DIR="$DATA_DIR/processed"

# Check raw CSV
CSV_PATH="$DATA_DIR/GlobalLandTemperaturesByCity.csv"
if [ -f "$CSV_PATH" ]; then
    size=$(du -h "$CSV_PATH" | cut -f1)
    write_success "CSV original encontrado: $size"
else
    write_fail "CSV original NO encontrado en $CSV_PATH"
fi

# Check processed files
if [ -d "$PROCESSED_DIR" ]; then
    write_success "Directorio processed/ existe"
    echo ""
    echo -e "${YELLOW}Contenido:${NC}"
    
    artifacts=("monthly.parquet" "yearly.parquet" "anomalies.parquet")
    for artifact in "${artifacts[@]}"; do
        artifact_path="$PROCESSED_DIR/$artifact"
        if [ -d "$artifact_path" ]; then
            size=$(du -sh "$artifact_path" | cut -f1)
            write_success "$artifact encontrado: $size"
        else
            write_fail "$artifact NO encontrado"
        fi
    done
else
    write_fail "Directorio processed/ NO existe"
    write_info "Ejecuta: ./scripts/linux/process_full_dataset.sh"
fi

# ============================================================================
# 6. Resumen y Recomendaciones
# ============================================================================

write_header "6. Resumen"

echo -e "${YELLOW}Estado del Sistema:${NC}"
echo ""

# Count issues
issues=0

# Check if dataset is uploaded
if ! docker exec climaxtreme-namenode hdfs dfs -test -e /data/climaxtreme/GlobalLandTemperaturesByCity.csv 2>/dev/null; then
    ((issues++))
    write_info "⚠ Dataset completo no está en HDFS"
fi

# Check if processed
if ! docker exec climaxtreme-namenode hdfs dfs -test -e /data/climaxtreme/processed/monthly.parquet 2>/dev/null; then
    ((issues++))
    write_info "⚠ Dataset no ha sido procesado"
fi

# Check if downloaded
if [ ! -d "$PROCESSED_DIR/monthly.parquet" ]; then
    ((issues++))
    write_info "⚠ Resultados procesados no están descargados localmente"
fi

echo ""

if [ $issues -eq 0 ]; then
    write_success "¡Todo está configurado correctamente!"
    echo ""
    write_info "El dashboard está disponible en: http://localhost:8501"
    write_info "Configuración HDFS:"
    write_info "  - Host: namenode"
    write_info "  - Port: 9000"
    write_info "  - Path: /data/climaxtreme/processed"
else
    echo -e "${YELLOW}Se encontraron $issues problemas.${NC}"
    echo ""
    write_info "Para configurar todo desde cero, ejecuta:"
    write_info "  ./scripts/linux/process_full_dataset.sh"
fi

echo ""
