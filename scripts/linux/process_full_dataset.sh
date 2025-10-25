#!/bin/bash

# Script completo para cargar y procesar el dataset completo en HDFS + Spark
#
# Este script automatiza todo el proceso:
# 1. Verifica que Docker esté corriendo
# 2. Inicia HDFS (namenode + datanode)
# 3. Sube el dataset COMPLETO a HDFS
# 4. Procesa el dataset con PySpark
# 5. Guarda los resultados procesados
# 6. (Opcional) Descarga los resultados a la carpeta local DATA/processed
#
# Usage:
#   ./process_full_dataset.sh [--csv-path PATH] [--skip-upload] [--skip-download]
#
# Options:
#   --csv-path PATH      Path to CSV file (default: DATA/GlobalLandTemperaturesByCity.csv)
#   --skip-upload        Skip upload (assume file is already in HDFS)
#   --skip-download      Don't download results (keep only in HDFS - recommended)

set -e

# Default values
CSV_PATH="DATA/GlobalLandTemperaturesByCity.csv"
SKIP_UPLOAD=false
SKIP_DOWNLOAD=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --csv-path)
            CSV_PATH="$2"
            shift 2
            ;;
        --skip-upload)
            SKIP_UPLOAD=true
            shift
            ;;
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

write_success() {
    echo -e "${GREEN}$1${NC}"
}

write_info() {
    echo -e "${CYAN}$1${NC}"
}

write_warning() {
    echo -e "${YELLOW}$1${NC}"
}

write_error() {
    echo -e "${RED}$1${NC}"
}

echo ""
echo -e "${MAGENTA}========================================${NC}"
echo -e "${MAGENTA}  climaXtreme - Procesamiento Completo${NC}"
echo -e "${MAGENTA}========================================${NC}"
echo ""

# Resolve paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
HDFS_SETUP_SCRIPT="$SCRIPT_DIR/hdfs_setup_and_load.sh"
OUTPUT_DIR="$REPO_ROOT/DATA/processed"

# HDFS paths
HDFS_INPUT_PATH="hdfs://climaxtreme-namenode:9000/data/climaxtreme/GlobalLandTemperaturesByCity.csv"
HDFS_OUTPUT_PATH="hdfs://climaxtreme-namenode:9000/data/climaxtreme/processed"

write_info "Configuración:"
echo "  Repo Root: $REPO_ROOT"
echo "  CSV Path: $CSV_PATH"
echo "  HDFS Input: $HDFS_INPUT_PATH"
echo "  HDFS Output: $HDFS_OUTPUT_PATH"
echo "  Local Output: $OUTPUT_DIR"
echo ""

# ============================================================================
# PASO 1: Cargar dataset a HDFS (si no se saltó)
# ============================================================================

if [ "$SKIP_UPLOAD" = false ]; then
    write_info "PASO 1/4: Cargando dataset COMPLETO a HDFS..."
    echo ""
    
    bash "$HDFS_SETUP_SCRIPT" --csv-path "$CSV_PATH" --full-file
    
    if [ $? -ne 0 ]; then
        write_error "Error al cargar el dataset a HDFS"
        exit 1
    fi
    
    write_success "✓ Dataset cargado exitosamente a HDFS"
    echo ""
else
    write_warning "PASO 1/4: Saltado (se asume que el archivo ya está en HDFS)"
    echo ""
fi

# ============================================================================
# PASO 2: Verificar que el contenedor processor esté corriendo
# ============================================================================

write_info "PASO 2/4: Verificando contenedor processor..."
echo ""

PROCESSOR_STATUS=$(docker inspect climaxtreme-processor --format '{{.State.Status}}' 2>/dev/null || echo "not-found")

if [ "$PROCESSOR_STATUS" != "running" ]; then
    write_warning "Contenedor processor no está corriendo. Iniciándolo..."
    docker start climaxtreme-processor >/dev/null
    sleep 5
fi

write_success "✓ Contenedor processor está listo"
echo ""

# ============================================================================
# PASO 3: Procesar dataset con PySpark
# ============================================================================

write_info "PASO 3/4: Procesando dataset con PySpark..."
write_info "  (Esto puede tardar varios minutos para datasets grandes)"
echo ""

SPARK_CMD="python -m climaxtreme.cli preprocess \
    --input-path '$HDFS_INPUT_PATH' \
    --output-path '$HDFS_OUTPUT_PATH' \
    --format city-csv"

echo "Ejecutando comando Spark en contenedor..."
docker exec climaxtreme-processor bash -c "$SPARK_CMD"

if [ $? -ne 0 ]; then
    write_error "Error al procesar el dataset con PySpark"
    exit 1
fi

write_success "✓ Procesamiento completado"
echo ""

# ============================================================================
# PASO 4: Descargar resultados procesados (OPCIONAL)
# ============================================================================

if [ "$SKIP_DOWNLOAD" = true ]; then
    write_info "PASO 4/4: Descarga omitida (modo HDFS-first)"
    echo ""
    write_success "Los archivos procesados están disponibles en HDFS:"
    echo "  $HDFS_OUTPUT_PATH/"
    echo ""
    write_info "Para ver el dashboard:"
    echo ""
    echo -e "${GREEN}  cd infra${NC}"
    echo -e "${GREEN}  docker-compose up -d dashboard${NC}"
    echo ""
    echo -e "${YELLOW}  Luego abre: http://localhost:8501${NC}"
    echo ""
    echo -e "${YELLOW}  En el sidebar del dashboard:${NC}"
    echo -e "${YELLOW}    1. Seleccionar: HDFS (Recommended)${NC}"
    echo -e "${YELLOW}    2. HDFS Host: namenode${NC}"
    echo -e "${YELLOW}    3. HDFS Port: 9000${NC}"
    echo -e "${YELLOW}    4. HDFS Base Path: /data/climaxtreme/processed${NC}"
    echo ""
else
    write_info "PASO 4/4: Descargando resultados procesados a carpeta local..."
    echo ""

    # Crear directorio de salida
    mkdir -p "$OUTPUT_DIR"

    # Descargar archivos Parquet desde HDFS (11 archivos: 8 agregaciones + 3 EDA)
    artifacts=(
        "monthly.parquet"
        "yearly.parquet"
        "anomalies.parquet"
        "climatology.parquet"
        "seasonal.parquet"
        "extreme_thresholds.parquet"
        "regional.parquet"
        "continental.parquet"
        "correlation_matrix.parquet"
        "descriptive_stats.parquet"
        "chi_square_tests.parquet"
    )

    for artifact in "${artifacts[@]}"; do
        echo "  Descargando $artifact..."
        
        # Get from HDFS to container temp
        container_temp_path="/tmp/$artifact"
        docker exec climaxtreme-namenode hdfs dfs -get "$HDFS_OUTPUT_PATH/$artifact" "$container_temp_path" 2>/dev/null || true
        
        # Copy from container to local
        local_artifact_path="$OUTPUT_DIR/$artifact"
        
        # Remove if exists
        if [ -d "$local_artifact_path" ]; then
            rm -rf "$local_artifact_path"
        fi
        
        docker cp "climaxtreme-namenode:$container_temp_path" "$local_artifact_path" >/dev/null 2>&1
        
        if [ -d "$local_artifact_path" ]; then
            write_success "    ✓ $artifact descargado"
        else
            write_warning "    ⚠ No se pudo descargar $artifact"
        fi
    done
fi

echo ""
write_success "========================================="
write_success "  ✓ PROCESAMIENTO COMPLETADO"
write_success "========================================="
echo ""

if [ "$SKIP_DOWNLOAD" = true ]; then
    write_info "Resultados disponibles en HDFS:"
    echo "  HDFS:   $HDFS_OUTPUT_PATH"
    echo ""
    write_info "Para ver el dashboard:"
    echo ""
    echo -e "${GREEN}  cd infra${NC}"
    echo -e "${GREEN}  docker-compose up -d dashboard${NC}"
    echo ""
    echo -e "${CYAN}  Abre: http://localhost:8501${NC}"
    echo ""
    echo -e "${YELLOW}  Luego en el sidebar del dashboard:${NC}"
    echo -e "${YELLOW}    1. Seleccionar: HDFS (Recommended)${NC}"
    echo -e "${YELLOW}    2. HDFS Host: namenode${NC}"
    echo -e "${YELLOW}    3. HDFS Port: 9000${NC}"
    echo -e "${YELLOW}    4. HDFS Base Path: /data/climaxtreme/processed${NC}"
    echo ""
else
    write_info "Resultados disponibles en:"
    echo "  Local:  $OUTPUT_DIR"
    echo "  HDFS:   $HDFS_OUTPUT_PATH"
    echo ""
    write_info "Para ver el dashboard:"
    echo ""
    echo -e "${GREEN}  cd infra${NC}"
    echo -e "${GREEN}  docker-compose up -d dashboard${NC}"
    echo ""
    echo -e "${CYAN}  Abre: http://localhost:8501${NC}"
    echo ""
    echo -e "${CYAN}  En el sidebar puedes seleccionar:${NC}"
    echo -e "${CYAN}    • HDFS (Recommended) - Lee directo desde Hadoop${NC}"
    echo -e "${CYAN}    • Local Files - Usa archivos en DATA/processed${NC}"
    echo ""
fi
