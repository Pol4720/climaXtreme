#!/bin/bash

# Setup and load data to HDFS
# 
# Usage:
#   ./hdfs_setup_and_load.sh [--csv-path PATH] [--head N | --full-file]
#
# Options:
#   --csv-path PATH    Path to CSV file (default: DATA/GlobalLandTemperaturesByCity.csv)
#   --hdfs-dir PATH    HDFS directory (default: /data/climaxtreme)
#   --head N           Upload first N rows as sample
#   --full-file        Upload complete file

set -e

# Default values
CSV_PATH="DATA/GlobalLandTemperaturesByCity.csv"
HDFS_DIR="/data/climaxtreme"
HEAD=0
FULL_FILE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --csv-path)
            CSV_PATH="$2"
            shift 2
            ;;
        --hdfs-dir)
            HDFS_DIR="$2"
            shift 2
            ;;
        --head)
            HEAD="$2"
            shift 2
            ;;
        --full-file)
            FULL_FILE=true
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
NC='\033[0m'

# Resolve repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
COMPOSE_FILE="$REPO_ROOT/infra/docker-compose.yml"

if [ ! -f "$COMPOSE_FILE" ]; then
    echo -e "${RED}No se encontró el archivo de Docker Compose en: $COMPOSE_FILE${NC}"
    exit 1
fi

# Resolve CSV path
if [[ ! "$CSV_PATH" = /* ]]; then
    CSV_PATH="$REPO_ROOT/$CSV_PATH"
fi

if [ ! -f "$CSV_PATH" ]; then
    echo -e "${RED}No se encontró el archivo CSV: $CSV_PATH${NC}"
    exit 1
fi

# Check Docker
if ! command -v docker &>/dev/null; then
    echo -e "${RED}Docker no está instalado o no está en PATH${NC}"
    exit 1
fi

if ! docker info &>/dev/null; then
    echo -e "${RED}Docker no está en ejecución. Inicia Docker y vuelve a intentarlo.${NC}"
    exit 1
fi

# Start HDFS
echo "Iniciando HDFS (Docker Compose)..."
echo "Descargando imágenes si es necesario (esto puede tardar 2-3 min la primera vez)..."

cd "$REPO_ROOT/infra"

if command -v docker-compose &>/dev/null; then
    docker-compose up -d namenode datanode
else
    docker compose up -d namenode datanode
fi

if [ $? -ne 0 ]; then
    echo ""
    echo -e "${RED}ERROR: Docker Compose falló al iniciar los contenedores.${NC}"
    echo ""
    echo -e "${YELLOW}Posibles causas:${NC}"
    echo "  1. Docker no está corriendo correctamente"
    echo "  2. Error al descargar imágenes (verifica tu conexión a internet)"
    echo "  3. Puerto 9870 o 9000 ya está en uso"
    echo ""
    echo -e "${YELLOW}Soluciones:${NC}"
    echo "  - Reinicia Docker y vuelve a intentar"
    echo "  - Verifica logs: docker compose logs"
    echo "  - Limpia contenedores previos: docker compose down -v"
    echo ""
    exit 1
fi

cd "$REPO_ROOT"

echo "Esperando a que HDFS inicialice..."

# Wait for NameNode
MAX_WAIT=90
START=$(date +%s)
OK=false

while [ $OK = false ]; do
    sleep 3
    STATUS=$(docker inspect climaxtreme-namenode --format '{{.State.Status}}' 2>/dev/null || echo "not-found")
    
    if [ "$STATUS" == "running" ]; then
        # Check health
        HEALTH=$(docker inspect climaxtreme-namenode --format '{{.State.Health.Status}}' 2>/dev/null || echo "no-healthcheck")
        
        if [ "$HEALTH" == "healthy" ]; then
            OK=true
        elif [ "$HEALTH" == "no-healthcheck" ]; then
            echo "Contenedor iniciado (sin healthcheck), esperando 10s adicionales..."
            sleep 10
            OK=true
        elif [ "$HEALTH" == "starting" ]; then
            echo "NameNode iniciando (health: starting)..."
        else
            echo "NameNode health: $HEALTH"
        fi
    elif [ "$STATUS" != "not-found" ]; then
        echo "NameNode status: $STATUS (esperando 'running')..."
    fi
    
    NOW=$(date +%s)
    ELAPSED=$((NOW - START))
    
    if [ $ELAPSED -ge $MAX_WAIT ] && [ $OK = false ]; then
        echo ""
        echo -e "${RED}ERROR: El contenedor NameNode no arrancó correctamente en $MAX_WAIT segundos.${NC}"
        echo ""
        echo -e "${YELLOW}Ver logs del contenedor:${NC}"
        echo "  docker logs climaxtreme-namenode"
        echo "  docker logs climaxtreme-datanode"
        echo ""
        exit 1
    fi
done

# Determine upload mode
UPLOAD_FULL_FILE=false
if [ "$FULL_FILE" = true ] || [ "$HEAD" -eq 0 ]; then
    UPLOAD_FULL_FILE=true
fi

if [ "$UPLOAD_FULL_FILE" = true ]; then
    echo "Modo: Subir archivo COMPLETO a HDFS (esto puede tardar varios minutos para archivos grandes)"
    FILE_TO_UPLOAD="$CSV_PATH"
    HDFS_FILENAME="GlobalLandTemperaturesByCity.csv"
else
    echo "Modo: Subir SAMPLE ($HEAD filas) a HDFS"
    SAMPLE="/tmp/climaxtreme_sample.csv"
    echo "Creando sample ($HEAD filas) desde $CSV_PATH -> $SAMPLE"
    head -n $((HEAD + 1)) "$CSV_PATH" > "$SAMPLE"
    FILE_TO_UPLOAD="$SAMPLE"
    HDFS_FILENAME="GlobalLandTemperaturesByCity_sample.csv"
fi

echo "Creando directorio en HDFS: $HDFS_DIR"
docker exec climaxtreme-namenode hdfs dfs -mkdir -p "$HDFS_DIR" &>/dev/null || true

echo "Subiendo archivo a HDFS..."
echo "  Origen: $FILE_TO_UPLOAD"
echo "  Destino HDFS: $HDFS_DIR/$HDFS_FILENAME"

# Get file size
FILE_SIZE=$(stat -c%s "$FILE_TO_UPLOAD" 2>/dev/null || stat -f%z "$FILE_TO_UPLOAD" 2>/dev/null)
FILE_SIZE_MB=$(echo "scale=2; $FILE_SIZE / 1048576" | bc)
echo "  Tamaño: ${FILE_SIZE_MB} MB"

if (( $(echo "$FILE_SIZE_MB > 100" | bc -l) )); then
    echo "  NOTA: Este archivo es grande. La carga puede tardar varios minutos..."
fi

# Copy file to container
echo "  Copiando a contenedor..."
docker cp "$FILE_TO_UPLOAD" climaxtreme-namenode:/tmp/upload_file.csv

# Upload to HDFS
echo "  Copiando a HDFS..."
docker exec climaxtreme-namenode hdfs dfs -put -f /tmp/upload_file.csv "$HDFS_DIR/$HDFS_FILENAME"

# Clean up
docker exec climaxtreme-namenode rm -f /tmp/upload_file.csv

echo "Contenido de ${HDFS_DIR}:"
docker exec climaxtreme-namenode hdfs dfs -ls "$HDFS_DIR"

echo ""
echo -e "${GREEN}✓ Listo. Archivo subido exitosamente a HDFS${NC}"
echo "  URL HDFS: hdfs://climaxtreme-namenode:9000$HDFS_DIR/$HDFS_FILENAME"
echo ""

if [ "$UPLOAD_FULL_FILE" = true ]; then
    echo -e "${CYAN}Para procesar este dataset completo, ejecuta:${NC}"
    echo -e "${YELLOW}  docker exec -it climaxtreme-processor python -m climaxtreme.cli preprocess \\
    --input-path hdfs://climaxtreme-namenode:9000$HDFS_DIR/$HDFS_FILENAME \\
    --output-path hdfs://climaxtreme-namenode:9000$HDFS_DIR/processed \\
    --format city-csv${NC}"
fi
