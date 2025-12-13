#!/bin/bash

# ============================================================================
# climaXtreme - Script 03: Cargar Datos en HDFS
# ============================================================================
#
# Este script carga el dataset de temperaturas en HDFS y genera datos sintéticos.
#
# Uso:
#   ./03_load_data.sh [--sample N] [--full] [--synthetic] [--all]
#
# Opciones:
#   --sample N       Cargar solo las primeras N filas (para pruebas rápidas)
#   --full           Cargar el dataset completo (~500MB)
#   --synthetic      Generar datos sintéticos con Spark
#   --all            Cargar dataset completo + generar sintéticos (recomendado)
#   --csv PATH       Ruta al archivo CSV (default: DATA/GlobalLandTemperaturesByCity.csv)
#
# Ejemplos:
#   ./03_load_data.sh --sample 100000    # Carga muestra de 100k filas
#   ./03_load_data.sh --all              # Carga todo y genera sintéticos
#   ./03_load_data.sh --synthetic        # Solo genera sintéticos (si ya hay datos)
#
# ============================================================================

set -e

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

SAMPLE_SIZE=0
FULL_LOAD=false
GENERATE_SYNTHETIC=false
CSV_PATH=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --sample)
            SAMPLE_SIZE="$2"
            shift 2
            ;;
        --full)
            FULL_LOAD=true
            shift
            ;;
        --synthetic)
            GENERATE_SYNTHETIC=true
            shift
            ;;
        --all)
            FULL_LOAD=true
            GENERATE_SYNTHETIC=true
            shift
            ;;
        --csv)
            CSV_PATH="$2"
            shift 2
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

# Resolver ruta CSV
if [ -z "$CSV_PATH" ]; then
    CSV_PATH="$REPO_ROOT/DATA/GlobalLandTemperaturesByCity.csv"
fi

write_header "climaXtreme - Carga de Datos"

# ============================================================================
# Verificar prerequisitos
# ============================================================================

# Verificar que HDFS está corriendo
if ! docker ps --format '{{.Names}}' | grep -q "climaxtreme-namenode"; then
    write_fail "HDFS no está corriendo. Ejecuta primero: ./02_start_infrastructure.sh"
    exit 1
fi

# Verificar dataset
if [ ! -f "$CSV_PATH" ]; then
    write_fail "Dataset no encontrado: $CSV_PATH"
    echo ""
    echo "Descarga el dataset de:"
    echo "  https://www.kaggle.com/berkeleyearth/climate-change-earth-surface-temperature-data"
    exit 1
fi

FILE_SIZE=$(du -h "$CSV_PATH" | cut -f1)
LINE_COUNT=$(wc -l < "$CSV_PATH")
write_success "Dataset encontrado: $FILE_SIZE ($LINE_COUNT líneas)"

# Si no se especificó ninguna opción, mostrar ayuda
if [ "$SAMPLE_SIZE" -eq 0 ] && [ "$FULL_LOAD" = false ] && [ "$GENERATE_SYNTHETIC" = false ]; then
    echo ""
    echo "Debes especificar qué hacer. Opciones:"
    echo ""
    echo "  --sample N     Cargar muestra de N filas (rápido para pruebas)"
    echo "  --full         Cargar dataset completo"
    echo "  --synthetic    Generar datos sintéticos"
    echo "  --all          Todo: dataset completo + sintéticos"
    echo ""
    echo "Ejemplo recomendado para primera vez:"
    echo "  ./03_load_data.sh --all"
    echo ""
    exit 0
fi

# ============================================================================
# Crear estructura en HDFS
# ============================================================================

write_header "1. Preparando HDFS"

echo "Creando directorios en HDFS..."

docker exec climaxtreme-namenode hdfs dfs -mkdir -p /data/climaxtreme/raw
docker exec climaxtreme-namenode hdfs dfs -mkdir -p /data/climaxtreme/processed
docker exec climaxtreme-namenode hdfs dfs -mkdir -p /data/climaxtreme/synthetic
docker exec climaxtreme-namenode hdfs dfs -mkdir -p /data/climaxtreme/streaming

write_success "Estructura HDFS creada"

# ============================================================================
# Cargar dataset
# ============================================================================

if [ "$SAMPLE_SIZE" -gt 0 ] || [ "$FULL_LOAD" = true ]; then
    write_header "2. Cargando Dataset en HDFS"
    
    TEMP_FILE="/tmp/climaxtreme_upload.csv"
    
    if [ "$FULL_LOAD" = true ]; then
        echo "Copiando dataset completo..."
        cp "$CSV_PATH" "$TEMP_FILE"
        HDFS_FILENAME="GlobalLandTemperaturesByCity.csv"
    else
        echo "Extrayendo muestra de $SAMPLE_SIZE filas..."
        head -n 1 "$CSV_PATH" > "$TEMP_FILE"  # Header
        tail -n +2 "$CSV_PATH" | head -n "$SAMPLE_SIZE" >> "$TEMP_FILE"
        HDFS_FILENAME="GlobalLandTemperaturesByCity_sample.csv"
    fi
    
    UPLOAD_SIZE=$(du -h "$TEMP_FILE" | cut -f1)
    echo "Tamaño a subir: $UPLOAD_SIZE"
    
    # Copiar al contenedor
    echo "Copiando a contenedor..."
    docker cp "$TEMP_FILE" climaxtreme-namenode:/tmp/upload.csv
    
    # Subir a HDFS
    echo "Subiendo a HDFS..."
    docker exec climaxtreme-namenode hdfs dfs -put -f /tmp/upload.csv "/data/climaxtreme/$HDFS_FILENAME"
    
    # También copiar al contenedor processor para acceso local
    docker cp "$TEMP_FILE" climaxtreme-processor:/data/GlobalLandTemperaturesByCity.csv
    
    # Limpiar
    rm -f "$TEMP_FILE"
    docker exec climaxtreme-namenode rm -f /tmp/upload.csv
    
    # Verificar
    HDFS_SIZE=$(docker exec climaxtreme-namenode hdfs dfs -du -h "/data/climaxtreme/$HDFS_FILENAME" | awk '{print $1}')
    write_success "Dataset cargado en HDFS: /data/climaxtreme/$HDFS_FILENAME ($HDFS_SIZE)"
fi

# ============================================================================
# Generar datos sintéticos
# ============================================================================

if [ "$GENERATE_SYNTHETIC" = true ]; then
    write_header "3. Generando Datos Sintéticos con Spark"
    
    echo "Esto puede tardar varios minutos..."
    echo ""
    
    # Ejecutar generación de datos sintéticos
    docker exec climaxtreme-processor python -c "
from climaxtreme.preprocessing.spark.synthetic_generator import (
    SyntheticClimateGenerator, 
    SyntheticConfig
)
from climaxtreme.preprocessing.spark.spark_session_manager import get_spark_session
from pyspark.sql import functions as F
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('synthetic_gen')

print('Iniciando generación de datos sintéticos...')

# Crear sesión Spark
spark = get_spark_session('SyntheticDataGeneration')

# Leer datos
print('Leyendo dataset...')
df = spark.read.csv('/data/GlobalLandTemperaturesByCity.csv', header=True, inferSchema=True)
print(f'Registros leídos: {df.count()}')

# Tomar muestra para generación más rápida
df_sample = df.filter(F.col('dt') >= '2010-01-01').limit(50000)
print(f'Registros para generar: {df_sample.count()}')

# Configurar generador
config = SyntheticConfig(seed=42, hourly_interpolation=True)
generator = SyntheticClimateGenerator(spark, config)

# Generar datos
print('Generando datos sintéticos (esto puede tardar)...')
synthetic_df, storm_tracks = generator.generate_full_synthetic_dataset(df_sample, generate_storms=True)

# Guardar
print('Guardando en HDFS...')
synthetic_df.write.mode('overwrite').parquet('hdfs://climaxtreme-namenode:9000/data/climaxtreme/synthetic/synthetic_hourly.parquet')

if storm_tracks is not None and storm_tracks.count() > 0:
    storm_tracks.write.mode('overwrite').parquet('hdfs://climaxtreme-namenode:9000/data/climaxtreme/synthetic/storm_tracks.parquet')
    print(f'Storm tracks guardados: {storm_tracks.count()} registros')

print(f'Datos sintéticos generados: {synthetic_df.count()} registros')
print('¡Completado!')

spark.stop()
"
    
    write_success "Datos sintéticos generados"
fi

# ============================================================================
# Verificar resultados
# ============================================================================

write_header "4. Verificación"

echo "Archivos en HDFS:"
docker exec climaxtreme-namenode hdfs dfs -ls -h /data/climaxtreme/

echo ""
echo "Datos sintéticos:"
docker exec climaxtreme-namenode hdfs dfs -ls -h /data/climaxtreme/synthetic/ 2>/dev/null || echo "  (no generados)"

# ============================================================================
# Resumen
# ============================================================================

write_header "✅ DATOS CARGADOS"

echo -e "El sistema está listo para usar."
echo ""
echo -e "Accede al dashboard en: ${CYAN}http://localhost:8501${NC}"
echo ""
echo "Otros comandos útiles:"
echo ""
echo -e "  ${CYAN}./04_start_streaming.sh${NC}  - Iniciar streaming Kafka en tiempo real"
echo -e "  ${CYAN}./check_status.sh${NC}        - Ver estado del sistema"
echo -e "  ${CYAN}./05_stop_all.sh${NC}         - Detener todos los servicios"
echo ""
