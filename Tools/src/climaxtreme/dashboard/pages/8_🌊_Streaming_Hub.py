"""
🌊 Streaming Hub - Centro de Control de Datos Sintéticos

Esta página es el centro de control para:
- Verificar estado de datos sintéticos en HDFS
- Configurar y lanzar generación de datos
- **Control de Kafka Streaming en Tiempo Real**
- Monitorear progreso en tiempo real
- Validar calidad de datos generados

Arquitectura:
    [Dashboard] → [Docker exec] → [Spark Container] → [HDFS]
         ↑                              ↓
    [Progress Polling] ← ← ← ← [Progress Events]
    
    [Kafka Producer] → [Kafka Topics] → [Dashboard Consumer]
         ↑                                    ↓
    [Spark Streaming]              [Gráficos Actualizándose]
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import subprocess
import json
import time
from datetime import datetime
from typing import Optional, Dict, Any
import sys
from pathlib import Path

# Configuración de página (DEBE ser lo primero)
st.set_page_config(
    page_title="Streaming Hub - climaXtreme",
    page_icon="🌊",
    layout="wide"
)

# Imports del proyecto
try:
    from climaxtreme.dashboard.synthetic_manager import (
        SyntheticDataManager, 
        HDFSStatus, 
        DataStatus,
        render_data_status_card,
        render_dataset_table,
        render_sufficiency_progress
    )
    from climaxtreme.dashboard.components.streaming_config import (
        StreamingConfig,
        StreamingPreset,
        PRESETS,
        render_full_config_ui,
        render_config_summary,
        get_current_config
    )
    from climaxtreme.dashboard.components.kafka_realtime import (
        get_kafka_state,
        check_kafka_available,
        render_kafka_status_card,
        render_realtime_controls,
        TOPICS
    )
    from climaxtreme.dashboard.components.kafka_manager import (
        get_streaming_manager,
        render_kafka_cluster_status,
        render_producer_controls,
        render_full_kafka_control_panel,
        KafkaStreamingConfig
    )
    from climaxtreme.dashboard.utils import configure_sidebar
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from climaxtreme.dashboard.synthetic_manager import (
        SyntheticDataManager, 
        HDFSStatus, 
        DataStatus,
        render_data_status_card,
        render_dataset_table,
        render_sufficiency_progress
    )
    from climaxtreme.dashboard.components.streaming_config import (
        StreamingConfig,
        StreamingPreset,
        PRESETS,
        render_full_config_ui,
        render_config_summary,
        get_current_config
    )
    from climaxtreme.dashboard.components.kafka_realtime import (
        get_kafka_state,
        check_kafka_available,
        render_kafka_status_card,
        render_realtime_controls,
        TOPICS
    )
    from climaxtreme.dashboard.components.kafka_manager import (
        get_streaming_manager,
        render_kafka_cluster_status,
        render_producer_controls,
        render_full_kafka_control_panel,
        KafkaStreamingConfig
    )
    from climaxtreme.dashboard.utils import configure_sidebar


# ============================================================================
# Session State Initialization
# ============================================================================

def init_session_state():
    """Inicializa el estado de la sesión."""
    if 'generation_active' not in st.session_state:
        st.session_state.generation_active = False
    if 'generation_progress' not in st.session_state:
        st.session_state.generation_progress = 0.0
    if 'generation_status' not in st.session_state:
        st.session_state.generation_status = ""
    if 'generation_result' not in st.session_state:
        st.session_state.generation_result = None
    if 'last_hdfs_check' not in st.session_state:
        st.session_state.last_hdfs_check = None
    if 'streaming_config' not in st.session_state:
        st.session_state.streaming_config = StreamingConfig()


# ============================================================================
# Funciones de Generación
# ============================================================================

def build_generation_script(config: StreamingConfig) -> str:
    """
    Construye el script Python para ejecutar en el contenedor Spark.
    
    Args:
        config: Configuración de streaming
        
    Returns:
        Script Python como string
    """
    script = f'''
import json
import sys
from datetime import datetime

# Configurar logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from pyspark.sql import SparkSession
    from climaxtreme.streaming.streaming_producer import (
        StreamingProducer, 
        ProducerConfig,
        ProgressEvent
    )
    
    # Crear SparkSession
    spark = SparkSession.builder \\
        .appName("climaXtreme-StreamingHub-Generation") \\
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \\
        .config("spark.sql.parquet.datetimeRebaseModeInRead", "CORRECTED") \\
        .config("spark.driver.memory", "2g") \\
        .getOrCreate()
    
    logger.info("SparkSession creada correctamente")
    
    # Configuración del producer
    config = ProducerConfig(
        hdfs_base="hdfs://climaxtreme-namenode:9000",
        output_path="/data/climaxtreme/synthetic",
        n_cities={config.n_cities},
        forecast_hours={config.forecast_hours},
        resolution_minutes={config.resolution_minutes},
        batch_size={config.batch_size},
        rain_probability={config.rain_probability},
        storm_probability={config.storm_probability},
        include_storms={str(config.include_storms)},
        include_alerts={str(config.include_alerts)},
        heat_threshold_yellow={config.heat_yellow_threshold},
        heat_threshold_orange={config.heat_orange_threshold},
        heat_threshold_red={config.heat_red_threshold},
        wind_threshold_yellow={config.wind_yellow_threshold},
        wind_threshold_orange={config.wind_orange_threshold},
        wind_threshold_red={config.wind_red_threshold},
        rain_gamma_scale={config.rain_gamma_scale},
        wind_weibull_shape={config.wind_weibull_shape},
        wind_weibull_scale={config.wind_weibull_scale},
        humidity_mean={config.humidity_mean},
        humidity_std={config.humidity_std},
        seed={config.seed}
    )
    
    # Callback para progreso
    def progress_callback(event: ProgressEvent):
        progress_pct = event.records_generated / event.total_records_target * 100 if event.total_records_target > 0 else 0
        print(f"PROGRESS:{progress_pct:.1f}|{{event.records_generated}}|{{event.status}}", flush=True)
    
    # Crear producer
    producer = StreamingProducer(spark, config, progress_callback)
    
    print("PROGRESS:0.0|0|Iniciando generación...", flush=True)
    
    # Generar datos
    result = producer.generate_and_write_to_hdfs(
        historical_path="hdfs://climaxtreme-namenode:9000/data/climaxtreme/processed/monthly.parquet"
    )
    
    # Resultado final
    result_json = json.dumps({{
        'success': result.success,
        'total_records': result.total_records,
        'total_batches': result.total_batches,
        'duration_seconds': result.duration_seconds,
        'output_paths': result.output_paths,
        'error_message': result.error_message
    }})
    
    print(f"RESULT:{{result_json}}", flush=True)
    print("PROGRESS:100.0|{{result.total_records}}|Generación completada", flush=True)
    
    spark.stop()
    
except Exception as e:
    import traceback
    error_msg = str(e)
    print(f"ERROR:{{error_msg}}", flush=True)
    print(f"TRACEBACK:{{traceback.format_exc()}}", flush=True)
    sys.exit(1)
'''
    return script


def run_generation(config: StreamingConfig) -> Dict[str, Any]:
    """
    Ejecuta la generación de datos en el contenedor Spark.
    
    Args:
        config: Configuración de streaming
        
    Returns:
        Diccionario con resultado
    """
    script = build_generation_script(config)
    
    # Ejecutar en Docker
    cmd = [
        "docker", "exec", "climaxtreme-processor",
        "python", "-c", script
    ]
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    result = {
        'success': False,
        'total_records': 0,
        'duration_seconds': 0,
        'error': None,
        'progress_log': []
    }
    
    # Leer output en tiempo real
    for line in process.stdout:
        line = line.strip()
        
        if line.startswith("PROGRESS:"):
            parts = line[9:].split("|")
            if len(parts) >= 3:
                progress = float(parts[0])
                records = int(parts[1])
                status = parts[2]
                
                st.session_state.generation_progress = progress
                st.session_state.generation_status = status
                result['progress_log'].append({
                    'progress': progress,
                    'records': records,
                    'status': status,
                    'timestamp': datetime.now().isoformat()
                })
        
        elif line.startswith("RESULT:"):
            try:
                result_data = json.loads(line[7:])
                result.update(result_data)
            except json.JSONDecodeError:
                pass
        
        elif line.startswith("ERROR:"):
            result['error'] = line[6:]
    
    # Esperar a que termine
    process.wait()
    
    # Capturar stderr
    stderr = process.stderr.read()
    if stderr and not result['success']:
        result['stderr'] = stderr
    
    return result


# ============================================================================
# Visualizaciones
# ============================================================================

def create_progress_chart(progress_log: list) -> go.Figure:
    """Crea un gráfico de progreso de la generación."""
    if not progress_log:
        return go.Figure()
    
    df = pd.DataFrame(progress_log)
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Progreso de Generación", "Registros Generados"),
        row_heights=[0.4, 0.6],
        vertical_spacing=0.15
    )
    
    # Progreso
    fig.add_trace(
        go.Scatter(
            x=list(range(len(df))),
            y=df['progress'],
            mode='lines+markers',
            name='Progreso %',
            line=dict(color='#3498DB', width=2),
            fill='tozeroy',
            fillcolor='rgba(52, 152, 219, 0.2)'
        ),
        row=1, col=1
    )
    
    # Registros
    fig.add_trace(
        go.Bar(
            x=list(range(len(df))),
            y=df['records'],
            name='Registros',
            marker_color='#2ECC71'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        margin=dict(l=60, r=20, t=40, b=40)
    )
    
    fig.update_yaxes(title_text="Progreso (%)", row=1, col=1)
    fig.update_yaxes(title_text="Registros", row=2, col=1)
    fig.update_xaxes(title_text="Batch", row=2, col=1)
    
    return fig


def create_data_distribution_preview(df: pd.DataFrame) -> go.Figure:
    """Crea un preview de distribuciones de los datos."""
    if df is None or df.empty:
        return go.Figure()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Distribución de Temperatura",
            "Distribución de Viento",
            "Ciudades por Zona Climática",
            "Alertas por Nivel"
        )
    )
    
    # Temperatura
    if 'temperature_hourly' in df.columns:
        fig.add_trace(
            go.Histogram(x=df['temperature_hourly'], nbinsx=50, name='Temperatura',
                        marker_color='#E74C3C'),
            row=1, col=1
        )
    
    # Viento
    if 'wind_speed_kmh' in df.columns:
        fig.add_trace(
            go.Histogram(x=df['wind_speed_kmh'], nbinsx=50, name='Viento',
                        marker_color='#3498DB'),
            row=1, col=2
        )
    
    # Zonas climáticas
    if 'climate_zone' in df.columns:
        zone_counts = df['climate_zone'].value_counts()
        fig.add_trace(
            go.Bar(x=zone_counts.index, y=zone_counts.values, name='Zonas',
                  marker_color='#2ECC71'),
            row=2, col=1
        )
    
    # Alertas
    if 'alert_level' in df.columns:
        alert_counts = df['alert_level'].value_counts()
        colors = {'NONE': '#27AE60', 'WATCH': '#F1C40F', 'WARNING': '#E67E22', 'EMERGENCY': '#C0392B'}
        fig.add_trace(
            go.Bar(
                x=alert_counts.index, 
                y=alert_counts.values, 
                name='Alertas',
                marker_color=[colors.get(a, '#95A5A6') for a in alert_counts.index]
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        height=500,
        showlegend=False,
        margin=dict(l=60, r=20, t=40, b=40)
    )
    
    return fig


# ============================================================================
# Página Principal
# ============================================================================

def main():
    """Función principal del Streaming Hub."""
    init_session_state()
    configure_sidebar()
    
    # Header
    st.title("🌊 Streaming Hub")
    st.markdown("""
    **Centro de Control de Datos Sintéticos y Streaming** - Genera, monitorea y transmite datos climáticos
    sintéticos usando Apache Spark sobre HDFS y Apache Kafka para streaming en tiempo real.
    """)
    
    # Tabs principales - CON KAFKA
    tab_kafka, tab1, tab2, tab3, tab4 = st.tabs([
        "🔴 Kafka Streaming",
        "📊 Estado HDFS",
        "⚙️ Configurar Generación",
        "🚀 Generar Datos",
        "📈 Validación EDA"
    ])
    
    # ========================================================================
    # TAB KAFKA: Control de Streaming en Tiempo Real
    # ========================================================================
    with tab_kafka:
        st.header("🔴 Control de Kafka Streaming")
        st.markdown("""
        **Pipeline Spark → Kafka → Dashboard**: Genera datos sintéticos con modelos estadísticos 
        en Spark y transmítelos en tiempo real a través de Kafka.
        """)
        
        # Diagrama de arquitectura
        st.info("""
        ```
        ┌─────────────────────┐    ┌─────────────────┐    ┌─────────────────┐
        │ SyntheticClimate    │    │   Kafka         │    │   Dashboard     │
        │ Generator (Spark)   │───▶│   Broker        │───▶│   (Real-time)   │
        │ - Ciclo diurno      │    │   Topics:       │    │   - Gráficos    │
        │ - Precipitación     │    │   • weather     │    │   - Mapas       │
        │ - Viento            │    │   • alerts      │    │   - Alertas     │
        │ - Eventos extremos  │    │   • storms      │    │                 │
        └─────────────────────┘    └─────────────────┘    └─────────────────┘
        ```
        """)
        
        # Estado del clúster
        st.subheader("📊 Estado del Clúster Kafka")
        render_kafka_cluster_status()
        
        st.markdown("---")
        
        # Estado de conexión del consumer
        st.subheader("📡 Estado del Consumer")
        render_kafka_status_card()
        
        st.markdown("---")
        
        # Controles del productor
        st.subheader("🎛️ Control del Productor Spark-Kafka")
        
        # Checkbox principal para usar Spark
        use_spark_generation = st.checkbox(
            "🚀 **Usar generación con Spark** (recomendado)",
            value=True,
            help="Genera datos usando SyntheticClimateGenerator con modelos estadísticos completos",
            key="use_spark_kafka"
        )
        
        if use_spark_generation:
            st.success("""
            **Modo Spark activo**: Generará datos de alta calidad con:
            - Interpolación horaria con ciclo diurno realista
            - Variables meteorológicas correlacionadas (precipitación, viento, humedad, presión)
            - Eventos extremos detectados por anomalías (olas de calor, frío, tormentas)
            - Tracking de tormentas con categorías Saffir-Simpson
            - Sistema de alertas basado en umbrales
            """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Configuración del Productor:**")
            
            kafka_n_cities = st.slider(
                "Número de ciudades",
                min_value=10, max_value=200, value=50, step=10,
                key="kafka_cities_hub",
                help="Ciudades a incluir en la generación"
            )
            
            kafka_interval = st.slider(
                "Intervalo entre batches (segundos)",
                min_value=0.5, max_value=5.0, value=1.0, step=0.5,
                key="kafka_interval_hub",
                help="Tiempo de espera entre grupos de eventos"
            )
        
        with col2:
            st.markdown("**Acciones:**")
            
            kafka_manager = get_streaming_manager()
            
            if kafka_manager.is_producing():
                st.success("🟢 **Productor activo** - Generando eventos")
                
                if st.button("⏹️ Detener Productor", type="secondary", use_container_width=True, key="stop_producer_hub"):
                    success, msg = kafka_manager.stop_producer()
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)
                    st.rerun()
            else:
                st.info("⚫ **Productor detenido**")
                
                if st.button("▶️ Iniciar Generación Spark→Kafka" if use_spark_generation else "▶️ Iniciar Productor", 
                            type="primary", use_container_width=True, key="start_producer_hub"):
                    config = KafkaStreamingConfig(
                        n_cities=kafka_n_cities,
                        interval_seconds=kafka_interval,
                        include_alerts=True,
                        include_storms=True
                    )
                    with st.spinner("Iniciando pipeline Spark→Kafka..." if use_spark_generation else "Iniciando productor..."):
                        success, msg = kafka_manager.start_producer(config, use_spark=use_spark_generation)
                    if success:
                        st.success(msg)
                        st.balloons()
                    else:
                        st.error(msg)
                    st.rerun()
            
            st.markdown("---")
            
            # Botones de utilidad
            if st.button("📋 Crear Topics de Kafka", use_container_width=True, key="create_topics_hub"):
                with st.spinner("Creando topics..."):
                    success, msg = kafka_manager.create_topics()
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)
        
        st.markdown("---")
        
        # Enlace a Live Dashboard
        st.subheader("🔗 Ver Datos en Tiempo Real")
        st.markdown("""
        Una vez que el productor esté activo, los datos fluirán a todas las páginas de streaming:
        
        - **🔴 [Live Streaming](/Live_Streaming)** - Dashboard principal de tiempo real
        - **🌊 [Streaming Forecast](/Streaming_Forecast)** - Pronósticos en vivo
        - **🌀 [Storm Tracking](/Storm_Tracking)** - Seguimiento de tormentas (pestaña En Vivo)
        - **🚨 [Active Alerts](/Active_Alerts)** - Alertas en tiempo real (pestaña En Vivo)
        """)
        
        # Guía rápida
        with st.expander("📚 Arquitectura y Guía de Inicio"):
            st.markdown("""
            ### Pipeline: Spark → Kafka → Dashboard
            
            **Componentes:**
            
            1. **SyntheticClimateGenerator** (Spark)
               - Lee datos históricos de temperatura
               - Genera variables sintéticas con modelos estadísticos
               - Detecta eventos extremos por anomalías
               - Simula tormentas con tracking
            
            2. **SparkKafkaStreamingProducer**
               - Toma DataFrames de Spark
               - Serializa a JSON
               - Envía a topics de Kafka
            
            3. **Kafka Broker**
               - Topics: weather, alerts, storms, predictions, progress
               - Permite múltiples consumidores
            
            4. **Dashboard Consumer**
               - Lee de Kafka en tiempo real
               - Actualiza gráficos automáticamente
            
            ### Inicio Rápido
            
            **1. Iniciar Kafka:**
            ```bash
            cd infra
            docker-compose up -d zookeeper kafka
            ```
            
            **2. Verificar estado** (indicadores arriba)
            
            **3. Configurar y hacer clic en "▶️ Iniciar Generación"**
            
            **4. Ver datos en páginas de streaming**
            
            ### Scripts Útiles
            ```bash
            # Gestión completa de Kafka
            powershell -ExecutionPolicy Bypass -File scripts/windows/manage_kafka_streaming.ps1
            
            # Ver logs del productor
            docker logs -f climaxtreme-processor
            ```
            """)
    
    # ========================================================================
    # TAB 1: Estado Actual de HDFS
    # ========================================================================
    with tab1:
        st.header("📊 Estado de Datos Sintéticos en HDFS")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            if st.button("🔄 Actualizar Estado", use_container_width=True):
                st.session_state.last_hdfs_check = None
                st.rerun()
        
        # Verificar estado
        manager = SyntheticDataManager()
        
        with st.spinner("Consultando HDFS..."):
            status = manager.get_hdfs_status()
        
        st.session_state.last_hdfs_check = datetime.now()
        
        # Mostrar estado
        render_data_status_card(status)
        
        st.markdown("---")
        
        # Datasets disponibles
        st.subheader("📁 Datasets Disponibles")
        render_dataset_table(status)
        
        st.markdown("---")
        
        # Suficiencia para casos de uso
        st.subheader("✅ Suficiencia por Caso de Uso")
        report = manager.get_sufficiency_report(status)
        render_sufficiency_progress(report)
        
        # Metadatos de última generación
        if status.last_generation:
            st.markdown("---")
            st.subheader("🕐 Última Generación")
            
            meta = status.last_generation
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Fecha", meta.timestamp[:10] if meta.timestamp else "N/A")
            with col2:
                st.metric("Registros", f"{meta.n_records_generated:,}")
            with col3:
                st.metric("Ciudades", meta.n_cities)
            with col4:
                st.metric("Duración", f"{meta.duration_seconds:.1f}s")
    
    # ========================================================================
    # TAB 2: Configuración
    # ========================================================================
    with tab2:
        st.header("⚙️ Configurar Generación de Datos")
        
        config = render_full_config_ui(st.session_state.streaming_config)
        st.session_state.streaming_config = config
        
        st.markdown("---")
        
        # Estimaciones
        st.subheader("📊 Estimaciones")
        
        records_per_city = config.forecast_hours * (60 // config.resolution_minutes)
        total_records = config.n_cities * records_per_city
        estimated_size_mb = total_records * 200 / (1024 * 1024)  # ~200 bytes por registro
        estimated_time_min = total_records / 50000  # ~50K registros por minuto
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Registros", f"{total_records:,}")
        with col2:
            st.metric("Tamaño Estimado", f"{estimated_size_mb:.1f} MB")
        with col3:
            st.metric("Tiempo Estimado", f"{estimated_time_min:.1f} min")
        with col4:
            st.metric("Batches", f"{total_records // config.batch_size + 1}")
        
        # Resumen de configuración
        with st.expander("📋 Resumen de Configuración"):
            render_config_summary(config)
    
    # ========================================================================
    # TAB 3: Generar Datos
    # ========================================================================
    with tab3:
        st.header("🚀 Generar Datos Sintéticos")
        
        config = st.session_state.streaming_config
        
        # Resumen rápido
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"🌍 **{config.n_cities}** ciudades")
        with col2:
            st.info(f"📅 **{config.forecast_hours // 24}** días")
        with col3:
            records = config.n_cities * config.forecast_hours
            st.info(f"📊 **~{records:,}** registros")
        
        st.markdown("---")
        
        # Control de generación
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            start_btn = st.button(
                "🚀 Iniciar Generación",
                type="primary",
                use_container_width=True,
                disabled=st.session_state.generation_active
            )
        
        with col2:
            stop_btn = st.button(
                "⏹️ Detener",
                use_container_width=True,
                disabled=not st.session_state.generation_active
            )
        
        with col3:
            clear_btn = st.button(
                "🗑️ Limpiar",
                use_container_width=True
            )
        
        if clear_btn:
            st.session_state.generation_result = None
            st.session_state.generation_progress = 0
            st.session_state.generation_status = ""
            st.rerun()
        
        st.markdown("---")
        
        # Área de progreso
        progress_container = st.container()
        
        with progress_container:
            if start_btn and not st.session_state.generation_active:
                st.session_state.generation_active = True
                st.session_state.generation_progress = 0
                st.session_state.generation_status = "Preparando..."
                
                # Placeholder para progreso
                progress_placeholder = st.empty()
                status_placeholder = st.empty()
                
                with st.spinner("Ejecutando generación en Spark..."):
                    # Mostrar progreso inicial
                    progress_placeholder.progress(0.0, text="Iniciando...")
                    
                    # Ejecutar generación
                    result = run_generation(config)
                    
                    st.session_state.generation_result = result
                    st.session_state.generation_active = False
                
                if result['success']:
                    st.success(f"""
                    ✅ **Generación Completada**
                    - Registros: {result['total_records']:,}
                    - Batches: {result.get('total_batches', 'N/A')}
                    - Duración: {result['duration_seconds']:.1f} segundos
                    """)
                else:
                    st.error(f"❌ Error en la generación: {result.get('error', 'Error desconocido')}")
                    if 'stderr' in result:
                        with st.expander("Ver logs de error"):
                            st.code(result['stderr'])
            
            # Mostrar resultado anterior si existe
            elif st.session_state.generation_result:
                result = st.session_state.generation_result
                
                if result['success']:
                    st.success(f"✅ Última generación exitosa: {result['total_records']:,} registros")
                    
                    # Gráfico de progreso
                    if result.get('progress_log'):
                        st.plotly_chart(
                            create_progress_chart(result['progress_log']),
                            use_container_width=True
                        )
                else:
                    st.error(f"❌ Última generación fallida: {result.get('error', 'Error desconocido')}")
            
            else:
                st.info("👆 Haz clic en **Iniciar Generación** para comenzar")
                
                # Mostrar qué se va a generar
                st.markdown("""
                ### 📋 Se generará:
                
                | Dataset | Descripción |
                |---------|-------------|
                | `synthetic_hourly.parquet` | Datos horarios de clima |
                | `synthetic_alerts.parquet` | Alertas meteorológicas |
                | `synthetic_storms.parquet` | Seguimiento de tormentas (si habilitado) |
                | `synthetic_events.parquet` | Eventos extremos (si habilitado) |
                """)
    
    # ========================================================================
    # TAB 4: Validación EDA
    # ========================================================================
    with tab4:
        st.header("📈 Validación de Calidad de Datos")
        
        manager = SyntheticDataManager()
        
        # Verificar si hay datos
        status = manager.get_hdfs_status()
        
        if status.status not in [DataStatus.AVAILABLE, DataStatus.INSUFFICIENT]:
            st.warning("⚠️ No hay datos sintéticos disponibles para validar. Genera datos primero.")
        else:
            # Cargar muestra
            with st.spinner("Cargando muestra de datos..."):
                df = manager.load_dataset_sample('synthetic_hourly.parquet', n_rows=10000)
            
            if df is not None and not df.empty:
                st.success(f"✅ Muestra cargada: {len(df):,} registros")
                
                # Distribuciones
                st.subheader("📊 Distribuciones de Variables")
                st.plotly_chart(
                    create_data_distribution_preview(df),
                    use_container_width=True
                )
                
                st.markdown("---")
                
                # Estadísticas descriptivas
                st.subheader("📋 Estadísticas Descriptivas")
                
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if numeric_cols:
                    stats_df = df[numeric_cols].describe().T
                    stats_df['missing'] = df[numeric_cols].isnull().sum()
                    stats_df['missing_%'] = (stats_df['missing'] / len(df) * 100).round(2)
                    
                    st.dataframe(stats_df, use_container_width=True)
                
                st.markdown("---")
                
                # Tests de validación
                st.subheader("✅ Tests de Validación")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Temperatura en rango razonable
                    if 'temperature_hourly' in df.columns:
                        temp_ok = df['temperature_hourly'].between(-60, 60).all()
                        if temp_ok:
                            st.success("🌡️ Temperatura: ✓ En rango [-60, 60]°C")
                        else:
                            st.error("🌡️ Temperatura: ✗ Valores fuera de rango")
                
                with col2:
                    # Humedad en [0, 100]
                    if 'humidity_pct' in df.columns:
                        hum_ok = df['humidity_pct'].between(0, 100).all()
                        if hum_ok:
                            st.success("💧 Humedad: ✓ En rango [0, 100]%")
                        else:
                            st.error("💧 Humedad: ✗ Valores fuera de rango")
                
                with col3:
                    # Alertas distribuidas correctamente
                    if 'alert_level' in df.columns:
                        none_pct = (df['alert_level'] == 'NONE').mean() * 100
                        if none_pct > 70:
                            st.success(f"🚨 Alertas: ✓ {none_pct:.0f}% sin alerta")
                        else:
                            st.warning(f"🚨 Alertas: ⚠️ Solo {none_pct:.0f}% sin alerta")
                
                st.markdown("---")
                
                # Correlaciones
                st.subheader("🔗 Matriz de Correlación")
                
                if len(numeric_cols) > 1:
                    corr_cols = [c for c in ['temperature_hourly', 'humidity_pct', 
                                            'wind_speed_kmh', 'rain_mm', 'pressure_hpa']
                               if c in df.columns]
                    
                    if len(corr_cols) > 1:
                        corr_matrix = df[corr_cols].corr()
                        
                        fig = px.imshow(
                            corr_matrix,
                            text_auto='.2f',
                            color_continuous_scale='RdBu_r',
                            zmin=-1, zmax=1,
                            title="Correlación entre Variables"
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                
                # Vista de datos
                st.markdown("---")
                st.subheader("👁️ Vista de Datos")
                
                with st.expander("Ver muestra de datos"):
                    st.dataframe(df.head(100), use_container_width=True)
            
            else:
                st.error("❌ No se pudieron cargar los datos para validación")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888;'>
        🌊 <strong>Streaming Hub</strong> | climaXtreme Dashboard | 
        Generación de datos con Apache Spark sobre HDFS
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
