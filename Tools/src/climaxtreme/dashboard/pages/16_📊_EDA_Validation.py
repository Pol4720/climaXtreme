"""
📊 EDA Validation - Validación de Datos Streaming

Valida la calidad de los datos generados via Kafka comparándolos
con las distribuciones esperadas basadas en datos históricos de HDFS.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from scipy import stats
import sys
from pathlib import Path

# Configuración de página
st.set_page_config(
    page_title="EDA Validation - climaXtreme",
    page_icon="📊",
    layout="wide"
)

# Imports del proyecto
try:
    from climaxtreme.dashboard.components.kafka_realtime import (
        get_kafka_state,
        check_kafka_available,
        TOPICS
    )
    from climaxtreme.dashboard.utils import configure_sidebar
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from climaxtreme.dashboard.components.kafka_realtime import (
        get_kafka_state,
        check_kafka_available,
        TOPICS
    )
    from climaxtreme.dashboard.utils import configure_sidebar


# ============================================================================
# Funciones de Carga de Datos Históricos (HDFS)
# ============================================================================

@st.cache_data(ttl=3600)  # Cache por 1 hora
def load_hdfs_climatology() -> Optional[pd.DataFrame]:
    """Cargar climatología histórica desde HDFS via Spark."""
    import subprocess
    import json
    
    script = '''
import json
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("LoadClimatology").getOrCreate()
try:
    df = spark.read.parquet("hdfs://climaxtreme-namenode:9000/data/climaxtreme/processed/climatology.parquet")
    result = df.toPandas().to_json(orient='records')
    print("DATA_START")
    print(result)
    print("DATA_END")
except Exception as e:
    print(f"ERROR:{e}")
spark.stop()
'''
    
    try:
        cmd = ["docker", "exec", "climaxtreme-processor", "python", "-c", script]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if "DATA_START" in result.stdout:
            start = result.stdout.find("DATA_START") + len("DATA_START")
            end = result.stdout.find("DATA_END")
            data_json = result.stdout[start:end].strip()
            return pd.DataFrame(json.loads(data_json))
    except Exception as e:
        st.error(f"Error cargando climatología: {e}")
    
    return None


@st.cache_data(ttl=3600)
def load_hdfs_descriptive_stats() -> Optional[pd.DataFrame]:
    """Cargar estadísticas descriptivas desde HDFS."""
    import subprocess
    import json
    
    script = '''
import json
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("LoadStats").getOrCreate()
try:
    df = spark.read.parquet("hdfs://climaxtreme-namenode:9000/data/climaxtreme/processed/descriptive_stats.parquet")
    result = df.toPandas().to_json(orient='records')
    print("DATA_START")
    print(result)
    print("DATA_END")
except Exception as e:
    print(f"ERROR:{e}")
spark.stop()
'''
    
    try:
        cmd = ["docker", "exec", "climaxtreme-processor", "python", "-c", script]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if "DATA_START" in result.stdout:
            start = result.stdout.find("DATA_START") + len("DATA_START")
            end = result.stdout.find("DATA_END")
            data_json = result.stdout[start:end].strip()
            return pd.DataFrame(json.loads(data_json))
    except Exception as e:
        pass
    
    return None


# ============================================================================
# Funciones de Validación
# ============================================================================

def validate_temperature_distribution(streaming_temps: List[float], historical_stats: Dict) -> Dict:
    """Validar distribución de temperatura contra histórico."""
    if not streaming_temps or not historical_stats:
        return {'passed': False, 'reason': 'Sin datos suficientes'}
    
    streaming_mean = np.mean(streaming_temps)
    streaming_std = np.std(streaming_temps)
    
    hist_mean = historical_stats.get('mean', 17)
    hist_std = historical_stats.get('std', 10)
    hist_min = historical_stats.get('min', -43)
    hist_max = historical_stats.get('max', 40)
    
    # Validaciones
    validations = []
    
    # 1. Rango válido
    temps_in_range = [t for t in streaming_temps if hist_min <= t <= hist_max]
    range_ratio = len(temps_in_range) / len(streaming_temps)
    validations.append({
        'test': 'Rango Válido',
        'passed': range_ratio >= 0.95,
        'value': f"{range_ratio*100:.1f}%",
        'expected': f"≥95% en [{hist_min:.1f}, {hist_max:.1f}]°C"
    })
    
    # 2. Media razonable (dentro de 2 std del histórico)
    mean_diff = abs(streaming_mean - hist_mean)
    mean_ok = mean_diff <= 2 * hist_std
    validations.append({
        'test': 'Media Razonable',
        'passed': mean_ok,
        'value': f"{streaming_mean:.2f}°C",
        'expected': f"{hist_mean:.2f}°C ± {2*hist_std:.1f}"
    })
    
    # 3. Variabilidad suficiente
    std_ok = streaming_std >= hist_std * 0.3
    validations.append({
        'test': 'Variabilidad Suficiente',
        'passed': std_ok,
        'value': f"σ={streaming_std:.2f}",
        'expected': f"σ≥{hist_std*0.3:.2f}"
    })
    
    # 4. Sin valores extremos imposibles
    extreme_count = len([t for t in streaming_temps if t < -60 or t > 60])
    validations.append({
        'test': 'Sin Extremos Imposibles',
        'passed': extreme_count == 0,
        'value': f"{extreme_count} valores",
        'expected': "0 valores fuera de [-60, 60]°C"
    })
    
    passed = sum(1 for v in validations if v['passed'])
    
    return {
        'validations': validations,
        'passed': passed,
        'total': len(validations),
        'score': passed / len(validations) * 100,
        'streaming_stats': {
            'mean': streaming_mean,
            'std': streaming_std,
            'min': min(streaming_temps),
            'max': max(streaming_temps),
            'count': len(streaming_temps)
        }
    }


def validate_humidity(streaming_data: List[Dict]) -> Dict:
    """Validar rango de humedad."""
    humidities = [e.get('humidity', e.get('humidity_pct', 0)) for e in streaming_data 
                  if e.get('humidity') is not None or e.get('humidity_pct') is not None]
    
    if not humidities:
        return {'passed': False, 'validations': []}
    
    validations = []
    
    # Humedad en rango 0-100
    in_range = len([h for h in humidities if 0 <= h <= 100])
    ratio = in_range / len(humidities)
    validations.append({
        'test': 'Rango 0-100%',
        'passed': ratio >= 0.99,
        'value': f"{ratio*100:.2f}%",
        'expected': '≥99% en rango'
    })
    
    # Variabilidad
    if len(humidities) > 10:
        std = np.std(humidities)
        validations.append({
            'test': 'Variabilidad',
            'passed': std >= 5,
            'value': f"σ={std:.2f}",
            'expected': 'σ≥5'
        })
    
    passed = sum(1 for v in validations if v['passed'])
    return {
        'validations': validations,
        'passed': passed,
        'total': len(validations),
        'score': passed / len(validations) * 100 if validations else 0
    }


def validate_geographic_coverage(streaming_data: List[Dict]) -> Dict:
    """Validar cobertura geográfica."""
    cities = set(e.get('city', e.get('City', '')) for e in streaming_data if e.get('city') or e.get('City'))
    zones = set(e.get('climate_zone', '') for e in streaming_data if e.get('climate_zone'))
    
    validations = []
    
    # Múltiples ciudades
    validations.append({
        'test': 'Múltiples Ciudades',
        'passed': len(cities) >= 10,
        'value': f"{len(cities)} ciudades",
        'expected': '≥10 ciudades'
    })
    
    # Múltiples zonas climáticas
    validations.append({
        'test': 'Zonas Climáticas',
        'passed': len(zones) >= 3,
        'value': f"{len(zones)} zonas",
        'expected': '≥3 zonas'
    })
    
    passed = sum(1 for v in validations if v['passed'])
    return {
        'validations': validations,
        'passed': passed,
        'total': len(validations),
        'score': passed / len(validations) * 100 if validations else 0
    }


# ============================================================================
# Visualizaciones
# ============================================================================

def create_distribution_comparison(streaming_temps: List[float], historical_stats: Dict) -> go.Figure:
    """Crear comparación de distribuciones."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        'Distribución Streaming', 'Comparación con Histórico'
    ))
    
    # Histograma streaming
    fig.add_trace(
        go.Histogram(x=streaming_temps, nbinsx=30, name='Streaming',
                    marker_color='#3498DB', opacity=0.7),
        row=1, col=1
    )
    
    # Líneas de referencia histórica
    hist_mean = historical_stats.get('mean', 17)
    hist_std = historical_stats.get('std', 10)
    
    # Box plot comparativo
    fig.add_trace(
        go.Box(y=streaming_temps, name='Streaming', marker_color='#3498DB'),
        row=1, col=2
    )
    
    # Agregar línea de media histórica
    fig.add_hline(y=hist_mean, line_dash='dash', line_color='red',
                 annotation_text=f'Media Histórica: {hist_mean:.1f}°C',
                 row=1, col=2)
    
    fig.update_layout(height=350, showlegend=True)
    
    return fig


def create_validation_gauge(score: float, title: str) -> go.Figure:
    """Crear gauge de validación."""
    color = '#27AE60' if score >= 80 else '#F39C12' if score >= 60 else '#E74C3C'
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': title},
        number={'suffix': '%'},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 60], 'color': '#FADBD8'},
                {'range': [60, 80], 'color': '#FCF3CF'},
                {'range': [80, 100], 'color': '#D5F5E3'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 2},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def create_streaming_vs_climatology(streaming_data: List[Dict], climatology: pd.DataFrame) -> go.Figure:
    """Comparar streaming con climatología mensual."""
    if climatology is None or climatology.empty:
        return go.Figure()
    
    # Agrupar streaming por mes simulado (usando hora actual)
    now = datetime.now()
    current_month = now.month
    
    # Obtener temperaturas streaming
    temps = [e.get('temperature', e.get('temperature_c', 0)) for e in streaming_data 
             if e.get('temperature') or e.get('temperature_c')]
    
    if not temps:
        return go.Figure()
    
    streaming_mean = np.mean(temps)
    
    fig = go.Figure()
    
    # Climatología mensual
    fig.add_trace(go.Scatter(
        x=climatology['month'],
        y=climatology['climatology_mean'],
        mode='lines+markers',
        name='Climatología Histórica',
        line=dict(color='#E74C3C', width=2),
        marker=dict(size=8)
    ))
    
    # Banda de variabilidad
    if 'climatology_std' in climatology.columns:
        upper = climatology['climatology_mean'] + climatology['climatology_std']
        lower = climatology['climatology_mean'] - climatology['climatology_std']
        
        fig.add_trace(go.Scatter(
            x=list(climatology['month']) + list(climatology['month'][::-1]),
            y=list(upper) + list(lower[::-1]),
            fill='toself',
            fillcolor='rgba(231, 76, 60, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='±1 Std Dev'
        ))
    
    # Punto actual del streaming
    fig.add_trace(go.Scatter(
        x=[current_month],
        y=[streaming_mean],
        mode='markers',
        name=f'Streaming Actual ({streaming_mean:.1f}°C)',
        marker=dict(size=15, color='#3498DB', symbol='star')
    ))
    
    fig.update_layout(
        title='📊 Streaming vs Climatología Histórica',
        xaxis_title='Mes',
        yaxis_title='Temperatura (°C)',
        height=400,
        xaxis=dict(tickmode='array', tickvals=list(range(1, 13)),
                  ticktext=['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                           'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'])
    )
    
    return fig


# ============================================================================
# Fragmentos Auto-actualizables
# ============================================================================

@st.fragment(run_every=timedelta(seconds=5))
def live_validation_fragment():
    """Validación en tiempo real."""
    kafka_state = get_kafka_state()
    events = kafka_state.get_weather_events(500)
    
    if not events:
        st.warning("⏳ Esperando datos de streaming...")
        return
    
    # Cargar estadísticas históricas
    historical_stats = {
        'mean': 16.74,  # De descriptive_stats
        'std': 10.35,
        'min': -42.7,
        'max': 39.65
    }
    
    # Extraer temperaturas
    temps = [e.get('temperature', e.get('temperature_c', 0)) for e in events 
             if e.get('temperature') is not None or e.get('temperature_c') is not None]
    
    # Validaciones
    temp_validation = validate_temperature_distribution(temps, historical_stats)
    humidity_validation = validate_humidity(events)
    geo_validation = validate_geographic_coverage(events)
    
    # Score total
    total_passed = temp_validation['passed'] + humidity_validation['passed'] + geo_validation['passed']
    total_tests = temp_validation['total'] + humidity_validation['total'] + geo_validation['total']
    overall_score = total_passed / total_tests * 100 if total_tests > 0 else 0
    
    # Mostrar gauges
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        fig = create_validation_gauge(overall_score, "Score Total")
        st.plotly_chart(fig, use_container_width=True, key="gauge_total")
    with col2:
        fig = create_validation_gauge(temp_validation['score'], "Temperatura")
        st.plotly_chart(fig, use_container_width=True, key="gauge_temp")
    with col3:
        fig = create_validation_gauge(humidity_validation['score'], "Humedad")
        st.plotly_chart(fig, use_container_width=True, key="gauge_humidity")
    with col4:
        fig = create_validation_gauge(geo_validation['score'], "Cobertura")
        st.plotly_chart(fig, use_container_width=True, key="gauge_geo")
    
    # Tabla de validaciones
    st.subheader("📋 Detalle de Validaciones")
    
    all_validations = (
        [{'Categoría': '🌡️ Temperatura', **v} for v in temp_validation['validations']] +
        [{'Categoría': '💧 Humedad', **v} for v in humidity_validation['validations']] +
        [{'Categoría': '🌍 Cobertura', **v} for v in geo_validation['validations']]
    )
    
    df_validations = pd.DataFrame(all_validations)
    df_validations['Estado'] = df_validations['passed'].apply(lambda x: '✅' if x else '❌')
    
    st.dataframe(
        df_validations[['Categoría', 'test', 'value', 'expected', 'Estado']].rename(columns={
            'test': 'Test', 'value': 'Valor', 'expected': 'Esperado'
        }),
        use_container_width=True,
        hide_index=True
    )


@st.fragment(run_every=timedelta(seconds=5))
def live_distribution_fragment():
    """Distribución en tiempo real vs histórico."""
    kafka_state = get_kafka_state()
    events = kafka_state.get_weather_events(500)
    
    if not events:
        return
    
    temps = [e.get('temperature', e.get('temperature_c', 0)) for e in events 
             if e.get('temperature') is not None or e.get('temperature_c') is not None]
    
    if not temps:
        return
    
    historical_stats = {'mean': 16.74, 'std': 10.35, 'min': -42.7, 'max': 39.65}
    
    fig = create_distribution_comparison(temps, historical_stats)
    st.plotly_chart(fig, use_container_width=True, key="dist_comparison")


@st.fragment(run_every=timedelta(seconds=8))
def live_climatology_comparison_fragment():
    """Comparación con climatología."""
    kafka_state = get_kafka_state()
    events = kafka_state.get_weather_events(300)
    
    climatology = load_hdfs_climatology()
    
    if events and climatology is not None:
        fig = create_streaming_vs_climatology(events, climatology)
        st.plotly_chart(fig, use_container_width=True, key="climatology_comp")


# ============================================================================
# Sidebar
# ============================================================================

def render_sidebar():
    """Renderizar sidebar."""
    st.sidebar.header("⚙️ Validación EDA")
    
    kafka_state = get_kafka_state()
    
    if kafka_state.is_running():
        st.sidebar.success("🟢 Consumer Activo")
        stats = kafka_state.get_stats()
        st.sidebar.metric("Eventos en Buffer", stats.get('weather_buffered', 0))
    else:
        st.sidebar.warning("🔴 Consumer Inactivo")
        if st.sidebar.button("▶️ Iniciar Consumer"):
            kafka_state.start([TOPICS['weather'], TOPICS['alerts']])
            st.rerun()
    
    st.sidebar.markdown("---")
    
    # Info sobre datos históricos
    st.sidebar.subheader("📚 Datos de Referencia")
    st.sidebar.info("""
    **Fuente HDFS:**
    - `climatology.parquet`
    - `descriptive_stats.parquet`
    
    Datos históricos procesados con Spark.
    """)
    
    if st.sidebar.button("🔄 Refrescar"):
        st.cache_data.clear()
        st.rerun()


# ============================================================================
# Main
# ============================================================================

def main():
    configure_sidebar()
    render_sidebar()
    
    st.title("📊 Validación EDA - Datos de Streaming")
    st.markdown("""
    Validación en tiempo real de los datos generados via **Kafka** comparándolos 
    con distribuciones históricas almacenadas en **HDFS**.
    """)
    
    # Verificar Kafka
    kafka_state = get_kafka_state()
    
    if not kafka_state.is_running():
        kafka_check = check_kafka_available()
        if kafka_check.get('available'):
            kafka_state.start([TOPICS['weather'], TOPICS['alerts']])
        else:
            st.warning("⚠️ Kafka no disponible. Ve a Streaming Hub para iniciar.")
            return
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "🧪 Validaciones en Vivo",
        "📈 Distribuciones",
        "📊 vs Climatología"
    ])
    
    with tab1:
        live_validation_fragment()
    
    with tab2:
        st.subheader("📈 Comparación de Distribuciones")
        live_distribution_fragment()
    
    with tab3:
        st.subheader("📊 Streaming vs Climatología Histórica (HDFS)")
        live_climatology_comparison_fragment()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888;'>
        📊 <strong>EDA Validation</strong> | Streaming Kafka vs Histórico HDFS
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
