"""
📈 Weather TimeSeries - Series Temporales en Tiempo Real

Visualiza series temporales de Kafka Streaming comparadas con
tendencias históricas de HDFS.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import deque
import sys
from pathlib import Path

# Configuración de página
st.set_page_config(
    page_title="Weather TimeSeries - climaXtreme",
    page_icon="📈",
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
# Carga de Datos HDFS
# ============================================================================

@st.cache_data(ttl=3600)
def load_hdfs_yearly() -> Optional[pd.DataFrame]:
    """Cargar tendencia anual desde HDFS."""
    import subprocess
    import json
    
    script = '''
import json
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("LoadYearly").getOrCreate()
try:
    df = spark.read.parquet("hdfs://climaxtreme-namenode:9000/data/climaxtreme/processed/yearly.parquet")
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
            return pd.DataFrame(json.loads(result.stdout[start:end].strip()))
    except Exception:
        pass
    return None


@st.cache_data(ttl=3600)
def load_hdfs_monthly() -> Optional[pd.DataFrame]:
    """Cargar datos mensuales desde HDFS."""
    import subprocess
    import json
    
    script = '''
import json
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("LoadMonthly").getOrCreate()
try:
    df = spark.read.parquet("hdfs://climaxtreme-namenode:9000/data/climaxtreme/processed/monthly.parquet")
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
            return pd.DataFrame(json.loads(result.stdout[start:end].strip()))
    except Exception:
        pass
    return None


@st.cache_data(ttl=3600)
def load_hdfs_climatology() -> Optional[pd.DataFrame]:
    """Cargar climatología desde HDFS."""
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
            return pd.DataFrame(json.loads(result.stdout[start:end].strip()))
    except Exception:
        pass
    return None


# ============================================================================
# Procesamiento de Streaming
# ============================================================================

def aggregate_streaming_by_city(events: List[Dict]) -> pd.DataFrame:
    """Agregar datos de streaming por ciudad."""
    if not events:
        return pd.DataFrame()
    
    city_data = {}
    
    for event in events:
        city = event.get('city', event.get('City', 'Unknown'))
        temp = event.get('temperature', event.get('temperature_c'))
        
        if temp is None:
            continue
        
        if city not in city_data:
            city_data[city] = {
                'temps': [],
                'humidities': [],
                'winds': []
            }
        
        city_data[city]['temps'].append(temp)
        city_data[city]['humidities'].append(event.get('humidity', event.get('humidity_pct', 50)))
        city_data[city]['winds'].append(event.get('wind_speed', event.get('wind_speed_kmh', 0)))
    
    results = []
    for city, data in city_data.items():
        results.append({
            'city': city,
            'avg_temp': np.mean(data['temps']),
            'min_temp': np.min(data['temps']),
            'max_temp': np.max(data['temps']),
            'std_temp': np.std(data['temps']) if len(data['temps']) > 1 else 0,
            'avg_humidity': np.mean(data['humidities']),
            'avg_wind': np.mean(data['winds']),
            'count': len(data['temps'])
        })
    
    return pd.DataFrame(results)


def create_streaming_timeseries(events: List[Dict], window: int = 100) -> pd.DataFrame:
    """Crear serie temporal del streaming."""
    if not events:
        return pd.DataFrame()
    
    data = []
    for i, event in enumerate(events[-window:]):
        temp = event.get('temperature', event.get('temperature_c', np.nan))
        data.append({
            'index': i,
            'temperature': temp,
            'humidity': event.get('humidity', event.get('humidity_pct', np.nan)),
            'wind_speed': event.get('wind_speed', event.get('wind_speed_kmh', np.nan)),
            'city': event.get('city', event.get('City', 'Unknown'))
        })
    
    return pd.DataFrame(data)


# ============================================================================
# Visualizaciones
# ============================================================================

def create_realtime_timeseries(df: pd.DataFrame) -> go.Figure:
    """Crear gráfico de serie temporal en tiempo real."""
    if df.empty:
        return go.Figure()
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                       subplot_titles=('🌡️ Temperatura', '💧 Humedad', '💨 Viento'),
                       vertical_spacing=0.08)
    
    # Temperatura
    fig.add_trace(
        go.Scatter(x=df['index'], y=df['temperature'], mode='lines',
                  name='Temperatura', line=dict(color='#E74C3C', width=2)),
        row=1, col=1
    )
    
    # Humedad
    fig.add_trace(
        go.Scatter(x=df['index'], y=df['humidity'], mode='lines',
                  name='Humedad', line=dict(color='#3498DB', width=2)),
        row=2, col=1
    )
    
    # Viento
    fig.add_trace(
        go.Scatter(x=df['index'], y=df['wind_speed'], mode='lines',
                  name='Viento', line=dict(color='#27AE60', width=2)),
        row=3, col=1
    )
    
    fig.update_layout(height=500, showlegend=True)
    fig.update_yaxes(title_text='°C', row=1, col=1)
    fig.update_yaxes(title_text='%', row=2, col=1)
    fig.update_yaxes(title_text='km/h', row=3, col=1)
    fig.update_xaxes(title_text='Evento #', row=3, col=1)
    
    return fig


def create_city_comparison(df: pd.DataFrame) -> go.Figure:
    """Crear comparación por ciudad."""
    if df.empty:
        return go.Figure()
    
    df_sorted = df.sort_values('avg_temp', ascending=True)
    
    fig = go.Figure()
    
    # Barras de temperatura
    fig.add_trace(go.Bar(
        y=df_sorted['city'],
        x=df_sorted['avg_temp'],
        orientation='h',
        name='Temperatura Promedio',
        marker_color='#E74C3C',
        text=[f'{t:.1f}°C' for t in df_sorted['avg_temp']],
        textposition='outside'
    ))
    
    # Error bars para min/max
    fig.add_trace(go.Scatter(
        y=df_sorted['city'],
        x=df_sorted['avg_temp'],
        error_x=dict(
            type='data',
            symmetric=False,
            array=df_sorted['max_temp'] - df_sorted['avg_temp'],
            arrayminus=df_sorted['avg_temp'] - df_sorted['min_temp'],
            color='rgba(0,0,0,0.3)'
        ),
        mode='markers',
        marker=dict(size=1, color='transparent'),
        showlegend=False
    ))
    
    fig.update_layout(
        title='🌍 Temperatura por Ciudad (Streaming)',
        xaxis_title='Temperatura (°C)',
        height=max(300, len(df_sorted) * 25),
        showlegend=False
    )
    
    return fig


def create_historical_comparison(streaming_avg: float, yearly_df: pd.DataFrame) -> go.Figure:
    """Comparar streaming con histórico anual."""
    if yearly_df is None or yearly_df.empty:
        return go.Figure()
    
    yearly_df = yearly_df.sort_values('year')
    
    fig = go.Figure()
    
    # Tendencia histórica
    fig.add_trace(go.Scatter(
        x=yearly_df['year'],
        y=yearly_df['avg_temperature'],
        mode='lines+markers',
        name='Histórico Anual',
        line=dict(color='#3498DB', width=2),
        marker=dict(size=4)
    ))
    
    # Área de variabilidad
    if 'min_temperature' in yearly_df.columns and 'max_temperature' in yearly_df.columns:
        fig.add_trace(go.Scatter(
            x=list(yearly_df['year']) + list(yearly_df['year'][::-1]),
            y=list(yearly_df['max_temperature']) + list(yearly_df['min_temperature'][::-1]),
            fill='toself',
            fillcolor='rgba(52, 152, 219, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Rango Min-Max'
        ))
    
    # Punto de streaming actual
    current_year = datetime.now().year
    fig.add_trace(go.Scatter(
        x=[current_year],
        y=[streaming_avg],
        mode='markers',
        name=f'Streaming Actual ({streaming_avg:.1f}°C)',
        marker=dict(size=15, color='#E74C3C', symbol='star',
                   line=dict(width=2, color='white'))
    ))
    
    fig.update_layout(
        title='📊 Streaming vs Tendencia Histórica (HDFS)',
        xaxis_title='Año',
        yaxis_title='Temperatura (°C)',
        height=400
    )
    
    return fig


def create_climatology_comparison(streaming_avg: float, climatology: pd.DataFrame) -> go.Figure:
    """Comparar streaming con climatología mensual."""
    if climatology is None or climatology.empty:
        return go.Figure()
    
    current_month = datetime.now().month
    
    fig = go.Figure()
    
    # Climatología mensual
    fig.add_trace(go.Bar(
        x=climatology['month'],
        y=climatology['climatology_mean'],
        name='Climatología Histórica',
        marker_color='#3498DB',
        opacity=0.7
    ))
    
    # Línea de streaming actual
    fig.add_hline(y=streaming_avg, line_dash='dash', line_color='#E74C3C',
                 annotation_text=f'Streaming: {streaming_avg:.1f}°C')
    
    # Destacar mes actual
    fig.add_trace(go.Bar(
        x=[current_month],
        y=[streaming_avg],
        name='Streaming Actual',
        marker_color='#E74C3C'
    ))
    
    fig.update_layout(
        title='📅 Streaming vs Climatología Mensual',
        xaxis_title='Mes',
        yaxis_title='Temperatura (°C)',
        barmode='overlay',
        height=350,
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(1, 13)),
            ticktext=['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
                     'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
        )
    )
    
    return fig


# ============================================================================
# Fragmentos Auto-actualizables
# ============================================================================

@st.fragment(run_every=timedelta(seconds=3))
def live_timeseries_fragment():
    """Serie temporal en tiempo real."""
    kafka_state = get_kafka_state()
    events = kafka_state.get_weather_events(200)
    
    if not events:
        st.warning("⏳ Esperando datos de streaming...")
        return
    
    df = create_streaming_timeseries(events)
    fig = create_realtime_timeseries(df)
    st.plotly_chart(fig, use_container_width=True, key="realtime_ts")


@st.fragment(run_every=timedelta(seconds=5))
def live_metrics_fragment():
    """Métricas en tiempo real."""
    kafka_state = get_kafka_state()
    events = kafka_state.get_weather_events(100)
    
    if not events:
        return
    
    temps = [e.get('temperature', e.get('temperature_c', 0)) for e in events 
             if e.get('temperature') is not None or e.get('temperature_c') is not None]
    humidities = [e.get('humidity', e.get('humidity_pct', 50)) for e in events]
    winds = [e.get('wind_speed', e.get('wind_speed_kmh', 0)) for e in events]
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("🌡️ Temp Actual", f"{temps[-1] if temps else 0:.1f}°C",
                 delta=f"{temps[-1] - temps[0]:.1f}°C" if len(temps) > 1 else None)
    with col2:
        st.metric("🌡️ Temp Media", f"{np.mean(temps):.1f}°C")
    with col3:
        st.metric("💧 Humedad Media", f"{np.mean(humidities):.0f}%")
    with col4:
        st.metric("💨 Viento Medio", f"{np.mean(winds):.0f} km/h")
    with col5:
        st.metric("📊 Eventos", len(events))


@st.fragment(run_every=timedelta(seconds=5))
def live_city_comparison_fragment():
    """Comparación por ciudad."""
    kafka_state = get_kafka_state()
    events = kafka_state.get_weather_events(300)
    
    if not events:
        return
    
    df = aggregate_streaming_by_city(events)
    if not df.empty:
        fig = create_city_comparison(df)
        st.plotly_chart(fig, use_container_width=True, key="city_comparison")


@st.fragment(run_every=timedelta(seconds=10))
def live_historical_comparison_fragment():
    """Comparación con histórico."""
    kafka_state = get_kafka_state()
    events = kafka_state.get_weather_events(200)
    yearly_df = load_hdfs_yearly()
    
    if not events:
        return
    
    temps = [e.get('temperature', e.get('temperature_c', 0)) for e in events 
             if e.get('temperature') is not None]
    streaming_avg = np.mean(temps) if temps else 0
    
    if yearly_df is not None:
        fig = create_historical_comparison(streaming_avg, yearly_df)
        st.plotly_chart(fig, use_container_width=True, key="hist_comparison")


# ============================================================================
# Sidebar
# ============================================================================

def render_sidebar():
    """Renderizar sidebar."""
    st.sidebar.header("⚙️ Series Temporales")
    
    kafka_state = get_kafka_state()
    
    if kafka_state.is_running():
        st.sidebar.success("🟢 Streaming Activo")
        stats = kafka_state.get_stats()
        st.sidebar.metric("Eventos Weather", stats.get('weather_buffered', 0))
    else:
        st.sidebar.warning("🔴 Consumer Inactivo")
        if st.sidebar.button("▶️ Iniciar Consumer"):
            kafka_state.start([TOPICS['weather']])
            st.rerun()
    
    st.sidebar.markdown("---")
    
    st.sidebar.subheader("📚 Datos Históricos")
    st.sidebar.info("""
    **Parquets HDFS:**
    - `yearly.parquet` - Tendencia anual
    - `monthly.parquet` - Datos mensuales
    - `climatology.parquet` - Climatología
    
    Usados para comparación.
    """)
    
    if st.sidebar.button("🗑️ Limpiar Cache"):
        st.cache_data.clear()
        st.rerun()


# ============================================================================
# Main
# ============================================================================

def main():
    configure_sidebar()
    render_sidebar()
    
    st.title("📈 Weather TimeSeries")
    st.markdown("""
    Series temporales en tiempo real desde **Kafka Streaming**, 
    comparadas con tendencias históricas de **HDFS**.
    """)
    
    # Verificar Kafka
    kafka_state = get_kafka_state()
    
    if not kafka_state.is_running():
        kafka_check = check_kafka_available()
        if kafka_check.get('available'):
            kafka_state.start([TOPICS['weather']])
        else:
            st.warning("⚠️ Kafka no disponible. Ve a Streaming Hub para iniciar.")
            return
    
    st.markdown("---")
    
    # Métricas principales
    live_metrics_fragment()
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Serie Temporal",
        "🌍 Por Ciudad",
        "📊 vs Histórico",
        "📅 vs Climatología"
    ])
    
    with tab1:
        st.subheader("📈 Serie Temporal en Tiempo Real")
        live_timeseries_fragment()
    
    with tab2:
        st.subheader("🌍 Comparación por Ciudad")
        live_city_comparison_fragment()
    
    with tab3:
        st.subheader("📊 Streaming vs Tendencia Histórica (HDFS)")
        live_historical_comparison_fragment()
    
    with tab4:
        st.subheader("📅 Streaming vs Climatología Mensual")
        kafka_state = get_kafka_state()
        events = kafka_state.get_weather_events(200)
        climatology = load_hdfs_climatology()
        
        if events and climatology is not None:
            temps = [e.get('temperature', e.get('temperature_c', 0)) for e in events 
                     if e.get('temperature') is not None]
            streaming_avg = np.mean(temps) if temps else 0
            
            fig = create_climatology_comparison(streaming_avg, climatology)
            st.plotly_chart(fig, use_container_width=True, key="climatology_comp")
        else:
            st.warning("Esperando datos...")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888;'>
        📈 <strong>Weather TimeSeries</strong> | Kafka Streaming + HDFS Historical
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
