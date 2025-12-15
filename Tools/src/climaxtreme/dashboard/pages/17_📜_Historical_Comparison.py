"""
📜 Historical Comparison - Comparación con Datos Históricos

Compara los datos en tiempo real de Kafka con los datos históricos
procesados almacenados en HDFS (parquets).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import sys
from pathlib import Path

# Configuración de página
st.set_page_config(
    page_title="Historical Comparison - climaXtreme",
    page_icon="📜",
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
# Funciones de Carga de Datos HDFS
# ============================================================================

def _is_running_in_container() -> bool:
    """Detectar si estamos ejecutando dentro de un contenedor Docker."""
    import os
    if os.path.exists("/.dockerenv"):
        return True
    try:
        with open("/proc/1/cgroup", "r") as f:
            return "docker" in f.read()
    except:
        pass
    return os.environ.get("HOSTNAME", "").startswith("climaxtreme-")


def _load_parquet_direct_spark(parquet_name: str) -> Optional[pd.DataFrame]:
    """Cargar parquet usando Spark directamente (dentro del contenedor)."""
    try:
        from pyspark.sql import SparkSession
        
        spark = SparkSession.builder \
            .appName(f"Load{parquet_name.replace('.', '')}") \
            .config("spark.driver.memory", "512m") \
            .config("spark.ui.enabled", "false") \
            .getOrCreate()
        
        try:
            hdfs_path = f"hdfs://climaxtreme-namenode:9000/data/climaxtreme/processed/{parquet_name}"
            df = spark.read.parquet(hdfs_path)
            
            # Limitar para parquets grandes
            count = df.count()
            if count > 5000:
                df = df.sample(False, 5000/count, seed=42).limit(5000)
            
            result = df.toPandas()
            return result
        finally:
            spark.stop()
            
    except Exception as e:
        st.warning(f"Error cargando {parquet_name} directamente: {e}")
        return None


def _load_parquet_via_subprocess(parquet_name: str) -> Optional[pd.DataFrame]:
    """Cargar parquet usando subprocess (fuera del contenedor)."""
    import subprocess
    import json
    
    script = f'''
import json
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("Load{parquet_name.replace(".", "")}").getOrCreate()
try:
    df = spark.read.parquet("hdfs://climaxtreme-namenode:9000/data/climaxtreme/processed/{parquet_name}")
    count = df.count()
    if count > 5000:
        df = df.sample(False, 5000/count, seed=42).limit(5000)
    result = df.toPandas().to_json(orient='records', date_format='iso')
    print("DATA_START")
    print(result)
    print("DATA_END")
except Exception as ex:
    print(f"ERROR:{{ex}}")
spark.stop()
'''
    
    try:
        cmd = ["docker", "exec", "climaxtreme-processor", "python", "-c", script]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if "DATA_START" in result.stdout:
            start = result.stdout.find("DATA_START") + len("DATA_START")
            end = result.stdout.find("DATA_END")
            data_json = result.stdout[start:end].strip()
            return pd.DataFrame(json.loads(data_json))
        elif "ERROR:" in result.stdout:
            error_msg = result.stdout.split("ERROR:")[1].split("\n")[0]
            st.warning(f"Error en Spark: {error_msg}")
    except Exception as e:
        st.warning(f"Error subprocess {parquet_name}: {e}")
    
    return None


@st.cache_data(ttl=3600)
def load_hdfs_parquet(parquet_name: str) -> Optional[pd.DataFrame]:
    """Cargar parquet desde HDFS. Detecta automáticamente el entorno."""
    if _is_running_in_container():
        return _load_parquet_direct_spark(parquet_name)
    else:
        return _load_parquet_via_subprocess(parquet_name)


@st.cache_data(ttl=3600)
def load_monthly_data() -> Optional[pd.DataFrame]:
    """Cargar datos mensuales."""
    return load_hdfs_parquet("monthly.parquet")


@st.cache_data(ttl=3600)
def load_yearly_data() -> Optional[pd.DataFrame]:
    """Cargar datos anuales."""
    return load_hdfs_parquet("yearly.parquet")


@st.cache_data(ttl=3600)
def load_climatology_data() -> Optional[pd.DataFrame]:
    """Cargar climatología."""
    return load_hdfs_parquet("climatology.parquet")


@st.cache_data(ttl=3600)
def load_country_data() -> Optional[pd.DataFrame]:
    """Cargar datos por país."""
    return load_hdfs_parquet("country.parquet")


@st.cache_data(ttl=3600)
def load_anomalies_sample() -> Optional[pd.DataFrame]:
    """Cargar muestra de anomalías."""
    return load_hdfs_parquet("anomalies.parquet")


# ============================================================================
# Funciones de Comparación
# ============================================================================

def compare_streaming_vs_monthly(streaming_data: List[Dict], monthly_df: pd.DataFrame) -> Dict:
    """Comparar streaming contra promedios mensuales históricos."""
    if monthly_df is None or monthly_df.empty:
        return {}
    
    temps = [e.get('temperature', e.get('temperature_c', 0)) for e in streaming_data 
             if e.get('temperature') is not None or e.get('temperature_c') is not None]
    
    if not temps:
        return {}
    
    current_month = datetime.now().month
    
    # Filtrar mes actual del histórico
    month_data = monthly_df[monthly_df['month'] == current_month]
    
    if month_data.empty:
        return {}
    
    hist_avg = month_data['avg_temperature'].mean()
    hist_min = month_data['min_temperature'].min()
    hist_max = month_data['max_temperature'].max()
    
    streaming_avg = np.mean(temps)
    streaming_min = np.min(temps)
    streaming_max = np.max(temps)
    
    diff = streaming_avg - hist_avg
    
    return {
        'month': current_month,
        'historical': {
            'avg': hist_avg,
            'min': hist_min,
            'max': hist_max
        },
        'streaming': {
            'avg': streaming_avg,
            'min': streaming_min,
            'max': streaming_max,
            'count': len(temps)
        },
        'diff': diff,
        'diff_pct': (diff / abs(hist_avg) * 100) if hist_avg != 0 else 0
    }


def compare_streaming_vs_climatology(streaming_data: List[Dict], climatology: pd.DataFrame) -> Dict:
    """Comparar con climatología mensual."""
    if climatology is None or climatology.empty:
        return {}
    
    temps = [e.get('temperature', e.get('temperature_c', 0)) for e in streaming_data 
             if e.get('temperature') is not None]
    
    if not temps:
        return {}
    
    current_month = datetime.now().month
    month_clima = climatology[climatology['month'] == current_month]
    
    if month_clima.empty:
        return {}
    
    clima_mean = month_clima['climatology_mean'].values[0]
    clima_std = month_clima.get('climatology_std', pd.Series([10])).values[0]
    
    streaming_mean = np.mean(temps)
    
    # Z-score del streaming respecto a la climatología
    z_score = (streaming_mean - clima_mean) / clima_std if clima_std > 0 else 0
    
    # Categoría
    if abs(z_score) < 1:
        category = 'Normal'
    elif abs(z_score) < 2:
        category = 'Ligeramente Anómalo'
    else:
        category = 'Muy Anómalo'
    
    return {
        'climatology_mean': clima_mean,
        'climatology_std': clima_std,
        'streaming_mean': streaming_mean,
        'z_score': z_score,
        'category': category
    }


def detect_streaming_anomalies(streaming_data: List[Dict], historical_stats: Dict) -> Tuple[List[Dict], Dict]:
    """Detectar anomalías en streaming basado en histórico."""
    if not historical_stats:
        historical_stats = {'mean': 16.74, 'std': 10.35}
    
    mean = historical_stats.get('mean', 16.74)
    std = historical_stats.get('std', 10.35)
    
    anomalies = []
    normal = []
    
    for event in streaming_data:
        temp = event.get('temperature', event.get('temperature_c'))
        if temp is None:
            continue
        
        z_score = (temp - mean) / std if std > 0 else 0
        
        event_copy = event.copy()
        event_copy['z_score'] = z_score
        event_copy['is_anomaly'] = abs(z_score) > 2
        
        if abs(z_score) > 2:
            anomalies.append(event_copy)
        else:
            normal.append(event_copy)
    
    total = len(anomalies) + len(normal)
    
    return anomalies, {
        'total': total,
        'anomalies_count': len(anomalies),
        'anomaly_rate': len(anomalies) / total * 100 if total > 0 else 0
    }


# ============================================================================
# Visualizaciones
# ============================================================================

def create_monthly_comparison_chart(streaming_data: List[Dict], monthly_df: pd.DataFrame) -> go.Figure:
    """Gráfico de comparación mensual."""
    if monthly_df is None:
        return go.Figure()
    
    # Agregar histórico por mes
    monthly_avg = monthly_df.groupby('month').agg({
        'avg_temperature': 'mean',
        'min_temperature': 'min',
        'max_temperature': 'max'
    }).reset_index()
    
    fig = go.Figure()
    
    # Área de rango histórico
    fig.add_trace(go.Scatter(
        x=list(monthly_avg['month']) + list(monthly_avg['month'][::-1]),
        y=list(monthly_avg['max_temperature']) + list(monthly_avg['min_temperature'][::-1]),
        fill='toself',
        fillcolor='rgba(52, 152, 219, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Rango Histórico'
    ))
    
    # Línea de promedio histórico
    fig.add_trace(go.Scatter(
        x=monthly_avg['month'],
        y=monthly_avg['avg_temperature'],
        mode='lines+markers',
        name='Promedio Histórico',
        line=dict(color='#3498DB', width=3)
    ))
    
    # Punto streaming actual
    temps = [e.get('temperature', e.get('temperature_c', 0)) for e in streaming_data 
             if e.get('temperature') is not None]
    if temps:
        current_month = datetime.now().month
        streaming_avg = np.mean(temps)
        
        fig.add_trace(go.Scatter(
            x=[current_month],
            y=[streaming_avg],
            mode='markers',
            name=f'Streaming Actual ({streaming_avg:.1f}°C)',
            marker=dict(size=20, color='#E74C3C', symbol='star',
                       line=dict(width=2, color='white'))
        ))
    
    fig.update_layout(
        title='📅 Comparación Mensual: Streaming vs Histórico',
        xaxis_title='Mes',
        yaxis_title='Temperatura (°C)',
        height=400,
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(1, 13)),
            ticktext=['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                     'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
        ),
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    
    return fig


def create_yearly_trend_chart(yearly_df: pd.DataFrame) -> go.Figure:
    """Gráfico de tendencia anual histórica."""
    if yearly_df is None:
        return go.Figure()
    
    yearly_df = yearly_df.sort_values('year')
    
    fig = go.Figure()
    
    # Área de rango
    fig.add_trace(go.Scatter(
        x=list(yearly_df['year']) + list(yearly_df['year'][::-1]),
        y=list(yearly_df['max_temperature']) + list(yearly_df['min_temperature'][::-1]),
        fill='toself',
        fillcolor='rgba(231, 76, 60, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Rango Min-Max'
    ))
    
    # Línea de promedio
    fig.add_trace(go.Scatter(
        x=yearly_df['year'],
        y=yearly_df['avg_temperature'],
        mode='lines',
        name='Promedio Anual',
        line=dict(color='#E74C3C', width=2)
    ))
    
    # Línea de tendencia
    if len(yearly_df) > 5:
        z = np.polyfit(yearly_df['year'], yearly_df['avg_temperature'], 1)
        p = np.poly1d(z)
        
        fig.add_trace(go.Scatter(
            x=yearly_df['year'],
            y=p(yearly_df['year']),
            mode='lines',
            name=f'Tendencia ({z[0]*100:.2f}°C/siglo)',
            line=dict(color='#2C3E50', width=2, dash='dash')
        ))
    
    fig.update_layout(
        title='📈 Tendencia de Temperatura Histórica (HDFS)',
        xaxis_title='Año',
        yaxis_title='Temperatura (°C)',
        height=400
    )
    
    return fig


def create_anomaly_scatter(anomalies: List[Dict], normal: List[Dict]) -> go.Figure:
    """Scatter plot de anomalías detectadas en streaming."""
    fig = go.Figure()
    
    # Datos normales
    if normal:
        normal_temps = [e['temperature'] if 'temperature' in e else e.get('temperature_c', 0) for e in normal]
        normal_times = list(range(len(normal)))
        
        fig.add_trace(go.Scatter(
            x=normal_times,
            y=normal_temps,
            mode='markers',
            name=f'Normal ({len(normal)})',
            marker=dict(color='#3498DB', size=6, opacity=0.5)
        ))
    
    # Anomalías
    if anomalies:
        anom_temps = [e['temperature'] if 'temperature' in e else e.get('temperature_c', 0) for e in anomalies]
        anom_times = list(range(len(normal), len(normal) + len(anomalies)))
        anom_zscores = [e.get('z_score', 0) for e in anomalies]
        
        fig.add_trace(go.Scatter(
            x=anom_times,
            y=anom_temps,
            mode='markers',
            name=f'Anomalías ({len(anomalies)})',
            marker=dict(color='#E74C3C', size=10, symbol='x'),
            text=[f'Z-score: {z:.2f}' for z in anom_zscores],
            hoverinfo='text+y'
        ))
    
    # Líneas de umbral
    fig.add_hline(y=16.74, line_dash='solid', line_color='green',
                 annotation_text='Media Histórica')
    fig.add_hline(y=16.74 + 2*10.35, line_dash='dash', line_color='orange',
                 annotation_text='+2σ')
    fig.add_hline(y=16.74 - 2*10.35, line_dash='dash', line_color='orange',
                 annotation_text='-2σ')
    
    fig.update_layout(
        title='🔍 Detección de Anomalías en Streaming (basado en histórico)',
        xaxis_title='Evento #',
        yaxis_title='Temperatura (°C)',
        height=400
    )
    
    return fig


def create_country_comparison(streaming_data: List[Dict], country_df: pd.DataFrame) -> go.Figure:
    """Comparar streaming por país con histórico."""
    if country_df is None:
        return go.Figure()
    
    # Agrupar streaming por país
    streaming_by_country = {}
    for e in streaming_data:
        country = e.get('country', e.get('Country', 'Unknown'))
        temp = e.get('temperature', e.get('temperature_c'))
        if temp is not None and country:
            if country not in streaming_by_country:
                streaming_by_country[country] = []
            streaming_by_country[country].append(temp)
    
    # Promediar
    streaming_avgs = {c: np.mean(t) for c, t in streaming_by_country.items()}
    
    # Histórico por país (último año disponible)
    latest_year = country_df['year'].max()
    hist_country = country_df[country_df['year'] == latest_year].set_index('country')
    
    # Crear comparación
    comparison_data = []
    for country, streaming_avg in streaming_avgs.items():
        if country in hist_country.index:
            hist_avg = hist_country.loc[country, 'avg_temperature']
            comparison_data.append({
                'País': country,
                'Histórico': hist_avg,
                'Streaming': streaming_avg,
                'Diferencia': streaming_avg - hist_avg
            })
    
    if not comparison_data:
        return go.Figure()
    
    df_comp = pd.DataFrame(comparison_data).sort_values('Diferencia')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df_comp['País'],
        x=df_comp['Histórico'],
        name='Histórico',
        orientation='h',
        marker_color='#3498DB'
    ))
    
    fig.add_trace(go.Bar(
        y=df_comp['País'],
        x=df_comp['Streaming'],
        name='Streaming',
        orientation='h',
        marker_color='#E74C3C'
    ))
    
    fig.update_layout(
        title='🌍 Comparación por País: Streaming vs Histórico',
        xaxis_title='Temperatura (°C)',
        barmode='group',
        height=max(300, len(df_comp) * 30)
    )
    
    return fig


# ============================================================================
# Fragmentos Auto-actualizables
# ============================================================================

@st.fragment(run_every=timedelta(seconds=5))
def live_comparison_metrics_fragment():
    """Métricas de comparación en tiempo real."""
    kafka_state = get_kafka_state()
    events = kafka_state.get_weather_events(300)
    
    monthly_df = load_monthly_data()
    climatology = load_climatology_data()
    
    if not events:
        st.warning("⏳ Esperando datos de streaming...")
        return
    
    # Comparaciones
    monthly_comp = compare_streaming_vs_monthly(events, monthly_df)
    clima_comp = compare_streaming_vs_climatology(events, climatology)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if monthly_comp:
            delta = f"{monthly_comp['diff']:+.2f}°C"
            st.metric(
                "Streaming vs Histórico",
                f"{monthly_comp['streaming']['avg']:.1f}°C",
                delta=delta
            )
        else:
            st.metric("Streaming vs Histórico", "N/A")
    
    with col2:
        if clima_comp:
            st.metric(
                "Z-Score Climático",
                f"{clima_comp['z_score']:+.2f}",
                delta=clima_comp['category']
            )
        else:
            st.metric("Z-Score", "N/A")
    
    with col3:
        st.metric(
            "Eventos Analizados",
            len(events),
            delta="Streaming activo"
        )
    
    with col4:
        # Detectar anomalías
        anomalies, stats = detect_streaming_anomalies(events, None)
        st.metric(
            "Anomalías Detectadas",
            stats['anomalies_count'],
            delta=f"{stats['anomaly_rate']:.1f}%"
        )


@st.fragment(run_every=timedelta(seconds=8))
def live_monthly_comparison_fragment():
    """Comparación mensual en tiempo real."""
    kafka_state = get_kafka_state()
    events = kafka_state.get_weather_events(300)
    monthly_df = load_monthly_data()
    
    if events and monthly_df is not None:
        fig = create_monthly_comparison_chart(events, monthly_df)
        st.plotly_chart(fig, use_container_width=True, key="monthly_comp_chart")


@st.fragment(run_every=timedelta(seconds=8))
def live_anomaly_detection_fragment():
    """Detección de anomalías en tiempo real."""
    kafka_state = get_kafka_state()
    events = kafka_state.get_weather_events(200)
    
    if events:
        anomalies, stats = detect_streaming_anomalies(events, None)
        normal = [e for e in events if not e.get('is_anomaly', False)]
        
        fig = create_anomaly_scatter(anomalies, normal)
        st.plotly_chart(fig, use_container_width=True, key="anomaly_scatter")
        
        if anomalies:
            st.warning(f"⚠️ Se detectaron {len(anomalies)} anomalías en los últimos {len(events)} eventos")


# ============================================================================
# Sidebar
# ============================================================================

def render_sidebar():
    """Renderizar sidebar."""
    st.sidebar.header("⚙️ Comparación Histórica")
    
    kafka_state = get_kafka_state()
    
    if kafka_state.is_running():
        st.sidebar.success("🟢 Consumer Activo")
        stats = kafka_state.get_stats()
        st.sidebar.metric("Weather", stats.get('weather_buffered', 0))
    else:
        st.sidebar.warning("🔴 Consumer Inactivo")
        if st.sidebar.button("▶️ Iniciar Consumer"):
            kafka_state.start([TOPICS['weather']])
            st.rerun()
    
    st.sidebar.markdown("---")
    
    st.sidebar.subheader("📚 Datos HDFS Disponibles")
    st.sidebar.info("""
    **Parquets Procesados:**
    - `monthly.parquet` (3,143 registros)
    - `yearly.parquet` (263 registros)
    - `climatology.parquet` (12 meses)
    - `country.parquet` (31,395 registros)
    - `anomalies.parquet` (8M+ registros)
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
    
    st.title("📜 Comparación Histórica")
    st.markdown("""
    Compara datos de **Kafka Streaming** en tiempo real con datos históricos 
    procesados almacenados en **HDFS** (parquets).
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
    live_comparison_metrics_fragment()
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📅 Comparación Mensual",
        "📈 Tendencia Histórica",
        "🔍 Anomalías",
        "🌍 Por País"
    ])
    
    with tab1:
        st.subheader("📅 Streaming vs Promedios Mensuales Históricos")
        live_monthly_comparison_fragment()
    
    with tab2:
        st.subheader("📈 Tendencia de Temperatura Histórica (Solo HDFS)")
        yearly_df = load_yearly_data()
        if yearly_df is not None:
            fig = create_yearly_trend_chart(yearly_df)
            st.plotly_chart(fig, use_container_width=True, key="yearly_trend")
            
            # Estadísticas
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Años de Datos", len(yearly_df))
            with col2:
                st.metric("Rango Temporal", 
                         f"{int(yearly_df['year'].min())}-{int(yearly_df['year'].max())}")
            with col3:
                # Calcular tendencia
                z = np.polyfit(yearly_df['year'], yearly_df['avg_temperature'], 1)
                st.metric("Tendencia", f"{z[0]*100:.2f}°C/siglo")
        else:
            st.warning("No se pudieron cargar los datos anuales de HDFS")
    
    with tab3:
        st.subheader("🔍 Detección de Anomalías en Streaming")
        st.info("Anomalías detectadas cuando |Z-score| > 2 respecto a estadísticas históricas")
        live_anomaly_detection_fragment()
    
    with tab4:
        st.subheader("🌍 Comparación por País")
        kafka_state = get_kafka_state()
        events = kafka_state.get_weather_events(500)
        country_df = load_country_data()
        
        if events and country_df is not None:
            fig = create_country_comparison(events, country_df)
            st.plotly_chart(fig, use_container_width=True, key="country_comp")
        else:
            st.warning("Esperando datos para comparación por país...")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888;'>
        📜 <strong>Historical Comparison</strong> | Kafka Streaming vs HDFS Parquets
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
