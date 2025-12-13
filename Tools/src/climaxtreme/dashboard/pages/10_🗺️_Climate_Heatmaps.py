"""
🗺️ Climate Heatmaps - Mapas de Calor en Tiempo Real

Visualización global interactiva de variables climáticas usando streaming de Kafka.
Los datos se actualizan automáticamente usando @st.fragment.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Optional, List, Dict

# Configuración de página
st.set_page_config(
    page_title="Climate Heatmaps - climaXtreme",
    page_icon="🗺️",
    layout="wide"
)

# Imports del proyecto
import sys
from pathlib import Path

try:
    from climaxtreme.dashboard.components.kafka_realtime import (
        get_kafka_state,
        init_realtime_stream,
        check_kafka_available,
        render_kafka_status_card,
        TOPICS
    )
    from climaxtreme.dashboard.utils import configure_sidebar
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from climaxtreme.dashboard.components.kafka_realtime import (
        get_kafka_state,
        init_realtime_stream,
        check_kafka_available,
        render_kafka_status_card,
        TOPICS
    )
    from climaxtreme.dashboard.utils import configure_sidebar


# ============================================================================
# Session State
# ============================================================================

def init_session_state():
    """Inicializar estado de sesión."""
    if 'heatmap_variable' not in st.session_state:
        st.session_state.heatmap_variable = 'temperature'
    if 'map_type' not in st.session_state:
        st.session_state.map_type = 'scatter'


# ============================================================================
# Funciones de Visualización
# ============================================================================

def create_global_heatmap(df: pd.DataFrame, variable: str, title: str) -> go.Figure:
    """
    Crear mapa de calor global interactivo.
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="Sin datos disponibles", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False, font=dict(size=20))
        return fig
    
    # Mapeo de variables a columnas
    var_mapping = {
        'temperature': ['temperature', 'temperature_c', 'temperature_hourly'],
        'humidity': ['humidity', 'humidity_pct'],
        'wind_speed': ['wind_speed', 'wind_speed_kmh'],
        'rain': ['rain_mm'],
        'pressure': ['pressure', 'pressure_hpa']
    }
    
    # Encontrar la columna correcta
    col_name = None
    for col in var_mapping.get(variable, [variable]):
        if col in df.columns:
            col_name = col
            break
    
    if col_name is None:
        col_name = variable
    
    if col_name not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text=f"Variable '{variable}' no disponible", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Coordenadas
    lat_col = 'lat_decimal' if 'lat_decimal' in df.columns else 'latitude'
    lon_col = 'lon_decimal' if 'lon_decimal' in df.columns else 'longitude'
    
    # Agregar por ciudad (último valor)
    city_col = 'City' if 'City' in df.columns else 'city'
    country_col = 'Country' if 'Country' in df.columns else 'country'
    
    if city_col in df.columns:
        agg_df = df.groupby([city_col]).agg({
            lat_col: 'last',
            lon_col: 'last',
            col_name: 'mean',
            country_col: 'last' if country_col in df.columns else 'first'
        }).reset_index()
    else:
        agg_df = df.copy()
    
    # Escala de colores según variable
    color_scales = {
        'temperature': ('RdYlBu_r', 'Temperatura (°C)'),
        'humidity': ('Blues', 'Humedad (%)'),
        'wind_speed': ('Viridis', 'Viento (km/h)'),
        'rain': ('Blues', 'Precipitación (mm)'),
        'pressure': ('Plasma', 'Presión (hPa)')
    }
    
    color_scale, color_label = color_scales.get(variable, ('Turbo', variable))
    
    fig = px.scatter_geo(
        agg_df,
        lat=lat_col,
        lon=lon_col,
        color=col_name,
        hover_name=city_col if city_col in agg_df.columns else None,
        hover_data={
            country_col: True if country_col in agg_df.columns else False,
            col_name: ':.1f',
        },
        color_continuous_scale=color_scale,
        title=title,
        projection='natural earth'
    )
    
    fig.update_layout(
        geo=dict(
            showland=True,
            landcolor='rgb(243, 243, 243)',
            countrycolor='rgb(204, 204, 204)',
            showocean=True,
            oceancolor='rgb(230, 245, 255)',
            showlakes=True,
            lakecolor='rgb(200, 230, 255)',
            showcountries=True,
            coastlinecolor='rgb(150, 150, 150)'
        ),
        height=500,
        margin=dict(l=0, r=0, t=50, b=0),
        coloraxis_colorbar=dict(title=color_label)
    )
    
    fig.update_traces(marker=dict(size=12, opacity=0.8))
    
    return fig


def create_distribution_chart(df: pd.DataFrame, variable: str) -> go.Figure:
    """Crear gráfico de distribución de la variable."""
    var_mapping = {
        'temperature': ['temperature', 'temperature_c', 'temperature_hourly'],
        'humidity': ['humidity', 'humidity_pct'],
        'wind_speed': ['wind_speed', 'wind_speed_kmh'],
        'rain': ['rain_mm'],
        'pressure': ['pressure', 'pressure_hpa']
    }
    
    col_name = None
    for col in var_mapping.get(variable, [variable]):
        if col in df.columns:
            col_name = col
            break
    
    if col_name is None or df.empty:
        return go.Figure()
    
    fig = go.Figure(data=[
        go.Histogram(x=df[col_name], nbinsx=30, marker_color='#3498DB')
    ])
    
    fig.update_layout(
        title=f"Distribución de {variable.replace('_', ' ').title()}",
        xaxis_title=variable.replace('_', ' ').title(),
        yaxis_title="Frecuencia",
        height=250,
        margin=dict(l=50, r=20, t=40, b=40)
    )
    
    return fig


def create_zone_comparison(df: pd.DataFrame, variable: str) -> go.Figure:
    """Crear gráfico de comparación por zona climática."""
    var_mapping = {
        'temperature': ['temperature', 'temperature_c', 'temperature_hourly'],
        'humidity': ['humidity', 'humidity_pct'],
        'wind_speed': ['wind_speed', 'wind_speed_kmh'],
        'rain': ['rain_mm'],
        'pressure': ['pressure', 'pressure_hpa']
    }
    
    col_name = None
    for col in var_mapping.get(variable, [variable]):
        if col in df.columns:
            col_name = col
            break
    
    if col_name is None or 'climate_zone' not in df.columns or df.empty:
        return go.Figure()
    
    zone_stats = df.groupby('climate_zone')[col_name].agg(['mean', 'std', 'count']).reset_index()
    zone_stats.columns = ['Zona', 'Media', 'Desv', 'Eventos']
    
    fig = px.bar(
        zone_stats,
        x='Zona',
        y='Media',
        error_y='Desv',
        color='Media',
        color_continuous_scale='RdYlBu_r',
        title=f"{variable.replace('_', ' ').title()} por Zona Climática"
    )
    
    fig.update_layout(height=250, margin=dict(l=50, r=20, t=40, b=40))
    
    return fig


# ============================================================================
# Fragmentos Auto-actualizables
# ============================================================================

@st.fragment(run_every=timedelta(seconds=3))
def live_heatmap_fragment():
    """Mapa de calor actualizándose en tiempo real."""
    kafka_state = get_kafka_state()
    events = kafka_state.get_weather_events(500)
    
    if not events:
        st.info("🔄 Esperando datos de streaming... Asegúrate de que el productor esté activo en Streaming Hub.")
        return
    
    # Convertir a DataFrame
    df = pd.DataFrame(events)
    
    # Obtener variable seleccionada
    variable = st.session_state.get('heatmap_variable', 'temperature')
    
    fig = create_global_heatmap(df, variable, f"🌍 {variable.replace('_', ' ').title()} en Tiempo Real")
    st.plotly_chart(fig, use_container_width=True, key="main_heatmap")
    
    # Estadísticas rápidas
    col1, col2, col3, col4 = st.columns(4)
    
    var_col = None
    for col in ['temperature', 'temperature_c', 'temperature_hourly', 'humidity', 'wind_speed', 'rain_mm']:
        if col in df.columns and variable in col:
            var_col = col
            break
    
    if var_col and var_col in df.columns:
        with col1:
            st.metric("Media", f"{df[var_col].mean():.1f}")
        with col2:
            st.metric("Mínimo", f"{df[var_col].min():.1f}")
        with col3:
            st.metric("Máximo", f"{df[var_col].max():.1f}")
        with col4:
            st.metric("Ciudades", df['city'].nunique() if 'city' in df.columns else len(df))


@st.fragment(run_every=timedelta(seconds=5))
def live_distribution_fragment():
    """Distribución de la variable en tiempo real."""
    kafka_state = get_kafka_state()
    events = kafka_state.get_weather_events(200)
    
    if not events:
        return
    
    df = pd.DataFrame(events)
    variable = st.session_state.get('heatmap_variable', 'temperature')
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_dist = create_distribution_chart(df, variable)
        st.plotly_chart(fig_dist, use_container_width=True, key="distribution_chart")
    
    with col2:
        fig_zone = create_zone_comparison(df, variable)
        st.plotly_chart(fig_zone, use_container_width=True, key="zone_comparison")


@st.fragment(run_every=timedelta(seconds=3))
def live_stats_fragment():
    """Estadísticas del streaming."""
    kafka_state = get_kafka_state()
    stats = kafka_state.get_stats()
    
    if kafka_state.is_running():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Eventos en Buffer", stats.get('weather_buffered', 0))
        with col2:
            st.metric("🌡️ Total Recibidos", stats.get('total_weather', 0))
        with col3:
            last_time = stats.get('last_event_time', 'N/A')
            if last_time != 'N/A':
                last_time = last_time.split('T')[1][:8] if 'T' in str(last_time) else last_time
            st.metric("🕐 Último Evento", last_time)


# ============================================================================
# Sidebar
# ============================================================================

def render_sidebar():
    """Renderizar sidebar con controles."""
    st.sidebar.header("⚙️ Configuración")
    
    # Estado de Kafka
    kafka_check = check_kafka_available()
    kafka_state = get_kafka_state()
    
    if kafka_check.get('available') and kafka_state.is_running():
        st.sidebar.success("🟢 Streaming Activo")
    else:
        st.sidebar.warning("🔴 Streaming Inactivo")
        if st.sidebar.button("▶️ Iniciar Consumer", key="start_consumer_heatmap"):
            kafka_state.start([TOPICS['weather']])
            st.rerun()
    
    st.sidebar.markdown("---")
    
    # Selección de variable
    st.sidebar.subheader("📊 Variable a Visualizar")
    st.session_state.heatmap_variable = st.sidebar.selectbox(
        "Variable",
        options=['temperature', 'humidity', 'wind_speed', 'rain', 'pressure'],
        format_func=lambda x: {
            'temperature': '🌡️ Temperatura',
            'humidity': '💧 Humedad',
            'wind_speed': '💨 Velocidad del Viento',
            'rain': '🌧️ Precipitación',
            'pressure': '📊 Presión'
        }.get(x, x),
        index=['temperature', 'humidity', 'wind_speed', 'rain', 'pressure'].index(
            st.session_state.get('heatmap_variable', 'temperature')
        )
    )
    
    st.sidebar.markdown("---")
    
    # Controles del consumer
    st.sidebar.subheader("🎛️ Control")
    
    if st.sidebar.button("🗑️ Limpiar Buffer", key="clear_buffer_heatmap"):
        kafka_state.clear_buffers()
        st.rerun()
    
    if st.sidebar.button("🔄 Refrescar Ahora", key="refresh_heatmap"):
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Info
    st.sidebar.info("""
    **Fuente de Datos:** Apache Kafka
    
    Los datos se actualizan automáticamente cada 3-5 segundos desde el stream de Kafka.
    
    Inicia el productor en **🌊 Streaming Hub** para ver datos.
    """)


# ============================================================================
# Vista sin streaming (fallback)
# ============================================================================

def render_no_streaming():
    """Mostrar vista cuando no hay streaming activo."""
    st.warning("⚠️ **Streaming no activo**")
    
    st.markdown("""
    ### Para ver los mapas de calor en tiempo real:
    
    1. Ve a **🌊 Streaming Hub**
    2. Verifica que Kafka esté funcionando (indicadores verdes)
    3. Haz clic en **▶️ Iniciar Productor**
    4. Luego haz clic en **▶️ Iniciar Consumer** aquí o en Streaming Hub
    5. Los datos comenzarán a fluir automáticamente
    
    ---
    """)
    
    # Mostrar demo con datos ficticios
    st.subheader("📊 Demo con Datos de Ejemplo")
    
    np.random.seed(42)
    demo_cities = [
        ("Madrid", "Spain", 40.42, -3.70, "TEMPERATE"),
        ("Tokyo", "Japan", 35.68, 139.69, "TEMPERATE"),
        ("Sydney", "Australia", -33.87, 151.21, "SUBTROPICAL"),
        ("New York", "USA", 40.71, -74.01, "CONTINENTAL"),
        ("Dubai", "UAE", 25.20, 55.27, "DESERT"),
        ("Singapore", "Singapore", 1.35, 103.82, "TROPICAL"),
    ]
    
    demo_data = []
    for city, country, lat, lon, zone in demo_cities:
        demo_data.append({
            'city': city, 'country': country,
            'latitude': lat, 'longitude': lon,
            'temperature': np.random.uniform(15, 35),
            'humidity': np.random.uniform(40, 80),
            'climate_zone': zone
        })
    
    demo_df = pd.DataFrame(demo_data)
    
    fig = create_global_heatmap(demo_df, 'temperature', '🌍 Demo: Mapa de Temperatura')
    st.plotly_chart(fig, use_container_width=True, key="demo_heatmap")
    
    st.caption("*Estos son datos de demostración. Inicia el streaming para ver datos reales.*")


# ============================================================================
# Main
# ============================================================================

def main():
    init_session_state()
    configure_sidebar()
    render_sidebar()
    
    # Header
    st.title("🗺️ Mapas de Calor Climáticos")
    st.markdown("""
    Visualización global interactiva de variables climáticas en **tiempo real**.
    Los datos fluyen desde Apache Kafka y se actualizan automáticamente.
    """)
    
    # Verificar estado de Kafka
    kafka_state = get_kafka_state()
    
    if not kafka_state.is_running():
        # Intentar iniciar automáticamente
        kafka_check = check_kafka_available()
        if kafka_check.get('available'):
            kafka_state.start([TOPICS['weather']])
        else:
            render_no_streaming()
            return
    
    # Streaming activo - mostrar fragmentos
    st.markdown("---")
    
    # Estadísticas del streaming
    live_stats_fragment()
    
    st.markdown("---")
    
    # Mapa principal
    live_heatmap_fragment()
    
    st.markdown("---")
    
    # Distribuciones
    st.subheader("📊 Análisis de Distribución")
    live_distribution_fragment()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888;'>
        🗺️ <strong>Climate Heatmaps</strong> | Datos en tiempo real via Apache Kafka
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
