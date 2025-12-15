"""
🌊 Streaming Forecast - Pronóstico en Tiempo Real

Visualización de pronósticos meteorológicos en tiempo real usando Apache Kafka.
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
from collections import defaultdict
import sys
from pathlib import Path

# Configuración de página
st.set_page_config(
    page_title="Streaming Forecast - climaXtreme",
    page_icon="🌊",
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
# Session State
# ============================================================================

def init_session_state():
    """Inicializar estado de sesión."""
    if 'forecast_city' not in st.session_state:
        st.session_state.forecast_city = 'ALL'
    if 'forecast_variable' not in st.session_state:
        st.session_state.forecast_variable = 'temperature'


# ============================================================================
# Funciones de Visualización
# ============================================================================

def create_multi_city_forecast(events: List[Dict], variable: str = 'temperature') -> go.Figure:
    """Crear gráfico de pronóstico multi-ciudad."""
    if not events:
        fig = go.Figure()
        fig.add_annotation(text="Sin datos", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Mapeo de variables
    var_cols = {
        'temperature': ['temperature', 'temperature_c', 'temperature_hourly'],
        'humidity': ['humidity', 'humidity_pct'],
        'wind_speed': ['wind_speed', 'wind_speed_kmh'],
        'rain': ['rain_mm'],
        'pressure': ['pressure', 'pressure_hpa']
    }
    
    # Agrupar eventos por ciudad
    city_data = defaultdict(list)
    for e in events:
        city = e.get('city', e.get('City', 'Unknown'))
        
        # Encontrar valor de la variable
        value = None
        for col in var_cols.get(variable, [variable]):
            if col in e:
                value = e[col]
                break
        
        if value is not None:
            city_data[city].append({
                'timestamp': e.get('timestamp', ''),
                'value': value
            })
    
    if not city_data:
        return go.Figure()
    
    fig = go.Figure()
    
    # Colores para ciudades
    colors = px.colors.qualitative.Set2
    
    for i, (city, data) in enumerate(list(city_data.items())[:10]):  # Max 10 ciudades
        values = [d['value'] for d in data]
        x_vals = list(range(len(values)))
        
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=values,
            mode='lines+markers',
            name=city,
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=4)
        ))
    
    var_labels = {
        'temperature': 'Temperatura (°C)',
        'humidity': 'Humedad (%)',
        'wind_speed': 'Viento (km/h)',
        'rain': 'Precipitación (mm)',
        'pressure': 'Presión (hPa)'
    }
    
    fig.update_layout(
        title=f"📈 {var_labels.get(variable, variable)} por Ciudad",
        xaxis_title="Eventos recientes",
        yaxis_title=var_labels.get(variable, variable),
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=80, b=40)
    )
    
    return fig


def create_city_detail_chart(events: List[Dict], city: str) -> go.Figure:
    """Crear gráfico detallado para una ciudad."""
    city_events = [e for e in events if e.get('city', e.get('City', '')) == city]
    
    if not city_events:
        fig = go.Figure()
        fig.add_annotation(text=f"Sin datos para {city}", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Extraer valores
    temps = [e.get('temperature', e.get('temperature_c', 0)) for e in city_events]
    humidities = [e.get('humidity', e.get('humidity_pct', 0)) for e in city_events]
    winds = [e.get('wind_speed', e.get('wind_speed_kmh', 0)) for e in city_events]
    
    x = list(range(len(temps)))
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('🌡️ Temperatura', '💧 Humedad', '💨 Viento'),
        vertical_spacing=0.1,
        shared_xaxes=True
    )
    
    # Temperatura
    fig.add_trace(
        go.Scatter(x=x, y=temps, mode='lines+markers', name='Temperatura',
                  line=dict(color='#E74C3C', width=2), marker=dict(size=4)),
        row=1, col=1
    )
    
    # Humedad
    fig.add_trace(
        go.Scatter(x=x, y=humidities, mode='lines+markers', name='Humedad',
                  line=dict(color='#3498DB', width=2), marker=dict(size=4)),
        row=2, col=1
    )
    
    # Viento
    fig.add_trace(
        go.Scatter(x=x, y=winds, mode='lines+markers', name='Viento',
                  line=dict(color='#27AE60', width=2), marker=dict(size=4)),
        row=3, col=1
    )
    
    fig.update_layout(
        title=f"📊 Pronóstico Detallado: {city}",
        height=450,
        showlegend=False,
        margin=dict(l=50, r=20, t=60, b=40)
    )
    
    fig.update_yaxes(title_text="°C", row=1, col=1)
    fig.update_yaxes(title_text="%", row=2, col=1)
    fig.update_yaxes(title_text="km/h", row=3, col=1)
    fig.update_xaxes(title_text="Eventos", row=3, col=1)
    
    return fig


def create_forecast_map(events: List[Dict]) -> go.Figure:
    """Crear mapa de pronóstico."""
    if not events:
        return go.Figure()
    
    # Agrupar por ciudad (último valor)
    city_latest = {}
    for e in events:
        city = e.get('city', e.get('City', 'Unknown'))
        city_latest[city] = e
    
    lats = [e.get('latitude', e.get('lat_decimal', 0)) for e in city_latest.values()]
    lons = [e.get('longitude', e.get('lon_decimal', 0)) for e in city_latest.values()]
    temps = [e.get('temperature', e.get('temperature_c', 20)) for e in city_latest.values()]
    cities = list(city_latest.keys())
    
    fig = px.scatter_geo(
        lat=lats,
        lon=lons,
        color=temps,
        hover_name=cities,
        color_continuous_scale='RdYlBu_r',
        title='🗺️ Mapa de Pronóstico',
        projection='natural earth'
    )
    
    fig.update_layout(
        height=350,
        margin=dict(l=0, r=0, t=50, b=0),
        geo=dict(
            showland=True,
            landcolor='rgb(243, 243, 243)',
            showocean=True,
            oceancolor='rgb(230, 245, 255)',
            showcountries=True
        ),
        coloraxis_colorbar=dict(title="°C")
    )
    
    fig.update_traces(marker=dict(size=12, opacity=0.8))
    
    return fig


def create_zone_summary(events: List[Dict]) -> go.Figure:
    """Crear resumen por zona climática."""
    if not events:
        return go.Figure()
    
    # Agrupar por zona
    zone_data = defaultdict(list)
    for e in events:
        zone = e.get('climate_zone', 'Unknown')
        temp = e.get('temperature', e.get('temperature_c'))
        if temp is not None:
            zone_data[zone].append(temp)
    
    if not zone_data:
        return go.Figure()
    
    zones = list(zone_data.keys())
    means = [np.mean(zone_data[z]) for z in zones]
    stds = [np.std(zone_data[z]) for z in zones]
    
    fig = go.Figure(data=[
        go.Bar(
            x=zones,
            y=means,
            error_y=dict(type='data', array=stds),
            marker_color='#3498DB',
            text=[f"{m:.1f}°C" for m in means],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="🌐 Temperatura por Zona Climática",
        xaxis_title="Zona",
        yaxis_title="Temperatura (°C)",
        height=300,
        margin=dict(l=50, r=20, t=50, b=40)
    )
    
    return fig


# ============================================================================
# Fragmentos Auto-actualizables
# ============================================================================

@st.fragment(run_every=timedelta(seconds=2))
def live_forecast_stats_fragment():
    """Estadísticas del pronóstico en tiempo real."""
    kafka_state = get_kafka_state()
    events = kafka_state.get_weather_events(200)
    stats = kafka_state.get_stats()
    
    if not events:
        st.info("🔄 Esperando datos...")
        return
    
    # Calcular estadísticas
    temps = [e.get('temperature', e.get('temperature_c', 0)) for e in events if e.get('temperature') or e.get('temperature_c')]
    cities = set(e.get('city', e.get('City', '')) for e in events)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        avg_temp = np.mean(temps) if temps else 0
        st.metric("🌡️ Temp Media", f"{avg_temp:.1f}°C")
    with col2:
        max_temp = max(temps) if temps else 0
        st.metric("🔥 Temp Máx", f"{max_temp:.1f}°C")
    with col3:
        min_temp = min(temps) if temps else 0
        st.metric("❄️ Temp Mín", f"{min_temp:.1f}°C")
    with col4:
        st.metric("🌍 Ciudades", len(cities))
    with col5:
        st.metric("📊 Eventos", stats.get('weather_buffered', 0))


@st.fragment(run_every=timedelta(seconds=3))
def live_multi_city_chart_fragment():
    """Gráfico multi-ciudad en tiempo real."""
    kafka_state = get_kafka_state()
    events = kafka_state.get_weather_events(300)
    
    variable = st.session_state.get('forecast_variable', 'temperature')
    fig = create_multi_city_forecast(events, variable)
    st.plotly_chart(fig, use_container_width=True, key="multi_city_chart")


@st.fragment(run_every=timedelta(seconds=3))
def live_forecast_map_fragment():
    """Mapa de pronóstico en tiempo real."""
    kafka_state = get_kafka_state()
    events = kafka_state.get_weather_events(200)
    
    fig = create_forecast_map(events)
    st.plotly_chart(fig, use_container_width=True, key="forecast_map")


@st.fragment(run_every=timedelta(seconds=4))
def live_city_detail_fragment():
    """Detalle de ciudad seleccionada."""
    city = st.session_state.get('forecast_city', 'ALL')
    
    if city == 'ALL':
        st.info("Selecciona una ciudad en el sidebar para ver el pronóstico detallado")
        return
    
    kafka_state = get_kafka_state()
    events = kafka_state.get_weather_events(500)
    
    fig = create_city_detail_chart(events, city)
    st.plotly_chart(fig, use_container_width=True, key="city_detail_chart")


@st.fragment(run_every=timedelta(seconds=5))
def live_zone_summary_fragment():
    """Resumen por zona climática."""
    kafka_state = get_kafka_state()
    events = kafka_state.get_weather_events(300)
    
    fig = create_zone_summary(events)
    st.plotly_chart(fig, use_container_width=True, key="zone_summary_chart")


# ============================================================================
# Sidebar
# ============================================================================

def render_sidebar():
    """Renderizar sidebar con controles."""
    st.sidebar.header("⚙️ Configuración")
    
    # Estado de Kafka
    kafka_state = get_kafka_state()
    
    if kafka_state.is_running():
        st.sidebar.success("🟢 Streaming Activo")
    else:
        st.sidebar.warning("🔴 Streaming Inactivo")
        if st.sidebar.button("▶️ Iniciar Consumer"):
            kafka_state.start([TOPICS['weather'], TOPICS['alerts']])
            st.rerun()
    
    st.sidebar.markdown("---")
    
    # Variable a visualizar
    st.sidebar.subheader("📊 Variable")
    st.session_state.forecast_variable = st.sidebar.selectbox(
        "Seleccionar",
        options=['temperature', 'humidity', 'wind_speed', 'rain', 'pressure'],
        format_func=lambda x: {
            'temperature': '🌡️ Temperatura',
            'humidity': '💧 Humedad',
            'wind_speed': '💨 Viento',
            'rain': '🌧️ Precipitación',
            'pressure': '📊 Presión'
        }.get(x, x),
        index=['temperature', 'humidity', 'wind_speed', 'rain', 'pressure'].index(
            st.session_state.get('forecast_variable', 'temperature')
        )
    )
    
    st.sidebar.markdown("---")
    
    # Selección de ciudad
    st.sidebar.subheader("🏙️ Ciudad")
    events = kafka_state.get_weather_events(100)
    cities = sorted(set(e.get('city', e.get('City', '')) for e in events if e.get('city') or e.get('City')))
    
    city_options = ['ALL'] + cities
    st.session_state.forecast_city = st.sidebar.selectbox(
        "Ver detalle de",
        options=city_options,
        format_func=lambda x: "📊 Todas las ciudades" if x == 'ALL' else f"🏙️ {x}"
    )
    
    st.sidebar.markdown("---")
    
    # Controles
    if st.sidebar.button("🗑️ Limpiar Buffer"):
        kafka_state.clear_buffers()
        st.rerun()
    
    if st.sidebar.button("🔄 Refrescar"):
        st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.info("Los gráficos se actualizan automáticamente cada 2-5 segundos.")


# ============================================================================
# Vista sin streaming
# ============================================================================

def render_no_streaming():
    """Mostrar cuando no hay streaming."""
    st.warning("⚠️ **Streaming no activo**")
    
    st.markdown("""
    ### Para ver pronósticos en tiempo real:
    
    1. Ve a **🌊 Streaming Hub**
    2. Inicia el **Productor**
    3. Inicia el **Consumer** aquí
    4. Los datos aparecerán automáticamente
    """)


# ============================================================================
# Main
# ============================================================================

def main():
    init_session_state()
    configure_sidebar()
    render_sidebar()
    
    # Header
    st.title("🌊 Pronóstico en Tiempo Real")
    st.markdown("""
    Visualización de pronósticos meteorológicos con datos **en tiempo real** de Apache Kafka.
    """)
    
    # Verificar Kafka
    kafka_state = get_kafka_state()
    
    if not kafka_state.is_running():
        kafka_check = check_kafka_available()
        if kafka_check.get('available'):
            kafka_state.start([TOPICS['weather'], TOPICS['alerts']])
        else:
            render_no_streaming()
            return
    
    st.markdown("---")
    
    # Estadísticas
    live_forecast_stats_fragment()
    
    st.markdown("---")
    
    # Layout: Mapa y Gráfico principal
    col1, col2 = st.columns([1, 2])
    
    with col1:
        live_forecast_map_fragment()
    
    with col2:
        live_multi_city_chart_fragment()
    
    st.markdown("---")
    
    # Layout: Detalle ciudad y Zona
    col3, col4 = st.columns([2, 1])
    
    with col3:
        st.subheader("📊 Detalle por Ciudad")
        live_city_detail_fragment()
    
    with col4:
        st.subheader("🌐 Por Zona Climática")
        live_zone_summary_fragment()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888;'>
        🌊 <strong>Streaming Forecast</strong> | Datos en tiempo real via Apache Kafka
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
