"""
🚨 Active Alerts - Panel de Alertas en Tiempo Real

Dashboard de alertas meteorológicas activas con streaming de Kafka.
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
from collections import Counter
import sys
from pathlib import Path

# Configuración de página
st.set_page_config(
    page_title="Active Alerts - climaXtreme",
    page_icon="🚨",
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


# Estilos de alertas
ALERT_STYLES = {
    'EMERGENCY': {'color': '#C0392B', 'icon': '🔴', 'bg': '#FADBD8', 'priority': 1},
    'WARNING': {'color': '#E67E22', 'icon': '🟠', 'bg': '#FDEBD0', 'priority': 2},
    'WATCH': {'color': '#F1C40F', 'icon': '🟡', 'bg': '#FEF9E7', 'priority': 3}
}

ALERT_TYPE_ICONS = {
    'HEAT': '🔥',
    'COLD': '❄️',
    'STORM': '🌀',
    'FLOOD': '🌊',
    'WIND': '💨',
    'RAIN': '🌧️',
    'WEATHER': '⛈️'
}


# ============================================================================
# Session State
# ============================================================================

def init_session_state():
    """Inicializar estado de sesión."""
    if 'alert_filter_level' not in st.session_state:
        st.session_state.alert_filter_level = 'ALL'
    if 'alert_filter_type' not in st.session_state:
        st.session_state.alert_filter_type = 'ALL'


# ============================================================================
# Funciones de Visualización
# ============================================================================

def create_alert_card_html(alert: Dict) -> str:
    """Crear HTML para una tarjeta de alerta."""
    level = alert.get('alert_level', 'WATCH')
    alert_type = alert.get('alert_type', 'WEATHER')
    
    style = ALERT_STYLES.get(level, ALERT_STYLES['WATCH'])
    icon = ALERT_TYPE_ICONS.get(alert_type, '⚠️')
    
    city = alert.get('city', alert.get('City', 'Unknown'))
    country = alert.get('country', alert.get('Country', ''))
    temp = alert.get('temperature', alert.get('temperature_hourly', 0))
    wind = alert.get('wind_speed', alert.get('wind_speed_kmh', 0))
    rain = alert.get('rain_mm', 0)
    intensity = alert.get('event_intensity', 0)
    
    timestamp = alert.get('timestamp', 'N/A')
    if isinstance(timestamp, str) and 'T' in timestamp:
        timestamp = timestamp.split('T')[1][:8]
    
    return f"""
    <div style='
        background-color:{style["bg"]}; 
        border-left: 4px solid {style["color"]}; 
        padding: 12px; 
        margin: 8px 0; 
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    '>
        <div style='display: flex; justify-content: space-between; align-items: center;'>
            <div>
                <span style='font-size: 20px;'>{icon} {style["icon"]}</span>
                <strong style='font-size: 16px; color: {style["color"]};'> {level}</strong>
                <span style='color: #666;'> - {alert_type}</span>
            </div>
            <span style='color: #888; font-size: 11px;'>🕐 {timestamp}</span>
        </div>
        <div style='margin-top: 8px;'>
            <strong>{city}</strong>{f', {country}' if country else ''}
        </div>
        <div style='margin-top: 5px; color: #555; font-size: 13px;'>
            🌡️ {temp:.1f}°C | 💨 {wind:.1f} km/h | 🌧️ {rain:.1f} mm
            {f' | ⚡ {intensity:.2f}' if intensity else ''}
        </div>
    </div>
    """


def create_alerts_map(alerts: List[Dict]) -> go.Figure:
    """Crear mapa de ubicación de alertas."""
    fig = go.Figure()
    
    if not alerts:
        fig.add_annotation(
            text="Sin alertas activas",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=18, color="#888")
        )
        fig.update_layout(height=400)
        return fig
    
    # Agrupar por nivel de alerta
    for level in ['EMERGENCY', 'WARNING', 'WATCH']:
        level_alerts = [a for a in alerts if a.get('alert_level') == level]
        
        if not level_alerts:
            continue
        
        style = ALERT_STYLES.get(level, ALERT_STYLES['WATCH'])
        
        lats = [a.get('latitude', a.get('lat_decimal', 0)) for a in level_alerts]
        lons = [a.get('longitude', a.get('lon_decimal', 0)) for a in level_alerts]
        cities = [a.get('city', a.get('City', 'Unknown')) for a in level_alerts]
        types = [a.get('alert_type', 'WEATHER') for a in level_alerts]
        
        fig.add_trace(go.Scattergeo(
            lat=lats,
            lon=lons,
            mode='markers',
            marker=dict(
                size=12 if level == 'EMERGENCY' else (10 if level == 'WARNING' else 8),
                color=style['color'],
                symbol='circle',
                line=dict(width=1, color='white')
            ),
            name=f"{style['icon']} {level} ({len(level_alerts)})",
            hovertemplate=[
                f"<b>{city}</b><br>{level} - {atype}<extra></extra>"
                for city, atype in zip(cities, types)
            ]
        ))
    
    fig.update_layout(
        title="🗺️ Mapa de Alertas Activas",
        geo=dict(
            showland=True,
            landcolor='rgb(243, 243, 243)',
            countrycolor='rgb(204, 204, 204)',
            showocean=True,
            oceancolor='rgb(230, 245, 255)',
            showcountries=True,
            projection_type='natural earth'
        ),
        height=400,
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig


def create_alerts_summary_chart(alerts: List[Dict]) -> go.Figure:
    """Crear gráfico resumen de alertas."""
    if not alerts:
        return go.Figure()
    
    # Contar por nivel
    level_counts = Counter(a.get('alert_level', 'WATCH') for a in alerts)
    
    # Contar por tipo
    type_counts = Counter(a.get('alert_type', 'WEATHER') for a in alerts)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Por Nivel de Severidad', 'Por Tipo de Alerta'),
        specs=[[{"type": "pie"}, {"type": "bar"}]]
    )
    
    # Pie chart de niveles
    levels = list(level_counts.keys())
    colors = [ALERT_STYLES.get(l, {}).get('color', '#888') for l in levels]
    
    fig.add_trace(
        go.Pie(
            labels=levels,
            values=[level_counts[l] for l in levels],
            marker=dict(colors=colors),
            textinfo='label+value',
            hole=0.4
        ),
        row=1, col=1
    )
    
    # Bar chart de tipos
    types = list(type_counts.keys())
    fig.add_trace(
        go.Bar(
            x=types,
            y=[type_counts[t] for t in types],
            marker_color='#3498DB',
            text=[type_counts[t] for t in types],
            textposition='auto'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=300,
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


# ============================================================================
# Fragmentos Auto-actualizables
# ============================================================================

@st.fragment(run_every=timedelta(seconds=2))
def live_alerts_stats_fragment():
    """Estadísticas de alertas en tiempo real."""
    kafka_state = get_kafka_state()
    alerts = kafka_state.get_alert_events(100)
    stats = kafka_state.get_stats()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Contar por nivel
    emergency_count = sum(1 for a in alerts if a.get('alert_level') == 'EMERGENCY')
    warning_count = sum(1 for a in alerts if a.get('alert_level') == 'WARNING')
    watch_count = sum(1 for a in alerts if a.get('alert_level') == 'WATCH')
    
    with col1:
        st.metric("🔴 Emergencias", emergency_count)
    with col2:
        st.metric("🟠 Avisos", warning_count)
    with col3:
        st.metric("🟡 Vigilancias", watch_count)
    with col4:
        st.metric("📊 Total Buffer", stats.get('alerts_buffered', 0))
    with col5:
        st.metric("🌐 Ciudades", len(set(a.get('city', '') for a in alerts)))


@st.fragment(run_every=timedelta(seconds=2))
def live_alerts_list_fragment():
    """Lista de alertas en tiempo real."""
    kafka_state = get_kafka_state()
    alerts = kafka_state.get_alert_events(50)
    
    if not alerts:
        st.info("✅ Sin alertas activas - Todo en orden")
        return
    
    # Aplicar filtros
    filter_level = st.session_state.get('alert_filter_level', 'ALL')
    filter_type = st.session_state.get('alert_filter_type', 'ALL')
    
    filtered_alerts = alerts
    if filter_level != 'ALL':
        filtered_alerts = [a for a in filtered_alerts if a.get('alert_level') == filter_level]
    if filter_type != 'ALL':
        filtered_alerts = [a for a in filtered_alerts if a.get('alert_type') == filter_type]
    
    # Ordenar por prioridad
    def get_priority(alert):
        level = alert.get('alert_level', 'WATCH')
        return ALERT_STYLES.get(level, {}).get('priority', 99)
    
    sorted_alerts = sorted(filtered_alerts, key=get_priority)
    
    # Mostrar últimas 15 alertas
    for alert in sorted_alerts[:15]:
        st.markdown(create_alert_card_html(alert), unsafe_allow_html=True)
    
    if len(sorted_alerts) > 15:
        st.caption(f"... y {len(sorted_alerts) - 15} alertas más")


@st.fragment(run_every=timedelta(seconds=3))
def live_alerts_map_fragment():
    """Mapa de alertas en tiempo real."""
    kafka_state = get_kafka_state()
    alerts = kafka_state.get_alert_events(100)
    
    fig = create_alerts_map(alerts)
    st.plotly_chart(fig, use_container_width=True, key="alerts_map")


@st.fragment(run_every=timedelta(seconds=4))
def live_alerts_chart_fragment():
    """Gráficos de alertas en tiempo real."""
    kafka_state = get_kafka_state()
    alerts = kafka_state.get_alert_events(100)
    
    if not alerts:
        st.info("Esperando alertas...")
        return
    
    fig = create_alerts_summary_chart(alerts)
    st.plotly_chart(fig, use_container_width=True, key="alerts_summary_chart")


# ============================================================================
# Sidebar
# ============================================================================

def render_sidebar():
    """Renderizar sidebar con controles."""
    st.sidebar.header("⚙️ Filtros y Control")
    
    # Estado de Kafka
    kafka_state = get_kafka_state()
    
    if kafka_state.is_running():
        st.sidebar.success("🟢 Streaming Activo")
    else:
        st.sidebar.warning("🔴 Streaming Inactivo")
        if st.sidebar.button("▶️ Iniciar Consumer"):
            kafka_state.start([TOPICS['weather'], TOPICS['alerts'], TOPICS['storms']])
            st.rerun()
    
    st.sidebar.markdown("---")
    
    # Filtro por nivel
    st.sidebar.subheader("🎚️ Filtrar por Nivel")
    st.session_state.alert_filter_level = st.sidebar.selectbox(
        "Nivel de Alerta",
        options=['ALL', 'EMERGENCY', 'WARNING', 'WATCH'],
        format_func=lambda x: {
            'ALL': '📊 Todas',
            'EMERGENCY': '🔴 Emergencias',
            'WARNING': '🟠 Avisos',
            'WATCH': '🟡 Vigilancias'
        }.get(x, x)
    )
    
    # Filtro por tipo
    st.sidebar.subheader("🏷️ Filtrar por Tipo")
    st.session_state.alert_filter_type = st.sidebar.selectbox(
        "Tipo de Alerta",
        options=['ALL', 'HEAT', 'COLD', 'WIND', 'RAIN', 'FLOOD', 'STORM'],
        format_func=lambda x: {
            'ALL': '📊 Todos',
            'HEAT': '🔥 Calor',
            'COLD': '❄️ Frío',
            'WIND': '💨 Viento',
            'RAIN': '🌧️ Lluvia',
            'FLOOD': '🌊 Inundación',
            'STORM': '🌀 Tormenta'
        }.get(x, x)
    )
    
    st.sidebar.markdown("---")
    
    # Controles
    if st.sidebar.button("🗑️ Limpiar Alertas"):
        kafka_state.clear_buffers()
        st.rerun()
    
    if st.sidebar.button("🔄 Refrescar"):
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Leyenda
    st.sidebar.subheader("📋 Leyenda")
    for level, style in ALERT_STYLES.items():
        st.sidebar.markdown(f"{style['icon']} **{level}**")
    
    st.sidebar.markdown("---")
    st.sidebar.info("Los datos se actualizan automáticamente cada 2-4 segundos.")


# ============================================================================
# Vista sin streaming
# ============================================================================

def render_no_streaming():
    """Mostrar cuando no hay streaming."""
    st.warning("⚠️ **Streaming no activo**")
    
    st.markdown("""
    ### Para ver alertas en tiempo real:
    
    1. Ve a **🌊 Streaming Hub**
    2. Inicia el **Productor** con alertas habilitadas
    3. Inicia el **Consumer** aquí
    4. Las alertas aparecerán automáticamente
    """)
    
    # Demo
    st.subheader("📊 Demo")
    
    demo_alerts = [
        {'alert_level': 'EMERGENCY', 'alert_type': 'HEAT', 'city': 'Dubai', 'country': 'UAE', 
         'temperature': 45.2, 'wind_speed': 15, 'rain_mm': 0, 'timestamp': '2024-01-01T12:00:00'},
        {'alert_level': 'WARNING', 'alert_type': 'WIND', 'city': 'Tokyo', 'country': 'Japan',
         'temperature': 28.5, 'wind_speed': 85, 'rain_mm': 12, 'timestamp': '2024-01-01T12:05:00'},
        {'alert_level': 'WATCH', 'alert_type': 'RAIN', 'city': 'London', 'country': 'UK',
         'temperature': 12.3, 'wind_speed': 25, 'rain_mm': 35, 'timestamp': '2024-01-01T12:10:00'},
    ]
    
    for alert in demo_alerts:
        st.markdown(create_alert_card_html(alert), unsafe_allow_html=True)
    
    st.caption("*Datos de demostración*")


# ============================================================================
# Main
# ============================================================================

def main():
    init_session_state()
    configure_sidebar()
    render_sidebar()
    
    # Header
    st.title("🚨 Panel de Alertas Activas")
    st.markdown("""
    Monitor de alertas meteorológicas en **tiempo real** con datos de streaming de Kafka.
    """)
    
    # Verificar Kafka
    kafka_state = get_kafka_state()
    
    if not kafka_state.is_running():
        kafka_check = check_kafka_available()
        if kafka_check.get('available'):
            kafka_state.start([TOPICS['weather'], TOPICS['alerts'], TOPICS['storms']])
        else:
            render_no_streaming()
            return
    
    st.markdown("---")
    
    # Estadísticas principales
    live_alerts_stats_fragment()
    
    st.markdown("---")
    
    # Layout principal: Mapa y Lista
    col_map, col_list = st.columns([3, 2])
    
    with col_map:
        live_alerts_map_fragment()
    
    with col_list:
        st.subheader("📋 Alertas Recientes")
        live_alerts_list_fragment()
    
    st.markdown("---")
    
    # Gráficos de resumen
    st.subheader("📊 Resumen de Alertas")
    live_alerts_chart_fragment()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888;'>
        🚨 <strong>Active Alerts</strong> | Datos en tiempo real via Apache Kafka
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
