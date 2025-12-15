"""
🌀 Storm Tracking - Seguimiento de Tormentas en Tiempo Real

Visualización de tormentas y eventos extremos con trayectorias en vivo.
Los datos fluyen desde Apache Kafka y se actualizan automáticamente.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import sys
from pathlib import Path

# Configuración de página
st.set_page_config(
    page_title="Storm Tracking - climaXtreme",
    page_icon="🌀",
    layout="wide"
)

# Imports del proyecto
try:
    from climaxtreme.dashboard.components.kafka_realtime import (
        get_kafka_state,
        init_realtime_stream,
        check_kafka_available,
        TOPICS
    )
    from climaxtreme.dashboard.utils import configure_sidebar
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from climaxtreme.dashboard.components.kafka_realtime import (
        get_kafka_state,
        init_realtime_stream,
        check_kafka_available,
        TOPICS
    )
    from climaxtreme.dashboard.utils import configure_sidebar


# Escala Saffir-Simpson
STORM_CATEGORIES = {
    1: {'name': 'Categoría 1', 'color': '#F7DC6F', 'wind': '119-153 km/h', 'icon': '🌀'},
    2: {'name': 'Categoría 2', 'color': '#F5B041', 'wind': '154-177 km/h', 'icon': '🌀🌀'},
    3: {'name': 'Categoría 3', 'color': '#E74C3C', 'wind': '178-208 km/h', 'icon': '🌀🌀🌀'},
    4: {'name': 'Categoría 4', 'color': '#8E44AD', 'wind': '209-251 km/h', 'icon': '🌀🌀🌀🌀'},
    5: {'name': 'Categoría 5', 'color': '#1B2631', 'wind': '>252 km/h', 'icon': '🌀🌀🌀🌀🌀'}
}


# ============================================================================
# Session State
# ============================================================================

def init_session_state():
    """Inicializar estado de sesión."""
    if 'selected_storm' not in st.session_state:
        st.session_state.selected_storm = None
    if 'storm_history' not in st.session_state:
        st.session_state.storm_history = {}  # storm_id -> list of events


# ============================================================================
# Funciones de Visualización
# ============================================================================

def create_all_storms_map(storms: List[Dict]) -> go.Figure:
    """Crear mapa con todas las tormentas activas."""
    fig = go.Figure()
    
    if not storms:
        fig.add_annotation(
            text="Sin tormentas activas",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="#888")
        )
        fig.update_layout(height=500)
        return fig
    
    # Agrupar por tormenta (último evento de cada una)
    storm_latest = {}
    for s in storms:
        sid = s.get('storm_id', 'unknown')
        storm_latest[sid] = s
    
    for storm_id, storm in storm_latest.items():
        category = storm.get('category', 1)
        cat_info = STORM_CATEGORIES.get(category, STORM_CATEGORIES[1])
        
        fig.add_trace(go.Scattergeo(
            lat=[storm.get('latitude', 0)],
            lon=[storm.get('longitude', 0)],
            mode='markers+text',
            marker=dict(
                size=15 + category * 3,
                color=cat_info['color'],
                symbol='circle',
                line=dict(width=2, color='white')
            ),
            text=storm.get('storm_name', storm_id),
            textposition='top center',
            name=f"{storm.get('storm_name', storm_id)} (Cat {category})",
            hovertemplate=(
                f"<b>{storm.get('storm_name', storm_id)}</b><br>"
                f"Categoría: {category}<br>"
                f"Viento máx: {storm.get('max_wind_kmh', 0):.0f} km/h<br>"
                f"Presión: {storm.get('central_pressure', 0):.0f} hPa<br>"
                f"<extra></extra>"
            )
        ))
    
    fig.update_layout(
        title="🌀 Tormentas Activas en Tiempo Real",
        geo=dict(
            showland=True,
            landcolor='rgb(243, 243, 243)',
            countrycolor='rgb(204, 204, 204)',
            showocean=True,
            oceancolor='rgb(210, 235, 255)',
            showlakes=True,
            lakecolor='rgb(180, 215, 255)',
            showcountries=True,
            projection_type='natural earth'
        ),
        height=500,
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig


def create_storm_trajectory_map(storm_history: List[Dict], storm_name: str) -> go.Figure:
    """Crear mapa de trayectoria de una tormenta."""
    fig = go.Figure()
    
    if not storm_history:
        return fig
    
    # Ordenar por timestamp
    sorted_history = sorted(storm_history, key=lambda x: x.get('timestamp', ''))
    
    lats = [s.get('latitude', 0) for s in sorted_history]
    lons = [s.get('longitude', 0) for s in sorted_history]
    categories = [s.get('category', 1) for s in sorted_history]
    
    # Línea de trayectoria
    fig.add_trace(go.Scattergeo(
        lat=lats,
        lon=lons,
        mode='lines',
        line=dict(width=3, color='#E74C3C'),
        name='Trayectoria',
        hoverinfo='skip'
    ))
    
    # Puntos coloreados por categoría
    for i, (lat, lon, cat) in enumerate(zip(lats, lons, categories)):
        cat_info = STORM_CATEGORIES.get(cat, STORM_CATEGORIES[1])
        fig.add_trace(go.Scattergeo(
            lat=[lat],
            lon=[lon],
            mode='markers',
            marker=dict(
                size=8 + cat * 2,
                color=cat_info['color'],
                line=dict(width=1, color='white')
            ),
            name=f"Cat {cat}" if i == 0 else None,
            showlegend=(i == 0),
            hovertemplate=f"Cat {cat}<br>Lat: {lat:.2f}<br>Lon: {lon:.2f}<extra></extra>"
        ))
    
    # Marcador de inicio
    if lats:
        fig.add_trace(go.Scattergeo(
            lat=[lats[0]],
            lon=[lons[0]],
            mode='markers+text',
            marker=dict(size=15, color='green', symbol='star'),
            text=['INICIO'],
            textposition='bottom center',
            name='Inicio',
            showlegend=True
        ))
        
        # Marcador de posición actual
        fig.add_trace(go.Scattergeo(
            lat=[lats[-1]],
            lon=[lons[-1]],
            mode='markers+text',
            marker=dict(size=18, color='red', symbol='circle'),
            text=['ACTUAL'],
            textposition='top center',
            name='Posición Actual',
            showlegend=True
        ))
    
    # Calcular centro
    center_lat = sum(lats) / len(lats) if lats else 0
    center_lon = sum(lons) / len(lons) if lons else 0
    
    fig.update_layout(
        title=f"🌀 Trayectoria de {storm_name}",
        geo=dict(
            showland=True,
            landcolor='rgb(243, 243, 243)',
            showocean=True,
            oceancolor='rgb(210, 235, 255)',
            showcountries=True,
            projection_type='natural earth',
            center=dict(lat=center_lat, lon=center_lon),
            projection_scale=2.5
        ),
        height=450,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    return fig


def create_storm_intensity_chart(storm_history: List[Dict]) -> go.Figure:
    """Crear gráfico de evolución de intensidad."""
    if not storm_history:
        return go.Figure()
    
    sorted_history = sorted(storm_history, key=lambda x: x.get('timestamp', ''))
    
    updates = list(range(len(sorted_history)))
    categories = [s.get('category', 1) for s in sorted_history]
    winds = [s.get('max_wind_kmh', 0) for s in sorted_history]
    pressures = [s.get('central_pressure', 1000) for s in sorted_history]
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Categoría e Intensidad', 'Presión Central'),
        row_heights=[0.6, 0.4],
        vertical_spacing=0.12
    )
    
    # Categoría
    fig.add_trace(
        go.Scatter(
            x=updates, y=categories,
            mode='lines+markers',
            name='Categoría',
            line=dict(color='#E74C3C', width=3),
            marker=dict(size=8)
        ),
        row=1, col=1
    )
    
    # Viento máximo (eje secundario simulado)
    fig.add_trace(
        go.Scatter(
            x=updates, y=[w/50 for w in winds],  # Escalar para visualizar
            mode='lines',
            name='Viento (escala)',
            line=dict(color='#3498DB', width=2, dash='dot')
        ),
        row=1, col=1
    )
    
    # Presión
    fig.add_trace(
        go.Scatter(
            x=updates, y=pressures,
            mode='lines+markers',
            name='Presión (hPa)',
            line=dict(color='#9B59B6', width=2),
            marker=dict(size=6),
            fill='tozeroy',
            fillcolor='rgba(155, 89, 182, 0.2)'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=350,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=60, b=40)
    )
    
    fig.update_yaxes(title_text="Categoría", row=1, col=1)
    fig.update_yaxes(title_text="Presión (hPa)", row=2, col=1)
    fig.update_xaxes(title_text="Actualizaciones", row=2, col=1)
    
    return fig


# ============================================================================
# Fragmentos Auto-actualizables
# ============================================================================

@st.fragment(run_every=timedelta(seconds=3))
def live_storms_map_fragment():
    """Mapa de tormentas actualizándose en tiempo real."""
    kafka_state = get_kafka_state()
    storms = kafka_state.get_storm_events(100)
    
    # Actualizar historial
    for storm in storms:
        sid = storm.get('storm_id', 'unknown')
        if sid not in st.session_state.storm_history:
            st.session_state.storm_history[sid] = []
        # Evitar duplicados
        if storm not in st.session_state.storm_history[sid]:
            st.session_state.storm_history[sid].append(storm)
            # Limitar historial a 100 eventos por tormenta
            if len(st.session_state.storm_history[sid]) > 100:
                st.session_state.storm_history[sid] = st.session_state.storm_history[sid][-100:]
    
    fig = create_all_storms_map(storms)
    st.plotly_chart(fig, use_container_width=True, key="storms_map")


@st.fragment(run_every=timedelta(seconds=3))
def live_storms_stats_fragment():
    """Estadísticas de tormentas en tiempo real."""
    kafka_state = get_kafka_state()
    storms = kafka_state.get_storm_events(50)
    stats = kafka_state.get_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Contar tormentas únicas
    unique_storms = set(s.get('storm_id', '') for s in storms) if storms else set()
    
    # Categoría máxima
    max_cat = max((s.get('category', 1) for s in storms), default=0) if storms else 0
    
    # Viento máximo
    max_wind = max((s.get('max_wind_kmh', 0) for s in storms), default=0) if storms else 0
    
    with col1:
        st.metric("🌀 Tormentas Activas", len(unique_storms))
    with col2:
        st.metric("⚡ Categoría Máxima", max_cat if max_cat > 0 else "N/A")
    with col3:
        st.metric("💨 Viento Máximo", f"{max_wind:.0f} km/h" if max_wind > 0 else "N/A")
    with col4:
        st.metric("📊 Eventos Totales", stats.get('total_storms', 0))


@st.fragment(run_every=timedelta(seconds=3))
def live_storms_list_fragment():
    """Lista de tormentas activas."""
    kafka_state = get_kafka_state()
    storms = kafka_state.get_storm_events(50)
    
    if not storms:
        st.warning("""
        **Sin tormentas en el stream**
        
        Asegúrate de que el **Productor** esté corriendo en Streaming Hub.
        """)
        return
    
    # Obtener último evento de cada tormenta
    storm_latest = {}
    for s in storms:
        sid = s.get('storm_id', 'unknown')
        storm_latest[sid] = s
    
    for storm_id, storm in storm_latest.items():
        cat = storm.get('category', 1)
        cat_info = STORM_CATEGORIES.get(cat, STORM_CATEGORIES[1])
        name = storm.get('storm_name', storm_id)
        
        with st.container():
            st.markdown(f"""
            <div style='background-color:{cat_info["color"]}22; padding:10px; 
                        border-radius:5px; margin-bottom:10px; border-left: 4px solid {cat_info["color"]}'>
                <strong>{cat_info["icon"]} {name}</strong> - {cat_info["name"]}<br>
                <small>
                📍 Lat: {storm.get('latitude', 0):.2f}° | Lon: {storm.get('longitude', 0):.2f}°<br>
                💨 Viento: {storm.get('max_wind_kmh', 0):.0f} km/h | 
                📊 Presión: {storm.get('central_pressure', 0):.0f} hPa
                </small>
            </div>
            """, unsafe_allow_html=True)
            
            # Botón para seleccionar tormenta
            if st.button(f"Ver detalles de {name}", key=f"select_{storm_id}"):
                st.session_state.selected_storm = storm_id


@st.fragment(run_every=timedelta(seconds=4))
def live_selected_storm_fragment():
    """Detalle de tormenta seleccionada."""
    selected = st.session_state.get('selected_storm')
    
    if not selected or selected not in st.session_state.storm_history:
        st.info("Selecciona una tormenta de la lista para ver su trayectoria")
        return
    
    history = st.session_state.storm_history[selected]
    
    if not history:
        return
    
    # Nombre de la tormenta
    storm_name = history[-1].get('storm_name', selected)
    
    st.subheader(f"🌀 Detalles de {storm_name}")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_traj = create_storm_trajectory_map(history, storm_name)
        st.plotly_chart(fig_traj, use_container_width=True, key="storm_trajectory")
    
    with col2:
        fig_intensity = create_storm_intensity_chart(history)
        st.plotly_chart(fig_intensity, use_container_width=True, key="storm_intensity")
        
        # Info actual
        latest = history[-1]
        st.markdown(f"""
        **Estado Actual:**
        - Categoría: **{latest.get('category', 'N/A')}**
        - Viento: **{latest.get('max_wind_kmh', 0):.0f} km/h**
        - Presión: **{latest.get('central_pressure', 0):.0f} hPa**
        - Dirección: {latest.get('movement_direction', 0):.0f}°
        - Velocidad: {latest.get('movement_speed', 0):.0f} km/h
        - Actualizaciones: {len(history)}
        """)


# ============================================================================
# Sidebar
# ============================================================================

def render_sidebar():
    """Renderizar sidebar con controles."""
    st.sidebar.header("⚙️ Control")
    
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
    
    # Limpiar historial
    if st.sidebar.button("🗑️ Limpiar Historial"):
        st.session_state.storm_history = {}
        st.session_state.selected_storm = None
        kafka_state.clear_buffers()
        st.rerun()
    
    if st.sidebar.button("🔄 Refrescar"):
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Leyenda de categorías
    st.sidebar.subheader("📊 Escala Saffir-Simpson")
    for cat, info in STORM_CATEGORIES.items():
        st.sidebar.markdown(f"""
        <div style='background-color:{info["color"]}33; padding:5px; margin:2px; border-radius:3px;'>
            <small><strong>Cat {cat}</strong>: {info["wind"]}</small>
        </div>
        """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Fuente:** Apache Kafka Stream
    
    Las tormentas se generan y rastrean en tiempo real.
    """)


# ============================================================================
# Vista sin streaming
# ============================================================================

def render_no_streaming():
    """Mostrar cuando no hay streaming."""
    st.warning("⚠️ **Streaming no activo**")
    
    st.markdown("""
    ### Para ver el tracking de tormentas:
    
    1. Ve a **🌊 Streaming Hub**
    2. Inicia el **Productor** con tormentas habilitadas
    3. Inicia el **Consumer** aquí o en Streaming Hub
    4. Las tormentas aparecerán automáticamente
    """)


# ============================================================================
# Main
# ============================================================================

def main():
    init_session_state()
    configure_sidebar()
    render_sidebar()
    
    # Header
    st.title("🌀 Seguimiento de Tormentas")
    st.markdown("""
    Visualización en tiempo real de tormentas y eventos extremos con trayectorias actualizándose automáticamente.
    """)
    
    # Verificar Kafka y asegurar que estamos suscritos al topic de storms
    kafka_state = get_kafka_state()
    required_topics = [TOPICS['weather'], TOPICS['alerts'], TOPICS['storms']]
    
    if not kafka_state.is_running():
        kafka_check = check_kafka_available()
        if kafka_check.get('available'):
            kafka_state.start(required_topics)
        else:
            render_no_streaming()
            return
    else:
        # Asegurar que estamos suscritos al topic de storms
        kafka_state.ensure_topics(required_topics)
    
    st.markdown("---")
    
    # Estadísticas
    live_storms_stats_fragment()
    
    st.markdown("---")
    
    # Layout principal
    col_map, col_list = st.columns([2, 1])
    
    with col_map:
        live_storms_map_fragment()
    
    with col_list:
        st.subheader("📋 Tormentas Activas")
        live_storms_list_fragment()
    
    st.markdown("---")
    
    # Detalle de tormenta seleccionada
    live_selected_storm_fragment()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888;'>
        🌀 <strong>Storm Tracking</strong> | Datos en tiempo real via Apache Kafka
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
