"""
🔴 Live Streaming Dashboard - Streaming en Tiempo Real con Kafka

Esta página muestra datos climáticos que se actualizan en TIEMPO REAL
consumiendo de Apache Kafka. Los gráficos se actualizan automáticamente
usando @st.fragment para una experiencia fluida tipo video.

Arquitectura:
    [Kafka Producer] → [Kafka Topics] → [Este Dashboard]
         ↑                                    ↓
    [Spark Streaming]              [Gráficos Actualizándose]
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
import sys
from pathlib import Path

# Configuración de página
st.set_page_config(
    page_title="Live Streaming - climaXtreme",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Imports del proyecto
try:
    from climaxtreme.dashboard.components.kafka_realtime import (
        get_kafka_state,
        init_realtime_stream,
        render_kafka_status_card,
        render_realtime_controls,
        create_realtime_weather_chart,
        create_realtime_map,
        create_realtime_alerts_panel,
        create_realtime_metrics_row,
        create_city_comparison_chart,
        create_time_series_chart,
        check_kafka_available,
        render_kafka_setup_guide,
        TOPICS
    )
    from climaxtreme.dashboard.utils import configure_sidebar
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from climaxtreme.dashboard.components.kafka_realtime import (
        get_kafka_state,
        init_realtime_stream,
        render_kafka_status_card,
        render_realtime_controls,
        create_realtime_weather_chart,
        create_realtime_map,
        create_realtime_alerts_panel,
        create_realtime_metrics_row,
        create_city_comparison_chart,
        create_time_series_chart,
        check_kafka_available,
        render_kafka_setup_guide,
        TOPICS
    )
    from climaxtreme.dashboard.utils import configure_sidebar


# ============================================================================
# Session State
# ============================================================================

def init_session_state():
    """Inicializar estado de sesión."""
    if 'kafka_connected' not in st.session_state:
        st.session_state.kafka_connected = False
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = 2
    if 'view_mode' not in st.session_state:
        st.session_state.view_mode = 'dashboard'


# ============================================================================
# Fragmentos que se auto-actualizan (solo estas partes se refrescan)
# ============================================================================

@st.fragment(run_every=timedelta(seconds=2))
def live_metrics_fragment():
    """Fragmento de métricas que se actualiza automáticamente."""
    kafka_state = get_kafka_state()
    weather_events = kafka_state.get_weather_events(500)
    
    if weather_events:
        temps = [e.get('temperature', e.get('temperature_c', 0)) for e in weather_events if 'temperature' in e or 'temperature_c' in e]
        humidities = [e.get('humidity', e.get('humidity_pct', 0)) for e in weather_events if 'humidity' in e or 'humidity_pct' in e]
        winds = [e.get('wind_speed', e.get('wind_speed_kmh', 0)) for e in weather_events if 'wind_speed' in e or 'wind_speed_kmh' in e]
        cities = list(set(e.get('city', '') for e in weather_events if e.get('city')))
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("🏙️ Ciudades", len(cities))
        with col2:
            if temps:
                st.metric("🌡️ Temp Media", f"{np.mean(temps):.1f}°C", 
                         delta=f"{temps[-1] - np.mean(temps):.1f}°C" if len(temps) > 1 else None)
        with col3:
            if temps:
                st.metric("🔥 Máxima", f"{max(temps):.1f}°C")
        with col4:
            if temps:
                st.metric("❄️ Mínima", f"{min(temps):.1f}°C")
        with col5:
            st.metric("📊 Eventos", len(weather_events))
    else:
        st.info("⏳ Esperando datos...")


@st.fragment(run_every=timedelta(seconds=2))
def live_temperature_chart_fragment():
    """Fragmento del gráfico de temperatura que se actualiza automáticamente."""
    kafka_state = get_kafka_state()
    weather_events = kafka_state.get_weather_events(200)
    
    if weather_events:
        # Preparar datos
        data = []
        for e in weather_events[-100:]:  # Últimos 100 eventos
            temp = e.get('temperature', e.get('temperature_c'))
            city = e.get('city', 'Unknown')
            timestamp = e.get('timestamp', e.get('generated_at', datetime.now().isoformat()))
            if temp is not None:
                data.append({
                    'timestamp': timestamp,
                    'temperature': temp,
                    'city': city
                })
        
        if data:
            df = pd.DataFrame(data)
            
            fig = px.line(
                df, 
                x='timestamp', 
                y='temperature',
                color='city',
                title='🌡️ Temperatura en Tiempo Real',
                labels={'temperature': 'Temperatura (°C)', 'timestamp': 'Tiempo'}
            )
            fig.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
            )
            fig.update_traces(mode='lines+markers', marker=dict(size=4))
            
            st.plotly_chart(fig, use_container_width=True, key=f"temp_live_{time.time()}")
        else:
            st.info("⏳ Procesando datos de temperatura...")
    else:
        st.info("⏳ Esperando eventos de temperatura...")


@st.fragment(run_every=timedelta(seconds=2))
def live_alerts_fragment():
    """Fragmento de alertas que se actualiza automáticamente."""
    kafka_state = get_kafka_state()
    alerts = kafka_state.get_alert_events(15)
    
    if alerts:
        for alert in alerts[:8]:  # Mostrar últimas 8
            level = alert.get('alert_level', 'WATCH')
            city = alert.get('city', 'Unknown')
            alert_type = alert.get('alert_type', 'WEATHER')
            
            colors = {
                'EMERGENCY': '🔴',
                'WARNING': '🟠', 
                'WATCH': '🟡'
            }
            icon = colors.get(level, '⚪')
            
            st.markdown(f"""
            <div style='padding: 8px; margin: 4px 0; border-radius: 5px; 
                        background: rgba(128,128,128,0.1); border-left: 3px solid 
                        {"#e74c3c" if level=="EMERGENCY" else "#f39c12" if level=="WARNING" else "#f1c40f"};'>
                <strong>{icon} {city}</strong><br>
                <small>{alert_type} - {level}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ Sin alertas activas")


@st.fragment(run_every=timedelta(seconds=3))
def live_city_comparison_fragment():
    """Fragmento de comparación por ciudad que se actualiza automáticamente."""
    kafka_state = get_kafka_state()
    weather_events = kafka_state.get_weather_events(300)
    
    if weather_events:
        # Agrupar por ciudad
        city_temps = {}
        for e in weather_events:
            city = e.get('city', 'Unknown')
            temp = e.get('temperature', e.get('temperature_c'))
            if temp is not None:
                if city not in city_temps:
                    city_temps[city] = []
                city_temps[city].append(temp)
        
        if city_temps:
            # Calcular medias
            city_means = {city: np.mean(temps) for city, temps in city_temps.items()}
            
            # Ordenar por temperatura
            sorted_cities = sorted(city_means.items(), key=lambda x: x[1], reverse=True)[:10]
            
            cities = [c[0] for c in sorted_cities]
            temps = [c[1] for c in sorted_cities]
            
            # Colores según temperatura
            colors = ['#e74c3c' if t > 30 else '#3498db' if t < 10 else '#2ecc71' for t in temps]
            
            fig = go.Figure(data=[
                go.Bar(x=cities, y=temps, marker_color=colors)
            ])
            fig.update_layout(
                title='🏙️ Temperatura por Ciudad',
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig, use_container_width=True, key=f"city_comp_{time.time()}")
        else:
            st.info("⏳ Recopilando datos...")
    else:
        st.info("⏳ Esperando datos de ciudades...")


@st.fragment(run_every=timedelta(seconds=3))
def live_map_fragment():
    """Fragmento del mapa que se actualiza automáticamente."""
    kafka_state = get_kafka_state()
    weather_events = kafka_state.get_weather_events(200)
    
    if weather_events:
        # Filtrar eventos con coordenadas
        map_data = []
        for e in weather_events:
            lat = e.get('latitude', e.get('lat'))
            lon = e.get('longitude', e.get('lon'))
            temp = e.get('temperature', e.get('temperature_c'))
            city = e.get('city', 'Unknown')
            
            if lat and lon and temp:
                map_data.append({
                    'lat': lat,
                    'lon': lon,
                    'temperature': temp,
                    'city': city
                })
        
        if map_data:
            df = pd.DataFrame(map_data)
            
            # Tomar último valor por ciudad
            df = df.groupby('city').last().reset_index()
            
            fig = px.scatter_geo(
                df,
                lat='lat',
                lon='lon',
                color='temperature',
                hover_name='city',
                size=[15] * len(df),
                color_continuous_scale='RdYlBu_r',
                projection='natural earth',
                title='🗺️ Mapa de Temperaturas en Vivo'
            )
            fig.update_layout(
                height=400,
                margin=dict(l=0, r=0, t=40, b=0),
                geo=dict(
                    showland=True,
                    landcolor='rgb(243, 243, 243)',
                    countrycolor='rgb(204, 204, 204)',
                )
            )
            st.plotly_chart(fig, use_container_width=True, key=f"map_live_{time.time()}")
        else:
            st.info("⏳ Esperando datos con coordenadas...")
    else:
        st.info("⏳ Esperando datos para el mapa...")


@st.fragment(run_every=timedelta(seconds=2))
def live_status_indicator():
    """Indicador de estado en vivo."""
    kafka_state = get_kafka_state()
    
    if kafka_state.is_running():
        stats = kafka_state.get_stats()
        total = stats.get('total_weather', 0) + stats.get('total_alerts', 0)
        
        st.markdown(f"""
        <div style='text-align: center; padding: 15px; background: linear-gradient(135deg, #2E7D32, #1B5E20); 
                    border-radius: 10px; color: white; box-shadow: 0 2px 10px rgba(0,0,0,0.2);'>
            <div style='font-size: 24px; font-weight: bold;'>🔴 LIVE</div>
            <div style='font-size: 14px; opacity: 0.9;'>{total} eventos</div>
            <div style='font-size: 11px; opacity: 0.7;'>{datetime.now().strftime('%H:%M:%S')}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='text-align: center; padding: 15px; background: #424242; 
                    border-radius: 10px; color: white;'>
            <div style='font-size: 24px; font-weight: bold;'>⏸️ OFFLINE</div>
            <div style='font-size: 12px;'>Conectar para iniciar</div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# Sidebar (no se actualiza automáticamente)
# ============================================================================

def render_sidebar():
    """Renderizar sidebar con controles."""
    st.sidebar.header("🎛️ Control Panel")
    
    # Estado de Kafka
    kafka_check = check_kafka_available()
    
    if kafka_check.get('available'):
        st.sidebar.success("✅ Kafka disponible")
        if kafka_check.get('climaxtreme_topics'):
            st.sidebar.caption(f"Topics: {', '.join(kafka_check['climaxtreme_topics'][:3])}")
    else:
        st.sidebar.error("❌ Kafka no disponible")
        st.sidebar.caption(kafka_check.get('error', 'Verifica la conexión'))
    
    st.sidebar.markdown("---")
    
    # Controles de stream
    kafka_state = get_kafka_state()
    
    if kafka_state.is_running():
        if st.sidebar.button("⏹️ Detener Streaming", type="secondary", use_container_width=True):
            kafka_state.stop()
            st.session_state.kafka_connected = False
            st.rerun()
    else:
        if st.sidebar.button("▶️ Iniciar Streaming", type="primary", use_container_width=True):
            topics = [TOPICS['weather'], TOPICS['alerts'], TOPICS['storms']]
            if kafka_state.start(topics):
                st.session_state.kafka_connected = True
                st.rerun()
            else:
                st.sidebar.error("Error al conectar")
    
    st.sidebar.markdown("---")
    
    # Vista
    st.sidebar.subheader("📊 Vista")
    st.session_state.view_mode = st.sidebar.radio(
        "Modo de visualización",
        options=['dashboard', 'mapa', 'alertas'],
        format_func=lambda x: {
            'dashboard': '📊 Dashboard General',
            'mapa': '🗺️ Mapa en Vivo',
            'alertas': '🚨 Panel de Alertas'
        }.get(x, x)
    )
    
    st.sidebar.markdown("---")
    
    # Info
    st.sidebar.info("""
    💡 **Auto-refresh**: Los gráficos se actualizan 
    automáticamente cada 2-3 segundos sin 
    recargar la página completa.
    """)
    
    # Estadísticas
    if kafka_state.is_running():
        st.sidebar.subheader("📈 Estadísticas")
        stats = kafka_state.get_stats()
        
        st.sidebar.metric("Eventos Clima", stats.get('weather_buffered', 0))
        st.sidebar.metric("Alertas", stats.get('alerts_buffered', 0))
        st.sidebar.metric("Tormentas", stats.get('storms_buffered', 0))
        
        if st.sidebar.button("🗑️ Limpiar Buffers"):
            kafka_state.clear_buffers()
            st.rerun()


# ============================================================================
# Vistas principales
# ============================================================================

def render_dashboard_view():
    """Renderizar vista principal del dashboard con fragmentos auto-actualizables."""
    # Métricas (se actualiza cada 2s)
    st.subheader("📊 Métricas en Tiempo Real")
    live_metrics_fragment()
    
    st.markdown("---")
    
    # Grid principal
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Gráfico de temperatura (se actualiza cada 2s)
        live_temperature_chart_fragment()
    
    with col2:
        st.subheader("🚨 Alertas Activas")
        live_alerts_fragment()
    
    st.markdown("---")
    
    # Segunda fila
    col3, col4 = st.columns(2)
    
    with col3:
        live_city_comparison_fragment()
    
    with col4:
        st.subheader("📈 Resumen")
        kafka_state = get_kafka_state()
        stats = kafka_state.get_stats()
        
        st.markdown(f"""
        **Estado del Stream:**
        - 🌡️ Eventos clima en buffer: `{stats.get('weather_buffered', 0)}`
        - 🚨 Alertas en buffer: `{stats.get('alerts_buffered', 0)}`
        - 🌀 Tormentas en buffer: `{stats.get('storms_buffered', 0)}`
        - 📊 Total procesados: `{stats.get('total_weather', 0) + stats.get('total_alerts', 0)}`
        """)


def render_map_view():
    """Renderizar vista de mapa en vivo con fragmento auto-actualizable."""
    st.subheader("🗺️ Mapa de Temperaturas en Tiempo Real")
    
    # Mapa (se actualiza cada 3s)
    live_map_fragment()
    
    st.markdown("---")
    
    # Métricas debajo
    live_metrics_fragment()


def render_alerts_view():
    """Renderizar vista de alertas con fragmento auto-actualizable."""
    st.subheader("🚨 Panel de Alertas en Tiempo Real")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Panel de alertas grande
        kafka_state = get_kafka_state()
        alerts = kafka_state.get_alert_events(50)
        
        if alerts:
            # Contadores por nivel
            emergency = sum(1 for a in alerts if a.get('alert_level') == 'EMERGENCY')
            warning = sum(1 for a in alerts if a.get('alert_level') == 'WARNING')
            watch = sum(1 for a in alerts if a.get('alert_level') == 'WATCH')
            
            c1, c2, c3 = st.columns(3)
            c1.metric("🔴 Emergencias", emergency)
            c2.metric("🟠 Advertencias", warning)
            c3.metric("🟡 Vigilancia", watch)
            
            st.markdown("---")
            
            # Lista expandida de alertas
            live_alerts_fragment()
        else:
            st.success("✅ No hay alertas activas")
    
    with col2:
        st.markdown("### 📊 Distribución")
        
        kafka_state = get_kafka_state()
        alerts = kafka_state.get_alert_events(100)
        
        if alerts:
            # Distribución por tipo
            types = {}
            for a in alerts:
                t = a.get('alert_type', 'OTHER')
                types[t] = types.get(t, 0) + 1
            
            if types:
                fig = px.pie(
                    values=list(types.values()),
                    names=list(types.keys()),
                    title="Por Tipo",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_layout(height=300, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)


def render_not_connected():
    """Renderizar estado no conectado."""
    st.warning("⚠️ **No conectado a Kafka Streaming**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🚀 Para iniciar el streaming:
        
        1. **Ve al Streaming Hub** y activa el productor
        2. **Haz clic en "Iniciar Streaming"** en el sidebar
        3. ¡Los gráficos se actualizarán automáticamente!
        
        ---
        
        ### 📋 Requisitos:
        - Kafka debe estar corriendo
        - El productor debe estar activo
        - Los topics deben existir
        """)
    
    with col2:
        st.markdown("### 📊 Preview de Demo")
        
        # Datos de demo estáticos
        demo_temps = [20 + np.random.randn() * 5 for _ in range(20)]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=demo_temps, 
            mode='lines+markers',
            line=dict(color='#3498db', width=2),
            marker=dict(size=6)
        ))
        fig.update_layout(
            title='📊 Datos de Demo (no en vivo)',
            height=250,
            xaxis_title='Eventos',
            yaxis_title='Temperatura (°C)',
            margin=dict(l=20, r=20, t=40, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    render_kafka_setup_guide()


# ============================================================================
# Main
# ============================================================================

def main():
    """Punto de entrada principal."""
    init_session_state()
    
    # Sidebar
    render_sidebar()
    
    # Header
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.title("🔴 Live Streaming Dashboard")
        st.markdown("""
        **Visualización en tiempo real** de datos climáticos desde Apache Kafka.
        Los gráficos se actualizan automáticamente cada 2-3 segundos.
        """)
    
    with col2:
        # Indicador de estado (se actualiza automáticamente)
        live_status_indicator()
    
    st.markdown("---")
    
    # Contenido principal basado en estado de conexión
    kafka_state = get_kafka_state()
    
    if kafka_state.is_running():
        # Mostrar vista seleccionada
        if st.session_state.view_mode == 'dashboard':
            render_dashboard_view()
        elif st.session_state.view_mode == 'mapa':
            render_map_view()
        elif st.session_state.view_mode == 'alertas':
            render_alerts_view()
    else:
        render_not_connected()


if __name__ == "__main__":
    main()
