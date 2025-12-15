"""
🌊 Streaming Hub - Centro de Control de Kafka Streaming

Centro de control para streaming en tiempo real con Apache Kafka:
- Verificar estado del clúster Kafka
- Iniciar/detener productor de datos
- Monitorear eventos en tiempo real
- Acceso rápido a dashboards de streaming

Arquitectura:
    [Kafka Producer] → [Kafka Topics] → [Dashboard Consumer]
         ↑                                    ↓
    [Generador Simple]              [Gráficos Actualizándose]
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
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
        KafkaStreamingConfig
    )
    from climaxtreme.dashboard.utils import configure_sidebar
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
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
        KafkaStreamingConfig
    )
    from climaxtreme.dashboard.utils import configure_sidebar


# ============================================================================
# Session State Initialization
# ============================================================================

def init_session_state():
    """Inicializa el estado de la sesión."""
    if 'kafka_config' not in st.session_state:
        st.session_state.kafka_config = KafkaStreamingConfig()


# ============================================================================
# Fragmentos Auto-actualizables
# ============================================================================

@st.fragment(run_every=timedelta(seconds=3))
def live_events_counter():
    """Contador de eventos en tiempo real."""
    kafka_state = get_kafka_state()
    stats = kafka_state.get_stats()
    
    if kafka_state.is_running():
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🌡️ Eventos Clima", f"{stats.get('total_weather', 0):,}")
        with col2:
            st.metric("🚨 Alertas", f"{stats.get('total_alerts', 0):,}")
        with col3:
            st.metric("🌀 Tormentas", f"{stats.get('total_storms', 0):,}")
        with col4:
            buffer_total = stats.get('weather_buffered', 0) + stats.get('alerts_buffered', 0) + stats.get('storms_buffered', 0)
            st.metric("📦 En Buffer", f"{buffer_total:,}")
        
        if stats.get('last_event_time'):
            st.caption(f"🕐 Último evento: {stats['last_event_time']}")
    else:
        st.info("⏸️ Consumer no activo - Los contadores se actualizarán cuando inicie el stream")


@st.fragment(run_every=timedelta(seconds=5))
def live_temperature_preview():
    """Preview de temperaturas en tiempo real."""
    kafka_state = get_kafka_state()
    
    if not kafka_state.is_running():
        return
    
    events = kafka_state.get_weather_events(100)
    
    if not events:
        st.info("Esperando datos de temperatura...")
        return
    
    # Extraer temperaturas
    temps = []
    cities = []
    for e in events[-50:]:  # Últimos 50
        temp = e.get('temperature') or e.get('temperature_c')
        if temp is not None:
            temps.append(temp)
            cities.append(e.get('city', 'Unknown'))
    
    if temps:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=list(range(len(temps))),
            y=temps,
            mode='lines+markers',
            line=dict(color='#E74C3C', width=2),
            marker=dict(size=4),
            hovertemplate='%{y:.1f}°C - %{text}<extra></extra>',
            text=cities
        ))
        
        fig.update_layout(
            title="🌡️ Temperaturas Recientes",
            xaxis_title="Eventos recientes",
            yaxis_title="Temperatura (°C)",
            height=250,
            margin=dict(l=50, r=20, t=40, b=40)
        )
        
        st.plotly_chart(fig, use_container_width=True)


@st.fragment(run_every=timedelta(seconds=5))
def live_alerts_preview():
    """Preview de alertas en tiempo real."""
    kafka_state = get_kafka_state()
    
    if not kafka_state.is_running():
        return
    
    alerts = kafka_state.get_alert_events(10)
    
    if not alerts:
        st.success("✅ Sin alertas recientes")
        return
    
    alert_styles = {
        'EMERGENCY': ('🔴', '#FADBD8'),
        'WARNING': ('🟠', '#FDEBD0'),
        'WATCH': ('🟡', '#FEF9E7')
    }
    
    for alert in reversed(alerts[-5:]):
        level = alert.get('alert_level', 'WATCH')
        alert_type = alert.get('alert_type', 'WEATHER')
        city = alert.get('city', 'Unknown')
        
        icon, bg = alert_styles.get(level, ('⚪', '#F5F5F5'))
        
        st.markdown(f"""
        <div style='background-color:{bg}; padding:8px; border-radius:5px; margin-bottom:5px;'>
            {icon} <strong>{level}</strong> - {alert_type} en {city}
        </div>
        """, unsafe_allow_html=True)


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
    **Centro de Control de Kafka Streaming** - Genera y transmite datos climáticos sintéticos
    en tiempo real usando Apache Kafka.
    """)
    
    # Diagrama de arquitectura
    with st.expander("📐 Arquitectura del Sistema", expanded=False):
        st.info("""
        ```
        ┌─────────────────────┐    ┌─────────────────┐    ┌─────────────────┐
        │ Generador de Datos  │    │   Kafka         │    │   Dashboard     │
        │ (Python Thread)     │───▶│   Broker        │───▶│   (Real-time)   │
        │ - Ciclo diurno      │    │   Topics:       │    │   - Live Stream │
        │ - Variables meteo   │    │   • weather     │    │   - Heatmaps    │
        │ - Alertas           │    │   • alerts      │    │   - Tormentas   │
        │ - Tormentas         │    │   • storms      │    │   - Alertas     │
        └─────────────────────┘    └─────────────────┘    └─────────────────┘
        ```
        """)
    
    st.markdown("---")
    
    # ========================================================================
    # Sección 1: Estado del Clúster
    # ========================================================================
    st.header("📊 Estado del Clúster Kafka")
    
    col_status1, col_status2 = st.columns([2, 1])
    
    with col_status1:
        render_kafka_cluster_status()
    
    with col_status2:
        kafka_manager = get_streaming_manager()
        
        if st.button("🔄 Refrescar Estado", use_container_width=True, key="refresh_cluster"):
            st.rerun()
        
        if st.button("📋 Crear Topics", use_container_width=True, key="create_topics"):
            with st.spinner("Creando topics..."):
                success, msg = kafka_manager.create_topics()
                if success:
                    st.success(msg)
                else:
                    st.error(msg)
    
    st.markdown("---")
    
    # ========================================================================
    # Sección 2: Control del Productor
    # ========================================================================
    st.header("🎛️ Control del Productor")
    
    col_prod1, col_prod2 = st.columns([2, 1])
    
    with col_prod1:
        st.markdown("**Configuración del Productor:**")
        
        col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
        
        with col_cfg1:
            n_cities = st.slider(
                "Número de ciudades",
                min_value=5, max_value=50, value=20, step=5,
                key="producer_cities",
                help="Ciudades a incluir en la generación"
            )
        
        with col_cfg2:
            interval = st.slider(
                "Intervalo (segundos)",
                min_value=0.5, max_value=5.0, value=1.0, step=0.5,
                key="producer_interval",
                help="Tiempo entre batches de eventos"
            )
        
        with col_cfg3:
            alert_prob = st.slider(
                "Prob. alertas",
                min_value=0.0, max_value=0.5, value=0.1, step=0.05,
                key="producer_alert_prob",
                help="Probabilidad de generar alertas"
            )
    
    with col_prod2:
        kafka_manager = get_streaming_manager()
        
        st.markdown("**Acciones:**")
        
        if kafka_manager.is_producing():
            st.success("🟢 **Productor Activo**")
            
            if st.button("⏹️ Detener Productor", type="secondary", use_container_width=True, key="stop_prod"):
                success, msg = kafka_manager.stop_producer()
                if success:
                    st.success(msg)
                else:
                    st.error(msg)
                st.rerun()
        else:
            st.warning("⚫ **Productor Detenido**")
            
            if st.button("▶️ Iniciar Productor", type="primary", use_container_width=True, key="start_prod"):
                config = KafkaStreamingConfig(
                    n_cities=n_cities,
                    interval_seconds=interval,
                    include_alerts=True,
                    include_storms=True,
                    alert_probability=alert_prob,
                    storm_probability=0.05
                )
                with st.spinner("Iniciando productor..."):
                    success, msg = kafka_manager.start_producer(config, use_spark=False)
                if success:
                    st.success(msg)
                    st.balloons()
                else:
                    st.error(msg)
                st.rerun()
    
    st.markdown("---")
    
    # ========================================================================
    # Sección 3: Consumer y Monitoreo
    # ========================================================================
    st.header("📡 Consumer y Monitoreo en Tiempo Real")
    
    col_cons1, col_cons2 = st.columns([2, 1])
    
    with col_cons1:
        render_kafka_status_card()
    
    with col_cons2:
        kafka_state = get_kafka_state()
        
        if kafka_state.is_running():
            if st.button("⏹️ Detener Consumer", use_container_width=True, key="stop_consumer"):
                kafka_state.stop()
                st.rerun()
        else:
            if st.button("▶️ Iniciar Consumer", type="primary", use_container_width=True, key="start_consumer"):
                kafka_state.start([TOPICS['weather'], TOPICS['alerts'], TOPICS['storms']])
                st.rerun()
        
        if st.button("🗑️ Limpiar Buffers", use_container_width=True, key="clear_buffers"):
            kafka_state.clear_buffers()
            st.rerun()
    
    st.markdown("---")
    
    # ========================================================================
    # Sección 4: Estadísticas en Tiempo Real
    # ========================================================================
    st.header("📈 Estadísticas en Tiempo Real")
    
    live_events_counter()
    
    col_charts1, col_charts2 = st.columns(2)
    
    with col_charts1:
        live_temperature_preview()
    
    with col_charts2:
        live_alerts_preview()
    
    st.markdown("---")
    
    # ========================================================================
    # Sección 5: Accesos Rápidos
    # ========================================================================
    st.header("🔗 Dashboards de Streaming")
    
    st.markdown("""
    Una vez que el productor esté activo, los datos fluirán a todas estas páginas:
    """)
    
    col_link1, col_link2, col_link3, col_link4 = st.columns(4)
    
    with col_link1:
        st.markdown("""
        ### 🔴 Live Streaming
        Dashboard principal con métricas, gráficos y alertas actualizándose en tiempo real.
        
        [Abrir →](/Live_Streaming)
        """)
    
    with col_link2:
        st.markdown("""
        ### 🗺️ Climate Heatmaps
        Mapas globales de temperatura, viento y otras variables en tiempo real.
        
        [Abrir →](/Climate_Heatmaps)
        """)
    
    with col_link3:
        st.markdown("""
        ### 🌀 Storm Tracking
        Seguimiento de tormentas y eventos extremos con trayectorias en vivo.
        
        [Abrir →](/Storm_Tracking)
        """)
    
    with col_link4:
        st.markdown("""
        ### 🚨 Active Alerts
        Panel de alertas meteorológicas activas con filtros y mapas.
        
        [Abrir →](/Active_Alerts)
        """)
    
    # ========================================================================
    # Sección 6: Guía Rápida
    # ========================================================================
    with st.expander("📚 Guía de Inicio Rápido"):
        st.markdown("""
        ### 🚀 Pasos para iniciar el streaming
        
        1. **Verificar Kafka** - Asegúrate de que Zookeeper y Kafka estén corriendo
           - Los indicadores de estado deben estar en verde
        
        2. **Crear Topics** - Si es la primera vez, haz clic en "Crear Topics"
        
        3. **Iniciar Productor** - Configura los parámetros y haz clic en "Iniciar Productor"
           - El productor generará datos sintéticos y los enviará a Kafka
        
        4. **Iniciar Consumer** - Haz clic en "Iniciar Consumer" para empezar a recibir datos
           - Los contadores empezarán a incrementarse
        
        5. **Ver Dashboards** - Navega a cualquier página de streaming para ver los datos en vivo
        
        ### 🔧 Solución de Problemas
        
        **Kafka no disponible:**
        ```bash
        cd infra
        docker-compose up -d zookeeper kafka
        ```
        
        **Topics no creados:**
        - Haz clic en "Crear Topics" en esta página
        
        **Sin datos en dashboards:**
        - Verifica que el productor esté activo (indicador verde)
        - Verifica que el consumer esté activo
        - Revisa los contadores de eventos
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888;'>
        🌊 <strong>Streaming Hub</strong> | climaXtreme Dashboard | 
        Apache Kafka Streaming en Tiempo Real
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
