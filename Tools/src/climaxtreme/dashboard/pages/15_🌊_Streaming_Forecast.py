"""
🌊 Real-Time Streaming Forecast Dashboard Page

This page provides REAL-TIME weather visualization using Apache Kafka:
- Live weather data streaming from Kafka topics
- Real-time temperature and weather charts
- Continuous updates without page refresh
- Integration with the Big Data Kafka pipeline
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time
import numpy as np
import sys
from pathlib import Path

# Page config - MUST be first
st.set_page_config(
    page_title="Streaming Forecast - climaXtreme",
    page_icon="🌊",
    layout="wide"
)

# Imports
try:
    from climaxtreme.dashboard.components.data_checker import show_hdfs_connection_status
    from climaxtreme.dashboard.components.kafka_realtime import (
        get_kafka_state,
        init_realtime_stream,
        check_kafka_available,
        create_realtime_weather_chart,
        create_realtime_map,
        create_time_series_chart,
        create_city_comparison_chart,
        TOPICS
    )
    from climaxtreme.dashboard.components.kafka_manager import (
        get_streaming_manager,
        render_kafka_cluster_status,
        render_producer_controls,
        KafkaStreamingConfig
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from climaxtreme.dashboard.components.data_checker import show_hdfs_connection_status
    from climaxtreme.dashboard.components.kafka_realtime import (
        get_kafka_state,
        init_realtime_stream,
        check_kafka_available,
        create_realtime_weather_chart,
        create_realtime_map,
        create_time_series_chart,
        create_city_comparison_chart,
        TOPICS
    )
    from climaxtreme.dashboard.components.kafka_manager import (
        get_streaming_manager,
        render_kafka_cluster_status,
        render_producer_controls,
        KafkaStreamingConfig
    )

# Show HDFS status in sidebar
show_hdfs_connection_status()

# Session state
if 'auto_refresh_forecast' not in st.session_state:
    st.session_state.auto_refresh_forecast = 2
if 'forecast_view' not in st.session_state:
    st.session_state.forecast_view = 'live'


# ============================================================================
# Header
# ============================================================================

st.title("🌊 Pronóstico en Tiempo Real (Streaming)")

kafka_state = get_kafka_state()

col_header1, col_header2 = st.columns([4, 1])

with col_header1:
    st.markdown("""
    Visualiza pronósticos meteorológicos **en tiempo real** usando Apache Kafka.
    Los datos fluyen continuamente desde el productor de streaming hacia esta visualización.
    """)

with col_header2:
    if kafka_state.is_running():
        stats = kafka_state.get_stats()
        st.markdown(f"""
        <div style='text-align: center; padding: 8px; background: #2E7D32; 
                    border-radius: 8px; color: white;'>
            <strong>🔴 LIVE</strong><br>
            <small>{stats['total_weather']} eventos</small>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='text-align: center; padding: 8px; background: #424242; 
                    border-radius: 8px; color: white;'>
            <strong>⏸️ OFFLINE</strong>
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# Sidebar - Controls
# ============================================================================

st.sidebar.header("⚙️ Control del Streaming")

# Kafka status
kafka_check = check_kafka_available()
if kafka_check.get('available'):
    st.sidebar.success("✅ Kafka disponible")
else:
    st.sidebar.error("❌ Kafka no disponible")
    st.sidebar.caption(kafka_check.get('error', ''))

st.sidebar.markdown("---")

# Stream controls
if kafka_state.is_running():
    if st.sidebar.button("⏹️ Detener Stream", type="secondary", use_container_width=True):
        kafka_state.stop()
        st.rerun()
else:
    if st.sidebar.button("▶️ Iniciar Stream", type="primary", use_container_width=True):
        topics = [TOPICS['weather'], TOPICS['alerts'], TOPICS['predictions']]
        if kafka_state.start(topics):
            st.rerun()
        else:
            st.sidebar.error("Error al conectar")

st.sidebar.markdown("---")

# Auto-refresh
st.sidebar.subheader("🔄 Auto-Refresh")
st.session_state.auto_refresh_forecast = st.sidebar.select_slider(
    "Intervalo",
    options=[0, 1, 2, 3, 5, 10],
    value=2,
    format_func=lambda x: "Off" if x == 0 else f"{x}s"
)

if st.sidebar.button("🔄 Refrescar Ahora", use_container_width=True):
    st.rerun()

st.sidebar.markdown("---")

# View selector
st.sidebar.subheader("📊 Vista")
st.session_state.forecast_view = st.sidebar.radio(
    "Modo",
    options=['live', 'analysis', 'map'],
    format_func=lambda x: {
        'live': '📈 Tiempo Real',
        'analysis': '📊 Análisis',
        'map': '🗺️ Mapa'
    }.get(x, x)
)

# Stats
if kafka_state.is_running():
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Stats")
    stats = kafka_state.get_stats()
    st.sidebar.metric("Eventos Clima", stats['weather_buffered'])
    st.sidebar.metric("Alertas", stats['alerts_buffered'])


# ============================================================================
# Main Content
# ============================================================================

st.markdown("---")

if kafka_state.is_running():
    # Obtener datos en vivo
    weather_events = kafka_state.get_weather_events(500)
    alert_events = kafka_state.get_alert_events(50)
    stats = kafka_state.get_stats()
    
    # Métricas principales
    if weather_events:
        temps = [e.get('temperature_c', 20) for e in weather_events if 'temperature_c' in e]
        winds = [e.get('wind_speed_kmh', 0) for e in weather_events if 'wind_speed_kmh' in e]
        humidities = [e.get('humidity_pct', 50) for e in weather_events if 'humidity_pct' in e]
        
        m1, m2, m3, m4, m5 = st.columns(5)
        
        with m1:
            if temps:
                delta = temps[-1] - temps[0] if len(temps) > 1 else 0
                st.metric("🌡️ Temp Actual", f"{temps[-1]:.1f}°C", f"{delta:+.1f}°C")
        with m2:
            if temps:
                st.metric("📊 Temp Media", f"{np.mean(temps):.1f}°C")
        with m3:
            if winds:
                st.metric("💨 Viento", f"{np.mean(winds):.1f} km/h")
        with m4:
            if humidities:
                st.metric("💧 Humedad", f"{np.mean(humidities):.0f}%")
        with m5:
            st.metric("📍 Ciudades", len(set(e.get('city', '') for e in weather_events)))
    
    st.markdown("---")
    
    # Contenido basado en vista seleccionada
    if st.session_state.forecast_view == 'live':
        # Vista en tiempo real
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📈 Temperatura en Tiempo Real")
            temp_chart = create_realtime_weather_chart(weather_events)
            if temp_chart:
                st.plotly_chart(temp_chart, use_container_width=True, key="forecast_temp")
            else:
                st.info("⏳ Esperando datos...")
        
        with col2:
            st.subheader("🚨 Alertas Recientes")
            if alert_events:
                for alert in alert_events[:5]:
                    level = alert.get('alert_level', 'WATCH')
                    colors = {
                        'EMERGENCY': ('🔴', '#FADBD8'),
                        'WARNING': ('🟠', '#FDEBD0'),
                        'WATCH': ('🟡', '#FEF9E7')
                    }
                    icon, bg = colors.get(level, ('⚪', '#ecf0f1'))
                    
                    st.markdown(f"""
                    <div style='background:{bg}; padding:8px; border-radius:5px; margin-bottom:5px;'>
                        {icon} <strong>{level}</strong> - {alert.get('city', 'Unknown')}<br>
                        <small>{alert.get('alert_type', 'WEATHER')}</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("✅ Sin alertas activas")
        
        # Segunda fila - Series temporales
        st.markdown("---")
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("💨 Velocidad del Viento")
            wind_chart = create_time_series_chart(weather_events, 'wind_speed_kmh')
            if wind_chart:
                st.plotly_chart(wind_chart, use_container_width=True, key="forecast_wind")
        
        with col4:
            st.subheader("💧 Humedad Relativa")
            hum_chart = create_time_series_chart(weather_events, 'humidity_pct')
            if hum_chart:
                st.plotly_chart(hum_chart, use_container_width=True, key="forecast_hum")
    
    elif st.session_state.forecast_view == 'analysis':
        # Vista de análisis
        st.subheader("📊 Análisis de Pronósticos en Tiempo Real")
        
        if weather_events:
            # Convertir a DataFrame
            df = pd.DataFrame(weather_events)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Comparación por ciudad
                city_chart = create_city_comparison_chart(weather_events, top_n=8)
                if city_chart:
                    st.plotly_chart(city_chart, use_container_width=True, key="city_comp")
                
                # Distribución de temperatura
                if 'temperature_c' in df.columns:
                    fig_hist = px.histogram(
                        df, x='temperature_c', nbins=30,
                        title="📊 Distribución de Temperaturas",
                        color_discrete_sequence=['#e74c3c']
                    )
                    fig_hist.update_layout(height=300)
                    st.plotly_chart(fig_hist, use_container_width=True, key="temp_hist")
            
            with col2:
                # Scatter de temperatura vs humedad
                if 'temperature_c' in df.columns and 'humidity_pct' in df.columns:
                    fig_scatter = px.scatter(
                        df.tail(200), 
                        x='temperature_c', 
                        y='humidity_pct',
                        color='city' if 'city' in df.columns else None,
                        title="🌡️ Temperatura vs Humedad",
                        opacity=0.6
                    )
                    fig_scatter.update_layout(height=300, showlegend=False)
                    st.plotly_chart(fig_scatter, use_container_width=True, key="scatter")
                
                # Box plot por clima
                if 'temperature_c' in df.columns:
                    cities = df['city'].unique()[:5] if 'city' in df.columns else []
                    if len(cities) > 0:
                        fig_box = px.box(
                            df[df['city'].isin(cities)],
                            x='city', y='temperature_c',
                            title="📦 Box Plot por Ciudad",
                            color='city'
                        )
                        fig_box.update_layout(height=300, showlegend=False)
                        st.plotly_chart(fig_box, use_container_width=True, key="boxplot")
            
            # Estadísticas
            st.markdown("### 📈 Estadísticas en Tiempo Real")
            
            col_s1, col_s2, col_s3 = st.columns(3)
            
            with col_s1:
                st.markdown("**Temperatura (°C)**")
                if 'temperature_c' in df.columns:
                    temps = df['temperature_c'].dropna()
                    st.write(f"- Media: {temps.mean():.2f}")
                    st.write(f"- Std: {temps.std():.2f}")
                    st.write(f"- Min: {temps.min():.2f}")
                    st.write(f"- Max: {temps.max():.2f}")
            
            with col_s2:
                st.markdown("**Viento (km/h)**")
                if 'wind_speed_kmh' in df.columns:
                    winds = df['wind_speed_kmh'].dropna()
                    st.write(f"- Media: {winds.mean():.2f}")
                    st.write(f"- Std: {winds.std():.2f}")
                    st.write(f"- Min: {winds.min():.2f}")
                    st.write(f"- Max: {winds.max():.2f}")
            
            with col_s3:
                st.markdown("**Humedad (%)**")
                if 'humidity_pct' in df.columns:
                    hums = df['humidity_pct'].dropna()
                    st.write(f"- Media: {hums.mean():.1f}")
                    st.write(f"- Std: {hums.std():.1f}")
                    st.write(f"- Min: {hums.min():.1f}")
                    st.write(f"- Max: {hums.max():.1f}")
    
    elif st.session_state.forecast_view == 'map':
        # Vista de mapa
        st.subheader("🗺️ Mapa de Pronósticos en Tiempo Real")
        
        map_fig = create_realtime_map(weather_events)
        if map_fig:
            st.plotly_chart(map_fig, use_container_width=True, key="forecast_map")
        
        # Info debajo del mapa
        if weather_events:
            col1, col2, col3 = st.columns(3)
            
            temps = [e.get('temperature_c', 0) for e in weather_events if 'temperature_c' in e]
            cities = list(set(e.get('city', '') for e in weather_events if e.get('city')))
            
            with col1:
                st.metric("🏙️ Ciudades Monitoreadas", len(cities))
            with col2:
                if temps:
                    st.metric("🔥 Temperatura Máxima", f"{max(temps):.1f}°C")
            with col3:
                if temps:
                    st.metric("❄️ Temperatura Mínima", f"{min(temps):.1f}°C")
    
    # Footer con timestamp
    st.markdown("---")
    st.caption(f"🕐 Última actualización: {datetime.now().strftime('%H:%M:%S')} | "
               f"📊 {stats['weather_buffered']} eventos en buffer | "
               f"🚨 {stats['alerts_buffered']} alertas")
    
    # Auto-refresh
    if st.session_state.auto_refresh_forecast > 0:
        time.sleep(st.session_state.auto_refresh_forecast)
        st.rerun()

else:
    # No conectado - mostrar instrucciones
    st.warning("⚠️ No hay conexión activa a Kafka")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🚀 Cómo iniciar el streaming:
        
        **1. Inicia Kafka (si no está corriendo):**
        ```bash
        cd infra
        docker-compose up -d zookeeper kafka
        ```
        
        **2. Inicia el productor de datos:**
        ```bash
        docker exec climaxtreme-processor python -c "
        from climaxtreme.streaming import KafkaStreamingProducer
        producer = KafkaStreamingProducer(
            n_cities=50,
            interval_seconds=1.0,
            include_alerts=True
        )
        producer.start_streaming()
        import time
        time.sleep(600)  # 10 minutos
        producer.stop_streaming()
        "
        ```
        
        **3. Haz clic en "▶️ Iniciar Stream" en el sidebar**
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Preview (datos simulados)
        
        Mientras tanto, aquí hay una preview con datos de ejemplo:
        """)
        
        # Generar datos de demo
        demo_temps = [20 + np.sin(i/5) * 10 + np.random.randn() * 2 for i in range(50)]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=demo_temps, 
            mode='lines+markers',
            line=dict(color='#e74c3c'),
            name='Temperatura (Demo)'
        ))
        fig.update_layout(
            title="📊 Datos de Demo",
            height=300,
            xaxis_title="Eventos",
            yaxis_title="Temperatura (°C)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("""
    ### 🏗️ Arquitectura del Streaming
    
    ```
    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │   Kafka         │────▶│   Dashboard     │────▶│  Visualización  │
    │   Producer      │     │   Consumer      │     │  Tiempo Real    │
    │   (Spark)       │     │   (Python)      │     │  (Plotly)       │
    └─────────────────┘     └─────────────────┘     └─────────────────┘
           │                                               
           ▼                                               
    ┌─────────────────┐                                    
    │   Kafka Topics  │                                    
    │   - weather     │                                    
    │   - alerts      │                                    
    │   - storms      │                                    
    └─────────────────┘                                    
    ```
    
    Los datos fluyen continuamente desde el productor (que genera datos sintéticos basados 
    en patrones históricos) a través de Apache Kafka hacia este dashboard, donde los 
    gráficos se actualizan automáticamente.
    """)
