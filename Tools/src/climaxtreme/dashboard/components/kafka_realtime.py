"""
Kafka Realtime Dashboard Component - Componente para consumo en tiempo real de Kafka.

Este módulo proporciona:
- Consumidor de Kafka para el dashboard de Streamlit
- Buffer de eventos para visualización
- Auto-refresh de componentes
- Integración con gráficos Plotly en tiempo real
"""

import os
import json
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from collections import deque
import streamlit as st

logger = logging.getLogger(__name__)

# ============================================================================
# Configuración
# ============================================================================

KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'climaxtreme-kafka:9092')

TOPICS = {
    'weather': 'climaxtreme-weather',
    'alerts': 'climaxtreme-alerts',
    'storms': 'climaxtreme-storms',
    'predictions': 'climaxtreme-predictions',
    'progress': 'climaxtreme-progress'
}


# ============================================================================
# Estado Global del Consumer (Singleton para Streamlit)
# ============================================================================

class KafkaStreamState:
    """
    Estado singleton del stream de Kafka para Streamlit.
    
    Mantiene el estado entre reruns de Streamlit y proporciona
    acceso thread-safe a los eventos.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self._consumer = None
        self._consumer_thread = None
        self._running = False
        
        # Buffers de eventos (thread-safe con deque)
        self._max_buffer_size = 1000
        self._weather_events = deque(maxlen=self._max_buffer_size)
        self._alert_events = deque(maxlen=self._max_buffer_size)
        self._storm_events = deque(maxlen=self._max_buffer_size)
        
        # Estadísticas
        self._stats = {
            'total_weather': 0,
            'total_alerts': 0,
            'total_storms': 0,
            'last_event_time': None,
            'start_time': None,
            'errors': 0
        }
        
        # Lock para acceso thread-safe
        self._data_lock = threading.Lock()
    
    def is_running(self) -> bool:
        return self._running
    
    def start(self, topics: List[str] = None) -> bool:
        """Iniciar consumo de Kafka."""
        if self._running:
            return True
        
        topics = topics or [TOPICS['weather'], TOPICS['alerts']]
        
        try:
            from kafka import KafkaConsumer
            
            self._consumer = KafkaConsumer(
                *topics,
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS.split(','),
                group_id='climaxtreme-dashboard-realtime',
                auto_offset_reset='latest',
                enable_auto_commit=True,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                consumer_timeout_ms=1000
            )
            
            self._running = True
            self._stats['start_time'] = datetime.now().isoformat()
            
            # Iniciar thread de consumo
            self._consumer_thread = threading.Thread(target=self._consume_loop, daemon=True)
            self._consumer_thread.start()
            
            logger.info(f"Kafka stream started, consuming from: {topics}")
            return True
            
        except ImportError:
            logger.error("kafka-python not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to start Kafka consumer: {e}")
            self._stats['errors'] += 1
            return False
    
    def stop(self):
        """Detener consumo de Kafka."""
        self._running = False
        if self._consumer_thread:
            self._consumer_thread.join(timeout=2)
        if self._consumer:
            self._consumer.close()
            self._consumer = None
        logger.info("Kafka stream stopped")
    
    def _consume_loop(self):
        """Loop de consumo en background."""
        while self._running:
            try:
                # Poll con timeout corto para poder verificar _running
                for message in self._consumer:
                    if not self._running:
                        break
                    
                    self._process_message(message)
                    
            except Exception as e:
                if self._running:  # Solo loguear si no es por shutdown
                    logger.error(f"Error in consume loop: {e}")
                    self._stats['errors'] += 1
                time.sleep(0.5)
    
    def _process_message(self, message):
        """Procesar mensaje recibido."""
        try:
            topic = message.topic
            value = message.value
            
            with self._data_lock:
                if topic == TOPICS['weather']:
                    self._weather_events.append(value)
                    self._stats['total_weather'] += 1
                elif topic == TOPICS['alerts']:
                    self._alert_events.append(value)
                    self._stats['total_alerts'] += 1
                elif topic == TOPICS['storms']:
                    self._storm_events.append(value)
                    self._stats['total_storms'] += 1
                
                self._stats['last_event_time'] = datetime.now().isoformat()
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self._stats['errors'] += 1
    
    def get_weather_events(self, n: int = 100) -> List[Dict]:
        """Obtener últimos eventos de clima."""
        with self._data_lock:
            return list(self._weather_events)[-n:]
    
    def get_alert_events(self, n: int = 100) -> List[Dict]:
        """Obtener últimos eventos de alerta."""
        with self._data_lock:
            return list(self._alert_events)[-n:]
    
    def get_storm_events(self, n: int = 100) -> List[Dict]:
        """Obtener últimos eventos de tormenta."""
        with self._data_lock:
            return list(self._storm_events)[-n:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        with self._data_lock:
            return {
                **self._stats,
                'is_running': self._running,
                'weather_buffered': len(self._weather_events),
                'alerts_buffered': len(self._alert_events),
                'storms_buffered': len(self._storm_events)
            }
    
    def clear_buffers(self):
        """Limpiar buffers."""
        with self._data_lock:
            self._weather_events.clear()
            self._alert_events.clear()
            self._storm_events.clear()


# Instancia global
_kafka_state = None

def get_kafka_state() -> KafkaStreamState:
    """Obtener instancia del estado de Kafka."""
    global _kafka_state
    if _kafka_state is None:
        _kafka_state = KafkaStreamState()
    return _kafka_state


# ============================================================================
# Componentes de Streamlit para Tiempo Real
# ============================================================================

def init_realtime_stream(
    topics: List[str] = None,
    auto_start: bool = True
) -> KafkaStreamState:
    """
    Inicializar stream de Kafka para la página actual.
    
    Args:
        topics: Lista de tópicos a consumir
        auto_start: Iniciar automáticamente si no está corriendo
        
    Returns:
        Estado del stream
    """
    state = get_kafka_state()
    
    if auto_start and not state.is_running():
        state.start(topics)
    
    return state


def render_kafka_status_card():
    """Renderizar tarjeta de estado de conexión Kafka."""
    state = get_kafka_state()
    stats = state.get_stats()
    
    if stats['is_running']:
        st.success("🟢 **Kafka Conectado** - Streaming activo")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Eventos Clima", stats['total_weather'])
        with col2:
            st.metric("🚨 Alertas", stats['total_alerts'])
        with col3:
            st.metric("🌀 Tormentas", stats['total_storms'])
        with col4:
            st.metric("⚠️ Errores", stats['errors'])
            
        if stats['last_event_time']:
            st.caption(f"Último evento: {stats['last_event_time']}")
    else:
        st.warning("🔴 **Kafka Desconectado**")
        if st.button("🔌 Conectar a Kafka", key="connect_kafka"):
            if state.start():
                st.rerun()
            else:
                st.error("No se pudo conectar a Kafka")


def render_realtime_controls():
    """Renderizar controles de streaming en tiempo real."""
    state = get_kafka_state()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if state.is_running():
            if st.button("⏹️ Detener Stream", key="stop_stream"):
                state.stop()
                st.rerun()
        else:
            if st.button("▶️ Iniciar Stream", key="start_stream", type="primary"):
                state.start()
                st.rerun()
    
    with col2:
        if st.button("🗑️ Limpiar Buffer", key="clear_buffer"):
            state.clear_buffers()
            st.rerun()
    
    with col3:
        refresh_rate = st.selectbox(
            "🔄 Auto-refresh",
            options=[0, 1, 2, 5, 10],
            format_func=lambda x: "Desactivado" if x == 0 else f"{x} segundos",
            key="refresh_rate"
        )
        
        return refresh_rate


def create_realtime_weather_chart(events: List[Dict], max_points: int = 100):
    """
    Crear gráfico de temperatura en tiempo real.
    
    Args:
        events: Lista de eventos de clima
        max_points: Máximo de puntos a mostrar
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    if not events:
        st.info("Esperando datos de clima...")
        return None
    
    # Preparar datos
    recent_events = events[-max_points:]
    
    timestamps = []
    temperatures = []
    cities = []
    
    for e in recent_events:
        try:
            timestamps.append(e.get('timestamp', ''))
            temperatures.append(e.get('temperature_c', 0))
            cities.append(e.get('city', 'Unknown'))
        except:
            continue
    
    if not timestamps:
        return None
    
    # Crear figura con subplots
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=('🌡️ Temperatura en Tiempo Real', '📊 Distribución por Ciudad'),
        vertical_spacing=0.15
    )
    
    # Gráfico de línea de temperatura
    fig.add_trace(
        go.Scatter(
            x=list(range(len(temperatures))),
            y=temperatures,
            mode='lines+markers',
            name='Temperatura',
            line=dict(color='#e74c3c', width=2),
            marker=dict(size=4),
            hovertemplate='%{y:.1f}°C<br>%{text}<extra></extra>',
            text=cities
        ),
        row=1, col=1
    )
    
    # Agregar líneas de referencia
    fig.add_hline(y=35, line_dash="dash", line_color="orange", 
                  annotation_text="Alerta Calor", row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="blue",
                  annotation_text="Congelación", row=1, col=1)
    
    # Histograma de temperaturas
    fig.add_trace(
        go.Histogram(
            x=temperatures,
            nbinsx=20,
            name='Distribución',
            marker_color='#3498db'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=500,
        showlegend=False,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    fig.update_xaxes(title_text="Eventos recientes", row=1, col=1)
    fig.update_yaxes(title_text="Temperatura (°C)", row=1, col=1)
    fig.update_xaxes(title_text="Temperatura (°C)", row=2, col=1)
    fig.update_yaxes(title_text="Frecuencia", row=2, col=1)
    
    return fig


def create_realtime_map(events: List[Dict], max_points: int = 200):
    """
    Crear mapa de eventos en tiempo real.
    
    Args:
        events: Lista de eventos
        max_points: Máximo de puntos
    """
    import plotly.express as px
    import pandas as pd
    
    if not events:
        return None
    
    recent_events = events[-max_points:]
    
    # Convertir a DataFrame
    data = []
    for e in recent_events:
        try:
            data.append({
                'city': e.get('city', 'Unknown'),
                'lat': e.get('latitude', 0),
                'lon': e.get('longitude', 0),
                'temp': e.get('temperature_c', 20),
                'timestamp': e.get('timestamp', '')
            })
        except:
            continue
    
    if not data:
        return None
    
    df = pd.DataFrame(data)
    
    # Agregar por ciudad (último valor)
    df_latest = df.groupby('city').last().reset_index()
    
    fig = px.scatter_geo(
        df_latest,
        lat='lat',
        lon='lon',
        color='temp',
        size=[10] * len(df_latest),
        hover_name='city',
        color_continuous_scale='RdYlBu_r',
        projection='natural earth',
        title='🗺️ Mapa de Temperaturas en Tiempo Real'
    )
    
    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
        geo=dict(
            showland=True,
            landcolor='rgb(243, 243, 243)',
            showocean=True,
            oceancolor='rgb(210, 235, 255)'
        )
    )
    
    return fig


def create_realtime_alerts_panel(alerts: List[Dict], max_alerts: int = 10):
    """
    Crear panel de alertas en tiempo real.
    
    Args:
        alerts: Lista de alertas
        max_alerts: Máximo de alertas a mostrar
    """
    if not alerts:
        st.info("✅ Sin alertas activas")
        return
    
    recent_alerts = alerts[-max_alerts:][::-1]  # Más recientes primero
    
    alert_styles = {
        'EMERGENCY': ('🔴', '#e74c3c', '#FADBD8'),
        'WARNING': ('🟠', '#e67e22', '#FDEBD0'),
        'WATCH': ('🟡', '#f1c40f', '#FEF9E7')
    }
    
    for alert in recent_alerts:
        level = alert.get('alert_level', 'WATCH')
        alert_type = alert.get('alert_type', 'WEATHER')
        city = alert.get('city', 'Unknown')
        desc = alert.get('description', '')
        timestamp = alert.get('timestamp', '')
        
        icon, color, bg = alert_styles.get(level, ('⚪', '#95a5a6', '#ecf0f1'))
        
        st.markdown(f"""
        <div style='background-color:{bg}; padding:10px; border-radius:5px; 
                    margin-bottom:5px; border-left:4px solid {color};'>
            <strong>{icon} {level}</strong> - {alert_type}<br>
            <small>📍 {city}</small><br>
            <small>{desc}</small><br>
            <small style='color:#666;'>🕐 {timestamp}</small>
        </div>
        """, unsafe_allow_html=True)


def create_realtime_metrics_row(events: List[Dict]):
    """
    Crear fila de métricas en tiempo real.
    
    Args:
        events: Lista de eventos de clima
    """
    if not events:
        return
    
    # Calcular estadísticas de los últimos eventos
    recent = events[-100:]
    
    temps = [e.get('temperature_c', 0) for e in recent if 'temperature_c' in e]
    winds = [e.get('wind_speed_kmh', 0) for e in recent if 'wind_speed_kmh' in e]
    humidities = [e.get('humidity_pct', 0) for e in recent if 'humidity_pct' in e]
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if temps:
            avg_temp = sum(temps) / len(temps)
            delta = temps[-1] - temps[0] if len(temps) > 1 else 0
            st.metric("🌡️ Temp Media", f"{avg_temp:.1f}°C", f"{delta:+.1f}°C")
    
    with col2:
        if temps:
            st.metric("🔥 Temp Máx", f"{max(temps):.1f}°C")
    
    with col3:
        if temps:
            st.metric("❄️ Temp Mín", f"{min(temps):.1f}°C")
    
    with col4:
        if winds:
            avg_wind = sum(winds) / len(winds)
            st.metric("💨 Viento Medio", f"{avg_wind:.1f} km/h")
    
    with col5:
        if humidities:
            avg_hum = sum(humidities) / len(humidities)
            st.metric("💧 Humedad Media", f"{avg_hum:.0f}%")


def create_city_comparison_chart(events: List[Dict], top_n: int = 10):
    """
    Crear gráfico de comparación entre ciudades.
    
    Args:
        events: Lista de eventos
        top_n: Top N ciudades a mostrar
    """
    import plotly.express as px
    import pandas as pd
    from collections import defaultdict
    
    if not events:
        return None
    
    # Agregar por ciudad
    city_data = defaultdict(list)
    for e in events:
        city = e.get('city', 'Unknown')
        temp = e.get('temperature_c')
        if temp is not None:
            city_data[city].append(temp)
    
    if not city_data:
        return None
    
    # Calcular estadísticas por ciudad
    stats = []
    for city, temps in city_data.items():
        if temps:
            stats.append({
                'city': city,
                'temp_avg': sum(temps) / len(temps),
                'temp_min': min(temps),
                'temp_max': max(temps),
                'count': len(temps)
            })
    
    if not stats:
        return None
    
    # Ordenar por cantidad de eventos
    stats.sort(key=lambda x: x['count'], reverse=True)
    top_stats = stats[:top_n]
    
    df = pd.DataFrame(top_stats)
    
    fig = px.bar(
        df,
        x='city',
        y='temp_avg',
        error_y=df['temp_max'] - df['temp_avg'],
        error_y_minus=df['temp_avg'] - df['temp_min'],
        color='temp_avg',
        color_continuous_scale='RdYlBu_r',
        title=f'🏙️ Comparación de Temperaturas (Top {top_n} ciudades)',
        labels={'temp_avg': 'Temperatura Media (°C)', 'city': 'Ciudad'}
    )
    
    fig.update_layout(
        height=350,
        showlegend=False,
        xaxis_tickangle=-45
    )
    
    return fig


def create_time_series_chart(events: List[Dict], variable: str = 'temperature_c'):
    """
    Crear gráfico de serie temporal.
    
    Args:
        events: Lista de eventos
        variable: Variable a graficar
    """
    import plotly.graph_objects as go
    
    if not events:
        return None
    
    # Extraer datos
    values = []
    timestamps = []
    
    for i, e in enumerate(events[-200:]):
        val = e.get(variable)
        if val is not None:
            values.append(val)
            timestamps.append(i)
    
    if not values:
        return None
    
    # Calcular media móvil
    window = min(10, len(values) // 5) if len(values) > 10 else 1
    moving_avg = []
    for i in range(len(values)):
        start = max(0, i - window)
        moving_avg.append(sum(values[start:i+1]) / (i - start + 1))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=values,
        mode='lines',
        name='Valores',
        line=dict(color='rgba(52, 152, 219, 0.5)', width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=moving_avg,
        mode='lines',
        name='Media Móvil',
        line=dict(color='#e74c3c', width=2)
    ))
    
    var_labels = {
        'temperature_c': 'Temperatura (°C)',
        'humidity_pct': 'Humedad (%)',
        'wind_speed_kmh': 'Viento (km/h)',
        'pressure_hpa': 'Presión (hPa)',
        'rain_mm': 'Lluvia (mm)'
    }
    
    fig.update_layout(
        title=f'📈 Serie Temporal: {var_labels.get(variable, variable)}',
        xaxis_title='Eventos',
        yaxis_title=var_labels.get(variable, variable),
        height=300,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig


# ============================================================================
# Auto-refresh para Streamlit
# ============================================================================

def setup_auto_refresh(interval_seconds: int):
    """
    Configurar auto-refresh de la página.
    
    Args:
        interval_seconds: Intervalo en segundos (0 para desactivar)
    """
    if interval_seconds > 0:
        st.markdown(f"""
        <script>
            setTimeout(function() {{
                window.location.reload();
            }}, {interval_seconds * 1000});
        </script>
        """, unsafe_allow_html=True)
        
        # También usar el método nativo de Streamlit si está disponible
        try:
            import time
            time.sleep(interval_seconds)
            st.rerun()
        except:
            pass


def auto_refresh_component(interval_seconds: int = 2):
    """
    Componente de auto-refresh usando st.empty().
    
    Args:
        interval_seconds: Intervalo de refresh
    """
    import time
    
    if interval_seconds <= 0:
        return
    
    # Usar session state para tracking
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()
    
    current_time = time.time()
    if current_time - st.session_state.last_refresh >= interval_seconds:
        st.session_state.last_refresh = current_time
        st.rerun()


# ============================================================================
# Verificación de conexión Kafka
# ============================================================================

def check_kafka_available() -> Dict[str, Any]:
    """
    Verificar si Kafka está disponible.
    
    Returns:
        Estado de la conexión
    """
    try:
        from kafka import KafkaConsumer
        from kafka.errors import NoBrokersAvailable
        
        consumer = KafkaConsumer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS.split(','),
            consumer_timeout_ms=3000
        )
        
        topics = list(consumer.topics())
        consumer.close()
        
        climaxtreme_topics = [t for t in topics if 'climaxtreme' in t]
        
        return {
            'available': True,
            'bootstrap_servers': KAFKA_BOOTSTRAP_SERVERS,
            'total_topics': len(topics),
            'climaxtreme_topics': climaxtreme_topics,
            'topics_ready': len(climaxtreme_topics) >= 2
        }
        
    except ImportError:
        return {
            'available': False,
            'error': 'kafka-python no instalado'
        }
    except Exception as e:
        return {
            'available': False,
            'error': str(e)
        }


def render_kafka_setup_guide():
    """Renderizar guía de configuración de Kafka."""
    st.markdown("""
    ### 🔧 Configuración de Kafka
    
    Para usar el streaming en tiempo real, sigue estos pasos:
    
    **1. Iniciar el clúster de Kafka:**
    ```bash
    cd infra
    docker-compose up -d zookeeper kafka
    ```
    
    **2. Crear los tópicos (automático con auto.create.topics.enable=true):**
    ```bash
    docker exec climaxtreme-kafka kafka-topics --create \\
        --bootstrap-server localhost:9092 \\
        --topic climaxtreme-weather \\
        --partitions 3
    ```
    
    **3. Iniciar el producer de streaming:**
    ```bash
    docker exec climaxtreme-processor python -c "
    from climaxtreme.streaming import KafkaStreamingProducer
    producer = KafkaStreamingProducer(n_cities=50, interval_seconds=1)
    producer.start_streaming()
    import time
    time.sleep(300)  # 5 minutos
    producer.stop_streaming()
    "
    ```
    
    **4. Ver UI de Kafka (opcional):**
    ```bash
    docker-compose --profile monitoring up -d kafka-ui
    # Acceder a http://localhost:8080
    ```
    """)
