"""
StreamingConfigUI - Componentes de configuración de streaming para Streamlit.

Proporciona widgets intuitivos para configurar la generación de datos sintéticos:
- Configuración de datos (ciudades, horas, resolución)
- Parámetros de eventos (probabilidades, umbrales)
- Configuración avanzada (seed, distribuciones)
- Presets predefinidos
"""

import streamlit as st
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
from enum import Enum
import json


# ============================================================================
# Configuración de Streaming
# ============================================================================

class StreamingPreset(Enum):
    """Presets predefinidos para generación."""
    DEMO_RAPIDO = "demo_rapido"
    VISUALIZACION = "visualizacion"
    ESTADISTICAS = "estadisticas"
    ML_TRAINING = "ml_training"
    STRESS_TEST = "stress_test"
    PERSONALIZADO = "personalizado"


@dataclass
class StreamingConfig:
    """Configuración completa para generación de datos streaming."""
    
    # === Datos básicos ===
    n_cities: int = 50
    forecast_hours: int = 168  # 1 semana
    resolution_minutes: int = 60  # Horario
    batch_size: int = 1000
    
    # === Rango temporal ===
    start_year: int = 2020
    end_year: int = 2024
    
    # === Probabilidades de eventos ===
    rain_probability: float = 0.25
    storm_probability: float = 0.05
    extreme_event_probability: float = 0.02
    
    # === Umbrales de alertas ===
    heat_yellow_threshold: float = 35.0
    heat_orange_threshold: float = 40.0
    heat_red_threshold: float = 45.0
    cold_yellow_threshold: float = 0.0
    cold_orange_threshold: float = -10.0
    cold_red_threshold: float = -20.0
    wind_yellow_threshold: float = 60.0
    wind_orange_threshold: float = 90.0
    wind_red_threshold: float = 120.0
    
    # === Parámetros de distribución ===
    rain_gamma_scale: float = 5.0
    wind_weibull_shape: float = 2.0
    wind_weibull_scale: float = 15.0
    humidity_mean: float = 65.0
    humidity_std: float = 15.0
    pressure_mean: float = 1013.25
    pressure_std: float = 10.0
    
    # === Zonas climáticas (pesos) ===
    tropical_weight: float = 0.20
    subtropical_weight: float = 0.25
    temperate_weight: float = 0.30
    continental_weight: float = 0.20
    polar_weight: float = 0.05
    
    # === Configuración avanzada ===
    seed: int = 42
    include_storms: bool = True
    include_alerts: bool = True
    include_events: bool = True
    
    # === Streaming específico ===
    trigger_interval_seconds: int = 10
    micro_batch_size: int = 100
    continuous_mode: bool = False
    max_duration_minutes: int = 30
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StreamingConfig':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'StreamingConfig':
        return cls.from_dict(json.loads(json_str))


# ============================================================================
# Presets Predefinidos
# ============================================================================

PRESETS: Dict[StreamingPreset, StreamingConfig] = {
    StreamingPreset.DEMO_RAPIDO: StreamingConfig(
        n_cities=10,
        forecast_hours=48,
        batch_size=500,
        include_storms=False,
        include_events=False,
        max_duration_minutes=5
    ),
    StreamingPreset.VISUALIZACION: StreamingConfig(
        n_cities=30,
        forecast_hours=168,
        batch_size=1000,
        include_storms=True,
        include_alerts=True
    ),
    StreamingPreset.ESTADISTICAS: StreamingConfig(
        n_cities=50,
        forecast_hours=720,  # 1 mes
        batch_size=2000,
        include_storms=True,
        include_alerts=True,
        include_events=True
    ),
    StreamingPreset.ML_TRAINING: StreamingConfig(
        n_cities=100,
        forecast_hours=2160,  # 3 meses
        batch_size=5000,
        start_year=2015,
        end_year=2024,
        include_storms=True,
        include_alerts=True,
        include_events=True
    ),
    StreamingPreset.STRESS_TEST: StreamingConfig(
        n_cities=500,
        forecast_hours=8760,  # 1 año
        batch_size=10000,
        start_year=2010,
        end_year=2024,
        include_storms=True,
        include_alerts=True,
        include_events=True,
        continuous_mode=True,
        max_duration_minutes=60
    )
}


# ============================================================================
# Componentes UI
# ============================================================================

def render_preset_selector() -> StreamingPreset:
    """
    Renderiza selector de presets.
    
    Returns:
        Preset seleccionado
    """
    preset_descriptions = {
        StreamingPreset.DEMO_RAPIDO: "🚀 **Demo Rápido** - 10 ciudades, 2 días (~30 seg)",
        StreamingPreset.VISUALIZACION: "📊 **Visualización** - 30 ciudades, 1 semana (~2 min)",
        StreamingPreset.ESTADISTICAS: "📈 **Estadísticas** - 50 ciudades, 1 mes (~5 min)",
        StreamingPreset.ML_TRAINING: "🤖 **ML Training** - 100 ciudades, 3 meses (~15 min)",
        StreamingPreset.STRESS_TEST: "💪 **Stress Test** - 500 ciudades, 1 año (~1 hora)",
        StreamingPreset.PERSONALIZADO: "⚙️ **Personalizado** - Configura todos los parámetros"
    }
    
    st.markdown("### 📋 Selecciona un Preset")
    
    selected = st.radio(
        "Tipo de generación",
        options=list(StreamingPreset),
        format_func=lambda x: preset_descriptions[x],
        key="streaming_preset",
        label_visibility="collapsed"
    )
    
    return selected


def render_basic_config(config: StreamingConfig) -> StreamingConfig:
    """
    Renderiza configuración básica.
    
    Args:
        config: Configuración actual
        
    Returns:
        Configuración actualizada
    """
    st.markdown("### 🌍 Configuración de Datos")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        config.n_cities = st.slider(
            "Número de Ciudades",
            min_value=5,
            max_value=500,
            value=config.n_cities,
            step=5,
            help="Más ciudades = más diversidad geográfica pero mayor tiempo de generación"
        )
    
    with col2:
        # Convertir horas a días para mejor UX
        days = config.forecast_hours // 24
        days = st.slider(
            "Horizonte (días)",
            min_value=1,
            max_value=365,
            value=days,
            help="Número de días de datos a generar por ciudad"
        )
        config.forecast_hours = days * 24
    
    with col3:
        resolution_options = {
            "15 minutos": 15,
            "30 minutos": 30,
            "1 hora": 60,
            "3 horas": 180,
            "6 horas": 360
        }
        resolution_label = st.selectbox(
            "Resolución Temporal",
            options=list(resolution_options.keys()),
            index=2,  # 1 hora por defecto
            help="Intervalo entre mediciones"
        )
        config.resolution_minutes = resolution_options[resolution_label]
    
    # Estimación de registros
    records_per_city = config.forecast_hours * (60 // config.resolution_minutes)
    total_records = config.n_cities * records_per_city
    
    st.info(f"""
    📊 **Estimación:** {total_records:,} registros totales
    ({config.n_cities} ciudades × {records_per_city:,} registros/ciudad)
    """)
    
    return config


def render_event_config(config: StreamingConfig) -> StreamingConfig:
    """
    Renderiza configuración de eventos.
    
    Args:
        config: Configuración actual
        
    Returns:
        Configuración actualizada
    """
    st.markdown("### 🌪️ Eventos Meteorológicos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Incluir en la generación:**")
        config.include_storms = st.checkbox("🌀 Tormentas", value=config.include_storms)
        config.include_alerts = st.checkbox("🚨 Alertas", value=config.include_alerts)
        config.include_events = st.checkbox("⚡ Eventos Extremos", value=config.include_events)
    
    with col2:
        st.markdown("**Probabilidades:**")
        config.rain_probability = st.slider(
            "🌧️ Lluvia",
            min_value=0.0,
            max_value=1.0,
            value=config.rain_probability,
            format="%.2f",
            help="Probabilidad base de precipitación"
        )
        
        if config.include_storms:
            config.storm_probability = st.slider(
                "🌀 Tormentas",
                min_value=0.0,
                max_value=0.2,
                value=config.storm_probability,
                format="%.3f",
                help="Probabilidad de tormenta severa"
            )
        
        if config.include_events:
            config.extreme_event_probability = st.slider(
                "⚡ Eventos Extremos",
                min_value=0.0,
                max_value=0.1,
                value=config.extreme_event_probability,
                format="%.3f",
                help="Probabilidad de evento extremo (ola de calor, etc.)"
            )
    
    return config


def render_alert_thresholds(config: StreamingConfig) -> StreamingConfig:
    """
    Renderiza configuración de umbrales de alertas.
    
    Args:
        config: Configuración actual
        
    Returns:
        Configuración actualizada
    """
    if not config.include_alerts:
        return config
    
    st.markdown("### 🚨 Umbrales de Alertas")
    
    tab1, tab2, tab3 = st.tabs(["🔥 Calor", "❄️ Frío", "💨 Viento"])
    
    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            config.heat_yellow_threshold = st.number_input(
                "🟡 Amarillo (°C)", value=config.heat_yellow_threshold, step=1.0
            )
        with col2:
            config.heat_orange_threshold = st.number_input(
                "🟠 Naranja (°C)", value=config.heat_orange_threshold, step=1.0
            )
        with col3:
            config.heat_red_threshold = st.number_input(
                "🔴 Rojo (°C)", value=config.heat_red_threshold, step=1.0
            )
    
    with tab2:
        col1, col2, col3 = st.columns(3)
        with col1:
            config.cold_yellow_threshold = st.number_input(
                "🟡 Amarillo (°C)", value=config.cold_yellow_threshold, step=1.0, key="cold_yellow"
            )
        with col2:
            config.cold_orange_threshold = st.number_input(
                "🟠 Naranja (°C)", value=config.cold_orange_threshold, step=1.0, key="cold_orange"
            )
        with col3:
            config.cold_red_threshold = st.number_input(
                "🔴 Rojo (°C)", value=config.cold_red_threshold, step=1.0, key="cold_red"
            )
    
    with tab3:
        col1, col2, col3 = st.columns(3)
        with col1:
            config.wind_yellow_threshold = st.number_input(
                "🟡 Amarillo (km/h)", value=config.wind_yellow_threshold, step=5.0
            )
        with col2:
            config.wind_orange_threshold = st.number_input(
                "🟠 Naranja (km/h)", value=config.wind_orange_threshold, step=5.0
            )
        with col3:
            config.wind_red_threshold = st.number_input(
                "🔴 Rojo (km/h)", value=config.wind_red_threshold, step=5.0
            )
    
    return config


def render_distribution_params(config: StreamingConfig) -> StreamingConfig:
    """
    Renderiza parámetros de distribución estadística.
    
    Args:
        config: Configuración actual
        
    Returns:
        Configuración actualizada
    """
    st.markdown("### 📊 Parámetros de Distribución")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Precipitación (Gamma)**")
        config.rain_gamma_scale = st.slider(
            "Escala", min_value=1.0, max_value=20.0, 
            value=config.rain_gamma_scale, step=0.5,
            help="Mayor escala = lluvias más intensas"
        )
        
        st.markdown("**Viento (Weibull)**")
        config.wind_weibull_shape = st.slider(
            "Forma", min_value=1.0, max_value=5.0,
            value=config.wind_weibull_shape, step=0.1,
            help="Forma de la distribución del viento"
        )
        config.wind_weibull_scale = st.slider(
            "Escala", min_value=5.0, max_value=30.0,
            value=config.wind_weibull_scale, step=1.0,
            help="Mayor escala = vientos más fuertes"
        )
    
    with col2:
        st.markdown("**Humedad (Normal)**")
        config.humidity_mean = st.slider(
            "Media (%)", min_value=30.0, max_value=90.0,
            value=config.humidity_mean, step=1.0
        )
        config.humidity_std = st.slider(
            "Desviación", min_value=5.0, max_value=30.0,
            value=config.humidity_std, step=1.0
        )
        
        st.markdown("**Presión (Normal)**")
        config.pressure_mean = st.slider(
            "Media (hPa)", min_value=980.0, max_value=1040.0,
            value=config.pressure_mean, step=1.0
        )
        config.pressure_std = st.slider(
            "Desviación", min_value=3.0, max_value=20.0,
            value=config.pressure_std, step=0.5
        )
    
    return config


def render_climate_zones(config: StreamingConfig) -> StreamingConfig:
    """
    Renderiza configuración de zonas climáticas.
    
    Args:
        config: Configuración actual
        
    Returns:
        Configuración actualizada
    """
    st.markdown("### 🌍 Distribución de Zonas Climáticas")
    
    st.markdown("Ajusta el peso de cada zona (la suma se normaliza automáticamente)")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        config.tropical_weight = st.slider(
            "🌴 Tropical", 0.0, 1.0, config.tropical_weight, 0.05
        )
    with col2:
        config.subtropical_weight = st.slider(
            "🏖️ Subtropical", 0.0, 1.0, config.subtropical_weight, 0.05
        )
    with col3:
        config.temperate_weight = st.slider(
            "🌲 Templado", 0.0, 1.0, config.temperate_weight, 0.05
        )
    with col4:
        config.continental_weight = st.slider(
            "🏔️ Continental", 0.0, 1.0, config.continental_weight, 0.05
        )
    with col5:
        config.polar_weight = st.slider(
            "❄️ Polar", 0.0, 1.0, config.polar_weight, 0.05
        )
    
    # Normalizar pesos
    total = (config.tropical_weight + config.subtropical_weight + 
             config.temperate_weight + config.continental_weight + config.polar_weight)
    
    if total > 0:
        st.markdown(f"""
        **Distribución normalizada:** 
        Tropical {config.tropical_weight/total*100:.0f}% | 
        Subtropical {config.subtropical_weight/total*100:.0f}% | 
        Templado {config.temperate_weight/total*100:.0f}% | 
        Continental {config.continental_weight/total*100:.0f}% | 
        Polar {config.polar_weight/total*100:.0f}%
        """)
    
    return config


def render_streaming_config(config: StreamingConfig) -> StreamingConfig:
    """
    Renderiza configuración de streaming en tiempo real.
    
    Args:
        config: Configuración actual
        
    Returns:
        Configuración actualizada
    """
    st.markdown("### 🌊 Configuración de Streaming")
    
    col1, col2 = st.columns(2)
    
    with col1:
        config.batch_size = st.number_input(
            "Tamaño de Batch",
            min_value=100,
            max_value=50000,
            value=config.batch_size,
            step=100,
            help="Registros por batch de escritura a HDFS"
        )
        
        config.micro_batch_size = st.number_input(
            "Micro-batch Size",
            min_value=10,
            max_value=1000,
            value=config.micro_batch_size,
            step=10,
            help="Registros por micro-batch en streaming"
        )
    
    with col2:
        config.trigger_interval_seconds = st.slider(
            "Intervalo de Trigger (seg)",
            min_value=1,
            max_value=60,
            value=config.trigger_interval_seconds,
            help="Tiempo entre emisiones de datos"
        )
        
        config.continuous_mode = st.checkbox(
            "Modo Continuo",
            value=config.continuous_mode,
            help="Generar datos indefinidamente hasta detener"
        )
        
        if config.continuous_mode:
            config.max_duration_minutes = st.number_input(
                "Duración Máxima (min)",
                min_value=1,
                max_value=1440,
                value=config.max_duration_minutes,
                help="Tiempo máximo de generación continua"
            )
    
    return config


def render_advanced_config(config: StreamingConfig) -> StreamingConfig:
    """
    Renderiza configuración avanzada.
    
    Args:
        config: Configuración actual
        
    Returns:
        Configuración actualizada
    """
    st.markdown("### ⚙️ Configuración Avanzada")
    
    col1, col2 = st.columns(2)
    
    with col1:
        config.seed = st.number_input(
            "Semilla (Seed)",
            min_value=0,
            max_value=999999,
            value=config.seed,
            help="Semilla para reproducibilidad"
        )
        
        st.markdown("**Rango Temporal:**")
        config.start_year = st.number_input(
            "Año Inicio",
            min_value=1900,
            max_value=2024,
            value=config.start_year
        )
        config.end_year = st.number_input(
            "Año Fin",
            min_value=config.start_year,
            max_value=2030,
            value=config.end_year
        )
    
    with col2:
        st.markdown("**Exportar/Importar Configuración:**")
        
        # Exportar
        config_json = config.to_json()
        st.download_button(
            "📥 Descargar Configuración",
            data=config_json,
            file_name="streaming_config.json",
            mime="application/json"
        )
        
        # Importar
        uploaded_file = st.file_uploader(
            "📤 Cargar Configuración",
            type=['json'],
            help="Cargar una configuración previamente exportada"
        )
        
        if uploaded_file is not None:
            try:
                loaded_config = StreamingConfig.from_json(uploaded_file.read().decode())
                st.success("✅ Configuración cargada")
                return loaded_config
            except Exception as e:
                st.error(f"Error cargando configuración: {e}")
    
    return config


# ============================================================================
# Componente Principal
# ============================================================================

def render_full_config_ui(initial_config: Optional[StreamingConfig] = None) -> StreamingConfig:
    """
    Renderiza la UI completa de configuración.
    
    Args:
        initial_config: Configuración inicial opcional
        
    Returns:
        Configuración final
    """
    # Inicializar config en session state si no existe
    if 'streaming_config' not in st.session_state:
        st.session_state.streaming_config = initial_config or StreamingConfig()
    
    config = st.session_state.streaming_config
    
    # Selector de preset
    preset = render_preset_selector()
    
    # Cargar preset si no es personalizado
    if preset != StreamingPreset.PERSONALIZADO:
        config = PRESETS[preset]
        st.session_state.streaming_config = config
    
    st.markdown("---")
    
    # Mostrar configuración según preset
    if preset == StreamingPreset.PERSONALIZADO:
        # Todas las opciones disponibles
        with st.expander("🌍 Datos Básicos", expanded=True):
            config = render_basic_config(config)
        
        with st.expander("🌪️ Eventos Meteorológicos"):
            config = render_event_config(config)
        
        if config.include_alerts:
            with st.expander("🚨 Umbrales de Alertas"):
                config = render_alert_thresholds(config)
        
        with st.expander("📊 Distribuciones Estadísticas"):
            config = render_distribution_params(config)
        
        with st.expander("🌍 Zonas Climáticas"):
            config = render_climate_zones(config)
        
        with st.expander("🌊 Streaming"):
            config = render_streaming_config(config)
        
        with st.expander("⚙️ Avanzado"):
            config = render_advanced_config(config)
    else:
        # Mostrar resumen de la configuración del preset
        st.markdown("#### 📋 Configuración del Preset")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🌍 Ciudades", config.n_cities)
        with col2:
            st.metric("📅 Días", config.forecast_hours // 24)
        with col3:
            records = config.n_cities * config.forecast_hours
            st.metric("📊 Registros Est.", f"{records:,}")
        with col4:
            st.metric("🌀 Tormentas", "✓" if config.include_storms else "✗")
        
        # Permitir ajustes rápidos
        with st.expander("⚡ Ajustes Rápidos"):
            config = render_basic_config(config)
    
    # Guardar config actualizada
    st.session_state.streaming_config = config
    
    return config


def get_current_config() -> StreamingConfig:
    """
    Obtiene la configuración actual del session state.
    
    Returns:
        Configuración actual o por defecto
    """
    return st.session_state.get('streaming_config', StreamingConfig())


def render_config_summary(config: StreamingConfig):
    """
    Muestra un resumen compacto de la configuración.
    
    Args:
        config: Configuración a mostrar
    """
    records_estimate = config.n_cities * config.forecast_hours * (60 // config.resolution_minutes)
    
    st.markdown(f"""
    | Parámetro | Valor |
    |-----------|-------|
    | 🌍 Ciudades | {config.n_cities} |
    | 📅 Horizonte | {config.forecast_hours // 24} días |
    | ⏱️ Resolución | {config.resolution_minutes} min |
    | 📊 Registros Est. | {records_estimate:,} |
    | 🌧️ Prob. Lluvia | {config.rain_probability:.0%} |
    | 🌀 Tormentas | {'✓' if config.include_storms else '✗'} |
    | 🚨 Alertas | {'✓' if config.include_alerts else '✗'} |
    | 🎲 Seed | {config.seed} |
    """)
