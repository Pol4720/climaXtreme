"""
Componentes UI reutilizables para el dashboard de climaXtreme.

Incluye:
- Configuración de streaming
- Componentes de Kafka en tiempo real
- Manager de Kafka streaming
"""

from climaxtreme.dashboard.components.streaming_config import (
    StreamingConfig,
    StreamingPreset,
    PRESETS,
    render_full_config_ui,
    render_config_summary,
    get_current_config,
    render_preset_selector,
    render_basic_config,
    render_event_config,
    render_alert_thresholds,
    render_streaming_config,
    render_advanced_config
)

# Kafka Realtime Components
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

# Kafka Manager
from climaxtreme.dashboard.components.kafka_manager import (
    KafkaStreamingConfig,
    KafkaStreamingManager,
    get_streaming_manager,
    render_kafka_cluster_status,
    render_streaming_config_form,
    render_producer_controls,
    render_quick_start_guide,
    render_full_kafka_control_panel
)

__all__ = [
    # Streaming Config
    'StreamingConfig',
    'StreamingPreset', 
    'PRESETS',
    'render_full_config_ui',
    'render_config_summary',
    'get_current_config',
    'render_preset_selector',
    'render_basic_config',
    'render_event_config',
    'render_alert_thresholds',
    'render_streaming_config',
    'render_advanced_config',
    
    # Kafka Realtime
    'get_kafka_state',
    'init_realtime_stream',
    'render_kafka_status_card',
    'render_realtime_controls',
    'create_realtime_weather_chart',
    'create_realtime_map',
    'create_realtime_alerts_panel',
    'create_realtime_metrics_row',
    'create_city_comparison_chart',
    'create_time_series_chart',
    'check_kafka_available',
    'render_kafka_setup_guide',
    'TOPICS',
    
    # Kafka Manager
    'KafkaStreamingConfig',
    'KafkaStreamingManager',
    'get_streaming_manager',
    'render_kafka_cluster_status',
    'render_streaming_config_form',
    'render_producer_controls',
    'render_quick_start_guide',
    'render_full_kafka_control_panel'
]
