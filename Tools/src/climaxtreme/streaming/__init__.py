"""
Streaming module for real-time climate data simulation.

This module provides tools for simulating streaming climate data,
useful for testing real-time dashboards and alert systems.

Example usage:
    from climaxtreme.streaming import StreamingConfig, StreamingSimulator
    
    config = StreamingConfig(n_cities=5, records_per_second=10)
    simulator = StreamingSimulator(config)
    stats = simulator.run_simulation(duration_seconds=60)
"""

# Lazy imports to avoid import errors when spark is not available
def __getattr__(name):
    """Lazy import streaming components."""
    if name in ('StreamingConfig', 'StreamingSimulator', 'run_streaming_demo',
                'create_spark_streaming_reader', 'create_alert_aggregation_query'):
        from .streaming_demo import (
            StreamingConfig,
            StreamingSimulator,
            run_streaming_demo,
            create_spark_streaming_reader,
            create_alert_aggregation_query
        )
        globals().update({
            'StreamingConfig': StreamingConfig,
            'StreamingSimulator': StreamingSimulator,
            'run_streaming_demo': run_streaming_demo,
            'create_spark_streaming_reader': create_spark_streaming_reader,
            'create_alert_aggregation_query': create_alert_aggregation_query
        })
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    'StreamingConfig',
    'StreamingSimulator', 
    'run_streaming_demo',
    'create_spark_streaming_reader',
    'create_alert_aggregation_query'
]
