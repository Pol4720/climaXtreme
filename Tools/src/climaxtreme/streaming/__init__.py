"""
Streaming module for real-time climate data simulation and generation using PySpark.

This module provides tools for:
- Real-time streaming climate data generation with Spark Structured Streaming
- On-demand synthetic data for forecasting models
- EDA validation of synthetic data distributions using Spark SQL
- Dashboard integration with HDFS

Example usage:
    # PySpark streaming generator
    from climaxtreme.streaming import SparkStreamingGenerator, StreamingConfig
    
    spark = SparkSession.builder.appName("climaXtreme").getOrCreate()
    config = StreamingConfig(forecast_horizon_hours=24)
    generator = SparkStreamingGenerator(spark, config)
    
    # Generate forecast from HDFS data
    forecast_df = generator.generate_streaming_forecast(n_cities=50)
    
    # Validate synthetic data with Spark
    from climaxtreme.streaming import SparkEDAValidator
    validator = SparkEDAValidator(spark)
    report = validator.run_full_validation(forecast_df)
    
    # Legacy streaming demo
    from climaxtreme.streaming import StreamingSimulator
    simulator = StreamingSimulator(config)
    stats = simulator.run_simulation(duration_seconds=60)
"""

# Lazy imports to avoid import errors when dependencies are not available
def __getattr__(name):
    """Lazy import streaming components."""
    
    # PySpark streaming generator components
    if name in ('SparkStreamingGenerator', 'StreamingConfig', 'SparkEDAValidator',
                'create_streaming_generator', 'generate_forecast_from_hdfs',
                'validate_synthetic_data_spark', 'CLIMATE_ZONES'):
        from .realtime_generator import (
            SparkStreamingGenerator,
            StreamingConfig,
            SparkEDAValidator,
            create_streaming_generator,
            generate_forecast_from_hdfs,
            validate_synthetic_data_spark,
            CLIMATE_ZONES
        )
        globals().update({
            'SparkStreamingGenerator': SparkStreamingGenerator,
            'StreamingConfig': StreamingConfig,
            'SparkEDAValidator': SparkEDAValidator,
            'create_streaming_generator': create_streaming_generator,
            'generate_forecast_from_hdfs': generate_forecast_from_hdfs,
            'validate_synthetic_data_spark': validate_synthetic_data_spark,
            'CLIMATE_ZONES': CLIMATE_ZONES
        })
        return globals()[name]
    
    # EDA Validator components (legacy pandas-based)
    if name in ('SyntheticDataValidator', 'ValidationResult', 'EDAReport',
                'validate_synthetic_data', 'get_distribution_plots_data'):
        from .eda_validator import (
            SyntheticDataValidator,
            ValidationResult,
            EDAReport,
            validate_synthetic_data,
            get_distribution_plots_data
        )
        globals().update({
            'SyntheticDataValidator': SyntheticDataValidator,
            'ValidationResult': ValidationResult,
            'EDAReport': EDAReport,
            'validate_synthetic_data': validate_synthetic_data,
            'get_distribution_plots_data': get_distribution_plots_data
        })
        return globals()[name]
    
    # Legacy streaming demo components
    if name in ('StreamingSimulator', 'run_streaming_demo',
                'create_spark_streaming_reader', 'create_alert_aggregation_query'):
        from .streaming_demo import (
            StreamingSimulator,
            run_streaming_demo,
            create_spark_streaming_reader,
            create_alert_aggregation_query
        )
        globals().update({
            'StreamingSimulator': StreamingSimulator,
            'run_streaming_demo': run_streaming_demo,
            'create_spark_streaming_reader': create_spark_streaming_reader,
            'create_alert_aggregation_query': create_alert_aggregation_query
        })
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # PySpark streaming
    'SparkStreamingGenerator',
    'StreamingConfig',
    'SparkEDAValidator',
    'create_streaming_generator',
    'generate_forecast_from_hdfs',
    'validate_synthetic_data_spark',
    'CLIMATE_ZONES',
    # Legacy 
    'StreamingSimulator', 
    'run_streaming_demo',
    'create_spark_streaming_reader',
    'create_alert_aggregation_query',
    # EDA
    'SyntheticDataValidator',
    'validate_synthetic_data'
]
