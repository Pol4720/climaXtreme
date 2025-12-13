"""
Streaming module for real-time climate data generation and Kafka integration.

This module provides tools for:
- Real-time streaming via Apache Kafka
- Synthetic data generation with Spark (SyntheticClimateGenerator)
- EDA validation of synthetic data distributions
- Dashboard integration for live data visualization

Architecture:
    ┌─────────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │ SyntheticClimate    │    │   Kafka         │    │   Dashboard     │
    │ Generator (Spark)   │───▶│   Broker        │───▶│   (Real-time)   │
    └─────────────────────┘    └─────────────────┘    └─────────────────┘

Example usage:
    # Spark-Kafka streaming (RECOMMENDED)
    from climaxtreme.streaming import SparkKafkaStreamingProducer, SparkKafkaConfig
    
    config = SparkKafkaConfig(
        kafka_bootstrap_servers="localhost:9092",
        max_cities=50
    )
    producer = SparkKafkaStreamingProducer(config)
    producer.start()
    
    # Kafka consumer for dashboard
    from climaxtreme.streaming import ClimateKafkaConsumer
    
    consumer = ClimateKafkaConsumer(bootstrap_servers="localhost:9092")
    for event in consumer.consume_weather():
        print(event)
    
    # EDA validation
    from climaxtreme.streaming import SyntheticDataValidator
    validator = SyntheticDataValidator()
    report = validator.validate(df)
"""

# Lazy imports to avoid import errors when dependencies are not available
def __getattr__(name):
    """Lazy import streaming components."""
    
    # Spark-Kafka Producer (RECOMMENDED - uses SyntheticClimateGenerator)
    if name in ('SparkKafkaStreamingProducer', 'ContinuousSparkKafkaProducer',
                'SparkKafkaConfig'):
        from .spark_kafka_producer import (
            SparkKafkaStreamingProducer,
            ContinuousSparkKafkaProducer,
            SparkKafkaConfig
        )
        globals().update({
            'SparkKafkaStreamingProducer': SparkKafkaStreamingProducer,
            'ContinuousSparkKafkaProducer': ContinuousSparkKafkaProducer,
            'SparkKafkaConfig': SparkKafkaConfig
        })
        return globals()[name]
    
    # Kafka Producer components (Python-based, simpler)
    if name in ('ClimateKafkaProducer', 'KafkaStreamingProducer', 'KafkaConfig',
                'WeatherEvent', 'AlertEvent', 'StormEvent',
                'create_kafka_topics', 'check_kafka_connection', 'TOPICS'):
        from .kafka_producer import (
            ClimateKafkaProducer,
            KafkaStreamingProducer,
            KafkaConfig,
            WeatherEvent,
            AlertEvent,
            StormEvent,
            create_kafka_topics,
            check_kafka_connection,
            TOPICS
        )
        globals().update({
            'ClimateKafkaProducer': ClimateKafkaProducer,
            'KafkaStreamingProducer': KafkaStreamingProducer,
            'KafkaConfig': KafkaConfig,
            'WeatherEvent': WeatherEvent,
            'AlertEvent': AlertEvent,
            'StormEvent': StormEvent,
            'create_kafka_topics': create_kafka_topics,
            'check_kafka_connection': check_kafka_connection,
            'TOPICS': TOPICS
        })
        return globals()[name]
    
    # Kafka Consumer components
    if name in ('ClimateKafkaConsumer', 'SparkKafkaConsumer', 'RealtimeWeatherProcessor',
                'ConsumerConfig', 'EventBuffer', 'list_kafka_topics', 'get_topic_offsets'):
        from .kafka_consumer import (
            ClimateKafkaConsumer,
            SparkKafkaConsumer,
            RealtimeWeatherProcessor,
            ConsumerConfig,
            EventBuffer,
            list_kafka_topics,
            get_topic_offsets
        )
        globals().update({
            'ClimateKafkaConsumer': ClimateKafkaConsumer,
            'SparkKafkaConsumer': SparkKafkaConsumer,
            'RealtimeWeatherProcessor': RealtimeWeatherProcessor,
            'ConsumerConfig': ConsumerConfig,
            'EventBuffer': EventBuffer,
            'list_kafka_topics': list_kafka_topics,
            'get_topic_offsets': get_topic_offsets
        })
        return globals()[name]
    
    # EDA Validator components
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
    
    # Streaming demo components
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
    # Spark-Kafka Producer (RECOMMENDED)
    'SparkKafkaStreamingProducer',
    'ContinuousSparkKafkaProducer',
    'SparkKafkaConfig',
    # Kafka Producer (Python-based)
    'ClimateKafkaProducer',
    'KafkaStreamingProducer',
    'KafkaConfig',
    'WeatherEvent',
    'AlertEvent',
    'StormEvent',
    'create_kafka_topics',
    'check_kafka_connection',
    'TOPICS',
    # Kafka Consumer
    'ClimateKafkaConsumer',
    'SparkKafkaConsumer',
    'RealtimeWeatherProcessor',
    'ConsumerConfig',
    'EventBuffer',
    'list_kafka_topics',
    'get_topic_offsets',
    # EDA Validation
    'SyntheticDataValidator',
    'ValidationResult',
    'EDAReport',
    'validate_synthetic_data',
    'get_distribution_plots_data',
    # Demo
    'StreamingSimulator',
    'run_streaming_demo',
    'create_spark_streaming_reader',
    'create_alert_aggregation_query',
]
