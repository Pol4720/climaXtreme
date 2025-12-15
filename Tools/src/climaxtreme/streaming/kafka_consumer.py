"""
Kafka Consumer - Consumidor de datos climáticos desde Kafka.

Este módulo proporciona:
- Consumo de eventos desde tópicos Kafka
- Procesamiento en tiempo real con callbacks
- Integración con Spark Structured Streaming
- Persistencia a HDFS
- Soporte para grupos de consumidores
"""

import os
import json
import logging
import time
import threading
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable, Generator
from dataclasses import dataclass, asdict
from queue import Queue, Empty
from collections import defaultdict

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


@dataclass
class ConsumerConfig:
    """Configuración del consumer Kafka."""
    bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS
    group_id: str = 'climaxtreme-consumers'
    
    # Tópicos a consumir
    topics: List[str] = None
    
    # Configuración
    auto_offset_reset: str = 'latest'  # 'earliest' para leer desde el inicio
    enable_auto_commit: bool = True
    auto_commit_interval_ms: int = 5000
    session_timeout_ms: int = 30000
    max_poll_records: int = 500
    max_poll_interval_ms: int = 300000
    
    # Buffer
    buffer_size: int = 10000
    
    def __post_init__(self):
        if self.topics is None:
            self.topics = [TOPICS['weather'], TOPICS['alerts']]


# ============================================================================
# Event Handlers
# ============================================================================

class EventBuffer:
    """Buffer thread-safe para eventos consumidos."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._weather_events: List[Dict] = []
        self._alert_events: List[Dict] = []
        self._storm_events: List[Dict] = []
        self._lock = threading.Lock()
        self._stats = defaultdict(int)
    
    def add_weather_event(self, event: Dict) -> None:
        with self._lock:
            self._weather_events.append(event)
            if len(self._weather_events) > self.max_size:
                self._weather_events.pop(0)
            self._stats['weather_total'] += 1
    
    def add_alert_event(self, event: Dict) -> None:
        with self._lock:
            self._alert_events.append(event)
            if len(self._alert_events) > self.max_size:
                self._alert_events.pop(0)
            self._stats['alerts_total'] += 1
    
    def add_storm_event(self, event: Dict) -> None:
        with self._lock:
            self._storm_events.append(event)
            if len(self._storm_events) > self.max_size:
                self._storm_events.pop(0)
            self._stats['storms_total'] += 1
    
    def get_weather_events(self, n: int = 100) -> List[Dict]:
        with self._lock:
            return list(self._weather_events[-n:])
    
    def get_alert_events(self, n: int = 100) -> List[Dict]:
        with self._lock:
            return list(self._alert_events[-n:])
    
    def get_storm_events(self, n: int = 100) -> List[Dict]:
        with self._lock:
            return list(self._storm_events[-n:])
    
    def get_all_events(self, n: int = 100) -> Dict[str, List[Dict]]:
        return {
            'weather': self.get_weather_events(n),
            'alerts': self.get_alert_events(n),
            'storms': self.get_storm_events(n)
        }
    
    def get_stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                **dict(self._stats),
                'weather_buffered': len(self._weather_events),
                'alerts_buffered': len(self._alert_events),
                'storms_buffered': len(self._storm_events)
            }
    
    def clear(self) -> None:
        with self._lock:
            self._weather_events.clear()
            self._alert_events.clear()
            self._storm_events.clear()


# ============================================================================
# Kafka Consumer Class
# ============================================================================

class ClimateKafkaConsumer:
    """
    Consumidor Kafka para eventos climáticos.
    
    Consume eventos de los tópicos y los procesa mediante
    callbacks configurables.
    """
    
    def __init__(self, config: Optional[ConsumerConfig] = None):
        """
        Inicializar consumer.
        
        Args:
            config: Configuración del consumer
        """
        self.config = config or ConsumerConfig()
        self._consumer = None
        self._is_connected = False
        self._running = False
        self._thread = None
        
        # Buffer de eventos
        self.buffer = EventBuffer(self.config.buffer_size)
        
        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Estadísticas
        self._stats = {
            'messages_consumed': 0,
            'errors': 0,
            'start_time': None,
            'last_message_time': None
        }
        
        logger.info(f"KafkaConsumer initialized (group: {self.config.group_id})")
    
    def connect(self) -> bool:
        """
        Conectar al broker Kafka.
        
        Returns:
            True si la conexión fue exitosa
        """
        try:
            from kafka import KafkaConsumer
            
            self._consumer = KafkaConsumer(
                *self.config.topics,
                bootstrap_servers=self.config.bootstrap_servers.split(','),
                group_id=self.config.group_id,
                auto_offset_reset=self.config.auto_offset_reset,
                enable_auto_commit=self.config.enable_auto_commit,
                auto_commit_interval_ms=self.config.auto_commit_interval_ms,
                session_timeout_ms=self.config.session_timeout_ms,
                max_poll_records=self.config.max_poll_records,
                value_deserializer=lambda m: json.loads(m.decode('utf-8'))
            )
            
            self._is_connected = True
            logger.info(f"Connected to Kafka, subscribed to: {self.config.topics}")
            return True
            
        except ImportError:
            logger.error("kafka-python not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            return False
    
    def disconnect(self) -> None:
        """Cerrar conexión con Kafka."""
        if self._consumer:
            self._consumer.close()
            self._consumer = None
            self._is_connected = False
            logger.info("Disconnected from Kafka")
    
    def register_callback(self, topic: str, callback: Callable[[Dict], None]) -> None:
        """
        Registrar callback para un tópico.
        
        Args:
            topic: Nombre del tópico
            callback: Función a ejecutar con cada mensaje
        """
        self._callbacks[topic].append(callback)
        logger.info(f"Registered callback for topic: {topic}")
    
    def _process_message(self, message) -> None:
        """Procesar mensaje recibido."""
        try:
            topic = message.topic
            value = message.value
            
            # Agregar a buffer según tipo
            if topic == TOPICS['weather']:
                self.buffer.add_weather_event(value)
            elif topic == TOPICS['alerts']:
                self.buffer.add_alert_event(value)
            elif topic == TOPICS['storms']:
                self.buffer.add_storm_event(value)
            
            # Ejecutar callbacks
            for callback in self._callbacks.get(topic, []):
                try:
                    callback(value)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
            
            self._stats['messages_consumed'] += 1
            self._stats['last_message_time'] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self._stats['errors'] += 1
    
    def _consume_loop(self) -> None:
        """Loop principal de consumo."""
        logger.info("Starting Kafka consume loop")
        
        while self._running:
            try:
                # Poll con timeout
                messages = self._consumer.poll(timeout_ms=1000)
                
                for topic_partition, records in messages.items():
                    for record in records:
                        self._process_message(record)
                
            except Exception as e:
                logger.error(f"Error in consume loop: {e}")
                self._stats['errors'] += 1
                time.sleep(1)
        
        logger.info("Kafka consume loop stopped")
    
    def start_consuming(self) -> bool:
        """
        Iniciar consumo en background.
        
        Returns:
            True si se inició correctamente
        """
        if self._running:
            logger.warning("Consumer already running")
            return False
        
        if not self._is_connected:
            if not self.connect():
                return False
        
        self._running = True
        self._stats['start_time'] = datetime.now().isoformat()
        
        self._thread = threading.Thread(target=self._consume_loop, daemon=True)
        self._thread.start()
        
        logger.info("Kafka consumer started")
        return True
    
    def stop_consuming(self) -> None:
        """Detener consumo."""
        self._running = False
        
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        
        self.disconnect()
        logger.info("Kafka consumer stopped")
    
    def consume_batch(self, timeout_ms: int = 5000, max_records: int = 100) -> List[Dict]:
        """
        Consumir batch de mensajes (síncrono).
        
        Args:
            timeout_ms: Timeout para poll
            max_records: Máximo de registros a retornar
            
        Returns:
            Lista de mensajes consumidos
        """
        if not self._is_connected:
            if not self.connect():
                return []
        
        messages = []
        
        try:
            records = self._consumer.poll(timeout_ms=timeout_ms, max_records=max_records)
            
            for topic_partition, batch in records.items():
                for record in batch:
                    messages.append({
                        'topic': record.topic,
                        'partition': record.partition,
                        'offset': record.offset,
                        'timestamp': record.timestamp,
                        'value': record.value
                    })
                    self._process_message(record)
            
        except Exception as e:
            logger.error(f"Error consuming batch: {e}")
        
        return messages
    
    def consume_stream(self, batch_size: int = 100) -> Generator[List[Dict], None, None]:
        """
        Generador que produce batches de mensajes.
        
        Args:
            batch_size: Tamaño del batch
            
        Yields:
            Batches de mensajes
        """
        if not self._is_connected:
            if not self.connect():
                return
        
        while True:
            batch = self.consume_batch(timeout_ms=1000, max_records=batch_size)
            if batch:
                yield batch
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del consumer."""
        return {
            **self._stats,
            'is_connected': self._is_connected,
            'is_running': self._running,
            'buffer_stats': self.buffer.get_stats()
        }
    
    def get_latest_events(self, n: int = 100) -> Dict[str, List[Dict]]:
        """Obtener últimos eventos del buffer."""
        return self.buffer.get_all_events(n)
    
    def __enter__(self):
        self.start_consuming()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_consuming()


# ============================================================================
# Spark Structured Streaming con Kafka
# ============================================================================

class SparkKafkaConsumer:
    """
    Consumer que usa Spark Structured Streaming para leer de Kafka.
    
    Ideal para procesamiento distribuido y persistencia a HDFS.
    """
    
    def __init__(
        self,
        bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS,
        topics: Optional[List[str]] = None,
        checkpoint_path: str = "hdfs://climaxtreme-namenode:9000/checkpoints/kafka",
        output_path: str = "hdfs://climaxtreme-namenode:9000/data/climaxtreme/streaming"
    ):
        """
        Inicializar Spark Kafka consumer.
        
        Args:
            bootstrap_servers: Servidores Kafka
            topics: Lista de tópicos a consumir
            checkpoint_path: Ruta para checkpoints de Spark
            output_path: Ruta de salida en HDFS
        """
        self.bootstrap_servers = bootstrap_servers
        self.topics = topics or [TOPICS['weather']]
        self.checkpoint_path = checkpoint_path
        self.output_path = output_path
        
        self._spark = None
        self._stream_query = None
    
    def _get_spark(self):
        """Obtener o crear SparkSession."""
        if self._spark is None:
            from pyspark.sql import SparkSession
            
            self._spark = SparkSession.builder \
                .appName("climaXtreme-Kafka-Consumer") \
                .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
                .config("spark.sql.streaming.checkpointLocation", self.checkpoint_path) \
                .getOrCreate()
            
            self._spark.sparkContext.setLogLevel("WARN")
        
        return self._spark
    
    def create_stream(self):
        """
        Crear DataFrame de streaming desde Kafka.
        
        Returns:
            Streaming DataFrame
        """
        spark = self._get_spark()
        
        from pyspark.sql import functions as F
        from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType
        
        # Schema del evento de clima
        weather_schema = StructType([
            StructField("city", StringType(), True),
            StructField("country", StringType(), True),
            StructField("latitude", DoubleType(), True),
            StructField("longitude", DoubleType(), True),
            StructField("timestamp", StringType(), True),
            StructField("temperature_c", DoubleType(), True),
            StructField("humidity_pct", DoubleType(), True),
            StructField("pressure_hpa", DoubleType(), True),
            StructField("wind_speed_kmh", DoubleType(), True),
            StructField("wind_direction", DoubleType(), True),
            StructField("rain_mm", DoubleType(), True),
            StructField("cloud_cover_pct", DoubleType(), True),
            StructField("uv_index", DoubleType(), True),
            StructField("climate_zone", StringType(), True),
            StructField("event_id", StringType(), True)
        ])
        
        # Leer stream de Kafka
        df = spark.readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.bootstrap_servers) \
            .option("subscribe", ",".join(self.topics)) \
            .option("startingOffsets", "latest") \
            .load()
        
        # Parsear JSON
        parsed_df = df.select(
            F.col("topic"),
            F.col("partition"),
            F.col("offset"),
            F.col("timestamp").alias("kafka_timestamp"),
            F.from_json(F.col("value").cast("string"), weather_schema).alias("data")
        ).select(
            "topic", "partition", "offset", "kafka_timestamp",
            "data.*"
        )
        
        return parsed_df
    
    def start_hdfs_sink(
        self,
        output_format: str = "parquet",
        trigger_interval: str = "10 seconds",
        partition_by: Optional[List[str]] = None
    ):
        """
        Iniciar sink a HDFS.
        
        Args:
            output_format: Formato de salida (parquet, json, csv)
            trigger_interval: Intervalo de trigger
            partition_by: Columnas para particionar
        """
        stream_df = self.create_stream()
        
        writer = stream_df.writeStream \
            .outputMode("append") \
            .format(output_format) \
            .option("path", self.output_path) \
            .option("checkpointLocation", self.checkpoint_path) \
            .trigger(processingTime=trigger_interval)
        
        if partition_by:
            writer = writer.partitionBy(*partition_by)
        
        self._stream_query = writer.start()
        
        logger.info(f"Started HDFS sink: {self.output_path}")
        return self._stream_query
    
    def start_console_sink(self, trigger_interval: str = "5 seconds"):
        """
        Iniciar sink a consola (para debugging).
        """
        stream_df = self.create_stream()
        
        self._stream_query = stream_df.writeStream \
            .outputMode("append") \
            .format("console") \
            .option("truncate", False) \
            .trigger(processingTime=trigger_interval) \
            .start()
        
        return self._stream_query
    
    def start_memory_sink(self, table_name: str = "weather_events"):
        """
        Iniciar sink en memoria (para queries interactivas).
        
        Args:
            table_name: Nombre de la tabla temporal
        """
        stream_df = self.create_stream()
        
        self._stream_query = stream_df.writeStream \
            .outputMode("append") \
            .format("memory") \
            .queryName(table_name) \
            .start()
        
        return self._stream_query
    
    def query_memory_table(self, table_name: str, sql: str):
        """
        Ejecutar query SQL en tabla de memoria.
        
        Args:
            table_name: Nombre de la tabla
            sql: Query SQL
        """
        spark = self._get_spark()
        return spark.sql(sql)
    
    def stop(self):
        """Detener streaming."""
        if self._stream_query:
            self._stream_query.stop()
            self._stream_query = None
            logger.info("Streaming query stopped")
    
    def await_termination(self, timeout: Optional[int] = None):
        """
        Esperar terminación del stream.
        
        Args:
            timeout: Timeout en segundos
        """
        if self._stream_query:
            self._stream_query.awaitTermination(timeout)
    
    def get_status(self) -> Dict[str, Any]:
        """Obtener estado del stream."""
        if not self._stream_query:
            return {'status': 'not_started'}
        
        return {
            'status': 'running' if self._stream_query.isActive else 'stopped',
            'name': self._stream_query.name,
            'id': str(self._stream_query.id),
            'recent_progress': self._stream_query.recentProgress
        }


# ============================================================================
# Consumer con procesamiento en tiempo real
# ============================================================================

class RealtimeWeatherProcessor:
    """
    Procesador en tiempo real de eventos meteorológicos.
    
    Consume de Kafka, procesa y genera agregaciones/alertas.
    """
    
    def __init__(
        self,
        config: Optional[ConsumerConfig] = None,
        aggregation_window_seconds: int = 60
    ):
        """
        Inicializar procesador.
        
        Args:
            config: Configuración del consumer
            aggregation_window_seconds: Ventana de agregación
        """
        self.consumer = ClimateKafkaConsumer(config)
        self.aggregation_window = aggregation_window_seconds
        
        # Agregaciones en memoria
        self._city_stats: Dict[str, Dict] = {}
        self._alert_counts: Dict[str, int] = defaultdict(int)
        self._window_start = datetime.now()
        
        # Callbacks
        self._on_aggregation: Optional[Callable] = None
        self._on_anomaly: Optional[Callable] = None
    
    def set_aggregation_callback(self, callback: Callable[[Dict], None]) -> None:
        """Establecer callback para agregaciones."""
        self._on_aggregation = callback
    
    def set_anomaly_callback(self, callback: Callable[[Dict], None]) -> None:
        """Establecer callback para anomalías."""
        self._on_anomaly = callback
    
    def _process_weather_event(self, event: Dict) -> None:
        """Procesar evento de clima."""
        city = event.get('city', 'unknown')
        
        # Actualizar estadísticas de la ciudad
        if city not in self._city_stats:
            self._city_stats[city] = {
                'count': 0,
                'temp_sum': 0,
                'temp_min': float('inf'),
                'temp_max': float('-inf'),
                'last_event': None
            }
        
        stats = self._city_stats[city]
        temp = event.get('temperature_c', 0)
        
        stats['count'] += 1
        stats['temp_sum'] += temp
        stats['temp_min'] = min(stats['temp_min'], temp)
        stats['temp_max'] = max(stats['temp_max'], temp)
        stats['last_event'] = event
        
        # Detectar anomalías (temperatura extrema)
        if temp > 40 or temp < -20:
            if self._on_anomaly:
                self._on_anomaly({
                    'type': 'extreme_temperature',
                    'city': city,
                    'temperature': temp,
                    'event': event
                })
    
    def _process_alert_event(self, event: Dict) -> None:
        """Procesar evento de alerta."""
        alert_type = event.get('alert_type', 'unknown')
        self._alert_counts[alert_type] += 1
    
    def _check_window_aggregation(self) -> None:
        """Verificar si se debe emitir agregación."""
        now = datetime.now()
        elapsed = (now - self._window_start).total_seconds()
        
        if elapsed >= self.aggregation_window:
            if self._on_aggregation and self._city_stats:
                # Calcular agregaciones
                aggregation = {
                    'window_start': self._window_start.isoformat(),
                    'window_end': now.isoformat(),
                    'cities_count': len(self._city_stats),
                    'total_events': sum(s['count'] for s in self._city_stats.values()),
                    'alert_counts': dict(self._alert_counts),
                    'city_summaries': {
                        city: {
                            'count': s['count'],
                            'temp_avg': s['temp_sum'] / s['count'] if s['count'] > 0 else 0,
                            'temp_min': s['temp_min'] if s['temp_min'] != float('inf') else None,
                            'temp_max': s['temp_max'] if s['temp_max'] != float('-inf') else None
                        }
                        for city, s in self._city_stats.items()
                    }
                }
                
                self._on_aggregation(aggregation)
            
            # Reset ventana
            self._city_stats.clear()
            self._alert_counts.clear()
            self._window_start = now
    
    def start(self) -> None:
        """Iniciar procesamiento."""
        # Registrar callbacks internos
        self.consumer.register_callback(
            TOPICS['weather'],
            self._process_weather_event
        )
        self.consumer.register_callback(
            TOPICS['alerts'],
            self._process_alert_event
        )
        
        # Iniciar consumer
        self.consumer.start_consuming()
        
        # Thread para verificar ventana de agregación
        def aggregation_checker():
            while self.consumer._running:
                self._check_window_aggregation()
                time.sleep(1)
        
        self._agg_thread = threading.Thread(target=aggregation_checker, daemon=True)
        self._agg_thread.start()
        
        logger.info("Realtime processor started")
    
    def stop(self) -> None:
        """Detener procesamiento."""
        self.consumer.stop_consuming()
        logger.info("Realtime processor stopped")
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas actuales."""
        return {
            'window_start': self._window_start.isoformat(),
            'cities_monitored': len(self._city_stats),
            'total_events_in_window': sum(s['count'] for s in self._city_stats.values()),
            'alert_counts': dict(self._alert_counts),
            'consumer_stats': self.consumer.get_stats()
        }


# ============================================================================
# Utilidades
# ============================================================================

def list_kafka_topics(
    bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS
) -> List[str]:
    """
    Listar tópicos disponibles en Kafka.
    
    Returns:
        Lista de nombres de tópicos
    """
    try:
        from kafka import KafkaConsumer
        
        consumer = KafkaConsumer(
            bootstrap_servers=bootstrap_servers.split(','),
            consumer_timeout_ms=5000
        )
        topics = list(consumer.topics())
        consumer.close()
        
        return topics
        
    except ImportError:
        logger.error("kafka-python not installed")
        return []
    except Exception as e:
        logger.error(f"Failed to list topics: {e}")
        return []


def get_topic_offsets(
    topic: str,
    bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS
) -> Dict[int, Dict[str, int]]:
    """
    Obtener offsets de un tópico.
    
    Args:
        topic: Nombre del tópico
        bootstrap_servers: Servidores Kafka
        
    Returns:
        Offsets por partición
    """
    try:
        from kafka import KafkaConsumer
        from kafka.structs import TopicPartition
        
        consumer = KafkaConsumer(
            bootstrap_servers=bootstrap_servers.split(','),
            consumer_timeout_ms=5000
        )
        
        partitions = consumer.partitions_for_topic(topic)
        if not partitions:
            consumer.close()
            return {}
        
        result = {}
        for partition in partitions:
            tp = TopicPartition(topic, partition)
            consumer.assign([tp])
            
            beginning = consumer.beginning_offsets([tp])
            end = consumer.end_offsets([tp])
            
            result[partition] = {
                'beginning': beginning.get(tp, 0),
                'end': end.get(tp, 0),
                'messages': end.get(tp, 0) - beginning.get(tp, 0)
            }
        
        consumer.close()
        return result
        
    except ImportError:
        logger.error("kafka-python not installed")
        return {}
    except Exception as e:
        logger.error(f"Failed to get offsets: {e}")
        return {}
