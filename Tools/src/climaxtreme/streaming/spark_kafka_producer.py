"""
Spark-Kafka Streaming Producer.

Este módulo integra el SyntheticClimateGenerator con Kafka,
permitiendo generar datos sintéticos con Spark y enviarlos
a Kafka para su consumo en tiempo real por el dashboard.

Es el puente entre:
- SyntheticClimateGenerator (genera datos con modelos estadísticos)
- Kafka (transmite eventos en tiempo real)
- Dashboard Streamlit (consume y visualiza)
"""

import os
import sys
import json
import time
import logging
import threading
from datetime import datetime
from typing import Optional, Dict, Any, List, Generator
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor

# Kafka
try:
    from kafka import KafkaProducer
    from kafka.errors import KafkaError, NoBrokersAvailable
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    KafkaProducer = None

# PySpark
try:
    from pyspark.sql import SparkSession, DataFrame
    from pyspark.sql import functions as F
    from pyspark.sql.types import StructType
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
    SparkSession = None
    DataFrame = None

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class SparkKafkaConfig:
    """Configuration for Spark-Kafka producer."""
    # Kafka settings
    kafka_bootstrap_servers: str = "localhost:9092"
    topic_weather: str = "climaxtreme-weather"
    topic_alerts: str = "climaxtreme-alerts"
    topic_storms: str = "climaxtreme-storms"
    topic_progress: str = "climaxtreme-progress"
    
    # Generation settings
    batch_size: int = 100
    delay_between_batches: float = 0.5  # seconds
    cities_per_batch: int = 5
    hours_to_generate: int = 24
    
    # Spark settings
    spark_master: str = "local[*]"
    app_name: str = "ClimaxtremeKafkaProducer"
    
    # Data source
    input_csv_path: str = "/app/DATA/GlobalLandTemperaturesByCity.csv"
    sample_cities: Optional[List[str]] = None
    max_cities: int = 20


# ============================================================================
# Event Serializers
# ============================================================================

def serialize_weather_event(row: Dict[str, Any]) -> bytes:
    """Serialize a weather row to JSON bytes."""
    event = {
        "event_type": "weather_update",
        "timestamp": row.get("timestamp", datetime.now().isoformat()),
        "city": row.get("city", "Unknown"),
        "country": row.get("country", "Unknown"),
        "latitude": row.get("lat_decimal", 0.0),
        "longitude": row.get("lon_decimal", 0.0),
        "temperature": row.get("temperature_hourly", row.get("avg_temperature", 20.0)),
        "humidity": row.get("humidity_pct", 50.0),
        "pressure": row.get("pressure_hpa", 1013.0),
        "wind_speed": row.get("wind_speed_kmh", 10.0),
        "wind_direction": row.get("wind_direction_deg", 180.0),
        "rain_mm": row.get("rain_mm", 0.0),
        "cloud_cover": row.get("cloud_cover_pct", 30.0),
        "climate_zone": row.get("climate_zone", "TEMPERATE"),
        "season": row.get("season", "SUMMER"),
        "event_classification": row.get("event_type", "NORMAL"),
        "anomaly_score": row.get("anomaly_score", 0.0),
        "generated_at": datetime.now().isoformat()
    }
    return json.dumps(event, default=str).encode('utf-8')


def serialize_alert_event(row: Dict[str, Any]) -> bytes:
    """Serialize an alert row to JSON bytes."""
    event = {
        "event_type": "alert",
        "alert_id": f"ALT-{row.get('city', 'UNK')[:3].upper()}-{int(time.time()*1000)}",
        "timestamp": row.get("timestamp", datetime.now().isoformat()),
        "city": row.get("city", "Unknown"),
        "country": row.get("country", "Unknown"),
        "latitude": row.get("lat_decimal", 0.0),
        "longitude": row.get("lon_decimal", 0.0),
        "alert_type": row.get("alert_type", "WEATHER"),
        "alert_level": row.get("alert_level", "WATCH"),
        "event_classification": row.get("event_type", "NORMAL"),
        "temperature": row.get("temperature_hourly", 20.0),
        "wind_speed": row.get("wind_speed_kmh", 10.0),
        "rain_mm": row.get("rain_mm", 0.0),
        "intensity": row.get("event_intensity", 0.0),
        "generated_at": datetime.now().isoformat()
    }
    return json.dumps(event, default=str).encode('utf-8')


def serialize_storm_event(row: Dict[str, Any]) -> bytes:
    """Serialize a storm track row to JSON bytes."""
    event = {
        "event_type": "storm_update",
        "storm_id": row.get("storm_id", f"STM-{int(time.time())}"),
        "storm_name": row.get("storm_name", "Unknown"),
        "timestamp": row.get("timestamp", datetime.now().isoformat()),
        "latitude": row.get("latitude", row.get("lat_decimal", 0.0)),
        "longitude": row.get("longitude", row.get("lon_decimal", 0.0)),
        "category": row.get("storm_category", row.get("category", 0)),
        "max_wind_kmh": row.get("max_wind_kmh", row.get("wind_speed_kmh", 50.0)),
        "central_pressure": row.get("central_pressure_hpa", row.get("pressure_hpa", 1000.0)),
        "movement_speed": row.get("movement_speed_kmh", 20.0),
        "movement_direction": row.get("movement_direction_deg", 45.0),
        "radius_km": row.get("radius_km", 100.0),
        "lifecycle_stage": row.get("lifecycle_stage", "MATURE"),
        "generated_at": datetime.now().isoformat()
    }
    return json.dumps(event, default=str).encode('utf-8')


def serialize_progress_event(
    phase: str,
    progress: float,
    message: str,
    stats: Optional[Dict] = None
) -> bytes:
    """Serialize a progress event."""
    event = {
        "event_type": "progress",
        "phase": phase,
        "progress": progress,
        "message": message,
        "stats": stats or {},
        "timestamp": datetime.now().isoformat()
    }
    return json.dumps(event).encode('utf-8')


# ============================================================================
# Main Producer Class
# ============================================================================

class SparkKafkaStreamingProducer:
    """
    Producer that uses Spark's SyntheticClimateGenerator to generate
    high-quality synthetic data and streams it to Kafka.
    
    This bridges the gap between:
    - The statistical models in SyntheticClimateGenerator
    - Real-time streaming to Kafka for dashboard consumption
    """
    
    def __init__(self, config: Optional[SparkKafkaConfig] = None):
        """Initialize the Spark-Kafka producer."""
        self.config = config or SparkKafkaConfig()
        self.spark = None  # SparkSession when initialized
        self.producer = None  # KafkaProducer when initialized
        self.generator = None
        
        self._running = False
        self._stop_event = threading.Event()
        
        # Statistics
        self.stats = {
            "weather_events_sent": 0,
            "alerts_sent": 0,
            "storms_sent": 0,
            "errors": 0,
            "start_time": None,
            "last_event_time": None
        }
        
        logger.info("SparkKafkaStreamingProducer initialized")
    
    def _init_spark(self) -> bool:
        """Initialize Spark session."""
        if not SPARK_AVAILABLE:
            logger.error("PySpark not available")
            return False
        
        try:
            self.spark = SparkSession.builder \
                .appName(self.config.app_name) \
                .master(self.config.spark_master) \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.driver.memory", "2g") \
                .getOrCreate()
            
            self.spark.sparkContext.setLogLevel("WARN")
            logger.info(f"Spark session created: {self.spark.version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create Spark session: {e}")
            return False
    
    def _init_kafka(self) -> bool:
        """Initialize Kafka producer."""
        if not KAFKA_AVAILABLE:
            logger.error("kafka-python not available")
            return False
        
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.config.kafka_bootstrap_servers,
                value_serializer=lambda v: v,  # Already serialized
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks='all',
                retries=3,
                max_in_flight_requests_per_connection=1,
                compression_type='gzip'
            )
            logger.info(f"Kafka producer connected to {self.config.kafka_bootstrap_servers}")
            return True
            
        except NoBrokersAvailable:
            logger.error(f"No Kafka brokers available at {self.config.kafka_bootstrap_servers}")
            return False
        except Exception as e:
            logger.error(f"Failed to create Kafka producer: {e}")
            return False
    
    def _init_generator(self) -> bool:
        """Initialize the synthetic data generator."""
        try:
            # Import here to avoid circular imports
            from climaxtreme.preprocessing.spark.synthetic_generator import (
                SyntheticClimateGenerator,
                SyntheticConfig
            )
            
            config = SyntheticConfig(
                seed=int(time.time()) % 10000,  # Different seed each run
                hourly_interpolation=True
            )
            
            self.generator = SyntheticClimateGenerator(self.spark, config)
            logger.info("SyntheticClimateGenerator initialized")
            return True
            
        except ImportError as e:
            logger.error(f"Failed to import SyntheticClimateGenerator: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize generator: {e}")
            return False
    
    def _load_base_data(self):
        """Load and sample the base temperature data."""
        try:
            input_path = self.config.input_csv_path
            
            # Check if file exists
            if not os.path.exists(input_path) and not input_path.startswith("/"):
                # Try common paths
                possible_paths = [
                    "/app/DATA/GlobalLandTemperaturesByCity.csv",
                    "/data/GlobalLandTemperaturesByCity.csv",
                    "DATA/GlobalLandTemperaturesByCity.csv",
                    "../DATA/GlobalLandTemperaturesByCity.csv"
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        input_path = path
                        break
            
            logger.info(f"Loading data from: {input_path}")
            
            # Read CSV
            df = self.spark.read.csv(
                input_path,
                header=True,
                inferSchema=True
            )
            
            # Sample cities if specified
            if self.config.sample_cities:
                df = df.filter(F.col("City").isin(self.config.sample_cities))
            else:
                # Get unique cities and sample
                cities = df.select("City").distinct().limit(self.config.max_cities)
                city_list = [row.City for row in cities.collect()]
                df = df.filter(F.col("City").isin(city_list))
            
            # Sample time range (last few years for variety)
            df = df.filter(F.col("dt") >= "2010-01-01")
            
            # Limit records for streaming demo
            df = df.limit(1000)
            
            logger.info(f"Loaded {df.count()} base records")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load base data: {e}")
            return None
    
    def _send_progress(self, phase: str, progress: float, message: str):
        """Send progress update to Kafka."""
        if self.producer:
            try:
                event = serialize_progress_event(phase, progress, message, self.stats)
                self.producer.send(
                    self.config.topic_progress,
                    key="progress",
                    value=event
                )
            except Exception as e:
                logger.warning(f"Failed to send progress: {e}")
    
    def _stream_dataframe_to_kafka(
        self,
        df,  # Spark DataFrame
        topic: str,
        serializer,
        event_type: str
    ) -> int:
        """
        Convert DataFrame rows to events and send to Kafka.
        
        Args:
            df: Spark DataFrame
            topic: Kafka topic
            serializer: Function to serialize rows
            event_type: Type of event for stats
            
        Returns:
            Number of events sent
        """
        sent_count = 0
        
        try:
            # Collect in batches to avoid memory issues
            total_rows = df.count()
            batch_size = self.config.batch_size
            
            # Convert to list of dicts
            rows = df.collect()
            
            for i, row in enumerate(rows):
                if self._stop_event.is_set():
                    break
                
                try:
                    row_dict = row.asDict()
                    event_bytes = serializer(row_dict)
                    
                    # Determine key (city or storm_id)
                    key = row_dict.get("city", row_dict.get("storm_id", "unknown"))
                    
                    self.producer.send(
                        topic,
                        key=str(key),
                        value=event_bytes
                    )
                    
                    sent_count += 1
                    self.stats["last_event_time"] = datetime.now()
                    
                    # Update stats
                    if event_type == "weather":
                        self.stats["weather_events_sent"] += 1
                    elif event_type == "alert":
                        self.stats["alerts_sent"] += 1
                    elif event_type == "storm":
                        self.stats["storms_sent"] += 1
                    
                    # Delay between events for realistic streaming
                    if i % batch_size == 0:
                        self.producer.flush()
                        progress = (i + 1) / total_rows
                        self._send_progress(
                            f"streaming_{event_type}",
                            progress,
                            f"Sent {sent_count} {event_type} events"
                        )
                        time.sleep(self.config.delay_between_batches)
                        
                except Exception as e:
                    self.stats["errors"] += 1
                    logger.warning(f"Failed to send event: {e}")
            
            self.producer.flush()
            
        except Exception as e:
            logger.error(f"Failed to stream DataFrame: {e}")
            self.stats["errors"] += 1
        
        return sent_count
    
    def start(self) -> bool:
        """
        Start the Spark-Kafka streaming pipeline.
        
        Returns:
            True if started successfully
        """
        if self._running:
            logger.warning("Producer already running")
            return False
        
        logger.info("=" * 60)
        logger.info("Starting Spark-Kafka Streaming Producer")
        logger.info("=" * 60)
        
        # Initialize components
        self._send_progress("initialization", 0.1, "Initializing Spark...")
        
        if not self._init_spark():
            return False
        
        self._send_progress("initialization", 0.2, "Connecting to Kafka...")
        
        if not self._init_kafka():
            return False
        
        self._send_progress("initialization", 0.3, "Initializing generator...")
        
        if not self._init_generator():
            return False
        
        # Load base data
        self._send_progress("loading", 0.4, "Loading temperature data...")
        
        base_df = self._load_base_data()
        if base_df is None:
            return False
        
        self._running = True
        self._stop_event.clear()
        self.stats["start_time"] = datetime.now()
        
        try:
            # Generate synthetic data
            self._send_progress("generating", 0.5, "Generating synthetic data with Spark...")
            
            synthetic_df, storm_tracks = self.generator.generate_full_synthetic_dataset(
                base_df,
                generate_storms=True
            )
            
            self._send_progress("generating", 0.7, "Synthetic data generated!")
            
            # Stream weather events
            logger.info("Streaming weather events to Kafka...")
            self._send_progress("streaming_weather", 0.0, "Starting weather stream...")
            
            weather_count = self._stream_dataframe_to_kafka(
                synthetic_df,
                self.config.topic_weather,
                serialize_weather_event,
                "weather"
            )
            
            logger.info(f"Sent {weather_count} weather events")
            
            # Stream alerts
            if not self._stop_event.is_set():
                alerts_df = synthetic_df.filter(F.col("alert_active") == True)
                alert_count = alerts_df.count()
                
                if alert_count > 0:
                    logger.info(f"Streaming {alert_count} alerts to Kafka...")
                    self._send_progress("streaming_alerts", 0.0, f"Streaming {alert_count} alerts...")
                    
                    self._stream_dataframe_to_kafka(
                        alerts_df,
                        self.config.topic_alerts,
                        serialize_alert_event,
                        "alert"
                    )
            
            # Stream storm tracks
            if not self._stop_event.is_set() and storm_tracks is not None:
                storm_count = storm_tracks.count()
                
                if storm_count > 0:
                    logger.info(f"Streaming {storm_count} storm tracks to Kafka...")
                    self._send_progress("streaming_storms", 0.0, f"Streaming {storm_count} storm tracks...")
                    
                    self._stream_dataframe_to_kafka(
                        storm_tracks,
                        self.config.topic_storms,
                        serialize_storm_event,
                        "storm"
                    )
            
            # Complete
            self._send_progress("complete", 1.0, "Streaming complete!")
            
            logger.info("=" * 60)
            logger.info("Spark-Kafka Streaming Complete!")
            logger.info(f"Weather events: {self.stats['weather_events_sent']}")
            logger.info(f"Alerts: {self.stats['alerts_sent']}")
            logger.info(f"Storms: {self.stats['storms_sent']}")
            logger.info(f"Errors: {self.stats['errors']}")
            logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            self._send_progress("error", 0.0, f"Error: {str(e)}")
            return False
            
        finally:
            self._running = False
    
    def stop(self):
        """Stop the streaming pipeline."""
        logger.info("Stopping Spark-Kafka producer...")
        self._stop_event.set()
        self._running = False
        
        if self.producer:
            self.producer.flush()
            self.producer.close()
            self.producer = None
        
        if self.spark:
            self.spark.stop()
            self.spark = None
    
    def is_running(self) -> bool:
        """Check if producer is running."""
        return self._running
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        stats = self.stats.copy()
        if stats["start_time"]:
            elapsed = (datetime.now() - stats["start_time"]).total_seconds()
            stats["elapsed_seconds"] = elapsed
            total_events = stats["weather_events_sent"] + stats["alerts_sent"] + stats["storms_sent"]
            stats["events_per_second"] = total_events / elapsed if elapsed > 0 else 0
        return stats


# ============================================================================
# Continuous Streaming Mode
# ============================================================================

class ContinuousSparkKafkaProducer(SparkKafkaStreamingProducer):
    """
    Extended producer that continuously generates and streams data
    in a loop, simulating real-time data ingestion.
    """
    
    def __init__(self, config: Optional[SparkKafkaConfig] = None):
        super().__init__(config)
        self._loop_count = 0
        self._loop_delay = 30  # seconds between loops
    
    def start_continuous(self, max_loops: int = 0):
        """
        Start continuous streaming.
        
        Args:
            max_loops: Maximum loops (0 = infinite)
        """
        logger.info("Starting continuous Spark-Kafka streaming...")
        
        # Initialize once
        if not self._init_spark():
            return
        if not self._init_kafka():
            return
        if not self._init_generator():
            return
        
        self._running = True
        self._stop_event.clear()
        self.stats["start_time"] = datetime.now()
        
        while not self._stop_event.is_set():
            if max_loops > 0 and self._loop_count >= max_loops:
                break
            
            try:
                self._loop_count += 1
                logger.info(f"Starting streaming loop {self._loop_count}")
                
                # Load fresh data sample
                base_df = self._load_base_data()
                if base_df is None:
                    continue
                
                # Generate and stream
                synthetic_df, storm_tracks = self.generator.generate_full_synthetic_dataset(
                    base_df,
                    generate_storms=True
                )
                
                # Stream all data types
                self._stream_dataframe_to_kafka(
                    synthetic_df,
                    self.config.topic_weather,
                    serialize_weather_event,
                    "weather"
                )
                
                alerts_df = synthetic_df.filter(F.col("alert_active") == True)
                if alerts_df.count() > 0:
                    self._stream_dataframe_to_kafka(
                        alerts_df,
                        self.config.topic_alerts,
                        serialize_alert_event,
                        "alert"
                    )
                
                if storm_tracks is not None and storm_tracks.count() > 0:
                    self._stream_dataframe_to_kafka(
                        storm_tracks,
                        self.config.topic_storms,
                        serialize_storm_event,
                        "storm"
                    )
                
                # Wait before next loop
                logger.info(f"Loop {self._loop_count} complete, waiting {self._loop_delay}s...")
                self._stop_event.wait(self._loop_delay)
                
            except Exception as e:
                logger.error(f"Error in loop {self._loop_count}: {e}")
                self.stats["errors"] += 1
                time.sleep(5)
        
        self.stop()


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """Main entry point for CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Spark-Kafka Streaming Producer")
    parser.add_argument("--kafka-servers", default="localhost:9092",
                       help="Kafka bootstrap servers")
    parser.add_argument("--input-csv", default="/data/GlobalLandTemperaturesByCity.csv",
                       help="Input CSV path")
    parser.add_argument("--max-cities", type=int, default=20,
                       help="Maximum cities to process")
    parser.add_argument("--batch-size", type=int, default=100,
                       help="Batch size for streaming")
    parser.add_argument("--delay", type=float, default=0.5,
                       help="Delay between batches (seconds)")
    parser.add_argument("--continuous", action="store_true",
                       help="Run in continuous mode")
    parser.add_argument("--max-loops", type=int, default=0,
                       help="Max loops in continuous mode (0=infinite)")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create config
    config = SparkKafkaConfig(
        kafka_bootstrap_servers=args.kafka_servers,
        input_csv_path=args.input_csv,
        max_cities=args.max_cities,
        batch_size=args.batch_size,
        delay_between_batches=args.delay
    )
    
    if args.continuous:
        producer = ContinuousSparkKafkaProducer(config)
        try:
            producer.start_continuous(args.max_loops)
        except KeyboardInterrupt:
            producer.stop()
    else:
        producer = SparkKafkaStreamingProducer(config)
        try:
            producer.start()
        finally:
            producer.stop()


if __name__ == "__main__":
    main()
