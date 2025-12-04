"""
Structured Streaming Demo for Real-Time Climate Data Simulation.

This module demonstrates how to use Spark Structured Streaming to simulate
real-time climate data generation and processing. This is useful for:
- Testing real-time alert systems
- Demonstrating streaming analytics capabilities
- Simulating sensor data feeds

Usage:
    # Run streaming simulation
    python -m climaxtreme.streaming.streaming_demo --duration 60
    
    # Or from CLI
    climaxtreme stream-demo --duration 60
"""

import time
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class StreamingConfig:
    """Configuration for streaming simulation."""
    
    # Output settings
    output_path: str = "DATA/streaming"
    checkpoint_path: str = "DATA/streaming/checkpoints"
    
    # Rate settings
    records_per_second: int = 10
    batch_interval_seconds: int = 5
    
    # Simulation settings
    n_cities: int = 10
    seed: int = 42
    
    # Alert thresholds
    heat_alert_threshold: float = 35.0
    wind_alert_threshold: float = 60.0
    rain_alert_threshold: float = 25.0


class StreamingSimulator:
    """
    Simulates streaming climate data for testing and demonstration.
    
    Can generate data to:
    - File system (for Structured Streaming file source)
    - Kafka (if available)
    - Memory (for testing)
    """
    
    def __init__(self, config: Optional[StreamingConfig] = None):
        """Initialize the streaming simulator."""
        self.config = config or StreamingConfig()
        np.random.seed(self.config.seed)
        
        # Initialize city metadata
        self.cities = self._initialize_cities()
        
        # Current state for Markov chain
        self.current_state: Dict[str, Dict[str, Any]] = {}
        self._initialize_state()
    
    def _initialize_cities(self) -> list:
        """Initialize city metadata for simulation."""
        cities = [
            {'name': 'New York', 'country': 'USA', 'lat': 40.71, 'lon': -74.01, 'zone': 'Temperate'},
            {'name': 'London', 'country': 'UK', 'lat': 51.51, 'lon': -0.13, 'zone': 'Temperate'},
            {'name': 'Tokyo', 'country': 'Japan', 'lat': 35.68, 'lon': 139.69, 'zone': 'Subtropical'},
            {'name': 'Sydney', 'country': 'Australia', 'lat': -33.87, 'lon': 151.21, 'zone': 'Subtropical'},
            {'name': 'Mumbai', 'country': 'India', 'lat': 19.08, 'lon': 72.88, 'zone': 'Tropical'},
            {'name': 'Cairo', 'country': 'Egypt', 'lat': 30.04, 'lon': 31.24, 'zone': 'Arid'},
            {'name': 'Moscow', 'country': 'Russia', 'lat': 55.75, 'lon': 37.62, 'zone': 'Continental'},
            {'name': 'São Paulo', 'country': 'Brazil', 'lat': -23.55, 'lon': -46.64, 'zone': 'Subtropical'},
            {'name': 'Beijing', 'country': 'China', 'lat': 39.90, 'lon': 116.41, 'zone': 'Continental'},
            {'name': 'Lagos', 'country': 'Nigeria', 'lat': 6.52, 'lon': 3.38, 'zone': 'Tropical'},
        ]
        return cities[:self.config.n_cities]
    
    def _initialize_state(self) -> None:
        """Initialize Markov chain state for each city."""
        for city in self.cities:
            # Base temperature depends on latitude and zone
            base_temp = self._get_base_temperature(city['lat'], city['zone'])
            
            self.current_state[city['name']] = {
                'temperature': base_temp,
                'humidity': 60.0,
                'pressure': 1013.25,
                'wind_speed': 10.0,
                'rain_state': 'dry',  # Markov state: dry, light, moderate, heavy
                'rain_mm': 0.0
            }
    
    def _get_base_temperature(self, latitude: float, zone: str) -> float:
        """Get base temperature for a location."""
        zone_temps = {
            'Tropical': 28.0,
            'Subtropical': 22.0,
            'Temperate': 15.0,
            'Continental': 10.0,
            'Polar': -5.0,
            'Arid': 30.0
        }
        return zone_temps.get(zone, 15.0)
    
    def _update_state(self, city_name: str, current_time: datetime) -> Dict[str, Any]:
        """Update state using Markov chain transitions."""
        state = self.current_state[city_name]
        
        # Time-based factors
        hour = current_time.hour
        month = current_time.month
        
        # Diurnal temperature cycle
        diurnal = 8 * np.sin(2 * np.pi * (hour - 6) / 24)
        
        # Seasonal adjustment (Northern Hemisphere assumption)
        seasonal = 10 * np.sin(2 * np.pi * (month - 4) / 12)
        
        # Random walk for temperature
        temp_change = np.random.normal(0, 0.5)
        new_temp = state['temperature'] + temp_change + 0.1 * (diurnal - state['temperature'] + seasonal)
        
        # Humidity (inverse correlation with temperature)
        humidity_change = np.random.normal(0, 2)
        new_humidity = np.clip(state['humidity'] + humidity_change - 0.5 * temp_change, 0, 100)
        
        # Pressure random walk
        pressure_change = np.random.normal(0, 1)
        new_pressure = np.clip(state['pressure'] + pressure_change, 970, 1050)
        
        # Wind speed (Weibull-like perturbation)
        wind_change = np.random.normal(0, 2)
        new_wind = max(0, state['wind_speed'] + wind_change)
        
        # Rain state transitions (Markov chain)
        rain_states = ['dry', 'light', 'moderate', 'heavy']
        transition_matrix = {
            'dry': [0.9, 0.08, 0.02, 0.0],
            'light': [0.3, 0.5, 0.15, 0.05],
            'moderate': [0.1, 0.3, 0.4, 0.2],
            'heavy': [0.05, 0.15, 0.4, 0.4]
        }
        
        current_rain_state = state['rain_state']
        probabilities = transition_matrix[current_rain_state]
        new_rain_state = np.random.choice(rain_states, p=probabilities)
        
        # Rain amount based on state
        rain_amounts = {'dry': 0, 'light': 2, 'moderate': 8, 'heavy': 20}
        base_rain = rain_amounts[new_rain_state]
        new_rain = max(0, base_rain + np.random.exponential(base_rain * 0.5)) if base_rain > 0 else 0
        
        # Update state
        self.current_state[city_name] = {
            'temperature': new_temp,
            'humidity': new_humidity,
            'pressure': new_pressure,
            'wind_speed': new_wind,
            'rain_state': new_rain_state,
            'rain_mm': new_rain
        }
        
        return self.current_state[city_name]
    
    def _generate_alerts(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate alerts based on current state."""
        alerts = {
            'alert_level': 'green',
            'alert_type': 'none',
            'alert_message': ''
        }
        
        # Heat alert
        if state['temperature'] >= 45:
            alerts = {'alert_level': 'red', 'alert_type': 'heat', 
                     'alert_message': f"Extreme heat warning: {state['temperature']:.1f}°C"}
        elif state['temperature'] >= 40:
            alerts = {'alert_level': 'orange', 'alert_type': 'heat',
                     'alert_message': f"Heat warning: {state['temperature']:.1f}°C"}
        elif state['temperature'] >= self.config.heat_alert_threshold:
            alerts = {'alert_level': 'yellow', 'alert_type': 'heat',
                     'alert_message': f"Heat advisory: {state['temperature']:.1f}°C"}
        
        # Wind alert (can override heat if more severe)
        if state['wind_speed'] >= 120:
            alerts = {'alert_level': 'red', 'alert_type': 'wind',
                     'alert_message': f"Extreme wind warning: {state['wind_speed']:.1f} km/h"}
        elif state['wind_speed'] >= 90 and alerts['alert_level'] != 'red':
            alerts = {'alert_level': 'orange', 'alert_type': 'wind',
                     'alert_message': f"Wind warning: {state['wind_speed']:.1f} km/h"}
        elif state['wind_speed'] >= self.config.wind_alert_threshold and alerts['alert_level'] == 'green':
            alerts = {'alert_level': 'yellow', 'alert_type': 'wind',
                     'alert_message': f"Wind advisory: {state['wind_speed']:.1f} km/h"}
        
        # Rain/flood alert
        if state['rain_mm'] >= 100:
            alerts = {'alert_level': 'red', 'alert_type': 'flood',
                     'alert_message': f"Flood warning: {state['rain_mm']:.1f} mm"}
        elif state['rain_mm'] >= 50 and alerts['alert_level'] not in ['red']:
            alerts = {'alert_level': 'orange', 'alert_type': 'flood',
                     'alert_message': f"Heavy rain warning: {state['rain_mm']:.1f} mm"}
        
        return alerts
    
    def generate_record(self, timestamp: Optional[datetime] = None) -> list:
        """Generate a batch of streaming records for all cities."""
        if timestamp is None:
            timestamp = datetime.now()
        
        records = []
        
        for city in self.cities:
            # Update state
            state = self._update_state(city['name'], timestamp)
            
            # Generate alerts
            alerts = self._generate_alerts(state)
            
            record = {
                'timestamp': timestamp.isoformat(),
                'city': city['name'],
                'country': city['country'],
                'latitude': city['lat'],
                'longitude': city['lon'],
                'climate_zone': city['zone'],
                'temperature_c': round(state['temperature'], 2),
                'humidity_pct': round(state['humidity'], 2),
                'pressure_hpa': round(state['pressure'], 2),
                'wind_speed_kmh': round(state['wind_speed'], 2),
                'rain_mm': round(state['rain_mm'], 2),
                'rain_state': state['rain_state'],
                'alert_level': alerts['alert_level'],
                'alert_type': alerts['alert_type'],
                'alert_message': alerts['alert_message']
            }
            
            records.append(record)
        
        return records
    
    def write_batch_to_file(self, records: list, output_dir: Path) -> str:
        """Write a batch of records to a JSON file."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = f"batch_{timestamp}.json"
        filepath = output_dir / filename
        
        with open(filepath, 'w') as f:
            for record in records:
                f.write(json.dumps(record) + '\n')
        
        return str(filepath)
    
    def run_simulation(
        self, 
        duration_seconds: int = 60,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run streaming simulation for a specified duration.
        
        Args:
            duration_seconds: How long to run the simulation
            output_dir: Directory to write output files
        
        Returns:
            Statistics about the simulation run
        """
        output_dir = Path(output_dir or self.config.output_path)
        
        logger.info(f"Starting streaming simulation for {duration_seconds} seconds")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Records per batch: {len(self.cities)} cities")
        logger.info(f"Batch interval: {self.config.batch_interval_seconds} seconds")
        
        start_time = time.time()
        batch_count = 0
        record_count = 0
        alert_count = 0
        
        simulated_time = datetime.now()
        
        try:
            while (time.time() - start_time) < duration_seconds:
                # Generate batch
                records = self.generate_record(simulated_time)
                
                # Write to file
                filepath = self.write_batch_to_file(records, output_dir)
                
                batch_count += 1
                record_count += len(records)
                alert_count += sum(1 for r in records if r['alert_level'] != 'green')
                
                logger.info(f"Batch {batch_count}: {len(records)} records written to {filepath}")
                
                # Advance simulated time
                simulated_time += timedelta(minutes=10)
                
                # Wait for next batch
                time.sleep(self.config.batch_interval_seconds)
                
        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")
        
        elapsed = time.time() - start_time
        
        stats = {
            'duration_seconds': elapsed,
            'batch_count': batch_count,
            'record_count': record_count,
            'alert_count': alert_count,
            'records_per_second': record_count / elapsed if elapsed > 0 else 0,
            'output_directory': str(output_dir)
        }
        
        logger.info(f"Simulation completed: {record_count} records in {batch_count} batches")
        
        return stats


def create_spark_streaming_reader(
    spark,
    input_path: str,
    schema: Optional[Any] = None
):
    """
    Create a Spark Structured Streaming reader for JSON files.
    
    Args:
        spark: SparkSession instance
        input_path: Directory to read JSON files from
        schema: Optional schema for the JSON data
    
    Returns:
        Streaming DataFrame
    """
    from pyspark.sql.types import (
        StructType, StructField, StringType, DoubleType, TimestampType
    )
    
    if schema is None:
        schema = StructType([
            StructField("timestamp", StringType(), True),
            StructField("city", StringType(), True),
            StructField("country", StringType(), True),
            StructField("latitude", DoubleType(), True),
            StructField("longitude", DoubleType(), True),
            StructField("climate_zone", StringType(), True),
            StructField("temperature_c", DoubleType(), True),
            StructField("humidity_pct", DoubleType(), True),
            StructField("pressure_hpa", DoubleType(), True),
            StructField("wind_speed_kmh", DoubleType(), True),
            StructField("rain_mm", DoubleType(), True),
            StructField("rain_state", StringType(), True),
            StructField("alert_level", StringType(), True),
            StructField("alert_type", StringType(), True),
            StructField("alert_message", StringType(), True),
        ])
    
    return (
        spark
        .readStream
        .schema(schema)
        .json(input_path)
    )


def create_alert_aggregation_query(streaming_df, output_path: str, checkpoint_path: str):
    """
    Create a streaming query that aggregates alerts.
    
    Args:
        streaming_df: Input streaming DataFrame
        output_path: Path to write aggregated results
        checkpoint_path: Path for checkpoints
    
    Returns:
        StreamingQuery
    """
    from pyspark.sql import functions as F
    
    # Aggregate alerts by city and level
    aggregated = (
        streaming_df
        .groupBy("city", "alert_level")
        .agg(
            F.count("*").alias("alert_count"),
            F.max("temperature_c").alias("max_temperature"),
            F.max("wind_speed_kmh").alias("max_wind_speed"),
            F.sum("rain_mm").alias("total_rain")
        )
    )
    
    # Write to parquet with append mode
    query = (
        aggregated
        .writeStream
        .outputMode("complete")
        .format("parquet")
        .option("path", output_path)
        .option("checkpointLocation", checkpoint_path)
        .start()
    )
    
    return query


def run_streaming_demo(
    duration_seconds: int = 60,
    output_path: str = "DATA/streaming",
    n_cities: int = 10
) -> Dict[str, Any]:
    """
    Run the streaming demo simulation.
    
    Args:
        duration_seconds: Duration to run simulation
        output_path: Output directory for streaming data
        n_cities: Number of cities to simulate
    
    Returns:
        Simulation statistics
    """
    config = StreamingConfig(
        output_path=output_path,
        n_cities=n_cities,
        batch_interval_seconds=5
    )
    
    simulator = StreamingSimulator(config)
    return simulator.run_simulation(duration_seconds, output_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Streaming Climate Data Demo")
    parser.add_argument("--duration", type=int, default=60, help="Duration in seconds")
    parser.add_argument("--output", type=str, default="DATA/streaming", help="Output directory")
    parser.add_argument("--cities", type=int, default=10, help="Number of cities")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    stats = run_streaming_demo(
        duration_seconds=args.duration,
        output_path=args.output,
        n_cities=args.cities
    )
    
    print("\nSimulation Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
