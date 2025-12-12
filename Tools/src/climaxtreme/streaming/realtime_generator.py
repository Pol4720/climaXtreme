"""
Real-Time Streaming Synthetic Data Generator using PySpark.

This module provides on-demand synthetic data generation for Big Data pipelines:
- Real-time forecasting using Spark Structured Streaming
- Dashboard streaming displays with HDFS integration
- Alert system testing at scale

Key Features:
- Uses PySpark for distributed data generation
- Integrates with existing synthetic_generator.py for consistency
- Reads/writes from HDFS following Big Data architecture
- Supports micro-batch and continuous streaming modes
- Statistical validation using Spark SQL

Architecture:
    HDFS Historical Data -> Spark Streaming -> Synthetic Generation -> HDFS/Dashboard
"""

import logging
import math
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime, timedelta

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, FloatType,
    TimestampType, DoubleType
)

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class StreamingConfig:
    """Configuration for PySpark streaming generator."""
    
    # Spark settings
    app_name: str = "climaXtreme-StreamingGenerator"
    
    # HDFS paths
    hdfs_base: str = "hdfs://climaxtreme-namenode:9000"
    historical_data_path: str = "/data/climaxtreme/synthetic/synthetic_hourly.parquet"
    checkpoint_location: str = "/data/climaxtreme/streaming/checkpoints"
    output_path: str = "/data/climaxtreme/streaming/output"
    
    # Streaming settings
    trigger_interval: str = "10 seconds"
    output_mode: str = "append"
    
    # Generation settings  
    forecast_horizon_hours: int = 24
    cities_per_batch: int = 50
    seed: int = 42
    
    # Weather parameters (same as synthetic_generator)
    rain_gamma_scale: float = 5.0
    wind_weibull_shape: float = 2.0
    wind_weibull_scale: float = 15.0
    humidity_mean: float = 65.0
    humidity_std: float = 15.0
    pressure_mean: float = 1013.25
    pressure_std: float = 10.0
    
    # Alert thresholds
    heat_threshold_yellow: float = 35.0
    heat_threshold_orange: float = 40.0
    heat_threshold_red: float = 45.0
    wind_threshold_yellow: float = 60.0
    wind_threshold_orange: float = 90.0
    wind_threshold_red: float = 120.0


# ============================================================================
# Climate Zone Parameters
# ============================================================================

CLIMATE_ZONES = {
    'TROPICAL': {'lat_range': (-23.5, 23.5), 'diurnal_amp': 10.0, 'rain_prob': 0.35},
    'SUBTROPICAL': {'lat_range': (23.5, 35), 'diurnal_amp': 12.0, 'rain_prob': 0.20},
    'TEMPERATE': {'lat_range': (35, 55), 'diurnal_amp': 14.0, 'rain_prob': 0.25},
    'CONTINENTAL': {'lat_range': (55, 66.5), 'diurnal_amp': 16.0, 'rain_prob': 0.20},
    'POLAR': {'lat_range': (66.5, 90), 'diurnal_amp': 8.0, 'rain_prob': 0.15},
}


# ============================================================================
# UDF Functions
# ============================================================================

def get_climate_zone_udf():
    """UDF to classify climate zone based on latitude."""
    def classify_zone(lat: float) -> str:
        if lat is None:
            return "TEMPERATE"
        abs_lat = abs(lat)
        if abs_lat <= 23.5:
            return "TROPICAL"
        elif abs_lat <= 35:
            return "SUBTROPICAL"
        elif abs_lat <= 55:
            return "TEMPERATE"
        elif abs_lat <= 66.5:
            return "CONTINENTAL"
        else:
            return "POLAR"
    return F.udf(classify_zone, StringType())


def get_season_udf():
    """UDF to determine season based on month and hemisphere."""
    def get_season(month: int, lat: float) -> str:
        if month is None:
            return "UNKNOWN"
        
        if month in [12, 1, 2]:
            season = "WINTER"
        elif month in [3, 4, 5]:
            season = "SPRING"
        elif month in [6, 7, 8]:
            season = "SUMMER"
        else:
            season = "AUTUMN"
        
        # Flip for southern hemisphere
        if lat is not None and lat < 0:
            flip = {"SUMMER": "WINTER", "WINTER": "SUMMER", 
                   "SPRING": "AUTUMN", "AUTUMN": "SPRING"}
            season = flip.get(season, season)
        
        return season
    return F.udf(get_season, StringType())


def get_diurnal_amplitude_udf():
    """UDF to calculate diurnal temperature amplitude."""
    def calc_amplitude(climate_zone: str, season: str) -> float:
        base_amp = {
            "TROPICAL": 10.0, "SUBTROPICAL": 12.0, "TEMPERATE": 14.0,
            "CONTINENTAL": 16.0, "POLAR": 8.0
        }.get(climate_zone, 12.0)
        
        season_factor = {
            "SUMMER": 1.2, "WINTER": 0.8, "SPRING": 1.0, "AUTUMN": 1.0
        }.get(season, 1.0)
        
        return base_amp * season_factor
    return F.udf(calc_amplitude, FloatType())


# ============================================================================
# Main PySpark Streaming Generator
# ============================================================================

class SparkStreamingGenerator:
    """
    Real-time synthetic data generator using PySpark.
    
    This class uses Spark for distributed data generation,
    integrating with HDFS and the existing Big Data architecture.
    
    Features:
    - Learns from historical data in HDFS
    - Generates synthetic data using Spark transformations
    - Supports batch and streaming modes
    - Compatible with synthetic_generator.py models
    """
    
    def __init__(
        self, 
        spark: SparkSession,
        config: Optional[StreamingConfig] = None
    ):
        """
        Initialize the Spark streaming generator.
        
        Args:
            spark: Active SparkSession
            config: Generator configuration
        """
        self.spark = spark
        self.config = config or StreamingConfig()
        self.seed = self.config.seed
        
        # Cache for historical statistics
        self._historical_stats_df: Optional[DataFrame] = None
        self._city_base_df: Optional[DataFrame] = None
        
        logger.info(f"SparkStreamingGenerator initialized with seed={self.seed}")
    
    def load_historical_data(self, path: Optional[str] = None) -> DataFrame:
        """
        Load historical data from HDFS.
        
        Args:
            path: HDFS path to historical data (parquet)
            
        Returns:
            DataFrame with historical climate data
        """
        hdfs_path = path or f"{self.config.hdfs_base}{self.config.historical_data_path}"
        
        logger.info(f"Loading historical data from: {hdfs_path}")
        
        try:
            df = self.spark.read.parquet(hdfs_path)
            logger.info(f"Loaded {df.count()} historical records")
            return df
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            raise
    
    def compute_historical_statistics(self, historical_df: DataFrame) -> DataFrame:
        """
        Compute per-city statistics from historical data.
        
        Args:
            historical_df: Historical temperature data
            
        Returns:
            DataFrame with city-level statistics
        """
        logger.info("Computing historical statistics...")
        
        # Detect column names dynamically
        cols = historical_df.columns
        
        # Find temperature column
        temp_col = None
        for candidate in ['temperature_hourly', 'avg_temperature', 'AverageTemperature', 'temperature']:
            if candidate in cols:
                temp_col = candidate
                break
        if temp_col is None:
            raise ValueError(f"No temperature column found. Available columns: {cols}")
        
        # Find city column
        city_col = None
        for candidate in ['city', 'City', 'location']:
            if candidate in cols:
                city_col = candidate
                break
        if city_col is None:
            raise ValueError(f"No city column found. Available columns: {cols}")
        
        # Find country column (optional)
        country_col = None
        for candidate in ['country', 'Country']:
            if candidate in cols:
                country_col = candidate
                break
        
        # Find lat/lon columns
        lat_col = None
        for candidate in ['lat_decimal', 'latitude', 'Latitude']:
            if candidate in cols:
                lat_col = candidate
                break
        
        lon_col = None
        for candidate in ['lon_decimal', 'longitude', 'Longitude']:
            if candidate in cols:
                lon_col = candidate
                break
        
        # Build groupBy columns
        group_cols = [city_col]
        if country_col:
            group_cols.append(country_col)
        
        # Build aggregation expressions
        agg_exprs = [
            F.avg(temp_col).alias("temp_mean"),
            F.stddev(temp_col).alias("temp_std"),
            F.min(temp_col).alias("temp_min"),
            F.max(temp_col).alias("temp_max"),
            F.count("*").alias("record_count")
        ]
        
        if lat_col:
            agg_exprs.append(F.first(lat_col).alias("latitude"))
        if lon_col:
            agg_exprs.append(F.first(lon_col).alias("longitude"))
        
        # Compute statistics per city
        stats_df = historical_df.groupBy(*group_cols).agg(*agg_exprs)
        
        # Add climate zone classification if we have latitude
        if lat_col or "latitude" in stats_df.columns:
            climate_zone_udf = get_climate_zone_udf()
            lat_col_final = "latitude" if "latitude" in stats_df.columns else lat_col
            stats_df = stats_df.withColumn("climate_zone", climate_zone_udf(F.col(lat_col_final)))
        else:
            stats_df = stats_df.withColumn("climate_zone", F.lit("TEMPERATE"))
        
        # Rename columns for consistency
        if city_col != "city":
            stats_df = stats_df.withColumnRenamed(city_col, "city")
        if country_col and country_col != "country":
            stats_df = stats_df.withColumnRenamed(country_col, "country")
        
        # Add country if not present
        if "country" not in stats_df.columns:
            stats_df = stats_df.withColumn("country", F.lit("Unknown"))
        
        # Add coordinates if not present
        if "latitude" not in stats_df.columns:
            stats_df = stats_df.withColumn("latitude", F.lit(0.0))
        if "longitude" not in stats_df.columns:
            stats_df = stats_df.withColumn("longitude", F.lit(0.0))
        
        # Handle null std
        stats_df = stats_df.withColumn(
            "temp_std",
            F.when(F.col("temp_std").isNull(), 5.0).otherwise(F.col("temp_std"))
        )
        
        self._historical_stats_df = stats_df.cache()
        logger.info(f"Computed statistics for {stats_df.count()} cities")
        
        return self._historical_stats_df
    
    def generate_forecast_batch(
        self,
        cities_df: DataFrame,
        forecast_hours: int = 24,
        base_timestamp: Optional[datetime] = None
    ) -> DataFrame:
        """
        Generate forecast data for a batch of cities.
        
        Args:
            cities_df: DataFrame with city statistics
            forecast_hours: Hours to forecast ahead
            base_timestamp: Starting timestamp (default: now)
            
        Returns:
            DataFrame with synthetic forecast data
        """
        logger.info(f"Generating {forecast_hours}-hour forecast batch...")
        
        if base_timestamp is None:
            base_timestamp = datetime.now()
        
        # Create hour sequence
        hours = list(range(forecast_hours))
        hours_df = self.spark.createDataFrame(
            [(h, base_timestamp + timedelta(hours=h)) for h in hours],
            ["hour_offset", "forecast_timestamp"]
        )
        
        # Cross join cities with hours
        forecast_df = cities_df.crossJoin(hours_df)
        
        # Add time components
        forecast_df = forecast_df.withColumn("hour", F.hour("forecast_timestamp"))
        forecast_df = forecast_df.withColumn("month", F.month("forecast_timestamp"))
        
        # Get season
        season_udf = get_season_udf()
        forecast_df = forecast_df.withColumn(
            "season", 
            season_udf(F.col("month"), F.col("latitude"))
        )
        
        # Get diurnal amplitude
        amplitude_udf = get_diurnal_amplitude_udf()
        forecast_df = forecast_df.withColumn(
            "diurnal_amplitude",
            amplitude_udf(F.col("climate_zone"), F.col("season"))
        )
        
        # Generate temperature with diurnal cycle
        h_max = 14.0  # Peak temperature hour
        forecast_df = forecast_df.withColumn(
            "temperature",
            F.col("temp_mean") + 
            F.col("diurnal_amplitude") * F.sin(2 * math.pi * (F.col("hour") - h_max) / 24.0) +
            F.randn(self.seed) * F.col("temp_std") * 0.3
        )
        
        # Generate weather variables
        forecast_df = self._add_weather_variables(forecast_df)
        
        # Generate alerts
        forecast_df = self._add_alerts(forecast_df)
        
        # Select final columns
        forecast_df = forecast_df.select(
            "city",
            "country",
            "latitude",
            "longitude",
            "climate_zone",
            "forecast_timestamp",
            "hour_offset",
            "temperature",
            "humidity",
            "pressure",
            "wind_speed",
            "rain_mm",
            "rain_state",
            "alert_level",
            "alert_type"
        )
        
        logger.info(f"Generated {forecast_df.count()} forecast records")
        return forecast_df
    
    def _add_weather_variables(self, df: DataFrame) -> DataFrame:
        """Add synthetic weather variables to DataFrame."""
        
        config = self.config
        
        # Rain probability based on climate zone
        df = df.withColumn(
            "rain_probability",
            F.when(F.col("climate_zone") == "TROPICAL", 0.35)
            .when(F.col("climate_zone") == "SUBTROPICAL", 0.20)
            .when(F.col("climate_zone") == "TEMPERATE", 0.25)
            .when(F.col("climate_zone") == "CONTINENTAL", 0.20)
            .when(F.col("climate_zone") == "POLAR", 0.15)
            .otherwise(0.20)
        )
        
        # Rain state (Markov-like with random)
        df = df.withColumn("rain_random", F.rand(self.seed + 1))
        df = df.withColumn(
            "rain_state",
            F.when(F.col("rain_random") < F.col("rain_probability") * 0.5, "heavy")
            .when(F.col("rain_random") < F.col("rain_probability") * 0.8, "moderate")
            .when(F.col("rain_random") < F.col("rain_probability"), "light")
            .otherwise("dry")
        )
        
        # Rain amount
        df = df.withColumn(
            "rain_mm",
            F.when(F.col("rain_state") == "heavy", 
                   20.0 + F.rand(self.seed + 2) * 30)
            .when(F.col("rain_state") == "moderate",
                   8.0 + F.rand(self.seed + 3) * 12)
            .when(F.col("rain_state") == "light",
                   1.0 + F.rand(self.seed + 4) * 5)
            .otherwise(0.0)
        )
        
        # Wind speed (Weibull approximation)
        df = df.withColumn(
            "wind_speed",
            config.wind_weibull_scale * 
            F.pow(-F.log(F.rand(self.seed + 5) + 0.001), 1.0 / config.wind_weibull_shape)
        )
        df = df.withColumn(
            "wind_speed",
            F.when(F.col("wind_speed") > 150, 150.0).otherwise(F.col("wind_speed"))
        )
        
        # Humidity (correlated with rain and temperature)
        df = df.withColumn(
            "humidity",
            F.when(F.col("rain_state") != "dry",
                   config.humidity_mean + 20 + F.randn(self.seed + 6) * 5)
            .otherwise(
                config.humidity_mean - 
                (F.col("temperature") - 15) * 0.5 + 
                F.randn(self.seed + 7) * config.humidity_std
            )
        )
        df = df.withColumn(
            "humidity",
            F.when(F.col("humidity") > 100, 100.0)
            .when(F.col("humidity") < 0, 0.0)
            .otherwise(F.col("humidity"))
        )
        
        # Pressure (lower during rain)
        df = df.withColumn(
            "pressure",
            config.pressure_mean +
            F.when(F.col("rain_state") != "dry", -10).otherwise(0) +
            F.randn(self.seed + 8) * config.pressure_std
        )
        
        # Cleanup
        df = df.drop("rain_probability", "rain_random")
        
        return df
    
    def _add_alerts(self, df: DataFrame) -> DataFrame:
        """Add weather alerts based on conditions."""
        
        config = self.config
        
        # Determine alert level
        df = df.withColumn(
            "alert_level",
            F.when(
                (F.col("temperature") >= config.heat_threshold_red) |
                (F.col("wind_speed") >= config.wind_threshold_red),
                "red"
            )
            .when(
                (F.col("temperature") >= config.heat_threshold_orange) |
                (F.col("wind_speed") >= config.wind_threshold_orange) |
                (F.col("rain_mm") >= 50),
                "orange"
            )
            .when(
                (F.col("temperature") >= config.heat_threshold_yellow) |
                (F.col("wind_speed") >= config.wind_threshold_yellow) |
                (F.col("rain_mm") >= 25),
                "yellow"
            )
            .otherwise("green")
        )
        
        # Determine alert type
        df = df.withColumn(
            "alert_type",
            F.when(F.col("alert_level") == "green", "none")
            .when(F.col("temperature") >= config.heat_threshold_yellow, "heat")
            .when(F.col("wind_speed") >= config.wind_threshold_yellow, "wind")
            .when(F.col("rain_mm") >= 25, "rain")
            .otherwise("weather")
        )
        
        return df
    
    def generate_streaming_forecast(
        self,
        historical_path: Optional[str] = None,
        n_cities: int = 50,
        forecast_hours: int = 24
    ) -> DataFrame:
        """
        Generate streaming forecast from historical data.
        
        This is the main entry point for dashboard integration.
        
        Args:
            historical_path: Path to historical data in HDFS
            n_cities: Number of cities to include
            forecast_hours: Hours to forecast
            
        Returns:
            DataFrame with synthetic forecast data
        """
        # Load and compute statistics if not cached
        if self._historical_stats_df is None:
            historical_df = self.load_historical_data(historical_path)
            self.compute_historical_statistics(historical_df)
        
        # Sample cities
        cities_df = self._historical_stats_df.limit(n_cities)
        
        # Generate forecast
        forecast_df = self.generate_forecast_batch(
            cities_df,
            forecast_hours=forecast_hours
        )
        
        return forecast_df
    
    def write_to_hdfs(
        self,
        df: DataFrame,
        path: Optional[str] = None,
        mode: str = "overwrite",
        partition_by: Optional[List[str]] = None
    ) -> str:
        """
        Write generated data to HDFS.
        
        Args:
            df: DataFrame to write
            path: Output path (default from config)
            mode: Write mode
            partition_by: Partition columns
            
        Returns:
            Output path
        """
        output_path = path or f"{self.config.hdfs_base}{self.config.output_path}"
        
        logger.info(f"Writing to HDFS: {output_path}")
        
        writer = df.write.mode(mode)
        
        if partition_by:
            writer = writer.partitionBy(*partition_by)
        
        writer.parquet(output_path)
        
        logger.info(f"Successfully wrote to {output_path}")
        return output_path
    
    def start_structured_streaming(
        self,
        source_path: str,
        output_path: Optional[str] = None,
        trigger_interval: Optional[str] = None
    ):
        """
        Start Spark Structured Streaming for continuous generation.
        
        Args:
            source_path: Path to watch for new data
            output_path: Path to write streaming output
            trigger_interval: Trigger processing interval
            
        Returns:
            StreamingQuery
        """
        trigger = trigger_interval or self.config.trigger_interval
        output = output_path or f"{self.config.hdfs_base}{self.config.output_path}"
        checkpoint = f"{self.config.hdfs_base}{self.config.checkpoint_location}"
        
        logger.info(f"Starting structured streaming from {source_path}")
        logger.info(f"Trigger interval: {trigger}")
        
        # Define schema for streaming source
        schema = StructType([
            StructField("city", StringType(), True),
            StructField("country", StringType(), True),
            StructField("latitude", DoubleType(), True),
            StructField("longitude", DoubleType(), True),
            StructField("temperature", DoubleType(), True),
            StructField("timestamp", TimestampType(), True)
        ])
        
        # Read stream
        stream_df = (
            self.spark
            .readStream
            .schema(schema)
            .parquet(source_path)
        )
        
        # Process: add weather variables and alerts
        processed_df = self._add_weather_variables(stream_df)
        processed_df = self._add_alerts(processed_df)
        
        # Write stream
        query = (
            processed_df
            .writeStream
            .outputMode(self.config.output_mode)
            .format("parquet")
            .option("path", output)
            .option("checkpointLocation", checkpoint)
            .trigger(processingTime=trigger)
            .start()
        )
        
        logger.info(f"Streaming query started: {query.id}")
        return query


# ============================================================================
# EDA Statistics using Spark SQL
# ============================================================================

class SparkEDAValidator:
    """
    EDA validation for synthetic data using Spark SQL.
    
    Computes distribution statistics and validates data quality
    using distributed Spark operations.
    """
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
    
    def compute_distribution_stats(self, df: DataFrame, column: str) -> Dict[str, float]:
        """Compute distribution statistics for a column."""
        stats = df.select(
            F.count(column).alias("count"),
            F.mean(column).alias("mean"),
            F.stddev(column).alias("std"),
            F.min(column).alias("min"),
            F.max(column).alias("max"),
            F.expr(f"percentile_approx({column}, 0.25)").alias("q25"),
            F.expr(f"percentile_approx({column}, 0.5)").alias("median"),
            F.expr(f"percentile_approx({column}, 0.75)").alias("q75"),
            F.skewness(column).alias("skewness"),
            F.kurtosis(column).alias("kurtosis")
        ).first()
        
        return stats.asDict()
    
    def validate_temperature_distribution(self, df: DataFrame) -> Dict[str, Any]:
        """Validate temperature follows expected distribution."""
        temp_col = 'temperature' if 'temperature' in df.columns else 'avg_temperature'
        
        stats = self.compute_distribution_stats(df, temp_col)
        
        # Check for reasonable values
        is_valid = (
            -60 <= stats['min'] <= 60 and
            -40 <= stats['max'] <= 60 and
            stats['std'] > 0
        )
        
        return {
            'test_name': 'Temperature Distribution',
            'passed': is_valid,
            'statistics': stats
        }
    
    def validate_humidity_bounds(self, df: DataFrame) -> Dict[str, Any]:
        """Validate humidity is within 0-100%."""
        if 'humidity' not in df.columns:
            return {'test_name': 'Humidity Bounds', 'passed': True, 'message': 'No humidity column'}
        
        violations = df.filter(
            (F.col("humidity") < 0) | (F.col("humidity") > 100)
        ).count()
        
        total = df.count()
        violation_rate = violations / total if total > 0 else 0
        
        return {
            'test_name': 'Humidity Bounds',
            'passed': violation_rate < 0.01,
            'violation_rate': violation_rate,
            'violations': violations,
            'total': total
        }
    
    def validate_alert_distribution(self, df: DataFrame) -> Dict[str, Any]:
        """Validate alert distribution is reasonable."""
        if 'alert_level' not in df.columns:
            return {'test_name': 'Alert Distribution', 'passed': True, 'message': 'No alerts column'}
        
        distribution = df.groupBy("alert_level").count().collect()
        dist_dict = {row['alert_level']: row['count'] for row in distribution}
        
        total = sum(dist_dict.values())
        
        # Most should be green
        green_ratio = dist_dict.get('green', 0) / total if total > 0 else 0
        
        return {
            'test_name': 'Alert Distribution',
            'passed': green_ratio > 0.7,  # At least 70% should be green
            'distribution': dist_dict,
            'green_ratio': green_ratio
        }
    
    def run_full_validation(self, df: DataFrame) -> Dict[str, Any]:
        """Run all validation tests."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'record_count': df.count(),
            'validations': []
        }
        
        # Run each validation
        results['validations'].append(self.validate_temperature_distribution(df))
        results['validations'].append(self.validate_humidity_bounds(df))
        results['validations'].append(self.validate_alert_distribution(df))
        
        # Calculate overall score
        passed = sum(1 for v in results['validations'] if v['passed'])
        total = len(results['validations'])
        results['overall_score'] = (passed / total * 100) if total > 0 else 0
        
        return results


# ============================================================================
# Convenience Functions
# ============================================================================

def create_streaming_generator(
    spark: SparkSession,
    config: Optional[StreamingConfig] = None
) -> SparkStreamingGenerator:
    """Factory function to create a streaming generator."""
    return SparkStreamingGenerator(spark, config)


def generate_forecast_from_hdfs(
    spark: SparkSession,
    historical_path: str,
    n_cities: int = 50,
    forecast_hours: int = 24
) -> DataFrame:
    """
    One-shot function to generate forecast from HDFS historical data.
    
    Args:
        spark: SparkSession
        historical_path: HDFS path to historical data
        n_cities: Number of cities
        forecast_hours: Forecast horizon
        
    Returns:
        DataFrame with synthetic forecast
    """
    generator = SparkStreamingGenerator(spark)
    return generator.generate_streaming_forecast(
        historical_path=historical_path,
        n_cities=n_cities,
        forecast_hours=forecast_hours
    )


def validate_synthetic_data_spark(spark: SparkSession, df: DataFrame) -> Dict[str, Any]:
    """Validate synthetic data using Spark."""
    validator = SparkEDAValidator(spark)
    return validator.run_full_validation(df)
