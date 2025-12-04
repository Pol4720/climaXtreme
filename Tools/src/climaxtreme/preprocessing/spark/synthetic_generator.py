"""
Synthetic Climate Data Generator using PySpark.

This module generates synthetic climate data including:
- Hourly temperature interpolation
- Precipitation simulation
- Storm events and tracking
- Weather alerts
- Extended meteorological variables

Based on statistical models documented in SYNTHETIC_DATA_AND_MODELS.md
"""

import logging
import math
import uuid
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, FloatType, IntegerType,
    TimestampType, BooleanType, ArrayType, DoubleType
)
from pyspark.sql.window import Window

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Dataclasses
# ============================================================================

@dataclass
class WeatherParams:
    """Parameters for weather variable generation."""
    rain_gamma_shape: float = 2.0
    rain_gamma_scale: float = 5.0
    wind_weibull_shape: float = 2.0
    wind_weibull_scale: float = 15.0
    humidity_mean: float = 65.0
    humidity_std: float = 15.0
    pressure_mean: float = 1013.25
    pressure_std: float = 10.0


@dataclass
class EventRates:
    """Event occurrence rates by region."""
    storm_per_year_tropical: float = 12.0
    storm_per_year_temperate: float = 4.0
    storm_per_year_polar: float = 1.0
    heatwave_probability: float = 0.02
    coldsnap_probability: float = 0.02
    flood_probability: float = 0.01


@dataclass
class SyntheticConfig:
    """Main configuration for synthetic data generation."""
    seed: int = 42
    hourly_interpolation: bool = True
    hours_per_day: int = 24
    weather_params: WeatherParams = None
    event_rates: EventRates = None
    output_partitions: List[str] = None
    
    def __post_init__(self):
        if self.weather_params is None:
            self.weather_params = WeatherParams()
        if self.event_rates is None:
            self.event_rates = EventRates()
        if self.output_partitions is None:
            self.output_partitions = ["year", "month", "country"]


# ============================================================================
# Climate Zone Classification
# ============================================================================

CLIMATE_ZONES = {
    'TROPICAL': {'lat_range': (-23.5, 23.5), 'diurnal_amp': 10.0, 'rain_prob': 0.35},
    'SUBTROPICAL': {'lat_range': (23.5, 35), 'diurnal_amp': 12.0, 'rain_prob': 0.20},
    'TEMPERATE': {'lat_range': (35, 55), 'diurnal_amp': 14.0, 'rain_prob': 0.25},
    'CONTINENTAL': {'lat_range': (55, 66.5), 'diurnal_amp': 16.0, 'rain_prob': 0.20},
    'POLAR': {'lat_range': (66.5, 90), 'diurnal_amp': 8.0, 'rain_prob': 0.15},
}

# Storm names pool (Atlantic style)
STORM_NAMES = [
    "Alberto", "Beryl", "Chris", "Debby", "Ernesto", "Florence", "Gordon",
    "Helene", "Isaac", "Joyce", "Kirk", "Leslie", "Michael", "Nadine",
    "Oscar", "Patty", "Rafael", "Sara", "Tony", "Valerie", "William"
]


# ============================================================================
# UDF Functions for Spark
# ============================================================================

def get_climate_zone_udf():
    """Returns UDF to classify climate zone based on latitude."""
    def classify_zone(lat: float) -> str:
        if lat is None:
            return "UNKNOWN"
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
    """Returns UDF to determine season based on month and hemisphere."""
    def get_season(month: int, lat: float) -> str:
        if month is None:
            return "UNKNOWN"
        # Northern hemisphere seasons
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
            flip = {"WINTER": "SUMMER", "SUMMER": "WINTER", 
                    "SPRING": "AUTUMN", "AUTUMN": "SPRING"}
            season = flip.get(season, season)
        
        return season
    return F.udf(get_season, StringType())


def get_diurnal_amplitude_udf():
    """Returns UDF to calculate diurnal temperature amplitude."""
    def calc_amplitude(climate_zone: str, season: str) -> float:
        base_amp = {
            "TROPICAL": 10.0,
            "SUBTROPICAL": 12.0,
            "TEMPERATE": 14.0,
            "CONTINENTAL": 16.0,
            "POLAR": 8.0
        }.get(climate_zone, 12.0)
        
        # Seasonal adjustment
        season_factor = {
            "SUMMER": 1.2,
            "WINTER": 0.8,
            "SPRING": 1.0,
            "AUTUMN": 1.0
        }.get(season, 1.0)
        
        return base_amp * season_factor
    
    return F.udf(calc_amplitude, FloatType())


# ============================================================================
# Main Generator Class
# ============================================================================

class SyntheticClimateGenerator:
    """
    Generates synthetic climate data using PySpark.
    
    Supports:
    - Hourly temperature interpolation with diurnal cycle
    - Precipitation using Markov chains + Gamma distribution
    - Storm simulation with trajectory tracking
    - Weather alerts based on thresholds
    - Extended meteorological variables (wind, humidity, pressure)
    """
    
    def __init__(self, spark: SparkSession, config: Optional[SyntheticConfig] = None):
        """
        Initialize the synthetic generator.
        
        Args:
            spark: Active SparkSession
            config: Configuration parameters
        """
        self.spark = spark
        self.config = config or SyntheticConfig()
        
        # Set random seed for reproducibility
        self._set_seed()
        
        logger.info(f"SyntheticClimateGenerator initialized with seed={self.config.seed}")
    
    def _set_seed(self):
        """Set random seed for Spark operations."""
        # Spark doesn't have a global seed, but we can use it in functions
        self.seed = self.config.seed
    
    # ========================================================================
    # Data Preparation
    # ========================================================================
    
    def prepare_base_data(self, df: DataFrame) -> DataFrame:
        """
        Prepare the original dataset with parsed coordinates and classifications.
        
        Args:
            df: Original temperature DataFrame
            
        Returns:
            DataFrame with parsed lat/lon and climate zones
        """
        logger.info("Preparing base data with coordinate parsing and classifications...")
        
        # Parse latitude (e.g., "57.05N" -> 57.05, "23.5S" -> -23.5)
        df = df.withColumn(
            "lat_decimal",
            F.when(F.col("Latitude").endswith("N"),
                   F.regexp_extract("Latitude", r"([\d.]+)", 1).cast(FloatType()))
            .when(F.col("Latitude").endswith("S"),
                  -F.regexp_extract("Latitude", r"([\d.]+)", 1).cast(FloatType()))
            .otherwise(F.col("Latitude").cast(FloatType()))
        )
        
        # Parse longitude (e.g., "10.33E" -> 10.33, "75.5W" -> -75.5)
        df = df.withColumn(
            "lon_decimal",
            F.when(F.col("Longitude").endswith("E"),
                   F.regexp_extract("Longitude", r"([\d.]+)", 1).cast(FloatType()))
            .when(F.col("Longitude").endswith("W"),
                  -F.regexp_extract("Longitude", r"([\d.]+)", 1).cast(FloatType()))
            .otherwise(F.col("Longitude").cast(FloatType()))
        )
        
        # Parse date and extract components
        df = df.withColumn("date", F.to_date("dt", "yyyy-MM-dd"))
        df = df.withColumn("year", F.year("date"))
        df = df.withColumn("month", F.month("date"))
        df = df.withColumn("day", F.dayofmonth("date"))
        df = df.withColumn("day_of_week", F.dayofweek("date"))
        
        # Classify climate zone
        climate_zone_udf = get_climate_zone_udf()
        df = df.withColumn("climate_zone", climate_zone_udf(F.col("lat_decimal")))
        
        # Determine season
        season_udf = get_season_udf()
        df = df.withColumn("season", season_udf(F.col("month"), F.col("lat_decimal")))
        
        # Rename temperature columns for consistency
        df = df.withColumnRenamed("AverageTemperature", "avg_temperature")
        df = df.withColumnRenamed("AverageTemperatureUncertainty", "temp_uncertainty")
        
        logger.info(f"Base data prepared: {df.count()} records")
        return df
    
    # ========================================================================
    # Hourly Temperature Generation
    # ========================================================================
    
    def generate_hourly_temperatures(self, df: DataFrame) -> DataFrame:
        """
        Expand daily data to hourly resolution using diurnal cycle model.
        
        Model: T(h) = T_mean + A * sin(2π(h - h_max)/24) + ε
        
        Args:
            df: Daily temperature DataFrame
            
        Returns:
            DataFrame with hourly temperatures
        """
        logger.info("Generating hourly temperature data...")
        
        if not self.config.hourly_interpolation:
            logger.info("Hourly interpolation disabled, skipping")
            return df
        
        # Get diurnal amplitude based on climate zone and season
        amplitude_udf = get_diurnal_amplitude_udf()
        df = df.withColumn("diurnal_amplitude", 
                          amplitude_udf(F.col("climate_zone"), F.col("season")))
        
        # Create array of hours [0, 1, 2, ..., 23]
        hours = list(range(24))
        
        # Explode to create hourly records
        df = df.withColumn("hour", F.explode(F.array([F.lit(h) for h in hours])))
        
        # Calculate hourly temperature using diurnal cycle
        # Peak temperature around 14:00 (h_max = 14)
        h_max = 14.0
        df = df.withColumn(
            "temperature_hourly",
            F.col("avg_temperature") + 
            F.col("diurnal_amplitude") * 
            F.sin(2 * math.pi * (F.col("hour") - h_max) / 24.0) +
            # Add random noise based on uncertainty
            (F.rand(self.seed) - 0.5) * F.col("temp_uncertainty")
        )
        
        # Create full timestamp
        df = df.withColumn(
            "timestamp",
            F.to_timestamp(
                F.concat(
                    F.col("date").cast(StringType()),
                    F.lit(" "),
                    F.lpad(F.col("hour").cast(StringType()), 2, "0"),
                    F.lit(":00:00")
                ),
                "yyyy-MM-dd HH:mm:ss"
            )
        )
        
        logger.info(f"Hourly data generated: {df.count()} records")
        return df
    
    # ========================================================================
    # Weather Variables Generation
    # ========================================================================
    
    def generate_weather_variables(self, df: DataFrame) -> DataFrame:
        """
        Generate synthetic weather variables (rain, wind, humidity, pressure).
        
        Args:
            df: DataFrame with temperature data
            
        Returns:
            DataFrame with additional weather variables
        """
        logger.info("Generating weather variables...")
        
        params = self.config.weather_params
        
        # --- Precipitation ---
        # Simple model: probability based on climate zone + random amount
        df = df.withColumn(
            "rain_probability",
            F.when(F.col("climate_zone") == "TROPICAL", 0.35)
            .when(F.col("climate_zone") == "SUBTROPICAL", 0.20)
            .when(F.col("climate_zone") == "TEMPERATE", 0.25)
            .when(F.col("climate_zone") == "CONTINENTAL", 0.20)
            .when(F.col("climate_zone") == "POLAR", 0.15)
            .otherwise(0.20)
        )
        
        # Determine if it's raining (wet/dry state)
        df = df.withColumn("is_raining", F.rand(self.seed + 1) < F.col("rain_probability"))
        
        # Generate rain amount using exponential approximation of Gamma
        # rain_mm ~ Gamma(shape, scale) ≈ -scale * shape * ln(U) for simple approximation
        df = df.withColumn(
            "rain_mm",
            F.when(
                F.col("is_raining"),
                -params.rain_gamma_scale * F.log(F.rand(self.seed + 2) + 0.001) * 
                (1 + F.rand(self.seed + 3))
            ).otherwise(0.0)
        )
        
        # Clip rain values to reasonable range
        df = df.withColumn("rain_mm", 
                          F.when(F.col("rain_mm") > 100, 100.0)
                          .when(F.col("rain_mm") < 0, 0.0)
                          .otherwise(F.col("rain_mm")))
        
        # --- Wind Speed ---
        # Weibull distribution approximation
        df = df.withColumn(
            "wind_speed_kmh",
            params.wind_weibull_scale * 
            F.pow(-F.log(F.rand(self.seed + 4) + 0.001), 1.0 / params.wind_weibull_shape)
        )
        df = df.withColumn("wind_speed_kmh", 
                          F.when(F.col("wind_speed_kmh") > 200, 200.0)
                          .otherwise(F.col("wind_speed_kmh")))
        
        # --- Wind Direction ---
        df = df.withColumn("wind_direction_deg", F.rand(self.seed + 5) * 360.0)
        
        # --- Humidity ---
        # Higher when raining, correlated with temperature
        df = df.withColumn(
            "humidity_pct",
            F.when(
                F.col("is_raining"),
                F.lit(params.humidity_mean) + 20 + F.randn(self.seed + 6) * 5
            ).otherwise(
                F.lit(params.humidity_mean) - 
                (F.col("temperature_hourly") - 15) * 0.5 + 
                F.randn(self.seed + 7) * params.humidity_std
            )
        )
        df = df.withColumn("humidity_pct",
                          F.when(F.col("humidity_pct") > 100, 100.0)
                          .when(F.col("humidity_pct") < 0, 0.0)
                          .otherwise(F.col("humidity_pct")))
        
        # --- Atmospheric Pressure ---
        # Lower during storms/rain
        df = df.withColumn(
            "pressure_hpa",
            F.lit(params.pressure_mean) +
            F.when(F.col("is_raining"), -10).otherwise(0) +
            F.randn(self.seed + 8) * params.pressure_std
        )
        
        # --- Cloud Cover ---
        df = df.withColumn(
            "cloud_cover_pct",
            F.when(F.col("is_raining"), 70 + F.rand(self.seed + 9) * 30)
            .otherwise(F.rand(self.seed + 10) * 70)
        )
        
        # Clean up intermediate columns
        df = df.drop("rain_probability", "is_raining")
        
        logger.info("Weather variables generated")
        return df
    
    # ========================================================================
    # Event Detection and Simulation
    # ========================================================================
    
    def simulate_extreme_events(self, df: DataFrame) -> DataFrame:
        """
        Simulate extreme weather events based on temperature anomalies.
        
        Args:
            df: DataFrame with weather data
            
        Returns:
            DataFrame with event classifications
        """
        logger.info("Simulating extreme events...")
        
        # Calculate climatology statistics per city/month
        climatology_window = Window.partitionBy("City", "month")
        
        df = df.withColumn("temp_mean_clim", 
                          F.avg("temperature_hourly").over(climatology_window))
        df = df.withColumn("temp_std_clim", 
                          F.stddev("temperature_hourly").over(climatology_window))
        
        # Handle null std (single values)
        df = df.withColumn("temp_std_clim",
                          F.when(F.col("temp_std_clim").isNull(), 1.0)
                          .otherwise(F.col("temp_std_clim")))
        
        # Calculate anomaly score (z-score)
        df = df.withColumn(
            "anomaly_score",
            (F.col("temperature_hourly") - F.col("temp_mean_clim")) / 
            F.col("temp_std_clim")
        )
        
        # Classify event type
        df = df.withColumn(
            "event_type",
            F.when(F.col("anomaly_score") >= 2.5, "HEATWAVE")
            .when(F.col("anomaly_score") <= -2.5, "COLDSNAP")
            .when((F.col("rain_mm") > 30) & (F.col("wind_speed_kmh") > 50), "STORM")
            .when(F.col("rain_mm") > 50, "FLOOD")
            .otherwise("NORMAL")
        )
        
        # Calculate event intensity (normalized 0-1)
        df = df.withColumn(
            "event_intensity",
            F.when(F.col("event_type") == "NORMAL", 0.0)
            .when(F.col("event_type") == "HEATWAVE", 
                  F.least(F.lit(1.0), (F.col("anomaly_score") - 2.5) / 2.5))
            .when(F.col("event_type") == "COLDSNAP",
                  F.least(F.lit(1.0), (-F.col("anomaly_score") - 2.5) / 2.5))
            .when(F.col("event_type") == "STORM",
                  F.least(F.lit(1.0), F.col("wind_speed_kmh") / 150.0))
            .when(F.col("event_type") == "FLOOD",
                  F.least(F.lit(1.0), F.col("rain_mm") / 100.0))
            .otherwise(0.0)
        )
        
        # Trend direction based on recent history
        trend_window = Window.partitionBy("City").orderBy("timestamp").rowsBetween(-24, 0)
        df = df.withColumn("temp_rolling_mean", 
                          F.avg("temperature_hourly").over(trend_window))
        
        df = df.withColumn(
            "trend_direction",
            F.when(F.col("temperature_hourly") > F.col("temp_rolling_mean") + 1, "UP")
            .when(F.col("temperature_hourly") < F.col("temp_rolling_mean") - 1, "DOWN")
            .otherwise("STABLE")
        )
        
        # Clean up intermediate columns
        df = df.drop("temp_mean_clim", "temp_std_clim", "temp_rolling_mean")
        
        logger.info("Extreme events simulated")
        return df
    
    # ========================================================================
    # Storm Generation
    # ========================================================================
    
    def generate_storms(self, df: DataFrame) -> Tuple[DataFrame, DataFrame]:
        """
        Generate synthetic storm events with tracking data.
        
        Args:
            df: DataFrame with event classifications
            
        Returns:
            Tuple of (main DataFrame with storm IDs, storm tracks DataFrame)
        """
        logger.info("Generating storm events...")
        
        # Identify storm events
        storm_events = df.filter(F.col("event_type") == "STORM")
        
        if storm_events.count() == 0:
            logger.warning("No storm events found to generate tracks")
            # Add empty storm columns
            df = df.withColumn("storm_id", F.lit(None).cast(StringType()))
            df = df.withColumn("storm_name", F.lit(None).cast(StringType()))
            df = df.withColumn("storm_category", F.lit(0).cast(IntegerType()))
            
            # Create empty storm tracks schema
            storm_schema = StructType([
                StructField("storm_id", StringType(), False),
                StructField("storm_name", StringType(), True),
                StructField("timestamp", TimestampType(), False),
                StructField("latitude", FloatType(), False),
                StructField("longitude", FloatType(), False),
                StructField("category", IntegerType(), False),
                StructField("max_wind_kmh", FloatType(), False),
                StructField("central_pressure_hpa", FloatType(), False),
                StructField("movement_speed_kmh", FloatType(), True),
                StructField("movement_direction_deg", FloatType(), True),
                StructField("radius_km", FloatType(), True),
                StructField("lifecycle_stage", StringType(), True)
            ])
            empty_storms = self.spark.createDataFrame([], storm_schema)
            return df, empty_storms
        
        # Group nearby storm events into storm systems
        # Simple approach: window by time and location
        storm_window = Window.partitionBy(
            F.date_trunc("day", "timestamp"),
            F.round(F.col("lat_decimal"), 0),
            F.round(F.col("lon_decimal"), 0)
        ).orderBy("timestamp")
        
        storm_events = storm_events.withColumn(
            "storm_group_id",
            F.concat(
                F.date_format("timestamp", "yyyyMMdd"),
                F.lit("_"),
                F.round(F.col("lat_decimal"), 0).cast(StringType()),
                F.lit("_"),
                F.round(F.col("lon_decimal"), 0).cast(StringType())
            )
        )
        
        # Assign storm IDs using UUID-like identifier
        storm_events = storm_events.withColumn(
            "storm_id",
            F.concat(F.lit("STM-"), F.md5(F.col("storm_group_id")))
        )
        
        # Assign storm names (cycling through list)
        storm_events = storm_events.withColumn(
            "storm_name_idx",
            F.abs(F.hash(F.col("storm_id"))) % len(STORM_NAMES)
        )
        
        # Calculate storm category based on wind speed (Saffir-Simpson simplified)
        storm_events = storm_events.withColumn(
            "storm_category",
            F.when(F.col("wind_speed_kmh") < 63, 0)  # Tropical depression
            .when(F.col("wind_speed_kmh") < 118, 1)  # Category 1
            .when(F.col("wind_speed_kmh") < 154, 2)  # Category 2
            .when(F.col("wind_speed_kmh") < 178, 3)  # Category 3
            .when(F.col("wind_speed_kmh") < 209, 4)  # Category 4
            .otherwise(5)  # Category 5
        )
        
        # Create storm name mapping
        names_df = self.spark.createDataFrame(
            [(i, name) for i, name in enumerate(STORM_NAMES)],
            ["idx", "storm_name"]
        )
        
        storm_events = storm_events.join(
            names_df, 
            storm_events.storm_name_idx == names_df.idx,
            "left"
        ).drop("idx", "storm_name_idx", "storm_group_id")
        
        # Create storm tracks DataFrame
        storm_tracks = storm_events.select(
            "storm_id",
            "storm_name",
            "timestamp",
            F.col("lat_decimal").alias("latitude"),
            F.col("lon_decimal").alias("longitude"),
            "storm_category",
            F.col("wind_speed_kmh").alias("max_wind_kmh"),
            F.col("pressure_hpa").alias("central_pressure_hpa"),
            (F.rand(self.seed + 20) * 30 + 10).alias("movement_speed_kmh"),
            (F.rand(self.seed + 21) * 360).alias("movement_direction_deg"),
            (F.rand(self.seed + 22) * 200 + 50).alias("radius_km"),
            F.when(F.col("storm_category") <= 1, "FORMING")
            .when(F.col("storm_category") <= 3, "MATURE")
            .otherwise("INTENSE").alias("lifecycle_stage")
        ).distinct()
        
        # Rename columns before join
        storm_events_for_join = storm_events.select(
            "timestamp", "City", "storm_id", "storm_name", 
            F.col("storm_category").alias("category_from_storm")
        )
        
        # Join storm info back to main dataframe
        df = df.join(
            storm_events_for_join.select("timestamp", "City", "storm_id", "storm_name", "category_from_storm"),
            on=["timestamp", "City"],
            how="left"
        )
        
        df = df.withColumn(
            "storm_category",
            F.coalesce(F.col("category_from_storm"), F.lit(0))
        ).drop("category_from_storm")
        
        logger.info(f"Generated {storm_tracks.count()} storm track points")
        return df, storm_tracks
    
    # ========================================================================
    # Alert Generation
    # ========================================================================
    
    def generate_alerts(self, df: DataFrame) -> DataFrame:
        """
        Generate weather alerts based on conditions.
        
        Args:
            df: DataFrame with weather data and events
            
        Returns:
            DataFrame with alert information
        """
        logger.info("Generating weather alerts...")
        
        # Determine if alert is active
        df = df.withColumn(
            "alert_active",
            (F.col("event_type") != "NORMAL") | 
            (F.col("wind_speed_kmh") > 60) |
            (F.col("rain_mm") > 30)
        )
        
        # Determine alert level
        df = df.withColumn(
            "alert_level",
            F.when(~F.col("alert_active"), "NONE")
            .when(
                (F.col("event_intensity") > 0.7) | 
                (F.col("storm_category") >= 4) |
                (F.col("wind_speed_kmh") > 120),
                "EMERGENCY"
            )
            .when(
                (F.col("event_intensity") > 0.4) | 
                (F.col("storm_category") >= 2) |
                (F.col("wind_speed_kmh") > 90),
                "WARNING"
            )
            .otherwise("WATCH")
        )
        
        # Determine alert type
        df = df.withColumn(
            "alert_type",
            F.when(~F.col("alert_active"), "NONE")
            .when(F.col("event_type") == "HEATWAVE", "HEAT")
            .when(F.col("event_type") == "COLDSNAP", "COLD")
            .when((F.col("event_type") == "STORM") | (F.col("storm_category") > 0), "STORM")
            .when(F.col("event_type") == "FLOOD", "FLOOD")
            .when(F.col("wind_speed_kmh") > 60, "WIND")
            .otherwise("WEATHER")
        )
        
        # Alert issued timestamp (same as event timestamp for synthetic data)
        df = df.withColumn(
            "alert_issued_at",
            F.when(F.col("alert_active"), F.col("timestamp")).otherwise(F.lit(None))
        )
        
        logger.info("Alerts generated")
        return df
    
    # ========================================================================
    # Main Pipeline
    # ========================================================================
    
    def generate_full_synthetic_dataset(
        self, 
        input_df: DataFrame,
        generate_storms: bool = True
    ) -> Tuple[DataFrame, Optional[DataFrame]]:
        """
        Run the full synthetic data generation pipeline.
        
        Args:
            input_df: Original temperature DataFrame
            generate_storms: Whether to generate storm tracking data
            
        Returns:
            Tuple of (synthetic data DataFrame, storm tracks DataFrame or None)
        """
        logger.info("=" * 60)
        logger.info("Starting synthetic data generation pipeline")
        logger.info("=" * 60)
        
        # Step 1: Prepare base data
        df = self.prepare_base_data(input_df)
        
        # Step 2: Generate hourly temperatures
        df = self.generate_hourly_temperatures(df)
        
        # Step 3: Generate weather variables
        df = self.generate_weather_variables(df)
        
        # Step 4: Simulate extreme events
        df = self.simulate_extreme_events(df)
        
        # Step 5: Generate storms (optional)
        storm_tracks = None
        if generate_storms:
            df, storm_tracks = self.generate_storms(df)
        else:
            df = df.withColumn("storm_id", F.lit(None).cast(StringType()))
            df = df.withColumn("storm_name", F.lit(None).cast(StringType()))
            df = df.withColumn("storm_category", F.lit(0).cast(IntegerType()))
        
        # Step 6: Generate alerts
        df = self.generate_alerts(df)
        
        # Select and order final columns
        final_columns = [
            # Original identifiers
            "timestamp", "date", "year", "month", "day", "hour", "day_of_week",
            "City", "Country", "lat_decimal", "lon_decimal",
            
            # Climate classification
            "climate_zone", "season",
            
            # Temperature
            "avg_temperature", "temp_uncertainty", "temperature_hourly", "diurnal_amplitude",
            
            # Weather variables
            "rain_mm", "wind_speed_kmh", "wind_direction_deg",
            "humidity_pct", "pressure_hpa", "cloud_cover_pct",
            
            # Events
            "event_type", "event_intensity", "anomaly_score", "trend_direction",
            
            # Storms
            "storm_id", "storm_name", "storm_category",
            
            # Alerts
            "alert_active", "alert_level", "alert_type", "alert_issued_at"
        ]
        
        # Filter to existing columns (in case some weren't created)
        existing_cols = [c for c in final_columns if c in df.columns]
        df = df.select(existing_cols)
        
        logger.info("=" * 60)
        logger.info("Synthetic data generation complete!")
        logger.info(f"Total records: {df.count()}")
        logger.info("=" * 60)
        
        return df, storm_tracks
    
    # ========================================================================
    # Output Functions
    # ========================================================================
    
    def write_to_parquet(
        self, 
        df: DataFrame, 
        output_path: str,
        partition_by: Optional[List[str]] = None,
        mode: str = "overwrite"
    ) -> str:
        """
        Write DataFrame to Parquet format.
        
        Args:
            df: DataFrame to write
            output_path: Output path (local or HDFS)
            partition_by: Columns to partition by
            mode: Write mode (overwrite, append, etc.)
            
        Returns:
            Output path
        """
        partition_by = partition_by or self.config.output_partitions
        
        logger.info(f"Writing to {output_path} with partitions: {partition_by}")
        
        writer = df.write.mode(mode)
        
        if partition_by:
            writer = writer.partitionBy(*partition_by)
        
        writer.parquet(output_path)
        
        logger.info(f"Data written successfully to {output_path}")
        return output_path
    
    def write_storm_tracks(
        self,
        storm_df: DataFrame,
        output_path: str,
        mode: str = "overwrite"
    ) -> str:
        """
        Write storm tracks to Parquet.
        
        Args:
            storm_df: Storm tracks DataFrame
            output_path: Output path
            mode: Write mode
            
        Returns:
            Output path
        """
        if storm_df is None or storm_df.count() == 0:
            logger.warning("No storm tracks to write")
            return output_path
        
        logger.info(f"Writing storm tracks to {output_path}")
        
        storm_df.write.mode(mode).partitionBy("storm_id").parquet(output_path)
        
        logger.info(f"Storm tracks written to {output_path}")
        return output_path


# ============================================================================
# Convenience Functions
# ============================================================================

def create_generator(
    spark: SparkSession,
    seed: int = 42,
    hourly: bool = True
) -> SyntheticClimateGenerator:
    """
    Factory function to create a configured generator.
    
    Args:
        spark: SparkSession
        seed: Random seed
        hourly: Enable hourly interpolation
        
    Returns:
        Configured SyntheticClimateGenerator
    """
    config = SyntheticConfig(
        seed=seed,
        hourly_interpolation=hourly
    )
    return SyntheticClimateGenerator(spark, config)


def generate_synthetic_data(
    spark: SparkSession,
    input_path: str,
    output_base_path: str,
    config: Optional[SyntheticConfig] = None
) -> Dict[str, str]:
    """
    One-shot function to generate synthetic data from CSV.
    
    Args:
        spark: SparkSession
        input_path: Path to input CSV
        output_base_path: Base path for outputs
        config: Optional configuration
        
    Returns:
        Dictionary of output paths
    """
    from .readers import read_city_temperature_csv_path
    
    # Read input data
    input_df = read_city_temperature_csv_path(spark, input_path)
    
    # Create generator
    generator = SyntheticClimateGenerator(spark, config)
    
    # Generate data
    synthetic_df, storm_tracks = generator.generate_full_synthetic_dataset(input_df)
    
    # Write outputs
    outputs = {}
    
    # Main synthetic data
    main_output = f"{output_base_path}/synthetic_hourly.parquet"
    generator.write_to_parquet(synthetic_df, main_output)
    outputs["synthetic_hourly"] = main_output
    
    # Storm tracks
    if storm_tracks is not None:
        storm_output = f"{output_base_path}/storm_tracks.parquet"
        generator.write_storm_tracks(storm_tracks, storm_output)
        outputs["storm_tracks"] = storm_output
    
    # Alerts summary
    alerts_df = synthetic_df.filter(F.col("alert_active") == True)
    if alerts_df.count() > 0:
        alerts_output = f"{output_base_path}/alerts_history.parquet"
        alerts_df.write.mode("overwrite").partitionBy("year", "month").parquet(alerts_output)
        outputs["alerts_history"] = alerts_output
    
    # Event summary
    events_df = synthetic_df.filter(F.col("event_type") != "NORMAL")
    if events_df.count() > 0:
        events_output = f"{output_base_path}/event_summary.parquet"
        events_df.write.mode("overwrite").partitionBy("year", "event_type").parquet(events_output)
        outputs["event_summary"] = events_output
    
    return outputs
