"""
PySpark-based data preprocessing for climate data.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, avg, count, min as spark_min, max as spark_max,
    when, isnan, isnull, year, month, dayofmonth,
    regexp_replace, trim, split, desc, abs as spark_abs
)
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, 
    IntegerType, DateType, TimestampType
)


logger = logging.getLogger(__name__)


class SparkPreprocessor:
    """
    PySpark-based preprocessor for large-scale climate data processing.
    
    Handles:
    - Data cleaning and validation
    - Format standardization  
    - Aggregation and summarization
    - Outlier detection and handling
    """
    
    def __init__(self, app_name: str = "climaXtreme") -> None:
        """
        Initialize Spark session and preprocessor.
        
        Args:
            app_name: Name for the Spark application
        """
        self.app_name = app_name
        self.spark: Optional[SparkSession] = None
        
    def get_spark_session(self) -> SparkSession:
        """
        Get or create Spark session with optimized configuration.
        
        Returns:
            SparkSession instance
        """
        if self.spark is None:
            self.spark = (SparkSession.builder
                         .appName(self.app_name)
                         .config("spark.sql.adaptive.enabled", "true")
                         .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
                         .config("spark.sql.adaptive.skewJoin.enabled", "true")
                         .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
                         .config("spark.driver.memory", "4g")
                         .config("spark.executor.memory", "4g")
                         .config("spark.sql.shuffle.partitions", "200")
                         .config("spark.default.parallelism", "100")
                         .config("spark.sql.files.maxPartitionBytes", "134217728")  # 128 MB
                         .config("spark.hadoop.fs.hdfs.impl", "org.apache.hadoop.hdfs.DistributedFileSystem")
                         .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem")
                         .getOrCreate())
            
            # Set log level to reduce verbosity
            self.spark.sparkContext.setLogLevel("WARN")
            logger.info("Spark session initialized with optimized settings for large datasets")
        
        return self.spark
    
    def stop_spark_session(self) -> None:
        """Stop the Spark session."""
        if self.spark:
            self.spark.stop()
            self.spark = None
            logger.info("Spark session stopped")
    
    def read_berkeley_earth_file(self, filepath: str) -> DataFrame:
        """
        Read Berkeley Earth data file into Spark DataFrame.
        
        Args:
            filepath: Path to the data file
            
        Returns:
            Spark DataFrame with the data
        """
        spark = self.get_spark_session()
        
        try:
            # Berkeley Earth files are typically space-separated with comments
            df = (spark.read
                  .option("header", "false")
                  .option("comment", "%")
                  .option("delimiter", " ")
                  .option("multiline", "true")
                  .text(filepath))
            
            # Split the text into columns (assuming standard BE format)
            # Year Month Temperature Uncertainty
            split_col = split(trim(regexp_replace(col("value"), r"\s+", " ")), " ")
            
            processed_df = (df
                           .filter(~col("value").startswith("%"))  # Remove comments
                           .filter(col("value") != "")  # Remove empty lines
                           .withColumn("year", split_col.getItem(0).cast(IntegerType()))
                           .withColumn("month", split_col.getItem(1).cast(IntegerType()))
                           .withColumn("temperature", split_col.getItem(2).cast(DoubleType()))
                           .withColumn("uncertainty", split_col.getItem(3).cast(DoubleType()))
                           .drop("value"))
            
            logger.info(f"Successfully loaded {filepath} with {processed_df.count()} records")
            return processed_df
            
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
            raise

    def read_berkeley_earth_path(self, input_path: str) -> DataFrame:
        """
        Read Berkeley Earth formatted text files from a path or glob pattern.

        Supports local or HDFS paths (e.g., hdfs://host:9000/path/*.txt).
        """
        spark = self.get_spark_session()
        try:
            df = (
                spark.read
                .option("header", "false")
                .option("comment", "%")
                .option("delimiter", " ")
                .option("multiline", "true")
                .text(input_path)
            )

            split_col = split(trim(regexp_replace(col("value"), r"\s+", " ")), " ")

            processed_df = (
                df
                .filter(~col("value").startswith("%"))
                .filter(col("value") != "")
                .withColumn("year", split_col.getItem(0).cast(IntegerType()))
                .withColumn("month", split_col.getItem(1).cast(IntegerType()))
                .withColumn("temperature", split_col.getItem(2).cast(DoubleType()))
                .withColumn("uncertainty", split_col.getItem(3).cast(DoubleType()))
                .drop("value")
            )
            return processed_df
        except Exception as e:
            logger.error(f"Error reading path {input_path}: {e}")
            raise

    def read_city_temperature_csv_path(self, input_path: str) -> DataFrame:
        """
        Read GlobalLandTemperaturesByCity.csv (or compatible) from local or HDFS.
        Optimized for large files (500+ MB).
        """
        spark = self.get_spark_session()
        try:
            logger.info(f"Reading CSV from {input_path} (this may take a few minutes for large files)...")
            
            df = (
                spark.read
                .option("header", True)
                .option("inferSchema", False)  # Manual schema for speed
                .option("mode", "DROPMALFORMED")
                .csv(input_path)
            )

            from pyspark.sql.functions import to_date, year as pyear, month as pmonth

            # Manual type casting for better performance
            df = (
                df
                .select(
                    col("dt").alias("dt"),
                    col("AverageTemperature").cast(DoubleType()).alias("temperature"),
                    col("AverageTemperatureUncertainty").cast(DoubleType()).alias("uncertainty"),
                    col("City").alias("city"),
                    col("Country").alias("country"),
                    col("Latitude").alias("latitude"),
                    col("Longitude").alias("longitude")
                )
                .withColumn("date", to_date(col("dt")))
                .dropna(subset=["date", "temperature"])  # keep valid rows
                .withColumn("year", pyear(col("date")))
                .withColumn("month", pmonth(col("date")))
            )
            
            row_count = df.count()
            logger.info(f"Successfully loaded {row_count:,} records from {input_path}")
            
            return df
        except Exception as e:
            logger.error(f"Error reading CSV path {input_path}: {e}")
            raise
    
    def clean_temperature_data(self, df: DataFrame) -> DataFrame:
        """
        Clean temperature data by removing outliers and invalid values.
        
        Args:
            df: Input DataFrame with temperature data
            
        Returns:
            Cleaned DataFrame
        """
        # Remove records with null or extreme temperature values
        cleaned_df = (df
                     .filter(col("temperature").isNotNull())
                     .filter(col("year").isNotNull() & (col("year") > 1750) & (col("year") <= 2030))
                     .filter(col("month").isNotNull() & (col("month") >= 1) & (col("month") <= 12))
                     .filter(col("temperature") >= -100.0)  # Reasonable temperature bounds
                     .filter(col("temperature") <= 60.0))
        
        # Log cleaning results
        original_count = df.count()
        cleaned_count = cleaned_df.count()
        removed_count = original_count - cleaned_count
        
        logger.info(f"Data cleaning: {original_count} -> {cleaned_count} records "
                   f"({removed_count} removed, {removed_count/original_count*100:.1f}%)")
        
        return cleaned_df
    
    def aggregate_monthly_data(self, df: DataFrame) -> DataFrame:
        """
        Aggregate temperature data by year and month.
        
        Args:
            df: Input DataFrame with temperature data
            
        Returns:
            DataFrame with monthly aggregations including std deviation
        """
        from pyspark.sql.functions import stddev
        
        monthly_agg = (df
                      .groupBy("year", "month")
                      .agg(
                          avg("temperature").alias("avg_temperature"),
                          spark_min("temperature").alias("min_temperature"),
                          spark_max("temperature").alias("max_temperature"),
                          stddev("temperature").alias("std_temperature"),
                          count("temperature").alias("record_count"),
                          avg("uncertainty").alias("avg_uncertainty")
                      )
                      .orderBy("year", "month"))
        
        return monthly_agg
    
    def aggregate_yearly_data(self, df: DataFrame) -> DataFrame:
        """
        Aggregate temperature data by year.
        
        Args:
            df: Input DataFrame with temperature data
            
        Returns:
            DataFrame with yearly aggregations including std deviation
        """
        from pyspark.sql.functions import stddev
        
        yearly_agg = (df
                     .groupBy("year")
                     .agg(
                         avg("temperature").alias("avg_temperature"),
                         spark_min("temperature").alias("min_temperature"),
                         spark_max("temperature").alias("max_temperature"),
                         stddev("temperature").alias("std_temperature"),
                         count("temperature").alias("record_count"),
                         avg("uncertainty").alias("avg_uncertainty")
                     )
                     .orderBy("year"))
        
        return yearly_agg
    
    def detect_anomalies(self, df: DataFrame, threshold_std: float = 3.0) -> DataFrame:
        """
        Detect temperature anomalies using statistical methods.
        
        Args:
            df: Input DataFrame with temperature data
            threshold_std: Standard deviation threshold for anomaly detection
            
        Returns:
            DataFrame with anomaly flags
        """
        # Calculate statistics for anomaly detection
        stats = df.select(
            avg("temperature").alias("mean_temp"),
            # Using approxQuantile for standard deviation approximation
        ).collect()[0]
        
        mean_temp = stats["mean_temp"]
        
        # Calculate standard deviation using aggregation
        std_df = (df
                 .select(((col("temperature") - mean_temp) ** 2).alias("squared_diff"))
                 .agg(avg("squared_diff").alias("variance")))
        
        variance = std_df.collect()[0]["variance"]
        std_temp = variance ** 0.5
        
        # Flag anomalies
        anomaly_df = (df
                     .withColumn("temp_zscore", 
                               (col("temperature") - mean_temp) / std_temp)
                     .withColumn("is_anomaly",
                               when(spark_abs(col("temp_zscore")) > threshold_std, True)
                               .otherwise(False)))
        
        anomaly_count = anomaly_df.filter(col("is_anomaly")).count()
        total_count = anomaly_df.count()
        
        logger.info(f"Detected {anomaly_count} anomalies out of {total_count} records "
                   f"({anomaly_count/total_count*100:.2f}%)")
        
        return anomaly_df
    
    def compute_climatology_stats(self, df: DataFrame) -> DataFrame:
        """
        Compute climatology statistics by month (for seasonal analysis).
        
        Args:
            df: Input DataFrame with temperature data
            
        Returns:
            DataFrame with monthly climatology stats
        """
        from pyspark.sql.functions import stddev
        
        climatology = (df
                      .groupBy("month")
                      .agg(
                          avg("temperature").alias("climatology_mean"),
                          stddev("temperature").alias("climatology_std"),
                          spark_min("temperature").alias("climatology_min"),
                          spark_max("temperature").alias("climatology_max"),
                          count("temperature").alias("climatology_count")
                      )
                      .orderBy("month"))
        
        logger.info(f"Computed monthly climatology statistics")
        return climatology
    
    def compute_seasonal_stats(self, df: DataFrame) -> DataFrame:
        """
        Compute seasonal temperature statistics.
        
        Args:
            df: Input DataFrame with temperature data
            
        Returns:
            DataFrame with seasonal aggregations
        """
        from pyspark.sql.functions import stddev, when
        
        # Map months to seasons
        df_with_season = df.withColumn(
            "season",
            when((col("month") == 12) | (col("month") == 1) | (col("month") == 2), "Winter")
            .when((col("month") >= 3) & (col("month") <= 5), "Spring")
            .when((col("month") >= 6) & (col("month") <= 8), "Summer")
            .otherwise("Fall")
        )
        
        seasonal_stats = (df_with_season
                         .groupBy("season")
                         .agg(
                             avg("temperature").alias("avg_temperature"),
                             stddev("temperature").alias("std_temperature"),
                             spark_min("temperature").alias("min_temperature"),
                             spark_max("temperature").alias("max_temperature"),
                             count("temperature").alias("record_count")
                         ))
        
        logger.info(f"Computed seasonal statistics")
        return seasonal_stats
    
    def compute_extreme_thresholds(self, df: DataFrame, percentiles: List[float] = [90.0, 95.0, 99.0]) -> DataFrame:
        """
        Compute temperature thresholds for extreme event detection.
        
        Args:
            df: Input DataFrame with temperature data
            percentiles: List of percentiles to compute (default: [90, 95, 99])
            
        Returns:
            DataFrame with percentile thresholds
        """
        from pyspark.sql.functions import lit, percentile_approx
        
        # Compute percentiles
        thresholds_data = []
        
        for p in percentiles:
            high_p = p / 100.0
            low_p = (100 - p) / 100.0
            
            # Get actual percentile values
            percentiles_result = df.select(
                percentile_approx("temperature", high_p).alias(f"p{int(p)}_high"),
                percentile_approx("temperature", low_p).alias(f"p{int(p)}_low")
            ).collect()[0]
            
            thresholds_data.append({
                "percentile": p,
                "high_threshold": float(percentiles_result[f"p{int(p)}_high"]),
                "low_threshold": float(percentiles_result[f"p{int(p)}_low"])
            })
        
        # Create DataFrame from computed thresholds
        spark = self.get_spark_session()
        thresholds_df = spark.createDataFrame(thresholds_data)
        
        logger.info(f"Computed extreme event thresholds for percentiles: {percentiles}")
        return thresholds_df
    
    def compute_trend_line(self, df: DataFrame) -> DataFrame:
        """
        Compute linear trend line for yearly temperature data.
        
        Args:
            df: Input DataFrame with year and temperature columns
            
        Returns:
            DataFrame with trend line values
        """
        # Collect yearly data for trend calculation
        yearly_data = df.select("year", "avg_temperature").orderBy("year").collect()
        
        if len(yearly_data) < 2:
            logger.warning("Insufficient data for trend calculation")
            return df
        
        # Calculate trend using least squares
        years = [row.year for row in yearly_data]
        temps = [row.avg_temperature for row in yearly_data]
        
        n = len(years)
        sum_x = sum(years)
        sum_y = sum(temps)
        sum_xy = sum(x * y for x, y in zip(years, temps))
        sum_x2 = sum(x * x for x in years)
        
        # Linear regression coefficients
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n
        
        # Add trend values
        from pyspark.sql.functions import lit
        
        df_with_trend = df.withColumn(
            "trend_line",
            lit(slope) * col("year") + lit(intercept)
        ).withColumn(
            "trend_slope_per_year",
            lit(slope)
        ).withColumn(
            "trend_slope_per_decade",
            lit(slope * 10)
        )
        
        logger.info(f"Computed trend: {slope*10:.4f}°C per decade")
        return df_with_trend
    
    def process_directory(self, input_dir: str, output_dir: str) -> Dict[str, str]:
        """
        Process all data files in a directory.
        
        Args:
            input_dir: Directory containing raw data files
            output_dir: Directory to store processed data
            
        Returns:
            Dictionary mapping input files to output files
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        processed_files = {}
        
        # Process text files (Berkeley Earth format)
        for txt_file in input_path.glob("*.txt"):
            try:
                logger.info(f"Processing {txt_file.name}...")
                
                # Read and clean data
                df = self.read_berkeley_earth_file(str(txt_file))
                cleaned_df = self.clean_temperature_data(df)
                
                # Generate different aggregations
                monthly_df = self.aggregate_monthly_data(cleaned_df)
                yearly_df = self.aggregate_yearly_data(cleaned_df)
                anomaly_df = self.detect_anomalies(cleaned_df)
                
                # Save processed data
                base_name = txt_file.stem
                
                monthly_output = output_path / f"{base_name}_monthly.parquet"
                yearly_output = output_path / f"{base_name}_yearly.parquet"
                anomaly_output = output_path / f"{base_name}_anomalies.parquet"
                
                # Write as Parquet for efficient storage
                monthly_df.coalesce(1).write.mode("overwrite").parquet(str(monthly_output))
                yearly_df.coalesce(1).write.mode("overwrite").parquet(str(yearly_output))
                anomaly_df.coalesce(1).write.mode("overwrite").parquet(str(anomaly_output))
                
                processed_files[str(txt_file)] = {
                    "monthly": str(monthly_output),
                    "yearly": str(yearly_output),
                    "anomalies": str(anomaly_output)
                }
                
                logger.info(f"Successfully processed {txt_file.name}")
                
            except Exception as e:
                logger.error(f"Error processing {txt_file}: {e}")
                continue
        
        logger.info(f"Processed {len(processed_files)} files")
        return processed_files

    def parse_coordinates(self, df: DataFrame) -> DataFrame:
        """
        Parse latitude and longitude strings to numeric values.
        
        Converts:
        - '57.05N' -> 57.05
        - '10.33E' -> 10.33
        - '33.45S' -> -33.45
        - '122.41W' -> -122.41
        
        Args:
            df: DataFrame with 'latitude' and 'longitude' string columns
            
        Returns:
            DataFrame with 'lat_numeric' and 'lon_numeric' numeric columns
        """
        from pyspark.sql.functions import regexp_replace, when, substring, length
        
        # Parse latitude
        df = df.withColumn(
            "lat_numeric",
            when(
                col("latitude").endswith("S"),
                -regexp_replace(col("latitude"), "[NS]", "").cast(DoubleType())
            ).otherwise(
                regexp_replace(col("latitude"), "[NS]", "").cast(DoubleType())
            )
        )
        
        # Parse longitude
        df = df.withColumn(
            "lon_numeric",
            when(
                col("longitude").endswith("W"),
                -regexp_replace(col("longitude"), "[EW]", "").cast(DoubleType())
            ).otherwise(
                regexp_replace(col("longitude"), "[EW]", "").cast(DoubleType())
            )
        )
        
        return df
    
    def assign_continent(self, df: DataFrame) -> DataFrame:
        """
        Assign continent based on latitude and longitude coordinates.
        
        Uses geographic boundaries to determine continent:
        - Europe: lat 35-71, lon -10-60
        - Asia: lat -10-80, lon 60-180
        - Africa: lat -35-37, lon -18-52
        - North America: lat 15-75, lon -170 to -50
        - South America: lat -56 to 13, lon -82 to -34
        - Oceania: lat -50 to 0, lon 110-180
        - Antarctica: lat < -60
        
        Args:
            df: DataFrame with 'lat_numeric' and 'lon_numeric' columns
            
        Returns:
            DataFrame with 'continent' column
        """
        df = df.withColumn(
            "continent",
            when(
                (col("lat_numeric") < -60), "Antarctica"
            ).when(
                (col("lat_numeric").between(35, 71)) & (col("lon_numeric").between(-10, 60)),
                "Europe"
            ).when(
                (col("lat_numeric").between(-10, 80)) & (col("lon_numeric").between(60, 180)),
                "Asia"
            ).when(
                (col("lat_numeric").between(-35, 37)) & (col("lon_numeric").between(-18, 52)),
                "Africa"
            ).when(
                (col("lat_numeric").between(15, 75)) & (col("lon_numeric").between(-170, -50)),
                "North America"
            ).when(
                (col("lat_numeric").between(-56, 13)) & (col("lon_numeric").between(-82, -34)),
                "South America"
            ).when(
                (col("lat_numeric").between(-50, 0)) & (col("lon_numeric").between(110, 180)),
                "Oceania"
            ).otherwise("Other")
        )
        
        return df
    
    def assign_region(self, df: DataFrame) -> DataFrame:
        """
        Assign geographic region based on continent and coordinates.
        
        Divides each continent into sub-regions for more granular analysis.
        
        Args:
            df: DataFrame with 'continent', 'lat_numeric', 'lon_numeric' columns
            
        Returns:
            DataFrame with 'region' column
        """
        # Europe regions
        df = df.withColumn(
            "region",
            when(
                (col("continent") == "Europe") & (col("lat_numeric") >= 55),
                "Northern Europe"
            ).when(
                (col("continent") == "Europe") & (col("lat_numeric") < 55) & (col("lat_numeric") >= 45),
                "Central Europe"
            ).when(
                (col("continent") == "Europe") & (col("lat_numeric") < 45),
                "Southern Europe"
            # Asia regions
            ).when(
                (col("continent") == "Asia") & (col("lat_numeric") >= 50),
                "Northern Asia"
            ).when(
                (col("continent") == "Asia") & (col("lat_numeric") < 50) & (col("lat_numeric") >= 30),
                "Central Asia"
            ).when(
                (col("continent") == "Asia") & (col("lat_numeric") < 30) & (col("lon_numeric") < 100),
                "South Asia"
            ).when(
                (col("continent") == "Asia") & (col("lat_numeric") < 30) & (col("lon_numeric") >= 100),
                "East Asia"
            # Africa regions
            ).when(
                (col("continent") == "Africa") & (col("lat_numeric") >= 20),
                "Northern Africa"
            ).when(
                (col("continent") == "Africa") & (col("lat_numeric") < 20) & (col("lat_numeric") >= 0),
                "Central Africa"
            ).when(
                (col("continent") == "Africa") & (col("lat_numeric") < 0),
                "Southern Africa"
            # North America regions
            ).when(
                (col("continent") == "North America") & (col("lat_numeric") >= 50),
                "Northern North America"
            ).when(
                (col("continent") == "North America") & (col("lat_numeric") < 50) & (col("lat_numeric") >= 30),
                "Central North America"
            ).when(
                (col("continent") == "North America") & (col("lat_numeric") < 30),
                "Caribbean & Central America"
            # South America regions
            ).when(
                (col("continent") == "South America") & (col("lat_numeric") >= -10),
                "Northern South America"
            ).when(
                (col("continent") == "South America") & (col("lat_numeric") < -10) & (col("lat_numeric") >= -30),
                "Central South America"
            ).when(
                (col("continent") == "South America") & (col("lat_numeric") < -30),
                "Southern South America"
            # Oceania regions
            ).when(
                (col("continent") == "Oceania") & (col("lat_numeric") >= -25),
                "Northern Oceania"
            ).when(
                (col("continent") == "Oceania") & (col("lat_numeric") < -25),
                "Southern Oceania"
            # Antarctica and Other
            ).when(
                col("continent") == "Antarctica", "Antarctica"
            ).otherwise("Other")
        )
        
        return df
    
    def compute_regional_aggregations(self, df: DataFrame) -> DataFrame:
        """
        Compute temperature aggregations by region and year.
        
        Args:
            df: DataFrame with 'region', 'year', 'temperature' columns
            
        Returns:
            DataFrame with regional aggregations
        """
        from pyspark.sql.functions import stddev
        
        regional_agg = (
            df
            .filter(col("region") != "Other")  # Exclude unclassified regions
            .groupBy("region", "continent", "year")
            .agg(
                avg("temperature").alias("avg_temperature"),
                spark_min("temperature").alias("min_temperature"),
                spark_max("temperature").alias("max_temperature"),
                stddev("temperature").alias("std_temperature"),
                count("temperature").alias("record_count")
            )
            .orderBy("region", "year")
        )
        
        logger.info(f"Computed regional aggregations: {regional_agg.count()} records")
        return regional_agg
    
    def compute_continental_aggregations(self, df: DataFrame) -> DataFrame:
        """
        Compute temperature aggregations by continent and year.
        
        Args:
            df: DataFrame with 'continent', 'year', 'temperature' columns
            
        Returns:
            DataFrame with continental aggregations
        """
        from pyspark.sql.functions import stddev
        
        continental_agg = (
            df
            .filter(col("continent") != "Other")  # Exclude unclassified
            .groupBy("continent", "year")
            .agg(
                avg("temperature").alias("avg_temperature"),
                spark_min("temperature").alias("min_temperature"),
                spark_max("temperature").alias("max_temperature"),
                stddev("temperature").alias("std_temperature"),
                count("temperature").alias("record_count")
            )
            .orderBy("continent", "year")
        )
        
        logger.info(f"Computed continental aggregations: {continental_agg.count()} records")
        return continental_agg

    def process_path(self, input_path: str, output_dir: str, *, fmt: str = "auto") -> Dict[str, str]:
        """
        Process data from a local/HDFS path (file, directory, or glob pattern).
        Generates ALL statistics needed for dashboard (no calculations in dashboard).

        Args:
            input_path: Path or glob (supports hdfs:// URLs)
            output_dir: Output directory (local or hdfs://)
            fmt: 'auto' | 'berkeley-txt' | 'city-csv'

        Returns:
            Mapping of output artifacts.
        """
        # Decide format
        f = fmt
        if f == "auto":
            if input_path.lower().endswith(".csv"):
                f = "city-csv"
            else:
                f = "berkeley-txt"

        spark = self.get_spark_session()
        logger.info(f"Processing {input_path} with format {f}")
        
        try:
            # Read data
            if f == "city-csv":
                df = self.read_city_temperature_csv_path(input_path)
            elif f == "berkeley-txt":
                df = self.read_berkeley_earth_path(input_path)
            else:
                raise ValueError(f"Unknown format '{fmt}'")

            # Clean data
            cleaned_df = self.clean_temperature_data(df)
            
            # Generate all aggregations and statistics
            logger.info("Generating monthly aggregations...")
            monthly_df = self.aggregate_monthly_data(cleaned_df)
            
            logger.info("Generating yearly aggregations...")
            yearly_df = self.aggregate_yearly_data(cleaned_df)
            
            logger.info("Computing trend line...")
            yearly_with_trend_df = self.compute_trend_line(yearly_df)
            
            logger.info("Detecting anomalies...")
            anomaly_df = self.detect_anomalies(cleaned_df)
            
            logger.info("Computing climatology statistics...")
            climatology_df = self.compute_climatology_stats(cleaned_df)
            
            logger.info("Computing seasonal statistics...")
            seasonal_df = self.compute_seasonal_stats(cleaned_df)
            
            logger.info("Computing extreme event thresholds...")
            extremes_thresholds_df = self.compute_extreme_thresholds(cleaned_df, percentiles=[90.0, 95.0, 99.0])
            
            # NEW: Regional and Continental Analysis
            logger.info("Parsing coordinates and assigning geographic classifications...")
            df_with_coords = self.parse_coordinates(cleaned_df)
            df_with_continent = self.assign_continent(df_with_coords)
            df_with_region = self.assign_region(df_with_continent)
            
            logger.info("Computing regional aggregations...")
            regional_df = self.compute_regional_aggregations(df_with_region)
            
            logger.info("Computing continental aggregations...")
            continental_df = self.compute_continental_aggregations(df_with_region)

            # Define output paths (8 files now instead of 6)
            base = output_dir.rstrip("/")
            monthly_out = f"{base}/monthly.parquet"
            yearly_out = f"{base}/yearly.parquet"
            anomaly_out = f"{base}/anomalies.parquet"
            climatology_out = f"{base}/climatology.parquet"
            seasonal_out = f"{base}/seasonal.parquet"
            extremes_out = f"{base}/extreme_thresholds.parquet"
            regional_out = f"{base}/regional.parquet"
            continental_out = f"{base}/continental.parquet"

            # Save all outputs
            logger.info("Saving monthly data...")
            monthly_df.coalesce(1).write.mode("overwrite").parquet(monthly_out)
            
            logger.info("Saving yearly data with trend...")
            yearly_with_trend_df.coalesce(1).write.mode("overwrite").parquet(yearly_out)
            
            logger.info("Saving anomalies data...")
            anomaly_df.repartition(10).write.mode("overwrite").parquet(anomaly_out)
            
            logger.info("Saving climatology data...")
            climatology_df.coalesce(1).write.mode("overwrite").parquet(climatology_out)
            
            logger.info("Saving seasonal data...")
            seasonal_df.coalesce(1).write.mode("overwrite").parquet(seasonal_out)
            
            logger.info("Saving extreme thresholds...")
            extremes_thresholds_df.coalesce(1).write.mode("overwrite").parquet(extremes_out)
            
            logger.info("Saving regional data...")
            regional_df.coalesce(1).write.mode("overwrite").parquet(regional_out)
            
            logger.info("Saving continental data...")
            continental_df.coalesce(1).write.mode("overwrite").parquet(continental_out)

            logger.info("✓ All processing completed successfully (8 output files generated)")
            
            return {
                "monthly": monthly_out,
                "yearly": yearly_out,
                "anomalies": anomaly_out,
                "climatology": climatology_out,
                "seasonal": seasonal_out,
                "extreme_thresholds": extremes_out,
                "regional": regional_out,
                "continental": continental_out
            }
        finally:
            # Do not stop session here; leave lifecycle to caller/CLI context
            pass
    
    def get_data_summary(self, df: DataFrame) -> Dict[str, any]:
        """
        Generate summary statistics for a DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with summary statistics
        """
        try:
            row_count = df.count()
            
            if "temperature" in df.columns:
                temp_stats = (df
                            .select(
                                avg("temperature").alias("mean_temp"),
                                spark_min("temperature").alias("min_temp"),
                                spark_max("temperature").alias("max_temp"),
                                count("temperature").alias("temp_count")
                            ).collect()[0])
                
                return {
                    "total_records": row_count,
                    "temperature_stats": {
                        "mean": float(temp_stats["mean_temp"]) if temp_stats["mean_temp"] else None,
                        "min": float(temp_stats["min_temp"]) if temp_stats["min_temp"] else None,
                        "max": float(temp_stats["max_temp"]) if temp_stats["max_temp"] else None,
                        "count": int(temp_stats["temp_count"])
                    },
                    "columns": df.columns
                }
            else:
                return {
                    "total_records": row_count,
                    "columns": df.columns
                }
                
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return {"error": str(e)}
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup Spark session."""
        self.stop_spark_session()