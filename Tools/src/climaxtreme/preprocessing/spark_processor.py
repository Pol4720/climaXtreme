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
    regexp_replace, trim, split, desc
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
                         .getOrCreate())
            
            # Set log level to reduce verbosity
            self.spark.sparkContext.setLogLevel("WARN")
            logger.info("Spark session initialized")
        
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
            DataFrame with monthly aggregations
        """
        monthly_agg = (df
                      .groupBy("year", "month")
                      .agg(
                          avg("temperature").alias("avg_temperature"),
                          spark_min("temperature").alias("min_temperature"),
                          spark_max("temperature").alias("max_temperature"),
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
            DataFrame with yearly aggregations
        """
        yearly_agg = (df
                     .groupBy("year")
                     .agg(
                         avg("temperature").alias("avg_temperature"),
                         spark_min("temperature").alias("min_temperature"),
                         spark_max("temperature").alias("max_temperature"),
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
                               when(abs(col("temp_zscore")) > threshold_std, True)
                               .otherwise(False)))
        
        anomaly_count = anomaly_df.filter(col("is_anomaly")).count()
        total_count = anomaly_df.count()
        
        logger.info(f"Detected {anomaly_count} anomalies out of {total_count} records "
                   f"({anomaly_count/total_count*100:.2f}%)")
        
        return anomaly_df
    
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