"""
PySpark-based data preprocessing for climate data.
"""

import logging
from typing import Dict, List
from pyspark.sql import DataFrame

from .spark_session_manager import SparkSessionManager
from .readers import (
    read_berkeley_earth_file, read_berkeley_earth_path, read_city_temperature_csv_path
)
from .cleaning import clean_temperature_data
from .aggregation import aggregate_monthly_data, aggregate_yearly_data
from .analysis import (
    detect_anomalies, compute_climatology_stats, compute_seasonal_stats, 
    compute_extreme_thresholds, compute_trend_line
)
from .processing import process_directory

logger = logging.getLogger(__name__)

class SparkPreprocessor:
    """
    PySpark-based preprocessor for large-scale climate data processing.
    """
    
    def __init__(self, app_name: str = "climaXtreme"):
        """
        Initializes the Spark preprocessor.
        
        Args:
            app_name: The name of the Spark application.
        """
        self.session_manager = SparkSessionManager(app_name)
        self.spark = self.session_manager.get_spark_session()

    def stop_spark_session(self):
        """
        Stops the Spark session.
        """
        self.session_manager.stop_spark_session()

    def read_berkeley_earth_file(self, filepath: str) -> DataFrame:
        return read_berkeley_earth_file(self.spark, filepath)

    def read_berkeley_earth_path(self, input_path: str) -> DataFrame:
        return read_berkeley_earth_path(self.spark, input_path)

    def read_city_temperature_csv_path(self, input_path: str) -> DataFrame:
        return read_city_temperature_csv_path(self.spark, input_path)

    def clean_temperature_data(self, df: DataFrame) -> DataFrame:
        return clean_temperature_data(df)

    def aggregate_monthly_data(self, df: DataFrame) -> DataFrame:
        return aggregate_monthly_data(df)

    def aggregate_yearly_data(self, df: DataFrame) -> DataFrame:
        return aggregate_yearly_data(df)

    def detect_anomalies(self, df: DataFrame, threshold_std: float = 3.0) -> DataFrame:
        return detect_anomalies(df, threshold_std)

    def compute_climatology_stats(self, df: DataFrame) -> DataFrame:
        return compute_climatology_stats(df)

    def compute_seasonal_stats(self, df: DataFrame) -> DataFrame:
        return compute_seasonal_stats(df)

    def compute_extreme_thresholds(self, df: DataFrame, percentiles: List[float] = [90.0, 95.0, 99.0]) -> DataFrame:
        return compute_extreme_thresholds(df, percentiles)

    def compute_trend_line(self, df: DataFrame) -> DataFrame:
        return compute_trend_line(df)

    def process_directory(self, input_dir: str, output_dir: str) -> Dict[str, str]:
        return process_directory(self.spark, input_dir, output_dir)

    def process_path(self, input_path: str, output_base: str, fmt: str = "auto") -> Dict[str, str]:
        """
        Process climate data from input_path (local or HDFS) and write Parquet outputs.
        
        Args:
            input_path: Path to input data (local file/dir or hdfs://)
            output_base: Base path for output Parquet files
            fmt: Format of input data ("auto", "berkeley-txt", "city-csv")
            
        Returns:
            Dictionary mapping artifact names to output paths
        """
        logger.info(f"Processing path: {input_path} -> {output_base}")
        
        # Read data based on format
        if fmt == "city-csv" or (fmt == "auto" and input_path.endswith(".csv")):
            df = self.read_city_temperature_csv_path(input_path)
        elif fmt == "berkeley-txt" or (fmt == "auto" and input_path.endswith(".txt")):
            df = self.read_berkeley_earth_path(input_path)
        else:
            # Try city CSV by default for HDFS paths
            df = self.read_city_temperature_csv_path(input_path)
        
        logger.info(f"Data loaded: {df.count()} records")
        
        # Clean data
        cleaned_df = self.clean_temperature_data(df)
        
        # Add geographic columns (region, continent)
        from .cleaning import add_geographic_columns
        cleaned_df = add_geographic_columns(cleaned_df)
        cleaned_df.cache()
        logger.info(f"Data cleaned and enriched: {cleaned_df.count()} records")
        
        artifacts = {}
        
        # Monthly aggregations
        logger.info("Computing monthly aggregations...")
        monthly_df = self.aggregate_monthly_data(cleaned_df)
        monthly_path = f"{output_base}/monthly.parquet"
        monthly_df.write.mode("overwrite").parquet(monthly_path)
        artifacts["monthly"] = monthly_path
        
        # Yearly aggregations
        logger.info("Computing yearly aggregations...")
        yearly_df = self.aggregate_yearly_data(cleaned_df)
        yearly_path = f"{output_base}/yearly.parquet"
        yearly_df.write.mode("overwrite").parquet(yearly_path)
        artifacts["yearly"] = yearly_path
        
        # Anomalies detection
        logger.info("Detecting anomalies...")
        anomalies_df = self.detect_anomalies(cleaned_df)
        anomalies_path = f"{output_base}/anomalies.parquet"
        anomalies_df.write.mode("overwrite").parquet(anomalies_path)
        artifacts["anomalies"] = anomalies_path
        
        # Climatology stats
        logger.info("Computing climatology stats...")
        climatology_df = self.compute_climatology_stats(cleaned_df)
        climatology_path = f"{output_base}/climatology.parquet"
        climatology_df.write.mode("overwrite").parquet(climatology_path)
        artifacts["climatology"] = climatology_path
        
        # Seasonal stats
        logger.info("Computing seasonal stats...")
        seasonal_df = self.compute_seasonal_stats(cleaned_df)
        seasonal_path = f"{output_base}/seasonal.parquet"
        seasonal_df.write.mode("overwrite").parquet(seasonal_path)
        artifacts["seasonal"] = seasonal_path
        
        # Extreme thresholds
        logger.info("Computing extreme thresholds...")
        extreme_df = self.compute_extreme_thresholds(cleaned_df)
        extreme_path = f"{output_base}/extreme_thresholds.parquet"
        extreme_df.write.mode("overwrite").parquet(extreme_path)
        artifacts["extreme_thresholds"] = extreme_path
        
        # Import additional aggregations
        from .aggregation import aggregate_by_region, aggregate_by_continent, aggregate_by_country
        
        # Regional aggregations
        logger.info("Computing regional aggregations...")
        regional_df = aggregate_by_region(cleaned_df)
        regional_path = f"{output_base}/regional.parquet"
        regional_df.write.mode("overwrite").parquet(regional_path)
        artifacts["regional"] = regional_path
        
        # Continental aggregations
        logger.info("Computing continental aggregations...")
        continental_df = aggregate_by_continent(cleaned_df)
        continental_path = f"{output_base}/continental.parquet"
        continental_df.write.mode("overwrite").parquet(continental_path)
        artifacts["continental"] = continental_path
        
        # Country aggregations
        logger.info("Computing country aggregations...")
        country_df = aggregate_by_country(cleaned_df)
        country_path = f"{output_base}/country.parquet"
        country_df.write.mode("overwrite").parquet(country_path)
        artifacts["country"] = country_path
        
        # EDA: correlation matrix, descriptive stats, chi-square tests
        from .analysis import compute_correlation_matrix, compute_descriptive_stats, compute_chi_square_tests
        
        logger.info("Computing EDA statistics...")
        
        corr_df = compute_correlation_matrix(cleaned_df)
        corr_path = f"{output_base}/correlation_matrix.parquet"
        corr_df.write.mode("overwrite").parquet(corr_path)
        artifacts["correlation_matrix"] = corr_path
        
        desc_df = compute_descriptive_stats(cleaned_df)
        desc_path = f"{output_base}/descriptive_stats.parquet"
        desc_df.write.mode("overwrite").parquet(desc_path)
        artifacts["descriptive_stats"] = desc_path
        
        chi_df = compute_chi_square_tests(cleaned_df)
        chi_path = f"{output_base}/chi_square_tests.parquet"
        chi_df.write.mode("overwrite").parquet(chi_path)
        artifacts["chi_square_tests"] = chi_path
        
        cleaned_df.unpersist()
        logger.info(f"Processing complete. Generated {len(artifacts)} artifacts.")
        
        return artifacts