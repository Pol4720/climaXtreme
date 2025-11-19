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
from .processing import process_directory, process_city_data

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

    def process_city_data(self, input_path: str, output_dir: str):
        return process_city_data(self.spark, input_path, output_dir)