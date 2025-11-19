"""
Spark-related modules for climate data preprocessing.
"""

from .spark_processor import SparkPreprocessor
from .spark_session_manager import SparkSessionManager
from .readers import (
    read_berkeley_earth_file, read_berkeley_earth_path, read_city_temperature_csv_path
)
from .cleaning import clean_temperature_data
from .aggregation import aggregate_monthly_data, aggregate_yearly_data
from .analysis import (
    detect_anomalies,
    compute_climatology_stats,
    compute_seasonal_stats,
    compute_extreme_thresholds,
    compute_trend_line,
)
from .processing import process_directory

__all__ = [
    "SparkPreprocessor",
    "SparkSessionManager",
    "read_berkeley_earth_file",
    "read_berkeley_earth_path",
    "read_city_temperature_csv_path",
    "clean_temperature_data",
    "aggregate_monthly_data",
    "aggregate_yearly_data",
    "detect_anomalies",
    "compute_climatology_stats",
    "compute_seasonal_stats",
    "compute_extreme_thresholds",
    "compute_trend_line",
    "process_directory",
]