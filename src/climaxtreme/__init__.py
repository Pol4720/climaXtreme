"""
climaXtreme: Climate analysis and extreme event modeling using Hadoop and PySpark.

This package provides tools for:
- Ingesting and preprocessing Berkeley Earth climate data
- PySpark-based climate analysis (heatmaps, time series)
- Interactive Streamlit dashboard
- Machine learning models for climate prediction
"""

__version__ = "0.1.0"
__author__ = "climaXtreme Team"
__email__ = "team@climaxtreme.com"

from .data import DataIngestion, DataValidator
from .preprocessing import DataPreprocessor, SparkPreprocessor
from .analysis import HeatmapAnalyzer, TimeSeriesAnalyzer
from .ml import BaselineModel, ClimatePredictor

__all__ = [
    "DataIngestion",
    "DataValidator", 
    "DataPreprocessor",
    "SparkPreprocessor",
    "HeatmapAnalyzer",
    "TimeSeriesAnalyzer",
    "BaselineModel",
    "ClimatePredictor"
]