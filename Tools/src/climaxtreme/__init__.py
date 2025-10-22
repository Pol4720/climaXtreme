"""
climaXtreme: Climate analysis and extreme event modeling using pandas and PySpark.

This package provides tools for:
- Ingesting and preprocessing Berkeley Earth climate data
- PySpark-based climate analysis (heatmaps, time series)
- Interactive Streamlit dashboard
- Machine learning models for climate prediction

Note: Heavy optional dependencies (e.g., PySpark, Streamlit) are NOT imported at
package import time to keep lightweight modules (like `climaxtreme.utils`) usable
without requiring the full stack to be installed. Import subpackages directly, e.g.:
    from climaxtreme.data import DataValidator
    from climaxtreme.preprocessing import DataPreprocessor
"""

__version__ = "0.1.0"
__author__ = "climaXtreme Team"
__email__ = "team@climaxtreme.com"

# Avoid importing heavy submodules at package import time. Downstream code should
# import from the specific subpackages (e.g., `from climaxtreme.data import ...`).
# We still attempt to expose a few common classes when dependencies are available,
# but failures here are non-fatal so that light modules (like utils) can be used
# without PySpark/Streamlit installed.

__all__ = []

try:  # Light-weight data utilities
    from .data import DataIngestion, DataValidator  # type: ignore
    __all__ += ["DataIngestion", "DataValidator"]
except Exception:
    # Optional: not available without full dependency stack
    pass

try:  # Pandas preprocessor (light-weight)
    from .preprocessing import DataPreprocessor  # type: ignore
    __all__ += ["DataPreprocessor"]
except Exception:
    pass

try:  # Spark preprocessor (requires pyspark)
    from .preprocessing import SparkPreprocessor  # type: ignore
    __all__ += ["SparkPreprocessor"]
except Exception:
    pass

try:  # Analysis modules may require plotting libs
    from .analysis import HeatmapAnalyzer, TimeSeriesAnalyzer  # type: ignore
    __all__ += ["HeatmapAnalyzer", "TimeSeriesAnalyzer"]
except Exception:
    pass

try:  # ML modules may require sklearn
    from .ml import BaselineModel, ClimatePredictor  # type: ignore
    __all__ += ["BaselineModel", "ClimatePredictor"]
except Exception:
    pass