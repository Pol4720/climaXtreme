"""
Data cleaning functions for climate data.
"""

import logging
from pyspark.sql import DataFrame
from pyspark.sql.functions import col

logger = logging.getLogger(__name__)

def clean_temperature_data(df: DataFrame) -> DataFrame:
    """
    Cleans temperature data by removing outliers and invalid values.
    
    Args:
        df: Input DataFrame with temperature data.
        
    Returns:
        A cleaned DataFrame.
    """
    cleaned_df = (
        df
        .filter(col("temperature").isNotNull())
        .filter(col("year").isNotNull() & (col("year") > 1750) & (col("year") <= 2030))
        .filter(col("month").isNotNull() & (col("month") >= 1) & (col("month") <= 12))
        .filter(col("temperature") >= -100.0)
        .filter(col("temperature") <= 60.0)
    )
    
    original_count = df.count()
    cleaned_count = cleaned_df.count()
    removed_count = original_count - cleaned_count
    
    if original_count > 0:
        logger.info(
            f"Data cleaning: {original_count} -> {cleaned_count} records "
            f"({removed_count} removed, {removed_count/original_count*100:.1f}%)"
        )
    
    return cleaned_df