"""
Data analysis functions for climate data.
"""

import logging
from typing import List
from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    avg, col, count, lit, percentile_approx, stddev, when, 
    min as spark_min, max as spark_max, abs as spark_abs
)

logger = logging.getLogger(__name__)

def detect_anomalies(df: DataFrame, threshold_std: float = 3.0) -> DataFrame:
    """
    Detects temperature anomalies using statistical methods.
    
    Args:
        df: Input DataFrame with temperature data.
        threshold_std: Standard deviation threshold for anomaly detection.
        
    Returns:
        A DataFrame with anomaly flags.
    """
    stats = df.select(
        avg("temperature").alias("mean_temp"),
    ).collect()[0]
    
    mean_temp = stats["mean_temp"]
    
    std_df = (
        df
        .select(((col("temperature") - mean_temp) ** 2).alias("squared_diff"))
        .agg(avg("squared_diff").alias("variance"))
    )
    
    variance = std_df.collect()[0]["variance"]
    std_temp = variance ** 0.5
    
    anomaly_df = (
        df
        .withColumn("temp_zscore", (col("temperature") - mean_temp) / std_temp)
        .withColumn("is_anomaly", when(spark_abs(col("temp_zscore")) > threshold_std, True).otherwise(False))
    )
    
    anomaly_count = anomaly_df.filter(col("is_anomaly")).count()
    total_count = anomaly_df.count()
    
    if total_count > 0:
        logger.info(
            f"Detected {anomaly_count} anomalies out of {total_count} records "
            f"({anomaly_count/total_count*100:.2f}%)"
        )
    
    return anomaly_df

def compute_climatology_stats(df: DataFrame) -> DataFrame:
    """
    Computes climatology statistics by month.
    
    Args:
        df: Input DataFrame with temperature data.
        
    Returns:
        A DataFrame with monthly climatology stats.
    """
    climatology = (
        df
        .groupBy("month")
        .agg(
            avg("temperature").alias("climatology_mean"),
            stddev("temperature").alias("climatology_std"),
            spark_min("temperature").alias("climatology_min"),
            spark_max("temperature").alias("climatology_max"),
            count("temperature").alias("climatology_count")
        )
        .orderBy("month")
    )
    
    logger.info("Computed monthly climatology statistics")
    return climatology

def compute_seasonal_stats(df: DataFrame) -> DataFrame:
    """
    Computes seasonal temperature statistics.
    
    Args:
        df: Input DataFrame with temperature data.
        
    Returns:
        A DataFrame with seasonal aggregations.
    """
    df_with_season = df.withColumn(
        "season",
        when((col("month") == 12) | (col("month") == 1) | (col("month") == 2), "Winter")
        .when((col("month") >= 3) & (col("month") <= 5), "Spring")
        .when((col("month") >= 6) & (col("month") <= 8), "Summer")
        .otherwise("Fall")
    )
    
    seasonal_stats = (
        df_with_season
        .groupBy("season")
        .agg(
            avg("temperature").alias("avg_temperature"),
            stddev("temperature").alias("std_temperature"),
            spark_min("temperature").alias("min_temperature"),
            spark_max("temperature").alias("max_temperature"),
            count("temperature").alias("record_count")
        )
    )
    
    logger.info("Computed seasonal statistics")
    return seasonal_stats

def compute_extreme_thresholds(df: DataFrame, percentiles: List[float] = [90.0, 95.0, 99.0]) -> DataFrame:
    """
    Computes temperature thresholds for extreme event detection.
    
    Args:
        df: Input DataFrame with temperature data.
        percentiles: List of percentiles to compute.
        
    Returns:
        A DataFrame with percentile thresholds.
    """
    thresholds_data = []
    
    for p in percentiles:
        high_p = p / 100.0
        low_p = (100 - p) / 100.0
        
        percentiles_result = df.select(
            percentile_approx("temperature", high_p).alias(f"p{int(p)}_high"),
            percentile_approx("temperature", low_p).alias(f"p{int(p)}_low")
        ).collect()[0]
        
        thresholds_data.append({
            "percentile": p,
            "high_threshold": float(percentiles_result[f"p{int(p)}_high"]),
            "low_threshold": float(percentiles_result[f"p{int(p)}_low"])
        })
    
    spark = df.sparkSession
    thresholds_df = spark.createDataFrame(thresholds_data)
    
    logger.info(f"Computed extreme event thresholds for percentiles: {percentiles}")
    return thresholds_df

def compute_trend_line(df: DataFrame) -> DataFrame:
    """
    Computes a linear trend line for yearly temperature data.
    
    Args:
        df: Input DataFrame with year and avg_temperature columns.
        
    Returns:
        A DataFrame with trend line values.
    """
    yearly_data = df.select("year", "avg_temperature").orderBy("year").collect()
    
    if len(yearly_data) < 2:
        logger.warning("Insufficient data for trend calculation")
        return df
    
    years = [row.year for row in yearly_data]
    temps = [row.avg_temperature for row in yearly_data]
    
    n = len(years)
    sum_x = sum(years)
    sum_y = sum(temps)
    sum_xy = sum(x * y for x, y in zip(years, temps))
    sum_x2 = sum(x * x for x in years)
    
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
    intercept = (sum_y - slope * sum_x) / n
    
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


def compute_correlation_matrix(df: DataFrame) -> DataFrame:
    """
    Computes correlation matrix for numeric columns.
    
    Args:
        df: Input DataFrame with temperature data.
        
    Returns:
        DataFrame with correlation coefficients.
    """
    from pyspark.sql.functions import corr
    
    # Select numeric columns for correlation
    numeric_cols = ["year", "month", "temperature"]
    
    # Check if uncertainty column exists
    if "uncertainty" in df.columns:
        numeric_cols.append("uncertainty")
    
    correlations = []
    for col1 in numeric_cols:
        row_data = {"variable": col1}
        for col2 in numeric_cols:
            corr_value = df.select(corr(col1, col2)).collect()[0][0]
            row_data[col2] = float(corr_value) if corr_value is not None else 0.0
        correlations.append(row_data)
    
    spark = df.sparkSession
    corr_df = spark.createDataFrame(correlations)
    
    logger.info("Computed correlation matrix")
    return corr_df


def compute_descriptive_stats(df: DataFrame) -> DataFrame:
    """
    Computes descriptive statistics for temperature data.
    
    Args:
        df: Input DataFrame with temperature data.
        
    Returns:
        DataFrame with descriptive statistics.
    """
    from pyspark.sql.functions import skewness, kurtosis, variance
    
    stats = df.select(
        count("temperature").alias("count"),
        avg("temperature").alias("mean"),
        stddev("temperature").alias("std"),
        variance("temperature").alias("variance"),
        spark_min("temperature").alias("min"),
        spark_max("temperature").alias("max"),
        percentile_approx("temperature", 0.25).alias("q1"),
        percentile_approx("temperature", 0.50).alias("median"),
        percentile_approx("temperature", 0.75).alias("q3"),
        skewness("temperature").alias("skewness"),
        kurtosis("temperature").alias("kurtosis")
    )
    
    logger.info("Computed descriptive statistics")
    return stats


def compute_chi_square_tests(df: DataFrame) -> DataFrame:
    """
    Computes chi-square test for categorical variables.
    
    Args:
        df: Input DataFrame with temperature data.
        
    Returns:
        DataFrame with chi-square test results.
    """
    # Create temperature categories
    df_cat = df.withColumn(
        "temp_category",
        when(col("temperature") < 0, "Cold")
        .when(col("temperature") < 15, "Mild")
        .when(col("temperature") < 25, "Warm")
        .otherwise("Hot")
    )
    
    # Create season from month if not exists
    df_cat = df_cat.withColumn(
        "season",
        when((col("month") == 12) | (col("month") == 1) | (col("month") == 2), "Winter")
        .when((col("month") >= 3) & (col("month") <= 5), "Spring")
        .when((col("month") >= 6) & (col("month") <= 8), "Summer")
        .otherwise("Fall")
    )
    
    # Compute contingency table counts
    contingency = (
        df_cat
        .groupBy("season", "temp_category")
        .count()
        .orderBy("season", "temp_category")
    )
    
    # Create a summary of the contingency table
    results = []
    seasons = ["Winter", "Spring", "Summer", "Fall"]
    categories = ["Cold", "Mild", "Warm", "Hot"]
    
    for season in seasons:
        row_data = {"season": season}
        for cat in categories:
            cnt = contingency.filter(
                (col("season") == season) & (col("temp_category") == cat)
            ).select("count").collect()
            row_data[cat] = cnt[0][0] if cnt else 0
        results.append(row_data)
    
    spark = df.sparkSession
    chi_df = spark.createDataFrame(results)
    
    logger.info("Computed chi-square contingency table")
    return chi_df