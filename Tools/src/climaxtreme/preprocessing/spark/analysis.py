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
    Computes comprehensive descriptive statistics for all numeric variables.
    Includes distribution analysis, normality indicators, and outlier detection.
    
    Args:
        df: Input DataFrame with temperature data.
        
    Returns:
        DataFrame with descriptive statistics for each variable.
    """
    from pyspark.sql.functions import skewness, kurtosis, variance, sqrt, sum as spark_sum
    from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType
    
    # Define numeric columns to analyze
    numeric_cols = ["temperature"]
    
    # Check for uncertainty column
    if "uncertainty" in df.columns:
        numeric_cols.append("uncertainty")
    
    results = []
    
    for var_name in numeric_cols:
        # Basic statistics
        basic_stats = df.select(
            count(col(var_name)).alias("count"),
            avg(col(var_name)).alias("mean"),
            stddev(col(var_name)).alias("std"),
            variance(col(var_name)).alias("variance"),
            spark_min(col(var_name)).alias("min"),
            spark_max(col(var_name)).alias("max"),
            percentile_approx(col(var_name), 0.25).alias("q1"),
            percentile_approx(col(var_name), 0.50).alias("median"),
            percentile_approx(col(var_name), 0.75).alias("q3"),
            percentile_approx(col(var_name), 0.05).alias("p5"),
            percentile_approx(col(var_name), 0.10).alias("p10"),
            percentile_approx(col(var_name), 0.90).alias("p90"),
            percentile_approx(col(var_name), 0.95).alias("p95"),
            skewness(col(var_name)).alias("skewness"),
            kurtosis(col(var_name)).alias("kurtosis")
        ).collect()[0]
        
        n = basic_stats["count"]
        mean_val = basic_stats["mean"]
        std_val = basic_stats["std"]
        min_val = basic_stats["min"]
        max_val = basic_stats["max"]
        q1 = basic_stats["q1"]
        median_val = basic_stats["median"]
        q3 = basic_stats["q3"]
        skew = basic_stats["skewness"]
        kurt = basic_stats["kurtosis"]
        
        # Derived statistics
        iqr = q3 - q1 if q1 is not None and q3 is not None else None
        range_val = max_val - min_val if min_val is not None and max_val is not None else None
        cv = (std_val / mean_val * 100) if mean_val and mean_val != 0 and std_val else None  # Coefficient of variation
        
        # Standard error of mean
        se_mean = std_val / (n ** 0.5) if std_val and n else None
        
        # Outlier bounds (IQR method)
        lower_fence = q1 - 1.5 * iqr if q1 is not None and iqr is not None else None
        upper_fence = q3 + 1.5 * iqr if q3 is not None and iqr is not None else None
        
        # Count outliers
        outlier_count = 0
        if lower_fence is not None and upper_fence is not None:
            outlier_count = df.filter(
                (col(var_name) < lower_fence) | (col(var_name) > upper_fence)
            ).count()
        
        outlier_pct = (outlier_count / n * 100) if n > 0 else 0
        
        # Normality indicators based on skewness and kurtosis
        # Jarque-Bera approximation: JB = n/6 * (S^2 + K^2/4)
        # For normal distribution: skewness ≈ 0, excess kurtosis ≈ 0
        jb_statistic = None
        jb_pvalue_approx = None
        normality_assessment = "Unknown"
        
        if skew is not None and kurt is not None and n > 0:
            # Jarque-Bera statistic
            jb_statistic = (n / 6.0) * (skew ** 2 + (kurt ** 2) / 4.0)
            
            # Approximate p-value using chi-square distribution with 2 df
            # This is a rough approximation
            if jb_statistic < 6:
                jb_pvalue_approx = 0.05  # Likely normal
            else:
                jb_pvalue_approx = 0.001  # Likely not normal
            
            # Practical assessment
            if abs(skew) < 0.5 and abs(kurt) < 1:
                normality_assessment = "Approximately Normal"
            elif abs(skew) < 1 and abs(kurt) < 2:
                normality_assessment = "Moderately Non-Normal"
            else:
                normality_assessment = "Significantly Non-Normal"
        
        # Distribution shape description
        if skew is not None:
            if skew < -1:
                skew_desc = "Highly Left-Skewed"
            elif skew < -0.5:
                skew_desc = "Moderately Left-Skewed"
            elif skew < 0.5:
                skew_desc = "Approximately Symmetric"
            elif skew < 1:
                skew_desc = "Moderately Right-Skewed"
            else:
                skew_desc = "Highly Right-Skewed"
        else:
            skew_desc = "Unknown"
        
        if kurt is not None:
            if kurt < -1:
                kurt_desc = "Platykurtic (Light Tails)"
            elif kurt < 1:
                kurt_desc = "Mesokurtic (Normal Tails)"
            else:
                kurt_desc = "Leptokurtic (Heavy Tails)"
        else:
            kurt_desc = "Unknown"
        
        results.append({
            "variable": var_name,
            "count": int(n) if n else 0,
            "mean": float(mean_val) if mean_val is not None else None,
            "std": float(std_val) if std_val is not None else None,
            "variance": float(basic_stats["variance"]) if basic_stats["variance"] is not None else None,
            "se_mean": float(se_mean) if se_mean is not None else None,
            "cv_percent": float(cv) if cv is not None else None,
            "min": float(min_val) if min_val is not None else None,
            "p5": float(basic_stats["p5"]) if basic_stats["p5"] is not None else None,
            "p10": float(basic_stats["p10"]) if basic_stats["p10"] is not None else None,
            "q1": float(q1) if q1 is not None else None,
            "median": float(median_val) if median_val is not None else None,
            "q3": float(q3) if q3 is not None else None,
            "p90": float(basic_stats["p90"]) if basic_stats["p90"] is not None else None,
            "p95": float(basic_stats["p95"]) if basic_stats["p95"] is not None else None,
            "max": float(max_val) if max_val is not None else None,
            "range": float(range_val) if range_val is not None else None,
            "iqr": float(iqr) if iqr is not None else None,
            "skewness": float(skew) if skew is not None else None,
            "kurtosis": float(kurt) if kurt is not None else None,
            "skewness_interpretation": skew_desc,
            "kurtosis_interpretation": kurt_desc,
            "lower_fence": float(lower_fence) if lower_fence is not None else None,
            "upper_fence": float(upper_fence) if upper_fence is not None else None,
            "outlier_count": int(outlier_count),
            "outlier_percent": float(outlier_pct),
            "jb_statistic": float(jb_statistic) if jb_statistic is not None else None,
            "normality_assessment": normality_assessment
        })
    
    spark = df.sparkSession
    stats_df = spark.createDataFrame(results)
    
    logger.info(f"Computed comprehensive descriptive statistics for {len(numeric_cols)} variables")
    return stats_df


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