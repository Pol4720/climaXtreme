"""
Data aggregation functions for climate data.
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import avg, count, min as spark_min, max as spark_max, stddev, first

def aggregate_monthly_data(df: DataFrame) -> DataFrame:
    """
    Aggregates temperature data by year and month.
    
    Args:
        df: Input DataFrame with temperature data.
        
    Returns:
        A DataFrame with monthly aggregations.
    """
    monthly_agg = (
        df
        .groupBy("year", "month")
        .agg(
            avg("temperature").alias("avg_temperature"),
            spark_min("temperature").alias("min_temperature"),
            spark_max("temperature").alias("max_temperature"),
            stddev("temperature").alias("std_temperature"),
            count("temperature").alias("record_count"),
            avg("uncertainty").alias("avg_uncertainty")
        )
        .orderBy("year", "month")
    )
    
    return monthly_agg

def aggregate_yearly_data(df: DataFrame) -> DataFrame:
    """
    Aggregates temperature data by year.
    
    Args:
        df: Input DataFrame with temperature data.
        
    Returns:
        A DataFrame with yearly aggregations.
    """
    yearly_agg = (
        df
        .groupBy("year")
        .agg(
            avg("temperature").alias("avg_temperature"),
            spark_min("temperature").alias("min_temperature"),
            spark_max("temperature").alias("max_temperature"),
            stddev("temperature").alias("std_temperature"),
            count("temperature").alias("record_count"),
            avg("uncertainty").alias("avg_uncertainty")
        )
        .orderBy("year")
    )
    
    return yearly_agg

def aggregate_by_country(df: DataFrame) -> DataFrame:
    """
    Aggregates temperature data by year and country.
    """
    country_agg = (
        df
        .groupBy("year", "country")
        .agg(
            avg("temperature").alias("avg_temperature"),
            first("country_code").alias("country_code"),
            first("continent").alias("continent")
        )
        .orderBy("year", "country")
    )
    return country_agg

def aggregate_by_continent(df: DataFrame) -> DataFrame:
    """
    Aggregates temperature data by year and continent.
    """
    continent_agg = (
        df
        .groupBy("year", "continent")
        .agg(
            avg("temperature").alias("avg_temperature")
        )
        .orderBy("year", "continent")
    )
    return continent_agg

def aggregate_by_region(df: DataFrame) -> DataFrame:
    """
    Aggregates temperature data by year and region.
    """
    region_agg = (
        df
        .groupBy("year", "region", "continent")
        .agg(
            avg("temperature").alias("avg_temperature")
        )
        .orderBy("year", "region")
    )
    return region_agg