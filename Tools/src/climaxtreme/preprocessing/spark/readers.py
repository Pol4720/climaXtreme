"""
Data reading functions for various climate data formats.
"""

import logging
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, split, trim, regexp_replace, to_date, year as pyear, month as pmonth
from pyspark.sql.types import IntegerType, DoubleType

logger = logging.getLogger(__name__)

def read_berkeley_earth_file(spark: SparkSession, filepath: str) -> DataFrame:
    """
    Reads a Berkeley Earth data file into a Spark DataFrame.
    
    Args:
        spark: The Spark session.
        filepath: The path to the data file.
        
    Returns:
        A Spark DataFrame with the data.
    """
    try:
        df = (
            spark.read
            .option("header", "false")
            .option("comment", "%")
            .option("delimiter", " ")
            .option("multiline", "true")
            .text(filepath)
        )
        
        split_col = split(trim(regexp_replace(col("value"), r"\s+", " ")), " ")
        
        processed_df = (
            df
            .filter(~col("value").startswith("%"))
            .filter(col("value") != "")
            .withColumn("year", split_col.getItem(0).cast(IntegerType()))
            .withColumn("month", split_col.getItem(1).cast(IntegerType()))
            .withColumn("temperature", split_col.getItem(2).cast(DoubleType()))
            .withColumn("uncertainty", split_col.getItem(3).cast(DoubleType()))
            .drop("value")
        )
        
        logger.info(f"Successfully loaded {filepath} with {processed_df.count()} records")
        return processed_df
        
    except Exception as e:
        logger.error(f"Error reading {filepath}: {e}")
        raise

def read_berkeley_earth_path(spark: SparkSession, input_path: str) -> DataFrame:
    """
    Reads Berkeley Earth formatted text files from a path or glob pattern.
    """
    try:
        df = (
            spark.read
            .option("header", "false")
            .option("comment", "%")
            .option("delimiter", " ")
            .option("multiline", "true")
            .text(input_path)
        )

        split_col = split(trim(regexp_replace(col("value"), r"\s+", " ")), " ")

        processed_df = (
            df
            .filter(~col("value").startswith("%"))
            .filter(col("value") != "")
            .withColumn("year", split_col.getItem(0).cast(IntegerType()))
            .withColumn("month", split_col.getItem(1).cast(IntegerType()))
            .withColumn("temperature", split_col.getItem(2).cast(DoubleType()))
            .withColumn("uncertainty", split_col.getItem(3).cast(DoubleType()))
            .drop("value")
        )
        return processed_df
    except Exception as e:
        logger.error(f"Error reading path {input_path}: {e}")
        raise

def read_city_temperature_csv_path(spark: SparkSession, input_path: str) -> DataFrame:
    """
    Reads GlobalLandTemperaturesByCity.csv (or compatible) from local or HDFS.
    """
    try:
        logger.info(f"Reading CSV from {input_path}...")
        
        df = (
            spark.read
            .option("header", True)
            .option("inferSchema", False)
            .option("mode", "DROPMALFORMED")
            .csv(input_path)
        )

        df = (
            df
            .select(
                col("dt").alias("dt"),
                col("AverageTemperature").cast(DoubleType()).alias("temperature"),
                col("AverageTemperatureUncertainty").cast(DoubleType()).alias("uncertainty"),
                col("City").alias("city"),
                col("Country").alias("country"),
                col("Latitude").alias("latitude"),
                col("Longitude").alias("longitude")
            )
            .withColumn("date", to_date(col("dt")))
            .dropna(subset=["date", "temperature"])
            .withColumn("year", pyear(col("date")))
            .withColumn("month", pmonth(col("date")))
        )
        
        row_count = df.count()
        logger.info(f"Successfully loaded {row_count:,} records from {input_path}")
        
        return df
    except Exception as e:
        logger.error(f"Error reading CSV path {input_path}: {e}")
        raise