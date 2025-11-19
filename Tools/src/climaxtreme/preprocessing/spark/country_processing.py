import logging
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr, when
import pycountry
import pycountry_convert as pc

from .readers import read_city_temperature_csv_path
from .cleaning import clean_temperature_data
from .aggregation import aggregate_by_country, aggregate_by_continent, aggregate_by_region

logger = logging.getLogger(__name__)

def get_country_code(country_name):
    try:
        return pycountry.countries.get(name=country_name).alpha_3
    except AttributeError:
        return None

def get_continent_name(country_name):
    try:
        country_alpha2 = pc.country_name_to_country_alpha2(country_name)
        continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
        continent_name = pc.convert_continent_code_to_continent_name(continent_code)
        return continent_name
    except (KeyError, AttributeError):
        return None

def process_city_data(spark: SparkSession, input_path: str, output_dir: str):
    df = read_city_temperature_csv_path(spark, input_path)
    df = clean_temperature_data(df)

    # Add country codes
    spark.udf.register("get_country_code", get_country_code)
    df = df.withColumn("country_code", expr("get_country_code(country)"))

    # Add continent column
    spark.udf.register("get_continent_name", get_continent_name)
    df = df.withColumn("continent", expr("get_continent_name(country)"))

    country_df = aggregate_by_country(df)
    continent_df = aggregate_by_continent(df)
    regional_df = aggregate_by_region(df)

    output_path = Path(output_dir)
    country_df.write.mode("overwrite").parquet(str(output_path / "country.parquet"))
    continent_df.write.mode("overwrite").parquet(str(output_path / "continental.parquet"))
    regional_df.write.mode("overwrite").parquet(str(output_path / "regional.parquet"))

    logger.info(f"Successfully processed city data and saved to {output_dir}")