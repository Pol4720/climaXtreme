"""
Main processing pipeline for climate data.
"""

import logging
from pathlib import Path
from typing import Dict
from pyspark.sql import SparkSession

from .readers import read_berkeley_earth_file
from .cleaning import clean_temperature_data
from .aggregation import aggregate_monthly_data, aggregate_yearly_data, aggregate_by_country, aggregate_by_continent, aggregate_by_region
from .analysis import detect_anomalies

logger = logging.getLogger(__name__)

def process_directory(spark: SparkSession, input_dir: str, output_dir: str) -> Dict[str, str]:
    """
    Processes all data files in a directory.
    
    Args:
        spark: The Spark session.
        input_dir: Directory containing raw data files.
        output_dir: Directory to store processed data.
        
    Returns:
        A dictionary mapping input files to output files.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    processed_files = {}
    
    for txt_file in input_path.glob("*.txt"):
        try:
            logger.info(f"Processing {txt_file.name}...")
            
            df = read_berkeley_earth_file(spark, str(txt_file))
            cleaned_df = clean_temperature_data(df)
            
            monthly_df = aggregate_monthly_data(cleaned_df)
            yearly_df = aggregate_yearly_data(cleaned_df)
            anomaly_df = detect_anomalies(cleaned_df)
            
            base_name = txt_file.stem
            
            monthly_output = output_path / f"{base_name}_monthly.parquet"
            yearly_output = output_path / f"{base_name}_yearly.parquet"
            anomaly_output = output_path / f"{base_name}_anomalies.parquet"
            
            monthly_df.write.mode("overwrite").parquet(str(monthly_output))
            yearly_df.write.mode("overwrite").parquet(str(yearly_output))
            anomaly_df.write.mode("overwrite").parquet(str(anomaly_output))
            
            processed_files[str(txt_file)] = {
                "monthly": str(monthly_output),
                "yearly": str(yearly_output),
                "anomalies": str(anomaly_output)
            }
            
            logger.info(f"Finished processing {txt_file.name}")
            
        except Exception as e:
            logger.error(f"Failed to process {txt_file.name}: {e}")
            
    return processed_files