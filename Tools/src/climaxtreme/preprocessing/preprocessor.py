"""
Traditional pandas-based preprocessing for smaller datasets.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Traditional pandas-based preprocessor for smaller climate datasets.
    
    Provides similar functionality to SparkPreprocessor but uses pandas
    for datasets that fit in memory.
    """
    
    def __init__(self) -> None:
        """Initialize the preprocessor."""
        pass
    
    def read_berkeley_earth_file(self, filepath: str) -> pd.DataFrame:
        """
        Read Berkeley Earth data file into pandas DataFrame.
        
        Args:
            filepath: Path to the data file
            
        Returns:
            pandas DataFrame with the data
        """
        try:
            # Read file, skipping comment lines
            with open(filepath, 'r') as f:
                lines = [line for line in f if not line.startswith('%') and line.strip()]
            
            # Parse the data - typical format: Year Month Temperature Uncertainty
            data = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 3:
                    try:
                        year = int(float(parts[0]))
                        month = int(float(parts[1]))
                        temperature = float(parts[2])
                        uncertainty = float(parts[3]) if len(parts) > 3 else np.nan
                        
                        data.append({
                            'year': year,
                            'month': month,
                            'temperature': temperature,
                            'uncertainty': uncertainty
                        })
                    except (ValueError, IndexError):
                        continue
            
            df = pd.DataFrame(data)
            logger.info(f"Successfully loaded {filepath} with {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
            raise
    
    def clean_temperature_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean temperature data by removing outliers and invalid values.
        
        Args:
            df: Input DataFrame with temperature data
            
        Returns:
            Cleaned DataFrame
        """
        original_count = len(df)
        
        # Remove records with invalid values
        cleaned_df = df[
            df['temperature'].notna() &
            df['year'].notna() &
            (df['year'] > 1750) &
            (df['year'] <= 2030) &
            df['month'].notna() &
            (df['month'] >= 1) &
            (df['month'] <= 12) &
            (df['temperature'] >= -100.0) &
            (df['temperature'] <= 60.0)
        ].copy()
        
        cleaned_count = len(cleaned_df)
        removed_count = original_count - cleaned_count
        
        logger.info(f"Data cleaning: {original_count} -> {cleaned_count} records "
                   f"({removed_count} removed, {removed_count/original_count*100:.1f}%)")
        
        return cleaned_df
    
    def aggregate_monthly_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate temperature data by year and month.
        
        Args:
            df: Input DataFrame with temperature data
            
        Returns:
            DataFrame with monthly aggregations
        """
        monthly_agg = (df
                      .groupby(['year', 'month'])
                      .agg({
                          'temperature': ['mean', 'min', 'max', 'count'],
                          'uncertainty': 'mean'
                      })
                      .round(3))
        
        # Flatten column names
        monthly_agg.columns = [
            'avg_temperature', 'min_temperature', 'max_temperature', 
            'record_count', 'avg_uncertainty'
        ]
        
        return monthly_agg.reset_index()
    
    def aggregate_yearly_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate temperature data by year.
        
        Args:
            df: Input DataFrame with temperature data
            
        Returns:
            DataFrame with yearly aggregations
        """
        yearly_agg = (df
                     .groupby('year')
                     .agg({
                         'temperature': ['mean', 'min', 'max', 'count'],
                         'uncertainty': 'mean'
                     })
                     .round(3))
        
        # Flatten column names
        yearly_agg.columns = [
            'avg_temperature', 'min_temperature', 'max_temperature',
            'record_count', 'avg_uncertainty'
        ]
        
        return yearly_agg.reset_index()
    
    def detect_anomalies(self, df: pd.DataFrame, threshold_std: float = 3.0) -> pd.DataFrame:
        """
        Detect temperature anomalies using statistical methods.
        
        Args:
            df: Input DataFrame with temperature data
            threshold_std: Standard deviation threshold for anomaly detection
            
        Returns:
            DataFrame with anomaly flags
        """
        df = df.copy()
        
        # Calculate z-scores
        mean_temp = df['temperature'].mean()
        std_temp = df['temperature'].std()
        
        df['temp_zscore'] = (df['temperature'] - mean_temp) / std_temp
        df['is_anomaly'] = np.abs(df['temp_zscore']) > threshold_std
        
        anomaly_count = df['is_anomaly'].sum()
        total_count = len(df)
        
        logger.info(f"Detected {anomaly_count} anomalies out of {total_count} records "
                   f"({anomaly_count/total_count*100:.2f}%)")
        
        return df
    
    def calculate_temperature_trends(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate temperature trends over time.
        
        Args:
            df: DataFrame with temperature data
            
        Returns:
            Dictionary with trend statistics
        """
        # Use yearly averages for trend calculation
        yearly_data = self.aggregate_yearly_data(df)
        
        if len(yearly_data) < 2:
            return {"error": "Insufficient data for trend calculation"}
        
        # Linear regression for trend
        years = yearly_data['year'].values
        temps = yearly_data['avg_temperature'].values
        
        # Remove any NaN values
        valid_mask = ~np.isnan(temps)
        years = years[valid_mask]
        temps = temps[valid_mask]
        
        if len(years) < 2:
            return {"error": "Insufficient valid data for trend calculation"}
        
        # Calculate linear trend
        coeffs = np.polyfit(years, temps, 1)
        trend_per_year = coeffs[0]
        
        # Calculate trend over decades
        trend_per_decade = trend_per_year * 10
        
        # Calculate R-squared
        y_pred = np.polyval(coeffs, years)
        ss_res = np.sum((temps - y_pred) ** 2)
        ss_tot = np.sum((temps - np.mean(temps)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            "trend_per_year": round(float(trend_per_year), 6),
            "trend_per_decade": round(float(trend_per_decade), 4),
            "r_squared": round(float(r_squared), 4),
            "start_year": int(years[0]),
            "end_year": int(years[-1]),
            "years_of_data": len(years)
        }
    
    def process_file(self, input_file: str, output_dir: str) -> Dict[str, str]:
        """
        Process a single data file.
        
        Args:
            input_file: Path to input file
            output_dir: Directory to store processed data
            
        Returns:
            Dictionary with output file paths
        """
        input_path = Path(input_file)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info(f"Processing {input_path.name}...")
            
            # Read and clean data
            df = self.read_berkeley_earth_file(str(input_path))
            cleaned_df = self.clean_temperature_data(df)
            
            # Generate different aggregations
            monthly_df = self.aggregate_monthly_data(cleaned_df)
            yearly_df = self.aggregate_yearly_data(cleaned_df)
            anomaly_df = self.detect_anomalies(cleaned_df)
            
            # Calculate trends
            trends = self.calculate_temperature_trends(cleaned_df)
            
            # Save processed data
            base_name = input_path.stem
            
            monthly_output = output_path / f"{base_name}_monthly.csv"
            yearly_output = output_path / f"{base_name}_yearly.csv"
            anomaly_output = output_path / f"{base_name}_anomalies.csv"
            trends_output = output_path / f"{base_name}_trends.json"
            
            monthly_df.to_csv(monthly_output, index=False)
            yearly_df.to_csv(yearly_output, index=False)
            anomaly_df.to_csv(anomaly_output, index=False)
            
            # Save trends as JSON
            import json
            with open(trends_output, 'w') as f:
                json.dump(trends, f, indent=2)
            
            output_files = {
                "monthly": str(monthly_output),
                "yearly": str(yearly_output),
                "anomalies": str(anomaly_output),
                "trends": str(trends_output)
            }
            
            logger.info(f"Successfully processed {input_path.name}")
            return output_files
            
        except Exception as e:
            logger.error(f"Error processing {input_file}: {e}")
            raise
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Generate summary statistics for a DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with summary statistics
        """
        try:
            summary = {
                "total_records": len(df),
                "columns": list(df.columns),
                "dtypes": df.dtypes.to_dict(),
                "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024**2), 2)
            }
            
            if "temperature" in df.columns:
                temp_stats = df["temperature"].describe()
                summary["temperature_stats"] = {
                    "count": int(temp_stats["count"]),
                    "mean": round(float(temp_stats["mean"]), 3),
                    "std": round(float(temp_stats["std"]), 3),
                    "min": round(float(temp_stats["min"]), 3),
                    "max": round(float(temp_stats["max"]), 3),
                    "25%": round(float(temp_stats["25%"]), 3),
                    "50%": round(float(temp_stats["50%"]), 3),
                    "75%": round(float(temp_stats["75%"]), 3)
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return {"error": str(e)}