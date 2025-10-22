"""
Data validation module for climate data quality checks.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validates climate data for quality and completeness.
    
    Performs various checks including:
    - Missing value analysis
    - Outlier detection
    - Temporal consistency
    - Spatial coverage validation
    """
    
    def __init__(self) -> None:
        """Initialize the data validator."""
        self.validation_results: Dict[str, dict] = {}
    
    def validate_file(self, filepath: Path) -> Dict[str, any]:
        """
        Validate a single data file.
        
        Args:
            filepath: Path to the data file
            
        Returns:
            Dictionary containing validation results
        """
        if not filepath.exists():
            return {"error": f"File {filepath} does not exist"}
        
        try:
            # Determine file type and load accordingly
            if filepath.suffix == '.csv':
                df = pd.read_csv(filepath)
            elif filepath.suffix in ['.txt']:
                df = pd.read_csv(filepath, sep=r'\s+', comment='%')
            else:
                return {"error": f"Unsupported file format: {filepath.suffix}"}
            
            results = {
                "file": str(filepath),
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": df.dtypes.to_dict(),
                "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2)
            }
            
            # Perform validation checks
            results.update(self._check_missing_values(df))
            results.update(self._check_data_ranges(df)) 
            results.update(self._check_temporal_consistency(df))
            results.update(self._check_duplicates(df))
            
            self.validation_results[str(filepath)] = results
            return results
            
        except Exception as e:
            error_msg = f"Error validating {filepath}: {e}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def _check_missing_values(self, df: pd.DataFrame) -> Dict[str, any]:
        """Check for missing values in the dataset."""
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        return {
            "missing_values": {
                "total_missing": int(missing_counts.sum()),
                "missing_by_column": missing_counts.to_dict(),
                "missing_percentage_by_column": missing_percentages.round(2).to_dict(),
                "complete_rows": int(len(df) - df.isnull().any(axis=1).sum())
            }
        }
    
    def _check_data_ranges(self, df: pd.DataFrame) -> Dict[str, any]:
        """Check if data values are within expected ranges."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        range_checks = {}
        
        for col in numeric_cols:
            col_stats = {
                "min": float(df[col].min()) if not df[col].empty else None,
                "max": float(df[col].max()) if not df[col].empty else None,
                "mean": float(df[col].mean()) if not df[col].empty else None,
                "std": float(df[col].std()) if not df[col].empty else None
            }
            
            # Climate-specific range checks
            if 'temp' in col.lower():
                # Temperature typically between -100°C and 60°C
                outliers = df[(df[col] < -100) | (df[col] > 60)][col].count()
                col_stats["temperature_outliers"] = int(outliers)
            
            elif 'lat' in col.lower():
                # Latitude between -90 and 90
                outliers = df[(df[col] < -90) | (df[col] > 90)][col].count()
                col_stats["latitude_outliers"] = int(outliers)
            
            elif 'lon' in col.lower():
                # Longitude between -180 and 180
                outliers = df[(df[col] < -180) | (df[col] > 180)][col].count()
                col_stats["longitude_outliers"] = int(outliers)
            
            range_checks[col] = col_stats
        
        return {"data_ranges": range_checks}
    
    def _check_temporal_consistency(self, df: pd.DataFrame) -> Dict[str, any]:
        """Check temporal consistency of the data."""
        temporal_checks = {}
        
        # Look for date/time columns
        date_cols = []
        for col in df.columns:
            if any(term in col.lower() for term in ['date', 'time', 'year', 'month']):
                date_cols.append(col)
        
        if date_cols:
            for col in date_cols:
                if df[col].dtype == 'object':
                    try:
                        # Use robust mixed-format parsing for common date columns (e.g., 'dt')
                        try:
                            from climaxtreme.utils import parse_mixed_date_column
                            date_series = parse_mixed_date_column(df[col])
                        except Exception:
                            date_series = pd.to_datetime(df[col], errors='coerce')

                        temporal_checks[col] = {
                            "date_range": {
                                "start": str(date_series.min()) if not date_series.empty else None,
                                "end": str(date_series.max()) if not date_series.empty else None
                            },
                            "invalid_dates": int(date_series.isnull().sum()),
                            "unique_dates": int(date_series.nunique())
                        }
                    except Exception:
                        temporal_checks[col] = {"error": "Could not parse as dates"}
                else:
                    # Numeric year/month columns
                    temporal_checks[col] = {
                        "range": {
                            "min": int(df[col].min()) if not df[col].empty else None,
                            "max": int(df[col].max()) if not df[col].empty else None
                        },
                        "unique_values": int(df[col].nunique())
                    }
        
        return {"temporal_consistency": temporal_checks}
    
    def _check_duplicates(self, df: pd.DataFrame) -> Dict[str, any]:
        """Check for duplicate records."""
        total_duplicates = df.duplicated().sum()
        
        return {
            "duplicates": {
                "total_duplicate_rows": int(total_duplicates),
                "duplicate_percentage": round((total_duplicates / len(df)) * 100, 2),
                "unique_rows": int(len(df) - total_duplicates)
            }
        }
    
    def validate_directory(self, directory: Path) -> Dict[str, dict]:
        """
        Validate all data files in a directory.
        
        Args:
            directory: Path to directory containing data files
            
        Returns:
            Dictionary with validation results for each file
        """
        if not directory.exists():
            logger.error(f"Directory {directory} does not exist")
            return {}
        
        results = {}
        data_files = list(directory.glob("*.txt")) + list(directory.glob("*.csv"))
        
        logger.info(f"Validating {len(data_files)} files in {directory}")
        
        for filepath in data_files:
            results[str(filepath)] = self.validate_file(filepath)
        
        return results
    
    def generate_validation_summary(self) -> Dict[str, any]:
        """
        Generate a summary of all validation results.
        
        Returns:
            Summary dictionary with overall statistics
        """
        if not self.validation_results:
            return {"error": "No validation results available"}
        
        total_files = len(self.validation_results)
        files_with_errors = sum(
            1 for result in self.validation_results.values() 
            if "error" in result
        )
        
        total_rows = sum(
            result.get("shape", [0])[0] 
            for result in self.validation_results.values()
            if "shape" in result
        )
        
        total_missing = sum(
            result.get("missing_values", {}).get("total_missing", 0)
            for result in self.validation_results.values()
        )
        
        return {
            "summary": {
                "total_files_validated": total_files,
                "files_with_errors": files_with_errors,
                "success_rate": round((total_files - files_with_errors) / total_files * 100, 2),
                "total_data_rows": total_rows,
                "total_missing_values": total_missing,
                "validation_timestamp": pd.Timestamp.now().isoformat()
            },
            "detailed_results": self.validation_results
        }