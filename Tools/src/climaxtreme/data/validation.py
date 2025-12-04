"""
Data validation module for climate data quality checks.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


# Validation ranges for synthetic climate data columns
SYNTHETIC_DATA_RANGES = {
    # Weather variables
    'temperature_hourly': {'min': -80, 'max': 60, 'unit': '°C'},
    'rain_mm': {'min': 0, 'max': 500, 'unit': 'mm'},
    'wind_speed_kmh': {'min': 0, 'max': 400, 'unit': 'km/h'},
    'humidity_pct': {'min': 0, 'max': 100, 'unit': '%'},
    'pressure_hpa': {'min': 870, 'max': 1084, 'unit': 'hPa'},
    'uv_index': {'min': 0, 'max': 15, 'unit': 'index'},
    'cloud_cover_pct': {'min': 0, 'max': 100, 'unit': '%'},
    'visibility_km': {'min': 0, 'max': 100, 'unit': 'km'},
    'dew_point_c': {'min': -80, 'max': 40, 'unit': '°C'},
    'feels_like_c': {'min': -90, 'max': 70, 'unit': '°C'},
    
    # Event and storm variables
    'event_intensity': {'min': 0, 'max': 10, 'unit': 'scale'},
    'storm_category': {'min': 0, 'max': 5, 'unit': 'Saffir-Simpson'},
    
    # Geographic
    'Latitude': {'min': -90, 'max': 90, 'unit': 'degrees'},
    'Longitude': {'min': -180, 'max': 180, 'unit': 'degrees'}
}

# Valid categorical values for synthetic data
SYNTHETIC_CATEGORICAL_VALUES = {
    'climate_zone': ['Tropical', 'Subtropical', 'Temperate', 'Continental', 'Polar', 'Arid'],
    'event_type': ['none', 'heatwave', 'cold_snap', 'drought', 'extreme_precipitation', 'hurricane', 'tornado'],
    'alert_level': ['green', 'yellow', 'orange', 'red'],
    'alert_type': ['none', 'heat', 'cold', 'storm', 'flood', 'wind', 'fire', 'air_quality'],
    'season': ['Spring', 'Summer', 'Fall', 'Winter', 'Dry', 'Wet']
}


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
            
            # Synthetic data specific checks
            elif col in SYNTHETIC_DATA_RANGES:
                limits = SYNTHETIC_DATA_RANGES[col]
                outliers = df[(df[col] < limits['min']) | (df[col] > limits['max'])][col].count()
                col_stats["outliers"] = int(outliers)
                col_stats["expected_range"] = f"{limits['min']} - {limits['max']} {limits['unit']}"
            
            range_checks[col] = col_stats
        
        # Check categorical columns for synthetic data
        categorical_checks = {}
        for col in df.columns:
            if col in SYNTHETIC_CATEGORICAL_VALUES:
                valid_values = SYNTHETIC_CATEGORICAL_VALUES[col]
                actual_values = df[col].dropna().unique().tolist()
                invalid_values = [v for v in actual_values if v not in valid_values]
                
                categorical_checks[col] = {
                    "valid_values": valid_values,
                    "actual_values": actual_values[:20],  # Limit for display
                    "invalid_values": invalid_values[:10],
                    "n_invalid": len(invalid_values)
                }
        
        return {"data_ranges": range_checks, "categorical_validation": categorical_checks}
    
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


class SyntheticDataValidator(DataValidator):
    """
    Specialized validator for synthetic climate data.
    
    Extends DataValidator with additional checks specific to 
    synthetically generated weather data including storms, alerts,
    and extreme events.
    """
    
    def __init__(self) -> None:
        """Initialize the synthetic data validator."""
        super().__init__()
    
    def validate_synthetic_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate that DataFrame contains expected synthetic data columns.
        
        Args:
            df: DataFrame to validate
        
        Returns:
            Dictionary with schema validation results
        """
        expected_columns = {
            'core': ['timestamp', 'year', 'month', 'day', 'hour'],
            'location': ['City', 'Country', 'Latitude', 'Longitude'],
            'weather': ['temperature_hourly', 'rain_mm', 'wind_speed_kmh', 'humidity_pct'],
            'optional': ['pressure_hpa', 'uv_index', 'cloud_cover_pct', 'visibility_km'],
            'events': ['event_type', 'event_intensity'],
            'alerts': ['alert_level', 'alert_type'],
            'storms': ['storm_id', 'storm_category']
        }
        
        results = {
            'schema_valid': True,
            'columns_present': {},
            'columns_missing': {},
            'total_columns': len(df.columns)
        }
        
        for category, cols in expected_columns.items():
            present = [c for c in cols if c in df.columns]
            missing = [c for c in cols if c not in df.columns]
            
            results['columns_present'][category] = present
            results['columns_missing'][category] = missing
            
            # Core and location are required
            if category in ['core', 'location'] and missing:
                results['schema_valid'] = False
        
        return results
    
    def validate_temporal_coverage(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate temporal coverage and resolution of synthetic data.
        
        Args:
            df: DataFrame with timestamp column
        
        Returns:
            Dictionary with temporal coverage results
        """
        results = {}
        
        if 'timestamp' not in df.columns:
            return {'error': 'No timestamp column found'}
        
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Date range
        results['date_range'] = {
            'start': str(df['timestamp'].min()),
            'end': str(df['timestamp'].max()),
            'span_days': (df['timestamp'].max() - df['timestamp'].min()).days
        }
        
        # Resolution check
        time_diffs = df['timestamp'].diff().dropna()
        if len(time_diffs) > 0:
            median_diff = time_diffs.median()
            results['resolution'] = {
                'median_interval': str(median_diff),
                'is_hourly': median_diff <= pd.Timedelta(hours=1.5),
                'min_interval': str(time_diffs.min()),
                'max_interval': str(time_diffs.max())
            }
        
        # Coverage by year/month
        results['coverage'] = {
            'years': sorted(df['year'].unique().tolist()) if 'year' in df.columns else [],
            'records_per_year': df.groupby('year').size().to_dict() if 'year' in df.columns else {}
        }
        
        return results
    
    def validate_physical_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate physical consistency of weather variables.
        
        Args:
            df: DataFrame with weather columns
        
        Returns:
            Dictionary with consistency check results
        """
        results = {
            'checks_passed': [],
            'checks_failed': [],
            'warnings': []
        }
        
        # Check: Temperature and feels_like should be correlated
        if 'temperature_hourly' in df.columns and 'feels_like_c' in df.columns:
            corr = df['temperature_hourly'].corr(df['feels_like_c'])
            if corr > 0.8:
                results['checks_passed'].append('Temperature-FeelsLike correlation')
            else:
                results['warnings'].append(f'Low Temperature-FeelsLike correlation: {corr:.2f}')
        
        # Check: Humidity should not exceed 100%
        if 'humidity_pct' in df.columns:
            invalid_humidity = (df['humidity_pct'] > 100) | (df['humidity_pct'] < 0)
            if invalid_humidity.any():
                results['checks_failed'].append(f'Invalid humidity values: {invalid_humidity.sum()}')
            else:
                results['checks_passed'].append('Humidity range valid')
        
        # Check: Rain should be positive
        if 'rain_mm' in df.columns:
            negative_rain = df['rain_mm'] < 0
            if negative_rain.any():
                results['checks_failed'].append(f'Negative rain values: {negative_rain.sum()}')
            else:
                results['checks_passed'].append('Rain values valid')
        
        # Check: Wind speed should be positive
        if 'wind_speed_kmh' in df.columns:
            negative_wind = df['wind_speed_kmh'] < 0
            if negative_wind.any():
                results['checks_failed'].append(f'Negative wind speed values: {negative_wind.sum()}')
            else:
                results['checks_passed'].append('Wind speed values valid')
        
        # Check: Pressure should be within realistic range
        if 'pressure_hpa' in df.columns:
            invalid_pressure = (df['pressure_hpa'] < 870) | (df['pressure_hpa'] > 1084)
            if invalid_pressure.any():
                results['warnings'].append(f'Extreme pressure values: {invalid_pressure.sum()}')
            else:
                results['checks_passed'].append('Pressure range valid')
        
        # Check: Event intensity should be 0-10
        if 'event_intensity' in df.columns:
            invalid_intensity = (df['event_intensity'] < 0) | (df['event_intensity'] > 10)
            if invalid_intensity.any():
                results['checks_failed'].append(f'Invalid event intensity: {invalid_intensity.sum()}')
            else:
                results['checks_passed'].append('Event intensity valid')
        
        results['is_consistent'] = len(results['checks_failed']) == 0
        
        return results
    
    def validate_storm_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate storm tracking data.
        
        Args:
            df: DataFrame with storm columns
        
        Returns:
            Dictionary with storm validation results
        """
        if 'storm_id' not in df.columns:
            return {'has_storms': False}
        
        storm_df = df[df['storm_id'].notna()]
        
        results = {
            'has_storms': len(storm_df) > 0,
            'n_storms': storm_df['storm_id'].nunique(),
            'n_storm_records': len(storm_df),
            'storm_ids': storm_df['storm_id'].unique().tolist()[:10]  # First 10
        }
        
        if 'storm_category' in storm_df.columns:
            results['category_distribution'] = storm_df['storm_category'].value_counts().to_dict()
        
        # Check storm trajectory continuity
        if 'Latitude' in storm_df.columns and 'Longitude' in storm_df.columns:
            trajectory_valid = True
            for storm_id in storm_df['storm_id'].unique()[:5]:  # Check first 5 storms
                storm_track = storm_df[storm_df['storm_id'] == storm_id].sort_values('timestamp')
                if len(storm_track) > 1:
                    lat_changes = storm_track['Latitude'].diff().abs()
                    lon_changes = storm_track['Longitude'].diff().abs()
                    # Check for unrealistic jumps (>10 degrees in one step)
                    if (lat_changes > 10).any() or (lon_changes > 10).any():
                        trajectory_valid = False
                        break
            
            results['trajectories_valid'] = trajectory_valid
        
        return results
    
    def validate_alert_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate weather alert data.
        
        Args:
            df: DataFrame with alert columns
        
        Returns:
            Dictionary with alert validation results
        """
        if 'alert_level' not in df.columns:
            return {'has_alerts': False}
        
        alert_df = df[df['alert_level'] != 'green'] if 'green' in df['alert_level'].values else df
        
        results = {
            'has_alerts': len(alert_df) > 0,
            'n_alerts': len(alert_df),
            'level_distribution': df['alert_level'].value_counts().to_dict()
        }
        
        if 'alert_type' in df.columns:
            results['type_distribution'] = df['alert_type'].value_counts().to_dict()
        
        # Check alert level hierarchy (red should be rarer than orange, etc.)
        if 'alert_level' in df.columns:
            level_counts = df['alert_level'].value_counts()
            hierarchy_valid = True
            
            if 'green' in level_counts and 'yellow' in level_counts:
                if level_counts.get('yellow', 0) > level_counts.get('green', 0):
                    hierarchy_valid = False
            
            if 'yellow' in level_counts and 'red' in level_counts:
                if level_counts.get('red', 0) > level_counts.get('yellow', 0):
                    hierarchy_valid = False
            
            results['hierarchy_valid'] = hierarchy_valid
        
        return results
    
    def full_validation_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run all validation checks and generate comprehensive report.
        
        Args:
            df: DataFrame to validate
        
        Returns:
            Dictionary with complete validation report
        """
        report = {
            'validation_timestamp': pd.Timestamp.now().isoformat(),
            'data_shape': {'rows': len(df), 'columns': len(df.columns)},
            'column_list': df.columns.tolist()
        }
        
        # Run all validation checks
        report['schema'] = self.validate_synthetic_schema(df)
        report['temporal'] = self.validate_temporal_coverage(df)
        report['physical'] = self.validate_physical_consistency(df)
        report['storms'] = self.validate_storm_data(df)
        report['alerts'] = self.validate_alert_data(df)
        report['missing_values'] = self._check_missing_values(df)
        report['data_ranges'] = self._check_data_ranges(df)
        
        # Overall validation status
        report['overall_valid'] = (
            report['schema']['schema_valid'] and
            report['physical']['is_consistent']
        )
        
        return report