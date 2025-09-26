"""
Heatmap analysis module for climate data visualization.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession, DataFrame as SparkDataFrame


logger = logging.getLogger(__name__)


class HeatmapAnalyzer:
    """
    Generates heatmap visualizations for climate data analysis.
    
    Supports:
    - Global temperature heatmaps
    - Seasonal temperature patterns
    - Anomaly visualization
    - Regional comparison heatmaps
    """
    
    def __init__(self) -> None:
        """Initialize the heatmap analyzer."""
        self.figure_style = {
            'figure.figsize': (12, 8),
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10
        }
        plt.rcParams.update(self.figure_style)
    
    def generate_global_heatmap(
        self, 
        data_path: str, 
        output_dir: str,
        time_period: Optional[Tuple[int, int]] = None
    ) -> str:
        """
        Generate global temperature heatmap.
        
        Args:
            data_path: Path to processed climate data
            output_dir: Directory to save heatmap
            time_period: Optional (start_year, end_year) tuple
            
        Returns:
            Path to generated heatmap file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info("Generating global temperature heatmap...")
            
            # Load data
            df = self._load_temperature_data(data_path, time_period)
            
            if df.empty:
                raise ValueError("No data available for the specified time period")
            
            # Create heatmap data
            heatmap_data = self._prepare_heatmap_data(df)
            
            # Generate heatmap
            fig, ax = plt.subplots(figsize=(14, 10))
            
            sns.heatmap(
                heatmap_data,
                cmap='RdYlBu_r',
                center=0,
                annot=False,
                fmt='.1f',
                cbar_kws={'label': 'Temperature (°C)'},
                ax=ax
            )
            
            # Customize plot
            period_str = f"{time_period[0]}-{time_period[1]}" if time_period else "All Years"
            ax.set_title(f'Global Temperature Heatmap ({period_str})', fontsize=16, pad=20)
            ax.set_xlabel('Month', fontsize=12)
            ax.set_ylabel('Year', fontsize=12)
            
            # Set month labels
            month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            ax.set_xticklabels(month_labels)
            
            plt.tight_layout()
            
            # Save heatmap
            output_file = output_path / f"global_temperature_heatmap_{period_str.replace('-', '_')}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Global heatmap saved to {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Error generating global heatmap: {e}")
            raise
    
    def generate_seasonal_heatmap(
        self, 
        data_path: str, 
        output_dir: str,
        seasons: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Generate seasonal temperature heatmaps.
        
        Args:
            data_path: Path to processed climate data
            output_dir: Directory to save heatmaps
            seasons: List of seasons to analyze (default: all)
            
        Returns:
            Dictionary mapping season names to heatmap file paths
        """
        if seasons is None:
            seasons = ['Spring', 'Summer', 'Fall', 'Winter']
        
        season_months = {
            'Spring': [3, 4, 5],
            'Summer': [6, 7, 8],
            'Fall': [9, 10, 11],
            'Winter': [12, 1, 2]
        }
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        output_files = {}
        
        try:
            # Load data
            df = self._load_temperature_data(data_path)
            
            for season in seasons:
                if season not in season_months:
                    logger.warning(f"Unknown season: {season}")
                    continue
                
                logger.info(f"Generating {season} heatmap...")
                
                # Filter data for season
                season_df = df[df['month'].isin(season_months[season])]
                
                if season_df.empty:
                    logger.warning(f"No data available for {season}")
                    continue
                
                # Create seasonal heatmap data
                heatmap_data = self._prepare_seasonal_heatmap_data(season_df, season)
                
                # Generate heatmap
                fig, ax = plt.subplots(figsize=(12, 8))
                
                sns.heatmap(
                    heatmap_data,
                    cmap='RdYlBu_r',
                    center=0,
                    annot=True,
                    fmt='.1f',
                    cbar_kws={'label': 'Temperature (°C)'},
                    ax=ax
                )
                
                ax.set_title(f'{season} Temperature Patterns', fontsize=16, pad=20)
                ax.set_xlabel('Month', fontsize=12)
                ax.set_ylabel('Year', fontsize=12)
                
                plt.tight_layout()
                
                # Save heatmap
                output_file = output_path / f"{season.lower()}_temperature_heatmap.png"
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                output_files[season] = str(output_file)
                logger.info(f"{season} heatmap saved to {output_file}")
        
        except Exception as e:
            logger.error(f"Error generating seasonal heatmaps: {e}")
            raise
        
        return output_files
    
    def generate_anomaly_heatmap(
        self, 
        data_path: str, 
        output_dir: str,
        baseline_period: Optional[Tuple[int, int]] = None
    ) -> str:
        """
        Generate temperature anomaly heatmap.
        
        Args:
            data_path: Path to processed climate data
            output_dir: Directory to save heatmap
            baseline_period: Optional (start_year, end_year) for baseline calculation
            
        Returns:
            Path to generated anomaly heatmap file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info("Generating temperature anomaly heatmap...")
            
            # Load data
            df = self._load_temperature_data(data_path)
            
            # Calculate anomalies
            anomaly_data = self._calculate_temperature_anomalies(df, baseline_period)
            
            # Generate heatmap
            fig, ax = plt.subplots(figsize=(14, 10))
            
            sns.heatmap(
                anomaly_data,
                cmap='RdBu_r',
                center=0,
                annot=False,
                fmt='.2f',
                cbar_kws={'label': 'Temperature Anomaly (°C)'},
                ax=ax
            )
            
            # Customize plot
            baseline_str = f"{baseline_period[0]}-{baseline_period[1]}" if baseline_period else "Long-term Average"
            ax.set_title(f'Temperature Anomalies (relative to {baseline_str})', fontsize=16, pad=20)
            ax.set_xlabel('Month', fontsize=12)
            ax.set_ylabel('Year', fontsize=12)
            
            # Set month labels
            month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            ax.set_xticklabels(month_labels)
            
            plt.tight_layout()
            
            # Save heatmap
            output_file = output_path / "temperature_anomaly_heatmap.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Anomaly heatmap saved to {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Error generating anomaly heatmap: {e}")
            raise
    
    def _load_temperature_data(
        self, 
        data_path: str, 
        time_period: Optional[Tuple[int, int]] = None
    ) -> pd.DataFrame:
        """Load temperature data from processed files."""
        data_path_obj = Path(data_path)
        
        # Look for monthly aggregated data
        monthly_files = list(data_path_obj.glob("*_monthly.csv")) + \
                      list(data_path_obj.glob("*_monthly.parquet"))
        
        if not monthly_files:
            # Fallback to original processed files
            csv_files = list(data_path_obj.glob("*.csv"))
            if csv_files:
                df = pd.read_csv(csv_files[0])
            else:
                raise FileNotFoundError(f"No suitable data files found in {data_path}")
        else:
            # Load monthly data
            if monthly_files[0].suffix == '.csv':
                df = pd.read_csv(monthly_files[0])
            else:
                df = pd.read_parquet(monthly_files[0])
        
        # Filter by time period if specified
        if time_period:
            start_year, end_year = time_period
            df = df[(df['year'] >= start_year) & (df['year'] <= end_year)]
        
        return df
    
    def _prepare_heatmap_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for heatmap visualization."""
        # Use average temperature if available, otherwise temperature
        temp_col = 'avg_temperature' if 'avg_temperature' in df.columns else 'temperature'
        
        # Pivot data for heatmap
        heatmap_data = df.pivot_table(
            values=temp_col,
            index='year',
            columns='month',
            aggfunc='mean'
        )
        
        return heatmap_data
    
    def _prepare_seasonal_heatmap_data(self, df: pd.DataFrame, season: str) -> pd.DataFrame:
        """Prepare seasonal data for heatmap visualization."""
        temp_col = 'avg_temperature' if 'avg_temperature' in df.columns else 'temperature'
        
        # Create decade groupings for better visualization
        df = df.copy()
        df['decade'] = (df['year'] // 10) * 10
        
        # Pivot data
        heatmap_data = df.pivot_table(
            values=temp_col,
            index='decade',
            columns='month',
            aggfunc='mean'
        )
        
        return heatmap_data
    
    def _calculate_temperature_anomalies(
        self, 
        df: pd.DataFrame,
        baseline_period: Optional[Tuple[int, int]] = None
    ) -> pd.DataFrame:
        """Calculate temperature anomalies relative to baseline period."""
        temp_col = 'avg_temperature' if 'avg_temperature' in df.columns else 'temperature'
        
        # Calculate baseline (climatology)
        if baseline_period:
            start_year, end_year = baseline_period
            baseline_df = df[(df['year'] >= start_year) & (df['year'] <= end_year)]
        else:
            baseline_df = df
        
        # Calculate monthly climatology
        climatology = baseline_df.groupby('month')[temp_col].mean()
        
        # Calculate anomalies
        df = df.copy()
        df['anomaly'] = df.apply(
            lambda row: row[temp_col] - climatology[row['month']], 
            axis=1
        )
        
        # Pivot anomalies for heatmap
        anomaly_data = df.pivot_table(
            values='anomaly',
            index='year',
            columns='month',
            aggfunc='mean'
        )
        
        return anomaly_data