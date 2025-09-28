"""
Time series analysis module for climate data.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


logger = logging.getLogger(__name__)


class TimeSeriesAnalyzer:
    """
    Performs time series analysis on climate data.
    
    Capabilities:
    - Temperature trend analysis
    - Seasonal decomposition
    - Extreme event detection
    - Long-term pattern identification
    """
    
    def __init__(self) -> None:
        """Initialize the time series analyzer."""
        self.figure_style = {
            'figure.figsize': (12, 8),
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10
        }
        plt.rcParams.update(self.figure_style)
    
    def analyze_temperature_trends(
        self, 
        data_path: str, 
        output_dir: str,
        time_period: Optional[Tuple[int, int]] = None
    ) -> Dict[str, any]:
        """
        Analyze long-term temperature trends.
        
        Args:
            data_path: Path to processed climate data
            output_dir: Directory to save analysis results
            time_period: Optional (start_year, end_year) tuple
            
        Returns:
            Dictionary with trend analysis results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info("Analyzing temperature trends...")
            
            # Load data
            df = self._load_temperature_data(data_path, time_period)
            
            if df.empty:
                raise ValueError("No data available for the specified time period")
            
            # Prepare yearly data
            yearly_data = self._prepare_yearly_data(df)
            
            # Calculate trends
            trend_results = self._calculate_trends(yearly_data)
            
            # Generate trend visualization
            trend_plot_path = self._plot_temperature_trends(yearly_data, output_path, time_period)
            
            # Add plot path to results
            trend_results['plot_path'] = trend_plot_path
            
            # Save results
            results_file = output_path / "temperature_trends.json"
            import json
            with open(results_file, 'w') as f:
                json.dump(trend_results, f, indent=2)
            
            logger.info(f"Trend analysis completed. Results saved to {results_file}")
            return trend_results
            
        except Exception as e:
            logger.error(f"Error analyzing temperature trends: {e}")
            raise
    
    def analyze_seasonal_patterns(
        self, 
        data_path: str, 
        output_dir: str
    ) -> Dict[str, any]:
        """
        Analyze seasonal temperature patterns.
        
        Args:
            data_path: Path to processed climate data
            output_dir: Directory to save analysis results
            
        Returns:
            Dictionary with seasonal analysis results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info("Analyzing seasonal patterns...")
            
            # Load data
            df = self._load_temperature_data(data_path)
            
            # Calculate seasonal statistics
            seasonal_stats = self._calculate_seasonal_statistics(df)
            
            # Generate seasonal visualization
            seasonal_plot_path = self._plot_seasonal_patterns(df, output_path)
            
            # Add plot path to results
            seasonal_stats['plot_path'] = seasonal_plot_path
            
            # Save results
            results_file = output_path / "seasonal_patterns.json"
            import json
            with open(results_file, 'w') as f:
                json.dump(seasonal_stats, f, indent=2)
            
            logger.info(f"Seasonal analysis completed. Results saved to {results_file}")
            return seasonal_stats
            
        except Exception as e:
            logger.error(f"Error analyzing seasonal patterns: {e}")
            raise
    
    def detect_extreme_events(
        self, 
        data_path: str, 
        output_dir: str,
        threshold_percentile: float = 95.0
    ) -> Dict[str, any]:
        """
        Detect extreme temperature events.
        
        Args:
            data_path: Path to processed climate data
            output_dir: Directory to save analysis results
            threshold_percentile: Percentile threshold for extreme events
            
        Returns:
            Dictionary with extreme event analysis results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info("Detecting extreme temperature events...")
            
            # Load data
            df = self._load_temperature_data(data_path)
            
            # Detect extreme events
            extreme_events = self._identify_extreme_events(df, threshold_percentile)
            
            # Generate extreme events visualization
            extreme_plot_path = self._plot_extreme_events(df, extreme_events, output_path)
            
            # Calculate extreme event statistics
            event_stats = self._calculate_extreme_event_statistics(extreme_events)
            
            results = {
                'extreme_events': extreme_events.to_dict('records'),
                'statistics': event_stats,
                'plot_path': extreme_plot_path,
                'threshold_percentile': threshold_percentile
            }
            
            # Save results
            results_file = output_path / "extreme_events.json"
            import json
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Extreme event analysis completed. Results saved to {results_file}")
            return results
            
        except Exception as e:
            logger.error(f"Error detecting extreme events: {e}")
            raise
    
    def _load_temperature_data(
        self, 
        data_path: str, 
        time_period: Optional[Tuple[int, int]] = None
    ) -> pd.DataFrame:
        """Load temperature data from processed files."""
        data_path_obj = Path(data_path)
        
        # Look for monthly aggregated data first
        monthly_files = list(data_path_obj.glob("*_monthly.csv")) + \
                      list(data_path_obj.glob("*_monthly.parquet"))
        
        if monthly_files:
            # Load monthly data
            if monthly_files[0].suffix == '.csv':
                df = pd.read_csv(monthly_files[0])
            else:
                df = pd.read_parquet(monthly_files[0])
        else:
            # Fallback to original processed files
            csv_files = list(data_path_obj.glob("*.csv"))
            if csv_files:
                df = pd.read_csv(csv_files[0])
            else:
                raise FileNotFoundError(f"No suitable data files found in {data_path}")
        
        # Filter by time period if specified
        if time_period:
            start_year, end_year = time_period
            df = df[(df['year'] >= start_year) & (df['year'] <= end_year)]
        
        return df
    
    def _prepare_yearly_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare yearly aggregated data."""
        temp_col = 'avg_temperature' if 'avg_temperature' in df.columns else 'temperature'
        
        yearly_data = (df.groupby('year')[temp_col]
                      .agg(['mean', 'min', 'max', 'std'])
                      .reset_index())
        
        yearly_data.columns = ['year', 'avg_temperature', 'min_temperature', 
                              'max_temperature', 'std_temperature']
        
        return yearly_data
    
    def _calculate_trends(self, yearly_data: pd.DataFrame) -> Dict[str, any]:
        """Calculate temperature trends using linear regression."""
        years = yearly_data['year'].values.reshape(-1, 1)
        temps = yearly_data['avg_temperature'].values
        
        # Linear trend
        linear_model = LinearRegression()
        linear_model.fit(years, temps)
        
        linear_trend = linear_model.coef_[0]
        linear_intercept = linear_model.intercept_
        linear_r2 = linear_model.score(years, temps)
        
        # Polynomial trend (degree 2)
        poly_features = PolynomialFeatures(degree=2)
        years_poly = poly_features.fit_transform(years)
        
        poly_model = LinearRegression()
        poly_model.fit(years_poly, temps)
        poly_r2 = poly_model.score(years_poly, temps)
        
        # Statistical significance test
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            yearly_data['year'], yearly_data['avg_temperature']
        )
        
        return {
            'linear_trend': {
                'slope_per_year': round(float(linear_trend), 6),
                'slope_per_decade': round(float(linear_trend * 10), 4),
                'intercept': round(float(linear_intercept), 3),
                'r_squared': round(float(linear_r2), 4),
                'p_value': round(float(p_value), 6),
                'std_error': round(float(std_err), 6),
                'significant': p_value < 0.05
            },
            'polynomial_trend': {
                'r_squared': round(float(poly_r2), 4),
                'coefficients': [round(float(c), 6) for c in poly_model.coef_]
            },
            'period': {
                'start_year': int(yearly_data['year'].min()),
                'end_year': int(yearly_data['year'].max()),
                'years_of_data': len(yearly_data)
            },
            'temperature_stats': {
                'mean': round(float(yearly_data['avg_temperature'].mean()), 3),
                'std': round(float(yearly_data['avg_temperature'].std()), 3),
                'min': round(float(yearly_data['avg_temperature'].min()), 3),
                'max': round(float(yearly_data['avg_temperature'].max()), 3)
            }
        }
    
    def _plot_temperature_trends(
        self, 
        yearly_data: pd.DataFrame, 
        output_path: Path,
        time_period: Optional[Tuple[int, int]] = None
    ) -> str:
        """Generate temperature trend visualization."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        years = yearly_data['year']
        temps = yearly_data['avg_temperature']
        
        # Main trend plot
        ax1.plot(years, temps, 'o-', color='steelblue', alpha=0.7, linewidth=2, markersize=4)
        
        # Add trend line
        z = np.polyfit(years, temps, 1)
        p = np.poly1d(z)
        ax1.plot(years, p(years), 'r--', linewidth=2, alpha=0.8)
        
        # Add confidence interval
        ax1.fill_between(years, 
                        temps - yearly_data['std_temperature'], 
                        temps + yearly_data['std_temperature'],
                        alpha=0.2, color='steelblue')
        
        period_str = f"{time_period[0]}-{time_period[1]}" if time_period else "All Years"
        ax1.set_title(f'Temperature Trends ({period_str})', fontsize=14, pad=20)
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Average Temperature (°C)')
        ax1.grid(True, alpha=0.3)
        
        # Add trend information
        trend_per_decade = z[0] * 10
        ax1.text(0.02, 0.98, f'Trend: {trend_per_decade:.3f}°C/decade', 
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Temperature range plot
        ax2.fill_between(years, yearly_data['min_temperature'], yearly_data['max_temperature'],
                        alpha=0.3, color='orange', label='Temperature Range')
        ax2.plot(years, temps, 'o-', color='red', linewidth=2, markersize=4, label='Average')
        
        ax2.set_title('Temperature Variability', fontsize=14)
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Temperature (°C)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = output_path / f"temperature_trends_{period_str.replace('-', '_')}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_file)
    
    def _calculate_seasonal_statistics(self, df: pd.DataFrame) -> Dict[str, any]:
        """Calculate seasonal temperature statistics."""
        temp_col = 'avg_temperature' if 'avg_temperature' in df.columns else 'temperature'
        
        # Define seasons
        season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                     3: 'Spring', 4: 'Spring', 5: 'Spring',
                     6: 'Summer', 7: 'Summer', 8: 'Summer',
                     9: 'Fall', 10: 'Fall', 11: 'Fall'}
        
        df = df.copy()
        df['season'] = df['month'].map(season_map)
        
        seasonal_stats = {}
        for season in ['Spring', 'Summer', 'Fall', 'Winter']:
            season_data = df[df['season'] == season][temp_col]
            
            seasonal_stats[season] = {
                'mean': round(float(season_data.mean()), 3),
                'std': round(float(season_data.std()), 3),
                'min': round(float(season_data.min()), 3),
                'max': round(float(season_data.max()), 3),
                'count': int(len(season_data))
            }
        
        return seasonal_stats
    
    def _plot_seasonal_patterns(self, df: pd.DataFrame, output_path: Path) -> str:
        """Generate seasonal pattern visualization."""
        temp_col = 'avg_temperature' if 'avg_temperature' in df.columns else 'temperature'
        
        # Calculate monthly climatology
        monthly_clim = df.groupby('month')[temp_col].agg(['mean', 'std']).reset_index()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        months = monthly_clim['month']
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Monthly climatology
        ax1.plot(months, monthly_clim['mean'], 'o-', color='darkblue', linewidth=3, markersize=6)
        ax1.fill_between(months, 
                        monthly_clim['mean'] - monthly_clim['std'],
                        monthly_clim['mean'] + monthly_clim['std'],
                        alpha=0.3, color='lightblue')
        
        ax1.set_title('Seasonal Temperature Patterns', fontsize=14, pad=20)
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Temperature (°C)')
        ax1.set_xticks(months)
        ax1.set_xticklabels(month_labels)
        ax1.grid(True, alpha=0.3)
        
        # Box plot by month
        monthly_data = [df[df['month'] == month][temp_col].values for month in range(1, 13)]
        ax2.boxplot(monthly_data, labels=month_labels)
        ax2.set_title('Monthly Temperature Distribution', fontsize=14)
        ax2.set_ylabel('Temperature (°C)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = output_path / "seasonal_patterns.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_file)
    
    def _identify_extreme_events(self, df: pd.DataFrame, threshold_percentile: float) -> pd.DataFrame:
        """Identify extreme temperature events."""
        temp_col = 'avg_temperature' if 'avg_temperature' in df.columns else 'temperature'
        
        # Calculate thresholds
        high_threshold = np.percentile(df[temp_col], threshold_percentile)
        low_threshold = np.percentile(df[temp_col], 100 - threshold_percentile)
        
        # Identify extreme events
        extreme_events = df[
            (df[temp_col] >= high_threshold) | (df[temp_col] <= low_threshold)
        ].copy()
        
        extreme_events['event_type'] = np.where(
            extreme_events[temp_col] >= high_threshold, 'Hot', 'Cold'
        )
        
        return extreme_events
    
    def _calculate_extreme_event_statistics(self, extreme_events: pd.DataFrame) -> Dict[str, any]:
        """Calculate statistics for extreme events."""
        total_events = len(extreme_events)
        hot_events = len(extreme_events[extreme_events['event_type'] == 'Hot'])
        cold_events = len(extreme_events[extreme_events['event_type'] == 'Cold'])
        
        return {
            'total_events': total_events,
            'hot_events': hot_events,
            'cold_events': cold_events,
            'hot_event_percentage': round(hot_events / total_events * 100, 2) if total_events > 0 else 0,
            'cold_event_percentage': round(cold_events / total_events * 100, 2) if total_events > 0 else 0,
            'events_by_decade': extreme_events['year'].apply(lambda x: (x // 10) * 10).value_counts().to_dict()
        }
    
    def _plot_extreme_events(self, df: pd.DataFrame, extreme_events: pd.DataFrame, output_path: Path) -> str:
        """Generate extreme events visualization."""
        temp_col = 'avg_temperature' if 'avg_temperature' in df.columns else 'temperature'
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot all data
        ax.plot(df['year'], df[temp_col], 'o-', color='lightblue', alpha=0.6, markersize=3)
        
        # Highlight extreme events
        hot_events = extreme_events[extreme_events['event_type'] == 'Hot']
        cold_events = extreme_events[extreme_events['event_type'] == 'Cold']
        
        if not hot_events.empty:
            ax.scatter(hot_events['year'], hot_events[temp_col], 
                      color='red', s=50, alpha=0.8, label='Hot Extremes')
        
        if not cold_events.empty:
            ax.scatter(cold_events['year'], cold_events[temp_col], 
                      color='blue', s=50, alpha=0.8, label='Cold Extremes')
        
        ax.set_title('Extreme Temperature Events', fontsize=14, pad=20)
        ax.set_xlabel('Year')
        ax.set_ylabel('Temperature (°C)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = output_path / "extreme_events.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_file)