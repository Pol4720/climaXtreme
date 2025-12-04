"""
Shared utilities for the climaXtreme dashboard.
Handles HDFS connection, data loading, and common configurations.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import logging
import os

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DATA_DIR = Path("DATA")
DEFAULT_PROCESSED_DIR = DEFAULT_DATA_DIR / "processed"
DEFAULT_SYNTHETIC_DIR = DEFAULT_DATA_DIR / "synthetic"


class DataSource:
    """Manages data loading from HDFS or local filesystem."""
    
    def __init__(self):
        """Initialize data source configuration from session state."""
        if 'data_source_configured' not in st.session_state:
            st.session_state.data_source_configured = False
            st.session_state.use_hdfs = True
            st.session_state.hdfs_host = "climaxtreme-namenode"
            st.session_state.hdfs_port = 9000
            st.session_state.hdfs_base_path = "/data/climaxtreme/processed"
            st.session_state.local_data_path = None
            st.session_state.synthetic_data_path = str(DEFAULT_SYNTHETIC_DIR)
    
    @property
    def use_hdfs(self) -> bool:
        return st.session_state.get('use_hdfs', True)
    
    @property
    def hdfs_host(self) -> str:
        return st.session_state.get('hdfs_host', 'climaxtreme-namenode')
    
    @property
    def hdfs_port(self) -> int:
        return st.session_state.get('hdfs_port', 9000)
    
    @property
    def hdfs_base_path(self) -> str:
        return st.session_state.get('hdfs_base_path', '/data/climaxtreme/processed')
    
    @property
    def synthetic_data_path(self) -> str:
        return st.session_state.get('synthetic_data_path', str(DEFAULT_SYNTHETIC_DIR))
    
    @st.cache_data(ttl=300, show_spinner="📂 Loading data from HDFS...")
    def load_parquet(_self, filename: str) -> Optional[pd.DataFrame]:
        """
        Load a parquet file from HDFS or local filesystem.
        
        Args:
            filename: Name of the parquet file (e.g., 'monthly.parquet')
        
        Returns:
            DataFrame or None if loading fails
        """
        if _self.use_hdfs:
            return _self._load_from_hdfs(filename)
        else:
            return _self._load_from_local(filename)
    
    def load_csv(_self, filename: str) -> Optional[pd.DataFrame]:
        """
        Load a CSV file from local filesystem.
        
        Args:
            filename: Name of the CSV file
        
        Returns:
            DataFrame or None if loading fails
        """
        try:
            # Try different possible paths
            possible_paths = [
                DEFAULT_DATA_DIR / filename,
                Path(filename),
                Path(st.session_state.get('local_data_path', 'DATA')) / filename
            ]
            
            for path in possible_paths:
                if path.exists():
                    logger.info(f"Loading CSV from {path}")
                    return pd.read_csv(path)
            
            logger.warning(f"CSV file not found: {filename}")
            return None
            
        except Exception as e:
            logger.error(f"Error loading CSV {filename}: {e}")
            return None
    
    def load_synthetic_data(_self, data_type: str = 'hourly') -> Optional[pd.DataFrame]:
        """
        Load synthetic climate data.
        
        Args:
            data_type: Type of synthetic data ('hourly', 'storms', 'alerts', 'events')
        
        Returns:
            DataFrame or None if loading fails
        """
        filename_map = {
            'hourly': 'synthetic_hourly.parquet',
            'storms': 'synthetic_storms.parquet',
            'alerts': 'synthetic_alerts.parquet',
            'events': 'synthetic_events.parquet',
            'full': 'synthetic_climate_data.parquet'
        }
        
        filename = filename_map.get(data_type, f'synthetic_{data_type}.parquet')
        
        # Try loading from synthetic data path
        try:
            synthetic_path = Path(_self.synthetic_data_path)
            file_path = synthetic_path / filename
            
            if file_path.exists():
                logger.info(f"Loading synthetic data from {file_path}")
                return pd.read_parquet(file_path)
            
            # Try in subdirectory
            file_path = synthetic_path / 'synthetic' / filename
            if file_path.exists():
                logger.info(f"Loading synthetic data from {file_path}")
                return pd.read_parquet(file_path)
            
            # Try HDFS if enabled
            if _self.use_hdfs:
                return _self._load_from_hdfs(f"synthetic/{filename}")
            
            logger.warning(f"Synthetic data file not found: {filename}")
            return None
            
        except Exception as e:
            logger.error(f"Error loading synthetic data {data_type}: {e}")
            return None
    
    def _load_from_hdfs(self, filename: str) -> Optional[pd.DataFrame]:
        """Load parquet from HDFS."""
        try:
            from climaxtreme.utils.hdfs_reader import HDFSReader
            
            reader = HDFSReader(self.hdfs_host, self.hdfs_port)
            hdfs_path = f"{self.hdfs_base_path}/{filename}"
            
            logger.info(f"Loading {hdfs_path} from HDFS")
            df = reader.read_parquet(hdfs_path)
            
            if df is not None and not df.empty:
                logger.info(f"Successfully loaded {len(df)} rows from {filename}")
                return df
            else:
                logger.warning(f"Empty dataframe returned from {filename}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading {filename} from HDFS: {e}", exc_info=True)
            st.error(f"Failed to load {filename} from HDFS: {e}")
            return None
    
    def _load_from_local(self, filename: str) -> Optional[pd.DataFrame]:
        """Load parquet from local filesystem."""
        try:
            local_path = Path(st.session_state.get('local_data_path', 'DATA/processed'))
            file_path = local_path / filename
            
            if not file_path.exists():
                st.warning(f"File not found: {file_path}")
                return None
            
            logger.info(f"Loading {file_path} from local filesystem")
            df = pd.read_parquet(file_path)
            
            if df is not None and not df.empty:
                logger.info(f"Successfully loaded {len(df)} rows from {filename}")
                return df
            else:
                logger.warning(f"Empty dataframe returned from {filename}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading {filename} from local: {e}", exc_info=True)
            st.error(f"Failed to load {filename} from local filesystem: {e}")
            return None


def configure_sidebar():
    """Configure the sidebar with data source settings."""
    st.sidebar.title("⚙️ Configuration")
    
    # Data source selection
    data_source_type = st.sidebar.radio(
        "Data Source",
        ["HDFS", "Local Files"],
        key="data_source_radio",
        help="Select where to load processed parquet files from"
    )
    
    st.session_state.use_hdfs = (data_source_type == "HDFS")
    
    if st.session_state.use_hdfs:
        st.sidebar.markdown("### 🐘 HDFS Settings")
        st.session_state.hdfs_host = st.sidebar.text_input(
            "NameNode Host", 
            value=st.session_state.get('hdfs_host', 'climaxtreme-namenode'),
            key="hdfs_host_input"
        )
        st.session_state.hdfs_port = st.sidebar.number_input(
            "NameNode Port", 
            value=st.session_state.get('hdfs_port', 9000),
            min_value=1,
            max_value=65535,
            key="hdfs_port_input"
        )
        st.session_state.hdfs_base_path = st.sidebar.text_input(
            "HDFS Base Path",
            value=st.session_state.get('hdfs_base_path', '/data/climaxtreme/processed'),
            key="hdfs_base_path_input",
            help="Path in HDFS where processed parquet files are stored"
        )
        
        # Connection test
        if st.sidebar.button("🔍 Test Connection", key="test_hdfs_connection"):
            with st.spinner("Testing HDFS connection..."):
                try:
                    from climaxtreme.utils.hdfs_reader import HDFSReader
                    reader = HDFSReader(st.session_state.hdfs_host, st.session_state.hdfs_port)
                    files = reader.list_files(st.session_state.hdfs_base_path)
                    
                    if files:
                        st.sidebar.success(f"✅ Connected! Found {len(files)} files")
                    else:
                        st.sidebar.warning("⚠️ Connected but no files found")
                except Exception as e:
                    st.sidebar.error(f"❌ Connection failed: {e}")
    else:
        st.sidebar.markdown("### 📁 Local Settings")
        st.session_state.local_data_path = st.sidebar.text_input(
            "Local Data Path",
            value=st.session_state.get('local_data_path', 'DATA/processed'),
            key="local_data_path_input",
            help="Local directory containing processed parquet files"
        )
    
    st.sidebar.markdown("---")
    st.session_state.data_source_configured = True


def get_available_parquets() -> dict:
    """
    Get list of available parquet files with descriptions.
    
    Returns:
        Dictionary mapping filename to description
    """
    return {
        # Original processed data
        'monthly.parquet': 'Monthly aggregations by city',
        'yearly.parquet': 'Yearly aggregations by city',
        'anomalies.parquet': 'Temperature anomalies vs climatology',
        'climatology.parquet': 'Climatological reference values',
        'seasonal.parquet': 'Seasonal aggregations',
        'extreme_thresholds.parquet': 'Extreme temperature thresholds (P10, P90)',
        'regional.parquet': 'Regional aggregations (16 regions)',
        'continental.parquet': 'Continental aggregations (7 continents)',
        'correlation_matrix.parquet': 'Correlation matrix',
        'descriptive_stats.parquet': 'Descriptive statistics',
        'chi_square_tests.parquet': 'Chi-square independence tests',
        # Synthetic data
        'synthetic_hourly.parquet': 'Synthetic hourly weather data',
        'synthetic_storms.parquet': 'Synthetic storm tracking data',
        'synthetic_alerts.parquet': 'Synthetic weather alerts',
        'synthetic_events.parquet': 'Synthetic extreme weather events',
        'synthetic_climate_data.parquet': 'Full synthetic climate dataset'
    }


def get_synthetic_data_columns() -> Dict[str, str]:
    """
    Get list of synthetic data columns with descriptions.
    
    Returns:
        Dictionary mapping column name to description
    """
    return {
        'temperature_hourly': 'Hourly temperature (°C)',
        'rain_mm': 'Precipitation amount (mm)',
        'wind_speed_kmh': 'Wind speed (km/h)',
        'humidity_pct': 'Relative humidity (%)',
        'pressure_hpa': 'Atmospheric pressure (hPa)',
        'uv_index': 'UV radiation index',
        'cloud_cover_pct': 'Cloud coverage (%)',
        'visibility_km': 'Visibility distance (km)',
        'dew_point_c': 'Dew point temperature (°C)',
        'feels_like_c': 'Apparent temperature (°C)',
        'climate_zone': 'Climate zone classification',
        'event_type': 'Extreme event type (heatwave, cold_snap, etc.)',
        'event_intensity': 'Event intensity (0-10 scale)',
        'storm_id': 'Storm identifier',
        'storm_category': 'Storm category (Saffir-Simpson scale)',
        'alert_level': 'Alert severity (green, yellow, orange, red)',
        'alert_type': 'Alert type (heat, storm, flood, etc.)'
    }


def format_dataframe_display(df: pd.DataFrame, max_rows: int = 100) -> None:
    """
    Display a dataframe with nice formatting.
    
    Args:
        df: DataFrame to display
        max_rows: Maximum number of rows to show
    """
    if df is None or df.empty:
        st.warning("No data to display")
        return
    
    # Show basic info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        memory_mb = df.memory_usage(deep=True).sum() / (1024**2)
        st.metric("Memory Usage", f"{memory_mb:.2f} MB")
    
    # Display sample data
    st.markdown("#### Sample Data")
    display_df = df.head(max_rows) if len(df) > max_rows else df
    st.dataframe(display_df, use_container_width=True, height=400)


def show_data_info(df: pd.DataFrame, title: str = "Dataset Information"):
    """
    Show comprehensive information about a dataset.
    
    Args:
        df: DataFrame to analyze
        title: Title for the info section
    """
    with st.expander(f"ℹ️ {title}", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Shape:**")
            st.write(f"- Rows: {len(df):,}")
            st.write(f"- Columns: {len(df.columns)}")
            
            st.markdown("**Data Types:**")
            dtype_counts = df.dtypes.value_counts()
            for dtype, count in dtype_counts.items():
                st.write(f"- {dtype}: {count} columns")
        
        with col2:
            st.markdown("**Missing Values:**")
            missing = df.isnull().sum()
            if missing.sum() == 0:
                st.write("✅ No missing values")
            else:
                for col in missing[missing > 0].index:
                    pct = (missing[col] / len(df)) * 100
                    st.write(f"- {col}: {missing[col]:,} ({pct:.2f}%)")
            
            st.markdown("**Memory:**")
            memory_mb = df.memory_usage(deep=True).sum() / (1024**2)
            st.write(f"{memory_mb:.2f} MB")


def create_metric_card(label: str, value: str, delta: Optional[str] = None):
    """
    Create a styled metric card.
    
    Args:
        label: Metric label
        value: Metric value
        delta: Optional delta value
    """
    st.metric(label=label, value=value, delta=delta)


def generate_demo_synthetic_data(n_records: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generate demo synthetic data for visualization when real data is not available.
    
    Args:
        n_records: Number of records to generate
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with synthetic weather data
    """
    np.random.seed(seed)
    
    dates = pd.date_range('2024-01-01', periods=n_records, freq='h')
    
    # Generate realistic patterns
    hours = np.arange(n_records)
    seasonal = 15 * np.sin(2 * np.pi * hours / (24 * 365))
    diurnal = 8 * np.sin(2 * np.pi * (hours - 6) / 24)
    noise = np.random.normal(0, 2, n_records)
    
    temp = 20 + seasonal + diurnal + noise
    
    # Generate precipitation with Markov chain
    is_wet = np.random.random(n_records) < 0.2
    rain = np.zeros(n_records)
    rain[is_wet] = np.random.exponential(5, is_wet.sum())
    
    # Generate wind speed (Weibull distribution)
    wind = np.random.weibull(2, n_records) * 15
    
    # Generate humidity (inverse correlation with temperature)
    humidity = np.clip(70 - (temp - 20) + np.random.normal(0, 10, n_records), 0, 100)
    
    # Generate pressure
    pressure = 1013.25 + np.random.normal(0, 10, n_records)
    
    # Climate zones
    climate_zones = ['Tropical', 'Subtropical', 'Temperate', 'Continental', 'Polar']
    zones = np.random.choice(climate_zones, n_records, p=[0.2, 0.25, 0.3, 0.2, 0.05])
    
    # Alerts (rare events)
    alert_types = ['None', 'Heat', 'Cold', 'Storm', 'Flood']
    alert_probs = [0.9, 0.04, 0.03, 0.02, 0.01]
    alerts = np.random.choice(alert_types, n_records, p=alert_probs)
    
    alert_level_map = {
        'None': 'green',
        'Heat': np.random.choice(['yellow', 'orange', 'red']),
        'Cold': np.random.choice(['yellow', 'orange', 'red']),
        'Storm': np.random.choice(['orange', 'red']),
        'Flood': np.random.choice(['orange', 'red'])
    }
    alert_levels = [alert_level_map.get(a, 'green') if a != 'None' else 'green' for a in alerts]
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'year': dates.year,
        'month': dates.month,
        'day': dates.day,
        'hour': dates.hour,
        'day_of_week': dates.dayofweek + 1,
        'temperature_hourly': temp,
        'rain_mm': rain,
        'wind_speed_kmh': wind,
        'humidity_pct': humidity,
        'pressure_hpa': pressure,
        'climate_zone': zones,
        'alert_type': alerts,
        'alert_level': alert_levels,
        'City': np.random.choice(['City_A', 'City_B', 'City_C', 'City_D', 'City_E'], n_records),
        'Country': np.random.choice(['Country_1', 'Country_2', 'Country_3'], n_records),
        'Latitude': np.random.uniform(-60, 60, n_records),
        'Longitude': np.random.uniform(-180, 180, n_records)
    })
    
    return df


def generate_demo_storm_data(n_storms: int = 20, seed: int = 42) -> pd.DataFrame:
    """
    Generate demo storm tracking data.
    
    Args:
        n_storms: Number of storms to generate
        seed: Random seed
    
    Returns:
        DataFrame with storm data
    """
    np.random.seed(seed)
    
    records = []
    
    for storm_id in range(1, n_storms + 1):
        # Storm parameters
        start_lat = np.random.uniform(-30, 30)
        start_lon = np.random.uniform(-80, 80)
        duration_hours = np.random.randint(24, 168)
        max_intensity = np.random.uniform(50, 200)
        
        # Generate trajectory
        for hour in range(duration_hours):
            # Move storm
            lat = start_lat + hour * np.random.uniform(-0.2, 0.3)
            lon = start_lon + hour * np.random.uniform(0.1, 0.4)
            
            # Intensity curve (build-up, peak, decay)
            t_norm = hour / duration_hours
            if t_norm < 0.3:
                intensity = max_intensity * (t_norm / 0.3)
            elif t_norm < 0.5:
                intensity = max_intensity
            else:
                intensity = max_intensity * (1 - (t_norm - 0.5) / 0.5)
            
            # Saffir-Simpson scale approximation
            if intensity < 74:
                category = 'TD'
            elif intensity < 96:
                category = 'TS'
            elif intensity < 111:
                category = '1'
            elif intensity < 130:
                category = '2'
            elif intensity < 157:
                category = '3'
            elif intensity < 178:
                category = '4'
            else:
                category = '5'
            
            records.append({
                'storm_id': f'STORM-{storm_id:03d}',
                'timestamp': pd.Timestamp('2024-01-01') + pd.Timedelta(hours=storm_id * 200 + hour),
                'latitude': lat,
                'longitude': lon,
                'wind_speed_kmh': intensity,
                'category': category,
                'pressure_hpa': 1013 - intensity * 0.5,
                'storm_name': f'Storm_{storm_id}'
            })
    
    return pd.DataFrame(records)


def generate_demo_alert_data(n_alerts: int = 100, seed: int = 42) -> pd.DataFrame:
    """
    Generate demo weather alert data.
    
    Args:
        n_alerts: Number of alerts to generate
        seed: Random seed
    
    Returns:
        DataFrame with alert data
    """
    np.random.seed(seed)
    
    alert_types = ['Heat', 'Cold', 'Storm', 'Flood', 'Wind', 'Fire']
    alert_levels = ['yellow', 'orange', 'red']
    
    records = []
    for i in range(n_alerts):
        alert_type = np.random.choice(alert_types)
        level = np.random.choice(alert_levels, p=[0.5, 0.35, 0.15])
        
        records.append({
            'alert_id': f'ALERT-{i+1:04d}',
            'timestamp': pd.Timestamp('2024-01-01') + pd.Timedelta(hours=np.random.randint(0, 8760)),
            'alert_type': alert_type,
            'alert_level': level,
            'latitude': np.random.uniform(-60, 60),
            'longitude': np.random.uniform(-180, 180),
            'City': f'City_{np.random.randint(1, 20)}',
            'Country': f'Country_{np.random.randint(1, 10)}',
            'description': f'{level.title()} {alert_type.lower()} warning',
            'duration_hours': np.random.randint(6, 72),
            'affected_population': np.random.randint(10000, 1000000)
        })
    
    return pd.DataFrame(records)
