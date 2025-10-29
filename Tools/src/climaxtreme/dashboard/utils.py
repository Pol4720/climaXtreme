"""
Shared utilities for the climaXtreme dashboard.
Handles HDFS connection, data loading, and common configurations.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


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
        'chi_square_tests.parquet': 'Chi-square independence tests'
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
