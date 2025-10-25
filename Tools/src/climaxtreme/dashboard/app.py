"""
Main Streamlit dashboard application for climaXtreme.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# Prefer absolute imports so the app works when launched directly via Streamlit
# Fallback: if the package isn't on sys.path (e.g., running the file directly),
# add the repository's "src" directory to sys.path and retry.
try:
    from climaxtreme.data import DataValidator
    from climaxtreme.analysis import HeatmapAnalyzer, TimeSeriesAnalyzer
except Exception:  # ImportError or other issues due to missing sys.path
    import sys
    from pathlib import Path as _Path

    _src_dir = _Path(__file__).resolve().parents[2]  # .../Tools/src
    if str(_src_dir) not in sys.path:
        sys.path.insert(0, str(_src_dir))
    from climaxtreme.data import DataValidator
    from climaxtreme.analysis import HeatmapAnalyzer, TimeSeriesAnalyzer


logger = logging.getLogger(__name__)


def run_dashboard(host: str = "localhost", port: int = 8501, data_dir: str = None) -> None:
    """
    Launch the Streamlit dashboard.
    
    Args:
        host: Host to run the dashboard on
        port: Port to run the dashboard on
        data_dir: (Deprecated) Directory containing climate data. 
                  Now the dashboard supports both HDFS and Local modes via UI.
    """
    import subprocess
    import sys
    from pathlib import Path
    
    # Get the path to this app.py file
    app_path = Path(__file__).resolve()
    
    # Build the streamlit run command
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.address", host,
        "--server.port", str(port),
        "--server.headless", "true"
    ]
    
    # Set data_dir as environment variable if provided (for backward compatibility)
    import os
    if data_dir:
        os.environ["CLIMAXTREME_DATA_DIR"] = data_dir
    
    print(f"Launching Streamlit dashboard at http://{host}:{port}")
    print(f"Dashboard supports both HDFS and Local Files modes")
    print(f"Select your data source from the sidebar")
    print("Press Ctrl+C to stop the server")
    
    # Run streamlit
    subprocess.run(cmd)


def main():
    """Main dashboard application."""
    
    # Set up Streamlit page config (MUST be first Streamlit command)
    st.set_page_config(
        page_title="climaXtreme Dashboard",
        page_icon="🌡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Page title and description
    st.title("🌡️ climaXtreme Dashboard")
    st.markdown("Interactive climate data analysis and visualization")
    
    # Sidebar configuration
    st.sidebar.title("Configuration")
    
    # Data source selection
    import os
    data_source = st.sidebar.radio(
        "Data Source",
        ["HDFS (Recommended)", "Local Files"],
        help="HDFS reads directly from Hadoop (Big Data best practice). Local is fallback only."
    )
    
    if data_source == "HDFS (Recommended)":
        # HDFS configuration
        st.sidebar.markdown("### HDFS Settings")
        hdfs_host = st.sidebar.text_input("NameNode Host", value="climaxtreme-namenode")
        hdfs_port = st.sidebar.number_input("NameNode Port", value=9000, min_value=1, max_value=65535)
        hdfs_base_path = st.sidebar.text_input(
            "HDFS Base Path", 
            value="/data/climaxtreme/processed",
            help="Base path in HDFS where processed files are stored"
        )
        use_hdfs = True
    else:
        # Local file configuration
        use_hdfs = False
        hdfs_host = None
        hdfs_port = None
        hdfs_base_path = None
    
    # Compute sensible default to repo-root/DATA
    _default_data_dir = os.environ.get("CLIMAXTREME_DATA_DIR")
    if not _default_data_dir:
        try:
            from climaxtreme.utils.config import default_dataset_dir as _default_dataset_dir
            _default_data_dir = str(_default_dataset_dir())
        except Exception:
            _default_data_dir = "DATA"
    
    # Data directory selection
    data_dir = st.sidebar.text_input(
        "Data Directory", 
        value=st.session_state.get('data_dir', _default_data_dir),
        help="Directory containing climate data (defaults to repo-root/DATA)"
    )
    
    # Check if data directory exists
    data_path = Path(data_dir)
    if not data_path.exists():
        st.error(f"Data directory '{data_dir}' does not exist!")
        st.stop()
    
    # Performance and safety controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("Memory & Performance")
    bigdata_mode = st.sidebar.checkbox(
        "Big data mode (memory-safe)", value=True,
        help="Enable chunked loading, downsampling, and vectorized ops to prevent RAM spikes."
    )
    max_rows_to_load = st.sidebar.number_input(
        "Max rows to load (CSV)", min_value=100_000, max_value=10_000_000,
        value=8_000_000, step=100_000,
        help="When reading CSV, load up to this many rows using chunking. Parquet will load fully then sample if needed."
    )
    max_points_to_plot = st.sidebar.number_input(
        "Max points to plot", min_value=50_000, max_value=1_000_000,
        value=500_000, step=50_000,
        help="Scatter/box plots will be downsampled to this limit to avoid browser/socket overload."
    )

    # Load available data files
    available_files = load_available_files(
        data_path=data_path if not use_hdfs else None,
        use_hdfs=use_hdfs,
        hdfs_host=hdfs_host if use_hdfs else None,
        hdfs_port=hdfs_port if use_hdfs else None,
        hdfs_base_path=hdfs_base_path if use_hdfs else None
    )
    
    if not available_files:
        st.warning("No climate data files found in the specified directory!")
        st.info("Please run data ingestion and preprocessing first.")
        st.stop()
    
    # File selection
    selected_file = st.sidebar.selectbox(
        "Select Data File",
        options=available_files,
        help="Choose a processed climate data file"
    )
    
    # Load selected data
    if use_hdfs:
        # Build HDFS path
        hdfs_file_path = f"{hdfs_base_path}/{selected_file}"
        df, meta = load_data_file(
            file_path=None,
            bigdata_mode=bigdata_mode,
            max_rows=max_rows_to_load,
            sample_seed=42,
            use_hdfs=True,
            hdfs_host=hdfs_host,
            hdfs_port=hdfs_port,
            hdfs_file_path=hdfs_file_path,
        )
    else:
        df, meta = load_data_file(
            data_path / selected_file,
            bigdata_mode=bigdata_mode,
            max_rows=max_rows_to_load,
            sample_seed=42,
        )
    
    if df is None or df.empty:
        st.error("Failed to load the selected data file!")
        st.stop()
    
    # Main dashboard content
    create_dashboard_content(df, selected_file, max_points_to_plot=max_points_to_plot)


def load_available_files(data_path: Path = None, use_hdfs: bool = False, 
                        hdfs_host: str = None, hdfs_port: int = None, 
                        hdfs_base_path: str = None) -> List[str]:
    """Load list of available data files from local or HDFS."""
    
    if use_hdfs and hdfs_host and hdfs_base_path:
        # Load from HDFS
        try:
            from climaxtreme.utils.hdfs_reader import HDFSReader
            reader = HDFSReader(hdfs_host, hdfs_port)
            
            # List all parquet directories in HDFS
            files = reader.list_files(hdfs_base_path)
            # Filter for parquet directories
            parquet_files = [f.split('/')[-1] for f in files if '.parquet' in f]
            # Get unique directory names
            unique_files = sorted(list(set(parquet_files)))
            
            return unique_files if unique_files else []
            
        except Exception as e:
            st.error(f"Could not connect to HDFS: {e}")
            st.info("Falling back to local files...")
            use_hdfs = False
    
    # Load from local filesystem
    if data_path:
        file_patterns = ['*.parquet']
        available_files = []
        
        for pattern in file_patterns:
            available_files.extend([f.name for f in data_path.glob(pattern)])
        
        return sorted(available_files)
    
    return []


@st.cache_data(show_spinner=True)
def _read_csv_chunked(
    file_path: str,
    max_rows: int,
) -> pd.DataFrame:
    """Read up to `max_rows` from a CSV using chunking to limit memory."""
    chunks: List[pd.DataFrame] = []
    total = 0
    for chunk in pd.read_csv(file_path, chunksize=100_000, low_memory=True):
        chunks.append(chunk)
        total += len(chunk)
        if total >= max_rows:
            break
    if not chunks:
        return pd.DataFrame()
    df = pd.concat(chunks, ignore_index=True)
    if len(df) > max_rows:
        df = df.iloc[:max_rows].reset_index(drop=True)
    return df


@st.cache_data(show_spinner=True)
def _read_parquet_sampled(
    file_path: str,
    max_rows: int,
    sample_seed: int,
) -> pd.DataFrame:
    """Read Parquet and downsample if it exceeds `max_rows`."""
    df = pd.read_parquet(file_path)
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=sample_seed).reset_index(drop=True)
    return df


@st.cache_data(show_spinner=True)
def load_data_file(
    file_path: Path = None,
    *,
    bigdata_mode: bool,
    max_rows: int,
    sample_seed: int = 42,
    use_hdfs: bool = False,
    hdfs_host: str = None,
    hdfs_port: int = None,
    hdfs_file_path: str = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Load a data file into a pandas DataFrame with memory-safe options.
    
    Supports reading from HDFS or local filesystem.

    Returns: (df, meta) where meta includes {"bigdata_mode", "loaded_rows", "source"}.
    """
    meta: Dict[str, object] = {"bigdata_mode": bigdata_mode}
    
    # HDFS Mode
    if use_hdfs and hdfs_host and hdfs_file_path:
        try:
            from climaxtreme.utils.hdfs_reader import HDFSReader
            reader = HDFSReader(hdfs_host, hdfs_port)
            
            # Read parquet from HDFS
            df = reader.read_parquet(hdfs_file_path)
            meta["source"] = hdfs_file_path.split('/')[-1]
            meta["read_mode"] = "hdfs_parquet"
            
            # Apply sampling if in bigdata mode
            if bigdata_mode and len(df) > max_rows:
                df = df.sample(n=max_rows, random_state=sample_seed).reset_index(drop=True)
            
            meta["loaded_rows"] = len(df)
            
        except Exception as e:
            st.error(f"Error reading from HDFS: {e}")
            st.info("Please check HDFS connection settings or use local files.")
            return pd.DataFrame(), meta
    
    # Local Mode
    elif file_path and file_path.exists():
        meta["source"] = file_path.name
        try:
            if file_path.suffix == '.csv':
                if bigdata_mode:
                    df = _read_csv_chunked(str(file_path), max_rows=max_rows)
                    meta["loaded_rows"] = len(df)
                    meta["read_mode"] = "csv_chunked"
                else:
                    df = pd.read_csv(file_path)
                    meta["loaded_rows"] = len(df)
                    meta["read_mode"] = "csv_full"
            elif file_path.suffix == '.parquet':
                if bigdata_mode:
                    df = _read_parquet_sampled(str(file_path), max_rows=max_rows, sample_seed=sample_seed)
                    meta["loaded_rows"] = len(df)
                    meta["read_mode"] = "parquet_sampled"
                else:
                    df = pd.read_parquet(file_path)
                    meta["loaded_rows"] = len(df)
                    meta["read_mode"] = "parquet_full"
            else:
                st.error(f"Unsupported file format: {file_path.suffix}")
                return pd.DataFrame(), meta
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return pd.DataFrame(), meta
    else:
        st.error("No valid file path or HDFS configuration provided")
        return pd.DataFrame(), meta

    # Downcast common numeric columns to save memory
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    # Normalize mixed-format date column commonly named 'dt'
    try:
        from climaxtreme.utils import add_date_parts
        if 'dt' in df.columns:
            # Add in-place to avoid doubling memory
            df = add_date_parts(df, date_col='dt', drop_invalid=False, in_place=True)
    except Exception:
        # Non-fatal; continue without normalization
        pass

    return df, meta


def create_dashboard_content(df: pd.DataFrame, filename: str, *, max_points_to_plot: int):
    """Create the main dashboard content."""
    
    # Data overview
    st.header("📊 Data Overview")
    create_data_overview(df, filename)
    
    # Navigation tabs (now 7 tabs: 6 + 1 for EDA)
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🌡️ Temperature Trends", 
        "🗺️ Heatmaps", 
        "📈 Seasonal Analysis", 
        "⚡ Extreme Events",
        "🌍 Regional Analysis",
        "🌐 Continental Analysis",
        "📊 Exploratory Analysis (EDA)"
    ])
    
    with tab1:
        create_temperature_trends_tab(df, max_points_to_plot=max_points_to_plot)
    
    with tab2:
        create_heatmaps_tab(df)
    
    with tab3:
        create_seasonal_analysis_tab(df, max_points_to_plot=max_points_to_plot)
    
    with tab4:
        create_extreme_events_tab(df, max_points_to_plot=max_points_to_plot)
    
    with tab5:
        create_regional_analysis_tab(df, max_points_to_plot=max_points_to_plot)
    
    with tab6:
        create_continental_analysis_tab(df, max_points_to_plot=max_points_to_plot)
    
    with tab7:
        create_eda_tab(df, max_points_to_plot=max_points_to_plot)


def create_data_overview(df: pd.DataFrame, filename: str):
    """Create data overview section."""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    
    with col2:
        if 'year' in df.columns:
            year_range = f"{df['year'].min()}-{df['year'].max()}"
            st.metric("Year Range", year_range)
        else:
            st.metric("Columns", len(df.columns))
    
    with col3:
        temp_col = get_temperature_column(df)
        if temp_col:
            avg_temp = df[temp_col].mean()
            st.metric("Avg Temperature", f"{avg_temp:.2f}°C")
        else:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    with col4:
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("Missing Data", f"{missing_pct:.1f}%")
    
    # Data quality indicator
    st.subheader("Data Quality")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Missing data by column
        missing_by_col = df.isnull().sum()
        if missing_by_col.sum() > 0:
            fig = px.bar(
                x=missing_by_col.index, 
                y=missing_by_col.values,
                title="Missing Values by Column",
                labels={'x': 'Columns', 'y': 'Missing Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No missing values detected!")
    
    with col2:
        # Data info
        st.subheader("Dataset Info")
        st.write(f"**File:** {filename}")
        st.write(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")
        st.write(f"**Columns:** {', '.join(df.columns[:3])}{'...' if len(df.columns) > 3 else ''}")


def _maybe_downsample(df: pd.DataFrame, *, max_points: int, sort_by: Optional[str] = None) -> pd.DataFrame:
    """Downsample a DataFrame to at most `max_points` rows for plotting."""
    n = len(df)
    if n <= max_points:
        return df
    if sort_by and sort_by in df.columns:
        # Even sampling after sort to preserve temporal order distribution
        idx = np.linspace(0, n - 1, num=max_points, dtype=int)
        return df.sort_values(sort_by).iloc[idx]
    # Random sample otherwise
    return df.sample(n=max_points, random_state=42)


def create_temperature_trends_tab(df: pd.DataFrame, *, max_points_to_plot: int):
    """Create temperature trends analysis tab using PRECALCULATED data."""
    
    st.subheader("Long-term Temperature Trends")
    
    # Check if this is the yearly precalculated data
    if 'trend_line' in df.columns and 'trend_slope_per_decade' in df.columns:
        # This is precalculated yearly data with trends - just display it!
        st.info("✓ Displaying precalculated trend data from Spark processing")
        
        # Time period selection
        if 'year' in df.columns:
            min_year, max_year = int(df['year'].min()), int(df['year'].max())
            
            col1, col2 = st.columns(2)
            with col1:
                start_year = st.slider("Start Year", min_year, max_year, min_year)
            with col2:
                end_year = st.slider("End Year", min_year, max_year, max_year)
            
            # Filter data
            yearly_data = df[(df['year'] >= start_year) & (df['year'] <= end_year)]
        else:
            yearly_data = df
        
        if yearly_data.empty:
            st.warning("No data available for the selected time period!")
            return
        
        # Get trend slope (already calculated)
        trend_slope = yearly_data['trend_slope_per_decade'].iloc[0] if 'trend_slope_per_decade' in yearly_data.columns else 0
        
        # Main trend plot with PRECALCULATED trend line
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Temperature Trends', 'Temperature Variability'),
            vertical_spacing=0.1
        )
        
        # Average temperature line
        fig.add_trace(
            go.Scatter(
                x=yearly_data['year'], 
                y=yearly_data['avg_temperature'],
                mode='lines+markers',
                name='Average Temperature',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Precalculated trend line
        fig.add_trace(
            go.Scatter(
                x=yearly_data['year'],
                y=yearly_data['trend_line'],
                mode='lines',
                name=f'Trend ({trend_slope:.3f}°C/decade)',
                line=dict(color='red', dash='dash', width=2)
            ),
            row=1, col=1
        )
        
        # Temperature range
        fig.add_trace(
            go.Scatter(
                x=yearly_data['year'],
                y=yearly_data['max_temperature'],
                fill=None,
                mode='lines',
                line_color='rgba(0,0,255,0)',
                showlegend=False
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=yearly_data['year'],
                y=yearly_data['min_temperature'],
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,0,255,0)',
                name='Temperature Range',
                fillcolor='rgba(0,100,80,0.2)'
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=yearly_data['year'],
                y=yearly_data['avg_temperature'],
                mode='lines+markers',
                name='Average',
                line=dict(color='red', width=2)
            ),
            row=2, col=1
        )
        
        fig.update_layout(height=600, showlegend=True)
        fig.update_xaxes(title_text="Year")
        fig.update_yaxes(title_text="Temperature (°C)")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Trend statistics (all precalculated)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Trend per Decade", f"{trend_slope:.3f}°C")
        
        with col2:
            temp_range = yearly_data['avg_temperature'].max() - yearly_data['avg_temperature'].min()
            st.metric("Temperature Range", f"{temp_range:.2f}°C")
        
        with col3:
            if 'std_temperature' in yearly_data.columns:
                temp_std = yearly_data['std_temperature'].mean()
                st.metric("Avg Variability (Std)", f"{temp_std:.3f}°C")
    else:
        st.warning("⚠ This file doesn't contain precalculated yearly trend data. Please use yearly.parquet file.")
        st.info("Expected columns: year, avg_temperature, trend_line, trend_slope_per_decade")


def create_heatmaps_tab(df: pd.DataFrame):
    """Create heatmaps analysis tab using PRECALCULATED monthly data."""
    
    st.subheader("Temperature Heatmaps")
    
    # Check if we have monthly precalculated data
    if 'year' not in df.columns or 'month' not in df.columns:
        st.error("This file doesn't contain year/month data. Please use monthly.parquet file.")
        return
    
    if 'avg_temperature' not in df.columns:
        st.error("This file doesn't contain avg_temperature. Please use monthly.parquet file.")
        return
    
    st.info("✓ Displaying precalculated monthly data from Spark processing")
    
    # Create pivot table for heatmap (just reshaping, not calculating)
    heatmap_data = df.pivot_table(
        values='avg_temperature',
        index='year',
        columns='month',
        aggfunc='first'  # Data is already aggregated, just take first value
    )
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        y=heatmap_data.index,
        colorscale='RdYlBu_r',
        hoverongaps=False,
        hovertemplate='Year: %{y}<br>Month: %{x}<br>Temperature: %{z:.2f}°C<extra></extra>'
    ))
    
    fig.update_layout(
        title="Temperature Heatmap (°C)",
        xaxis_title="Month",
        yaxis_title="Year",
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Heatmap statistics (simple operations on precalculated data)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        hottest_year = heatmap_data.mean(axis=1).idxmax()
        st.metric("Hottest Year", f"{hottest_year}")
    
    with col2:
        coldest_year = heatmap_data.mean(axis=1).idxmin()
        st.metric("Coldest Year", f"{coldest_year}")
    
    with col3:
        max_temp = heatmap_data.max().max()
        st.metric("Max Temperature", f"{max_temp:.2f}°C")


def create_seasonal_analysis_tab(df: pd.DataFrame, *, max_points_to_plot: int):
    """Create seasonal analysis tab using PRECALCULATED data."""
    
    st.subheader("Seasonal Temperature Analysis")
    
    # Check if this is climatology or seasonal precalculated data
    has_climatology = 'climatology_mean' in df.columns
    has_seasonal = 'season' in df.columns
    
    if not has_climatology and not has_seasonal:
        st.warning("⚠ Please load climatology.parquet or seasonal.parquet for seasonal analysis")
        st.info("These files contain precalculated seasonal statistics from Spark processing")
        return
    
    st.info("✓ Displaying precalculated seasonal data from Spark processing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Monthly climatology plot (if climatology data is loaded)
        if has_climatology:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df['month'],
                y=df['climatology_mean'],
                mode='lines+markers',
                name='Average Temperature',
                line=dict(color='blue', width=3),
                marker=dict(size=8)
            ))
            
            # Add error bars if std is available
            if 'climatology_std' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['month'],
                    y=df['climatology_mean'] + df['climatology_std'],
                    fill=None,
                    mode='lines',
                    line_color='rgba(0,0,255,0)',
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=df['month'],
                    y=df['climatology_mean'] - df['climatology_std'],
                    fill='tonexty',
                    mode='lines',
                    line_color='rgba(0,0,255,0)',
                    name='±1 Std Dev',
                    fillcolor='rgba(0,100,80,0.2)'
                ))
            
            fig.update_layout(
                title="Monthly Temperature Climatology",
                xaxis_title="Month",
                yaxis_title="Temperature (°C)",
                xaxis=dict(tickmode='array', tickvals=list(range(1, 13)),
                          ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Load climatology.parquet to see monthly climatology")
    
    with col2:
        # Seasonal box plot using precalculated seasonal stats
        if has_seasonal:
            fig = go.Figure()
            
            # Create simple bar chart with error bars from precalculated data
            seasons_order = ["Spring", "Summer", "Fall", "Winter"]
            df_sorted = df.set_index('season').reindex(seasons_order).reset_index()
            
            fig.add_trace(go.Bar(
                x=df_sorted['season'],
                y=df_sorted['avg_temperature'],
                error_y=dict(
                    type='data',
                    array=df_sorted['std_temperature'] if 'std_temperature' in df_sorted.columns else None
                ),
                marker_color=['lightgreen', 'orange', 'brown', 'lightblue']
            ))
            
            fig.update_layout(
                title="Seasonal Temperature Distribution",
                xaxis_title="Season",
                yaxis_title="Temperature (°C)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Load seasonal.parquet to see seasonal distribution")
    
    # Seasonal statistics table
    if has_seasonal:
        st.subheader("Seasonal Statistics")
        
        display_df = df[['season', 'avg_temperature', 'std_temperature', 'min_temperature', 'max_temperature']].copy()
        display_df.columns = ['Season', 'Mean (°C)', 'Std Dev (°C)', 'Min (°C)', 'Max (°C)']
        display_df = display_df.round(2)
        
        st.dataframe(display_df, use_container_width=True)


def create_extreme_events_tab(df: pd.DataFrame, *, max_points_to_plot: int):
    """Create extreme events analysis tab using PRECALCULATED thresholds."""
    
    st.subheader("Extreme Temperature Events")
    
    # Check if this is the anomalies data with precalculated z-scores
    if 'is_anomaly' in df.columns and 'temp_zscore' in df.columns:
        st.info("✓ Using precalculated anomalies and extreme events from Spark processing")
        
        # Show statistics
        total_events = len(df)
        extreme_events = df[df['is_anomaly'] == True]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Records", f"{total_events:,}")
        
        with col2:
            extreme_count = len(extreme_events)
            st.metric("Extreme Events", f"{extreme_count:,}")
        
        with col3:
            extreme_pct = (extreme_count / total_events * 100) if total_events > 0 else 0
            st.metric("Extreme Event Rate", f"{extreme_pct:.2f}%")
        
        # Visualization of extreme events
        if 'year' in df.columns:
            # Downsample for plotting
            plot_df = _maybe_downsample(df[['year', 'temperature', 'is_anomaly', 'temp_zscore']], 
                                       max_points=max_points_to_plot, sort_by='year')
            
            fig = go.Figure()
            
            # Normal data
            normal_data = plot_df[plot_df['is_anomaly'] == False]
            fig.add_trace(go.Scatter(
                x=normal_data['year'],
                y=normal_data['temperature'],
                mode='markers',
                name='Normal',
                marker=dict(color='lightblue', size=4, opacity=0.6)
            ))
            
            # Extreme events
            extremes = plot_df[plot_df['is_anomaly'] == True]
            if not extremes.empty:
                # Hot extremes (positive z-score)
                hot_extremes = extremes[extremes['temp_zscore'] > 0]
                if not hot_extremes.empty:
                    fig.add_trace(go.Scatter(
                        x=hot_extremes['year'],
                        y=hot_extremes['temperature'],
                        mode='markers',
                        name='Hot Extremes',
                        marker=dict(color='red', size=8, symbol='triangle-up')
                    ))
                
                # Cold extremes (negative z-score)
                cold_extremes = extremes[extremes['temp_zscore'] < 0]
                if not cold_extremes.empty:
                    fig.add_trace(go.Scatter(
                        x=cold_extremes['year'],
                        y=cold_extremes['temperature'],
                        mode='markers',
                        name='Cold Extremes',
                        marker=dict(color='blue', size=8, symbol='triangle-down')
                    ))
            
            fig.update_layout(
                title="Extreme Temperature Events Over Time (Precalculated)",
                xaxis_title="Year",
                yaxis_title="Temperature (°C)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Show sample of extreme events
        if not extreme_events.empty:
            st.subheader("Recent Extreme Events")
            display_cols = [c for c in ['year', 'month', 'temperature', 'temp_zscore'] if c in extreme_events.columns]
            sample_extremes = extreme_events[display_cols].head(100).sort_values('temp_zscore', ascending=False).head(10)
            st.dataframe(sample_extremes, use_container_width=True)
    
    else:
        st.warning("⚠ This file doesn't contain precalculated extreme events data.")
        st.info("Please load anomalies.parquet which contains precalculated anomaly detection from Spark processing.")
        st.info("Expected columns: is_anomaly, temp_zscore")


def create_regional_analysis_tab(df: pd.DataFrame, *, max_points_to_plot: int):
    """Create regional analysis tab showing temperature patterns by geographic region."""
    
    st.subheader("🌍 Temperature Analysis by Geographic Region")
    
    # Check if this is regional data
    if 'region' not in df.columns or 'continent' not in df.columns:
        st.warning("⚠ This file doesn't contain regional analysis data.")
        st.info("Please load regional.parquet which contains precalculated regional aggregations.")
        st.info("Expected columns: region, continent, year, avg_temperature, record_count")
        return
    
    # Overview metrics
    st.markdown("### 📊 Regional Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_regions = df['region'].nunique()
        st.metric("Total Regions", f"{n_regions}")
    
    with col2:
        n_years = df['year'].nunique() if 'year' in df.columns else 0
        st.metric("Years of Data", f"{n_years}")
    
    with col3:
        total_records = df['record_count'].sum() if 'record_count' in df.columns else len(df)
        st.metric("Total Records", f"{total_records:,.0f}")
    
    # World Map Visualization - NEW!
    st.markdown("### 🗺️ Global Temperature Distribution by Region")
    
    if 'year' in df.columns and 'avg_temperature' in df.columns:
        # Get latest year for map
        latest_year = df['year'].max()
        map_data = df[df['year'] == latest_year].copy()
        
        # Create world map using Plotly's choropleth
        fig = go.Figure()
        
        # Map region names to approximate center coordinates for scatter plot
        region_coords = {
            'Northern Europe': {'lat': 60, 'lon': 15},
            'Central Europe': {'lat': 50, 'lon': 10},
            'Southern Europe': {'lat': 40, 'lon': 15},
            'Northern Asia': {'lat': 60, 'lon': 100},
            'Central Asia': {'lat': 45, 'lon': 65},
            'South Asia': {'lat': 20, 'lon': 80},
            'East Asia': {'lat': 35, 'lon': 115},
            'Northern Africa': {'lat': 25, 'lon': 15},
            'Central Africa': {'lat': 5, 'lon': 20},
            'Southern Africa': {'lat': -25, 'lon': 25},
            'Northern North America': {'lat': 60, 'lon': -100},
            'Central North America': {'lat': 40, 'lon': -100},
            'Caribbean & Central America': {'lat': 15, 'lon': -80},
            'Northern South America': {'lat': 0, 'lon': -60},
            'Central South America': {'lat': -15, 'lon': -60},
            'Southern South America': {'lat': -40, 'lon': -65},
            'Northern Oceania': {'lat': -15, 'lon': 135},
            'Southern Oceania': {'lat': -35, 'lon': 145},
            'Antarctica': {'lat': -75, 'lon': 0}
        }
        
        # Add coordinates to map_data
        map_data['lat'] = map_data['region'].map(lambda x: region_coords.get(x, {}).get('lat', 0))
        map_data['lon'] = map_data['region'].map(lambda x: region_coords.get(x, {}).get('lon', 0))
        
        # Create scatter geo plot with bubble sizes
        fig = go.Figure(data=go.Scattergeo(
            lon=map_data['lon'],
            lat=map_data['lat'],
            text=map_data['region'] + '<br>Temp: ' + map_data['avg_temperature'].round(1).astype(str) + '°C',
            mode='markers',
            marker=dict(
                size=map_data['avg_temperature'].abs() * 2 + 10,  # Size based on temperature
                color=map_data['avg_temperature'],
                colorscale='RdYlBu_r',
                showscale=True,
                colorbar=dict(
                    title="Temp (°C)",
                    x=1.1
                ),
                line=dict(width=0.5, color='white')
            ),
            hoverinfo='text'
        ))
        
        fig.update_layout(
            title=f'Global Temperature Distribution by Region ({latest_year})',
            geo=dict(
                projection_type='natural earth',
                showland=True,
                landcolor='rgb(243, 243, 243)',
                coastlinecolor='rgb(204, 204, 204)',
                showocean=True,
                oceancolor='rgb(230, 245, 255)',
                showcountries=True,
                countrycolor='rgb(204, 204, 204)'
            ),
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Regional selector
    st.markdown("### 🔍 Explore by Region")
    
    regions = sorted(df['region'].unique())
    selected_region = st.selectbox("Select Region", regions)
    
    # Filter data for selected region
    region_data = df[df['region'] == selected_region].copy()
    
    if not region_data.empty and 'year' in region_data.columns and 'avg_temperature' in region_data.columns:
        # Temperature trend for selected region
        st.markdown(f"#### Temperature Trend: {selected_region}")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=region_data['year'],
            y=region_data['avg_temperature'],
            mode='lines+markers',
            name='Average Temperature',
            line=dict(color='#FF6B6B', width=2),
            marker=dict(size=4)
        ))
        
        # Add min/max if available
        if 'min_temperature' in region_data.columns and 'max_temperature' in region_data.columns:
            fig.add_trace(go.Scatter(
                x=region_data['year'],
                y=region_data['max_temperature'],
                mode='lines',
                name='Max Temperature',
                line=dict(color='#FF8787', width=1, dash='dash'),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=region_data['year'],
                y=region_data['min_temperature'],
                mode='lines',
                name='Min Temperature',
                line=dict(color='#4ECDC4', width=1, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(78, 205, 196, 0.1)',
                showlegend=False
            ))
        
        fig.update_layout(
            title=f"Temperature Evolution in {selected_region}",
            xaxis_title="Year",
            yaxis_title="Temperature (°C)",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics for selected region
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_temp = region_data['avg_temperature'].mean()
            st.metric("Average Temp", f"{avg_temp:.2f}°C")
        
        with col2:
            if 'min_temperature' in region_data.columns:
                min_temp = region_data['min_temperature'].min()
                st.metric("Lowest Recorded", f"{min_temp:.2f}°C")
        
        with col3:
            if 'max_temperature' in region_data.columns:
                max_temp = region_data['max_temperature'].max()
                st.metric("Highest Recorded", f"{max_temp:.2f}°C")
        
        with col4:
            if 'std_temperature' in region_data.columns:
                avg_std = region_data['std_temperature'].mean()
                st.metric("Avg Variability", f"{avg_std:.2f}°C")
    
    # Compare all regions
    st.markdown("### 📊 Compare All Regions")
    
    if 'year' in df.columns and 'avg_temperature' in df.columns:
        # Get most recent year
        latest_year = df['year'].max()
        latest_data = df[df['year'] == latest_year].sort_values('avg_temperature', ascending=False)
        
        st.markdown(f"#### Temperature by Region ({latest_year})")
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=latest_data['region'],
            y=latest_data['avg_temperature'],
            marker=dict(
                color=latest_data['avg_temperature'],
                colorscale='RdYlBu_r',
                showscale=True,
                colorbar=dict(title="Temp (°C)")
            ),
            text=latest_data['avg_temperature'].round(1),
            textposition='outside'
        ))
        
        fig.update_layout(
            title=f"Average Temperature by Region in {latest_year}",
            xaxis_title="Region",
            yaxis_title="Temperature (°C)",
            height=500,
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Heatmap over time
        st.markdown("#### Regional Temperature Evolution Heatmap")
        
        # Pivot data for heatmap
        pivot_data = df.pivot_table(
            index='region',
            columns='year',
            values='avg_temperature',
            aggfunc='first'
        )
        
        # Sample years if too many
        if len(pivot_data.columns) > 50:
            step = len(pivot_data.columns) // 50
            pivot_data = pivot_data.iloc[:, ::step]
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale='RdYlBu_r',
            colorbar=dict(title="Temp (°C)")
        ))
        
        fig.update_layout(
            title="Temperature Evolution Across Regions",
            xaxis_title="Year",
            yaxis_title="Region",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)


def create_continental_analysis_tab(df: pd.DataFrame, *, max_points_to_plot: int):
    """Create continental analysis tab showing temperature patterns by continent."""
    
    st.subheader("🌐 Temperature Analysis by Continent")
    
    # Check if this is continental data
    if 'continent' not in df.columns:
        st.warning("⚠ This file doesn't contain continental analysis data.")
        st.info("Please load continental.parquet which contains precalculated continental aggregations.")
        st.info("Expected columns: continent, year, avg_temperature, record_count")
        return
    
    # Overview metrics
    st.markdown("### 📊 Continental Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_continents = df['continent'].nunique()
        st.metric("Continents", f"{n_continents}")
    
    with col2:
        n_years = df['year'].nunique() if 'year' in df.columns else 0
        st.metric("Years of Data", f"{n_years}")
    
    with col3:
        total_records = df['record_count'].sum() if 'record_count' in df.columns else len(df)
        st.metric("Total Records", f"{total_records:,.0f}")
    
    # World Map Visualization by Continent - NEW!
    st.markdown("### 🗺️ Continental Temperature Overview")
    
    if 'year' in df.columns and 'avg_temperature' in df.columns:
        # Get latest year for map
        latest_year = df['year'].max()
        map_data = df[df['year'] == latest_year].copy()
        
        # Map continent names to approximate center coordinates
        continent_coords = {
            'Europe': {'lat': 50, 'lon': 15},
            'Asia': {'lat': 45, 'lon': 90},
            'Africa': {'lat': 5, 'lon': 20},
            'North America': {'lat': 50, 'lon': -100},
            'South America': {'lat': -15, 'lon': -60},
            'Oceania': {'lat': -25, 'lon': 135},
            'Antarctica': {'lat': -75, 'lon': 0}
        }
        
        # Add coordinates to map_data
        map_data['lat'] = map_data['continent'].map(lambda x: continent_coords.get(x, {}).get('lat', 0))
        map_data['lon'] = map_data['continent'].map(lambda x: continent_coords.get(x, {}).get('lon', 0))
        
        # Create scatter geo plot with larger bubbles for continents
        fig_map = go.Figure(data=go.Scattergeo(
            lon=map_data['lon'],
            lat=map_data['lat'],
            text=map_data['continent'] + '<br>Temp: ' + map_data['avg_temperature'].round(1).astype(str) + '°C' + 
                 '<br>Records: ' + map_data['record_count'].astype(str),
            mode='markers+text',
            marker=dict(
                size=map_data['avg_temperature'].abs() * 3 + 20,  # Larger bubbles for continents
                color=map_data['avg_temperature'],
                colorscale='RdYlBu_r',
                showscale=True,
                colorbar=dict(
                    title="Avg Temp (°C)",
                    x=1.1
                ),
                line=dict(width=1, color='black')
            ),
            text=map_data['continent'],
            textfont=dict(size=10, color='black'),
            textposition='middle center',
            hoverinfo='text'
        ))
        
        fig_map.update_layout(
            title=f'Continental Temperature Overview ({latest_year})',
            geo=dict(
                projection_type='natural earth',
                showland=True,
                landcolor='rgb(243, 243, 243)',
                coastlinecolor='rgb(100, 100, 100)',
                showocean=True,
                oceancolor='rgb(220, 235, 255)',
                showcountries=True,
                countrycolor='rgb(204, 204, 204)'
            ),
            height=600
        )
        
        st.plotly_chart(fig_map, use_container_width=True)
    
    # Continental comparison
    st.markdown("### 🌍 Temperature Trends by Continent")
    
    if 'year' in df.columns and 'avg_temperature' in df.columns:
        fig = go.Figure()
        
        # Define colors for continents
        continent_colors = {
            'Africa': '#FF6B6B',
            'Asia': '#4ECDC4',
            'Europe': '#45B7D1',
            'North America': '#F9CA24',
            'South America': '#6C5CE7',
            'Oceania': '#00B894',
            'Antarctica': '#636E72'
        }
        
        for continent in sorted(df['continent'].unique()):
            continent_data = df[df['continent'] == continent].sort_values('year')
            
            fig.add_trace(go.Scatter(
                x=continent_data['year'],
                y=continent_data['avg_temperature'],
                mode='lines+markers',
                name=continent,
                line=dict(
                    color=continent_colors.get(continent, '#000000'),
                    width=2
                ),
                marker=dict(size=4)
            ))
        
        fig.update_layout(
            title="Temperature Evolution by Continent",
            xaxis_title="Year",
            yaxis_title="Average Temperature (°C)",
            height=600,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Continental statistics
        st.markdown("### 📈 Continental Statistics")
        
        continents = sorted(df['continent'].unique())
        
        for continent in continents:
            with st.expander(f"🌍 {continent}"):
                cont_data = df[df['continent'] == continent]
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_temp = cont_data['avg_temperature'].mean()
                    st.metric("Average Temp", f"{avg_temp:.2f}°C")
                
                with col2:
                    if 'min_temperature' in cont_data.columns:
                        min_temp = cont_data['min_temperature'].min()
                        st.metric("Lowest Recorded", f"{min_temp:.2f}°C")
                
                with col3:
                    if 'max_temperature' in cont_data.columns:
                        max_temp = cont_data['max_temperature'].max()
                        st.metric("Highest Recorded", f"{max_temp:.2f}°C")
                
                with col4:
                    records = cont_data['record_count'].sum() if 'record_count' in cont_data.columns else len(cont_data)
                    st.metric("Total Records", f"{records:,.0f}")
                
                # Temperature distribution
                if len(cont_data) > 10:
                    fig = go.Figure()
                    
                    fig.add_trace(go.Histogram(
                        x=cont_data['avg_temperature'],
                        nbinsx=30,
                        name=continent,
                        marker=dict(color=continent_colors.get(continent, '#000000'))
                    ))
                    
                    fig.update_layout(
                        title=f"Temperature Distribution: {continent}",
                        xaxis_title="Temperature (°C)",
                        yaxis_title="Frequency",
                        height=300,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        # Latest year comparison
        st.markdown("### 🌡️ Current Year Comparison")
        
        latest_year = df['year'].max()
        latest_data = df[df['year'] == latest_year].sort_values('avg_temperature', ascending=False)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=latest_data['continent'],
            y=latest_data['avg_temperature'],
            marker=dict(
                color=[continent_colors.get(c, '#000000') for c in latest_data['continent']]
            ),
            text=latest_data['avg_temperature'].round(1),
            textposition='outside'
        ))
        
        fig.update_layout(
            title=f"Average Temperature by Continent ({latest_year})",
            xaxis_title="Continent",
            yaxis_title="Temperature (°C)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)


def create_eda_tab(df: pd.DataFrame, *, max_points_to_plot: int):
    """Create Exploratory Data Analysis (EDA) tab with correlation, descriptive stats, and chi-square tests."""
    
    st.subheader("📊 Exploratory Data Analysis")
    
    st.markdown("""
    This tab shows comprehensive statistical analysis computed in Spark:
    - **Pearson Correlation Matrix**: Linear relationships between numeric variables
    - **Descriptive Statistics**: Mean, median, quartiles, skewness, kurtosis
    - **Chi-Square Tests**: Independence tests for categorical variables
    """)
    
    # Check what type of EDA file is loaded
    if 'variable_1' in df.columns and 'variable_2' in df.columns and 'correlation' in df.columns:
        # This is correlation_matrix.parquet
        st.markdown("### 🔗 Pearson Correlation Matrix")
        
        # Pivot to create matrix format
        corr_pivot = df.pivot(index='variable_1', columns='variable_2', values='correlation')
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_pivot.values,
            x=corr_pivot.columns,
            y=corr_pivot.index,
            colorscale='RdBu_r',
            zmid=0,
            text=corr_pivot.values.round(3),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title="Correlation Matrix (Pearson r)",
            xaxis_title="Variable",
            yaxis_title="Variable",
            height=600,
            xaxis={'side': 'bottom'},
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show top correlations
        st.markdown("#### 📌 Strongest Correlations")
        
        # Filter for unique pairs (upper triangle)
        unique_pairs = df[df['variable_1'] < df['variable_2']].copy()
        unique_pairs['abs_corr'] = unique_pairs['correlation'].abs()
        top_corr = unique_pairs.nlargest(10, 'abs_corr')[['variable_1', 'variable_2', 'correlation']]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Positive Correlations**")
            positive = top_corr[top_corr['correlation'] > 0].head(5)
            if not positive.empty:
                for _, row in positive.iterrows():
                    st.write(f"• {row['variable_1']} ↔ {row['variable_2']}: **{row['correlation']:.3f}**")
            else:
                st.info("No strong positive correlations")
        
        with col2:
            st.markdown("**Negative Correlations**")
            negative = top_corr[top_corr['correlation'] < 0].head(5)
            if not negative.empty:
                for _, row in negative.iterrows():
                    st.write(f"• {row['variable_1']} ↔ {row['variable_2']}: **{row['correlation']:.3f}**")
            else:
                st.info("No strong negative correlations")
    
    elif 'variable' in df.columns and 'statistic' in df.columns:
        # This is descriptive_stats.parquet (already pivoted)
        st.markdown("### 📈 Descriptive Statistics")
        
        # Display as styled dataframe
        st.dataframe(
            df.style.format("{:.3f}", na_rep="-").background_gradient(cmap='YlOrRd', axis=None),
            use_container_width=True
        )
        
        # Visualize distributions
        if 'mean' in df.columns and 'std_dev' in df.columns:
            st.markdown("#### 📊 Variable Distributions (Mean ± Std Dev)")
            
            fig = go.Figure()
            
            for _, row in df.iterrows():
                if pd.notna(row.get('mean')) and pd.notna(row.get('std_dev')):
                    fig.add_trace(go.Bar(
                        name=row['variable'],
                        x=[row['variable']],
                        y=[row['mean']],
                        error_y=dict(
                            type='data',
                            array=[row['std_dev']],
                            visible=True
                        )
                    ))
            
            fig.update_layout(
                title="Mean Values with Standard Deviation",
                xaxis_title="Variable",
                yaxis_title="Value",
                height=500,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Box plot visualization
        if 'min' in df.columns and 'q1' in df.columns and 'median' in df.columns:
            st.markdown("#### 📦 Box Plot Representation")
            
            fig = go.Figure()
            
            for _, row in df.iterrows():
                if pd.notna(row.get('median')):
                    fig.add_trace(go.Box(
                        name=row['variable'],
                        q1=[row.get('q1', 0)],
                        median=[row.get('median', 0)],
                        q3=[row.get('q3', 0)],
                        lowerfence=[row.get('min', 0)],
                        upperfence=[row.get('max', 0)],
                        boxmean='sd'
                    ))
            
            fig.update_layout(
                title="Five-Number Summary (Min, Q1, Median, Q3, Max)",
                yaxis_title="Temperature (°C)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    elif 'test' in df.columns and 'chi_square_statistic' in df.columns:
        # This is chi_square_tests.parquet
        st.markdown("### 🧪 Chi-Square Independence Tests")
        
        st.markdown("""
        Chi-Square tests evaluate whether two categorical variables are independent.
        - **Null Hypothesis (H₀)**: Variables are independent
        - **Alternative (H₁)**: Variables are dependent
        - **Significance level**: α = 0.05
        """)
        
        # Display results table
        st.dataframe(
            df.style.format({
                'chi_square_statistic': '{:.4f}',
                'p_value': '{:.4e}',
                'degrees_of_freedom': '{:.0f}'
            }).applymap(
                lambda v: 'background-color: lightgreen' if v == True else '',
                subset=['is_significant']
            ),
            use_container_width=True
        )
        
        # Visualize test results
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=df['test'],
            y=df['chi_square_statistic'],
            marker=dict(
                color=df['is_significant'].map({True: 'red', False: 'green'}),
                line=dict(color='black', width=1)
            ),
            text=df['p_value'].apply(lambda p: f'p={p:.4f}'),
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Chi-Square Test Statistics",
            xaxis_title="Test",
            yaxis_title="χ² Statistic",
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretation
        st.markdown("#### 🔍 Interpretation")
        
        for _, row in df.iterrows():
            if row['is_significant']:
                st.warning(f"**{row['test']}**: Significant relationship detected (p < 0.05). "
                          f"Variables **{row['variable_1']}** and **{row['variable_2']}** are likely dependent.")
            else:
                st.success(f"**{row['test']}**: No significant relationship (p ≥ 0.05). "
                          f"Variables **{row['variable_1']}** and **{row['variable_2']}** appear independent.")
    
    else:
        st.warning("⚠ This file doesn't contain EDA statistics data.")
        st.info("""
        Please load one of the following EDA Parquet files:
        - **correlation_matrix.parquet**: Pearson correlation coefficients
        - **descriptive_stats.parquet**: Descriptive statistics (mean, median, quartiles, etc.)
        - **chi_square_tests.parquet**: Chi-square independence tests
        """)
        
        # Show available columns
        st.markdown("**Available columns in current file:**")
        st.code(", ".join(df.columns.tolist()))


def get_temperature_column(df: pd.DataFrame) -> Optional[str]:
    """Find the temperature column in the dataframe."""
    temp_columns = ['temperature', 'avg_temperature', 'mean_temperature', 'temp']
    
    for col in temp_columns:
        if col in df.columns:
            return col
    
    # Look for columns containing 'temp'
    for col in df.columns:
        if 'temp' in col.lower():
            return col
    
    return None


if __name__ == "__main__":
    main()