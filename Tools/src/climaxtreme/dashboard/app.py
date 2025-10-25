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


def run_dashboard(host: str = "localhost", port: int = 8501, data_dir: str = "data") -> None:
    """
    Launch the Streamlit dashboard.
    
    Args:
        host: Host to run the dashboard on
        port: Port to run the dashboard on
        data_dir: Directory containing climate data
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
    
    # Set data_dir as environment variable so main() can read it
    import os
    os.environ["CLIMAXTREME_DATA_DIR"] = data_dir
    
    print(f"Launching Streamlit dashboard at http://{host}:{port}")
    print(f"Data directory: {data_dir}")
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
    # Compute sensible default to repo-root/DATA
    import os
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
    available_files = load_available_files(data_path)
    
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


def load_available_files(data_path: Path) -> List[str]:
    """Load list of available data files."""
    file_patterns = ['*.csv', '*.parquet']
    available_files = []
    
    for pattern in file_patterns:
        available_files.extend([f.name for f in data_path.glob(pattern)])
    
    return sorted(available_files)


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
    file_path: Path,
    *,
    bigdata_mode: bool,
    max_rows: int,
    sample_seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Load a data file into a pandas DataFrame with memory-safe options.

    Returns: (df, meta) where meta includes {"bigdata_mode", "loaded_rows", "source"}.
    """
    meta: Dict[str, object] = {"bigdata_mode": bigdata_mode, "source": file_path.name}
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
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return pd.DataFrame(), meta


def create_dashboard_content(df: pd.DataFrame, filename: str, *, max_points_to_plot: int):
    """Create the main dashboard content."""
    
    # Data overview
    st.header("📊 Data Overview")
    create_data_overview(df, filename)
    
    # Navigation tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🌡️ Temperature Trends", "🗺️ Heatmaps", "📈 Seasonal Analysis", "⚡ Extreme Events"])
    
    with tab1:
        create_temperature_trends_tab(df, max_points_to_plot=max_points_to_plot)
    
    with tab2:
        create_heatmaps_tab(df)
    
    with tab3:
        create_seasonal_analysis_tab(df, max_points_to_plot=max_points_to_plot)
    
    with tab4:
        create_extreme_events_tab(df, max_points_to_plot=max_points_to_plot)


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