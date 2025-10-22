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
from typing import Dict, List, Optional
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
    import streamlit.web.cli as stcli
    import sys
    
    # Set up Streamlit configuration
    st.set_page_config(
        page_title="climaXtreme Dashboard",
        page_icon="🌡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Store data directory in session state
    if 'data_dir' not in st.session_state:
        st.session_state.data_dir = data_dir
    
    # Run the dashboard
    sys.argv = ["streamlit", "run", __file__, "--server.address", host, "--server.port", str(port)]
    sys.exit(stcli.main())


def main():
    """Main dashboard application."""
    
    # Page title and description
    st.title("🌡️ climaXtreme Dashboard")
    st.markdown("Interactive climate data analysis and visualization")
    
    # Sidebar configuration
    st.sidebar.title("Configuration")
    # Compute sensible default to repo-root/DATA
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
    df = load_data_file(data_path / selected_file)
    
    if df is None or df.empty:
        st.error("Failed to load the selected data file!")
        st.stop()
    
    # Main dashboard content
    create_dashboard_content(df, selected_file)


def load_available_files(data_path: Path) -> List[str]:
    """Load list of available data files."""
    file_patterns = ['*.csv', '*.parquet']
    available_files = []
    
    for pattern in file_patterns:
        available_files.extend([f.name for f in data_path.glob(pattern)])
    
    return sorted(available_files)


def load_data_file(file_path: Path) -> Optional[pd.DataFrame]:
    """Load a data file into a pandas DataFrame and normalize date columns if present."""
    try:
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix == '.parquet':
            df = pd.read_parquet(file_path)
        else:
            st.error(f"Unsupported file format: {file_path.suffix}")
            return None

        # Normalize mixed-format date column commonly named 'dt'
        try:
            from climaxtreme.utils import add_date_parts
            if 'dt' in df.columns:
                # Don't drop invalid rows here to allow user inspection; downstream filters can handle NaT
                df = add_date_parts(df, date_col='dt', drop_invalid=False)
        except Exception as _:
            # Non-fatal; continue without normalization
            pass

        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


def create_dashboard_content(df: pd.DataFrame, filename: str):
    """Create the main dashboard content."""
    
    # Data overview
    st.header("📊 Data Overview")
    create_data_overview(df, filename)
    
    # Navigation tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🌡️ Temperature Trends", "🗺️ Heatmaps", "📈 Seasonal Analysis", "⚡ Extreme Events"])
    
    with tab1:
        create_temperature_trends_tab(df)
    
    with tab2:
        create_heatmaps_tab(df)
    
    with tab3:
        create_seasonal_analysis_tab(df)
    
    with tab4:
        create_extreme_events_tab(df)


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


def create_temperature_trends_tab(df: pd.DataFrame):
    """Create temperature trends analysis tab."""
    
    st.subheader("Long-term Temperature Trends")
    
    temp_col = get_temperature_column(df)
    if not temp_col:
        st.error("No temperature column found in the dataset!")
        return
    
    # Time period selection
    if 'year' in df.columns:
        min_year, max_year = int(df['year'].min()), int(df['year'].max())
        
        col1, col2 = st.columns(2)
        with col1:
            start_year = st.slider("Start Year", min_year, max_year, min_year)
        with col2:
            end_year = st.slider("End Year", min_year, max_year, max_year)
        
        # Filter data
        filtered_df = df[(df['year'] >= start_year) & (df['year'] <= end_year)]
    else:
        filtered_df = df
        start_year, end_year = None, None
    
    if filtered_df.empty:
        st.warning("No data available for the selected time period!")
        return
    
    # Generate trends visualization
    if 'year' in filtered_df.columns:
        yearly_data = filtered_df.groupby('year')[temp_col].agg(['mean', 'min', 'max']).reset_index()
        
        # Main trend plot
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Temperature Trends', 'Temperature Variability'),
            vertical_spacing=0.1
        )
        
        # Trend line
        fig.add_trace(
            go.Scatter(
                x=yearly_data['year'], 
                y=yearly_data['mean'],
                mode='lines+markers',
                name='Average Temperature',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Add trend line
        z = np.polyfit(yearly_data['year'], yearly_data['mean'], 1)
        trend_line = np.poly1d(z)(yearly_data['year'])
        
        fig.add_trace(
            go.Scatter(
                x=yearly_data['year'],
                y=trend_line,
                mode='lines',
                name=f'Trend ({z[0]*10:.3f}°C/decade)',
                line=dict(color='red', dash='dash', width=2)
            ),
            row=1, col=1
        )
        
        # Temperature range
        fig.add_trace(
            go.Scatter(
                x=yearly_data['year'],
                y=yearly_data['max'],
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
                y=yearly_data['min'],
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
                y=yearly_data['mean'],
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
        
        # Trend statistics
        trend_per_decade = z[0] * 10
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Trend per Decade", f"{trend_per_decade:.3f}°C")
        
        with col2:
            temp_range = yearly_data['mean'].max() - yearly_data['mean'].min()
            st.metric("Temperature Range", f"{temp_range:.2f}°C")
        
        with col3:
            temp_std = yearly_data['mean'].std()
            st.metric("Variability (Std)", f"{temp_std:.3f}°C")
    
    else:
        # Simple histogram for non-time series data
        fig = px.histogram(filtered_df, x=temp_col, nbins=50, title="Temperature Distribution")
        st.plotly_chart(fig, use_container_width=True)


def create_heatmaps_tab(df: pd.DataFrame):
    """Create heatmaps analysis tab."""
    
    st.subheader("Temperature Heatmaps")
    
    temp_col = get_temperature_column(df)
    if not temp_col:
        st.error("No temperature column found in the dataset!")
        return
    
    if 'year' not in df.columns or 'month' not in df.columns:
        st.error("Year and month columns are required for heatmap analysis!")
        return
    
    # Heatmap type selection
    heatmap_type = st.selectbox(
        "Heatmap Type",
        ["Temperature", "Anomalies"],
        help="Choose between absolute temperature or temperature anomalies"
    )
    
    # Create heatmap data
    if heatmap_type == "Temperature":
        heatmap_data = df.pivot_table(
            values=temp_col,
            index='year',
            columns='month',
            aggfunc='mean'
        )
        title = "Temperature Heatmap (°C)"
        colorscale = 'RdYlBu_r'
    else:
        # Calculate anomalies
        monthly_climatology = df.groupby('month')[temp_col].mean()
        df_anomaly = df.copy()
        df_anomaly['anomaly'] = df_anomaly.apply(
            lambda row: row[temp_col] - monthly_climatology[row['month']], 
            axis=1
        )
        
        heatmap_data = df_anomaly.pivot_table(
            values='anomaly',
            index='year',
            columns='month',
            aggfunc='mean'
        )
        title = "Temperature Anomaly Heatmap (°C)"
        colorscale = 'RdBu_r'
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        y=heatmap_data.index,
        colorscale=colorscale,
        hoverongaps=False,
        hovertemplate='Year: %{y}<br>Month: %{x}<br>Temperature: %{z:.2f}°C<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Month",
        yaxis_title="Year",
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Heatmap statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Hottest Year", f"{heatmap_data.mean(axis=1).idxmax()}")
    
    with col2:
        st.metric("Coldest Year", f"{heatmap_data.mean(axis=1).idxmin()}")
    
    with col3:
        if heatmap_type == "Anomalies":
            max_anomaly = heatmap_data.max().max()
            st.metric("Max Anomaly", f"{max_anomaly:.2f}°C")
        else:
            max_temp = heatmap_data.max().max()
            st.metric("Max Temperature", f"{max_temp:.2f}°C")


def create_seasonal_analysis_tab(df: pd.DataFrame):
    """Create seasonal analysis tab."""
    
    st.subheader("Seasonal Temperature Analysis")
    
    temp_col = get_temperature_column(df)
    if not temp_col:
        st.error("No temperature column found in the dataset!")
        return
    
    if 'month' not in df.columns:
        st.error("Month column is required for seasonal analysis!")
        return
    
    # Define seasons
    season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                 3: 'Spring', 4: 'Spring', 5: 'Spring',
                 6: 'Summer', 7: 'Summer', 8: 'Summer',
                 9: 'Fall', 10: 'Fall', 11: 'Fall'}
    
    df_seasonal = df.copy()
    df_seasonal['season'] = df_seasonal['month'].map(season_map)
    
    # Monthly climatology
    monthly_stats = df.groupby('month')[temp_col].agg(['mean', 'std']).reset_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Monthly climatology plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=monthly_stats['month'],
            y=monthly_stats['mean'],
            mode='lines+markers',
            name='Average Temperature',
            line=dict(color='blue', width=3),
            marker=dict(size=8)
        ))
        
        # Add error bars
        fig.add_trace(go.Scatter(
            x=monthly_stats['month'],
            y=monthly_stats['mean'] + monthly_stats['std'],
            fill=None,
            mode='lines',
            line_color='rgba(0,0,255,0)',
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=monthly_stats['month'],
            y=monthly_stats['mean'] - monthly_stats['std'],
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
    
    with col2:
        # Seasonal box plot
        fig = px.box(
            df_seasonal, 
            x='season', 
            y=temp_col,
            title="Seasonal Temperature Distribution",
            category_orders={"season": ["Spring", "Summer", "Fall", "Winter"]}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Seasonal statistics
    st.subheader("Seasonal Statistics")
    
    seasonal_stats = df_seasonal.groupby('season')[temp_col].agg(['mean', 'std', 'min', 'max']).round(2)
    seasonal_stats.columns = ['Mean (°C)', 'Std Dev (°C)', 'Min (°C)', 'Max (°C)']
    
    st.dataframe(seasonal_stats, use_container_width=True)


def create_extreme_events_tab(df: pd.DataFrame):
    """Create extreme events analysis tab."""
    
    st.subheader("Extreme Temperature Events")
    
    temp_col = get_temperature_column(df)
    if not temp_col:
        st.error("No temperature column found in the dataset!")
        return
    
    # Threshold selection
    threshold_percentile = st.slider(
        "Extreme Event Threshold (Percentile)",
        min_value=90.0,
        max_value=99.9,
        value=95.0,
        step=0.1,
        help="Temperatures above/below this percentile are considered extreme"
    )
    
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
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Extreme Events", len(extreme_events))
    
    with col2:
        hot_events = len(extreme_events[extreme_events['event_type'] == 'Hot'])
        st.metric("Hot Extremes", hot_events)
    
    with col3:
        cold_events = len(extreme_events[extreme_events['event_type'] == 'Cold'])
        st.metric("Cold Extremes", cold_events)
    
    with col4:
        extreme_pct = (len(extreme_events) / len(df)) * 100
        st.metric("Extreme Event Rate", f"{extreme_pct:.2f}%")
    
    # Visualization
    if 'year' in df.columns:
        # Time series with extreme events highlighted
        fig = go.Figure()
        
        # All data
        fig.add_trace(go.Scatter(
            x=df['year'],
            y=df[temp_col],
            mode='markers',
            name='All Data',
            marker=dict(color='lightblue', size=4, opacity=0.6)
        ))
        
        # Hot extremes
        hot_extremes = extreme_events[extreme_events['event_type'] == 'Hot']
        if not hot_extremes.empty:
            fig.add_trace(go.Scatter(
                x=hot_extremes['year'],
                y=hot_extremes[temp_col],
                mode='markers',
                name='Hot Extremes',
                marker=dict(color='red', size=8, symbol='triangle-up')
            ))
        
        # Cold extremes
        cold_extremes = extreme_events[extreme_events['event_type'] == 'Cold']
        if not cold_extremes.empty:
            fig.add_trace(go.Scatter(
                x=cold_extremes['year'],
                y=cold_extremes[temp_col],
                mode='markers',
                name='Cold Extremes',
                marker=dict(color='blue', size=8, symbol='triangle-down')
            ))
        
        # Add threshold lines
        fig.add_hline(y=high_threshold, line_dash="dash", line_color="red", 
                     annotation_text=f"Hot Threshold ({high_threshold:.2f}°C)")
        fig.add_hline(y=low_threshold, line_dash="dash", line_color="blue",
                     annotation_text=f"Cold Threshold ({low_threshold:.2f}°C)")
        
        fig.update_layout(
            title="Extreme Temperature Events Over Time",
            xaxis_title="Year",
            yaxis_title="Temperature (°C)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        # Histogram with thresholds
        fig = px.histogram(df, x=temp_col, nbins=50, title="Temperature Distribution with Extreme Thresholds")
        fig.add_vline(x=high_threshold, line_dash="dash", line_color="red", 
                     annotation_text=f"Hot Threshold ({high_threshold:.2f}°C)")
        fig.add_vline(x=low_threshold, line_dash="dash", line_color="blue",
                     annotation_text=f"Cold Threshold ({low_threshold:.2f}°C)")
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Extreme events table
    if not extreme_events.empty:
        st.subheader("Recent Extreme Events")
        
        # Show most recent extreme events
        recent_extremes = extreme_events.nlargest(10, 'year' if 'year' in extreme_events.columns else extreme_events.index)
        
        display_cols = [col for col in ['year', 'month', temp_col, 'event_type'] if col in recent_extremes.columns]
        st.dataframe(recent_extremes[display_cols], use_container_width=True)


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