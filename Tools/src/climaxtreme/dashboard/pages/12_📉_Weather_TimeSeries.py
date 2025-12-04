"""
📉 Weather Time Series Page
Interactive time series visualization for temperature, precipitation, and other variables.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List

try:
    from climaxtreme.dashboard.utils import configure_sidebar, DataSource, show_data_info
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from climaxtreme.dashboard.utils import configure_sidebar, DataSource, show_data_info


def load_synthetic_data(data_source: DataSource) -> Optional[pd.DataFrame]:
    """Load synthetic hourly data."""
    try:
        df = data_source.load_parquet('synthetic_hourly.parquet')
        if df is None:
            df = data_source.load_parquet('synthetic/synthetic_hourly.parquet')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


def create_multi_variable_timeseries(
    df: pd.DataFrame, 
    variables: List[str],
    time_col: str = 'timestamp'
) -> go.Figure:
    """Create multi-variable time series chart with subplots."""
    n_vars = len(variables)
    
    fig = make_subplots(
        rows=n_vars,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[v.replace('_', ' ').title() for v in variables]
    )
    
    colors = px.colors.qualitative.Set2
    
    for i, var in enumerate(variables):
        fig.add_trace(
            go.Scatter(
                x=df[time_col],
                y=df[var],
                mode='lines',
                name=var.replace('_', ' ').title(),
                line=dict(color=colors[i % len(colors)], width=1),
                fill='tozeroy' if 'rain' in var.lower() else None
            ),
            row=i + 1, col=1
        )
    
    fig.update_layout(
        height=200 * n_vars,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02)
    )
    
    fig.update_xaxes(title_text="Time", row=n_vars, col=1)
    
    return fig


def create_comparison_chart(
    df: pd.DataFrame,
    variable: str,
    group_by: str,
    time_col: str = 'timestamp'
) -> go.Figure:
    """Create comparison chart grouped by category."""
    # Aggregate by time period and group
    df = df.copy()
    df['date'] = pd.to_datetime(df[time_col]).dt.date
    
    agg_df = df.groupby(['date', group_by])[variable].mean().reset_index()
    
    fig = px.line(
        agg_df,
        x='date',
        y=variable,
        color=group_by,
        title=f'{variable.replace("_", " ").title()} by {group_by.replace("_", " ").title()}',
        labels={variable: variable.replace('_', ' ').title()}
    )
    
    fig.update_layout(height=500)
    
    return fig


def create_seasonal_pattern_chart(df: pd.DataFrame, variable: str) -> go.Figure:
    """Create seasonal pattern analysis chart."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'By Month (Average)',
            'By Hour of Day',
            'By Day of Week',
            'Monthly Distribution'
        )
    )
    
    # By month
    monthly = df.groupby('month')[variable].mean().reset_index()
    fig.add_trace(
        go.Bar(x=monthly['month'], y=monthly[variable], marker_color='#3498DB'),
        row=1, col=1
    )
    
    # By hour
    if 'hour' in df.columns:
        hourly = df.groupby('hour')[variable].mean().reset_index()
        fig.add_trace(
            go.Scatter(
                x=hourly['hour'], y=hourly[variable],
                mode='lines+markers', marker_color='#E74C3C'
            ),
            row=1, col=2
        )
    
    # By day of week
    if 'day_of_week' in df.columns:
        daily = df.groupby('day_of_week')[variable].mean().reset_index()
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        fig.add_trace(
            go.Bar(
                x=[days[int(d)-1] if d <= 7 else str(d) for d in daily['day_of_week']], 
                y=daily[variable], 
                marker_color='#27AE60'
            ),
            row=2, col=1
        )
    
    # Monthly distribution (box plot)
    months_data = []
    for m in range(1, 13):
        month_data = df[df['month'] == m][variable].dropna().tolist()
        if month_data:
            months_data.append(go.Box(y=month_data, name=str(m), marker_color='#9B59B6'))
    
    for box in months_data:
        fig.add_trace(box, row=2, col=2)
    
    fig.update_layout(height=600, showlegend=False)
    fig.update_xaxes(title_text='Month', row=1, col=1)
    fig.update_xaxes(title_text='Hour', row=1, col=2)
    fig.update_xaxes(title_text='Day', row=2, col=1)
    fig.update_xaxes(title_text='Month', row=2, col=2)
    
    return fig


def create_anomaly_detection_chart(df: pd.DataFrame, variable: str) -> go.Figure:
    """Create chart highlighting anomalies in time series."""
    df = df.copy()
    
    # Calculate rolling statistics
    window = 24 * 7  # 1 week for hourly data
    df['rolling_mean'] = df[variable].rolling(window=window, center=True, min_periods=1).mean()
    df['rolling_std'] = df[variable].rolling(window=window, center=True, min_periods=1).std()
    
    # Define anomaly bounds
    df['upper_bound'] = df['rolling_mean'] + 2 * df['rolling_std']
    df['lower_bound'] = df['rolling_mean'] - 2 * df['rolling_std']
    
    # Identify anomalies
    df['is_anomaly'] = (df[variable] > df['upper_bound']) | (df[variable] < df['lower_bound'])
    
    fig = go.Figure()
    
    # Confidence band
    fig.add_trace(go.Scatter(
        x=list(df['timestamp']) + list(df['timestamp'][::-1]),
        y=list(df['upper_bound']) + list(df['lower_bound'][::-1]),
        fill='toself',
        fillcolor='rgba(52, 152, 219, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Normal Range (±2σ)'
    ))
    
    # Main series
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df[variable],
        mode='lines',
        line=dict(color='#3498DB', width=1),
        name='Observed'
    ))
    
    # Rolling mean
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['rolling_mean'],
        mode='lines',
        line=dict(color='#2C3E50', width=2, dash='dash'),
        name='Rolling Mean'
    ))
    
    # Anomalies
    anomalies = df[df['is_anomaly']]
    fig.add_trace(go.Scatter(
        x=anomalies['timestamp'],
        y=anomalies[variable],
        mode='markers',
        marker=dict(color='#E74C3C', size=8, symbol='x'),
        name='Anomalies'
    ))
    
    fig.update_layout(
        title=f'Anomaly Detection: {variable.replace("_", " ").title()}',
        xaxis_title='Time',
        yaxis_title=variable.replace('_', ' ').title(),
        height=500
    )
    
    return fig


def create_correlation_heatmap(df: pd.DataFrame, variables: List[str]) -> go.Figure:
    """Create correlation heatmap between variables."""
    corr_matrix = df[variables].corr()
    
    fig = px.imshow(
        corr_matrix,
        labels=dict(color="Correlation"),
        x=[v.replace('_', ' ').title() for v in variables],
        y=[v.replace('_', ' ').title() for v in variables],
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        title='Variable Correlation Matrix'
    )
    
    fig.update_layout(height=500)
    
    return fig


def main():
    st.set_page_config(
        page_title="Weather Time Series - climaXtreme",
        page_icon="📉",
        layout="wide"
    )
    
    configure_sidebar()
    
    st.title("📉 Weather Time Series Analysis")
    st.markdown("""
    Interactive visualization of temperature, precipitation, and other meteorological 
    time series from synthetic climate data.
    """)
    
    # Load data
    data_source = DataSource()
    
    with st.spinner("Loading time series data..."):
        df = load_synthetic_data(data_source)
    
    if df is None or df.empty:
        st.warning("""
        ⚠️ **No synthetic data found!**
        
        Please generate synthetic data first:
        ```bash
        climaxtreme generate-synthetic --input-path DATA/GlobalLandTemperaturesByCity.csv --output-path DATA/synthetic
        ```
        """)
        
        # Demo mode
        st.markdown("---")
        st.subheader("📊 Demo Mode")
        
        np.random.seed(42)
        n = 24 * 365  # 1 year hourly
        
        dates = pd.date_range('2024-01-01', periods=n, freq='h')
        
        # Generate realistic patterns
        seasonal = 15 * np.sin(2 * np.pi * np.arange(n) / (24 * 365))
        diurnal = 8 * np.sin(2 * np.pi * (np.arange(n) - 6) / 24)
        trend = np.linspace(0, 2, n)
        noise = np.random.normal(0, 2, n)
        
        temp = 20 + seasonal + diurnal + trend + noise
        
        df = pd.DataFrame({
            'timestamp': dates,
            'year': dates.year,
            'month': dates.month,
            'day': dates.day,
            'hour': dates.hour,
            'day_of_week': dates.dayofweek + 1,
            'temperature_hourly': temp,
            'rain_mm': np.maximum(0, np.random.exponential(3, n) * (np.random.random(n) < 0.2)),
            'wind_speed_kmh': np.abs(np.random.normal(15, 8, n)),
            'humidity_pct': np.clip(60 + np.random.normal(0, 15, n), 0, 100),
            'City': ['Demo City'] * n,
            'Country': ['Demo Country'] * n
        })
        
        st.info("Showing demo data for illustration")
    
    # Data info
    st.success(f"✅ Loaded {len(df):,} records")
    
    # Ensure timestamp column
    if 'timestamp' not in df.columns:
        if 'date' in df.columns and 'hour' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['hour'], unit='h')
        else:
            df['timestamp'] = pd.date_range('2020-01-01', periods=len(df), freq='h')
    
    # Sidebar filters
    st.sidebar.markdown("### 🎛️ Time Series Filters")
    
    # Variable selection
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    time_vars = ['year', 'month', 'day', 'hour', 'day_of_week']
    available_vars = [c for c in numeric_cols if c not in time_vars]
    
    selected_vars = st.sidebar.multiselect(
        "Variables to Plot",
        options=available_vars,
        default=['temperature_hourly', 'rain_mm'] if 'temperature_hourly' in available_vars else available_vars[:2]
    )
    
    # Time range
    if 'year' in df.columns:
        years = sorted(df['year'].unique())
        year_range = st.sidebar.slider(
            "Year Range",
            min_value=int(min(years)),
            max_value=int(max(years)),
            value=(int(min(years)), int(max(years)))
        )
        df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
    
    # Location filter
    if 'City' in df.columns:
        cities = ['All'] + sorted(df['City'].unique().tolist())
        selected_city = st.sidebar.selectbox("City", cities)
        if selected_city != 'All':
            df = df[df['City'] == selected_city]
    
    # Sample if too large
    max_points = 50000
    if len(df) > max_points:
        st.sidebar.info(f"Sampling {max_points:,} points for performance")
        df = df.sample(n=max_points, random_state=42).sort_values('timestamp')
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Time Series",
        "🔄 Seasonal Patterns",
        "🔍 Anomaly Detection",
        "📊 Comparisons",
        "🔗 Correlations"
    ])
    
    with tab1:
        st.subheader("Multi-Variable Time Series")
        
        if selected_vars:
            fig = create_multi_variable_timeseries(df, selected_vars, 'timestamp')
            st.plotly_chart(fig, use_container_width=True)
            
            # Quick stats
            st.markdown("### Quick Statistics")
            stats_df = df[selected_vars].describe().T
            stats_df.columns = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
            st.dataframe(stats_df.round(2), use_container_width=True)
        else:
            st.info("Select variables from the sidebar to visualize")
    
    with tab2:
        st.subheader("Seasonal Pattern Analysis")
        
        if selected_vars:
            var_for_season = st.selectbox(
                "Variable for Seasonal Analysis",
                options=selected_vars,
                key='seasonal_var'
            )
            
            fig_seasonal = create_seasonal_pattern_chart(df, var_for_season)
            st.plotly_chart(fig_seasonal, use_container_width=True)
        else:
            st.info("Select variables to analyze seasonal patterns")
    
    with tab3:
        st.subheader("Anomaly Detection")
        
        if selected_vars:
            var_for_anomaly = st.selectbox(
                "Variable for Anomaly Detection",
                options=selected_vars,
                key='anomaly_var'
            )
            
            # Limit data for anomaly detection
            anomaly_df = df.head(5000).copy()
            
            fig_anomaly = create_anomaly_detection_chart(anomaly_df, var_for_anomaly)
            st.plotly_chart(fig_anomaly, use_container_width=True)
            
            # Anomaly statistics
            st.markdown("### Anomaly Statistics")
            
            window = 24 * 7
            anomaly_df['rolling_mean'] = anomaly_df[var_for_anomaly].rolling(window=window, center=True, min_periods=1).mean()
            anomaly_df['rolling_std'] = anomaly_df[var_for_anomaly].rolling(window=window, center=True, min_periods=1).std()
            anomaly_df['z_score'] = (anomaly_df[var_for_anomaly] - anomaly_df['rolling_mean']) / anomaly_df['rolling_std']
            
            n_anomalies = len(anomaly_df[abs(anomaly_df['z_score']) > 2])
            pct_anomalies = n_anomalies / len(anomaly_df) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Anomalies", n_anomalies)
            with col2:
                st.metric("Anomaly Rate", f"{pct_anomalies:.2f}%")
            with col3:
                max_anomaly = anomaly_df['z_score'].abs().max()
                st.metric("Max Z-Score", f"{max_anomaly:.2f}")
        else:
            st.info("Select variables to detect anomalies")
    
    with tab4:
        st.subheader("Comparative Analysis")
        
        if selected_vars:
            var_for_compare = st.selectbox(
                "Variable to Compare",
                options=selected_vars,
                key='compare_var'
            )
            
            # Available grouping columns
            group_cols = [c for c in ['climate_zone', 'season', 'Country', 'event_type'] if c in df.columns]
            
            if group_cols:
                group_by = st.selectbox(
                    "Group By",
                    options=group_cols
                )
                
                fig_compare = create_comparison_chart(df, var_for_compare, group_by, 'timestamp')
                st.plotly_chart(fig_compare, use_container_width=True)
            else:
                st.info("No categorical columns available for comparison")
        else:
            st.info("Select variables to compare")
    
    with tab5:
        st.subheader("Variable Correlations")
        
        if len(selected_vars) >= 2:
            fig_corr = create_correlation_heatmap(df, selected_vars)
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Scatter matrix
            st.markdown("### Scatter Plot Matrix")
            
            if len(selected_vars) <= 5:
                scatter_df = df[selected_vars].sample(n=min(1000, len(df)), random_state=42)
                fig_scatter = px.scatter_matrix(
                    scatter_df,
                    dimensions=selected_vars,
                    title='Pairwise Scatter Plots'
                )
                fig_scatter.update_layout(height=600)
                fig_scatter.update_traces(diagonal_visible=False, marker=dict(size=3, opacity=0.5))
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info("Select up to 5 variables for scatter matrix")
        else:
            st.info("Select at least 2 variables to analyze correlations")
    
    # Download option
    st.markdown("---")
    if st.button("📥 Download Filtered Data"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="climate_timeseries.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()
