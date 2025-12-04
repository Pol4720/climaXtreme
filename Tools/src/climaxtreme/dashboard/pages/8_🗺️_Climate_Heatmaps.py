"""
🗺️ Real-Time Climate Heatmaps Page
Interactive global heatmaps with synthetic climate data.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

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
            # Try alternative path
            df = data_source.load_parquet('synthetic/synthetic_hourly.parquet')
        return df
    except Exception as e:
        st.error(f"Error loading synthetic data: {e}")
        return None


def create_global_heatmap(df: pd.DataFrame, variable: str, title: str) -> go.Figure:
    """
    Create an interactive global heatmap.
    
    Args:
        df: DataFrame with lat_decimal, lon_decimal, and variable
        variable: Column name to visualize
        title: Chart title
    """
    # Aggregate by location
    agg_df = df.groupby(['lat_decimal', 'lon_decimal', 'City', 'Country']).agg({
        variable: 'mean'
    }).reset_index()
    
    # Determine color scale based on variable
    if 'temperature' in variable.lower():
        color_scale = 'RdYlBu_r'
        color_label = 'Temperature (°C)'
    elif 'rain' in variable.lower():
        color_scale = 'Blues'
        color_label = 'Precipitation (mm)'
    elif 'wind' in variable.lower():
        color_scale = 'Viridis'
        color_label = 'Wind Speed (km/h)'
    elif 'humidity' in variable.lower():
        color_scale = 'Teal'
        color_label = 'Humidity (%)'
    else:
        color_scale = 'Plasma'
        color_label = variable
    
    fig = px.scatter_geo(
        agg_df,
        lat='lat_decimal',
        lon='lon_decimal',
        color=variable,
        hover_name='City',
        hover_data={
            'Country': True,
            variable: ':.2f',
            'lat_decimal': ':.2f',
            'lon_decimal': ':.2f'
        },
        color_continuous_scale=color_scale,
        title=title,
        projection='natural earth'
    )
    
    fig.update_layout(
        geo=dict(
            showland=True,
            landcolor='rgb(243, 243, 243)',
            countrycolor='rgb(204, 204, 204)',
            showocean=True,
            oceancolor='rgb(230, 245, 255)',
            showlakes=True,
            lakecolor='rgb(200, 230, 255)',
            showcountries=True,
            coastlinecolor='rgb(150, 150, 150)'
        ),
        height=600,
        margin=dict(l=0, r=0, t=50, b=0),
        coloraxis_colorbar=dict(title=color_label)
    )
    
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    
    return fig


def create_density_heatmap(df: pd.DataFrame, variable: str, title: str) -> go.Figure:
    """Create a density heatmap using hexagonal binning simulation."""
    fig = px.density_mapbox(
        df,
        lat='lat_decimal',
        lon='lon_decimal',
        z=variable,
        radius=20,
        center=dict(lat=20, lon=0),
        zoom=1,
        mapbox_style='carto-positron',
        title=title,
        color_continuous_scale='Turbo'
    )
    
    fig.update_layout(height=600, margin=dict(l=0, r=0, t=50, b=0))
    
    return fig


def create_animated_heatmap(df: pd.DataFrame, variable: str) -> go.Figure:
    """Create an animated heatmap over time."""
    # Aggregate by month and location
    df['month_year'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2)
    
    agg_df = df.groupby(['month_year', 'lat_decimal', 'lon_decimal', 'City']).agg({
        variable: 'mean'
    }).reset_index()
    
    # Sort by month_year
    agg_df = agg_df.sort_values('month_year')
    
    fig = px.scatter_geo(
        agg_df,
        lat='lat_decimal',
        lon='lon_decimal',
        color=variable,
        hover_name='City',
        animation_frame='month_year',
        color_continuous_scale='RdYlBu_r',
        title=f'Temporal Evolution of {variable}',
        projection='natural earth'
    )
    
    fig.update_layout(
        geo=dict(showland=True, landcolor='rgb(243, 243, 243)'),
        height=600,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    fig.update_traces(marker=dict(size=6, opacity=0.7))
    
    return fig


def main():
    st.set_page_config(
        page_title="Climate Heatmaps - climaXtreme",
        page_icon="🗺️",
        layout="wide"
    )
    
    configure_sidebar()
    
    st.title("🗺️ Real-Time Climate Heatmaps")
    st.markdown("""
    Interactive global visualization of climate variables using synthetic data.
    Explore temperature, precipitation, wind, and other meteorological patterns.
    """)
    
    # Load data
    data_source = DataSource()
    
    with st.spinner("Loading synthetic climate data..."):
        df = load_synthetic_data(data_source)
    
    if df is None or df.empty:
        st.warning("""
        ⚠️ **No synthetic data found!**
        
        Please generate synthetic data first using:
        ```bash
        climaxtreme generate-synthetic --input-path DATA/GlobalLandTemperaturesByCity.csv --output-path DATA/synthetic
        ```
        
        Or configure the correct data path in the sidebar.
        """)
        
        # Show demo with sample data
        st.markdown("---")
        st.subheader("📊 Demo Mode")
        st.info("Showing demo visualization with sample data")
        
        # Generate minimal demo data
        np.random.seed(42)
        demo_df = pd.DataFrame({
            'lat_decimal': np.random.uniform(-60, 70, 500),
            'lon_decimal': np.random.uniform(-180, 180, 500),
            'temperature_hourly': np.random.uniform(-20, 40, 500),
            'City': [f'City_{i}' for i in range(500)],
            'Country': ['Demo'] * 500
        })
        
        fig = create_global_heatmap(demo_df, 'temperature_hourly', '🌡️ Demo Temperature Heatmap')
        st.plotly_chart(fig, use_container_width=True)
        return
    
    # Show data info
    st.success(f"✅ Loaded {len(df):,} synthetic records")
    
    with st.expander("📊 Dataset Info", expanded=False):
        show_data_info(df, "Synthetic Climate Data")
    
    # Variable selection
    st.markdown("---")
    st.subheader("⚙️ Heatmap Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Available numeric columns for heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        display_vars = [c for c in numeric_cols if c not in ['year', 'month', 'day', 'hour', 'day_of_week']]
        
        variable = st.selectbox(
            "Select Variable",
            options=display_vars,
            index=display_vars.index('temperature_hourly') if 'temperature_hourly' in display_vars else 0,
            help="Choose the climate variable to visualize"
        )
    
    with col2:
        map_type = st.selectbox(
            "Map Type",
            options=["Scatter Points", "Density Map", "Animated (by month)"],
            help="Choose visualization style"
        )
    
    with col3:
        # Time filter
        if 'year' in df.columns:
            years = sorted(df['year'].unique())
            selected_years = st.multiselect(
                "Filter Years",
                options=years,
                default=years[-5:] if len(years) > 5 else years,
                help="Select years to include"
            )
            if selected_years:
                df = df[df['year'].isin(selected_years)]
    
    # Additional filters
    col4, col5 = st.columns(2)
    
    with col4:
        if 'climate_zone' in df.columns:
            zones = ['All'] + df['climate_zone'].unique().tolist()
            selected_zone = st.selectbox("Climate Zone", zones)
            if selected_zone != 'All':
                df = df[df['climate_zone'] == selected_zone]
    
    with col5:
        if 'Country' in df.columns:
            countries = ['All'] + sorted(df['Country'].unique().tolist())
            selected_country = st.selectbox("Country", countries)
            if selected_country != 'All':
                df = df[df['Country'] == selected_country]
    
    # Sample data if too large
    max_points = 50000
    if len(df) > max_points:
        st.info(f"📉 Sampling {max_points:,} points from {len(df):,} for visualization performance")
        df = df.sample(n=max_points, random_state=42)
    
    # Create visualization
    st.markdown("---")
    
    if map_type == "Scatter Points":
        fig = create_global_heatmap(df, variable, f'🌍 Global {variable.replace("_", " ").title()} Heatmap')
        st.plotly_chart(fig, use_container_width=True)
        
    elif map_type == "Density Map":
        try:
            fig = create_density_heatmap(df, variable, f'🔥 Density Map: {variable.replace("_", " ").title()}')
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Density map requires mapbox token. Showing scatter plot instead.")
            fig = create_global_heatmap(df, variable, f'🌍 Global {variable.replace("_", " ").title()} Heatmap')
            st.plotly_chart(fig, use_container_width=True)
            
    elif map_type == "Animated (by month)":
        with st.spinner("Creating animation... This may take a moment."):
            # Limit data for animation
            if len(df) > 10000:
                df = df.sample(n=10000, random_state=42)
            fig = create_animated_heatmap(df, variable)
            st.plotly_chart(fig, use_container_width=True)
    
    # Statistics summary
    st.markdown("---")
    st.subheader("📈 Variable Statistics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Mean", f"{df[variable].mean():.2f}")
    with col2:
        st.metric("Std Dev", f"{df[variable].std():.2f}")
    with col3:
        st.metric("Min", f"{df[variable].min():.2f}")
    with col4:
        st.metric("Max", f"{df[variable].max():.2f}")
    with col5:
        st.metric("Median", f"{df[variable].median():.2f}")
    
    # Distribution plot
    st.subheader("📊 Value Distribution")
    
    fig_dist = px.histogram(
        df, 
        x=variable, 
        nbins=50,
        title=f'Distribution of {variable.replace("_", " ").title()}',
        color_discrete_sequence=['#1f77b4']
    )
    fig_dist.update_layout(
        xaxis_title=variable.replace("_", " ").title(),
        yaxis_title="Count",
        height=300
    )
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Climate zone comparison
    if 'climate_zone' in df.columns:
        st.subheader("🌐 Comparison by Climate Zone")
        
        zone_stats = df.groupby('climate_zone')[variable].agg(['mean', 'std', 'count']).reset_index()
        zone_stats.columns = ['Climate Zone', 'Mean', 'Std Dev', 'Count']
        
        fig_zones = px.bar(
            zone_stats,
            x='Climate Zone',
            y='Mean',
            error_y='Std Dev',
            title=f'{variable.replace("_", " ").title()} by Climate Zone',
            color='Mean',
            color_continuous_scale='RdYlBu_r'
        )
        fig_zones.update_layout(height=400)
        st.plotly_chart(fig_zones, use_container_width=True)


if __name__ == "__main__":
    main()
