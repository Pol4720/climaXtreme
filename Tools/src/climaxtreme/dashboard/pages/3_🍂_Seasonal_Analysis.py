"""
Seasonal Analysis Page
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

try:
    from climaxtreme.dashboard.utils import DataSource, configure_sidebar, show_data_info
except ImportError:
    _src_dir = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(_src_dir))
    from climaxtreme.dashboard.utils import DataSource, configure_sidebar, show_data_info

st.set_page_config(page_title="Seasonal Analysis", page_icon="🍂", layout="wide")
configure_sidebar()

st.title("🍂 Seasonal Analysis")
data_source = DataSource()

seasonal_df = data_source.load_parquet('seasonal.parquet')

if seasonal_df is not None and not seasonal_df.empty:
    show_data_info(seasonal_df, "Seasonal Dataset Information")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        min_year = int(seasonal_df['year'].min())
        max_year = int(seasonal_df['year'].max())
        year_range = st.slider("Year Range", min_year, max_year, (max_year-30, max_year))
    with col2:
        countries = sorted(seasonal_df['Country'].unique())
        country = st.selectbox("Country", countries)
    with col3:
        cities = sorted(seasonal_df[seasonal_df['Country'] == country]['City'].unique())
        city = st.selectbox("City", cities)
    
    # Filter
    filtered = seasonal_df[
        (seasonal_df['year'] >= year_range[0]) &
        (seasonal_df['year'] <= year_range[1]) &
        (seasonal_df['Country'] == country) &
        (seasonal_df['City'] == city)
    ]
    
    if not filtered.empty:
        # Seasonal comparison
        st.markdown(f"#### Seasonal Temperature Patterns - {city}, {country}")
        
        season_avg = filtered.groupby('season')['avg_temperature'].mean().reset_index()
        
        fig = px.bar(
            season_avg,
            x='season',
            y='avg_temperature',
            title="Average Temperature by Season",
            color='avg_temperature',
            color_continuous_scale='RdYlBu_r',
            labels={'avg_temperature': 'Temperature (°C)', 'season': 'Season'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Seasonal trends over time
        st.markdown("#### Seasonal Trends Over Time")
        
        fig = px.line(
            filtered,
            x='year',
            y='avg_temperature',
            color='season',
            title="Temperature Evolution by Season",
            labels={'year': 'Year', 'avg_temperature': 'Temperature (°C)'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Box plot
        st.markdown("#### Temperature Distribution by Season")
        
        fig = px.box(
            filtered,
            x='season',
            y='avg_temperature',
            color='season',
            title="Temperature Distribution",
            labels={'avg_temperature': 'Temperature (°C)'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data for selected filters")
else:
    st.error("❌ Failed to load seasonal.parquet")
