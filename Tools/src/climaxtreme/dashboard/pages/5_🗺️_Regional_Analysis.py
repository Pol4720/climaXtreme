"""
Regional Analysis Page - 16 geographic regions
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

st.set_page_config(page_title="Regional Analysis", page_icon="🗺️", layout="wide")
configure_sidebar()

st.title("🗺️ Regional Analysis")
st.markdown("18 geographic regions worldwide")

data_source = DataSource()
regional_df = data_source.load_parquet('regional.parquet')

if regional_df is not None and not regional_df.empty:
    show_data_info(regional_df, "Regional Dataset")
    
    # Year filter
    min_year = int(regional_df['year'].min())
    max_year = int(regional_df['year'].max())
    selected_year = st.slider("Select Year", min_year, max_year, max_year)
    
    year_data = regional_df[regional_df['year'] == selected_year]
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Regions", year_data['region'].nunique())
    with col2:
        st.metric("Global Avg", f"{year_data['avg_temperature'].mean():.2f}°C")
    with col3:
        hottest = year_data.loc[year_data['avg_temperature'].idxmax(), 'region']
        st.metric("Hottest Region", hottest[:15])
    with col4:
        coldest = year_data.loc[year_data['avg_temperature'].idxmin(), 'region']
        st.metric("Coldest Region", coldest[:15])
    
    # Regional map
    st.markdown(f"#### Global Temperature Map - {selected_year}")
    
    # Add approximate coordinates for regions (simplified)
    region_coords = {
        'Northern Europe': (60, 10),
        'Central Europe': (50, 10),
        'Southern Europe': (40, 15),
        'Northern Asia': (65, 100),
        'Central Asia': (45, 65),
        'South Asia': (20, 80),
        'East Asia': (35, 110),
        'Northern Africa': (20, 10),
        'Central Africa': (0, 20),
        'Southern Africa': (-25, 25),
        'Northern North America': (55, -100),
        'Central North America': (40, -95),
        'Caribbean & Central America': (15, -80),
        'Northern South America': (0, -60),
        'Central South America': (-15, -60),
        'Southern South America': (-35, -65),
        'Northern Oceania': (5, 160),
        'Southern Oceania': (-30, 145),
        'Antarctica': (-75, 0)
    }
    
    year_data['lat'] = year_data['region'].map(lambda x: region_coords.get(x, (0, 0))[0])
    year_data['lon'] = year_data['region'].map(lambda x: region_coords.get(x, (0, 0))[1])
    
    fig = px.scatter_geo(
        year_data,
        lat='lat',
        lon='lon',
        size='avg_temperature',
        color='avg_temperature',
        hover_name='region',
        hover_data={'lat': False, 'lon': False, 'avg_temperature': ':.2f'},
        color_continuous_scale='RdYlBu_r',
        size_max=40,
        title=f"Regional Temperature Distribution - {selected_year}"
    )
    
    fig.update_geos(projection_type="natural earth", showcountries=True)
    fig.update_layout(height=600)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Regional comparison
    st.markdown("#### Regional Comparison")
    
    fig = px.bar(
        year_data.sort_values('avg_temperature', ascending=False),
        x='region',
        y='avg_temperature',
        color='continent',
        title=f"Temperature by Region - {selected_year}",
        labels={'avg_temperature': 'Temperature (°C)', 'region': 'Region'}
    )
    
    fig.update_xaxes(tickangle=45)
    fig.update_layout(height=500)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Regional trends
    st.markdown("#### Regional Temperature Trends")
    
    regions = sorted(regional_df['region'].unique())
    selected_regions = st.multiselect(
        "Select Regions to Compare",
        regions,
        default=regions[:4]
    )
    
    if selected_regions:
        trend_data = regional_df[regional_df['region'].isin(selected_regions)]
        
        fig = px.line(
            trend_data,
            x='year',
            y='avg_temperature',
            color='region',
            title="Regional Temperature Evolution",
            labels={'year': 'Year', 'avg_temperature': 'Temperature (°C)'}
        )
        
        fig.update_layout(height=500, hovermode='x unified')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics table
        st.markdown("#### Regional Statistics")
        
        stats = trend_data.groupby('region').agg({
            'avg_temperature': ['mean', 'std', 'min', 'max'],
            'record_count': 'sum'
        }).round(2)
        
        stats.columns = ['Mean (°C)', 'Std Dev (°C)', 'Min (°C)', 'Max (°C)', 'Total Records']
        
        st.dataframe(stats, use_container_width=True)

else:
    st.error("❌ Failed to load regional.parquet")
