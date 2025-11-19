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

st.set_page_config(page_title="Country Analysis", page_icon="🌍", layout="wide")
configure_sidebar()

st.title("🌍 Country Analysis")
st.markdown("Global temperature analysis by country.")

data_source = DataSource()
country_df = data_source.load_parquet('country.parquet')

if country_df is not None and not country_df.empty:
    show_dat-info(country_df, "Country Dataset")
    
    # Year filter
    min_year = int(country_df['year'].min())
    max_year = int(country_df['year'].max())
    selected_year = st.slider("Select Year", min_year, max_year, max_year)
    
    year_data = country_df[country_df['year'] == selected_year]
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Countries", year_data['country'].nunique())
    with col2:
        st.metric("Global Avg", f"{year_data['avg_temperature'].mean():.2f}°C")
    with col3:
        hottest = year_data.loc[year_data['avg_temperature'].idxmax(), 'country']
        st.metric("Hottest Country", hottest)
    with col4:
        coldest = year_data.loc[year_data['avg_temperature'].idxmin(), 'country']
        st.metric("Coldest Country", coldest)
    
    # Regional map
    st.markdown(f"#### Global Temperature Map - {selected_year}")
    fig = px.choropleth(
        year_data,
        locations="country_code",
        color="avg_temperature",
        hover_name="country",
        color_continuous_scale=px.colors.sequential.Plasma,
        title=f"Global Temperature Distribution - {selected_year}"
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Regional comparison
    st.markdown("#### Country Comparison")
    
    fig = px.bar(
        year_data.sort_values('avg_temperature', ascending=False).head(50),
        x='country',
        y='avg_temperature',
        title=f"Top 50 Hottest Countries - {selected_year}",
        labels={'avg_temperature': 'Temperature (°C)', 'country': 'Country'}
    )
    
    fig.update_xaxes(tickangle=45)
    fig.update_layout(height=500)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Regional trends
    st.markdown("#### Country Temperature Trends")
    
    countries = sorted(country_df['country'].unique())
    selected_countries = st.multiselect(
        "Select Countries to Compare",
        countries,
        default=countries[:4]
    )
    
    if selected_countries:
        trend_data = country_df[country_df['country'].isin(selected_countries)]
        
        fig = px.line(
            trend_data,
            x='year',
            y='avg_temperature',
            color='country',
            title="Country Temperature Evolution",
            labels={'year': 'Year', 'avg_temperature': 'Temperature (°C)'}
        )
        
        fig.update_layout(height=500, hovermode='x unified')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics table
        st.markdown("#### Country Statistics")
        
        stats = trend_data.groupby('country').agg({
            'avg_temperature': ['mean', 'std', 'min', 'max']
        }).round(2)
        
        stats.columns = ['Mean (°C)', 'Std Dev (°C)', 'Min (°C)', 'Max (°C)']
        
        st.dataframe(stats, use_container_width=True)

else:
    st.error("❌ Failed to load country.parquet")
