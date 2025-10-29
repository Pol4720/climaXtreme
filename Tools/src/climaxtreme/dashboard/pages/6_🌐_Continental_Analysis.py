"""
Continental Analysis Page - 7 continents
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

st.set_page_config(page_title="Continental Analysis", page_icon="🌐", layout="wide")
configure_sidebar()

st.title("🌐 Continental Analysis")
st.markdown("Global overview across 7 continents")

data_source = DataSource()
continental_df = data_source.load_parquet('continental.parquet')

if continental_df is not None and not continental_df.empty:
    show_data_info(continental_df, "Continental Dataset")
    
    # Year filter
    min_year = int(continental_df['year'].min())
    max_year = int(continental_df['year'].max())
    selected_year = st.slider("Select Year", min_year, max_year, max_year)
    
    year_data = continental_df[continental_df['year'] == selected_year]
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Continents", len(year_data))
    with col2:
        st.metric("Global Avg", f"{year_data['avg_temperature'].mean():.2f}°C")
    with col3:
        hottest = year_data.loc[year_data['avg_temperature'].idxmax(), 'continent']
        st.metric("Hottest", hottest)
    with col4:
        coldest = year_data.loc[year_data['avg_temperature'].idxmin(), 'continent']
        st.metric("Coldest", coldest)
    
    # Continental map
    st.markdown(f"#### Global Continental Temperature - {selected_year}")
    
    # Approximate coordinates for continents
    continent_coords = {
        'Europe': (50, 10),
        'Asia': (40, 100),
        'Africa': (0, 20),
        'North America': (45, -100),
        'South America': (-15, -60),
        'Oceania': (-25, 135),
        'Antarctica': (-75, 0)
    }
    
    year_data['lat'] = year_data['continent'].map(lambda x: continent_coords.get(x, (0, 0))[0])
    year_data['lon'] = year_data['continent'].map(lambda x: continent_coords.get(x, (0, 0))[1])
    
    fig = px.scatter_geo(
        year_data,
        lat='lat',
        lon='lon',
        size='avg_temperature',
        color='avg_temperature',
        hover_name='continent',
        hover_data={'lat': False, 'lon': False, 'avg_temperature': ':.2f'},
        color_continuous_scale='RdYlBu_r',
        size_max=60,
        text='continent',
        title=f"Continental Temperature Distribution - {selected_year}"
    )
    
    fig.update_traces(textposition='top center')
    fig.update_geos(projection_type="natural earth", showcountries=True)
    fig.update_layout(height=600)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Continental comparison
    st.markdown("#### Continental Comparison")
    
    fig = px.bar(
        year_data.sort_values('avg_temperature', ascending=False),
        x='continent',
        y='avg_temperature',
        color='avg_temperature',
        color_continuous_scale='RdYlBu_r',
        title=f"Temperature by Continent - {selected_year}",
        labels={'avg_temperature': 'Temperature (°C)', 'continent': 'Continent'}
    )
    
    fig.update_layout(height=400, showlegend=False)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Continental trends
    st.markdown("#### Continental Temperature Evolution")
    
    fig = px.line(
        continental_df,
        x='year',
        y='avg_temperature',
        color='continent',
        title="Temperature Trends by Continent",
        labels={'year': 'Year', 'avg_temperature': 'Temperature (°C)'}
    )
    
    fig.update_layout(height=500, hovermode='x unified')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Temperature change analysis
    st.markdown("#### Temperature Change Analysis")
    
    # Calculate change from first to last year
    first_year = continental_df['year'].min()
    last_year = continental_df['year'].max()
    
    # Get data for first and last year
    first_data = continental_df[continental_df['year'] == first_year][['continent', 'avg_temperature']]
    last_data = continental_df[continental_df['year'] == last_year][['continent', 'avg_temperature']]
    
    # Merge - use inner join to only include continents with data in both years
    change_df = first_data.merge(last_data, on='continent', suffixes=('_first', '_last'), how='inner')
    
    if not change_df.empty:
        change_df['change'] = change_df['avg_temperature_last'] - change_df['avg_temperature_first']
        change_df['change_pct'] = (change_df['change'] / change_df['avg_temperature_first'].abs()) * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                change_df.sort_values('change', ascending=False),
                x='continent',
                y='change',
                title=f"Temperature Change ({first_year} → {last_year})",
                labels={'change': 'Change (°C)', 'continent': 'Continent'},
                color='change',
                color_continuous_scale='RdBu_r',
                color_continuous_midpoint=0
            )
            
            fig.update_xaxes(tickangle=45)
            fig.update_layout(height=400)
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Statistics table
            st.markdown(f"**Temperature Change Statistics:**")
            st.caption(f"Comparing {first_year} vs {last_year}")
            
            display_df = change_df[['continent', 'avg_temperature_first', 'avg_temperature_last', 'change', 'change_pct']].copy()
            display_df.columns = ['Continent', f'{first_year} (°C)', f'{last_year} (°C)', 'Change (°C)', 'Change (%)']
            display_df = display_df.round(2)
            display_df = display_df.sort_values('Change (°C)', ascending=False)
            
            st.dataframe(display_df, hide_index=True, use_container_width=True)
    else:
        st.warning(f"⚠️ No continents have data for both {first_year} and {last_year}")
    
    # Decade analysis
    st.markdown("#### Temperature by Decade")
    
    continental_df['decade'] = (continental_df['year'] // 10) * 10
    decade_avg = continental_df.groupby(['decade', 'continent'])['avg_temperature'].mean().reset_index()
    
    fig = px.line(
        decade_avg,
        x='decade',
        y='avg_temperature',
        color='continent',
        markers=True,
        title="Decadal Average Temperature",
        labels={'decade': 'Decade', 'avg_temperature': 'Temperature (°C)'}
    )
    
    fig.update_layout(height=500)
    
    st.plotly_chart(fig, use_container_width=True)

else:
    st.error("❌ Failed to load continental.parquet")
