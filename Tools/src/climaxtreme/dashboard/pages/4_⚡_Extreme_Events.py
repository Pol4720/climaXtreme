"""
Extreme Events Page - Temperature extremes analysis at city level
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

try:
    from climaxtreme.dashboard.utils import DataSource, configure_sidebar, show_data_info
except ImportError:
    _src_dir = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(_src_dir))
    from climaxtreme.dashboard.utils import DataSource, configure_sidebar, show_data_info

st.set_page_config(page_title="Extreme Events", page_icon="⚡", layout="wide")
configure_sidebar()

st.title("⚡ Extreme Temperature Events")
st.markdown("Analysis of temperature extremes based on statistical thresholds at city level")

data_source = DataSource()
extreme_df = data_source.load_parquet('extreme_thresholds.parquet')
# Use anomalies.parquet which has city-level temperature data
anomalies_df = data_source.load_parquet('anomalies.parquet')

if extreme_df is not None and not extreme_df.empty:
    show_data_info(extreme_df, "Extreme Thresholds Dataset")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Thresholds",
        "🔥 Hot Extremes", 
        "❄️ Cold Extremes",
        "🗺️ Geographic Distribution"
    ])
    
    with tab1:
        st.markdown("#### Extreme Temperature Thresholds")
        st.markdown("""
        These thresholds are computed from **all city-level temperature records** (8.2M+ observations).
        - **Hot Extreme (P90)**: Temperature above the 90th percentile
        - **Cold Extreme (P10)**: Temperature below the 10th percentile
        """)
        
        # Display thresholds table with proper column order
        display_df = extreme_df[['percentile', 'high_threshold', 'low_threshold']].copy()
        display_df.columns = ['Percentile', 'Hot Threshold (°C)', 'Cold Threshold (°C)']
        st.dataframe(display_df, hide_index=True, use_container_width=True)
        
        # Plot thresholds
        st.markdown("#### Threshold Visualization")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=extreme_df['percentile'],
            y=extreme_df['high_threshold'],
            mode='lines+markers',
            name='Hot Extreme Threshold',
            line=dict(color='red', width=3),
            marker=dict(size=10)
        ))
        
        fig.add_trace(go.Scatter(
            x=extreme_df['percentile'],
            y=extreme_df['low_threshold'],
            mode='lines+markers',
            name='Cold Extreme Threshold',
            line=dict(color='blue', width=3),
            marker=dict(size=10)
        ))
        
        fig.update_layout(
            title="Temperature Thresholds by Percentile",
            xaxis_title="Percentile",
            yaxis_title="Temperature (°C)",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Check for anomalies data (city-level)
    if anomalies_df is not None and not anomalies_df.empty:
        # Get P90 thresholds
        p90_row = extreme_df[extreme_df['percentile'] == 90.0]
        if len(p90_row) > 0:
            hot_threshold = p90_row['high_threshold'].values[0]
            cold_threshold = p90_row['low_threshold'].values[0]
        else:
            hot_threshold = 27.56
            cold_threshold = 2.06
        
        # Identify extreme events
        anomalies_df['is_hot_extreme'] = anomalies_df['temperature'] > hot_threshold
        anomalies_df['is_cold_extreme'] = anomalies_df['temperature'] < cold_threshold
        
        hot_events = anomalies_df[anomalies_df['is_hot_extreme']]
        cold_events = anomalies_df[anomalies_df['is_cold_extreme']]
        
        # Summary metrics in sidebar
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Extreme Events Summary")
        st.sidebar.metric("🔥 Hot Threshold (P90)", f"{hot_threshold:.2f}°C")
        st.sidebar.metric("❄️ Cold Threshold (P10)", f"{cold_threshold:.2f}°C")
        st.sidebar.metric("🔥 Hot Events", f"{len(hot_events):,}")
        st.sidebar.metric("❄️ Cold Events", f"{len(cold_events):,}")
        
        with tab2:
            st.markdown("### 🔥 Hot Extreme Events")
            st.markdown(f"Events where temperature exceeds **{hot_threshold:.2f}°C** (P90 threshold)")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Hot Events", f"{len(hot_events):,}")
            with col2:
                st.metric("Max Temperature", f"{hot_events['temperature'].max():.2f}°C")
            with col3:
                st.metric("Cities Affected", f"{hot_events['city'].nunique():,}")
            
            # Top hottest events
            st.markdown("#### 🌡️ Top 20 Hottest Records")
            top_hot = hot_events.nlargest(20, 'temperature')[
                ['year', 'month', 'city', 'country', 'temperature', 'continent']
            ].copy()
            top_hot.columns = ['Year', 'Month', 'City', 'Country', 'Temperature (°C)', 'Continent']
            st.dataframe(top_hot, hide_index=True, use_container_width=True)
            
            # Hot events by year
            st.markdown("#### 📈 Hot Events by Year")
            hot_by_year = hot_events.groupby('year').size().reset_index(name='count')
            fig_hot_year = px.bar(
                hot_by_year,
                x='year',
                y='count',
                title='Number of Hot Extreme Events by Year',
                color_discrete_sequence=['#e74c3c']
            )
            fig_hot_year.update_layout(xaxis_title='Year', yaxis_title='Number of Events')
            st.plotly_chart(fig_hot_year, use_container_width=True)
            
            # Hot events by month
            st.markdown("#### 📅 Hot Events by Month")
            hot_by_month = hot_events.groupby('month').size().reset_index(name='count')
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            hot_by_month['month_name'] = hot_by_month['month'].apply(lambda x: month_names[x-1])
            fig_hot_month = px.bar(
                hot_by_month,
                x='month_name',
                y='count',
                title='Hot Extreme Events by Month',
                color_discrete_sequence=['#e74c3c']
            )
            st.plotly_chart(fig_hot_month, use_container_width=True)
            
            # Top countries with hot extremes
            st.markdown("#### 🌍 Countries with Most Hot Extremes")
            hot_by_country = hot_events.groupby('country').size().reset_index(name='count')
            top_hot_countries = hot_by_country.nlargest(15, 'count')
            fig_hot_countries = px.bar(
                top_hot_countries,
                x='country',
                y='count',
                title='Top 15 Countries with Hot Extreme Events',
                color='count',
                color_continuous_scale='Reds'
            )
            fig_hot_countries.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_hot_countries, use_container_width=True)
        
        with tab3:
            st.markdown("### ❄️ Cold Extreme Events")
            st.markdown(f"Events where temperature falls below **{cold_threshold:.2f}°C** (P10 threshold)")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Cold Events", f"{len(cold_events):,}")
            with col2:
                st.metric("Min Temperature", f"{cold_events['temperature'].min():.2f}°C")
            with col3:
                st.metric("Cities Affected", f"{cold_events['city'].nunique():,}")
            
            # Top coldest events
            st.markdown("#### 🥶 Top 20 Coldest Records")
            top_cold = cold_events.nsmallest(20, 'temperature')[
                ['year', 'month', 'city', 'country', 'temperature', 'continent']
            ].copy()
            top_cold.columns = ['Year', 'Month', 'City', 'Country', 'Temperature (°C)', 'Continent']
            st.dataframe(top_cold, hide_index=True, use_container_width=True)
            
            # Cold events by year
            st.markdown("#### 📈 Cold Events by Year")
            cold_by_year = cold_events.groupby('year').size().reset_index(name='count')
            fig_cold_year = px.bar(
                cold_by_year,
                x='year',
                y='count',
                title='Number of Cold Extreme Events by Year',
                color_discrete_sequence=['#3498db']
            )
            fig_cold_year.update_layout(xaxis_title='Year', yaxis_title='Number of Events')
            st.plotly_chart(fig_cold_year, use_container_width=True)
            
            # Cold events by month
            st.markdown("#### 📅 Cold Events by Month")
            cold_by_month = cold_events.groupby('month').size().reset_index(name='count')
            cold_by_month['month_name'] = cold_by_month['month'].apply(lambda x: month_names[x-1])
            fig_cold_month = px.bar(
                cold_by_month,
                x='month_name',
                y='count',
                title='Cold Extreme Events by Month',
                color_discrete_sequence=['#3498db']
            )
            st.plotly_chart(fig_cold_month, use_container_width=True)
            
            # Top countries with cold extremes
            st.markdown("#### 🌍 Countries with Most Cold Extremes")
            cold_by_country = cold_events.groupby('country').size().reset_index(name='count')
            top_cold_countries = cold_by_country.nlargest(15, 'count')
            fig_cold_countries = px.bar(
                top_cold_countries,
                x='country',
                y='count',
                title='Top 15 Countries with Cold Extreme Events',
                color='count',
                color_continuous_scale='Blues'
            )
            fig_cold_countries.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_cold_countries, use_container_width=True)
        
        with tab4:
            st.markdown("### 🗺️ Geographic Distribution of Extreme Events")
            
            # Filter for events with valid coordinates
            geo_hot = hot_events[hot_events['lat_numeric'].notna() & hot_events['lon_numeric'].notna()]
            geo_cold = cold_events[cold_events['lat_numeric'].notna() & cold_events['lon_numeric'].notna()]
            
            # Aggregate by city for visualization
            hot_cities = geo_hot.groupby(['city', 'country', 'lat_numeric', 'lon_numeric']).agg({
                'temperature': ['count', 'max']
            }).reset_index()
            hot_cities.columns = ['city', 'country', 'lat', 'lon', 'event_count', 'max_temp']
            
            cold_cities = geo_cold.groupby(['city', 'country', 'lat_numeric', 'lon_numeric']).agg({
                'temperature': ['count', 'min']
            }).reset_index()
            cold_cities.columns = ['city', 'country', 'lat', 'lon', 'event_count', 'min_temp']
            
            # Map selection
            map_type = st.radio(
                "Select Event Type",
                options=['🔥 Hot Extremes', '❄️ Cold Extremes', '🔀 Both'],
                horizontal=True
            )
            
            fig_map = go.Figure()
            
            if map_type in ['🔥 Hot Extremes', '🔀 Both']:
                # Sample for performance
                hot_sample = hot_cities.nlargest(500, 'event_count')
                fig_map.add_trace(go.Scattergeo(
                    lat=hot_sample['lat'],
                    lon=hot_sample['lon'],
                    mode='markers',
                    name='Hot Extremes',
                    marker=dict(
                        size=hot_sample['event_count'] / hot_sample['event_count'].max() * 20 + 5,
                        color='red',
                        opacity=0.6,
                        line=dict(width=1, color='darkred')
                    ),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b>, %{customdata[1]}<br>" +
                        "Events: %{customdata[2]}<br>" +
                        "Max Temp: %{customdata[3]:.1f}°C<extra></extra>"
                    ),
                    customdata=hot_sample[['city', 'country', 'event_count', 'max_temp']].values
                ))
            
            if map_type in ['❄️ Cold Extremes', '🔀 Both']:
                cold_sample = cold_cities.nlargest(500, 'event_count')
                fig_map.add_trace(go.Scattergeo(
                    lat=cold_sample['lat'],
                    lon=cold_sample['lon'],
                    mode='markers',
                    name='Cold Extremes',
                    marker=dict(
                        size=cold_sample['event_count'] / cold_sample['event_count'].max() * 20 + 5,
                        color='blue',
                        opacity=0.6,
                        line=dict(width=1, color='darkblue')
                    ),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b>, %{customdata[1]}<br>" +
                        "Events: %{customdata[2]}<br>" +
                        "Min Temp: %{customdata[3]:.1f}°C<extra></extra>"
                    ),
                    customdata=cold_sample[['city', 'country', 'event_count', 'min_temp']].values
                ))
            
            fig_map.update_layout(
                title="Geographic Distribution of Extreme Temperature Events",
                geo=dict(
                    showland=True,
                    landcolor='rgb(243, 243, 243)',
                    countrycolor='rgb(204, 204, 204)',
                    showocean=True,
                    oceancolor='rgb(230, 245, 255)',
                    showcountries=True,
                    projection_type='natural earth'
                ),
                height=600,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            
            st.plotly_chart(fig_map, use_container_width=True)
            
            # Continental breakdown
            st.markdown("#### 🌍 Extreme Events by Continent")
            
            col1, col2 = st.columns(2)
            
            with col1:
                hot_by_continent = hot_events.groupby('continent').size().reset_index(name='count')
                fig_hot_cont = px.pie(
                    hot_by_continent,
                    values='count',
                    names='continent',
                    title='🔥 Hot Extremes by Continent',
                    color_discrete_sequence=px.colors.sequential.Reds
                )
                st.plotly_chart(fig_hot_cont, use_container_width=True)
            
            with col2:
                cold_by_continent = cold_events.groupby('continent').size().reset_index(name='count')
                fig_cold_cont = px.pie(
                    cold_by_continent,
                    values='count',
                    names='continent',
                    title='❄️ Cold Extremes by Continent',
                    color_discrete_sequence=px.colors.sequential.Blues
                )
                st.plotly_chart(fig_cold_cont, use_container_width=True)
    
    else:
        st.warning("""
        ⚠️ **No city-level data found!**
        
        The `anomalies.parquet` file is required for detecting extreme events at city level.
        Please ensure data preprocessing has been completed.
        """)

else:
    st.error("❌ Failed to load extreme_thresholds.parquet")
