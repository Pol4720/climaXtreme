"""
Extreme Events Page - Temperature extremes analysis
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

st.set_page_config(page_title="Extreme Events", page_icon="⚡", layout="wide")
configure_sidebar()

st.title("⚡ Extreme Temperature Events")
data_source = DataSource()

extreme_df = data_source.load_parquet('extreme_thresholds.parquet')
monthly_df = data_source.load_parquet('monthly.parquet')

if extreme_df is not None and not extreme_df.empty:
    show_data_info(extreme_df, "Extreme Thresholds Dataset")
    
    # Select location
    col1, col2 = st.columns(2)
    with col1:
        countries = sorted(extreme_df['Country'].unique())
        country = st.selectbox("Country", countries)
    with col2:
        cities = sorted(extreme_df[extreme_df['Country'] == country]['City'].unique())
        city = st.selectbox("City", cities)
    
    # Filter
    city_extremes = extreme_df[
        (extreme_df['Country'] == country) & 
        (extreme_df['City'] == city)
    ].sort_values('month')
    
    if not city_extremes.empty:
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Hottest P90", f"{city_extremes['p90_temperature'].max():.2f}°C")
        with col2:
            st.metric("Coldest P10", f"{city_extremes['p10_temperature'].min():.2f}°C")
        with col3:
            range_val = city_extremes['p90_temperature'].max() - city_extremes['p10_temperature'].min()
            st.metric("Total Range", f"{range_val:.2f}°C")
        
        # Plot extreme thresholds
        st.markdown(f"#### Extreme Temperature Thresholds - {city}, {country}")
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=month_names,
            y=city_extremes['p90_temperature'],
            mode='lines+markers',
            name='P90 (Hot Extreme)',
            line=dict(color='red', width=2),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=month_names,
            y=city_extremes['p10_temperature'],
            mode='lines+markers',
            name='P10 (Cold Extreme)',
            line=dict(color='blue', width=2),
            marker=dict(size=8),
            fill='tonexty',
            fillcolor='rgba(128, 128, 128, 0.2)'
        ))
        
        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Temperature (°C)",
            height=500,
            hovermode='x'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # If we have monthly data, detect extreme events
        if monthly_df is not None:
            st.markdown("#### Detected Extreme Events")
            
            city_monthly = monthly_df[
                (monthly_df['Country'] == country) & 
                (monthly_df['City'] == city)
            ]
            
            if not city_monthly.empty:
                # Merge with thresholds
                merged = city_monthly.merge(
                    city_extremes[['month', 'p10_temperature', 'p90_temperature']],
                    on='month',
                    how='left'
                )
                
                # Identify extremes
                merged['is_hot_extreme'] = merged['avg_temperature'] > merged['p90_temperature']
                merged['is_cold_extreme'] = merged['avg_temperature'] < merged['p10_temperature']
                
                hot_events = merged[merged['is_hot_extreme']]
                cold_events = merged[merged['is_cold_extreme']]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("🔥 Hot Extreme Events", len(hot_events))
                with col2:
                    st.metric("❄️ Cold Extreme Events", len(cold_events))
                
                # Show recent extremes
                recent_extremes = merged[
                    (merged['is_hot_extreme'] | merged['is_cold_extreme']) &
                    (merged['year'] >= merged['year'].max() - 10)
                ].sort_values('year', ascending=False)
                
                if not recent_extremes.empty:
                    st.markdown("**Recent Extreme Events (Last 10 years):**")
                    display_df = recent_extremes[['year', 'month', 'avg_temperature', 'is_hot_extreme', 'is_cold_extreme']].copy()
                    display_df['Event Type'] = display_df.apply(
                        lambda row: '🔥 Hot' if row['is_hot_extreme'] else '❄️ Cold',
                        axis=1
                    )
                    display_df = display_df[['year', 'month', 'avg_temperature', 'Event Type']]
                    display_df.columns = ['Year', 'Month', 'Temperature (°C)', 'Type']
                    st.dataframe(display_df.head(20), hide_index=True)
            else:
                st.info("No monthly data available for extreme event detection")
        
        # Threshold table
        st.markdown("#### Monthly Extreme Thresholds")
        display_df = city_extremes[['month', 'p10_temperature', 'p90_temperature']].copy()
        display_df['month'] = display_df['month'].map(lambda x: month_names[int(x)-1])
        display_df.columns = ['Month', 'P10 Cold (°C)', 'P90 Hot (°C)']
        st.dataframe(display_df.round(2), hide_index=True, use_container_width=True)
    else:
        st.warning("No data for selected location")
else:
    st.error("❌ Failed to load extreme_thresholds.parquet")
