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
st.markdown("Analysis of temperature extremes based on statistical thresholds")

data_source = DataSource()
extreme_df = data_source.load_parquet('extreme_thresholds.parquet')
monthly_df = data_source.load_parquet('monthly.parquet')

if extreme_df is not None and not extreme_df.empty:
    show_data_info(extreme_df, "Extreme Thresholds Dataset")
    
    # Display thresholds table
    st.markdown("#### Extreme Temperature Thresholds")
    
    display_df = extreme_df.copy()
    display_df.columns = ['Percentile', 'Hot Threshold (°C)', 'Cold Threshold (°C)']
    st.dataframe(display_df, hide_index=True, use_container_width=True)
    
    # Plot thresholds
    st.markdown("#### Threshold Visualization")
    
    fig = go.Figure()
    
    # High thresholds (hot extremes)
    fig.add_trace(go.Scatter(
        x=extreme_df['percentile'],
        y=extreme_df['high_threshold'],
        mode='lines+markers',
        name='Hot Extreme Threshold',
        line=dict(color='red', width=3),
        marker=dict(size=10)
    ))
    
    # Low thresholds (cold extremes)
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
    
    # Detect extreme events in monthly data
    if monthly_df is not None and not monthly_df.empty:
        st.markdown("#### Extreme Events Detection")
        
        # Get P90 thresholds (most common for extreme detection)
        p90_row = extreme_df[extreme_df['percentile'] == 90.0]
        if len(p90_row) > 0:
            hot_threshold = p90_row['high_threshold'].values[0]
            cold_threshold = p90_row['low_threshold'].values[0]
        else:
            # Fallback to quantiles
            hot_threshold = monthly_df['avg_temperature'].quantile(0.9)
            cold_threshold = monthly_df['avg_temperature'].quantile(0.1)
        
        # Show thresholds being used
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🔥 Hot Extreme Threshold (P90)", f"{hot_threshold:.2f}°C")
        with col2:
            st.metric("❄️ Cold Extreme Threshold (P10)", f"{cold_threshold:.2f}°C")
        
        # Identify extremes
        monthly_df['is_hot_extreme'] = monthly_df['avg_temperature'] > hot_threshold
        monthly_df['is_cold_extreme'] = monthly_df['avg_temperature'] < cold_threshold
        
        hot_events = monthly_df[monthly_df['is_hot_extreme']]
        cold_events = monthly_df[monthly_df['is_cold_extreme']]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🔥 Hot Extreme Events", len(hot_events))
        with col2:
            st.metric("❄️ Cold Extreme Events", len(cold_events))
        
        # Show recent extremes
        st.markdown("**Recent Extreme Events:**")
        recent = monthly_df[(monthly_df['is_hot_extreme'] | monthly_df['is_cold_extreme'])].nlargest(20, 'year')
        recent['Type'] = recent.apply(lambda row: '🔥 Hot' if row['is_hot_extreme'] else '❄️ Cold', axis=1)
        display = recent[['year', 'month', 'avg_temperature', 'Type']]
        display.columns = ['Year', 'Month', 'Temperature (°C)', 'Type']
        st.dataframe(display, hide_index=True, use_container_width=True)
else:
    st.error("❌ Failed to load extreme_thresholds.parquet")
