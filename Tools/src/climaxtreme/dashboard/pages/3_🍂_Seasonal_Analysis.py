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
st.markdown("Temperature patterns across seasons")

data_source = DataSource()
seasonal_df = data_source.load_parquet('seasonal.parquet')

if seasonal_df is not None and not seasonal_df.empty:
    show_data_info(seasonal_df, "Seasonal Dataset Information")
    
    # Note: seasonal.parquet contains global seasonal aggregations (no year breakdown)
    st.info("📊 Showing global seasonal statistics across all years in the dataset")
    
    # Filter by season only
    seasons = sorted(seasonal_df['season'].unique()) if 'season' in seasonal_df.columns else []
    selected_seasons = st.multiselect("Select Seasons", seasons, default=seasons, key="season_filter")
    
    # Filter
    if selected_seasons:
        filtered = seasonal_df[seasonal_df['season'].isin(selected_seasons)]
    else:
        filtered = seasonal_df
    
    if not filtered.empty:
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Seasons", len(filtered))
        with col2:
            avg_temp = filtered['avg_temperature'].mean()
            st.metric("Avg Temperature", f"{avg_temp:.2f}°C")
        with col3:
            warmest = filtered.loc[filtered['avg_temperature'].idxmax(), 'season']
            st.metric("Warmest Season", warmest)
        with col4:
            coldest = filtered.loc[filtered['avg_temperature'].idxmin(), 'season']
            st.metric("Coldest Season", coldest)
        
        # Seasonal comparison
        st.markdown("#### Average Temperature by Season")
        
        fig = px.bar(filtered, x='season', y='avg_temperature', title="Average Temperature by Season",
                     color='avg_temperature', color_continuous_scale='RdYlBu_r',
                     labels={'avg_temperature': 'Temperature (°C)', 'season': 'Season'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Temperature range by season
        st.markdown("#### Temperature Range by Season")
        
        fig = go.Figure()
        
        for _, row in filtered.iterrows():
            fig.add_trace(go.Scatter(
                x=[row['season'], row['season']],
                y=[row['min_temperature'], row['max_temperature']],
                mode='lines+markers',
                name=row['season'],
                line=dict(width=10),
                marker=dict(size=10)
            ))
        
        fig.update_layout(
            title="Temperature Range by Season",
            xaxis_title="Season",
            yaxis_title="Temperature (°C)",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics table
        st.markdown("#### Seasonal Statistics Summary")
        
        display_df = filtered[['season', 'avg_temperature', 'std_temperature', 'min_temperature', 'max_temperature', 'record_count']].copy()
        display_df.columns = ['Season', 'Mean (°C)', 'Std Dev (°C)', 'Min (°C)', 'Max (°C)', 'Record Count']
        display_df = display_df.sort_values('Mean (°C)', ascending=False)
        
        st.dataframe(display_df, hide_index=True, use_container_width=True)
    else:
        st.warning("No data for selected filters")
else:
    st.error("❌ Failed to load seasonal.parquet")
