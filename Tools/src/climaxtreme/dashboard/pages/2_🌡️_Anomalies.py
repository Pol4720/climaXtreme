"""
Anomalies Analysis Page - Temperature anomalies detection
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

st.set_page_config(page_title="Anomalies - climaXtreme", page_icon="🌡️", layout="wide")
configure_sidebar()

st.title("🌡️ Temperature Anomalies Analysis")
st.markdown("Analyze temperature anomalies using statistical detection methods")

data_source = DataSource()

tab1, tab2 = st.tabs(["🔥 Anomalies Detection", "📊 Climatology Reference"])

# TAB 1: ANOMALIES
with tab1:
    st.subheader("Temperature Anomalies")
    
    monthly_df = data_source.load_parquet('monthly.parquet')
    climatology_df = data_source.load_parquet('climatology.parquet')
    
    if monthly_df is not None and not monthly_df.empty:
        show_data_info(monthly_df, "Monthly Dataset")
        
        # Calculate anomalies
        if climatology_df is not None and not climatology_df.empty:
            st.info("📊 Computing anomalies relative to climatological mean...")
            merged = monthly_df.merge(climatology_df[['month', 'climatology_mean']], on='month', how='left')
            merged['anomaly'] = merged['avg_temperature'] - merged['climatology_mean']
            mean_temp = monthly_df['avg_temperature'].mean()
            std_temp = monthly_df['avg_temperature'].std()
            threshold = 3.0
            merged['zscore'] = (merged['avg_temperature'] - mean_temp) / std_temp
            merged['is_anomaly'] = abs(merged['zscore']) > threshold
        else:
            st.warning("⚠️ Climatology data not available. Using z-score method.")
            mean_temp = monthly_df['avg_temperature'].mean()
            std_temp = monthly_df['avg_temperature'].std()
            threshold = 3.0
            merged = monthly_df.copy()
            merged['zscore'] = (merged['avg_temperature'] - mean_temp) / std_temp
            merged['is_anomaly'] = abs(merged['zscore']) > threshold
            merged['anomaly'] = merged['avg_temperature'] - mean_temp
        
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            min_year = int(merged['year'].min())
            max_year = int(merged['year'].max())
            year_range = st.slider("Year Range", min_year, max_year, (min_year, max_year), key="anom_years")
        with col2:
            threshold_selector = st.slider("Z-Score Threshold", 1.0, 5.0, threshold, 0.5, key="threshold")
            merged['is_anomaly'] = abs(merged['zscore']) > threshold_selector
        
        # Filter
        filtered_df = merged[(merged['year'] >= year_range[0]) & (merged['year'] <= year_range[1])].copy()
        
        # Metrics
        anomalies = filtered_df[filtered_df['is_anomaly']]
        hot_anomalies = filtered_df[(filtered_df['is_anomaly']) & (filtered_df['zscore'] > 0)]
        cold_anomalies = filtered_df[(filtered_df['is_anomaly']) & (filtered_df['zscore'] < 0)]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(filtered_df):,}")
        with col2:
            pct = (len(anomalies) / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0
            st.metric("Anomalies", f"{len(anomalies):,}", f"{pct:.2f}%")
        with col3:
            st.metric("🔥 Hot Anomalies", f"{len(hot_anomalies):,}")
        with col4:
            st.metric("❄️ Cold Anomalies", f"{len(cold_anomalies):,}")
        
        # Plot
        st.markdown("#### Anomalies Over Time")
        yearly_anom = filtered_df.groupby('year').agg({'anomaly': 'mean', 'avg_temperature': 'mean', 'is_anomaly': 'sum'}).reset_index()
        
        fig = go.Figure()
        colors = ['red' if x > 0 else 'blue' for x in yearly_anom['anomaly']]
        fig.add_trace(go.Bar(x=yearly_anom['year'], y=yearly_anom['anomaly'], marker_color=colors, name='Anomaly'))
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        fig.update_layout(title="Annual Mean Temperature Anomaly", xaxis_title="Year", yaxis_title="Anomaly (°C)", height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Z-score distribution
        st.markdown("#### Z-Score Distribution")
        fig = px.histogram(filtered_df, x='zscore', nbins=50, color='is_anomaly', color_discrete_map={True: 'red', False: 'blue'})
        fig.add_vline(x=threshold_selector, line_dash="dash", line_color="red")
        fig.add_vline(x=-threshold_selector, line_dash="dash", line_color="blue")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("❌ Failed to load monthly.parquet")

# TAB 2: CLIMATOLOGY
with tab2:
    st.subheader("Climatological Reference Values")
    climatology_df = data_source.load_parquet('climatology.parquet')
    
    if climatology_df is not None and not climatology_df.empty:
        show_data_info(climatology_df, "Climatology Dataset")
        st.markdown("#### Monthly Climatology Statistics")
        
        display_df = climatology_df.copy()
        if 'month' in display_df.columns:
            display_df['Month'] = display_df['month'].apply(lambda x: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][int(x)-1])
            display_df = display_df[['Month', 'climatology_mean', 'climatology_std', 'climatology_min', 'climatology_max', 'climatology_count']]
            display_df.columns = ['Month', 'Mean (°C)', 'Std Dev (°C)', 'Min (°C)', 'Max (°C)', 'Record Count']
        st.dataframe(display_df, hide_index=True, use_container_width=True)
        
        # Visualization
        st.markdown("#### Monthly Climatology Visualization")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=climatology_df['month'], y=climatology_df['climatology_mean'], mode='lines+markers', name='Mean Temperature', line=dict(color='green', width=3), marker=dict(size=8)))
        fig.add_trace(go.Scatter(x=climatology_df['month'], y=climatology_df['climatology_mean'] + climatology_df['climatology_std'], mode='lines', line=dict(color='rgba(0, 255, 0, 0.3)'), showlegend=False))
        fig.add_trace(go.Scatter(x=climatology_df['month'], y=climatology_df['climatology_mean'] - climatology_df['climatology_std'], mode='lines', line=dict(color='rgba(0, 255, 0, 0.3)'), fill='tonexty', fillcolor='rgba(0, 255, 0, 0.2)', name='±1 Std Dev'))
        fig.update_xaxes(tickmode='array', tickvals=list(range(1, 13)), ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        fig.update_layout(title="Monthly Climatology with Standard Deviation", xaxis_title="Month", yaxis_title="Temperature (°C)", height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("❌ Failed to load climatology.parquet")
