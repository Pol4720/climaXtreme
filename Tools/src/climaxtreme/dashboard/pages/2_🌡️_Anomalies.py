"""
Anomalies Analysis Page - Temperature anomalies vs climatology
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
st.markdown("Analyze temperature deviations from climatological normals")

data_source = DataSource()

tab1, tab2 = st.tabs(["🔥 Anomalies", "📊 Climatology"])

# TAB 1: ANOMALIES
with tab1:
    st.subheader("Temperature Anomalies")
    
    anomalies_df = data_source.load_parquet('anomalies.parquet')
    
    if anomalies_df is not None and not anomalies_df.empty:
        show_data_info(anomalies_df, "Anomalies Dataset Information")
        
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            min_year = int(anomalies_df['year'].min())
            max_year = int(anomalies_df['year'].max())
            year_range = st.slider("Year Range", min_year, max_year, (min_year, max_year), key="anom_years")
        
        with col2:
            agg_level = st.radio("Aggregation", ["Global", "By Country", "By City"], key="anom_agg")
        
        # Filter
        filtered_df = anomalies_df[(anomalies_df['year'] >= year_range[0]) & (anomalies_df['year'] <= year_range[1])]
        
        # Aggregate
        if agg_level == "Global":
            agg_df = filtered_df.groupby('year').agg({
                'anomaly': 'mean',
                'avg_temperature': 'mean',
                'climatology': 'mean'
            }).reset_index()
            title = "Global"
        elif agg_level == "By Country":
            country = st.selectbox("Country", sorted(filtered_df['Country'].unique()), key="anom_country")
            filtered_df = filtered_df[filtered_df['Country'] == country]
            agg_df = filtered_df.groupby('year').agg({
                'anomaly': 'mean',
                'avg_temperature': 'mean',
                'climatology': 'mean'
            }).reset_index()
            title = country
        else:
            country = st.selectbox("Country", sorted(filtered_df['Country'].unique()), key="anom_country_city")
            cities = sorted(filtered_df[filtered_df['Country'] == country]['City'].unique())
            city = st.selectbox("City", cities, key="anom_city")
            agg_df = filtered_df[(filtered_df['Country'] == country) & (filtered_df['City'] == city)].copy()
            title = f"{city}, {country}"
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Years", len(agg_df))
        with col2:
            avg_anomaly = agg_df['anomaly'].mean()
            st.metric("Mean Anomaly", f"{avg_anomaly:+.2f}°C")
        with col3:
            max_anomaly = agg_df['anomaly'].max()
            st.metric("Max Anomaly", f"{max_anomaly:+.2f}°C")
        with col4:
            min_anomaly = agg_df['anomaly'].min()
            st.metric("Min Anomaly", f"{min_anomaly:+.2f}°C")
        
        # Plot anomalies
        st.markdown(f"#### Anomalies Over Time - {title}")
        
        fig = go.Figure()
        
        # Bar chart with colors
        colors = ['red' if x > 0 else 'blue' for x in agg_df['anomaly']]
        fig.add_trace(go.Bar(
            x=agg_df['year'],
            y=agg_df['anomaly'],
            marker_color=colors,
            name='Anomaly',
            hovertemplate='Year: %{x}<br>Anomaly: %{y:.2f}°C<extra></extra>'
        ))
        
        # Zero line
        fig.add_hline(y=0, line_dash="dash", line_color="black", annotation_text="Climatology")
        
        fig.update_layout(
            title=f"Temperature Anomalies - {title}",
            xaxis_title="Year",
            yaxis_title="Anomaly (°C)",
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Temperature vs Climatology
        st.markdown("#### Observed vs Climatological Temperature")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=agg_df['year'],
            y=agg_df['avg_temperature'],
            mode='lines+markers',
            name='Observed Temperature',
            line=dict(color='red', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=agg_df['year'],
            y=agg_df['climatology'],
            mode='lines',
            name='Climatology',
            line=dict(color='blue', width=2, dash='dash')
        ))
        
        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Temperature (°C)",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribution
        st.markdown("#### Anomaly Distribution")
        
        fig = px.histogram(
            agg_df,
            x='anomaly',
            nbins=30,
            title="Distribution of Temperature Anomalies",
            labels={'anomaly': 'Anomaly (°C)', 'count': 'Frequency'}
        )
        fig.add_vline(x=0, line_dash="dash", line_color="black")
        
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.error("❌ Failed to load anomalies.parquet")

# TAB 2: CLIMATOLOGY
with tab2:
    st.subheader("Climatological Reference Values")
    
    climatology_df = data_source.load_parquet('climatology.parquet')
    
    if climatology_df is not None and not climatology_df.empty:
        show_data_info(climatology_df, "Climatology Dataset Information")
        
        # Select location
        col1, col2 = st.columns(2)
        with col1:
            countries = sorted(climatology_df['Country'].unique())
            country = st.selectbox("Select Country", countries, key="clim_country")
        
        with col2:
            cities = sorted(climatology_df[climatology_df['Country'] == country]['City'].unique())
            city = st.selectbox("Select City", cities, key="clim_city")
        
        # Filter
        city_clim = climatology_df[
            (climatology_df['Country'] == country) & 
            (climatology_df['City'] == city)
        ].sort_values('month')
        
        if not city_clim.empty:
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Annual Mean", f"{city_clim['climatology'].mean():.2f}°C")
            with col2:
                hottest_month = city_clim.loc[city_clim['climatology'].idxmax(), 'month']
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                st.metric("Hottest Month", month_names[int(hottest_month)-1])
            with col3:
                coldest_month = city_clim.loc[city_clim['climatology'].idxmin(), 'month']
                st.metric("Coldest Month", month_names[int(coldest_month)-1])
            
            # Seasonal cycle
            st.markdown(f"#### Climatological Seasonal Cycle - {city}, {country}")
            
            fig = go.Figure()
            
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            fig.add_trace(go.Scatter(
                x=month_names,
                y=city_clim['climatology'],
                mode='lines+markers',
                name='Climatology',
                line=dict(color='green', width=3),
                marker=dict(size=10),
                fill='tozeroy',
                fillcolor='rgba(0, 255, 0, 0.1)'
            ))
            
            fig.update_layout(
                xaxis_title="Month",
                yaxis_title="Temperature (°C)",
                height=400,
                hovermode='x'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show data table
            st.markdown("#### Monthly Climatology Values")
            
            display_df = city_clim[['month', 'climatology', 'record_count']].copy()
            display_df['month'] = display_df['month'].map(lambda x: month_names[int(x)-1])
            display_df.columns = ['Month', 'Temperature (°C)', 'Years Used']
            display_df = display_df.round(2)
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.warning("No data available for selected location")
    
    else:
        st.error("❌ Failed to load climatology.parquet")
