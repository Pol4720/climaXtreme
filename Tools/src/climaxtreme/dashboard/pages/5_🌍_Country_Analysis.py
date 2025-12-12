"""
🌍 Country Analysis Page - Global temperature analysis by country
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
st.markdown("Global temperature analysis by country over time.")

data_source = DataSource()
country_df = data_source.load_parquet('country.parquet')

if country_df is not None and not country_df.empty:
    show_data_info(country_df, "Country Dataset")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "🗺️ Global Map",
        "📊 Country Comparison",
        "📈 Temporal Trends",
        "🔥 Extremes by Country"
    ])
    
    with tab1:
        st.markdown("### 🗺️ Global Temperature Map")
        
        # Year filter
        years = sorted(country_df['year'].unique())
        min_year = int(min(years))
        max_year = int(max(years))
        
        if min_year < max_year:
            selected_year = st.slider("Select Year", min_year, max_year, max_year)
        else:
            selected_year = min_year
            st.info(f"Data only available for year {min_year}")
        
        year_data = country_df[country_df['year'] == selected_year]
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🌍 Countries", year_data['country'].nunique())
        with col2:
            st.metric("🌡️ Global Avg", f"{year_data['avg_temperature'].mean():.2f}°C")
        with col3:
            if len(year_data) > 0:
                hottest_idx = year_data['avg_temperature'].idxmax()
                hottest = year_data.loc[hottest_idx, 'country']
                st.metric("🔥 Hottest", hottest)
        with col4:
            if len(year_data) > 0:
                coldest_idx = year_data['avg_temperature'].idxmin()
                coldest = year_data.loc[coldest_idx, 'country']
                st.metric("❄️ Coldest", coldest)
        
        # Choropleth map
        fig = px.choropleth(
            year_data,
            locations="country",
            locationmode="country names",
            color="avg_temperature",
            hover_name="country",
            hover_data={
                'avg_temperature': ':.2f',
                'continent': True
            },
            color_continuous_scale="RdYlBu_r",
            title=f"Average Temperature by Country - {selected_year}"
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### 📊 Country Temperature Comparison")
        
        # Year selection for comparison
        compare_year = st.selectbox("Select Year for Comparison", sorted(years, reverse=True))
        year_compare = country_df[country_df['year'] == compare_year].copy()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top hottest countries
            st.markdown("#### 🔥 Top 20 Hottest Countries")
            top_hot = year_compare.nlargest(20, 'avg_temperature')
            fig_hot = px.bar(
                top_hot,
                x='country',
                y='avg_temperature',
                color='avg_temperature',
                color_continuous_scale='Reds',
                title=f"Hottest Countries - {compare_year}"
            )
            fig_hot.update_layout(xaxis_tickangle=45, height=500)
            st.plotly_chart(fig_hot, use_container_width=True)
        
        with col2:
            # Top coldest countries
            st.markdown("#### ❄️ Top 20 Coldest Countries")
            top_cold = year_compare.nsmallest(20, 'avg_temperature')
            fig_cold = px.bar(
                top_cold,
                x='country',
                y='avg_temperature',
                color='avg_temperature',
                color_continuous_scale='Blues_r',
                title=f"Coldest Countries - {compare_year}"
            )
            fig_cold.update_layout(xaxis_tickangle=45, height=500)
            st.plotly_chart(fig_cold, use_container_width=True)
        
        # Full table
        st.markdown("#### 📋 All Countries Data")
        display_df = year_compare[['country', 'avg_temperature', 'min_temperature', 'max_temperature', 'record_count']].copy()
        display_df.columns = ['Country', 'Avg Temp (°C)', 'Min Temp (°C)', 'Max Temp (°C)', 'Records']
        display_df = display_df.sort_values('Avg Temp (°C)', ascending=False)
        st.dataframe(display_df, hide_index=True, use_container_width=True)
    
    with tab3:
        st.markdown("### 📈 Temperature Trends by Country")
        
        # Country selection
        countries = sorted(country_df['country'].unique())
        default_countries = ['United States', 'China', 'Russia', 'Brazil', 'India']
        default_selection = [c for c in default_countries if c in countries][:4]
        if not default_selection:
            default_selection = countries[:4]
        
        selected_countries = st.multiselect(
            "Select Countries to Compare",
            countries,
            default=default_selection
        )
        
        if selected_countries:
            trend_data = country_df[country_df['country'].isin(selected_countries)].copy()
            # Sort by country and year for proper line plotting
            trend_data = trend_data.sort_values(['country', 'year'])
            
            # Line chart
            fig = px.line(
                trend_data,
                x='year',
                y='avg_temperature',
                color='country',
                title="Temperature Evolution by Country",
                labels={'year': 'Year', 'avg_temperature': 'Average Temperature (°C)'}
            )
            fig.update_layout(height=500, hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics table
            st.markdown("#### 📊 Country Statistics (All Years)")
            stats = trend_data.groupby('country').agg({
                'avg_temperature': ['mean', 'std', 'min', 'max'],
                'year': ['min', 'max']
            }).round(2)
            stats.columns = ['Mean (°C)', 'Std Dev', 'Min Avg (°C)', 'Max Avg (°C)', 'First Year', 'Last Year']
            st.dataframe(stats, use_container_width=True)
            
            # Temperature range comparison
            st.markdown("#### 🌡️ Temperature Range by Country")
            range_data = trend_data.groupby('country').agg({
                'min_temperature': 'min',
                'max_temperature': 'max'
            }).reset_index()
            range_data['range'] = range_data['max_temperature'] - range_data['min_temperature']
            
            fig_range = go.Figure()
            for _, row in range_data.iterrows():
                fig_range.add_trace(go.Bar(
                    name=row['country'],
                    x=[row['country']],
                    y=[row['range']],
                    text=f"{row['min_temperature']:.1f}°C to {row['max_temperature']:.1f}°C",
                    textposition='outside'
                ))
            fig_range.update_layout(
                title="Temperature Range (Max - Min) by Country",
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig_range, use_container_width=True)
    
    with tab4:
        st.markdown("### 🔥❄️ Temperature Extremes by Country")
        
        # Aggregate extremes across all years
        extremes = country_df.groupby('country').agg({
            'max_temperature': 'max',
            'min_temperature': 'min',
            'avg_temperature': 'mean',
            'record_count': 'sum'
        }).reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔥 Countries with Highest Max Temperatures")
            top_max = extremes.nlargest(15, 'max_temperature')
            fig_max = px.bar(
                top_max,
                x='country',
                y='max_temperature',
                color='max_temperature',
                color_continuous_scale='YlOrRd',
                title="Highest Recorded Temperatures"
            )
            fig_max.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_max, use_container_width=True)
        
        with col2:
            st.markdown("#### ❄️ Countries with Lowest Min Temperatures")
            top_min = extremes.nsmallest(15, 'min_temperature')
            fig_min = px.bar(
                top_min,
                x='country',
                y='min_temperature',
                color='min_temperature',
                color_continuous_scale='YlGnBu_r',
                title="Lowest Recorded Temperatures"
            )
            fig_min.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_min, use_container_width=True)
        
        # Countries with largest temperature variance
        st.markdown("#### 📊 Countries with Largest Temperature Range")
        extremes['temp_range'] = extremes['max_temperature'] - extremes['min_temperature']
        top_range = extremes.nlargest(20, 'temp_range')
        
        fig_range = px.bar(
            top_range,
            x='country',
            y='temp_range',
            color='temp_range',
            color_continuous_scale='Viridis',
            title="Countries with Greatest Temperature Variability",
            hover_data={
                'min_temperature': ':.2f',
                'max_temperature': ':.2f'
            }
        )
        fig_range.update_layout(xaxis_tickangle=45, height=500)
        st.plotly_chart(fig_range, use_container_width=True)

else:
    st.warning("""
    ⚠️ **No country.parquet data found!**
    
    Country-level data is not currently available. Please use:
    - **🌐 Continental Analysis** for regional temperature patterns
    - **📈 Temporal Analysis** for city-level temperature trends
    """)
