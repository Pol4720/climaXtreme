"""
Temporal Analysis Page - Monthly and Yearly Temperature Trends
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Import utilities
try:
    from climaxtreme.dashboard.utils import DataSource, configure_sidebar, show_data_info
except ImportError:
    _src_dir = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(_src_dir))
    from climaxtreme.dashboard.utils import DataSource, configure_sidebar, show_data_info


st.set_page_config(
    page_title="Temporal Analysis - climaXtreme",
    page_icon="📈",
    layout="wide"
)

configure_sidebar()

st.title("📈 Temporal Analysis")
st.markdown("Analyze temperature trends using monthly and yearly aggregated data")

# Initialize data source
data_source = DataSource()

# Tabs for different temporal views
tab1, tab2, tab3 = st.tabs(["📅 Monthly Trends", "📆 Yearly Trends", "🔍 Time Series Explorer"])

# ========== TAB 1: MONTHLY TRENDS ==========
with tab1:
    st.subheader("Monthly Temperature Patterns")
    
    # Load monthly data
    monthly_df = data_source.load_parquet('monthly.parquet')
    
    if monthly_df is not None and not monthly_df.empty:
        # Show data info
        show_data_info(monthly_df, "Monthly Dataset Information")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Year range
            min_year = int(monthly_df['year'].min())
            max_year = int(monthly_df['year'].max())
            year_range = st.slider(
                "Select Year Range",
                min_year, max_year,
                (max_year - 20, max_year),
                key="monthly_year_range"
            )
        
        with col2:
            # Country filter
            countries = sorted(monthly_df['Country'].dropna().unique())
            selected_countries = st.multiselect(
                "Filter by Country",
                countries,
                default=countries[:5] if len(countries) > 5 else countries,
                key="monthly_countries"
            )
        
        with col3:
            # City filter
            if selected_countries:
                cities = sorted(monthly_df[monthly_df['Country'].isin(selected_countries)]['City'].dropna().unique())
                selected_city = st.selectbox(
                    "Select City",
                    ["All Cities"] + list(cities),
                    key="monthly_city"
                )
            else:
                selected_city = "All Cities"
        
        # Filter data
        filtered_df = monthly_df[
            (monthly_df['year'] >= year_range[0]) &
            (monthly_df['year'] <= year_range[1])
        ]
        
        if selected_countries:
            filtered_df = filtered_df[filtered_df['Country'].isin(selected_countries)]
        
        if selected_city != "All Cities":
            filtered_df = filtered_df[filtered_df['City'] == selected_city]
        
        if filtered_df.empty:
            st.warning("No data available for selected filters")
        else:
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", f"{len(filtered_df):,}")
            with col2:
                avg_temp = filtered_df['avg_temperature'].mean()
                st.metric("Avg Temperature", f"{avg_temp:.2f}°C")
            with col3:
                temp_range = filtered_df['max_temperature'].max() - filtered_df['min_temperature'].min()
                st.metric("Temperature Range", f"{temp_range:.2f}°C")
            with col4:
                cities_count = filtered_df['City'].nunique()
                st.metric("Cities", f"{cities_count:,}")
            
            # Visualization 1: Monthly average over time
            st.markdown("#### Monthly Temperature Evolution")
            
            # Aggregate by year-month
            filtered_df['year_month'] = pd.to_datetime(
                filtered_df['year'].astype(str) + '-' + 
                filtered_df['month'].astype(str) + '-01'
            )
            
            monthly_avg = filtered_df.groupby('year_month').agg({
                'avg_temperature': 'mean',
                'min_temperature': 'min',
                'max_temperature': 'max'
            }).reset_index()
            
            fig = go.Figure()
            
            # Temperature range as filled area
            fig.add_trace(go.Scatter(
                x=monthly_avg['year_month'],
                y=monthly_avg['max_temperature'],
                fill=None,
                mode='lines',
                line_color='rgba(255, 0, 0, 0)',
                showlegend=False,
                name='Max'
            ))
            
            fig.add_trace(go.Scatter(
                x=monthly_avg['year_month'],
                y=monthly_avg['min_temperature'],
                fill='tonexty',
                mode='lines',
                line_color='rgba(0, 0, 255, 0)',
                fillcolor='rgba(100, 100, 255, 0.2)',
                name='Temperature Range'
            ))
            
            # Average line
            fig.add_trace(go.Scatter(
                x=monthly_avg['year_month'],
                y=monthly_avg['avg_temperature'],
                mode='lines',
                line=dict(color='red', width=2),
                name='Average Temperature'
            ))
            
            fig.update_layout(
                title="Monthly Temperature Trends",
                xaxis_title="Date",
                yaxis_title="Temperature (°C)",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Visualization 2: Seasonal patterns (heatmap)
            st.markdown("#### Seasonal Patterns by Month")
            
            # Create pivot for heatmap
            if selected_city == "All Cities":
                # Aggregate across all cities
                seasonal_pivot = filtered_df.groupby(['year', 'month'])['avg_temperature'].mean().reset_index()
                seasonal_pivot = seasonal_pivot.pivot(index='month', columns='year', values='avg_temperature')
            else:
                seasonal_pivot = filtered_df.pivot_table(
                    index='month',
                    columns='year',
                    values='avg_temperature',
                    aggfunc='mean'
                )
            
            fig = px.imshow(
                seasonal_pivot,
                labels=dict(x="Year", y="Month", color="Temperature (°C)"),
                title="Temperature Heatmap: Month vs Year",
                color_continuous_scale="RdYlBu_r",
                aspect="auto"
            )
            
            fig.update_yaxis(
                tickmode='array',
                tickvals=list(range(1, 13)),
                ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics by month
            st.markdown("#### Monthly Statistics")
            monthly_stats = filtered_df.groupby('month').agg({
                'avg_temperature': ['mean', 'std', 'min', 'max'],
                'record_count': 'sum'
            }).round(2)
            
            monthly_stats.columns = ['Mean (°C)', 'Std Dev (°C)', 'Min (°C)', 'Max (°C)', 'Total Records']
            monthly_stats.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            st.dataframe(monthly_stats, use_container_width=True)
    
    else:
        st.error("❌ Failed to load monthly.parquet. Check your data source configuration.")


# ========== TAB 2: YEARLY TRENDS ==========
with tab2:
    st.subheader("Yearly Temperature Trends")
    
    # Load yearly data
    yearly_df = data_source.load_parquet('yearly.parquet')
    
    if yearly_df is not None and not yearly_df.empty:
        # Show data info
        show_data_info(yearly_df, "Yearly Dataset Information")
        
        # Filters
        col1, col2 = st.columns(2)
        
        with col1:
            # Year range
            min_year = int(yearly_df['year'].min())
            max_year = int(yearly_df['year'].max())
            year_range = st.slider(
                "Select Year Range",
                min_year, max_year,
                (min_year, max_year),
                key="yearly_year_range"
            )
        
        with col2:
            # Aggregation level
            agg_level = st.radio(
                "Aggregation Level",
                ["Global", "By Country", "By City"],
                key="yearly_agg_level"
            )
        
        # Filter by year
        filtered_df = yearly_df[
            (yearly_df['year'] >= year_range[0]) &
            (yearly_df['year'] <= year_range[1])
        ]
        
        if filtered_df.empty:
            st.warning("No data available for selected filters")
        else:
            # Aggregate based on selection
            if agg_level == "Global":
                yearly_agg = filtered_df.groupby('year').agg({
                    'avg_temperature': 'mean',
                    'min_temperature': 'min',
                    'max_temperature': 'max',
                    'temperature_range': 'mean',
                    'record_count': 'sum'
                }).reset_index()
                title_suffix = "Global Average"
            elif agg_level == "By Country":
                # Let user select country
                countries = sorted(filtered_df['Country'].dropna().unique())
                selected_country = st.selectbox("Select Country", countries, key="yearly_country")
                filtered_df = filtered_df[filtered_df['Country'] == selected_country]
                
                yearly_agg = filtered_df.groupby('year').agg({
                    'avg_temperature': 'mean',
                    'min_temperature': 'min',
                    'max_temperature': 'max',
                    'temperature_range': 'mean',
                    'record_count': 'sum'
                }).reset_index()
                title_suffix = f"{selected_country}"
            else:  # By City
                countries = sorted(filtered_df['Country'].dropna().unique())
                selected_country = st.selectbox("Select Country", countries, key="yearly_country_city")
                cities = sorted(filtered_df[filtered_df['Country'] == selected_country]['City'].unique())
                selected_city = st.selectbox("Select City", cities, key="yearly_city")
                
                yearly_agg = filtered_df[
                    (filtered_df['Country'] == selected_country) &
                    (filtered_df['City'] == selected_city)
                ].copy()
                title_suffix = f"{selected_city}, {selected_country}"
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Years Analyzed", len(yearly_agg))
            with col2:
                avg_temp = yearly_agg['avg_temperature'].mean()
                st.metric("Mean Temperature", f"{avg_temp:.2f}°C")
            with col3:
                temp_change = yearly_agg['avg_temperature'].iloc[-1] - yearly_agg['avg_temperature'].iloc[0]
                st.metric("Total Change", f"{temp_change:+.2f}°C")
            with col4:
                # Calculate trend
                import numpy as np
                z = np.polyfit(yearly_agg['year'], yearly_agg['avg_temperature'], 1)
                trend_per_year = z[0]
                st.metric("Trend", f"{trend_per_year*10:+.3f}°C/decade")
            
            # Main chart: Temperature over time with trend
            st.markdown(f"#### Temperature Trends - {title_suffix}")
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Average Temperature with Trend', 'Temperature Range'),
                vertical_spacing=0.12,
                row_heights=[0.6, 0.4]
            )
            
            # Row 1: Average temp with trend
            fig.add_trace(
                go.Scatter(
                    x=yearly_agg['year'],
                    y=yearly_agg['avg_temperature'],
                    mode='lines+markers',
                    name='Average Temperature',
                    line=dict(color='blue', width=2),
                    marker=dict(size=4)
                ),
                row=1, col=1
            )
            
            # Add trend line
            z = np.polyfit(yearly_agg['year'], yearly_agg['avg_temperature'], 1)
            p = np.poly1d(z)
            fig.add_trace(
                go.Scatter(
                    x=yearly_agg['year'],
                    y=p(yearly_agg['year']),
                    mode='lines',
                    name=f'Trend ({z[0]*10:+.3f}°C/decade)',
                    line=dict(color='red', dash='dash', width=2)
                ),
                row=1, col=1
            )
            
            # Row 2: Temperature range
            fig.add_trace(
                go.Scatter(
                    x=yearly_agg['year'],
                    y=yearly_agg['max_temperature'],
                    mode='lines',
                    name='Max Temperature',
                    line=dict(color='rgba(255, 0, 0, 0.5)', width=1)
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=yearly_agg['year'],
                    y=yearly_agg['min_temperature'],
                    mode='lines',
                    name='Min Temperature',
                    line=dict(color='rgba(0, 0, 255, 0.5)', width=1),
                    fill='tonexty',
                    fillcolor='rgba(100, 100, 100, 0.2)'
                ),
                row=2, col=1
            )
            
            fig.update_xaxes(title_text="Year", row=2, col=1)
            fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
            fig.update_yaxes(title_text="Temperature (°C)", row=2, col=1)
            
            fig.update_layout(height=700, showlegend=True, hovermode='x unified')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Distribution over decades
            st.markdown("#### Temperature Distribution by Decade")
            
            yearly_agg['decade'] = (yearly_agg['year'] // 10) * 10
            
            fig = px.box(
                yearly_agg,
                x='decade',
                y='avg_temperature',
                title="Temperature Distribution by Decade",
                labels={'decade': 'Decade', 'avg_temperature': 'Temperature (°C)'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.error("❌ Failed to load yearly.parquet. Check your data source configuration.")


# ========== TAB 3: TIME SERIES EXPLORER ==========
with tab3:
    st.subheader("Interactive Time Series Explorer")
    
    st.info("🔍 This tool allows you to explore and compare temperature trends across multiple cities")
    
    # Load monthly data
    monthly_df = data_source.load_parquet('monthly.parquet')
    
    if monthly_df is not None and not monthly_df.empty:
        # City selection
        col1, col2 = st.columns(2)
        
        with col1:
            countries = sorted(monthly_df['Country'].dropna().unique())
            selected_countries = st.multiselect(
                "Select Countries",
                countries,
                default=countries[:3] if len(countries) > 3 else countries,
                key="explorer_countries"
            )
        
        with col2:
            if selected_countries:
                cities_filtered = monthly_df[monthly_df['Country'].isin(selected_countries)]
                cities = sorted(cities_filtered['City'].dropna().unique())
                selected_cities = st.multiselect(
                    "Select Cities to Compare",
                    cities,
                    default=cities[:5] if len(cities) > 5 else cities,
                    key="explorer_cities",
                    max_selections=10
                )
            else:
                selected_cities = []
        
        if selected_cities:
            # Filter data
            explorer_df = monthly_df[monthly_df['City'].isin(selected_cities)].copy()
            
            # Create date column
            explorer_df['date'] = pd.to_datetime(
                explorer_df['year'].astype(str) + '-' + 
                explorer_df['month'].astype(str) + '-01'
            )
            
            # Aggregate by city and date
            city_series = explorer_df.groupby(['City', 'Country', 'date'])['avg_temperature'].mean().reset_index()
            
            # Plot
            fig = px.line(
                city_series,
                x='date',
                y='avg_temperature',
                color='City',
                title=f"Temperature Comparison: {len(selected_cities)} Cities",
                labels={'date': 'Date', 'avg_temperature': 'Temperature (°C)'}
            )
            
            fig.update_layout(height=600, hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics comparison
            st.markdown("#### City Comparison Statistics")
            
            stats_comparison = explorer_df.groupby(['City', 'Country']).agg({
                'avg_temperature': ['mean', 'std', 'min', 'max'],
                'record_count': 'sum'
            }).round(2)
            
            stats_comparison.columns = ['Mean (°C)', 'Std Dev (°C)', 'Min (°C)', 'Max (°C)', 'Total Records']
            stats_comparison = stats_comparison.sort_values('Mean (°C)', ascending=False)
            
            st.dataframe(stats_comparison, use_container_width=True)
        else:
            st.warning("Please select at least one city to explore")
    
    else:
        st.error("❌ Failed to load monthly.parquet. Check your data source configuration.")
