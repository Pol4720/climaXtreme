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
        col1, col2 = st.columns(2)
        
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
            # Month filter
            all_months = sorted(monthly_df['month'].unique())
            selected_months = st.multiselect(
                "Filter by Month",
                all_months,
                default=all_months,
                key="monthly_months",
                format_func=lambda x: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][int(x)-1]
            )
        
        # Filter data
        filtered_df = monthly_df[
            (monthly_df['year'] >= year_range[0]) &
            (monthly_df['year'] <= year_range[1])
        ]
        
        if selected_months:
            filtered_df = filtered_df[filtered_df['month'].isin(selected_months)]
        
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
                years_count = filtered_df['year'].nunique()
                st.metric("Years Covered", f"{years_count:,}")
            
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
            seasonal_pivot = filtered_df.groupby(['year', 'month'])['avg_temperature'].mean().reset_index()
            seasonal_pivot = seasonal_pivot.pivot(index='month', columns='year', values='avg_temperature')
            
            fig = px.imshow(
                seasonal_pivot,
                labels=dict(x="Year", y="Month", color="Temperature (°C)"),
                title="Temperature Heatmap: Month vs Year",
                color_continuous_scale="RdYlBu_r",
                aspect="auto"
            )
            
            fig.update_yaxes(
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
            # Show trend line option
            show_trend = st.checkbox("Show Trend Line", value=True, key="show_trend")
        
        # Filter by year
        filtered_df = yearly_df[
            (yearly_df['year'] >= year_range[0]) &
            (yearly_df['year'] <= year_range[1])
        ].copy()
        
        if filtered_df.empty:
            st.warning("No data available for selected filters")
        else:
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Years", len(filtered_df))
            with col2:
                avg_temp = filtered_df['avg_temperature'].mean()
                st.metric("Mean Temperature", f"{avg_temp:.2f}°C")
            with col3:
                if 'trend_slope_per_decade' in filtered_df.columns:
                    trend = filtered_df['trend_slope_per_decade'].iloc[0]
                    st.metric("Trend (per decade)", f"{trend:+.3f}°C")
                else:
                    # Calculate trend manually
                    import numpy as np
                    z = np.polyfit(filtered_df['year'], filtered_df['avg_temperature'], 1)
                    st.metric("Trend (per decade)", f"{z[0]*10:+.3f}°C")
            with col4:
                temp_change = filtered_df['avg_temperature'].iloc[-1] - filtered_df['avg_temperature'].iloc[0]
                st.metric("Total Change", f"{temp_change:+.2f}°C")
            
            # Visualization: Yearly temperature with trend
            st.markdown("#### Yearly Temperature Evolution")
            
            fig = go.Figure()
            
            # Temperature range
            fig.add_trace(go.Scatter(
                x=filtered_df['year'],
                y=filtered_df['max_temperature'],
                fill=None,
                mode='lines',
                line_color='rgba(255, 0, 0, 0.2)',
                showlegend=False,
                name='Max'
            ))
            
            fig.add_trace(go.Scatter(
                x=filtered_df['year'],
                y=filtered_df['min_temperature'],
                fill='tonexty',
                mode='lines',
                line_color='rgba(0, 0, 255, 0.2)',
                fillcolor='rgba(200, 200, 200, 0.3)',
                name='Temp Range',
                showlegend=True
            ))
            
            # Average temperature
            fig.add_trace(go.Scatter(
                x=filtered_df['year'],
                y=filtered_df['avg_temperature'],
                mode='lines+markers',
                line=dict(color='red', width=3),
                marker=dict(size=5),
                name='Average Temperature'
            ))
            
            # Trend line
            if show_trend:
                import numpy as np
                z = np.polyfit(filtered_df['year'], filtered_df['avg_temperature'], 1)
                p = np.poly1d(z)
                fig.add_trace(go.Scatter(
                    x=filtered_df['year'],
                    y=p(filtered_df['year']),
                    mode='lines',
                    line=dict(color='black', width=2, dash='dash'),
                    name=f'Trend ({z[0]*10:+.3f}°C/decade)'
                ))
            
            fig.update_layout(
                title="Yearly Temperature Trends",
                xaxis_title="Year",
                yaxis_title="Temperature (°C)",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Annual statistics
            st.markdown("#### Annual Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Warmest years
                st.markdown("**🔥 Top 10 Warmest Years:**")
                warmest = filtered_df.nlargest(10, 'avg_temperature')[['year', 'avg_temperature']]
                warmest.columns = ['Year', 'Temperature (°C)']
                st.dataframe(warmest.reset_index(drop=True), hide_index=True, use_container_width=True)
            
            with col2:
                # Coldest years
                st.markdown("**❄️ Top 10 Coldest Years:**")
                coldest = filtered_df.nsmallest(10, 'avg_temperature')[['year', 'avg_temperature']]
                coldest.columns = ['Year', 'Temperature (°C)']
                st.dataframe(coldest.reset_index(drop=True), hide_index=True, use_container_width=True)
    
    else:
        st.error("❌ Failed to load yearly.parquet. Check your data source configuration.")


# ========== TAB 3: TIME SERIES EXPLORER ==========
with tab3:
    st.subheader("🔍 Interactive Time Series Explorer")
    
    # Load monthly data for detailed exploration
    monthly_df = data_source.load_parquet('monthly.parquet')
    
    if monthly_df is not None and not monthly_df.empty:
        # Date range selector
        min_year = int(monthly_df['year'].min())
        max_year = int(monthly_df['year'].max())
        
        col1, col2 = st.columns(2)
        with col1:
            start_year = st.number_input("Start Year", min_value=min_year, max_value=max_year, value=max_year-10)
        with col2:
            end_year = st.number_input("End Year", min_value=min_year, max_value=max_year, value=max_year)
        
        if start_year >= end_year:
            st.error("Start year must be less than end year")
        else:
            # Filter data
            filtered = monthly_df[
                (monthly_df['year'] >= start_year) &
                (monthly_df['year'] <= end_year)
            ].copy()
            
            # Create datetime column
            filtered['date'] = pd.to_datetime(
                filtered['year'].astype(str) + '-' + 
                filtered['month'].astype(str) + '-01'
            )
            
            # Visualization options
            viz_type = st.radio(
                "Visualization Type",
                ["Line Chart", "Area Chart", "Bar Chart", "Box Plot by Month"],
                horizontal=True
            )
            
            if viz_type == "Line Chart":
                fig = px.line(
                    filtered,
                    x='date',
                    y='avg_temperature',
                    title=f"Temperature Time Series ({start_year}-{end_year})",
                    labels={'date': 'Date', 'avg_temperature': 'Temperature (°C)'}
                )
                
            elif viz_type == "Area Chart":
                fig = px.area(
                    filtered,
                    x='date',
                    y='avg_temperature',
                    title=f"Temperature Time Series ({start_year}-{end_year})",
                    labels={'date': 'Date', 'avg_temperature': 'Temperature (°C)'}
                )
                
            elif viz_type == "Bar Chart":
                # Aggregate by year for bar chart
                yearly_agg = filtered.groupby('year')['avg_temperature'].mean().reset_index()
                fig = px.bar(
                    yearly_agg,
                    x='year',
                    y='avg_temperature',
                    title=f"Annual Average Temperature ({start_year}-{end_year})",
                    labels={'year': 'Year', 'avg_temperature': 'Temperature (°C)'},
                    color='avg_temperature',
                    color_continuous_scale='RdYlBu_r'
                )
                
            else:  # Box Plot
                fig = px.box(
                    filtered,
                    x='month',
                    y='avg_temperature',
                    title=f"Temperature Distribution by Month ({start_year}-{end_year})",
                    labels={'month': 'Month', 'avg_temperature': 'Temperature (°C)'}
                )
                fig.update_xaxes(
                    tickmode='array',
                    tickvals=list(range(1, 13)),
                    ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                )
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Data summary
            with st.expander("📊 Data Summary"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Records", f"{len(filtered):,}")
                with col2:
                    st.metric("Mean Temperature", f"{filtered['avg_temperature'].mean():.2f}°C")
                with col3:
                    st.metric("Std Deviation", f"{filtered['avg_temperature'].std():.2f}°C")
                
                # Show sample data
                st.markdown("**Sample Data:**")
                display_df = filtered[['year', 'month', 'avg_temperature', 'min_temperature', 'max_temperature', 'record_count']].head(20)
                st.dataframe(display_df, hide_index=True, use_container_width=True)
    
    else:
        st.error("❌ Failed to load monthly.parquet. Check your data source configuration.")
